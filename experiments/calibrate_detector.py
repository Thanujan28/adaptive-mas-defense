"""
experiments/calibrate_detector.py

Report-only calibration of the observable-evidence detector (P3/P4).

NOT part of CI. Does not tune, change, or recommend any threshold or
weight -- it only reports:

  * per-category and overall false-positive rate on CLEAN episodes,
    per response and per episode, across the four topologies
    (layered, fully_connected_p2p, centralized, shared_pool);
  * TPR (detection rate) of the default GENERIC-only detector on the
    held-out payload variants from evaluation/payload_variants.py
    (Task 2c).

Usage:
    python -m experiments.calibrate_detector
    python -m experiments.calibrate_detector --num-episodes 5
"""

from __future__ import annotations
import argparse
import csv
import json
import statistics
import traceback
from datetime import date
from pathlib import Path
from evaluation.payload_variants import PAYLOAD_VARIANTS
from security.content_detector import scan_text
from security.detector import SecurityDetector
TOPOLOGIES = (
    "layered",
    "fully_connected_p2p",
    "centralized",
    "shared_pool",
)

# Default seed + task used for the clean-run token baseline. The
# baseline file records these so it is reproducible ("generated:
# seeds, date").
DEFAULT_TOKEN_BASELINE_SEED = 20260906
DEFAULT_TOKEN_BASELINE_TASK = (
    "Write a short proposal about renewable energy adoption "
    "in mid-size cities (run {index})."
)


def _install_stub_llm() -> None:
    """
    Optionally replace the live LLM and the real tools with
    deterministic in-memory stubs (no Ollama, no network, no docx, no
    config mutation). Mirrors tests/conftest.py so an offline dry run
    and a live run exercise the same detector paths.
    """

    from tests.conftest import StubLLM
    import agents.analyst as analyst_module
    import agents.coordinator as coordinator_module
    import agents.executor as executor_module
    import agents.outline as outline_module
    import agents.researcher as researcher_module
    stub = StubLLM()
    for module in (
        coordinator_module,
        outline_module,
        researcher_module,
        analyst_module,
        executor_module,
    ):
        module.get_llm = lambda: stub

def _stub_tools(env) -> None:
    """Swap every real tool for an in-memory stub (no network)."""

    from tests.conftest import (
        StubCalendar,
        StubEmail,
        StubReportWriter,
        StubSearchTool,
    )

    tools = env.tool_manager.tools
    for name in ("internet_search", "academic_search"):
        if name in tools:
            tools[name] = StubSearchTool()
    if "report_writer" in tools:
        tools["report_writer"] = StubReportWriter()
    if "mock_calendar" in tools:
        tools["mock_calendar"] = StubCalendar()
    if "mock_calender" in tools:
        tools["mock_calender"] = StubCalendar()
    if "mock_email" in tools:
        tools["mock_email"] = StubEmail()


def _run_clean_episode(
    topology_name: str,
    task: str,
    stub_llm: bool = False,
    stub_encoder: bool = False,
):
    # Imported lazily: MASEnvironment pulls in the LLM backend, which
    # this script must still be able to report on (as "error") when
    # unavailable, rather than crashing entirely.
    from environment.mas_environment import MASEnvironment
    # Patch the LLM BEFORE building the env (agents call get_llm()
    # during __init__).
    if stub_llm:
        _install_stub_llm()

    env = MASEnvironment(topology_name=topology_name)

    if stub_llm:
        _stub_tools(env)

    env.execute_task(task)
    return env


def calibrate_clean_episodes(
    topologies: list[str],
    num_episodes: int,
    task_template: str,
    stub_llm: bool = False,
    stub_encoder: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Run ``num_episodes`` clean (no attack simulator) episodes per
    topology and score each with the default detector.

    Clean episodes carry no attack, so ANY evidence is a false
    positive. Returns (episode_rows, response_rows) where
    episode_rows is one row per episode and response_rows is one row
    per agent RESPONSE (per-response false-positive rate, P5).
    """

    detector = SecurityDetector()
    rows: list[dict] = []
    response_rows: list[dict] = []

    for topology in topologies:
        for index in range(num_episodes):

            task = task_template.format(index=index)

            try:
                env = _run_clean_episode(
                    topology,
                    task,
                    stub_llm=stub_llm,
                    stub_encoder=stub_encoder,
                )
            except Exception as exc:  # pragma: no cover - environment/LLM dependent
                rows.append(
                    {
                        "topology": topology,
                        "episode": index,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            events = env.get_observable_events()
            artifacts = env.get_observable_artifacts()

            result = detector.detect(events, artifacts=artifacts)

            rows.append(
                {
                    "topology": topology,
                    "episode": index,
                    "status": "ok",
                    "evidence_present": result["evidence_present"],
                    "content_evidence_count": result["content_evidence_count"],
                    "injection_evidence_count": result[
                        "injection_evidence_count"
                    ],
                    "high_confidence_evidence_count": result[
                        "high_confidence_evidence_count"
                    ],
                    "untrusted_source_evidence_count": result[
                        "untrusted_source_evidence_count"
                    ],
                    **{
                        f"category:{category}": count
                        for category, count in result[
                            "content_category_counts"
                        ].items()
                    },
                }
            )

            # Per-RESPONSE false positives: each recorded Observation
            # carries its own per-response detector_result (P5).
            for observation in getattr(env, "security_observations", []):
                detector_result = observation.detector_result or {}
                response_rows.append(
                    {
                        "topology": topology,
                        "episode": index,
                        "agent_id": observation.agent_id,
                        "evidence_present": bool(
                            detector_result.get("evidence_present")
                        ),
                        "injection_evidence_count": detector_result.get(
                            "injection_evidence_count", 0
                        ),
                        "content_evidence_count": detector_result.get(
                            "content_evidence_count", 0
                        ),
                        **{
                            f"category:{category}": count
                            for category, count in (
                                detector_result.get(
                                    "content_category_counts", {}
                                )
                            ).items()
                        },
                    }
                )

    return rows, response_rows


def tpr_on_held_out_variants() -> tuple[float, list[dict]]:
    """
    TPR of the default GENERIC-only detector on the Task 2c held-out
    payload variants (never used to tune detector patterns).
    """

    rows: list[dict] = []
    hits = 0

    for variant in PAYLOAD_VARIANTS:

        evidence = scan_text(variant.text, use_template_signatures=False)
        detected = evidence.total > 0
        hits += int(detected)

        rows.append(
            {
                "variant_id": variant.variant_id,
                "category": variant.category,
                "detected": detected,
                "matched_examples": ";".join(evidence.matched_examples),
            }
        )

    tpr = hits / len(PAYLOAD_VARIANTS) if PAYLOAD_VARIANTS else 0.0
    return tpr, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_token_baseline(
    topologies: list[str],
    num_episodes: int,
    task_template: str,
    output_path: str | Path,
    seed: int,
    stub_llm: bool = False,
    stub_encoder: bool = False,
) -> dict:
    """
    Run clean episodes and write ``configs/token_baseline.json`` (the
    per-agent mean/std token usage the detector uses for
    ``token_spike_count``; see security/detector.py).

    The file is marked ``generated`` with the seed, date and episode
    count so its provenance is explicit and reproducible.

    Uses only the OBSERVABLE ``llm_usage`` events. Ground truth is not
    involved. Returns the written mapping (empty if no episode
    completed).
    """

    tokens_by_agent: dict[str, list[int]] = {}

    for topology in topologies:
        for index in range(num_episodes):

            task = task_template.format(index=index)
            try:
                env = _run_clean_episode(
                    topology,
                    task,
                    stub_llm=stub_llm,
                    stub_encoder=stub_encoder,
                )
            except Exception:  # pragma: no cover - env/LLM dependent
                continue
            for event in env.get_observable_events():
                usage = event.get("token_usage") or 0
                agent = event.get("sender")
                if agent and usage:
                    tokens_by_agent.setdefault(agent, []).append(int(usage))

    baseline: dict[str, dict[str, float]] = {}
    for agent, values in tokens_by_agent.items():
        if len(values) < 2:
            continue
        baseline[agent] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }

    payload = {
        "generated": True,
        "seed": seed,
        "date": date.today().isoformat(),
        "episodes_per_topology": num_episodes,
        "topologies": list(topologies),
        "task_template": task_template,
        "stub_llm": stub_llm,
        "agents": baseline,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return payload

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=list(TOPOLOGIES),
    )
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument(
        "--task",
        default=(
            "Write a short proposal about renewable energy adoption "
            "in mid-size cities (run {index})."
        ),
    )
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--stub-llm",
        action="store_true",
        help=(
            "Run with a deterministic stub LLM and stub tools (no "
            "Ollama, no network) for offline dry runs."
        ),
    )
    parser.add_argument(
        "--stub-encoder",
        action="store_true",
        help="Use a deterministic stub semantic encoder (no download).",
    )
    parser.add_argument(
        "--token-baseline-out",
        default="configs/token_baseline.json",
        help="Where to write the generated per-agent token baseline.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TOKEN_BASELINE_SEED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "=== Clean-episode false-positive calibration "
        "(report only; no thresholds/weights changed) ==="
    )

    episode_rows, response_rows = calibrate_clean_episodes(
        args.topologies,
        args.num_episodes,
        args.task,
        stub_llm=args.stub_llm,
        stub_encoder=args.stub_encoder,
    )
    _write_csv(out_dir / "calibrate_detector_episodes.csv", episode_rows)
    _write_csv(out_dir / "calibrate_detector_responses.csv", response_rows)

    ok_rows = [row for row in episode_rows if row["status"] == "ok"]
    error_rows = [row for row in episode_rows if row["status"] != "ok"]

    if ok_rows:
        fp_episodes = sum(1 for row in ok_rows if row["evidence_present"])
        print(
            f"Per-episode FP rate: {fp_episodes}/{len(ok_rows)} "
            f"= {fp_episodes / len(ok_rows):.2%}"
        )

        category_counts: dict[str, int] = {}
        for row in ok_rows:
            for key, value in row.items():
                if key.startswith("category:"):
                    category_counts[key] = (
                        category_counts.get(key, 0) + (1 if value else 0)
                    )
        for key in sorted(category_counts):
            print(
                f"  per-episode {key}: "
                f"{category_counts[key]}/{len(ok_rows)} episodes"
            )
    else:
        print(
            "No clean episodes completed successfully; FP rate "
            "unavailable (see errors below). This script requires a "
            "live LLM backend (agents.llm.get_llm) to run full "
            "episodes, or pass --stub-llm for an offline dry run."
        )

    if response_rows:
        fp_responses = sum(
            1 for row in response_rows if row["evidence_present"]
        )
        print(
            f"Per-response FP rate: {fp_responses}/{len(response_rows)} "
            f"= {fp_responses / len(response_rows):.2%}"
        )
        response_categories: dict[str, int] = {}
        for row in response_rows:
            for key, value in row.items():
                if key.startswith("category:"):
                    response_categories[key] = (
                        response_categories.get(key, 0) + (1 if value else 0)
                    )
        for key in sorted(response_categories):
            print(
                f"  per-response {key}: "
                f"{response_categories[key]}/{len(response_rows)} responses"
            )

    for row in error_rows:
        print(
            f"  [SKIPPED] {row['topology']} episode {row['episode']}: "
            f"{row['error']}"
        )

    print()
    print(
        "=== TPR on held-out payload variants (Task 2c), "
        "GENERIC-only detector ==="
    )
    tpr, variant_rows = tpr_on_held_out_variants()
    _write_csv(out_dir / "calibrate_detector_variants.csv", variant_rows)
    print(f"TPR: {tpr:.2%} ({sum(r['detected'] for r in variant_rows)}/{len(variant_rows)})")
    for row in variant_rows:
        print(
            f"  {row['variant_id']:24s} [{row['category']:20s}] "
            f"detected={row['detected']}"
        )

    print()
    print(
        "=== Per-agent clean-run token baseline "
        "(configs/token_baseline.json) ==="
    )
    baseline = build_token_baseline(
        args.topologies,
        args.num_episodes,
        args.task,
        args.token_baseline_out,
        seed=args.seed,
        stub_llm=args.stub_llm,
        stub_encoder=args.stub_encoder,
    )
    if baseline["agents"]:
        for agent, stats in sorted(baseline["agents"].items()):
            print(
                f"  {agent:12s} mean={stats['mean']:.1f} "
                f"std={stats['std']:.1f} n={stats['n']}"
            )
    else:
        print(
            "  no token baseline written (no episodes completed; "
            "use --stub-llm for an offline run)"
        )
    print(
        f"  wrote {args.token_baseline_out} "
        f"(generated={baseline['generated']}, seed={baseline['seed']}, "
        f"date={baseline['date']})"
    )

    print()
    print(f"CSV reports written to {out_dir}/")

if __name__ == "__main__":
    main()
