"""
experiments/security_dashboard.py

A GUI dashboard (Tkinter, stdlib -- no new dependency) that shows, for a
run of the tiered security observer:

  * which AGENT produced each output,
  * each SECTION (chunk) of that output,
  * the SEMANTIC (Tier 1) result for that chunk,
  * the NLI (Tier 2) result for that chunk,
  * the JUDGE (Tier 3) result where the gate fired,
  * and how all of it is INCORPORATED into the observer's fused
    security-state result (rule evidence + semantic deviation -> the
    fused security_score and the investigation decision).

Design
------
The data extraction (``build_dashboard_rows`` / ``EpisodeDashboard``) is
pure and headless, so it is unit-testable without a display. The Tk
layer is a thin renderer over those rows.

Data sources
------------
Either run the pipeline from a JSONL of records (same shape as
experiments/show_security_logs.py: condition/task/subtask/output/evidence),
or run a LIVE episode (needs Ollama). The dashboard itself only consumes
``TieredObservation`` objects.

Usage
-----
    # offline demo (stub encoder + stub NLI + stub judge)
    python -m experiments.security_dashboard --stub

    # your own records
    python -m experiments.security_dashboard --stub --from-jsonl my.jsonl

    # headless: dump the same data the GUI shows, then exit
    python -m experiments.security_dashboard --stub --dump

    # real models (NLI download + Ollama)
    python -m experiments.security_dashboard --strategy formula
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_TASK = "Write a proposal on AI in cyber security"


# =================================================================
# PURE DATA LAYER (headless, testable)
# =================================================================

@dataclass
class ChunkRow:
    """Everything the dashboard shows for one section (chunk)."""

    chunk_index: int
    chunk_text: str
    tiers_ran: list[str] = field(default_factory=list)

    # ---- Tier 1: chunked semantic ----
    semantic_assessed: bool = False
    task_similarity: float = 0.0
    subtask_similarity: float = 0.0
    objective_deviation: float = 0.0
    scope_deviation: float = 0.0
    deviation_score: float = 0.0
    semantic_confidence: float = 0.0

    # ---- Tier 2: NLI contradiction ----
    nli_ran: bool = False
    nli_label: str = ""
    nli_confidence: float = 0.0
    nli_premise: str = ""
    nli_hypothesis: str = ""

    # ---- Tier 3: judge ----
    tier3_ran: bool = False
    tier3_contradicts: Optional[bool] = None
    tier3_reasoning: str = ""
    tier3_stubbed: bool = False


@dataclass
class AgentRow:
    """One agent response: its sections plus the fused state."""

    agent_id: str
    original_task: str
    assigned_subtask: str
    response_preview: str

    # Fused observer state for the whole response.
    content_score: float = 0.0
    semantic_score: float = 0.0
    security_score: float = 0.0
    investigation_required: bool = False

    detector_result: dict[str, Any] = field(default_factory=dict)

    # Per-response aggregates.
    worst_chunk_deviation: float = 0.0
    worst_chunk_index: int = -1
    contradiction_flagged_chunks: int = 0
    tier3_invocations: int = 0

    chunks: list[ChunkRow] = field(default_factory=list)


@dataclass
class EpisodeDashboard:
    """All agent rows for one run, plus episode roll-ups."""

    strategy: str = ""
    agents: list[AgentRow] = field(default_factory=list)

    # ---- roll-ups ----
    @property
    def worst_chunk_deviation(self) -> float:
        return max(
            (a.worst_chunk_deviation for a in self.agents), default=0.0
        )

    @property
    def total_contradiction_flagged_chunks(self) -> int:
        return sum(a.contradiction_flagged_chunks for a in self.agents)

    @property
    def total_tier3_invocations(self) -> int:
        return sum(a.tier3_invocations for a in self.agents)

    @property
    def total_judge_contradictions(self) -> int:
        return sum(
            1
            for a in self.agents
            for c in a.chunks
            if c.tier3_contradicts is True
        )


def _content_score(detector_result: dict) -> float:
    """
    The rule/content evidence score the observer fuses (mirrors
    experiments/eval_detector.py::_content_score so the dashboard shows
    the same rule component the fused score uses). NOT the 0.45 weight.
    """

    score = 0.0
    if detector_result.get("evidence_present"):
        score += 0.35
    score += min(
        float(detector_result.get("injection_evidence_count", 0)) * 0.10,
        0.30,
    )
    score += min(
        float(detector_result.get("untrusted_source_evidence_count", 0)) * 0.05,
        0.15,
    )
    return float(min(1.0, max(0.0, score)))


def chunk_row_from_decision(decision) -> ChunkRow:
    """Convert a ``ChunkDecision`` into a dashboard ``ChunkRow``."""

    row = ChunkRow(
        chunk_index=decision.chunk_index,
        chunk_text=decision.chunk_text,
        tiers_ran=list(decision.tiers_ran),
    )

    semantic = decision.semantic
    if semantic is not None and getattr(semantic, "assessed", False):
        row.semantic_assessed = True
        row.task_similarity = float(semantic.task_similarity)
        row.subtask_similarity = float(semantic.subtask_similarity)
        row.objective_deviation = float(semantic.objective_deviation)
        row.scope_deviation = float(semantic.scope_deviation)
        row.deviation_score = float(semantic.deviation_score)
        row.semantic_confidence = float(semantic.confidence)

    contradiction = decision.contradiction
    if contradiction is not None:
        row.nli_ran = True
        row.nli_label = contradiction.label
        row.nli_confidence = float(contradiction.confidence)
        row.nli_premise = contradiction.premise
        row.nli_hypothesis = contradiction.hypothesis

    verdict = decision.tier3_verdict
    if verdict is not None:
        row.tier3_ran = True
        row.tier3_contradicts = bool(verdict.contradicts_evidence)
        row.tier3_reasoning = verdict.reasoning
        row.tier3_stubbed = bool(verdict.stubbed)

    return row


def agent_row_from_tiered(tiered) -> AgentRow:
    """Convert a ``TieredObservation`` into a dashboard ``AgentRow``."""

    base = tiered.base
    detector_result = dict(base.detector_result or {})

    semantic_assessment = base.semantic_assessment
    semantic_score = 0.0
    if semantic_assessment is not None:
        semantic_score = (
            semantic_assessment.deviation_score
            * semantic_assessment.confidence
        )

    worst_index = -1
    if tiered.chunked is not None and tiered.chunked.worst_chunk_index >= 0:
        worst_index = tiered.chunked.worst_chunk_index

    return AgentRow(
        agent_id=base.agent_id,
        original_task=base.original_task,
        assigned_subtask=base.assigned_subtask,
        response_preview=(base.response or "")[:160],
        content_score=_content_score(detector_result),
        semantic_score=float(semantic_score),
        security_score=float(base.security_score),
        investigation_required=bool(base.investigation_required),
        detector_result=detector_result,
        worst_chunk_deviation=float(tiered.worst_chunk_deviation),
        worst_chunk_index=worst_index,
        contradiction_flagged_chunks=int(
            tiered.contradiction_flagged_chunks
        ),
        tier3_invocations=int(tiered.tier3_invocations),
        chunks=[chunk_row_from_decision(d) for d in tiered.chunk_decisions],
    )


def build_dashboard_rows(
    tiered_observations: Sequence[Any],
    strategy: str = "",
) -> EpisodeDashboard:
    """
    Build the full dashboard model from a sequence of
    ``TieredObservation`` objects. Pure: no I/O, no display.
    """

    return EpisodeDashboard(
        strategy=strategy,
        agents=[agent_row_from_tiered(t) for t in tiered_observations],
    )


# =================================================================
# RUNNERS (produce TieredObservation objects)
# =================================================================

def _build_assessor(stub: bool):
    from security.semantic_assessor import SemanticAssessor

    if stub:
        import numpy as np

        class _StubEncoder:
            KEYWORDS = (
                "security", "ai", "threat", "detection", "network",
                "monitoring", "malicious",
                "cookie", "recipe", "butter", "sugar", "bake", "oven",
            )

            def encode(self, texts, convert_to_numpy=True,
                       normalize_embeddings=False):
                rows = [
                    [float(str(t).lower().count(k)) for k in self.KEYWORDS]
                    for t in texts
                ]
                array = np.asarray(rows, dtype=np.float32)
                if array.size and not array.any(axis=1).all():
                    array[~array.any(axis=1), 0] = 1e-6
                return array

        return SemanticAssessor(model=_StubEncoder())

    return SemanticAssessor()


def _build_nli_checker(stub_nli: bool):
    from experiments.show_security_logs import (
        _StubNLIModel,
    )
    from security.contradiction_checker import ContradictionChecker

    if stub_nli:
        return ContradictionChecker(model=_StubNLIModel())
    return ContradictionChecker()


def _build_judge(stub_judge: bool):
    from security.llm_judge import LLMJudge

    return LLMJudge(stub=True) if stub_judge else LLMJudge()


def _build_allocator(strategy: str):
    from security.resource_allocator import AllocationState, ResourceAllocator

    return ResourceAllocator(strategy, state=AllocationState())


def run_from_records(
    records: Sequence[dict],
    *,
    stub_encoder: bool,
    stub_nli: bool,
    stub_judge: bool,
    strategy: str,
) -> list[Any]:
    """Run the tiered observer over JSONL-style records (offline)."""

    from security.observer import SecurityObserver

    observer = SecurityObserver(
        semantic_assessor=_build_assessor(stub_encoder),
        contradiction_checker=_build_nli_checker(stub_nli),
        llm_judge=_build_judge(stub_judge),
        resource_allocator=_build_allocator(strategy),
        log_enabled=False,
    )

    tiered_observations = []
    for record in records:
        tiered_observations.append(
            observer.observe_tiered(
                agent_id=str(record.get("condition", "agent")),
                response=str(record.get("output", "")),
                original_task=str(record.get("task", DEFAULT_TASK)),
                assigned_subtask=str(
                    record.get("subtask", record.get("condition", "agent"))
                ),
                evidence_chunks=list(record.get("evidence", []) or []),
                events=list(record.get("events", [])),
            )
        )
    return tiered_observations


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    if not records:
        raise RuntimeError(f"No records found in {path}.")
    return records


# =================================================================
# HEADLESS DUMP
# =================================================================

def dump_dashboard(dashboard: EpisodeDashboard) -> None:
    """Print the dashboard model as clear text (the headless twin)."""

    for agent in dashboard.agents:
        print("=" * 78)
        print(f"AGENT: {agent.agent_id}   subtask={agent.assigned_subtask!r}")
        print(
            f"  fused: rule={agent.content_score:.4f} "
            f"semantic={agent.semantic_score:.4f} "
            f"-> security_score={agent.security_score:.4f} "
            f"investigation_required={agent.investigation_required}"
        )
        print(
            f"  worst_chunk_deviation={agent.worst_chunk_deviation:.4f} "
            f"contradiction_flagged_chunks="
            f"{agent.contradiction_flagged_chunks} "
            f"tier3_invocations={agent.tier3_invocations}"
        )
        for chunk in agent.chunks:
            print("-" * 78)
            print(
                f"  SECTION chunk[{chunk.chunk_index}] "
                f"tiers_ran={','.join(chunk.tiers_ran)}"
            )
            text = chunk.chunk_text[:120].replace("\n", " ")
            print(f"    text      : {text!r}")
            if chunk.semantic_assessed:
                print(
                    f"    Tier1 sem : deviation={chunk.deviation_score:.4f} "
                    f"task_sim={chunk.task_similarity:.4f} "
                    f"subtask_sim={chunk.subtask_similarity:.4f} "
                    f"conf={chunk.semantic_confidence:.4f}"
                )
            else:
                print("    Tier1 sem : (not assessed)")
            if chunk.nli_ran:
                print(
                    f"    Tier2 NLI : label={chunk.nli_label} "
                    f"conf={chunk.nli_confidence:.4f} "
                    f"(premise=evidence, hypothesis=chunk)"
                )
            else:
                print("    Tier2 NLI : (not run)")
            if chunk.tier3_ran:
                print(
                    f"    Tier3 judge: contradicts="
                    f"{chunk.tier3_contradicts}"
                )
                print(f"               reasoning={chunk.tier3_reasoning!r}")
            else:
                print("    Tier3 judge: (not run)")
    print("=" * 78)
    print(
        f"EPISODE ROLL-UP  worst_chunk_deviation="
        f"{dashboard.worst_chunk_deviation:.4f} "
        f"contradiction_flagged_chunks="
        f"{dashboard.total_contradiction_flagged_chunks} "
        f"tier3_invocations={dashboard.total_tier3_invocations} "
        f"judge_contradictions={dashboard.total_judge_contradictions}"
    )


# =================================================================
# TKINTER GUI (thin renderer)
# =================================================================

def launch_gui(dashboard: EpisodeDashboard) -> int:
    """Render the dashboard in a Tkinter window."""

    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Security Observer Dashboard -- Tiered Pipeline")
    root.geometry("1280x820")

    # ---- top: strategy + episode roll-up ----
    header = tk.Frame(root, padx=10, pady=8)
    header.pack(fill="x")

    tk.Label(
        header,
        text=f"strategy: {dashboard.strategy or '(default)'}",
        font=("Segoe UI", 11, "bold"),
    ).pack(side="left")

    rollup = (
        f"   worst_chunk_deviation={dashboard.worst_chunk_deviation:.4f}"
        f"   contradiction_flagged_chunks="
        f"{dashboard.total_contradiction_flagged_chunks}"
        f"   tier3_invocations={dashboard.total_tier3_invocations}"
        f"   judge_contradictions={dashboard.total_judge_contradictions}"
    )
    tk.Label(header, text=rollup, font=("Consolas", 10)).pack(side="left")

    # ---- left: agent list ----
    panes = tk.PanedWindow(root, orient="horizontal", sashwidth=6)
    panes.pack(fill="both", expand=True, padx=8, pady=6)

    left = tk.Frame(panes)
    tk.Label(left, text="AGENTS", font=("Segoe UI", 10, "bold")).pack(
        anchor="w", padx=6, pady=(4, 2)
    )
    agent_list = tk.Listbox(left, width=34, font=("Consolas", 10))
    agent_list.pack(fill="both", expand=True, padx=6, pady=4)
    for agent in dashboard.agents:
        flag = "  [!investigate]" if agent.investigation_required else ""
        agent_list.insert(
            "end",
            f"{agent.agent_id}  score={agent.security_score:.3f}{flag}",
        )
    panes.add(left)

    # ---- right: detail ----
    right = tk.Frame(panes)
    panes.add(right)

    detail_title = tk.Label(
        right, text="Select an agent", font=("Segoe UI", 10, "bold"),
        anchor="w",
    )
    detail_title.pack(fill="x", padx=6, pady=(4, 0))

    fused_label = tk.Label(
        right, text="", font=("Consolas", 10), anchor="w", justify="left",
    )
    fused_label.pack(fill="x", padx=6)

    columns = (
        "chunk", "tiers", "deviation", "nli_label", "nli_conf", "judge",
    )
    tree = ttk.Treeview(right, columns=columns, show="headings", height=8)
    headings = {
        "chunk": ("section", 70),
        "tiers": ("tiers_ran", 150),
        "deviation": ("T1 deviation", 110),
        "nli_label": ("T2 NLI label", 130),
        "nli_conf": ("T2 conf", 90),
        "judge": ("T3 judge", 110),
    }
    for key, (text, width) in headings.items():
        tree.heading(key, text=text)
        tree.column(key, width=width, anchor="w")
    tree.pack(fill="x", padx=6, pady=4)

    section_text = tk.Text(right, height=16, font=("Consolas", 10), wrap="word")
    section_text.pack(fill="both", expand=True, padx=6, pady=(0, 8))

    def show_agent(index: int) -> None:
        agent = dashboard.agents[index]
        detail_title.config(
            text=f"AGENT: {agent.agent_id}   subtask={agent.assigned_subtask!r}"
        )
        fused_label.config(
            text=(
                f"FUSED SECURITY STATE:  rule(content)={agent.content_score:.4f}"
                f"  +  semantic(deviation*conf)={agent.semantic_score:.4f}"
                f"   ->  security_score={agent.security_score:.4f}"
                f"   investigation_required={agent.investigation_required}\n"
                f"per-response:  worst_chunk_deviation="
                f"{agent.worst_chunk_deviation:.4f}  "
                f"contradiction_flagged_chunks="
                f"{agent.contradiction_flagged_chunks}  "
                f"tier3_invocations={agent.tier3_invocations}"
            )
        )
        tree.delete(*tree.get_children())
        for chunk in agent.chunks:
            deviation = (
                f"{chunk.deviation_score:.3f}" if chunk.semantic_assessed
                else "-"
            )
            nli_label = chunk.nli_label if chunk.nli_ran else "-"
            nli_conf = (
                f"{chunk.nli_confidence:.3f}" if chunk.nli_ran else "-"
            )
            if chunk.tier3_ran:
                judge = (
                    "CONTRADICTS" if chunk.tier3_contradicts else "ok"
                )
            else:
                judge = "-"
            tree.insert(
                "", "end",
                values=(
                    f"[{chunk.chunk_index}]", ",".join(chunk.tiers_ran),
                    deviation, nli_label, nli_conf, judge,
                ),
            )

        # full per-section text
        section_text.delete("1.0", "end")
        for chunk in agent.chunks:
            section_text.insert("end", f"--- section [{chunk.chunk_index}] ---\n")
            section_text.insert("end", f"{chunk.chunk_text}\n")
            if chunk.semantic_assessed:
                section_text.insert(
                    "end",
                    f"  T1 semantic: deviation={chunk.deviation_score:.4f} "
                    f"task_sim={chunk.task_similarity:.4f} "
                    f"subtask_sim={chunk.subtask_similarity:.4f} "
                    f"conf={chunk.semantic_confidence:.4f}\n",
                )
            if chunk.nli_ran:
                section_text.insert(
                    "end",
                    f"  T2 NLI: {chunk.nli_label} "
                    f"({chunk.nli_confidence:.3f}) "
                    f"[premise=evidence, hypothesis=chunk]\n",
                )
                section_text.insert(
                    "end", f"    premise   : {chunk.nli_premise[:100]!r}\n"
                )
                section_text.insert(
                    "end", f"    hypothesis: {chunk.nli_hypothesis[:100]!r}\n"
                )
            if chunk.tier3_ran:
                section_text.insert(
                    "end",
                    f"  T3 judge: contradicts={chunk.tier3_contradicts} "
                    f"reasoning={chunk.tier3_reasoning!r}"
                    f"{' (stub)' if chunk.tier3_stubbed else ''}\n",
                )
            section_text.insert("end", "\n")

    agent_list.bind(
        "<<ListboxSelect>>",
        lambda _event: (
            show_agent(agent_list.curselection()[0])
            if agent_list.curselection() else None
        ),
    )

    if dashboard.agents:
        agent_list.selection_set(0)
        show_agent(0)

    root.mainloop()
    return 0


# =================================================================
# MAIN
# =================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--topology", default="centralized")
    parser.add_argument("--stub", action="store_true",
                        help="Stub encoder (offline).")
    parser.add_argument("--stub-nli", action="store_true",
                        help="Stub NLI checker (offline).")
    parser.add_argument("--stub-judge", action="store_true",
                        help="Stub Tier-3 judge (offline).")
    parser.add_argument("--strategy", default="formula",
                        choices=["no_investigation", "brute_force", "formula"])
    parser.add_argument("--from-jsonl", type=Path, default=None)
    parser.add_argument("--write-template", action="store_true",
                        help="Write a demo JSONL and exit.")
    parser.add_argument("--dump", action="store_true",
                        help="Print the dashboard headlessly and exit "
                             "(no GUI).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.write_template:
        from experiments.show_security_logs import write_template

        write_template(Path("outputs/demo_logs.jsonl"))
        return 0

    # Resolve input source.
    if args.from_jsonl is None:
        demo = Path("outputs/demo_logs.jsonl")
        if demo.exists():
            args.from_jsonl = demo
            print(f"[input] using {demo}")

    if args.from_jsonl is not None:
        records = load_records(args.from_jsonl)
        observations = run_from_records(
            records,
            stub_encoder=args.stub,
            stub_nli=args.stub_nli,
            stub_judge=args.stub_judge,
            strategy=args.strategy,
        )
    else:
        print(
            "[input] no --from-jsonl and no outputs/demo_logs.jsonl; "
            "running a LIVE episode (needs Ollama)."
        )
        observations = _run_live_episode(args)

    dashboard = build_dashboard_rows(
        observations, strategy=args.strategy
    )

    if args.dump:
        dump_dashboard(dashboard)
        return 0

    return launch_gui(dashboard)


def _run_live_episode(args) -> list[Any]:
    from environment.mas_environment import MASEnvironment
    from security.observer import SecurityObserver

    observer = SecurityObserver(
        semantic_assessor=_build_assessor(args.stub),
        contradiction_checker=_build_nli_checker(args.stub_nli),
        llm_judge=_build_judge(args.stub_judge),
        resource_allocator=_build_allocator(args.strategy),
        log_enabled=False,
    )
    environment = MASEnvironment(
        topology_name=args.topology, security_observer=observer
    )
    environment.execute_task(args.task)

    # Drive the tiered path per response, using the observable artifacts
    # actually delivered to each agent (one hop back; see Task J).
    artifacts = environment.get_observable_artifacts()

    observations = []
    for base in environment.security_observations:
        evidence_chunks = [
            artifact.get("text") or ""
            for artifact in artifacts
            if artifact.get("receiver") == base.agent_id
        ]
        observations.append(
            observer.observe_tiered(
                agent_id=base.agent_id,
                response=base.response
                if isinstance(base.response, str) else str(base.response),
                original_task=base.original_task,
                assigned_subtask=base.assigned_subtask,
                evidence_chunks=evidence_chunks,
                events=[],
            )
        )
    return observations


if __name__ == "__main__":
    raise SystemExit(main())