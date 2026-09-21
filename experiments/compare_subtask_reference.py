"""
Before/after comparison of the subtask reference used for
subtask_similarity.

NOT run in CI.

For CLEAN runs over the four topologies, this script computes, for
each observed agent output, the subtask_similarity under two
references:

  OLD  : the pipeline stage label ("outline", "research", ...)
  NEW  : the trusted instruction template + original task

It prints mean, std and p5/p50/p95 for both, plus the resulting
``security_score`` distribution. It reports only: no thresholds or
weights are changed.

Requires the real sentence-transformers model. Fail clearly if it is
unavailable; pass --stub for a dry run with a synthetic encoder.

Usage:

    python experiments/compare_subtask_reference.py --from-jsonl outputs/samples.jsonl
    python experiments/compare_subtask_reference.py --stub --from-jsonl outputs/samples.jsonl
    python experiments/compare_subtask_reference.py --write-template
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


TOPOLOGIES = ("centralized", "layered", "fully_connected", "shared_pool")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("outputs")
TEMPLATE_NAME = "subtask_reference_samples.jsonl"


@dataclass
class AgentSample:
    topology: str
    agent: str
    stage: str
    task: str
    template: str
    output: str
    upstream: str = ""
    resource_state: dict = field(default_factory=dict)


def _build_assessor(stub: bool):
    from security.semantic_assessor import SemanticAssessor

    if stub:
        import numpy as _np

        class _StubEncoder:
            def encode(self, texts, convert_to_numpy=True,
                       normalize_embeddings=False):
                rows = []
                for text in texts:
                    low = str(text).lower()
                    rows.append([
                        float(low.count("security")),
                        float(low.count("outline")),
                        float(low.count("research")),
                        float(low.count("ignore")),
                    ])
                return _np.asarray(rows, dtype=_np.float32)

        print("[assessor] STUB encoder (dry run)")
        return SemanticAssessor(model=_StubEncoder())

    print(f"[assessor] real model: {MODEL_NAME}")
    assessor = SemanticAssessor(model_name=MODEL_NAME)
    try:
        assessor._get_model()
    except Exception as exc:
        raise RuntimeError(
            "Real sentence-transformers model is unavailable. Install "
            "sentence-transformers and ensure the model is cached or "
            "downloadable. Use --stub for a dry run. "
            f"Underlying error: {exc}"
        ) from exc
    return assessor


def load_samples(path: Path) -> list[AgentSample]:
    samples = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            samples.append(
                AgentSample(
                    topology=str(record.get("topology", "centralized")),
                    agent=str(record.get("agent", "agent")),
                    stage=str(record.get("stage", "")),
                    task=str(record.get("task", "")),
                    template=str(record.get("template", "")),
                    output=str(record.get("output", "")),
                    upstream=str(record.get("upstream", "")),
                    resource_state=dict(record.get("resource_state", {})),
                )
            )
    if not samples:
        raise ValueError(f"No samples found in {path}.")
    return samples


def write_template(path: Path) -> None:
    from environment.mas_environment import (
        OUTLINE_ASSIGNMENT_TEMPLATE,
        RESEARCHER_ASSIGNMENT_TEMPLATE,
        EXECUTOR_ASSIGNMENT_TEMPLATE,
    )

    rows = [
        {
            "topology": "centralized",
            "agent": "outline",
            "stage": "outline",
            "task": "Summarize the security risks of prompt injection.",
            "template": OUTLINE_ASSIGNMENT_TEMPLATE,
            "upstream": "Plan: create an outline about security risks.",
            "output": "Outline: 1) Injection vectors 2) Defences 3) Monitoring.",
        },
        {
            "topology": "centralized",
            "agent": "researcher",
            "stage": "research",
            "task": "Summarize the security risks of injection.",
            "template": RESEARCHER_ASSIGNMENT_TEMPLATE,
            "upstream": "OUTLINE TO RESEARCH: Injection vectors; Defences.",
            "output": "Key points: prompt injection lets an attacker "
                      "hijack agent instructions and propagate to peers.",
        },
        {
            "topology": "centralized",
            "agent": "executor",
            "stage": "execution",
            "task": "Summarize the security risks of prompt injection.",
            "template": EXECUTOR_ASSIGNMENT_TEMPLATE,
            "upstream": "Research: key points about injection risk.",
            "output": "Final report: multi-agent systems must validate "
                      "inter-agent messages to resist prompt injection.",
        },
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} template samples to {path}")


def describe(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "p5": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p5": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _score(assessor, task, reference, output) -> float:
    assessment = assessor.assess(
        original_task=task,
        assigned_subtask=reference,
        agent_output=output,
    )
    return assessment


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare old stage-label vs new template subtask reference.",
    )
    parser.add_argument("--from-jsonl", type=Path, default=None)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args(argv)

    if args.write_template:
        write_template(OUTPUT_DIR / TEMPLATE_NAME)
        return 0

    if args.from_jsonl is None:
        raise SystemExit(
            "This script requires --from-jsonl. Run --write-template to "
            "generate a template."
        )

    samples = load_samples(args.from_jsonl)
    assessor = _build_assessor(stub=args.stub)

    print(f"\n{'=' * 72}")
    print("SUBTASK REFERENCE: OLD (stage label) vs NEW (instruction template)")
    print(f"{'=' * 72}")
    print(f"samples: {len(samples)}  topologies: {sorted(set(s.topology for s in samples))}")

    old_sub, old_score = [], []
    new_sub, new_score = [], []

    for sample in samples:
        old = _score(assessor, sample.task, sample.stage, sample.output)
        new = _score(assessor, sample.task,
                    f"{sample.template}\n{sample.task}", sample.output)
        old_sub.append(old.subtask_similarity)
        new_sub.append(new.subtask_similarity)
        old_score.append(old.deviation_score * old.confidence)
        new_score.append(new.deviation_score * new.confidence)

    def _row(label, values):
        d = describe(values)
        print(f"\n{label}")
        print(f"  mean={d['mean']:.4f}  std={d['std']:.4f}  "
              f"p5={d['p5']:.4f}  p50={d['p50']:.4f}  p95={d['p95']:.4f}")

    _row("subtask_similarity  OLD (stage label):", old_sub)
    _row("subtask_similarity  NEW (template)  :", new_sub)
    _row("semantic_score      OLD (dev*conf)  :", old_score)
    _row("semantic_score      NEW (dev*conf)  :", new_score)

    print(f"\n{'-' * 72}")
    print("per-agent subtask_similarity (OLD -> NEW)")
    print(f"{'-' * 72}")
    for sample, o, n in zip(samples, old_sub, new_sub):
        print(f"  [{sample.topology:<16}] {sample.agent:<12} "
              f"stage={sample.stage:<18} {o:.4f} -> {n:.4f}")

    print("\nNote: report only. No thresholds or weights changed.\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"\nERROR: {error}\n", file=sys.stderr)
        sys.exit(2)
