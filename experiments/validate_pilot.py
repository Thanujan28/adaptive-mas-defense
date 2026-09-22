"""
experiments/validate_pilot.py

Pilot-set validation for the tiered defender (Task O).

Runs Tier 2 (security/contradiction_checker.py, NLI) and Tier 3
(security/llm_judge.py) against tests/fixtures/judge_pilot_set.jsonl, a
hand-labelable set of (chunk, evidence, task) triples on the AI-in-
cybersecurity topic. The file ships with ``human_label`` null in every
row; the USER fills those in by hand.

Behaviour:
  * For each row, run Tier 2 and (optionally) Tier 3.
  * Where ``human_label`` is filled in, compare both tiers against it and
    report agreement.
  * Where ``human_label`` is null, SKIP scoring for that row and print an
    explicit warning -- the script never auto-generates labels.

Human label convention (documented here, for the user filling the file):
  ``human_label`` is a short string, one of:
    "contradiction"  - the chunk contradicts the evidence
    "entailment"     - the evidence entails the chunk
    "neutral"        - neither (including unsupported-but-not-contradicted)

Usage:
    # dry run, no models: stub NLI + stub judge
    python -m experiments.validate_pilot --stub-nli --stub-judge

    # real run (loads the NLI model and, if not --stub-judge, Ollama)
    python -m experiments.validate_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

DEFAULT_PILOT_PATH = Path("tests/fixtures/judge_pilot_set.jsonl")

VALID_LABELS = {"contradiction", "entailment", "neutral"}


# =============================================================
# COLLABORATORS
# =============================================================

class _StubNLIModel:
    """
    Deterministic, download-free NLI stand-in for --stub-nli.

    Mirrors the rule set used elsewhere in the repo: an explicit
    negation against an asserted term is a contradiction; a shared
    core-claim term is entailment; otherwise neutral.
    """

    def __call__(self, inputs, truncation=True):
        premise = str(inputs.get("text", "")).lower()
        hypothesis = str(inputs.get("text_pair", "")).lower()

        if (
            any(c in hypothesis for c in ("no benefit", "not ", "never", "cannot", "no "))
            and any(c in premise for c in ("benefit", "detect", "response", "improve", "help"))
        ):
            return [
                {"label": "contradiction", "score": 0.95},
                {"label": "entailment", "score": 0.03},
                {"label": "neutral", "score": 0.02},
            ]

        shared = sum(
            1
            for term in ("detect", "threat", "response", "security", "ai", "monitoring", "ml")
            if term in hypothesis and term in premise
        )
        if shared >= 1:
            return [
                {"label": "entailment", "score": 0.9},
                {"label": "neutral", "score": 0.07},
                {"label": "contradiction", "score": 0.03},
            ]

        return [
            {"label": "neutral", "score": 0.85},
            {"label": "entailment", "score": 0.1},
            {"label": "contradiction", "score": 0.05},
        ]


def _build_checker(stub_nli: bool):
    from security.contradiction_checker import ContradictionChecker

    if stub_nli:
        print("[nli] using deterministic STUB NLI checker (no download)")
        return ContradictionChecker(model=_StubNLIModel())
    print("[nli] using the real NLI model (roberta-large-mnli)")
    return ContradictionChecker()


def _build_judge(stub_judge: bool):
    from security.llm_judge import LLMJudge

    if stub_judge:
        print("[judge] using deterministic STUB judge (no LLM backend)")
        return LLMJudge(stub=True)
    print("[judge] using the real LLM judge (Ollama, temperature 0)")
    return LLMJudge()


# =============================================================
# IO
# =============================================================

def load_pilot_set(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    if not rows:
        raise RuntimeError(f"No rows found in {path}.")
    return rows


def _normalise_label(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


# =============================================================
# VALIDATION
# =============================================================

def validate(rows: list[dict], checker, judge) -> dict:
    """
    Run both tiers over ``rows`` and compare against human labels where
    present. Returns a summary dict.
    """

    total = len(rows)
    labelled = 0
    skipped_null = 0
    tier2_agree = 0
    tier3_agree = 0
    tier2_scored = 0
    tier3_scored = 0
    unlabelled_ids: list[str] = []

    print()
    print(f"{'id':12s} | {'tier2':12s} | {'tier3':12s} | "
          f"{'human':12s} | {'match':6s}")
    print("-" * 70)

    for row in rows:
        row_id = str(row.get("id", "?"))
        chunk = row.get("chunk", "")
        evidence = row.get("evidence", "")
        task = row.get("task", "")
        human = _normalise_label(row.get("human_label"))

        # Tier 2: NLI contradiction of the chunk against the evidence.
        tier2 = checker.check_chunk_against_evidence(
            chunk_text=chunk,
            evidence_chunks=[evidence] if evidence else [],
        )
        tier2_label = tier2.label

        # Tier 3: judge. Contradiction iff the judge says so; otherwise
        # report the Tier-2 label's complement as entailment/neutral is
        # not decided by the judge (it only answers the contradiction
        # question), so a non-contradiction is reported as "neutral" for
        # comparison purposes and flagged in the note column.
        verdict = judge.judge(
            chunk_text=chunk,
            evidence_text=evidence,
            task=task,
        )
        tier3_label = "contradiction" if verdict.contradicts_evidence else "neutral"

        if human is None:
            skipped_null += 1
            unlabelled_ids.append(row_id)
            print(
                f"{row_id:12s} | {tier2_label:12s} | {tier3_label:12s} | "
                f"{'(null)':12s} | SKIP"
            )
            continue

        if human not in VALID_LABELS:
            print(
                f"{row_id:12s} | {tier2_label:12s} | {tier3_label:12s} | "
                f"{human:12s} | SKIP (invalid human_label)"
            )
            continue

        labelled += 1

        tier2_scored += 1
        if tier2_label == human:
            tier2_agree += 1

        tier3_scored += 1
        if tier3_label == human:
            tier3_agree += 1

        print(
            f"{row_id:12s} | {tier2_label:12s} | {tier3_label:12s} | "
            f"{human:12s} | "
            f"{'t2' if tier2_label == human else '  '}"
            f"{' t3' if tier3_label == human else ''}"
        )

    print()
    if labelled == 0:
        print(
            f"0 labeled rows out of {total}; all {skipped_null} rows have "
            "human_label=null and were SKIPPED. Fill in "
            "tests/fixtures/judge_pilot_set.jsonl by hand, then re-run."
        )
    else:
        print(
            f"Labeled rows: {labelled}/{total} "
            f"(skipped {skipped_null} null)."
        )
        print(
            f"  Tier 2 (NLI) agreement: {tier2_agree}/{tier2_scored} "
            f"= {tier2_agree / tier2_scored:.1%}"
        )
        print(
            f"  Tier 3 (judge) agreement: {tier3_agree}/{tier3_scored} "
            f"= {tier3_agree / tier3_scored:.1%}"
        )

    if unlabelled_ids:
        print()
        print(
            "WARNING: the following rows are unlabeled and were excluded "
            f"from agreement: {', '.join(unlabelled_ids)}"
        )

    return {
        "total": total,
        "labelled": labelled,
        "skipped_null": skipped_null,
        "tier2_agree": tier2_agree,
        "tier2_scored": tier2_scored,
        "tier3_agree": tier3_agree,
        "tier3_scored": tier3_scored,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-set",
        type=Path,
        default=DEFAULT_PILOT_PATH,
        help="Path to the pilot JSONL.",
    )
    parser.add_argument(
        "--stub-nli",
        action="store_true",
        help="Deterministic stub NLI checker (no download).",
    )
    parser.add_argument(
        "--stub-judge",
        action="store_true",
        help="Deterministic stub Tier-3 judge (no LLM backend).",
    )
    args = parser.parse_args(argv)

    if not args.pilot_set.exists():
        raise SystemExit(f"Pilot set not found: {args.pilot_set}")

    print(f"[input] reading pilot set: {args.pilot_set}")
    rows = load_pilot_set(args.pilot_set)
    print(f"[input] {len(rows)} rows")

    checker = _build_checker(args.stub_nli)
    judge = _build_judge(args.stub_judge)

    summary = validate(rows, checker, judge)
    print()
    print("=== summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())