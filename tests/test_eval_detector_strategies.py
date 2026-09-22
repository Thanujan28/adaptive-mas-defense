"""
Tests for the Task-N strategy comparison added to
experiments/eval_detector.py.

These do NOT assert full CSV content -- they assert that the strategy
comparison logic wires the RIGHT numbers into the RIGHT columns, and
that it runs under stubs without touching the network.
"""

from __future__ import annotations

import unittest

from experiments.eval_detector import (
    TIER3_ESTIMATED_COMPLETION_TOKENS,
    TIER3_ESTIMATED_PROMPT_TOKENS,
    StrategySample,
    _build_strategy_rows,
    _strategy_ratio_rows,
)
from security.resource_allocator import AllocationStrategy


def _sample(
    strategy: str,
    tier3: int,
    fused: float,
    exposure: int,
    payload_kind: str = "all",
) -> StrategySample:
    return StrategySample(
        condition="prompt_infection",
        topology="layered",
        seed=1,
        agent_id="researcher",
        strategy=strategy,
        payload_kind=payload_kind,
        fused_score=fused,
        tier3_invocations=tier3,
        exposure_label=exposure,
        outcome_label=None,
        legacy_label=None,
    )


class StrategyRowTests(unittest.TestCase):

    def test_tier3_counts_land_in_the_tier3_column(self):
        samples = [
            _sample("brute_force", 2, 0.9, 1),
            _sample("brute_force", 2, 0.1, 0),
            _sample("formula", 1, 0.9, 1),
            _sample("formula", 0, 0.1, 0),
            _sample("no_investigation", 0, 0.9, 1),
        ]
        rows = _build_strategy_rows(samples)

        def find(strategy, label, payload):
            for row in rows:
                if (
                    row["strategy"] == strategy
                    and row["label"] == label
                    and row["payload"] == payload
                ):
                    return row
            return None

        brute = find("brute_force", "exposure", "all")
        formula = find("formula", "exposure", "all")
        none_ = find("no_investigation", "exposure", "all")

        # Real, summed Tier-3 counts.
        self.assertEqual(brute["tier3_invocations"], 4)
        self.assertEqual(formula["tier3_invocations"], 1)
        self.assertEqual(none_["tier3_invocations"], 0)

        # Estimated tokens derive FROM the real counts, not a constant.
        per_call = (
            TIER3_ESTIMATED_PROMPT_TOKENS + TIER3_ESTIMATED_COMPLETION_TOKENS
        )
        self.assertEqual(brute["estimated_tokens"], 4 * per_call)

    def test_indicator_legacy_is_flagged_as_circularity_reference(self):
        samples = [
            StrategySample(
                condition="prompt_infection",
                topology="layered",
                seed=1,
                agent_id="researcher",
                strategy="formula",
                payload_kind="all",
                fused_score=0.9,
                tier3_invocations=1,
                exposure_label=1,
                outcome_label=0,
                legacy_label=1,
            )
        ]
        rows = _build_strategy_rows(samples)
        legacy = [r for r in rows if r["label"] == "indicator_legacy"]
        self.assertTrue(legacy)
        for row in legacy:
            self.assertIn("circularity", row["note"])

    def test_original_and_variants_are_split(self):
        samples = [
            _sample("formula", 1, 0.9, 1, payload_kind="original"),
            _sample("formula", 1, 0.9, 1, payload_kind="variant:urgent_grammatically_off"),
        ]
        rows = _build_strategy_rows(samples)
        payloads = {r["payload"] for r in rows if r["strategy"] == "formula"}
        self.assertIn("original", payloads)
        self.assertIn("variants", payloads)
        self.assertIn("all", payloads)


class RatioRowTests(unittest.TestCase):

    def test_formula_ratio_row_present_and_correct(self):
        samples = [
            _sample("brute_force", 2, 0.8, 1),
            _sample("brute_force", 2, 0.2, 0),
            _sample("formula", 1, 0.8, 1),
            _sample("formula", 0, 0.2, 0),
        ]
        rows = _build_strategy_rows(samples)
        ratio = [
            r for r in rows if r["strategy"] == "formula_vs_brute_force"
        ]
        self.assertEqual(len(ratio), 1)

        # Both strategies separate the same way (1 positive, 1
        # negative, higher score on the positive), so each has AUROC
        # 1.0 and the recovered ratio is 1.0. Cost fraction = 1/4.
        self.assertAlmostEqual(ratio[0]["auroc"], 1.0)
        self.assertAlmostEqual(ratio[0]["tier3_invocations"], 0.25)

    def test_ratio_rows_empty_without_both_strategies(self):
        rows = [{"strategy": "formula", "label": "exposure",
                 "payload": "all", "auroc": 0.5, "tier3_invocations": 1,
                 "estimated_tokens": 0, "estimated_latency_seconds": 0.0}]
        self.assertEqual(_strategy_ratio_rows(rows), [])


class StrategyEndToEndStubTests(unittest.TestCase):

    def test_run_evaluation_under_stubs_produces_three_strategies(self):
        """
        A single tiny stubbed evaluation must produce strategy rows for
        all three strategies and a ratio row, without network access.
        """

        from experiments.eval_detector import run_evaluation

        response_rows, episode_rows, strategy_rows = run_evaluation(
            topologies=["layered"],
            seeds=[1],
            task="Write a proposal on AI in cyber security",
            stub_encoder=True,
            stub_judge=True,
            stub_llm=True,
            include_variants=False,
            stub_nli=True,
        )

        strategies = {
            row["strategy"]
            for row in strategy_rows
            if row["strategy"] in {
                AllocationStrategy.NO_INVESTIGATION.value,
                AllocationStrategy.BRUTE_FORCE.value,
                AllocationStrategy.FORMULA.value,
            }
        }
        self.assertEqual(
            strategies,
            {
                AllocationStrategy.NO_INVESTIGATION.value,
                AllocationStrategy.BRUTE_FORCE.value,
                AllocationStrategy.FORMULA.value,
            },
        )
        # A ratio row is present.
        self.assertTrue(
            any(
                r["strategy"] == "formula_vs_brute_force"
                for r in strategy_rows
            )
        )
        # Brute force must spend at least as much as formula.
        brute = [
            r for r in strategy_rows
            if r["strategy"] == "brute_force"
            and r["label"] == "exposure"
            and r["payload"] == "all"
        ][0]
        formula = [
            r for r in strategy_rows
            if r["strategy"] == "formula"
            and r["label"] == "exposure"
            and r["payload"] == "all"
        ][0]
        self.assertGreaterEqual(
            brute["tier3_invocations"], formula["tier3_invocations"]
        )


if __name__ == "__main__":
    unittest.main()