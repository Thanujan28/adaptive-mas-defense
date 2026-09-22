"""
Tests for the investigation resource allocator (Task M).

Pure policy logic -- no model, no network.
"""

from __future__ import annotations

import unittest

from security.contradiction_checker import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    ContradictionResult,
)
from security.resource_allocator import (
    ALL_STRATEGIES,
    FORMULA_DEFAULT_EPISODE_BUDGET,
    FORMULA_MAX_EVIDENCE_DEPTH,
    FORMULA_MIN_CONTRADICTION_CONFIDENCE,
    AllocationState,
    AllocationStrategy,
    ResourceAllocator,
)


def _contradiction(confidence: float) -> ContradictionResult:
    return ContradictionResult(
        label=LABEL_CONTRADICTION, confidence=confidence
    )


class NoInvestigationTests(unittest.TestCase):

    def test_never_runs_tier3(self):
        alloc = ResourceAllocator(AllocationStrategy.NO_INVESTIGATION)
        for contradiction in (
            None,
            _contradiction(0.99),
            ContradictionResult(label=LABEL_NEUTRAL, confidence=0.9),
        ):
            request = alloc.allocate_for_chunk(contradiction)
            self.assertFalse(request.run_tier3)


class BruteForceTests(unittest.TestCase):

    def test_runs_on_every_chunk(self):
        alloc = ResourceAllocator(AllocationStrategy.BRUTE_FORCE)
        for contradiction in (
            None,
            _contradiction(0.01),
            ContradictionResult(label=LABEL_ENTAILMENT, confidence=0.9),
        ):
            request = alloc.allocate_for_chunk(contradiction)
            self.assertTrue(request.run_tier3)
            self.assertEqual(
                request.evidence_depth, FORMULA_MAX_EVIDENCE_DEPTH
            )

    def test_ignores_budget_when_deciding(self):
        state = AllocationState(episode_budget=0, tier3_spent=100)
        alloc = ResourceAllocator(
            AllocationStrategy.BRUTE_FORCE, state=state
        )
        self.assertTrue(alloc.allocate_for_chunk(None).run_tier3)


class FormulaTests(unittest.TestCase):

    def test_only_flagged_contradictions_run(self):
        alloc = ResourceAllocator(AllocationStrategy.FORMULA)

        # Not flagged -> no Tier 3.
        for contradiction in (
            None,
            ContradictionResult(label=LABEL_NEUTRAL, confidence=0.99),
            ContradictionResult(label=LABEL_ENTAILMENT, confidence=0.99),
        ):
            self.assertFalse(
                alloc.allocate_for_chunk(contradiction).run_tier3
            )

        # Flagged above the threshold -> Tier 3.
        self.assertTrue(
            alloc.allocate_for_chunk(_contradiction(0.9)).run_tier3
        )

    def test_below_min_confidence_does_not_run(self):
        alloc = ResourceAllocator(AllocationStrategy.FORMULA)
        request = alloc.allocate_for_chunk(
            _contradiction(FORMULA_MIN_CONTRADICTION_CONFIDENCE - 0.05)
        )
        self.assertFalse(request.run_tier3)

    def test_depth_scales_with_confidence(self):
        alloc = ResourceAllocator(AllocationStrategy.FORMULA)
        low = alloc.allocate_for_chunk(_contradiction(0.60))
        high = alloc.allocate_for_chunk(_contradiction(0.95))
        self.assertTrue(low.run_tier3)
        self.assertTrue(high.run_tier3)
        self.assertLess(low.evidence_depth, high.evidence_depth)
        self.assertEqual(
            high.evidence_depth, FORMULA_MAX_EVIDENCE_DEPTH
        )

    def test_budget_exhaustion_stops_spending(self):
        state = AllocationState(episode_budget=1)
        alloc = ResourceAllocator(AllocationStrategy.FORMULA, state=state)

        first = alloc.allocate_for_chunk(_contradiction(0.9))
        self.assertTrue(first.run_tier3)
        alloc.record_investigation(first)

        second = alloc.allocate_for_chunk(_contradiction(0.9))
        self.assertFalse(second.run_tier3)
        self.assertIn("budget", second.reason)

    def test_reason_names_its_inputs(self):
        alloc = ResourceAllocator(AllocationStrategy.FORMULA)
        request = alloc.allocate_for_chunk(_contradiction(0.8))
        self.assertIn("confidence", request.reason)
        self.assertIn("depth", request.reason)


class StrategySelectabilityTests(unittest.TestCase):

    def test_strategy_can_be_set_from_string(self):
        self.assertEqual(
            ResourceAllocator("brute_force").strategy,
            AllocationStrategy.BRUTE_FORCE,
        )

    def test_all_three_strategies_present(self):
        self.assertEqual(len(ALL_STRATEGIES), 3)


class CostTests(unittest.TestCase):

    def test_formula_is_cheaper_than_brute_force_on_mixed_chunks(self):
        chunks = [
            _contradiction(0.9),   # flagged
            None,                  # not flagged
            _contradiction(0.2),   # flagged but low confidence
            ContradictionResult(label=LABEL_NEUTRAL, confidence=0.9),
            _contradiction(0.85),  # flagged
        ]

        brute = ResourceAllocator(AllocationStrategy.BRUTE_FORCE)
        formula = ResourceAllocator(
            AllocationStrategy.FORMULA,
            state=AllocationState(episode_budget=FORMULA_DEFAULT_EPISODE_BUDGET),
        )

        brute_reqs = [brute.allocate_for_chunk(c) for c in chunks]
        formula_reqs = [formula.allocate_for_chunk(c) for c in chunks]

        brute_cost = brute.estimate_cost_units(brute_reqs)
        formula_cost = formula.estimate_cost_units(formula_reqs)

        self.assertEqual(brute_cost, len(chunks))
        self.assertLess(formula_cost, brute_cost)
        self.assertEqual(formula_cost, 2)  # only the two high-conf flags


if __name__ == "__main__":
    unittest.main()
