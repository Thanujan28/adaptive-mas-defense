"""
security/resource_allocator.py

Investigation-resource allocation strategies for the tiered defender
(Task M).

The Tier-3 LLM judge (security/llm_judge.py) is the expensive tier. The
question this module answers is: on which chunks should it be spent?

Three strategies:
  * ``no_investigation`` - Tier 3 NEVER runs. The cheap baseline.
  * ``brute_force``      - Tier 3 runs on EVERY chunk. The expensive
                           ceiling used to measure what the formula
                           recovers, and at what cost.
  * ``formula``          - Tier 3 runs only on chunks that Tier 2
                           flagged (a contradiction at/above a named
                           confidence), with an investigation DEPTH
                           (how many linked-evidence chunks to attach,
                           i.e. how much context to give the judge)
                           scaled by the contradiction confidence and
                           the remaining per-episode budget.

Every input is explicit and named. There are no unexplained
coefficients: each scaling factor below is named, documented and given
a stated rationale. The strategy is selectable so evaluation code
(experiments/eval_detector.py) can run all three over the same episodes
and compare AUROC recovered against cost incurred.

This module does NOT call any model. It only DECIDES, per chunk, whether
Tier 3 should run and with what depth/budget. The caller performs the
call. That keeps the allocation policy testable and cost-accountable
independently of the judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .contradiction_checker import (
    ContradictionResult,
    LABEL_CONTRADICTION,
)


# =================================================================
# NAMED CONSTANTS (no inline magic numbers)
# =================================================================
#
# FORMULA strategy inputs, each named with its rationale:

# Only a chunk Tier 2 actually flags as a contradiction is a candidate
# for Tier 3. Below this Tier-2 contradiction confidence, the chunk is
# treated as "not worth the expensive tier".
FORMULA_MIN_CONTRADICTION_CONFIDENCE = 0.60

# Investigation depth is expressed as "how many linked-evidence chunks
# to attach for the judge". A depth of 1 is the minimum (one evidence
# chunk); the maximum bounds the prompt cost.
FORMULA_MIN_EVIDENCE_DEPTH = 1
FORMULA_MAX_EVIDENCE_DEPTH = 4

# Per-episode Tier-3 call budget for the FORMULA strategy. When the
# remaining budget falls below this floor, the formula stops spending
# (the caller may still have spent it via brute_force, which ignores
# budget by design).
FORMULA_DEFAULT_EPISODE_BUDGET = 10

# Confidence band over which depth scales from min to max. Chunks at or
# below FORMULA_LOW_CONFIDENCE get MIN depth; chunks at or above
# FORMULA_HIGH_CONFIDENCE get MAX depth; in between, depth is
# interpolated. This makes high-confidence contradictions get more
# evidence, which is a deliberate, stated policy -- not a hidden knob.
FORMULA_LOW_CONFIDENCE = 0.60
FORMULA_HIGH_CONFIDENCE = 0.95


class AllocationStrategy(str, Enum):
    """Selectable investigation strategies."""

    NO_INVESTIGATION = "no_investigation"
    BRUTE_FORCE = "brute_force"
    FORMULA = "formula"


@dataclass
class InvestigationRequest:
    """
    One allocation decision for a single chunk.

    ``run_tier3`` is the decision. ``evidence_depth`` is how many
    linked-evidence chunks to attach (>=1 when running). ``reason`` is a
    short human-readable justification naming the inputs that produced
    the decision.
    """

    run_tier3: bool
    evidence_depth: int = 0
    strategy: str = ""
    reason: str = ""


@dataclass
class AllocationState:
    """
    Per-episode allocation state.

    ``tier3_spent`` counts Tier-3 invocations already made in this
    episode; ``episode_budget`` is the formula strategy's cap. Both are
    explicit inputs to the formula, not globals.
    """

    episode_budget: int = FORMULA_DEFAULT_EPISODE_BUDGET
    tier3_spent: int = 0

    @property
    def remaining_budget(self) -> int:
        return max(0, self.episode_budget - self.tier3_spent)

    def record_spend(self, amount: int = 1) -> None:
        self.tier3_spent += max(0, int(amount))


class ResourceAllocator:
    """
    Decides, per chunk, whether and how deeply to run the Tier-3 judge.

    Construct with a strategy; call ``allocate_for_chunk`` for each
    chunk, passing the Tier-2 contradiction result (or None when Tier 2
    produced no verdict). The returned ``InvestigationRequest`` is the
    policy output; the caller acts on it.
    """

    def __init__(
        self,
        strategy: AllocationStrategy | str = AllocationStrategy.FORMULA,
        state: Optional[AllocationState] = None,
        min_contradiction_confidence: float = (
            FORMULA_MIN_CONTRADICTION_CONFIDENCE
        ),
        low_confidence: float = FORMULA_LOW_CONFIDENCE,
        high_confidence: float = FORMULA_HIGH_CONFIDENCE,
        min_evidence_depth: int = FORMULA_MIN_EVIDENCE_DEPTH,
        max_evidence_depth: int = FORMULA_MAX_EVIDENCE_DEPTH,
    ) -> None:
        self.strategy = AllocationStrategy(strategy)
        self.state = state or AllocationState()

        # Named, overridable inputs (each documented at module top).
        self.min_contradiction_confidence = min_contradiction_confidence
        self.low_confidence = low_confidence
        self.high_confidence = high_confidence
        self.min_evidence_depth = min_evidence_depth
        self.max_evidence_depth = max_evidence_depth

    # =========================================================
    # FORMULA DEPTH
    # =========================================================

    def _depth_for(self, confidence: float) -> int:
        """
        Interpolate investigation depth from the contradiction
        confidence.

        confidence <= low_confidence          -> min_evidence_depth
        confidence >= high_confidence         -> max_evidence_depth
        in between                            -> linear interpolation

        Inputs are the four named parameters above; no other term
        enters.
        """

        span = self.high_confidence - self.low_confidence
        if span <= 0.0:
            return self.max_evidence_depth

        frac = (confidence - self.low_confidence) / span
        frac = max(0.0, min(1.0, frac))

        depth = self.min_evidence_depth + round(
            frac * (self.max_evidence_depth - self.min_evidence_depth)
        )
        return int(
            max(self.min_evidence_depth, min(self.max_evidence_depth, depth))
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def allocate_for_chunk(
        self,
        contradiction: Optional[ContradictionResult],
    ) -> InvestigationRequest:
        """
        Decide the Tier-3 treatment for one chunk.

        ``contradiction`` is the chunk's Tier-2 NLI result (or None if
        Tier 2 produced no verdict).
        """

        if self.strategy is AllocationStrategy.NO_INVESTIGATION:
            return InvestigationRequest(
                run_tier3=False,
                strategy=self.strategy.value,
                reason="strategy=no_investigation: Tier 3 never runs",
            )

        if self.strategy is AllocationStrategy.BRUTE_FORCE:
            return InvestigationRequest(
                run_tier3=True,
                evidence_depth=self.max_evidence_depth,
                strategy=self.strategy.value,
                reason="strategy=brute_force: Tier 3 on every chunk",
            )

        # ---- FORMULA ----
        if contradiction is None:
            return InvestigationRequest(
                run_tier3=False,
                strategy=self.strategy.value,
                reason="no Tier-2 verdict for this chunk",
            )

        if contradiction.label != LABEL_CONTRADICTION:
            return InvestigationRequest(
                run_tier3=False,
                strategy=self.strategy.value,
                reason=(
                    f"Tier-2 label={contradiction.label} "
                    "(not a contradiction)"
                ),
            )

        confidence = float(contradiction.confidence)

        if confidence < self.min_contradiction_confidence:
            return InvestigationRequest(
                run_tier3=False,
                strategy=self.strategy.value,
                reason=(
                    f"contradiction confidence {confidence:.3f} < "
                    f"min {self.min_contradiction_confidence:.3f}"
                ),
            )

        if self.state.remaining_budget <= 0:
            return InvestigationRequest(
                run_tier3=False,
                strategy=self.strategy.value,
                reason=(
                    "contradiction flagged but per-episode budget "
                    f"exhausted ({self.state.tier3_spent}/"
                    f"{self.state.episode_budget})"
                ),
            )

        depth = self._depth_for(confidence)

        return InvestigationRequest(
            run_tier3=True,
            evidence_depth=depth,
            strategy=self.strategy.value,
            reason=(
                f"contradiction confidence {confidence:.3f} >= "
                f"{self.min_contradiction_confidence:.3f}; depth={depth} "
                f"(low={self.low_confidence}, high={self.high_confidence}, "
                f"min_depth={self.min_evidence_depth}, "
                f"max_depth={self.max_evidence_depth}); "
                f"budget {self.state.remaining_budget} remaining"
            ),
        )

    def record_investigation(self, request: InvestigationRequest) -> None:
        """
        Account for a Tier-3 call the caller actually made. Brute force
        still tracks spend (for cost reporting) even though it ignores
        the budget when deciding.
        """

        if request.run_tier3:
            self.state.record_spend(1)

    def estimate_cost_units(
        self,
        requests: list[InvestigationRequest],
    ) -> int:
        """
        A simple, stated cost proxy: one unit per Tier-3 call. Callers
        that have real token/latency numbers should prefer those; this
        is the strategy-level cost for comparing formula vs brute force.
        """

        return sum(1 for r in requests if r.run_tier3)


def make_allocator(
    strategy: AllocationStrategy | str,
    episode_budget: int = FORMULA_DEFAULT_EPISODE_BUDGET,
) -> ResourceAllocator:
    """Convenience constructor for the evaluation harness."""

    return ResourceAllocator(
        strategy=strategy,
        state=AllocationState(episode_budget=episode_budget),
    )


ALL_STRATEGIES: tuple[AllocationStrategy, ...] = (
    AllocationStrategy.NO_INVESTIGATION,
    AllocationStrategy.BRUTE_FORCE,
    AllocationStrategy.FORMULA,
)
