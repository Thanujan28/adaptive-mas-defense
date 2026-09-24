"""
security/root_cause_tracer.py

Root-cause tracer for the remediation/containment layer (Task S).

ROLE
----
After the detective agent (security/detective_agent.py) has attributed a
judge-confirmed contradiction to a specific upstream artifact, the tracer
walks BACKWARD through the evidence chain ONE HOP AT A TIME to find the
artifact that is actually corrupt.

At each hop it checks whether THAT artifact/prompt is itself
contradictory or corrupted, reusing the SAME contradiction checker
(security/contradiction_checker.py) on that artifact against its own
upstream evidence:

  * artifact at hop N is STILL contradictory  -> go back another hop;
  * artifact at hop N is NOT contradictory     -> THAT hop is the root;
  * hop limit reached without a clean source   -> stop, report
                                                  "hop limit exceeded";
  * the next hop would be a trusted root       -> stop at the plan,
                                                  report
                                                  "no identifiable source
                                                   before trusted root".

HARD LIMITS (named, documented, not magic numbers; constructor defaults
only -- they are NOT configurable per call)
------------------------------------------------------------------------
* ``MAX_TRACE_HOPS = 3``        - maximum 3 hops back from the point of
                                  detection. Reaching it without a clean
                                  source stops the walk; the pipeline's
                                  original task and the coordinator's
                                  initial plan are NEVER treated as
                                  suspect.
* ``TRUSTED_ROOT_LABEL``        - the label used for the trusted root.
  ``TRUSTED_ROOT_IDS``          - the task / coordinator-plan ids that
                                  are never traced into, removed or
                                  re-run.

CONVERGENCE SAFETY
------------------
A branching chain where two paths converge on the SAME artifact must not
loop forever. ``trace()`` keeps an explicit VISITED SET of artifact ids
and stops if it revisits one (reporting ``visited_cycle``), so the walk
is guaranteed to terminate.

This module is read-only with respect to the pipeline: it only inspects
evidence via the injected ``evidence_lookup`` and never mutates state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .contradiction_checker import (
    ContradictionChecker,
    LABEL_CONTRADICTION,
)


logger = logging.getLogger(__name__)


# =================================================================
# NAMED, HARD CONSTANTS
# =================================================================

# Maximum number of hops BACK from the point of detection. A named
# constant on purpose (no inline magic number). It is a constructor
# default only: the caller cannot change it per call.
MAX_TRACE_HOPS = 3

# Label/id for the trusted root: the original human task and the
# coordinator's initial plan. Tracing never goes INTO these and they are
# never removed or re-run.
TRUSTED_ROOT_LABEL = "trusted_root"
TRUSTED_ROOT_IDS = frozenset(
    {
        "trusted_root",
        "task",
        "agent:coordinator:plan",
        "coordinator:plan",
    }
)

# Reason strings (stable, so logs and reports are greppable).
REASON_ROOT_FOUND = "root_found"
REASON_HOP_LIMIT = "hop_limit_exceeded"
REASON_TRUSTED_ROOT = "reached_trusted_root"
REASON_NO_UPSTREAM = "no_upstream_evidence"
REASON_VISITED_CYCLE = "visited_cycle"


# =================================================================
# STRUCTURED VIEW OF ONE HOP
# =================================================================

@dataclass
class ArtifactView:
    """
    What the tracer needs to know about ONE artifact to take one hop:

      * ``artifact_id``     - this artifact's stable id (request id /
                              message id / ``agent:<name>`` / a trusted
                              root id);
      * ``text``            - the artifact's own text;
      * ``upstream_ids``    - ids of the artifacts this one was derived
                              from (its "own upstream evidence");
      * ``upstream_evidence`` - the TEXT of that upstream evidence, used
                              by the contradiction checker as the
                              premise;
      * ``is_trusted_root`` - True for the task / coordinator plan.
    """

    artifact_id: str
    text: str = ""
    upstream_ids: Sequence[str] = field(default_factory=tuple)
    upstream_evidence: Sequence[str] = field(default_factory=tuple)
    is_trusted_root: bool = False


@dataclass
class TraceResult:
    """
    Result of one backward walk.

    ``stopped_reason`` is one of the stable REASON_* strings above.
    ``root_artifact_id`` is set only when ``root_found`` is True.
    """

    root_found: bool
    root_artifact_id: Optional[str]
    hops_taken: int
    stopped_reason: str


class RootCauseTracer:
    """
    Hop-by-hop backward walk over the evidence chain.

    ``evidence_lookup(artifact_id)`` returns the ``ArtifactView`` for one
    artifact (or None when unknown). The tracer calls it one hop at a
    time; it never fetches more than one hop ahead.
    """

    def __init__(
        self,
        contradiction_checker: ContradictionChecker,
        evidence_lookup: Callable[[str], Optional[ArtifactView]],
        max_hops: int = MAX_TRACE_HOPS,
        trusted_root_ids: frozenset[str] = TRUSTED_ROOT_IDS,
    ) -> None:
        # HARD limits: these are constructor defaults, documented and
        # named. They are deliberately NOT per-call parameters.
        self.max_hops = int(max_hops)
        self.trusted_root_ids = frozenset(trusted_root_ids)
        self.checker = contradiction_checker
        self.evidence_lookup = evidence_lookup

    # =========================================================
    # PUBLIC API
    # =========================================================

    def trace(self, start_artifact_id: str) -> TraceResult:
        """
        Walk backward from ``start_artifact_id`` one hop at a time.

        Returns a ``TraceResult``. Never raises for an unknown id: an id
        with no view, or with no upstream evidence, stops the walk with
        ``no_upstream_evidence``.
        """

        if not start_artifact_id:
            return TraceResult(
                root_found=False,
                root_artifact_id=None,
                hops_taken=0,
                stopped_reason=REASON_NO_UPSTREAM,
            )

        visited: set[str] = set()
        current_id = start_artifact_id
        hops = 0

        while True:

            # ---- convergence safety: never revisit an artifact ----
            if current_id in visited:
                logger.info(
                    "tracer_visited_cycle artifact=%s hops=%s",
                    current_id,
                    hops,
                )
                return TraceResult(
                    root_found=False,
                    root_artifact_id=None,
                    hops_taken=hops,
                    stopped_reason=REASON_VISITED_CYCLE,
                )
            visited.add(current_id)

            view = self.evidence_lookup(current_id)

            # ---- unknown / no view: nothing to trace into ----
            if view is None:
                return TraceResult(
                    root_found=False,
                    root_artifact_id=None,
                    hops_taken=hops,
                    stopped_reason=REASON_NO_UPSTREAM,
                )

            # ---- trusted root: never traced into ----
            if view.is_trusted_root or view.artifact_id in self.trusted_root_ids:
                return TraceResult(
                    root_found=False,
                    root_artifact_id=None,
                    hops_taken=hops,
                    stopped_reason=REASON_TRUSTED_ROOT,
                )

            # ---- is THIS artifact itself contradictory? ----
            contradictory = self._is_contradictory(view)

            if not contradictory:
                # Clean hop -> this IS the root cause.
                return TraceResult(
                    root_found=True,
                    root_artifact_id=current_id,
                    hops_taken=hops,
                    stopped_reason=REASON_ROOT_FOUND,
                )

            # ---- this artifact is also corrupt: try to go one hop back ----
            upstream_ids = [str(u) for u in (view.upstream_ids or [])]
            if not upstream_ids:
                return TraceResult(
                    root_found=False,
                    root_artifact_id=None,
                    hops_taken=hops,
                    stopped_reason=REASON_NO_UPSTREAM,
                )

            # Hop limit is checked BEFORE taking another hop (max 3 hops
            # back from the point of detection).
            if hops >= self.max_hops:
                return TraceResult(
                    root_found=False,
                    root_artifact_id=None,
                    hops_taken=hops,
                    stopped_reason=REASON_HOP_LIMIT,
                )

            # Follow the FIRST upstream id (the chain is a walk, not a
            # breadth-first search); a branching chain converges only if
            # the chosen id is already visited, which the visited-set
            # check above catches.
            current_id = upstream_ids[0]
            hops += 1

    # =========================================================
    # ONE-HOP CONTRADICTION CHECK
    # =========================================================

    def _is_contradictory(self, view: ArtifactView) -> bool:
        """
        Reuse the contradiction checker on THIS artifact against its own
        upstream evidence (evidence is the PREMISE, the artifact text is
        the HYPOTHESIS -- see contradiction_checker's documented
        direction rule).
        """

        upstream = [
            str(e) for e in (view.upstream_evidence or []) if str(e).strip()
        ]
        if not upstream or not str(view.text or "").strip():
            # No upstream evidence to contradict -> not (verifiably)
            # corrupt here; this hop is the clean root.
            return False

        result = self.checker.check_chunk_against_evidence(
            chunk_text=view.text,
            evidence_chunks=upstream,
        )
        return result.label == LABEL_CONTRADICTION