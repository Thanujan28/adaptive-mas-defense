"""
security/remediation.py

Remediation executor for the containment layer (Task S).

ROLE
----
Given a CONFIRMED root cause (an upstream artifact identified by the
root-cause tracer, or directly by the detective), produce a
``RemediationPlan`` describing exactly which agents must re-run, in
dependency order, once that artifact is discarded.

This module DELIBERATELY does NOT touch ``MASEnvironment`` -- it only
PRODUCES the plan. The environment performs the actual re-run
(environment/mas_environment.py). Keeping the policy here makes it
testable without building an environment.

WHAT A PLAN CONTAINS
--------------------
* ``agents_to_rerun``  - every agent that must re-run, in DEPENDENCY
                         ORDER (upstream -> downstream). This is the
                         union of:
                           (a) the agents that DIRECTLY consumed the
                               discarded artifact, and
                           (b) every DOWNSTREAM agent reachable from
                               those via the consumer graph (remediation
                               cascades FORWARD from the fix point).
* ``discarded_artifact_id`` - the artifact to discard/exclude.
* ``attempt``           - the attempt number this plan represents.
* ``over_attempt_cap``  - True when the per-chunk/artifact attempt cap
                         has been reached (then ``agents_to_rerun`` is
                         empty and remediation must STOP, reporting the
                         chunk as unresolved).

ATTEMPT TRACKING (no global mutable state)
------------------------------------------
``MAX_REMEDIATION_ATTEMPTS`` is enforced per chunk/artifact.
``plan()`` RECEIVES the current attempt counter via the caller and
RETURNS an updated counter. It never reads or writes a module-level or
global counter, so two episodes never interfere and the counter is
explicit at every call site.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


# =================================================================
# NAMED CONSTANT
# =================================================================

# Maximum number of remediation attempts per chunk/artifact. Once this
# many attempts have been made for the same root, do not retry forever:
# report the chunk as unresolved. A named constant on purpose.
MAX_REMEDIATION_ATTEMPTS = 2


# =================================================================
# PLAN
# =================================================================

@dataclass
class RemediationPlan:
    """
    The set of agents to re-run and the artifact to discard.

    ``agents_to_rerun`` is in dependency order (upstream -> downstream).
    When ``over_attempt_cap`` is True, ``agents_to_rerun`` is EMPTY and
    the caller must not re-run anything.
    """

    agents_to_rerun: list[str]
    discarded_artifact_id: str
    attempt: int
    over_attempt_cap: bool = False


# =================================================================
# EXECUTOR (plan producer only -- never touches the environment)
# =================================================================

class RemediationExecutor:
    """
    Turn a confirmed root cause into a ``RemediationPlan``.

    Two mappings drive the cascade:

      * ``consumer_index``  - artifact_id -> agents that consumed it
                              DIRECTLY. This identifies the fix point's
                              immediate victims.
      * ``consumer_graph``  - agent -> agents it feeds downstream. This
                              is the forward cascade: if agent A re-runs,
                              every agent A feeds must re-run AFTER it.

    ``pipeline_order`` is the canonical dependency order of the pipeline
    (upstream -> downstream), used to emit ``agents_to_rerun`` in a
    stable order rather than dict iteration order.
    """

    def __init__(
        self,
        consumer_index: Mapping[str, Sequence[str]],
        consumer_graph: Optional[Mapping[str, Sequence[str]]] = None,
        pipeline_order: Optional[Sequence[str]] = None,
        max_attempts: int = MAX_REMEDIATION_ATTEMPTS,
    ) -> None:
        self.consumer_index = {
            str(k): [str(a) for a in (v or [])]
            for k, v in (consumer_index or {}).items()
        }
        self.consumer_graph = {
            str(k): [str(a) for a in (v or [])]
            for k, v in (consumer_graph or {}).items()
        }
        self.pipeline_order = list(pipeline_order or [])
        self.max_attempts = int(max_attempts)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def plan(
        self,
        *,
        root_artifact_id: str,
        attempts: Mapping[str, int],
    ) -> tuple[RemediationPlan, dict[str, int]]:
        """
        Produce a ``RemediationPlan`` for ``root_artifact_id`` and the
        UPDATED attempt counter.

        ``attempts`` is the caller's external counter (artifact_id ->
        attempts made so far). This method never mutates it; it returns
        a NEW dict with the counter incremented for this artifact.

        When the counter is already at/above ``max_attempts``, the plan
        is returned with ``over_attempt_cap=True`` and NO agents to
        re-run (the caller stops and reports the chunk unresolved).
        """

        updated = dict(attempts or {})
        current = int(updated.get(root_artifact_id, 0))

        if current >= self.max_attempts:
            logger.warning(
                "remediation_over_attempt_cap artifact=%s attempts=%s cap=%s",
                root_artifact_id,
                current,
                self.max_attempts,
            )
            return (
                RemediationPlan(
                    agents_to_rerun=[],
                    discarded_artifact_id=root_artifact_id,
                    attempt=current,
                    over_attempt_cap=True,
                ),
                updated,
            )

        direct = self.consumer_index.get(root_artifact_id, [])
        cascade = self._downstream_closure(direct)
        ordered = self._order(cascade)

        attempt = current + 1
        updated[root_artifact_id] = attempt

        logger.info(
            "remediation_plan artifact=%s agents=%s attempt=%s",
            root_artifact_id,
            ordered,
            attempt,
        )

        return (
            RemediationPlan(
                agents_to_rerun=ordered,
                discarded_artifact_id=root_artifact_id,
                attempt=attempt,
                over_attempt_cap=False,
            ),
            updated,
        )

    # =========================================================
    # CASCADE / ORDERING HELPERS
    # =========================================================

    def _downstream_closure(self, roots: Sequence[str]) -> set[str]:
        """
        All agents reachable from ``roots`` via ``consumer_graph``
        (including the roots themselves). Breadth-first with a visited
        set, so a branching/shared topology cannot loop.
        """

        visited: set[str] = set()
        queue: list[str] = [str(r) for r in roots]
        while queue:
            agent = queue.pop(0)
            if agent in visited:
                continue
            visited.add(agent)
            for downstream in self.consumer_graph.get(agent, []):
                if downstream not in visited:
                    queue.append(downstream)
        return visited

    def _order(self, agents: set[str]) -> list[str]:
        """
        Order ``agents`` by the canonical pipeline order (upstream ->
        downstream). Agents not present in ``pipeline_order`` keep
        alphabetical order AFTER the known ones, deterministically.
        """

        known = [a for a in self.pipeline_order if a in agents]
        unknown = sorted(a for a in agents if a not in self.pipeline_order)
        return known + unknown