"""
Fixed-dimensional security state for PPO.

The state is built ONLY from observable evidence: content,
behaviour, provenance, semantics and resources. Simulator ground
truth must never enter this state (enforced by
``assert_no_ground_truth``).

The vector length is unchanged from the original design; the
label-based features were re-derived from observable evidence and
renamed.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping, Optional, Sequence

from .detector import SecurityDetector
from .semantic_assessor import SemanticAssessment
from environment.visibility import assert_no_ground_truth


# Feature groups, used for ablation masking.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "content": (
        "injection_evidence_count",
        "high_confidence_evidence_count",
        "untrusted_source_evidence_count",
        "unexpected_url_count",
    ),
    "behaviour": (
        "relay_fanout",
        "token_spike_count",
        "tool_timeout_count",
    ),
    "semantic": (
        "task_similarity",
        "subtask_similarity",
        "objective_deviation",
        "scope_deviation",
        "semantic_deviation",
        "semantic_confidence",
    ),
    "resource": (
        "message_count",
        "relay_count",
        "affected_agent_count",
        "propagation_depth",
        "memory_count",
        "tokens_used",
        "tools_used",
    ),
    "defence": (
        "investigation_count",
        "containment_count",
        "resource_allocation_count",
    ),
}


class SecurityStateBuilder:
    """
    Build a fixed-dimensional security state for PPO.

    The state combines:
      1. Observable content/behaviour evidence
      2. Semantic deviation
      3. Propagation evidence
      4. Runtime/resource behaviour
      5. Current defensive activity
    """

    FEATURE_NAMES = (
        # Observable evidence (renamed from label-based features)
        "injection_evidence_count",
        "high_confidence_evidence_count",
        "untrusted_source_evidence_count",
        "tool_timeout_count",

        # Semantic deviation
        "task_similarity",
        "subtask_similarity",
        "objective_deviation",
        "scope_deviation",
        "semantic_deviation",
        "semantic_confidence",

        # Propagation
        "message_count",
        "relay_count",
        "affected_agent_count",
        "propagation_depth",

        # Runtime
        "relay_fanout",
        "memory_count",

        # Resources
        "tokens_used",
        "tools_used",

        # Defence activity
        "investigation_count",
        "containment_count",
        "resource_allocation_count",
    )

    def __init__(
        self,
        detector: Optional[SecurityDetector] = None,
    ) -> None:
        self.detector = detector or SecurityDetector()

    def build(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        artifacts: Optional[Iterable[Mapping[str, Any]]] = None,
        semantic: Optional[SemanticAssessment] = None,
        resource_state: Optional[Mapping[str, Any]] = None,
        memory_counts: Optional[Mapping[str, int]] = None,
        tool_limit: Optional[int] = None,
        mask_groups: Optional[Sequence[str]] = None,
        strict: bool = True,
    ) -> dict[str, float]:
        """
        Build the security state dictionary.

        Parameters
        ----------
        events:
            OBSERVABLE events only. Ground truth here raises
            ValueError when ``strict`` is True (the default).

        artifacts:
            Observable artifacts (P2, see
            ``MASEnvironment.get_observable_artifacts``). When given,
            content evidence is derived from these instead of the
            compact event summaries.

        semantic:
            Aggregated semantic assessment.

        resource_state / memory_counts:
            Runtime resource and memory summaries.

        mask_groups:
            Optional feature-group names (see FEATURE_GROUPS) to
            zero out, for ablations.

        strict:
            When True (default), reject any ground-truth leakage.

        P5 aggregation rules (this builder is intentionally
        EPISODE-level, aggregating over every agent's responses so
        far, unlike ``SecurityObserver.observe`` which scores a
        single response):

          * evidence counts (``injection_evidence_count``,
            ``high_confidence_evidence_count``,
            ``untrusted_source_evidence_count``,
            ``unexpected_url_count``/``_email_count``/
            ``_command_count``) are summed over unique artifacts
            across the whole episode (an artifact is scanned once
            each, regardless of how many agents later read the same
            memory);
          * ``affected_agent_count`` / ``propagation_depth`` are
            derived from the set of distinct receivers whose
            artifacts carried evidence (episode-wide, not per
            response);
          * ``semantic_*`` fields reflect the assessment passed in by
            the caller for the CURRENT step (typically the
            worst-of-episode deviation tracked by
            ``MASEnvironment.get_semantic_assessment``), not an
            average -- a single strongly deviating response should
            not be diluted by many benign ones;
          * ``tokens_used``/``tools_used``/``memory_count`` are
            episode-cumulative resource counters;
          * ``investigation_count``/``containment_count``/
            ``resource_allocation_count`` count defensive actions
            taken so far this episode.
        """

        events = list(events)
        artifacts = list(artifacts) if artifacts is not None else None
        resource_state = resource_state or {}
        memory_counts = memory_counts or {}

        if strict:
            assert_no_ground_truth(events, artifacts=artifacts)

        evidence = self.detector.detect(
            events,
            tool_limit=tool_limit,
            artifacts=artifacts,
        )

        # -----------------------------------------------------
        # affected_agent_count: agents whose observable events
        # triggered content evidence.
        # -----------------------------------------------------

        affected_agents = set(
            evidence.get("affected_agents", [])
        )

        # -----------------------------------------------------
        # propagation_depth (P4 redefinition): the number of
        # distinct agents reachable downstream, over the observable
        # message graph, from an agent whose artifact carried content
        # evidence. 0 when there is no evidence at all -- previously
        # this was pure communication topology (identical in clean
        # and attacked runs, P4).
        # -----------------------------------------------------

        propagation_depth = self._propagation_depth(
            events,
            affected_agents,
        )

        investigation_count = sum(
            1
            for event in events
            if event.get("event_type") == "investigation"
        )

        containment_count = sum(
            1
            for event in events
            if event.get("event_type") == "containment"
        )

        allocation_count = sum(
            1
            for event in events
            if event.get("event_type") == "resource_allocation"
        )

        semantic = semantic or SemanticAssessment()

        state = {
            # Observable evidence
            "injection_evidence_count": float(
                evidence["injection_evidence_count"]
            ),
            "high_confidence_evidence_count": float(
                evidence["high_confidence_evidence_count"]
            ),
            "untrusted_source_evidence_count": float(
                evidence["untrusted_source_evidence_count"]
            ),
            "tool_timeout_count": float(
                evidence["tool_timeout_count"]
            ),

            # Semantic
            "task_similarity": float(
                semantic.task_similarity
            ),
            "subtask_similarity": float(
                semantic.subtask_similarity
            ),
            "objective_deviation": float(
                semantic.objective_deviation
            ),
            "scope_deviation": float(
                semantic.scope_deviation
            ),
            "semantic_deviation": float(
                semantic.deviation_score
            ),
            "semantic_confidence": float(
                semantic.confidence
            ),

            # Propagation
            "message_count": float(
                evidence["message_count"]
            ),
            "relay_count": float(
                evidence["relay_count"]
            ),
            "affected_agent_count": float(
                len(affected_agents)
            ),
            "propagation_depth": float(
                propagation_depth
            ),

            # Runtime
            "relay_fanout": float(
                evidence["relay_fanout"]
            ),
            "memory_count": float(
                sum(memory_counts.values())
            ),

            # Resources
            "tokens_used": float(
                resource_state.get("tokens_used", 0)
            ),
            "tools_used": float(
                resource_state.get("tools_used", 0)
            ),

            # Defence
            "investigation_count": float(
                investigation_count
            ),
            "containment_count": float(
                containment_count
            ),
            "resource_allocation_count": float(
                allocation_count
            ),
        }

        self._apply_mask(state, mask_groups)

        return state

    def _apply_mask(
        self,
        state: dict[str, float],
        mask_groups: Optional[Sequence[str]],
    ) -> None:
        """
        Zero out the features belonging to the given groups.

        Unknown group names are ignored. Mutates ``state`` in place.
        """

        if not mask_groups:
            return

        for group in mask_groups:

            for feature in FEATURE_GROUPS.get(group, ()):

                if feature in state:
                    state[feature] = 0.0

    @staticmethod
    def _propagation_depth(
        events: Iterable[Mapping[str, Any]],
        affected_agents: Iterable[str],
    ) -> int:
        """
        Number of distinct agents reachable downstream, over the
        observable message graph, from an agent whose artifact
        carried content evidence (P4).

        Returns 0 when ``affected_agents`` is empty -- with no
        evidence there is nothing to propagate, so this is no longer
        identical between clean and attacked runs (previously it was
        pure communication topology, present regardless of evidence).
        """

        affected_agents = set(affected_agents)
        if not affected_agents:
            return 0

        edges: list[tuple[str, str]] = []

        for event in events:

            if event.get("event_type") not in {
                "message",
                "message_relay",
            }:
                continue

            sender = event.get("sender")
            receiver = event.get("receiver")

            if sender and receiver:
                edges.append((sender, receiver))

        adjacency: dict[str, list[str]] = {}
        for sender, receiver in edges:
            adjacency.setdefault(sender, []).append(receiver)

        reachable: set[str] = set()

        for start in affected_agents:

            visited = {start}
            queue = deque([start])

            while queue:
                node = queue.popleft()
                for neighbour in adjacency.get(node, []):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
                        reachable.add(neighbour)

        return len(reachable)

    def vector(
        self,
        state: Mapping[str, float],
        mask_groups: Optional[Sequence[str]] = None,
    ) -> list[float]:
        """
        Convert the state dictionary into the fixed PPO vector.

        ``mask_groups`` zeroes the same groups as ``build`` for
        ablations, without rebuilding the state.
        """

        masked: set[str] = set()

        for group in (mask_groups or ()):
            masked.update(FEATURE_GROUPS.get(group, ()))

        return [
            0.0
            if feature in masked
            else float(state.get(feature, 0.0))
            for feature in self.FEATURE_NAMES
        ]

    @classmethod
    def feature_count(cls) -> int:
        return len(cls.FEATURE_NAMES)
