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
        # propagation_depth: derived from the observable
        # message-relay chain (sender -> receiver hops), NOT from a
        # simulator-written field.
        # -----------------------------------------------------

        propagation_depth = self._propagation_depth(events)

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
    ) -> int:
        """
        Estimate propagation depth from the observable relay chain.

        The chain is reconstructed from ``message`` / ``message_relay``
        edges: each edge is (current hop sender -> next hop receiver).
        Depth is the length of the longest path through these edges.
        Returns 0 when no chain is present.
        """

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

        if not edges:
            return 0

        # Longest simple path over the observed edges (small graphs).
        adjacency: dict[str, list[str]] = {}
        for sender, receiver in edges:
            adjacency.setdefault(sender, []).append(receiver)

        best = 0

        def walk(node: str, depth: int, seen: frozenset) -> None:
            nonlocal best
            best = max(best, depth)
            for neighbour in adjacency.get(node, []):
                if neighbour in seen:
                    continue
                walk(
                    neighbour,
                    depth + 1,
                    seen | {neighbour},
                )

        for start in list(adjacency):
            walk(start, 0, frozenset({start}))

        return best

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
