from __future__ import annotations

from typing import Iterable, Mapping, Optional, Any

from .detector import SecurityDetector
from .semantic_assessor import SemanticAssessment


class SecurityStateBuilder:
    """
    Build a fixed-dimensional security state for PPO.

    The state combines:
      1. Rule-based threat evidence
      2. Semantic deviation
      3. Propagation evidence
      4. Runtime/resource behaviour
      5. Current defensive activity
    """

    FEATURE_NAMES = (
        # Threat evidence
        "attack_count",
        "explicit_attack_count",
        "external_injection_count",
        "suspicious_event_count",

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
        "tool_timeout_count",
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
        semantic: Optional[SemanticAssessment] = None,
        resource_state: Optional[Mapping[str, Any]] = None,
        memory_counts: Optional[Mapping[str, int]] = None,
    ) -> dict[str, float]:

        events = list(events)
        resource_state = resource_state or {}
        memory_counts = memory_counts or {}

        rule_state = self.detector.detect(events)

        affected_agents = {
            event.get("agent_id")
            for event in events
            if event.get("agent_id") is not None
            and event.get("event_type") in {
                "attack",
                "external_result_injection",
                "prompt_injection",
                "memory_poisoning",
                "suspicious_tool_result",
            }
        }

        propagation_depth = max(
            (
                int(event.get("propagation_depth", 0) or 0)
                for event in events
            ),
            default=0,
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

        return {
            # Threat evidence
            "attack_count": float(
                rule_state["attack_count"]
            ),
            "explicit_attack_count": float(
                rule_state["explicit_attack_count"]
            ),
            "external_injection_count": float(
                rule_state["external_injection_count"]
            ),
            "suspicious_event_count": float(
                rule_state["suspicious_event_count"]
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
                rule_state["message_count"]
            ),
            "relay_count": float(
                rule_state["relay_count"]
            ),
            "affected_agent_count": float(
                len(affected_agents)
            ),
            "propagation_depth": float(
                propagation_depth
            ),

            # Runtime
            "tool_timeout_count": float(
                rule_state["tool_timeout_count"]
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

    def vector(
        self,
        state: Mapping[str, float],
    ) -> list[float]:
        """
        Convert the state dictionary into the fixed PPO vector.
        """
        return [
            float(state.get(feature, 0.0))
            for feature in self.FEATURE_NAMES
        ]

    @classmethod
    def feature_count(cls) -> int:
        return len(cls.FEATURE_NAMES)