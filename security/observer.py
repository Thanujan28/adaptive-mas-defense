from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .detector import SecurityDetector
from .semantic_assessor import (
    SemanticAssessor,
    SemanticAssessment,
)

logger = logging.getLogger(__name__)


def _round_or_none(
    value: Optional[float],
):
    """
    Round a float to 4 decimals, passing None through unchanged.
    """

    if value is None:
        return None
    return round(float(value), 4)


def logger_from_module() -> logging.Logger:
    """
    Return the module-level logger used for security-state records.
    """

    return logger


@dataclass
class Observation:
    """
    Security observation generated for one agent response.
    """

    agent_id: str
    response: str

    original_task: str
    assigned_subtask: str

    detector_result: dict[str, Any] = field(default_factory=dict)

    semantic_assessment: Optional[SemanticAssessment] = None

    security_score: float = 0.0
    investigation_required: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


class SecurityObserver:
    """
    Central security observation point for all agent responses.

    Every agent response should pass through this class before
    being forwarded to another agent.

    The observer itself does not modify the response.
    It produces security evidence for the PPO state and logs the
    resulting security state for inspection/debugging.
    """

    def __init__(
        self,
        detector: Optional[SecurityDetector] = None,
        semantic_assessor: Optional[SemanticAssessor] = None,
        semantic_enabled: bool = True,
        log_enabled: bool = True,
        log_level: Optional[int] = None,
        log_logger: Optional[logging.Logger] = None,
    ) -> None:

        self.detector = detector or SecurityDetector()
        self.semantic_assessor = (
            semantic_assessor or SemanticAssessor()
        )

        self.semantic_enabled = semantic_enabled

        # Security-state logging. Can be disabled (log_enabled=False)
        # in tests or tight loops that do not want log noise.
        self.log_enabled = log_enabled
        self.log_level = log_level
        self.logger = log_logger or logger_from_module()

    def observe(
        self,
        *,
        agent_id: str,
        response: str,
        original_task: str,
        assigned_subtask: str = "",
        events: Optional[list[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Observation:
        """
        Observe one complete agent response.

        This is the main entry point.

        Parameters
        ----------
        agent_id:
            Agent producing the response.

        response:
            Complete LLM-generated response.

        original_task:
            Immutable user task.

        assigned_subtask:
            Task assigned to this agent.

        events:
            Security/runtime events associated with the response.

        metadata:
            Additional provenance information.
        """

        events = list(events or [])

        metadata = dict(metadata or {})

        # ---------------------------------------------------------
        # 1. Add the response itself as an observation event
        # ---------------------------------------------------------

        response_event = {
            "event_type": "agent_response",
            "agent_id": agent_id,
            "response": response,
            "original_task": original_task,
            "assigned_subtask": assigned_subtask,
        }

        events.append(response_event)

        # ---------------------------------------------------------
        # 2. Cheap rule-based detection
        # ---------------------------------------------------------

        detector_result = self.detector.detect(events)

        # ---------------------------------------------------------
        # 3. Semantic assessment
        # ---------------------------------------------------------

        semantic_assessment = None

        if self.semantic_enabled:

            semantic_assessment = (
                self.semantic_assessor.assess(
                    original_task=original_task,
                    assigned_subtask=assigned_subtask,
                    agent_output=response,
                )
            )

        # ---------------------------------------------------------
        # 4. Combine security evidence
        # ---------------------------------------------------------

        security_score = self._calculate_security_score(
            detector_result=detector_result,
            semantic_assessment=semantic_assessment,
        )

        # ---------------------------------------------------------
        # 5. Decide whether deeper investigation is useful
        # ---------------------------------------------------------

        investigation_required = (
            security_score >= 0.45
        )

        observation = Observation(
            agent_id=agent_id,
            response=response,
            original_task=original_task,
            assigned_subtask=assigned_subtask,
            detector_result=detector_result,
            semantic_assessment=semantic_assessment,
            security_score=security_score,
            investigation_required=investigation_required,
            metadata=metadata,
        )

        # ---------------------------------------------------------
        # 6. Log the security state
        # ---------------------------------------------------------

        self._log_security_state(observation)

        return observation

    # =========================================================
    # SECURITY STATE LOGGING
    # =========================================================

    def _log_security_state(
        self,
        observation: Observation,
    ) -> None:
        """
        Emit a structured log record for one security observation.

        The record captures the rule-based detector evidence, the
        semantic similarity check (task/subtask similarity and the
        derived deviations), the fused security score and the
        investigation decision.
        """

        if not self.log_enabled:
            return

        assessment = observation.semantic_assessment
        detector_result = observation.detector_result

        level = self._log_level_for(observation)

        self.logger.log(
            level,
            (
                "security_state "
                "agent=%s "
                "security_score=%.4f "
                "investigation_required=%s "
                "detected=%s "
                "attack_count=%s "
                "suspicious_event_count=%s "
                "semantic_assessed=%s "
                "task_similarity=%s "
                "subtask_similarity=%s "
                "objective_deviation=%s "
                "scope_deviation=%s "
                "semantic_deviation=%s "
                "semantic_confidence=%s"
            ),
            observation.agent_id,
            observation.security_score,
            observation.investigation_required,
            bool(detector_result.get("detected")),
            detector_result.get("attack_count"),
            detector_result.get("suspicious_event_count"),
            _round_or_none(
                assessment and assessment.assessed
            ),
            _round_or_none(
                assessment and assessment.task_similarity
            ),
            _round_or_none(
                assessment and assessment.subtask_similarity
            ),
            _round_or_none(
                assessment and assessment.objective_deviation
            ),
            _round_or_none(
                assessment and assessment.scope_deviation
            ),
            _round_or_none(
                assessment and assessment.deviation_score
            ),
            _round_or_none(
                assessment and assessment.confidence
            ),
        )

    def _log_level_for(
        self,
        observation: Observation,
    ) -> int:
        """
        Choose the log level for a security observation.

        An explicit ``log_level`` passed to the constructor always
        wins. Otherwise, observations that require investigation are
        logged at WARNING and the rest at DEBUG.
        """

        if self.log_level is not None:
            return self.log_level

        if observation.investigation_required:
            return logging.WARNING

        return logging.DEBUG

    @staticmethod
    def _calculate_security_score(
        *,
        detector_result: Mapping[str, Any],
        semantic_assessment: Optional[
            SemanticAssessment
        ],
    ) -> float:
        """
        Combine rule and semantic evidence.

        This is an evidence score, NOT a probability that
        the agent is compromised.
        """

        rule_score = 0.0

        if detector_result.get("detected"):
            rule_score += 0.35

        attack_count = float(
            detector_result.get(
                "attack_count",
                0,
            )
        )

        suspicious_count = float(
            detector_result.get(
                "suspicious_event_count",
                0,
            )
        )

        rule_score += min(
            attack_count * 0.10,
            0.30,
        )

        rule_score += min(
            suspicious_count * 0.05,
            0.15,
        )

        semantic_score = 0.0

        if semantic_assessment is not None:
            semantic_score = (
                semantic_assessment.deviation_score
                * semantic_assessment.confidence
            )

        # Evidence fusion.
        score = (
            0.45 * rule_score
            + 0.55 * semantic_score
        )

        return max(
            0.0,
            min(1.0, score),
        )
