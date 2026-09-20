from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .detector import SecurityDetector
from .semantic_assessor import (
    SemanticAssessor,
    SemanticAssessment,
)
from environment.visibility import assert_no_ground_truth

logger = logging.getLogger(__name__)


def _preview(
    value,
    limit: int = 120,
) -> str:
    # Short, truncated preview for DEBUG logs only.

    text = "" if value is None else str(value)
    return text[:limit]


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
        artifacts: Optional[list[Mapping[str, Any]]] = None,
        tool_limit: Optional[int] = None,
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
            P5: this should be scoped to the current step (e.g. []),
            not the cumulative episode log -- ``response`` and
            ``artifacts`` are the primary per-response evidence
            sources; ``events`` only contributes behavioural counts
            (tool timeouts, relay fan-out, ...).

        artifacts:
            Observable artifacts delivered to THIS agent for THIS
            response (P2/P5): tool results, messages and memory
            writes actually received by ``agent_id`` in this step,
            not the whole episode's artifacts. The detector scans
            these plus ``response`` itself (deduped by text hash so
            the same text is never counted twice).

        metadata:
            Additional provenance information.

        tool_limit:
            Optional tool-call budget (P4), forwarded to the detector
            so ``tool_volume_spike`` can be computed from this path
            too, not only from ``SecurityStateBuilder.build``.
        """

        events = list(events or [])
        artifacts = list(artifacts) if artifacts is not None else None

        metadata = dict(metadata or {})

        # ---------------------------------------------------------
        # 0. Strict guard: the observer may only ever see observable
        #    evidence. Ground-truth leakage raises ValueError.
        # ---------------------------------------------------------

        assert_no_ground_truth(events, artifacts=artifacts)

        # ---------------------------------------------------------
        # 1. Cheap rule-based detection, scoped to this response
        #    (P5): the response text is scanned directly here rather
        #    than depending on it being logged into `events` in the
        #    right order beforehand -- correctness no longer depends
        #    on log ordering, and an empty `events` list still
        #    detects an injected response.
        # ---------------------------------------------------------

        detector_result = self.detector.detect(
            events,
            artifacts=artifacts,
            response=response,
            tool_limit=tool_limit,
        )

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
        metadata = observation.metadata or {}

        level = self._log_level_for(observation)

        self.logger.log(
            level,
            (
                "security_state "
                "agent=%s "
                "stage=%s "
                "subtask_source=%s "
                "security_score=%.4f "
                "investigation_required=%s "
                "evidence_present=%s "
                "injection_evidence_count=%s "
                "untrusted_source_evidence_count=%s "
                "high_confidence_evidence_count=%s "
                "semantic_assessed=%s "
                "task_similarity=%s "
                "subtask_similarity=%s "
                "objective_deviation=%s "
                "scope_deviation=%s "
                "semantic_deviation=%s "
                "semantic_confidence=%s"
            ),
            observation.agent_id,
            metadata.get("stage"),
            metadata.get("subtask_source"),
            observation.security_score,
            observation.investigation_required,
            bool(detector_result.get("evidence_present")),
            detector_result.get("injection_evidence_count"),
            detector_result.get("untrusted_source_evidence_count"),
            detector_result.get("high_confidence_evidence_count"),
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

        # At DEBUG only, add short previews of the subtask and the
        # response. Truncated, and never the full prompt or secrets.
        if level == logging.DEBUG:
            self.logger.debug(
                "security_state_detail agent=%s stage=%s "
                "subtask[:120]=%r response[:120]=%r",
                observation.agent_id,
                metadata.get("stage"),
                _preview(observation.assigned_subtask),
                _preview(observation.response),
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
        if detector_result.get("evidence_present"):
            rule_score += 0.35
        # Observable evidence counts (renamed from label features).
        injection_count = float(
            detector_result.get(
                "injection_evidence_count",
                0,
            )
        )

        untrusted_source_count = float(
            detector_result.get(
                "untrusted_source_evidence_count",
                0,
            )
        )

        rule_score += min(
            injection_count * 0.10,
            0.30,
        )

        rule_score += min(
            untrusted_source_count * 0.05,
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
