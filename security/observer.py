from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .detector import SecurityDetector
from .semantic_assessor import (
    SemanticAssessor,
    SemanticAssessment,
    ChunkedSemanticAssessment,
    split_into_chunks,
    CHUNK_MIN_WORDS,
    CHUNK_MAX_WORDS,
)
from .contradiction_checker import (
    ContradictionChecker,
    ContradictionResult,
    LABEL_CONTRADICTION,
)
from .llm_judge import LLMJudge, JudgeVerdict
from .resource_allocator import (
    AllocationStrategy,
    ResourceAllocator,
    InvestigationRequest,
)
from environment.visibility import assert_no_ground_truth

logger = logging.getLogger(__name__)


# =================================================================
# TIERED PIPELINE CONFIGURATION
# =================================================================
#
# Tier 1: chunked semantic deviation (cheap, always runs).
# Tier 2: existing rule/content detector + NLI contradiction
#         (cheap-ish, always runs per chunk).
# Tier 3: the LLM judge (expensive) -- runs ONLY on chunks whose
#         Tier-2 contradiction confidence EXCEEDS the named threshold
#         below. This is a named constant on purpose: no magic number
#         inline, and one place to change the gate.
TIER3_CONTRADICTION_THRESHOLD = 0.60


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


@dataclass
class ChunkDecision:
    """
    Per-chunk record of which tiers ran and what they produced.

    ``tiers_ran`` lists the tier names that executed for this chunk
    ("tier1", "tier2", "tier3"). ``tier3_verdict`` is populated only
    when Tier 3 actually ran (i.e. the Tier-2 contradiction gate was
    exceeded).
    """

    chunk_index: int
    chunk_text: str
    semantic: Optional[SemanticAssessment] = None
    contradiction: Optional[ContradictionResult] = None
    tier3_verdict: Optional[JudgeVerdict] = None
    tiers_ran: list[str] = field(default_factory=list)


@dataclass
class TieredObservation:
    """
    Result of one tiered observation of an agent response.

    Wraps the base ``Observation`` and adds the chunk-level summary
    fields surfaced in the security_state log line:
    ``worst_chunk_deviation``, ``contradiction_flagged_chunks`` and
    ``tier3_invocations`` (the per-call count).
    """

    base: Observation
    chunked: Optional[ChunkedSemanticAssessment] = None
    chunk_decisions: list[ChunkDecision] = field(default_factory=list)
    worst_chunk_deviation: float = 0.0
    contradiction_flagged_chunks: int = 0
    tier3_invocations: int = 0

    @property
    def security_score(self) -> float:
        return self.base.security_score

    @property
    def investigation_required(self) -> bool:
        return self.base.investigation_required

    @property
    def agent_id(self) -> str:
        return self.base.agent_id


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
        contradiction_checker: Optional[ContradictionChecker] = None,
        llm_judge: Optional[LLMJudge] = None,
        tier3_contradiction_threshold: float = TIER3_CONTRADICTION_THRESHOLD,
        resource_allocator: Optional["ResourceAllocator"] = None,
    ) -> None:

        self.detector = detector or SecurityDetector()
        self.semantic_assessor = (
            semantic_assessor or SemanticAssessor()
        )

        self.semantic_enabled = semantic_enabled

        # Tier-2/3 collaborators. The NLI checker and the Tier-3 judge
        # are optional: when absent, observe_tiered() still runs Tier 1
        # (chunked semantic) + the rule detector, and skips the missing
        # tier gracefully.
        self.contradiction_checker = contradiction_checker
        self.llm_judge = llm_judge
        self.tier3_contradiction_threshold = tier3_contradiction_threshold

        # Optional investigation-resource allocator (Task M). When set,
        # observe() routes through the tiered pipeline and attaches the
        # TieredObservation to metadata["tiered"], while still returning
        # the SAME Observation object (so every existing caller is
        # unaffected). When None, observe() behaves exactly as before.
        self.resource_allocator = resource_allocator

        # Per-episode Tier-3 invocation counter. Reset per episode by
        # the caller (or via reset_tier_counters()).
        self.tier3_invocations = 0

        # Security-state logging. Can be disabled (log_enabled=False)
        # in tests or tight loops that do not want log noise.
        self.log_enabled = log_enabled
        self.log_level = log_level
        self.logger = log_logger or logger_from_module()

    def reset_tier_counters(self) -> None:
        """Reset the per-episode Tier-3 invocation counter."""
        self.tier3_invocations = 0

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
        # 0a. If an investigation-resource allocator is configured,
        #     route through the tiered pipeline. observe() still
        #     returns the SAME Observation object (the tiered result is
        #     attached to metadata["tiered"]), so every existing caller
        #     is unaffected. When no allocator is set this branch is
        #     skipped and the method behaves exactly as before.
        # ---------------------------------------------------------

        if self.resource_allocator is not None:
            tiered = self._run_tiered_pipeline(
                agent_id=agent_id,
                response=response,
                original_task=original_task,
                assigned_subtask=assigned_subtask,
                events=events,
                artifacts=artifacts,
                tool_limit=tool_limit,
                metadata=metadata,
                evidence_chunks=None,
                log=True,
            )
            return tiered.base

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
    # TIERED PIPELINE (Task L)
    #
    # observe() is kept UNCHANGED above for compatibility. The tiered
    # pipeline is a separate method so existing callers are unaffected.
    # =========================================================

    def observe_tiered(
        self,
        *,
        agent_id: str,
        response: str,
        original_task: str,
        assigned_subtask: str = "",
        events: Optional[list[Mapping[str, Any]]] = None,
        artifacts: Optional[list[Mapping[str, Any]]] = None,
        evidence_chunks: Optional[Sequence[str]] = None,
        tool_limit: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TieredObservation:
        """
        Run the TIERED detection pipeline over one agent response.

        Per chunk of ``response`` (paragraph-aware ~150-200 word chunks):

          * Tier 1 - chunked semantic deviation (always runs).
          * Tier 2 - the existing rule/content detector (scoped to the
            whole response) plus NLI contradiction of the chunk against
            ``evidence_chunks`` (always runs per chunk, when a
            contradiction checker is configured).
          * Tier 3 - the LLM judge, run ONLY on chunks whose Tier-2
            contradiction confidence EXCEEDS
            ``tier3_contradiction_threshold`` (a named constant, see
            the module top). Every Tier-3 call increments the
            per-episode ``tier3_invocations`` counter.

        Each ``ChunkDecision`` records which tiers ran and their
        outputs. The returned ``TieredObservation`` also exposes
        ``worst_chunk_deviation`` and ``contradiction_flagged_chunks``
        for the security_state log line.

        ``evidence_chunks`` is the linked evidence for this response
        (see security/evidence_linker.py). When empty, Tier 2's NLI and
        Tier 3 simply have nothing to check against and are skipped for
        the contradiction axis.
        """

        return self._run_tiered_pipeline(
            agent_id=agent_id,
            response=response,
            original_task=original_task,
            assigned_subtask=assigned_subtask,
            events=events,
            artifacts=artifacts,
            tool_limit=tool_limit,
            metadata=metadata,
            evidence_chunks=evidence_chunks,
            log=True,
        )

    def _run_tiered_pipeline(
        self,
        *,
        agent_id: str,
        response: str,
        original_task: str,
        assigned_subtask: str = "",
        events: Optional[list[Mapping[str, Any]]] = None,
        artifacts: Optional[list[Mapping[str, Any]]] = None,
        evidence_chunks: Optional[Sequence[str]] = None,
        tool_limit: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        log: bool = True,
    ) -> TieredObservation:
        """
        Shared tiered-pipeline implementation used by both
        ``observe_tiered`` and (when a resource allocator is configured)
        ``observe``.

        Tier-3 gating honours the configured ``resource_allocator``
        (Task M) when present: the allocator decides, per chunk, whether
        Tier 3 runs and with what evidence depth. When no allocator is
        set, the gate falls back to the named threshold
        ``tier3_contradiction_threshold`` (Task L behaviour, unchanged).
        """

        events = list(events or [])
        artifacts = list(artifacts) if artifacts is not None else None
        metadata = dict(metadata or {})
        evidence_list = [
            str(e) for e in (evidence_chunks or []) if str(e).strip()
        ]

        # Strict guard: the tiered path is a defender path and may only
        # ever see observable evidence.
        assert_no_ground_truth(events, artifacts=artifacts)

        # ---- Tier 2a: rule/content detector over the whole response ----
        detector_result = self.detector.detect(
            events,
            artifacts=artifacts,
            response=response,
            tool_limit=tool_limit,
        )

        # ---- Tier 1: chunked semantic assessment ----
        #
        # Honour ``semantic_enabled`` exactly as the legacy observe()
        # does: when semantic assessment is switched off, no semantic
        # work runs (Tier 1 or the fused base), so the state builder
        # falls back to its neutral defaults -- identical to the old
        # whole-response path with semantic disabled.
        if self.semantic_enabled:
            chunked = self.semantic_assessor.assess_chunked(
                original_task=original_task,
                assigned_subtask=assigned_subtask,
                agent_output=response,
            )
        else:
            chunked = ChunkedSemanticAssessment()

        chunk_decisions: list[ChunkDecision] = []
        contradiction_flagged_chunks = 0
        tier3_calls = 0

        for index, chunk_text in enumerate(chunked.chunk_texts):

            semantic = (
                chunked.chunks[index]
                if index < len(chunked.chunks)
                else None
            )

            decision = ChunkDecision(
                chunk_index=index,
                chunk_text=chunk_text,
                semantic=semantic,
                tiers_ran=["tier1"],
            )

            # ---- Tier 2b: NLI contradiction vs linked evidence ----
            if self.contradiction_checker is not None and evidence_list:
                contradiction = (
                    self.contradiction_checker.check_chunk_against_evidence(
                        chunk_text=chunk_text,
                        evidence_chunks=evidence_list,
                    )
                )
                decision.contradiction = contradiction
                decision.tiers_ran.append("tier2")

                if contradiction.label == LABEL_CONTRADICTION:
                    contradiction_flagged_chunks += 1

                # ---- Tier 3 gating ----
                if self.llm_judge is not None:
                    run_tier3, depth = self._gate_tier3(contradiction)
                    if run_tier3:
                        evidence_for_judge = evidence_list[:depth]
                        decision.tier3_verdict = self.llm_judge.judge(
                            chunk_text=chunk_text,
                            evidence_text="\n\n".join(evidence_for_judge),
                            task=original_task,
                        )
                        decision.tiers_ran.append("tier3")
                        tier3_calls += 1

            chunk_decisions.append(decision)

        self.tier3_invocations += tier3_calls

        # ---- Fused base observation (reuses observe()'s scoring) ----
        base_assessment = None
        if self.semantic_enabled:
            base_assessment = self.semantic_assessor.assess(
                original_task=original_task,
                assigned_subtask=assigned_subtask,
                agent_output=response,
            )

        security_score = self._calculate_security_score(
            detector_result=detector_result,
            semantic_assessment=base_assessment,
        )

        base = Observation(
            agent_id=agent_id,
            response=response,
            original_task=original_task,
            assigned_subtask=assigned_subtask,
            detector_result=detector_result,
            semantic_assessment=base_assessment,
            security_score=security_score,
            investigation_required=(security_score >= 0.45),
            metadata=metadata,
        )

        tiered = TieredObservation(
            base=base,
            chunked=chunked,
            chunk_decisions=chunk_decisions,
            worst_chunk_deviation=chunked.worst_chunk_deviation,
            contradiction_flagged_chunks=contradiction_flagged_chunks,
            tier3_invocations=tier3_calls,
        )

        # Attach the tiered result to the base observation's metadata so
        # a caller that only sees the Observation (e.g. the environment
        # call site) can still read the per-response Tier-3 count.
        base.metadata["tiered"] = tiered

        if log:
            self._log_tiered_security_state(tiered)

        return tiered

    def _gate_tier3(
        self,
        contradiction: ContradictionResult,
    ) -> tuple[bool, int]:
        """
        Decide whether Tier 3 runs for one chunk and at what evidence
        depth.

        With a ``resource_allocator`` (Task M) configured, the allocator
        owns the decision. Without one, the Task-L gate applies: run on
        a contradiction whose confidence EXCEEDS
        ``tier3_contradiction_threshold``, with all evidence attached.
        """

        if self.resource_allocator is not None:
            request = self.resource_allocator.allocate_for_chunk(
                contradiction
            )
            depth = max(1, request.evidence_depth)
            if request.run_tier3:
                self.resource_allocator.record_investigation(request)
            return request.run_tier3, depth

        run = (
            contradiction.label == LABEL_CONTRADICTION
            and contradiction.confidence
            > self.tier3_contradiction_threshold
        )
        return run, 10_000  # all evidence (Task-L behaviour)

    # =========================================================
    # SECURITY STATE LOGGING
    # =========================================================

    def _log_tiered_security_state(
        self,
        tiered: TieredObservation,
    ) -> None:
        """
        Emit the security_state log line for a tiered observation.

        The base fields are identical to ``_log_security_state`` (kept
        so any existing reader still parses them); three chunk-level
        summary fields are APPENDED, which is backward-compatible with
        any parser that reads named ``key=value`` fields:

          worst_chunk_deviation=...
          contradiction_flagged_chunks=...
          tier3_invocations=...
        """

        if not self.log_enabled:
            return

        self._log_security_state(tiered.base)

        self.logger.log(
            self._log_level_for(tiered.base),
            (
                "security_state_chunks "
                "agent=%s "
                "worst_chunk_deviation=%s "
                "contradiction_flagged_chunks=%s "
                "tier3_invocations=%s "
                "chunk_count=%s"
            ),
            tiered.base.agent_id,
            _round_or_none(tiered.worst_chunk_deviation),
            tiered.contradiction_flagged_chunks,
            tiered.tier3_invocations,
            tiered.chunked.chunk_count if tiered.chunked else 0,
        )

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
