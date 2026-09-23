"""
End-to-end test for the LIVE tiered detector (Task: make observe_tiered
the only detector that runs under a real pipeline execution).

This runs a FULL task through ``MASEnvironment.execute_task`` with a
stubbed LLM and stubbed tools (the same pattern used by
tests/test_no_attack_vocabulary_leak.py and tests/conftest.py), and
asserts that the TIERED detector -- not the legacy whole-response
``observe()`` -- actually ran on the live path:

  * the observer stored on each observation is a base ``Observation``
    whose ``metadata["tiered"]`` is a real ``TieredObservation``;
  * the chunk-level log fields (``worst_chunk_deviation``,
    ``contradiction_flagged_chunks``, ``tier3_invocations``) carry
    REAL, non-default values;
  * a chunk written to CONTRADICT its delivered evidence is flagged.

No model download and no Ollama: a deterministic stub encoder + a stub
NLI model are injected, and the Tier-3 judge is absent (its default
gated state) unless the test opts in with a stub judge.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from security.observer import SecurityObserver, TieredObservation
from security.semantic_assessor import SemanticAssessor
from security.contradiction_checker import ContradictionChecker


# =================================================================
# Deterministic, download-free collaborators
# =================================================================

class _KeywordEncoder:
    """Bag-of-keywords encoder (no sentence-transformers download)."""

    KEYWORDS = (
        "security", "cyber", "threat", "detection", "ai", "network",
        "cookie", "recipe", "butter", "sugar", "bake", "malicious",
        "traffic", "monitoring",
    )

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            rows.append([float(lowered.count(k)) for k in self.KEYWORDS])
        array = np.asarray(rows, dtype=np.float32)
        if array.size and not array.any(axis=1).all():
            array[~array.any(axis=1), 0] = 1e-6
        return array


class _MarkerNLI:
    """
    Stub NLI: confidence is driven by a marker in the hypothesis, so a
    test can place a chunk deterministically above the Tier-3 gate.

    Mirrors the rule set used elsewhere in the repo: an explicit
    negation against an asserted term is a contradiction; a shared
    core term is entailment; otherwise neutral.
    """

    def __call__(self, inputs, truncation=True):
        hypothesis = str(inputs.get("text_pair", "")).lower()
        if "strongcontradict" in hypothesis:
            return [
                {"label": "contradiction", "score": 0.95},
                {"label": "entailment", "score": 0.03},
                {"label": "neutral", "score": 0.02},
            ]
        return [
            {"label": "neutral", "score": 0.8},
            {"label": "entailment", "score": 0.1},
            {"label": "contradiction", "score": 0.1},
        ]


def _observer(logger=None, judge=None) -> SecurityObserver:
    """
    A tiered observer wired exactly as the live pipeline wires it:
    chunked semantic (Tier 1) + NLI contradiction (Tier 2) always on,
    the LLM judge (Tier 3) only when ``judge`` is supplied.
    """

    return SecurityObserver(
        semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
        contradiction_checker=ContradictionChecker(model=_MarkerNLI()),
        llm_judge=judge,
        log_enabled=logger is not None,
        log_logger=logger,
    )


def _install_stub_tools(env):
    from tests.conftest import (
        StubCalendar,
        StubEmail,
        StubReportWriter,
        StubSearchTool,
    )

    tools = env.tool_manager.tools
    for name in ("internet_search", "academic_search"):
        if name in tools:
            tools[name] = StubSearchTool()
    if "report_writer" in tools:
        tools["report_writer"] = StubReportWriter()
    if "mock_calendar" in tools:
        tools["mock_calendar"] = StubCalendar()
    if "mock_calender" in tools:
        tools["mock_calender"] = StubCalendar()
    if "mock_email" in tools:
        tools["mock_email"] = StubEmail()
    return env


def _install_researcher_marker_llm(marker_text: str, request_search: bool):
    """
    Install a deterministic stub LLM in every agent module.

    The researcher is made to (optionally) request ``internet_search``
    -- so a tool_result artifact is actually delivered to it as
    evidence -- and to emit ``marker_text`` as its own prose.
    """

    import agents.analyst as analyst_module
    import agents.coordinator as coordinator_module
    import agents.executor as executor_module
    import agents.outline as outline_module
    import agents.researcher as researcher_module
    from tests.conftest import StubLLM, _StubResponse

    class _ResearcherStub(StubLLM):
        def invoke(self, prompt: str):
            text = str(prompt)
            # The researcher's tool decision prompt.
            if '"need_tool"' in text or "need_tool" in text:
                self.prompts.append(text)
                if request_search:
                    return _StubResponse(
                        '{"need_tool": true, "tool_name": "internet_search",'
                        ' "arguments": {"query": "AI in cyber security"}}'
                    )
                return _StubResponse(
                    '{"need_tool": false, "tool_name": null, "arguments": {}}'
                )
            # The researcher's prose-generation prompt -> our marker.
            if "You are the Researcher agent" in text:
                self.prompts.append(text)
                return _StubResponse(marker_text)
            return super().invoke(prompt)

    stub = _ResearcherStub(researcher_text=marker_text)
    for module in (
        coordinator_module,
        outline_module,
        researcher_module,
        analyst_module,
        executor_module,
    ):
        module.get_llm = lambda: stub
    return stub


def _run_episode(observer, marker_text, request_search, topology="layered"):
    from environment.mas_environment import MASEnvironment

    _install_researcher_marker_llm(marker_text, request_search=request_search)

    env = MASEnvironment(
        topology_name=topology,
        security_observer=observer,
    )
    _install_stub_tools(env)
    env.execute_task("Write a proposal on AI in cyber security")
    return env


# =================================================================
# The new detector is the ONLY detector on the live path
# =================================================================

def test_live_pipeline_uses_tiered_detector():
    """
    Every observation recorded by a real episode must carry a real
    TieredObservation in metadata["tiered"], and its chunk-level log
    fields must be populated (non-default), proving the tiered path
    ran -- not the legacy whole-response observe().
    """

    env = _run_episode(
        _observer(),
        marker_text="AI strengthens cyber security threat detection.",
        request_search=False,
    )

    assert env.security_observations, "expected at least one observation"

    for observation in env.security_observations:
        tiered = (observation.metadata or {}).get("tiered")
        assert isinstance(tiered, TieredObservation), (
            "live observation is missing its TieredObservation: the "
            "tiered detector did not run on the live path"
        )
        # Base Observation fields every downstream reader relies on
        # must still be present and real.
        assert hasattr(observation, "semantic_assessment")
        assert hasattr(observation, "detector_result")
        # Chunks actually ran (Tier 1 always runs).
        assert tiered.chunk_decisions
        assert all(
            "tier1" in decision.tiers_ran
            for decision in tiered.chunk_decisions
        )


def test_live_pipeline_chunk_log_fields_are_real():
    """
    The tiered log line's chunk-level fields must be present with real
    values on a real run, not just in unit tests.
    """

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    test_logger = logging.getLogger("test.environment.tiered")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(_Capture())

    env = _run_episode(
        _observer(logger=test_logger),
        marker_text="AI strengthens cyber security threat detection.",
        request_search=False,
    )

    combined = "\n".join(records)
    assert "security_state_chunks" in combined
    assert "worst_chunk_deviation=" in combined
    assert "contradiction_flagged_chunks=" in combined
    assert "tier3_invocations=" in combined

    # At least one recorded tiered observation must expose a chunk count
    # AND a real (non-None) worst_chunk_deviation value.
    tiered_obs = [
        (o.metadata or {}).get("tiered")
        for o in env.security_observations
    ]
    tiered_obs = [t for t in tiered_obs if isinstance(t, TieredObservation)]
    assert tiered_obs
    assert any(t.chunked is not None for t in tiered_obs)
    assert all(t.worst_chunk_deviation is not None for t in tiered_obs)


def test_live_pipeline_flags_a_contradicting_chunk():
    """
    A chunk the researcher emits that contradicts the evidence it was
    delivered (via a search tool result) must be flagged as a
    contradiction by Tier 2, end to end.
    """

    marker = (
        "STRONGCONTRADICT AI offers no benefit to cyber security and "
        "should not be used for threat detection at all."
    )

    env = _run_episode(
        _observer(),  # judge absent => default gated state
        marker_text=marker,
        request_search=True,
    )

    flagged = [
        (o.metadata or {}).get("tiered")
        for o in env.security_observations
        if (o.metadata or {}).get("tiered") is not None
    ]

    total_flagged = sum(
        t.contradiction_flagged_chunks for t in flagged
    )
    assert total_flagged >= 1, (
        "the contradicting researcher chunk was not flagged by Tier 2"
    )

    # Without a judge, Tier 3 must NOT have run (default gated state).
    assert all(t.tier3_invocations == 0 for t in flagged)


def test_live_pipeline_tier3_runs_only_when_judge_wired():
    """
    With a (stub) judge wired in, a strongly-contradicting chunk must
    escalate to Tier 3 -- proving the gate works on the live path.
    """

    from security.llm_judge import LLMJudge

    env = _run_episode(
        _observer(judge=LLMJudge(stub=True)),
        marker_text=(
            "STRONGCONTRADICT AI offers no benefit to cyber security."
        ),
        request_search=True,
    )

    flagged = [
        (o.metadata or {}).get("tiered")
        for o in env.security_observations
        if (o.metadata or {}).get("tiered") is not None
    ]
    assert sum(t.tier3_invocations for t in flagged) >= 1, (
        "Tier 3 did not fire on a strongly-contradicting chunk even "
        "with a judge wired in"
    )


def test_ground_truth_artifact_still_rejected_on_live_path():
    """
    The tiered path keeps the strict guard: ground-truth evidence in an
    artifact delivered to the agent must raise before being observed.
    """

    from environment.mas_environment import MASEnvironment

    env = MASEnvironment(
        topology_name="layered",
        security_observer=_observer(),
    )
    _install_stub_tools(env)
    env.episode_state.task = "Analyze the security risk."
    env.security_observations = []

    # A ground-truth key on a delivered artifact must be rejected by the
    # observer's assert_no_ground_truth guard when the response is
    # observed. publish_agent_result drives _observe_agent_response.
    env.log_artifact(
        artifact_type="tool_result",
        source="internet_search",
        receiver="researcher",
        content="benign text",
        request_id="req-gt",
    )
    env.observable_artifacts[-1]["metadata"] = {"attack_type": "prompt_infection"}

    with pytest.raises(ValueError):
        env.publish_agent_result(
            sender="researcher",
            receiver="executor",
            content="Output text",
            metadata={"stage": "research"},
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
