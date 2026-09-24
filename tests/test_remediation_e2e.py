"""
End-to-end test for the containment/remediation layer (Task S).

This runs a FULL task through ``MASEnvironment.execute_task`` with:
  * ``MAS_REMEDIATION_ENABLED=1`` (the layer enabled),
  * a deterministic STUB judge that CONFIRMS a contradiction on the
    researcher's chunk,
  * a deterministic STUB detective + tracer path that finds a specific
    fake root cause (a delivered search tool result),

and asserts that:
  * the CORRECT agent (the researcher) re-ran,
  * the final output CHANGED after remediation,
  * the total token/call cost of remediation is reported and non-zero.

It also covers the two NON-activating paths:
  * "not attributable" -> no re-run, a clear log entry instead;
  * "hop limit exceeded" -> no re-run, a clear log entry instead.

No model download and no Ollama: the encoder, NLI, judge, detective LLM
and agents' LLM are all deterministic stubs.
"""

from __future__ import annotations

import os


# =================================================================
# Deterministic, download-free collaborators
# =================================================================

class _KeywordEncoder:
    KEYWORDS = (
        "security", "cyber", "threat", "detection", "ai", "network",
        "malicious", "traffic", "contradict", "benefit",
    )

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        import numpy as np

        rows = []
        for text in texts:
            lowered = str(text).lower()
            rows.append([float(lowered.count(k)) for k in self.KEYWORDS])
        array = np.asarray(rows, dtype=np.float32)
        if array.size and not array.any(axis=1).all():
            array[~array.any(axis=1), 0] = 1e-6
        return array


class _MarkerNLI:
    """Stub NLI: a marker in the hypothesis forces a contradiction.

    The marker drives Tier-2 confidence above the Tier-3 gate (0.60), so
    Tier 3 actually runs for the marked chunk.
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


class _ConfirmingJudge:
    """Stub Tier-3 judge that ALWAYS confirms a contradiction."""

    stub = True

    def judge(self, chunk_text, evidence_text, task):
        from security.llm_judge import JudgeVerdict

        return JudgeVerdict(
            contradicts_evidence=True,
            reasoning="stub judge: confirms a contradiction",
            stubbed=True,
        )


class _NeverConfirmingJudge:
    stub = True

    def judge(self, chunk_text, evidence_text, task):
        from security.llm_judge import JudgeVerdict

        return JudgeVerdict(
            contradicts_evidence=False,
            reasoning="stub judge: no contradiction",
            stubbed=True,
        )


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _EchoSourceDetectiveLLM:
    """
    A detective LLM stub that attributes to the FIRST upstream item it
    sees in the prompt (so the test does not need to know the random
    tool-result request id beforehand).

    ``trace`` controls whether it asks the tracer to walk further back.
    """

    def __init__(self, trace: bool) -> None:
        self.trace = trace
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        import re

        self.prompts.append(str(prompt))
        # Prefer a tool-result candidate ('req:<id>') -- the realistic
        # root cause -- and only fall back to the first candidate.
        match = re.search(r"\[(req:[^\]]+)\]", str(prompt))
        if match is None:
            match = re.search(r"\[(agent:[^\]]+)\]", str(prompt))
        source = match.group(1) if match else "NONE"
        return _StubResponse(
            f"SOURCE: [{source}]\n"
            f"TRACE: {'YES' if self.trace else 'NO'}\n"
            "REASONING: stub detective attribution"
        )


class _NoSourceDetectiveLLM:
    """A detective LLM stub that attributes the problem to nothing."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(str(prompt))
        return _StubResponse(
            "SOURCE: NONE\nTRACE: NO\nREASONING: not attributable (stub)"
        )


def _observer(judge):
    from security.observer import SecurityObserver
    from security.semantic_assessor import SemanticAssessor
    from security.contradiction_checker import ContradictionChecker

    return SecurityObserver(
        semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
        contradiction_checker=ContradictionChecker(model=_MarkerNLI()),
        llm_judge=judge,
        log_enabled=False,
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


def _install_agent_llm(first_text: str, later_text: str):
    """
    Install a stub LLM whose Researcher prose is ``first_text`` on the
    FIRST run and ``later_text`` on every subsequent (re-run) invocation
    -- so remediation visibly CHANGES the researcher's output.
    """

    import agents.analyst as analyst_module
    import agents.coordinator as coordinator_module
    import agents.executor as executor_module
    import agents.outline as outline_module
    import agents.researcher as researcher_module
    from tests.conftest import StubLLM

    class _ChangingResearcherStub(StubLLM):
        def __init__(self, first_text, later_text):
            super().__init__(researcher_text=first_text)
            self.first_text = first_text
            self.later_text = later_text
            self.researcher_calls = 0

        def invoke(self, prompt: str):
            text = str(prompt)
            if '"need_tool"' in text or "need_tool" in text:
                self.prompts.append(text)
                return _StubResponse(
                    '{"need_tool": true, "tool_name": "internet_search",'
                    ' "arguments": {"query": "AI in cyber security"}}'
                )
            if "You are the Researcher agent" in text:
                self.prompts.append(text)
                self.researcher_calls += 1
                if self.researcher_calls == 1:
                    return _StubResponse(self.first_text)
                return _StubResponse(self.later_text)
            return super().invoke(prompt)

    stub = _ChangingResearcherStub(first_text, later_text)
    for module in (
        coordinator_module,
        outline_module,
        researcher_module,
        analyst_module,
        executor_module,
    ):
        module.get_llm = lambda: stub
    return stub


MARKER = (
    "STRONGCONTRADICT AI offers no benefit to cyber security and should "
    "not be used for threat detection at all."
)
CLEAN = (
    "AI strengthens cyber security threat detection and supports "
    "malicious-traffic monitoring."
)


def _run_episode(
    *,
    judge,
    detective_llm,
    marker: str = MARKER,
    clean: str = CLEAN,
    topology: str = "shared_pool",
    remediation_enabled: bool = True,
):
    from environment.mas_environment import MASEnvironment

    if remediation_enabled:
        os.environ["MAS_REMEDIATION_ENABLED"] = "1"
    else:
        os.environ.pop("MAS_REMEDIATION_ENABLED", None)

    stub = _install_agent_llm(marker, clean)

    env = MASEnvironment(
        topology_name=topology,
        security_observer=_observer(judge),
    )
    _install_stub_tools(env)
    # The detective's LLM is injected as a per-environment attribute (the
    # containment layer reads it via getattr(self, "remediation_detective_llm")).
    env.remediation_detective_llm = detective_llm

    env.execute_task("Write a proposal on AI in cyber security")
    return env, stub


def _containment_events(env):
    return [
        event.to_dict()
        for event in env.events
        if event.event_type == "investigation"
        and (event.metadata or {}).get("stage") == "containment"
    ]


# =================================================================
# Activating path
# =================================================================

def test_remediation_reruns_correct_agent_and_changes_output():
    """
    A judge-confirmed contradiction on the researcher's chunk, attributed
    to its delivered search tool result, must:
      * re-run the RESEARCHER (the agent that consumed that artifact);
      * change the researcher's output (marker -> clean);
      * report a non-zero remediation cost.
    """

    env, stub = _run_episode(
        judge=_ConfirmingJudge(),
        detective_llm=_EchoSourceDetectiveLLM(trace=False),
    )

    # The researcher must have run at least twice (original + re-run).
    assert stub.researcher_calls >= 2, (
        "the researcher was not re-run by remediation"
    )

    # The containment layer must have logged a performed remediation.
    stages = [
        (event.get("metadata") or {}).get("containment_stage")
        for event in _containment_events(env)
    ]
    assert "remediation_performed" in stages, stages
    assert "rerun_start" in stages, stages

    # A remediation summary with non-zero cost must exist.
    assert env.remediation_summary, "no remediation summary was recorded"
    total_cost = {"tokens": 0, "llm_calls": 0}
    for entry in env.remediation_summary:
        cost = entry["summary"]["cost"]
        total_cost["tokens"] += cost.get("tokens", 0)
        total_cost["llm_calls"] += cost.get("llm_calls", 0)
    assert total_cost["tokens"] > 0 or total_cost["llm_calls"] > 0, (
        f"remediation cost was not reported as non-zero: {total_cost}"
    )

    # The final output must reflect the CLEAN (post-remediation) research.
    final = env.episode_state.result
    assert final is not None
    final_text = str(final.get("final_result") or "")
    assert "STRONGCONTRADICT" not in final_text, (
        "final output still contains the poisoned marker after remediation"
    )


def test_remediation_trace_finds_root_and_reruns():
    """
    With tracing REQUESTED, the tracer walks to the detected tool result
    (a clean root within 2 hops) and remediation still re-runs the
    researcher.
    """

    env, stub = _run_episode(
        judge=_ConfirmingJudge(),
        detective_llm=_EchoSourceDetectiveLLM(trace=True),
    )

    stages = [
        (event.get("metadata") or {}).get("containment_stage")
        for event in _containment_events(env)
    ]
    assert "trace_result" in stages, stages
    assert "remediation_performed" in stages, stages

    # The trace found a root (the tool result has no further upstream).
    trace_events = [
        event for event in _containment_events(env)
        if (event.get("metadata") or {}).get("containment_stage")
        == "trace_result"
    ]
    assert any(
        (event.get("metadata") or {}).get("root_found") is True
        for event in trace_events
    ), "tracer did not find a root"


# =================================================================
# Non-activating paths
# =================================================================

def test_not_attributable_produces_no_rerun_and_clear_log():
    """Detective finds nothing attributable -> NO re-run, clear log."""

    env, stub = _run_episode(
        judge=_ConfirmingJudge(),
        detective_llm=_NoSourceDetectiveLLM(),
    )

    stages = [
        (event.get("metadata") or {}).get("containment_stage")
        for event in _containment_events(env)
    ]
    assert "not_attributable" in stages, stages
    assert "remediation_performed" not in stages, (
        "a non-attributable finding must not trigger a re-run"
    )
    # The researcher ran only once (no re-run).
    assert stub.researcher_calls == 1, stub.researcher_calls


def test_hop_limit_exceeded_produces_no_rerun_and_clear_log():
    """
    The tracer is asked to walk but every hop is corrupt -> hop limit
    reached -> NO re-run, clear log entry ('no_root' / hop_limit_exceeded).

    Built directly (no episode run) so the tracer's evidence lookup can
    be forced to an endless corrupt chain deterministically.
    """

    from security.root_cause_tracer import ArtifactView, REASON_HOP_LIMIT
    from security.observer import (
        SecurityObserver,
        TieredObservation,
        ChunkDecision,
        Observation,
    )
    from security.semantic_assessor import SemanticAssessor
    from security.contradiction_checker import ContradictionChecker
    from security.llm_judge import JudgeVerdict
    from environment.mas_environment import MASEnvironment

    os.environ["MAS_REMEDIATION_ENABLED"] = "1"

    checker = ContradictionChecker(model=_MarkerNLI())
    env = MASEnvironment(
        topology_name="shared_pool",
        security_observer=SecurityObserver(
            semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
            contradiction_checker=checker,
            llm_judge=_ConfirmingJudge(),
            log_enabled=False,
        ),
    )
    _install_stub_tools(env)

    chunk_text = "STRONGCONTRADICT corrupted claim"
    # A delivered tool result the detective can attribute to.
    artifact = env.log_artifact(
        artifact_type="tool_result",
        source="internet_search",
        receiver="researcher",
        content=chunk_text,
        request_id="req-root",
    )

    # The detective attributes to that tool result and asks for tracing.
    class _AttrToTool:
        def invoke(self, prompt):
            return _StubResponse(
                "SOURCE: [req:req-root]\nTRACE: YES\nREASONING: stub"
            )

    env.remediation_detective_llm = _AttrToTool()

    # Force the tracer's lookup to an endless corrupt chain (never clean).
    def _endless_corrupt(artifact_id):
        if artifact_id in ("trusted_root", "task", "agent:coordinator:plan"):
            return ArtifactView(artifact_id=artifact_id, is_trusted_root=True)
        return ArtifactView(
            artifact_id=str(artifact_id),
            text="this artifact STRONGCONTRADICT corrupts its source",
            upstream_ids=[f"req:upstream-{artifact_id}"],
            upstream_evidence=["clean upstream"],
        )

    env._trace_evidence_lookup = _endless_corrupt

    tiered = TieredObservation(
        base=Observation(
            agent_id="researcher",
            response=chunk_text,
            original_task="task",
            assigned_subtask="task",
        ),
        chunk_decisions=[
            ChunkDecision(
                chunk_index=0,
                chunk_text=chunk_text,
                tier3_verdict=JudgeVerdict(
                    contradicts_evidence=True,
                    reasoning="stub",
                    stubbed=True,
                ),
            )
        ],
    )

    summary = env._maybe_remediate(
        tiered=tiered,
        agent_id="researcher",
        response=chunk_text,
        artifacts_for_agent=[artifact],
        original_task="task",
    )

    outcomes = [entry["outcome"] for entry in summary["chunks"]]
    assert "no_root" in outcomes, outcomes
    assert summary["reruns"] == [], "hop limit must not trigger a re-run"

    # A clear log entry with the hop-limit reason must exist.
    trace_events = [
        event for event in _containment_events(env)
        if (event.get("metadata") or {}).get("containment_stage")
        == "trace_result"
    ]
    assert any(
        (event.get("metadata") or {}).get("stopped_reason")
        == REASON_HOP_LIMIT
        for event in trace_events
    ), "hop-limit stop reason was not logged"


def test_disabled_flag_is_a_noop():
    """With MAS_REMEDIATION_ENABLED unset, the layer never runs."""

    env, stub = _run_episode(
        judge=_ConfirmingJudge(),
        detective_llm=_EchoSourceDetectiveLLM(trace=False),
        remediation_enabled=False,
    )

    assert env.remediation_enabled is False
    assert env.remediation_summary == []
    assert stub.researcher_calls == 1, (
        "the containment layer re-ran an agent while disabled"
    )
    assert _containment_events(env) == []


def test_no_tier3_confirmation_is_inert():
    """
    Even with the layer ENABLED, if Tier 3 does not CONFIRM (judge says
    no), the containment layer must never activate.
    """

    env, stub = _run_episode(
        judge=_NeverConfirmingJudge(),
        detective_llm=_EchoSourceDetectiveLLM(trace=False),
    )

    assert env.remediation_summary == []
    assert stub.researcher_calls == 1
    assert _containment_events(env) == []


if __name__ == "__main__":
    import unittest

    unittest.main()