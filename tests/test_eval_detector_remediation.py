"""
Tests for the Task-S remediation-comparison mode added to
experiments/eval_detector.py.

These assert the MODE produces well-formed rows and that the cost helper
reports non-zero cost when remediation actually triggered -- using the
deterministic E2E stubs, so no model download and no LLM backend.
"""

from __future__ import annotations

import io
import contextlib
import os

from experiments.eval_detector import (
    REMEDIATION_QUALITY_PROXY,
    _remediation_cost_from_env,
    _semantic_deviation,
    run_remediation_comparison,
)


def test_cost_helper_is_zero_when_nothing_remediated():
    class _Empty:
        remediation_summary = []

    cost = _remediation_cost_from_env(_Empty())
    assert cost == {
        "tokens": 0,
        "llm_calls": 0,
        "events": 0,
        "episodes_with_rerun": 0,
    }


def test_cost_helper_reports_nonzero_when_triggered():
    env, stub = _run_episode_triggering_remediation()
    cost = _remediation_cost_from_env(env)
    assert cost["episodes_with_rerun"] >= 1
    assert cost["tokens"] > 0 or cost["llm_calls"] > 0


def test_semantic_deviation_proxy_returns_a_float():
    from security.semantic_assessor import SemanticAssessor
    from tests.test_remediation_e2e import _KeywordEncoder

    assessor = SemanticAssessor(model=_KeywordEncoder())
    dev = _semantic_deviation(
        assessor,
        "Write a proposal on AI in cyber security",
        "AI strengthens cyber security threat detection.",
    )
    assert dev is None or isinstance(dev, float)


def test_comparison_rows_are_well_formed():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rows = run_remediation_comparison(
            topologies=["shared_pool"],
            seeds=[7],
            task="Write a proposal on AI in cyber security",
            stub_encoder=True,
            stub_llm=True,
            stub_nli=False,
            stub_judge=True,
        )

    assert rows, "expected at least one remediation-comparison row"
    for row in rows:
        for key in (
            "condition",
            "topology",
            "seed",
            "remediation_triggered",
            "added_tokens",
            "added_llm_calls",
            "output_changed",
            "quality_proxy",
            "better_after_remediation",
        ):
            assert key in row, key
        # The proxy is labelled explicitly so it cannot be mistaken for a
        # real quality score.
        assert row["quality_proxy"] == REMEDIATION_QUALITY_PROXY
        assert "PROXY" in row["note"] or "proxy" in row["note"]

    # The flag must never be left set for later runs.
    assert os.environ.get("MAS_REMEDIATION_ENABLED") is None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _run_episode_triggering_remediation():
    """Run one episode with the deterministic confirming judge so the
    containment layer actually triggers (reusing the E2E stubs)."""

    from tests.test_remediation_e2e import (
        _run_episode,
        _ConfirmingJudge,
        _EchoSourceDetectiveLLM,
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env, stub = _run_episode(
            judge=_ConfirmingJudge(),
            detective_llm=_EchoSourceDetectiveLLM(trace=False),
        )
    os.environ.pop("MAS_REMEDIATION_ENABLED", None)
    return env, stub