"""
Tests for the detective agent (Task S, containment layer).

No model download and no LLM backend: the shared encoder is injected
through ``SemanticAssessor(model=...)`` and the detective's LLM is a
deterministic stub.
"""

from __future__ import annotations

import unittest

import numpy as np

from security.detective_agent import (
    DETECTIVE_TOKEN_BUDGET,
    DetectiveAgent,
    DetectiveFinding,
)
from security.semantic_assessor import SemanticAssessor


class _KeywordEncoder:
    """Deterministic bag-of-keywords encoder (no download)."""

    KEYWORDS = (
        "ransomware",
        "encryption",
        "threat",
        "detection",
        "ai",
        "security",
        "network",
        "telemetry",
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


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubLLM:
    """Returns a fixed structured reply; records the prompts it saw."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(str(prompt))
        return _StubResponse(self.reply)


RELEVANT = (
    "AI-powered ransomware detection uses network telemetry to spot "
    "the encryption of files before the threat spreads across the "
    "network. Detection models watch network telemetry for the "
    "encryption behaviour of ransomware and raise a threat alert."
)

CHUNK = "Ransomware detection via AI and network telemetry."

ARTIFACTS = [
    {
        "artifact_type": "tool_result",
        "source": "internet_search",
        "receiver": "researcher",
        "text": RELEVANT,
        "request_id": "req-tool-1",
        "message_id": None,
    },
]


def _detective(llm) -> DetectiveAgent:
    return DetectiveAgent(
        assessor=SemanticAssessor(model=_KeywordEncoder()),
        llm=llm,
    )


class DetectiveAgentTests(unittest.TestCase):

    def test_clear_source_found_and_tracing_requested(self):
        """A cited tool result is attributable, tracing may be requested."""

        llm = _StubLLM(
            "SOURCE: [req:req-tool-1]\n"
            "TRACE: YES\n"
            "REASONING: the tool result fed this claim"
        )
        finding = _detective(llm).investigate(
            chunk_text=CHUNK,
            evidence_chunks=[RELEVANT],
            evidence_artifacts=ARTIFACTS,
            upstream_responses={},
            task="Write a proposal on AI in cyber security",
        )

        self.assertIsInstance(finding, DetectiveFinding)
        self.assertTrue(finding.attributable)
        self.assertEqual(finding.source_artifact_id, "req:req-tool-1")
        self.assertTrue(finding.requests_tracer)
        self.assertFalse(finding.budget_exhausted)
        self.assertGreater(finding.tokens_used, 0)

    def test_clear_source_found_but_tracing_declined(self):
        """Attributable, but the detective decides not to trace further."""

        llm = _StubLLM(
            "SOURCE: req:req-tool-1\n"
            "TRACE: NO\n"
            "REASONING: the tool result IS the root; do not go further"
        )
        finding = _detective(llm).investigate(
            chunk_text=CHUNK,
            evidence_chunks=[RELEVANT],
            evidence_artifacts=ARTIFACTS,
            upstream_responses={},
            task="Write a proposal on AI in cyber security",
        )

        self.assertTrue(finding.attributable)
        self.assertEqual(finding.source_artifact_id, "req:req-tool-1")
        self.assertFalse(finding.requests_tracer)

    def test_nothing_attributable(self):
        """NONE from the LLM -> not attributable, no tracing."""

        llm = _StubLLM(
            "SOURCE: NONE\n"
            "TRACE: NO\n"
            "REASONING: the chunk is the agent's own error"
        )
        finding = _detective(llm).investigate(
            chunk_text=CHUNK,
            evidence_chunks=[RELEVANT],
            evidence_artifacts=ARTIFACTS,
            upstream_responses={},
            task="Write a proposal on AI in cyber security",
        )

        self.assertFalse(finding.attributable)
        self.assertIsNone(finding.source_artifact_id)
        self.assertFalse(finding.requests_tracer)

    def test_no_evidence_is_not_attributable(self):
        """With no linked evidence there is nothing to attribute."""

        llm = _StubLLM("SOURCE: NONE\nTRACE: NO\nREASONING: n/a")
        finding = _detective(llm).investigate(
            chunk_text=CHUNK,
            evidence_chunks=[],
            evidence_artifacts=[],
            upstream_responses={},
            task="task",
        )
        self.assertFalse(finding.attributable)
        self.assertEqual(finding.tokens_used, 0)
        # The LLM is never called when there is nothing to inspect.
        self.assertEqual(llm.prompts, [])

    def test_budget_exhausted_returns_partial_findings(self):
        """A tiny budget must stop early with partial findings, no overspend."""

        llm = _StubLLM("SOURCE: [req:req-tool-1]\nTRACE: YES\nREASONING: x")
        detective = DetectiveAgent(
            assessor=SemanticAssessor(model=_KeywordEncoder()),
            llm=llm,
            token_budget=1,  # far below any realistic prompt
        )
        finding = detective.investigate(
            chunk_text=CHUNK,
            evidence_chunks=[RELEVANT],
            evidence_artifacts=ARTIFACTS,
            upstream_responses={},
            task="Write a proposal on AI in cyber security",
        )

        self.assertTrue(finding.budget_exhausted)
        self.assertFalse(finding.attributable)
        self.assertFalse(finding.requests_tracer)
        # The LLM was never called -- we stopped BEFORE overspending.
        self.assertEqual(llm.prompts, [])

    def test_unknown_cited_source_is_not_attributable(self):
        """A cited id that is not an upstream item must not be trusted."""

        llm = _StubLLM("SOURCE: [req:does-not-exist]\nTRACE: YES\nREASONING: x")
        finding = _detective(llm).investigate(
            chunk_text=CHUNK,
            evidence_chunks=[RELEVANT],
            evidence_artifacts=ARTIFACTS,
            upstream_responses={},
            task="task",
        )
        self.assertFalse(finding.attributable)
        self.assertIsNone(finding.source_artifact_id)

    def test_agent_output_can_be_attributed(self):
        """A prior agent's output is a valid attribution target."""

        llm = _StubLLM("SOURCE: [agent:researcher]\nTRACE: YES\nREASONING: x")
        finding = _detective(llm).investigate(
            chunk_text=CHUNK,
            evidence_chunks=[],
            evidence_artifacts=[],
            upstream_responses={"researcher": RELEVANT},
            task="task",
        )
        self.assertTrue(finding.attributable)
        self.assertEqual(finding.source_artifact_id, "agent:researcher")

    def test_default_budget_is_named_constant(self):
        detective = _detective(_StubLLM("SOURCE: NONE\nTRACE: NO\nREASONING: x"))
        self.assertEqual(detective.token_budget, DETECTIVE_TOKEN_BUDGET)


if __name__ == "__main__":
    unittest.main()