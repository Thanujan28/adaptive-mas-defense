"""
Tests proving Tier 1 (semantic) and Tier 2 (NLI) operate on the EXACT
SAME response chunk text.

Requirement
-----------
The authoritative response chunks are produced ONCE, by the existing
``SemanticAssessor`` chunking (``assess_chunked`` -> ``chunked.chunk_texts``).
Tier 2 must consume those chunk TEXTS -- it must NOT re-chunk the response
with a second mechanism.

These tests record the exact ``chunk_text`` STRING that crosses the
Tier 1/Tier 2 boundary inside ``SecurityObserver._run_tiered_pipeline``
by instrumenting the two boundaries directly:

  * Tier 1: the ``chunked.chunks[index]`` whose chunk text is recorded on
    the per-chunk decision (``ChunkDecision.chunk_text``).
  * Tier 2: the ``chunk_text=`` argument actually passed to
    ``ContradictionChecker.check_chunk_against_evidence``.

We compare the TEXT, not merely the chunk index, so a second/re-split
chunking mechanism would fail these tests.

NLI sentence splitting is explicitly ALLOWED: the ContradictionChecker
may internally split a chunk into sentences for the NLI hypotheses. That
is not a second chunking mechanism -- the WHOLE chunk text is what is
handed to Tier 2, and the sentence split happens inside the checker.

No model download: deterministic stub encoder + stub NLI model are
injected (same pattern as the rest of the suite).
"""

from __future__ import annotations

import unittest

import numpy as np

from security.contradiction_checker import ContradictionChecker
from security.observer import SecurityObserver
from security.semantic_assessor import SemanticAssessor


# =====================================================================
# Deterministic, download-free doubles
# =====================================================================

class _KeywordEncoder:
    KEYWORDS = (
        "security", "cyber", "threat", "detection", "ai", "network",
        "cookie", "recipe", "butter", "sugar", "bake",
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


class _SpyChecker(ContradictionChecker):
    """
    A real ``ContradictionChecker`` (deterministic stub model) that ALSO
    records the EXACT ``chunk_text`` argument it receives per call.

    This captures what Tier 2 actually consumes at the whole-chunk level,
    independent of the checker's internal sentence splitting.
    """

    def __init__(self, model=None, **kwargs):
        super().__init__(model=model or _EntailmentNLI(), **kwargs)
        self.seen_chunk_texts: list[str] = []

    def check_chunk_against_evidence(self, chunk_text, evidence_chunks):
        # Record BEFORE delegating, so the recorded text is exactly what
        # Tier 2 was asked about.
        self.seen_chunk_texts.append(str(chunk_text))
        return super().check_chunk_against_evidence(
            chunk_text=chunk_text,
            evidence_chunks=evidence_chunks,
        )


class _EntailmentNLI:
    """
    Deterministic stub NLI model (no download) that returns a decisive
    ENTAILMENT, so the winning ``premise``/``hypothesis`` pair is recorded
    on the result and the evidence -> premise direction is observable.
    """

    def __call__(self, inputs, truncation=True):
        return [
            {"label": "entailment", "score": 0.9},
            {"label": "neutral", "score": 0.07},
            {"label": "contradiction", "score": 0.03},
        ]


TASK = "Write a proposal on AI in cyber security"
EVIDENCE = ["AI detects threats in real time and improves response."]


def _multi_chunk_response() -> str:
    """Three paragraph-scale sections -> several chunks."""

    return (
        (("AI improves cyber security threat detection and network "
          "defence. ") * 40)
        + "\n\n"
        + (("Threat detection uses machine learning on network traffic. ")
           * 40)
        + "\n\n"
        + (("Deployment governance and monitoring keep model risk "
            "manageable for defenders. ") * 40)
    )


# =====================================================================
# THE CORE REQUIREMENT: same chunk text in both tiers
# =====================================================================

class SameChunkTextTests(unittest.TestCase):
    """Chunk i used by Tier 1 == Chunk i used by Tier 2, by TEXT."""

    def setUp(self):
        self.assessor = SemanticAssessor(model=_KeywordEncoder())
        self.checker = _SpyChecker()
        self.observer = SecurityObserver(
            semantic_assessor=self.assessor,
            contradiction_checker=self.checker,
            log_enabled=False,
        )
        self.response = _multi_chunk_response()

    def _run(self):
        return self.observer.observe_tiered(
            agent_id="researcher",
            response=self.response,
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )

    def test_authoritative_chunks_are_the_semantic_assessor_chunks(self):
        """The observer's chunks ARE ``chunked.chunk_texts`` -- no re-split."""

        expected = self.assessor.assess_chunked(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=self.response,
        ).chunk_texts
        self.assertGreaterEqual(len(expected), 2, "need a multi-chunk response")

        result = self._run()
        semantic_chunks = [d.chunk_text for d in result.chunk_decisions]
        self.assertEqual(semantic_chunks, expected)

    def test_semantic_chunks_equal_nli_chunks_textually(self):
        """
        ``semantic_chunks == nli_chunks`` verified on the ACTUAL TEXT.

        ``nli_chunks`` is the list of ``chunk_text`` arguments that Tier 2
        actually received (recorded on the checker), so this fails if NLI
        ever splits/re-chunks the response a second way.
        """

        result = self._run()

        semantic_chunks = [d.chunk_text for d in result.chunk_decisions]
        nli_chunks = list(self.checker.seen_chunk_texts)

        self.assertEqual(
            semantic_chunks,
            nli_chunks,
            "Tier 1 and Tier 2 did not receive the same chunk texts",
        )

        # And, chunk by chunk, by exact text (never just by index).
        for index, decision in enumerate(result.chunk_decisions):
            self.assertEqual(decision.chunk_text, nli_chunks[index])
            self.assertEqual(decision.chunk_text, semantic_chunks[index])

    def test_no_chunk_is_invented_or_dropped_by_nli(self):
        """NLI was called exactly once per Tier-1 chunk, in order."""

        result = self._run()
        self.assertEqual(
            len(self.checker.seen_chunk_texts),
            len(result.chunk_decisions),
        )

        for seen in self.checker.seen_chunk_texts:
            self.assertTrue(seen.strip())
            # Every chunk Tier 2 saw is one Tier 1 actually produced.
            self.assertIn(
                seen,
                [d.chunk_text for d in result.chunk_decisions],
            )

    def test_nli_hypothesis_is_a_sentence_of_the_same_tier1_chunk(self):
        """
        Tier 2's internal sentence split is allowed, but every NLI
        hypothesis must belong to the SAME whole chunk handed to Tier 2.
        """

        result = self._run()
        for decision in result.chunk_decisions:
            contradiction = decision.contradiction
            self.assertIsNotNone(contradiction)
            # Evidence is the premise; the response chunk (a sentence of
            # it) is the hypothesis.
            self.assertEqual(contradiction.premise, EVIDENCE[0])
            self.assertIn(contradiction.hypothesis, decision.chunk_text)

    def test_evidence_is_premise_and_chunk_is_hypothesis(self):
        """Direction: evidence -> premise, response chunk -> hypothesis."""

        result = self._run()
        for decision in result.chunk_decisions:
            c = decision.contradiction
            self.assertEqual(c.premise, EVIDENCE[0])
            # The hypothesis is drawn from the chunk, never the evidence.
            self.assertNotEqual(c.premise, c.hypothesis)
            self.assertIn(c.hypothesis, decision.chunk_text)
            # The whole chunk is never used as the NLI premise.
            self.assertNotEqual(c.premise, decision.chunk_text)


# =====================================================================
# Per-chunk result shape retained for each chunk
# =====================================================================

class PerChunkResultShapeTests(unittest.TestCase):
    """Each chunk keeps index, text, semantic, NLI and tiers_ran."""

    def test_each_chunk_retains_all_fields(self):
        observer = SecurityObserver(
            semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
            contradiction_checker=_SpyChecker(),
            log_enabled=False,
        )
        result = observer.observe_tiered(
            agent_id="researcher",
            response=_multi_chunk_response(),
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )

        self.assertTrue(result.chunk_decisions)
        for index, decision in enumerate(result.chunk_decisions):
            self.assertEqual(decision.chunk_index, index)
            self.assertTrue(decision.chunk_text.strip())
            # Tier 1 semantic result.
            self.assertIsNotNone(decision.semantic)
            self.assertTrue(decision.semantic.assessed)
            # Tier 2 NLI result.
            self.assertIsNotNone(decision.contradiction)
            self.assertTrue(decision.contradiction.label)
            # tiers_ran marks both tiers for the SAME chunk.
            self.assertIn("tier1", decision.tiers_ran)
            self.assertIn("tier2", decision.tiers_ran)


if __name__ == "__main__":
    unittest.main()