"""
Tests for chunked semantic assessment (Task H).

Motivation
----------
Whole-response ``SemanticAssessor.assess()`` embeds an ENTIRE agent
response into one vector. For a long report (~5000+ chars, hundreds of
words) a single off-topic or injected passage is diluted by the
surrounding on-topic prose, so the whole-response cosine similarity
barely moves. That is the "truncation blind spot": a locally deviant
chunk is invisible at the response level.

``assess_chunked()`` splits the response into paragraph-aware ~150-200
word chunks and assesses each one, returning the worst-chunk deviation.

These tests reproduce the blind spot DIRECTLY: they build a synthetic
long response modelled on the structure of the real researcher report
(multi-section proposal on AI in cybersecurity), bury an off-topic
chunk after word ~700, and assert that

  * whole-response ``assess()`` does NOT flag the response, while
  * ``assess_chunked()`` DOES flag the off-topic chunk as the worst.

No model download and no network: a deterministic keyword encoder is
injected via ``SemanticAssessor(model=...)``.
"""

from __future__ import annotations

import unittest

import numpy as np

from security.semantic_assessor import (
    CHUNK_MAX_WORDS,
    SemanticAssessor,
    ChunkedSemanticAssessment,
    split_into_chunks,
)


# =============================================================
# DETERMINISTIC STUB ENCODER
# =============================================================

class _KeywordEncoder:
    """
    Deterministic bag-of-keywords encoder.

    On-topic (security/AI) text shares the on-topic lexicon; the
    off-topic chunk shares none of it and lights up the off-topic
    lexicon instead, so it lands far from the task reference -- with no
    model download.
    """

    KEYWORDS = (
        # on-topic
        "security", "cyber", "threat", "detection", "ai",
        "machine", "learning", "agent", "attack", "network",
        # off-topic
        "cookie", "recipe", "butter", "sugar", "bake", "oven",
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


# =============================================================
# SYNTHETIC LONG RESPONSE (modelled on the researcher report)
# =============================================================

TASK = "Write a proposal on AI in cyber security"

SECTION = (
    "Artificial intelligence is reshaping cyber security across "
    "detection, response and governance. Machine learning models "
    "inspect network traffic and flag anomalies that signature-based "
    "tools miss, and continuous monitoring lets defenders detect "
    "threats in real time. A layered defence combines robust feature "
    "engineering, model monitoring and human review so that an "
    "attacker who poisons training data cannot silently degrade the "
    "detector. This section develops the case for AI-assisted threat "
    "detection, the security risks of deploying models online, and "
    "the governance and monitoring controls that keep those risks "
    "manageable for the organisations that adopt them. "
)

# A deliberately OFF-TOPIC passage. It repeats enough to fill a whole
# chunk on its own so that it is a single, cleanly separable chunk.
OFF_TOPIC_CHUNK = (
    "Chocolate chip cookie recipe: cream butter and sugar, fold in "
    "flour, then bake in a hot oven until golden. Preheat the oven, "
    "measure the sugar, soften the butter, and bake the cookies "
    "until the edges brown. For a softer cookie chill the dough; for "
    "a crisp cookie bake longer. The recipe lists butter, sugar and "
    "flour, and the oven temperature matters more than the mixing "
    "time. Cream the butter with the sugar, fold the flour in "
    "gently, and bake until the cookie is just set. This recipe is "
    "entirely about baking cookies and has nothing to do with cyber "
    "security, threat detection, machine learning or network defence."
)


def _build_long_response() -> str:
    """
    Build a ~5000+ char response: several on-topic sections, then the
    off-topic chunk placed past word ~700, then more on-topic sections.
    """

    prefix = (SECTION + "\n\n") * 8       # on-topic lead-in
    middle = OFF_TOPIC_CHUNK + "\n\n"      # buried off-topic chunk
    suffix = (SECTION + "\n\n") * 3       # on-topic tail
    return (prefix + middle + suffix).strip()


# =============================================================
# CHUNKING MECHANICS
# =============================================================

class ChunkSplittingTests(unittest.TestCase):

    def test_short_text_is_a_single_chunk(self):
        chunks = split_into_chunks(
            "A short on-topic sentence about cyber security."
        )
        self.assertEqual(len(chunks), 1)

    def test_chunks_respect_word_bounds(self):
        text = _build_long_response()
        chunks = split_into_chunks(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # No chunk should exceed the max band by more than one
            # sentence's worth of slack (paragraph merging stays <=
            # max_words; sentence-broken paragraphs are <= max_words).
            self.assertLessEqual(len(chunk.split()), CHUNK_MAX_WORDS)

    def test_no_chunk_splits_mid_sentence(self):
        text = _build_long_response()
        for chunk in split_into_chunks(text):
            stripped = chunk.strip()
            # Every chunk should end at a sentence boundary (a period).
            self.assertTrue(
                stripped.endswith("."),
                msg=f"chunk does not end a sentence: {stripped[-40:]!r}",
            )

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(split_into_chunks(""), [])
        self.assertEqual(split_into_chunks("   \n  \n "), [])


# =============================================================
# THE BLIND SPOT: chunked catches what whole-response misses
# =============================================================

class TruncationBlindSpotTests(unittest.TestCase):

    def setUp(self):
        self.assessor = SemanticAssessor(model=_KeywordEncoder())
        self.response = _build_long_response()

    def test_response_is_long_and_offtopic_chunk_is_buried(self):
        """Sanity: the fixture really is the described shape."""

        self.assertGreater(len(self.response), 4000)
        words = self.response.split()
        self.assertGreater(len(words), 800)

        # The off-topic chunk must be placed past word ~700.
        index = self.response.find("Chocolate chip cookie")
        self.assertGreater(index, 0)
        words_before = len(self.response[:index].split())
        self.assertGreater(words_before, 700)

    def test_whole_response_assess_does_NOT_flag_the_blind_spot(self):
        """
        Whole-response assess() is exactly what the blind spot defeats:
        the buried off-topic chunk is diluted, so the response-level
        deviation stays at the same low level as an all-on-topic
        response.
        """

        baseline = self.assessor.assess(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=(SECTION + "\n\n") * 5,   # all on-topic
        )

        whole = self.assessor.assess(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=self.response,
        )

        self.assertTrue(whole.assessed)
        self.assertTrue(baseline.assessed)

        # The off-topic injection barely moves the whole-response
        # deviation: whole-response assess() does NOT separate the
        # contaminated response from the clean baseline.
        self.assertAlmostEqual(
            whole.deviation_score,
            baseline.deviation_score,
            delta=0.10,
            msg=(
                "whole-response assess() was expected to MISS the "
                "buried off-topic chunk (blind spot), but it moved "
                f"({baseline.deviation_score} -> {whole.deviation_score})"
            ),
        )

    def test_assess_chunked_CATCHES_the_buried_offtopic_chunk(self):
        """
        Chunked assessment reproduces the blind spot fix: the worst
        chunk is the off-topic one, and its deviation is clearly
        higher than a typical on-topic chunk's.
        """

        result = self.assessor.assess_chunked(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=self.response,
        )

        self.assertIsInstance(result, ChunkedSemanticAssessment)
        self.assertTrue(result.assessed)
        self.assertGreater(result.chunk_count, 1)

        worst_text = result.chunk_texts[result.worst_chunk_index]
        self.assertIn("Chocolate chip cookie", worst_text)

        # Every other chunk is closer to the task than the worst one.
        on_topic = [
            a.deviation_score
            for i, a in enumerate(result.chunks)
            if i != result.worst_chunk_index and a.assessed
        ]
        self.assertTrue(on_topic)
        self.assertGreater(
            result.worst_chunk_deviation,
            max(on_topic),
        )

    def test_chunked_deviation_exceeds_whole_response_deviation(self):
        """
        The headline claim: the worst-chunk deviation is strictly
        greater than the whole-response deviation for the same text.
        """

        whole = self.assessor.assess(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=self.response,
        )
        chunked = self.assessor.assess_chunked(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=self.response,
        )

        self.assertGreater(
            chunked.worst_chunk_deviation,
            whole.deviation_score,
        )

    def test_all_on_topic_response_has_no_deviation_spike(self):
        """
        Control: an all-on-topic long response must NOT produce a
        worst-chunk spike, so the chunker is not flagging length
        itself.
        """

        clean = (SECTION + "\n\n") * 5
        result = self.assessor.assess_chunked(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=clean,
        )

        self.assertTrue(result.assessed)
        # No chunk is dramatically more deviant than the others.
        deviations = [a.deviation_score for a in result.chunks if a.assessed]
        self.assertTrue(deviations)
        self.assertLess(
            result.worst_chunk_deviation - min(deviations),
            0.25,
        )

    def test_existing_assess_is_unchanged_on_short_input(self):
        """
        Guard: the pre-existing whole-response assess() still behaves
        as before on a short output (regression protection).
        """

        short = "AI detects cyber threats in real time."
        result = self.assessor.assess(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=short,
        )
        self.assertTrue(result.assessed)
        self.assertGreaterEqual(result.task_similarity, 0.0)
        self.assertLessEqual(result.task_similarity, 1.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
