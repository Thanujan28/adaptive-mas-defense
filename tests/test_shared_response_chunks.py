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

from security.contradiction_checker import (
    ContradictionChecker,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
)
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

# Evidence chosen to decisively ENTAIL the Markdown bullet claims below, so
# the winning pair is observable (evidence -> premise, claim -> hypothesis).
MARKDOWN_EVIDENCE = [
    "AI can improve image analysis and support radiologists in diagnosis."
]


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


# =====================================================================
# MARKDOWN RESPONSES: claim extraction, not heading fragments
# =====================================================================
#
# A heading such as ``**VII. Conclusion`` carries no proposition, so it
# must NEVER become an NLI hypothesis. The response chunk is still handed
# to Tier 2 WHOLE (the same-chunk invariant is re-checked here); only the
# checker's internal claim extraction changes.

class _RecordingNLI:
    """
    Deterministic stub NLI (no download) that records every
    ``(premise, hypothesis)`` pair it is asked about, so a test can assert
    exactly which hypotheses the checker built from the chunk.
    """

    def __init__(self, label: str = LABEL_ENTAILMENT, score: float = 0.9):
        self.label = label
        self.score = score
        self.seen: list[tuple[str, str]] = []

    def __call__(self, inputs, truncation=True):
        premise = inputs.get("text", "")
        hypothesis = inputs.get("text_pair", "")
        self.seen.append((premise, hypothesis))
        return [
            {"label": self.label, "score": self.score},
            {"label": LABEL_NEUTRAL, "score": 1.0 - self.score},
        ]


class MarkdownClaimExtractionTests(unittest.TestCase):
    """Tests 1-5: real claims are used, headings are not."""

    def _check(self, chunk_text, evidence=None, model=None):
        model = model or _RecordingNLI()
        checker = ContradictionChecker(model=model)
        result = checker.check_chunk_against_evidence(
            chunk_text=chunk_text,
            evidence_chunks=evidence or MARKDOWN_EVIDENCE,
        )
        return result, model

    def _observer(self, checker):
        return SecurityObserver(
            semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
            contradiction_checker=checker,
            log_enabled=False,
        )

    # ---- Test 1: Markdown heading is ignored -------------------------

    def test_heading_is_ignored_and_real_sentence_is_the_hypothesis(self):
        chunk = (
            "**VII. Conclusion**\n\n"
            "The integration of AI in medical imaging can improve diagnosis."
        )
        result, model = self._check(chunk)

        # The heading fragment never reached the NLI model as a hypothesis.
        hypotheses = [h for _, h in model.seen]
        self.assertTrue(hypotheses)
        for hypothesis in hypotheses:
            self.assertFalse(hypothesis.startswith("**VII"))
            self.assertNotEqual(hypothesis, "**VII.")
            self.assertNotEqual(hypothesis, "VII.")
            self.assertNotEqual(hypothesis, "**VII. Conclusion**")

        # The real sentence IS the hypothesis reported by the result.
        self.assertEqual(
            result.hypothesis,
            "The integration of AI in medical imaging can improve diagnosis.",
        )
        self.assertNotEqual(result.hypothesis, "**VII.")

    def test_tier2_receives_the_original_complete_chunk_from_tier1(self):
        """The whole Markdown chunk reaches Tier 2; nothing is re-chunked."""

        chunk = (
            "**VII. Conclusion**\n\n"
            "The integration of AI in medical imaging can improve diagnosis."
        )
        checker = _SpyChecker()
        observer = self._observer(checker)
        result = observer.observe_tiered(
            agent_id="researcher",
            response=chunk,
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=MARKDOWN_EVIDENCE,
            events=[],
        )

        tier1_chunks = [d.chunk_text for d in result.chunk_decisions]
        self.assertEqual(checker.seen_chunk_texts, tier1_chunks)
        for seen in checker.seen_chunk_texts:
            self.assertIn("**VII. Conclusion**", seen)

    # ---- Test 2: Markdown bullet claims are preserved ----------------

    def test_bullet_claims_are_preserved(self):
        chunk = (
            "**VII. Conclusion**\n\n"
            "* AI can improve image analysis.\n"
            "* AI can support radiologists.\n"
            "* Further research is required."
        )
        result, model = self._check(chunk)

        hypotheses = {h for _, h in model.seen}
        self.assertIn("AI can improve image analysis.", hypotheses)
        self.assertIn("AI can support radiologists.", hypotheses)
        self.assertIn("Further research is required.", hypotheses)
        # No bullet markers leaked into the hypothesis.
        for hypothesis in hypotheses:
            self.assertFalse(hypothesis.startswith("*"))
            self.assertFalse(hypothesis.startswith("-"))

    # ---- Test 3: normal prose still works ----------------------------

    def test_normal_prose_still_splits_into_sentences(self):
        chunk = "AI can improve image analysis. It can also assist diagnosis."
        result, model = self._check(chunk)

        hypotheses = {h for _, h in model.seen}
        self.assertIn("AI can improve image analysis.", hypotheses)
        self.assertIn("It can also assist diagnosis.", hypotheses)

    # ---- Test 4: evidence remains the premise ------------------------

    def test_evidence_is_premise_and_hypothesis_is_in_response_chunk(self):
        chunk = (
            "**VII. Conclusion**\n\n"
            "* The integration of AI in medical imaging devices has the "
            "potential to revolutionize radiology.\n"
            "* Further research is needed."
        )
        result, model = self._check(chunk)

        # Every pair the checker built: evidence is the premise, and the
        # hypothesis is drawn from the response chunk (never the evidence).
        for premise, hypothesis in model.seen:
            self.assertIn(premise, MARKDOWN_EVIDENCE)
            self.assertIn(hypothesis, chunk)
            self.assertNotEqual(premise, hypothesis)

        # The reported winning pair matches those invariants too.
        self.assertIn(result.premise, MARKDOWN_EVIDENCE)
        self.assertIn(result.hypothesis, chunk)
        self.assertNotEqual(result.premise, result.hypothesis)

    # ---- Test 5: heading-only chunk ---------------------------------

    def test_heading_only_chunk_does_not_fabricate_a_hypothesis(self):
        for heading in ("**VII. Conclusion**", "**VII.**", "### III. Results"):
            result, model = self._check(heading)
            # No NLI call was made (nothing to check), and the result is a
            # safe neutral/no-claim result with no fabricated hypothesis.
            self.assertEqual(model.seen, [])
            self.assertEqual(result.label, LABEL_NEUTRAL)
            self.assertEqual(result.hypothesis, "")
            self.assertEqual(result.premise, "")

    def test_heading_only_chunk_through_observer(self):
        """The Streamlit path: a heading-only chunk stays claim-free."""

        checker = _SpyChecker(model=_RecordingNLI())
        observer = self._observer(checker)
        result = observer.observe_tiered(
            agent_id="researcher",
            response="**VII. Conclusion**",
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=MARKDOWN_EVIDENCE,
            events=[],
        )
        for decision in result.chunk_decisions:
            self.assertNotEqual(decision.contradiction.hypothesis, "**VII.")
            self.assertEqual(decision.contradiction.hypothesis, "")

    # ---- The user's full example ------------------------------------

    def test_user_example_produces_the_two_real_claims(self):
        chunk = (
            "**VII. Conclusion**\n\n"
            "* The integration of AI in medical imaging devices has the "
            "potential to revolutionize the field of radiology and improve "
            "patient care.\n"
            "* Further research is needed to fully understand the benefits "
            "and limitations of AI in medical imaging."
        )
        result, model = self._check(chunk)
        hypotheses = [h for _, h in model.seen]

        self.assertIn(
            "The integration of AI in medical imaging devices has the "
            "potential to revolutionize the field of radiology and improve "
            "patient care.",
            hypotheses,
        )
        self.assertIn(
            "Further research is needed to fully understand the benefits "
            "and limitations of AI in medical imaging.",
            hypotheses,
        )
        # And NEVER the heading fragment.
        for forbidden in ("**VII.", "VII.", "**VII. Conclusion**"):
            self.assertNotIn(forbidden, hypotheses)


if __name__ == "__main__":
    unittest.main()