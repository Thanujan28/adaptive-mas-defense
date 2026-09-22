"""
Tests for the NLI contradiction checker (Task I).

No model download: a deterministic stub NLI model is injected through
``ContradictionChecker(model=...)`` (same pattern as
``SemanticAssessor(model=...)``).

The three required cases use the fixed premise

    "AI-powered systems can detect threats in real-time, enabling rapid
     response and mitigation."

with hypotheses chosen to be entailed (a), contradicted (b) and
unrelated (c).
"""

from __future__ import annotations

import unittest

from security.contradiction_checker import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    ContradictionChecker,
    ContradictionResult,
)


PREMISE = (
    "AI-powered systems can detect threats in real-time, enabling "
    "rapid response and mitigation."
)


class _StubNLIModel:
    """
    Deterministic stand-in for a HuggingFace text-classification NLI
    pipeline.

    It parses the ``{"text": premise, "text_pair": hypothesis}`` call,
    then applies a small, explicit rule set that implements the intended
    NLI behaviour of the three test cases:

      * a hypothesis whose content words are largely covered by the
        premise (or whose key terms appear in it) -> entailment,
      * a hypothesis that NEGATES a proposition the premise asserts
        (detected by an explicit negation cue against an asserted term)
        -> contradiction,
      * otherwise -> neutral.

    The rule set is deliberately simple; it exists to exercise the
    checker's wiring and direction, NOT to model NLI.
    """

    # Terms the premise asserts positively; a hypothesis claiming "no
    # benefit" about these is a contradiction.
    NEGATION_CUES = ("no benefit", "not ", "never", "cannot", "does not", "no ")

    def __call__(self, inputs, truncation=True):
        premise = inputs["text"]
        hypothesis = inputs["text_pair"]
        return self._score(premise, hypothesis)

    # -- stub logic ----------------------------------------------------

    def _score(self, premise: str, hypothesis: str):
        p = premise.lower()
        h = hypothesis.lower()

        # (b) Contradiction: the hypothesis negates a benefit the
        # premise asserts. "AI offers no benefit in cybersecurity."
        if "benefit" in h and any(cue in h for cue in ("no ", "not", "never")):
            if "benefit" in p or "mitigation" in p or "response" in p:
                return [
                    {"label": "contradiction", "score": 0.97},
                    {"label": "entailment", "score": 0.02},
                    {"label": "neutral", "score": 0.01},
                ]

        # (a) Entailment: hypothesis shares the premise's core claim
        # (faster response / mitigation). "AI enables organizations to
        # respond to threats more quickly."
        entailment_terms = ("respond", "response", "threat", "quickly",
                            "faster", "detect")
        hits = sum(1 for term in entailment_terms if term in h and term in p)
        if hits >= 1:
            return [
                {"label": "entailment", "score": 0.95},
                {"label": "neutral", "score": 0.04},
                {"label": "contradiction", "score": 0.01},
            ]

        # (c) Neutral: unrelated hypothesis.
        return [
            {"label": "neutral", "score": 0.9},
            {"label": "entailment", "score": 0.06},
            {"label": "contradiction", "score": 0.04},
        ]


def _checker() -> ContradictionChecker:
    return ContradictionChecker(model=_StubNLIModel())


# =============================================================
# THE THREE REQUIRED CASES
# =============================================================

class RequiredCasesTests(unittest.TestCase):

    def test_case_a_entailment(self):
        result = _checker().check_contradiction(
            premise=PREMISE,
            hypothesis=(
                "AI enables organizations to respond to threats more "
                "quickly."
            ),
        )
        self.assertIsInstance(result, ContradictionResult)
        self.assertEqual(result.label, LABEL_ENTAILMENT)
        self.assertGreater(result.confidence, 0.5)

    def test_case_b_contradiction(self):
        result = _checker().check_contradiction(
            premise=PREMISE,
            hypothesis="AI offers no benefit in cybersecurity.",
        )
        self.assertEqual(result.label, LABEL_CONTRADICTION)
        self.assertGreater(result.confidence, 0.5)

    def test_case_c_neutral(self):
        result = _checker().check_contradiction(
            premise=PREMISE,
            hypothesis=(
                "The city council approved a new parking ordinance for "
                "downtown residents."
            ),
        )
        self.assertEqual(result.label, LABEL_NEUTRAL)


# =============================================================
# DIRECTION MATTERS
# =============================================================

class DirectionTests(unittest.TestCase):

    def test_direction_is_recorded(self):
        """Evidence must always be recorded as the premise."""

        result = _checker().check_contradiction(
            premise=PREMISE,
            hypothesis="AI offers no benefit in cybersecurity.",
        )
        self.assertEqual(result.premise, PREMISE)
        self.assertEqual(
            result.hypothesis, "AI offers no benefit in cybersecurity."
        )

    def test_check_chunk_puts_evidence_as_premise(self):
        checker = _checker()
        result = checker.check_chunk_against_evidence(
            chunk_text="AI offers no benefit in cybersecurity.",
            evidence_chunks=[PREMISE],
        )
        # The evidence text is the premise, the agent sentence the
        # hypothesis.
        self.assertEqual(result.premise, PREMISE)
        self.assertEqual(
            result.hypothesis, "AI offers no benefit in cybersecurity."
        )
        self.assertEqual(result.label, LABEL_CONTRADICTION)


# =============================================================
# CHUNK-LEVEL BEHAVIOUR
# =============================================================

class ChunkAgainstEvidenceTests(unittest.TestCase):

    def test_strongest_contradiction_across_sentences(self):
        chunk = (
            "AI speeds up threat response. "
            "AI offers no benefit in cybersecurity."
        )
        result = _checker().check_chunk_against_evidence(
            chunk_text=chunk,
            evidence_chunks=[PREMISE],
        )
        self.assertEqual(result.label, LABEL_CONTRADICTION)

    def test_best_entailment_when_no_contradiction(self):
        chunk = "AI helps detect threats quickly."
        result = _checker().check_chunk_against_evidence(
            chunk_text=chunk,
            evidence_chunks=[PREMISE],
        )
        self.assertEqual(result.label, LABEL_ENTAILMENT)

    def test_neutral_when_evidence_entails_nothing(self):
        result = _checker().check_chunk_against_evidence(
            chunk_text="Penguins live in the Antarctic.",
            evidence_chunks=[PREMISE],
        )
        self.assertEqual(result.label, LABEL_NEUTRAL)

    def test_empty_inputs_are_neutral_not_an_error(self):
        checker = _checker()
        self.assertEqual(
            checker.check_contradiction("", "x").label, LABEL_NEUTRAL
        )
        self.assertEqual(
            checker.check_contradiction("x", "").label, LABEL_NEUTRAL
        )
        self.assertEqual(
            checker.check_chunk_against_evidence("", [PREMISE]).label,
            LABEL_NEUTRAL,
        )
        self.assertEqual(
            checker.check_chunk_against_evidence("text", []).label,
            LABEL_NEUTRAL,
        )


# =============================================================
# LABEL NORMALISATION
# =============================================================

class LabelNormalisationTests(unittest.TestCase):

    def test_model_specific_label_spellings_map_to_canonical(self):
        class _Weird(_StubNLIModel):
            def _score(self, premise, hypothesis):
                return [
                    {"label": "LABEL_2", "score": 0.9},
                    {"label": "LABEL_0", "score": 0.1},
                ]

        # LABEL_2/LABEL_0 have no entail/contradict/neutral substring, so
        # they pass through lower-cased unchanged (documented fallback).
        result = ContradictionChecker(model=_Weird()).check_contradiction(
            "p", "h"
        )
        self.assertEqual(result.label, "label_2")

    def test_standard_spellings_are_canonicalised(self):
        class _Standard(_StubNLIModel):
            def _score(self, premise, hypothesis):
                return [
                    {"label": "entailment", "score": 0.1},
                    {"label": "neutral", "score": 0.2},
                    {"label": "contradiction", "score": 0.7},
                ]

        result = ContradictionChecker(model=_Standard()).check_contradiction(
            "p", "h"
        )
        self.assertEqual(result.label, LABEL_CONTRADICTION)


if __name__ == "__main__":
    unittest.main()
