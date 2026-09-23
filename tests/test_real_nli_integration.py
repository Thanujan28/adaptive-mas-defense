"""
OPTIONAL integration test: exercise the REAL roberta-large-mnli NLI model.

This test is SKIPPED unless ``MAS_TEST_REAL_NLI=1`` is set (and the
model is already available locally), so the normal unit-test suite never
downloads anything and CI stays hermetic. It exists to verify the
end-to-end claim "the production ContradictionChecker runs real NLI with
evidence as the premise and the agent claim as the hypothesis".

Run it explicitly with the model already cached:

    MAS_TEST_REAL_NLI=1 python -m pytest tests/test_real_nli_integration.py -v

(On Windows PowerShell: $env:MAS_TEST_REAL_NLI=1; python -m pytest ...)
"""

from __future__ import annotations

import os
import unittest

import pytest

from security.contradiction_checker import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    ContradictionChecker,
)


_real_nli_enabled = os.getenv("MAS_TEST_REAL_NLI", "0") == "1"


@pytest.mark.skipif(
    not _real_nli_enabled,
    reason="Set MAS_TEST_REAL_NLI=1 to run the real roberta-large-mnli test.",
)
class RealNLIIntegrationTests(unittest.TestCase):
    """Loads the REAL model; skipped by default to keep CI download-free."""

    PREMISE = (
        "The study found that the treatment reduced errors by 20%."
    )

    def test_production_checker_uses_roberta_large_mnli(self):
        checker = ContradictionChecker()
        self.assertEqual(checker.model_name, "roberta-large-mnli")
        self.assertEqual(checker.source, "real")
        self.assertFalse(checker.is_stub)
        # Force the lazy load so the real pipeline is actually exercised.
        checker._get_model()
        self.assertIsNotNone(checker._model)

    def test_supported_claim_is_entailment(self):
        result = ContradictionChecker().check_contradiction(
            premise=self.PREMISE,
            hypothesis="The treatment reduced errors by 20%.",
        )
        self.assertEqual(result.label, LABEL_ENTAILMENT)
        self.assertEqual(result.premise, self.PREMISE)
        self.assertEqual(
            result.hypothesis, "The treatment reduced errors by 20%."
        )

    def test_conflicting_claim_is_contradiction(self):
        result = ContradictionChecker().check_contradiction(
            premise=self.PREMISE,
            hypothesis="The treatment increased errors by 20%.",
        )
        self.assertEqual(result.label, LABEL_CONTRADICTION)
        self.assertTrue(result.is_contradiction)

    def test_chunk_direction_keeps_evidence_as_premise(self):
        checker = ContradictionChecker()
        result = checker.check_chunk_against_evidence(
            chunk_text="The treatment increased errors by 20%.",
            evidence_chunks=[self.PREMISE],
        )
        self.assertEqual(result.label, LABEL_CONTRADICTION)
        # Evidence is the premise; the agent claim is the hypothesis.
        self.assertEqual(result.premise, self.PREMISE)
        self.assertIn("increased errors", result.hypothesis)


if __name__ == "__main__":
    unittest.main()
