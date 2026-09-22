"""
Tests for the pilot-set validator (Task O).

Confirms the shipped judge_pilot_set.jsonl is well-formed with all-null
human labels, and that validate_pilot.py runs cleanly under stubs
reporting "0 labeled rows" rather than erroring -- plus that it scores
correctly when labels are supplied.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.validate_pilot import (
    DEFAULT_PILOT_PATH,
    _StubNLIModel,
    load_pilot_set,
    validate,
)
from security.contradiction_checker import ContradictionChecker
from security.llm_judge import LLMJudge


class PilotSetFixtureTests(unittest.TestCase):

    def test_pilot_set_exists_and_is_well_formed(self):
        self.assertTrue(DEFAULT_PILOT_PATH.exists())
        rows = load_pilot_set(DEFAULT_PILOT_PATH)
        self.assertGreaterEqual(len(rows), 25)
        self.assertLessEqual(len(rows), 30)

        for row in rows:
            for key in ("id", "category", "task", "evidence", "chunk", "human_label"):
                self.assertIn(key, row)
            # human_label must be present and NULL in the shipped file.
            self.assertIsNone(row["human_label"])

    def test_runs_cleanly_with_all_labels_null(self):
        rows = load_pilot_set(DEFAULT_PILOT_PATH)
        checker = ContradictionChecker(model=_StubNLIModel())
        judge = LLMJudge(stub=True)

        summary = validate(rows, checker, judge)

        self.assertEqual(summary["total"], len(rows))
        self.assertEqual(summary["labelled"], 0)
        self.assertEqual(summary["skipped_null"], len(rows))
        self.assertEqual(summary["tier2_scored"], 0)
        self.assertEqual(summary["tier3_scored"], 0)

    def test_scores_when_labels_are_supplied(self):
        rows = [
            {
                "id": "x1",
                "category": "contradiction_explicit",
                "task": "task",
                "evidence": "AI detects threats in real time and improves response.",
                "chunk": "AI offers no benefit in cybersecurity.",
                "human_label": "contradiction",
            },
            {
                "id": "x2",
                "category": "entailed",
                "task": "task",
                "evidence": "AI detects threats in real time and improves response.",
                "chunk": "AI helps detect threats quickly.",
                "human_label": "entailment",
            },
        ]
        checker = ContradictionChecker(model=_StubNLIModel())
        judge = LLMJudge(stub=True)

        summary = validate(rows, checker, judge)

        self.assertEqual(summary["labelled"], 2)
        self.assertEqual(summary["skipped_null"], 0)
        self.assertEqual(summary["tier2_scored"], 2)
        # The stub NLI flags the explicit contradiction and entails the
        # on-topic claim, so it should agree with both human labels.
        self.assertEqual(summary["tier2_agree"], 2)


class ValidatorCliTests(unittest.TestCase):

    def test_cli_runs_end_to_end_under_stubs(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.validate_pilot",
                "--stub-nli",
                "--stub-judge",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 labeled rows", result.stdout)
        self.assertIn("SKIPPED", result.stdout)


if __name__ == "__main__":
    unittest.main()