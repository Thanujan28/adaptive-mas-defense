"""
Tests for the tiered security-state viewer in
experiments/show_security_logs.py.

These confirm the viewer runs the full Tier 1/2/3 pipeline offline
(stub encoder + stub NLI + stub judge), that the detail view surfaces
each tier, and that the demo template exercises all three tiers.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.show_security_logs import (
    _StubNLIModel,
    parse_args,
    write_template,
)


class ArgumentTests(unittest.TestCase):

    def test_new_tiered_flags_parse(self):
        args = parse_args(
            [
                "--tiered",
                "--detail",
                "--stub-nli",
                "--stub-judge",
                "--strategy",
                "brute_force",
                "--chunk-chars",
                "80",
            ]
        )
        self.assertTrue(args.tiered)
        self.assertTrue(args.detail)
        self.assertTrue(args.stub_nli)
        self.assertTrue(args.stub_judge)
        self.assertEqual(args.strategy, "brute_force")
        self.assertEqual(args.chunk_chars, 80)

    def test_strategy_defaults_to_formula(self):
        self.assertEqual(parse_args([]).strategy, "formula")


class StubNLITests(unittest.TestCase):

    def test_contradiction_and_entailment_are_distinguished(self):
        model = _StubNLIModel()

        contradiction = model(
            {
                "text": "AI detects threats in real time and improves response.",
                "text_pair": "AI offers no benefit in cybersecurity.",
            }
        )
        labels = {item["label"]: item["score"] for item in contradiction}
        self.assertGreater(labels["contradiction"], labels["entailment"])

        entailment = model(
            {
                "text": "AI detects threats in real time.",
                "text_pair": "AI helps detect threats.",
            }
        )
        labels = {item["label"]: item["score"] for item in entailment}
        self.assertGreater(labels["entailment"], labels["contradiction"])


class TemplateTests(unittest.TestCase):

    def test_template_has_evidence_and_a_long_multi_chunk_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.jsonl"
            write_template(path)
            lines = [
                line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 3)

            import json

            records = [json.loads(line) for line in lines]
            # Every record carries an evidence list.
            for record in records:
                self.assertIn("evidence", record)
                self.assertTrue(record["evidence"])

            # At least one record is long enough to produce multiple
            # chunks.
            self.assertTrue(
                any(len(record["output"]) > 1000 for record in records)
            )


class ViewerEndToEndTests(unittest.TestCase):

    def test_viewer_runs_all_three_tiers_under_stubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "demo.jsonl"
            write_template(demo)

            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/show_security_logs.py",
                    "--stub",
                    "--tiered",
                    "--detail",
                    "--stub-nli",
                    "--stub-judge",
                    "--from-jsonl",
                    str(demo),
                ],
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout

        # The three tiers are all visible.
        self.assertIn("TIER 1 semantic", out)
        self.assertIn("TIER 2 NLI", out)
        self.assertIn("TIER 3 judge", out)

        # The judge actually ran on the contradiction demo record.
        self.assertIn("contradicts_evidence=True", out)
        self.assertIn("TIERED EPISODE ROLL-UP", out)
        self.assertIn("tier3_invocations", out)

        # Direction is stated explicitly.
        self.assertIn("premise=evidence, hypothesis=chunk", out)


if __name__ == "__main__":
    unittest.main()