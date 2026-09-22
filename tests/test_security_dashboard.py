"""
Tests for the security-observer GUI dashboard (Task P).

The data layer (build_dashboard_rows / chunk_row_from_decision /
agent_row_from_tiered) is pure and always tested. The Tk GUI is exercised
with a hidden root where a display is available; otherwise that single
test is skipped (headless CI).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.security_dashboard import (
    build_dashboard_rows,
    dump_dashboard,
    load_records,
    run_from_records,
)
from experiments.show_security_logs import write_template


def _run_demo_records():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.jsonl"
        write_template(path)
        records = load_records(path)
    return run_from_records(
        records,
        stub_encoder=True,
        stub_nli=True,
        stub_judge=True,
        strategy="formula",
    )


# =============================================================
# PURE DATA LAYER
# =============================================================

class ChunkRowTests(unittest.TestCase):

    def test_chunk_row_captures_all_three_tiers(self):
        observations = _run_demo_records()
        dashboard = build_dashboard_rows(observations, strategy="formula")

        # Find the chunk the judge ran on.
        judged = [
            chunk
            for agent in dashboard.agents
            for chunk in agent.chunks
            if chunk.tier3_ran
        ]
        self.assertTrue(judged, "expected at least one Tier-3-judged chunk")

        chunk = judged[0]
        # Tier 1 fields populated.
        self.assertTrue(chunk.semantic_assessed)
        self.assertGreaterEqual(chunk.deviation_score, 0.0)
        # Tier 2 fields populated and directional.
        self.assertTrue(chunk.nli_ran)
        self.assertEqual(chunk.nli_label, "contradiction")
        self.assertGreater(chunk.nli_confidence, 0.0)
        self.assertTrue(chunk.nli_premise)
        self.assertTrue(chunk.nli_hypothesis)
        self.assertIn("ai offers no benefit", chunk.nli_hypothesis.lower())
        # Tier 3 verdict.
        self.assertIs(chunk.tier3_contradicts, True)
        self.assertTrue(chunk.tier3_reasoning)

    def test_every_chunk_reports_tier1(self):
        observations = _run_demo_records()
        dashboard = build_dashboard_rows(observations)
        for agent in dashboard.agents:
            for chunk in agent.chunks:
                self.assertIn("tier1", chunk.tiers_ran)


class AgentRowTests(unittest.TestCase):

    def test_fused_state_is_exposed(self):
        observations = _run_demo_records()
        dashboard = build_dashboard_rows(observations)
        for agent in dashboard.agents:
            # The fused score and its components are present.
            self.assertGreaterEqual(agent.security_score, 0.0)
            self.assertLessEqual(agent.security_score, 1.0)
            self.assertGreaterEqual(agent.content_score, 0.0)
            self.assertGreaterEqual(agent.semantic_score, 0.0)
            self.assertIsInstance(agent.investigation_required, bool)

    def test_agent_aggregates_match_its_chunks(self):
        observations = _run_demo_records()
        dashboard = build_dashboard_rows(observations)
        for agent in dashboard.agents:
            flagged = sum(
                1 for c in agent.chunks if c.nli_label == "contradiction"
            )
            self.assertEqual(agent.contradiction_flagged_chunks, flagged)
            tier3 = sum(1 for c in agent.chunks if c.tier3_ran)
            self.assertEqual(agent.tier3_invocations, tier3)

    def test_worst_chunk_index_points_at_highest_deviation(self):
        observations = _run_demo_records()
        dashboard = build_dashboard_rows(observations)
        for agent in dashboard.agents:
            assessed = [
                (c.chunk_index, c.deviation_score)
                for c in agent.chunks if c.semantic_assessed
            ]
            if not assessed:
                continue
            expected_worst = max(assessed, key=lambda item: item[1])[0]
            self.assertEqual(agent.worst_chunk_index, expected_worst)


class EpisodeRollupTests(unittest.TestCase):

    def test_rollups_are_consistent_with_agents(self):
        observations = _run_demo_records()
        dashboard = build_dashboard_rows(observations, strategy="formula")

        self.assertEqual(
            dashboard.total_contradiction_flagged_chunks,
            sum(a.contradiction_flagged_chunks for a in dashboard.agents),
        )
        self.assertEqual(
            dashboard.total_tier3_invocations,
            sum(a.tier3_invocations for a in dashboard.agents),
        )
        self.assertGreaterEqual(dashboard.worst_chunk_deviation, 0.0)

    def test_strategy_is_carried_through(self):
        observations = _run_demo_records()
        for strategy in ("no_investigation", "brute_force", "formula"):
            dashboard = build_dashboard_rows(observations, strategy=strategy)
            self.assertEqual(dashboard.strategy, strategy)


class StrategyDifferenceTests(unittest.TestCase):

    def test_no_investigation_never_invokes_tier3(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.jsonl"
            write_template(path)
            records = load_records(path)
        observations = run_from_records(
            records,
            stub_encoder=True,
            stub_nli=True,
            stub_judge=True,
            strategy="no_investigation",
        )
        dashboard = build_dashboard_rows(observations)
        self.assertEqual(dashboard.total_tier3_invocations, 0)

    def test_brute_force_invokes_at_least_as_much_as_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.jsonl"
            write_template(path)
            records = load_records(path)

        brute = build_dashboard_rows(
            run_from_records(records, stub_encoder=True, stub_nli=True,
                             stub_judge=True, strategy="brute_force")
        )
        formula = build_dashboard_rows(
            run_from_records(records, stub_encoder=True, stub_nli=True,
                             stub_judge=True, strategy="formula")
        )
        self.assertGreaterEqual(
            brute.total_tier3_invocations, formula.total_tier3_invocations
        )


class HeadlessDumpTests(unittest.TestCase):

    def test_dump_prints_tiers_and_rollup(self):
        import io
        import contextlib

        observations = _run_demo_records()
        dashboard = build_dashboard_rows(observations)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dump_dashboard(dashboard)
        out = buf.getvalue()

        self.assertIn("Tier1 sem", out)
        self.assertIn("Tier2 NLI", out)
        self.assertIn("Tier3 judge", out)
        self.assertIn("EPISODE ROLL-UP", out)


# =============================================================
# CLI (headless)
# =============================================================

class CliDumpTests(unittest.TestCase):

    def test_cli_dump_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "demo.jsonl"
            write_template(demo)
            result = subprocess.run(
                [
                    sys.executable, "-m", "experiments.security_dashboard",
                    "--stub", "--stub-nli", "--stub-judge", "--dump",
                    "--from-jsonl", str(demo),
                ],
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EPISODE ROLL-UP", result.stdout)


# =============================================================
# GUI SMOKE TEST (skipped when no display)
# =============================================================

class GuiSmokeTests(unittest.TestCase):

    def test_gui_builds_widgets_without_entering_mainloop(self):
        """
        Exercise the Tk widget construction path in a hidden root.

        ``launch_gui`` is not called (it enters mainloop); instead a
        hidden root is created and the dashboard data is rendered into
        a Treeview, which is the part most likely to break. Skipped in
        headless environments where Tk cannot open a display.
        """

        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"tkinter unavailable: {exc}")

        try:
            root = tk.Tk()
            root.withdraw()
        except Exception as exc:  # pragma: no cover - headless
            self.skipTest(f"no display available: {exc}")

        try:
            observations = _run_demo_records()
            dashboard = build_dashboard_rows(observations, strategy="formula")

            tree = ttk.Treeview(
                root,
                columns=("chunk", "tiers", "deviation", "nli", "judge"),
                show="headings",
            )
            for agent in dashboard.agents:
                for chunk in agent.chunks:
                    tree.insert(
                        "", "end",
                        values=(
                            chunk.chunk_index,
                            ",".join(chunk.tiers_ran),
                            f"{chunk.deviation_score:.3f}",
                            chunk.nli_label or "-",
                            chunk.tier3_contradicts,
                        ),
                    )
            root.update_idletasks()
            self.assertEqual(
                len(tree.get_children()),
                sum(len(a.chunks) for a in dashboard.agents),
            )
        finally:
            root.destroy()


class LiveModeTests(unittest.TestCase):

    def test_live_flag_exists_and_defaults_off(self):
        from experiments.security_dashboard import parse_args

        self.assertFalse(parse_args([]).live)
        self.assertTrue(parse_args(["--live"]).live)
        self.assertFalse(parse_args([]).stub_tools)
        self.assertTrue(parse_args(["--live", "--stub-tools"]).stub_tools)
        self.assertIsNone(parse_args([]).nli_model)
        self.assertEqual(
            parse_args(["--nli-model", "x/y"]).nli_model, "x/y"
        )

    def test_live_does_not_fall_back_to_a_saved_run_or_demo(self):
        """
        --live must force a real episode even when outputs/last_run.jsonl
        or outputs/demo_logs.jsonl exist. We stub _run_live_episode so the
        test does not need Ollama, and assert it was called instead of the
        file-replay path.
        """

        import experiments.security_dashboard as dash

        called = {"live": False}

        def _fake_live(args):
            called["live"] = True
            return []  # no observations -> empty dashboard, fine

        original = dash._run_live_episode
        dash._run_live_episode = _fake_live
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # Create a demo file that WOULD be picked up if the code
                # wrongly fell back to it.
                demo = Path(tmp) / "demo.jsonl"
                write_template(demo)
                code = dash.main(
                    ["--live", "--dump", "--from-jsonl", str(demo)]
                )
            self.assertEqual(code, 0)
            self.assertTrue(
                called["live"],
                "--live did not run a real episode (fell back to a file)",
            )
        finally:
            dash._run_live_episode = original


if __name__ == "__main__":
    unittest.main()