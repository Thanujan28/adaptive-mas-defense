"""
Tests for the live event stream (security/live_events.py).

Covers the reliability requirements of the live dashboard:

  * the publisher appends complete JSON lines and is failure-isolated
    (a broken path must never affect the experiment);
  * the reader is incremental -- repeated refreshes never duplicate
    events, and a partially-written trailing line is never parsed;
  * a new run (fresh file / new run_id) is not mistaken for the old one;
  * missing files and completed runs are handled cleanly;
  * the run marker distinguishes running / completed / failed;
  * enabling the publisher on a real episode does not change the
    experiment's decisions or outcomes.

No live LLM is used: the shared ``stub_llm`` / ``stub_tools`` fixtures
from tests/conftest.py drive a deterministic offline episode.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from security.live_events import (
    EventReader,
    LiveEventPublisher,
    read_all_events,
    read_run_marker,
)


def _publisher(tmp: Path) -> LiveEventPublisher:
    return LiveEventPublisher(
        events_path=tmp / "live_events.jsonl",
        marker_path=tmp / "live_run.json",
    )


# =====================================================================
# PUBLISHER
# =====================================================================

class PublisherTests(unittest.TestCase):

    def test_start_run_writes_marker_and_truncates_events(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            # Pre-existing stale content from a previous run.
            pub.events_path.parent.mkdir(parents=True, exist_ok=True)
            pub.events_path.write_text('{"old": true}\n', encoding="utf-8")

            pub.start_run(task="t", topology="shared_pool")

            marker = read_run_marker(pub.marker_path)
            self.assertEqual(marker["status"], "running")
            self.assertEqual(marker["run_id"], pub.run_id)
            self.assertEqual(marker["task"], "t")

            events = read_all_events(pub.events_path)
            # Stale event gone; the start event is present.
            self.assertFalse(any(e.get("old") for e in events))
            self.assertEqual(events[0]["event_type"], "experiment_started")

    def test_emit_appends_one_complete_line_per_event(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("custom", {"a": 1})
            pub.emit("custom", {"a": 2})

            raw = pub.events_path.read_text(encoding="utf-8")
            self.assertTrue(raw.endswith("\n"))
            events = read_all_events(pub.events_path)
            customs = [e for e in events if e["event_type"] == "custom"]
            self.assertEqual([c["payload"]["a"] for c in customs], [1, 2])
            # seq is monotonic within a run.
            self.assertEqual(
                [c["seq"] for c in customs], sorted(c["seq"] for c in customs)
            )

    def test_finish_run_marks_completed(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.start_run(task="t", topology="shared_pool")
            pub.finish_run(status="completed", metadata={"n": 3})
            marker = read_run_marker(pub.marker_path)
            self.assertEqual(marker["status"], "completed")
            self.assertIsNotNone(marker["completed_at"])

    def test_finish_run_marks_failed_with_error(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.start_run(task="t", topology="shared_pool")
            pub.finish_run(status="failed", error="boom")
            marker = read_run_marker(pub.marker_path)
            self.assertEqual(marker["status"], "failed")
            self.assertEqual(marker["error"], "boom")

    def test_broken_path_never_raises_and_disables_streaming(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # A path whose parent is a FILE cannot be created as a dir.
            blocker = tmp / "blocker"
            blocker.write_text("x", encoding="utf-8")
            pub = LiveEventPublisher(
                events_path=blocker / "sub" / "events.jsonl",
                marker_path=blocker / "sub" / "marker.json",
            )
            # Must not raise.
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("custom", {"a": 1})
            pub.finish_run(status="completed")
            self.assertFalse(pub.enabled)

    def test_disabled_publisher_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.enabled = False
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("custom", {"a": 1})
            self.assertFalse(pub.events_path.exists())

    def test_unserialisable_payload_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.start_run(task="t", topology="shared_pool")

            class _Weird:
                pass

            pub.emit("custom", {"obj": _Weird()})  # default=str makes it OK
            # A truly unserialisable object (circular) must not crash:
            circular = {}
            circular["self"] = circular
            pub.emit("circular", circular)
            self.assertTrue(pub.enabled)
            events = read_all_events(pub.events_path)
            self.assertTrue(any(e["event_type"] == "custom" for e in events))


# =====================================================================
# READER
# =====================================================================

class ReaderTests(unittest.TestCase):

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            reader = EventReader(events_path=Path(d) / "nope.jsonl")
            self.assertEqual(reader.read_new(), [])

    def test_incremental_read_never_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("a", {})
            pub.emit("b", {})

            reader = EventReader(events_path=pub.events_path)
            first = reader.read_new()
            self.assertEqual(len(first), 3)  # start + a + b

            # Nothing new -> nothing returned (no duplication on refresh).
            self.assertEqual(reader.read_new(), [])

            pub.emit("c", {})
            second = reader.read_new()
            self.assertEqual([e["event_type"] for e in second], ["c"])
            self.assertEqual(reader.read_new(), [])

    def test_partial_trailing_line_is_not_parsed_until_complete(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            path = tmp / "events.jsonl"
            # A complete first line plus a partial second line.
            path.write_text(
                json.dumps({"event_type": "ok", "seq": 1}) + "\n"
                + '{"event_type": "pending", "seq": '
                ,
                encoding="utf-8",
            )
            reader = EventReader(events_path=path)
            events = reader.read_new()
            # Only the complete line is returned; the partial one is held.
            self.assertEqual([e["event_type"] for e in events], ["ok"])

            # The producer finishes the second line.
            with path.open("a", encoding="utf-8") as handle:
                handle.write('2}\n')

            events = reader.read_new()
            self.assertEqual([e["event_type"] for e in events], ["pending"])

    def test_corrupt_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            path = tmp / "events.jsonl"
            path.write_text(
                "not json\n"
                + json.dumps({"event_type": "good", "seq": 1}) + "\n",
                encoding="utf-8",
            )
            reader = EventReader(events_path=path)
            events = reader.read_new()
            self.assertEqual([e["event_type"] for e in events], ["good"])

    def test_new_run_resets_reader(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("old", {})

            reader = EventReader(events_path=pub.events_path)
            reader.read_new()

            # A SECOND run truncates the file -> reader restarts at 0.
            pub2 = LiveEventPublisher(
                events_path=pub.events_path, marker_path=pub.marker_path
            )
            pub2.start_run(task="t2", topology="shared_pool")
            pub2.emit("new", {})

            events = reader.read_new()
            kinds = [e["event_type"] for e in events]
            self.assertIn("experiment_started", kinds)
            self.assertIn("new", kinds)
            self.assertNotIn("old", kinds)
            # All visible events belong to the new run.
            self.assertTrue(all(e["run_id"] == pub2.run_id for e in events))

    def test_events_from_previous_run_are_distinguishable_by_run_id(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = _publisher(tmp)
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("old", {})
            old_events = read_all_events(pub.events_path)
            old_run = pub.run_id

            # New run writes to the SAME file (truncated by start_run).
            pub2 = LiveEventPublisher(
                events_path=pub.events_path, marker_path=pub.marker_path
            )
            pub2.start_run(task="t2", topology="shared_pool")
            new_events = read_all_events(pub.events_path)

            self.assertNotEqual(old_run, pub2.run_id)
            self.assertTrue(all(e["run_id"] == old_run for e in old_events))
            self.assertTrue(all(e["run_id"] == pub2.run_id for e in new_events))


# =====================================================================
# OBSERVER NON-INTERFERENCE (real episode, offline stubs)
# =====================================================================

class ObserverNonInterferenceTests(unittest.TestCase):
    """
    Enabling the live publisher must not change the experiment: the same
    task must yield the same final result and the same observable events,
    with and without a publisher attached.
    """

    def _run(self, tmp: Path, with_publisher: bool):
        from environment.mas_environment import MASEnvironment

        publisher = None
        if with_publisher:
            publisher = _publisher(tmp)
            publisher.start_run(task="T", topology="shared_pool")

        env = MASEnvironment(
            topology_name="shared_pool",
            event_publisher=publisher,
        )
        result = env.execute_task("Write a proposal on AI in cyber security")
        observable = env.get_observable_events()
        if publisher is not None:
            publisher.finish_run(status="completed")
        return result, observable, env.security_observations

    def test_publisher_does_not_change_result_or_events(self):
        # The stub_llm/stub_tools fixtures are not available in a plain
        # unittest method, so install the same offline stubs directly.
        import agents.coordinator as coordinator_module
        import agents.outline as outline_module
        import agents.researcher as researcher_module
        import agents.analyst as analyst_module
        import agents.executor as executor_module
        from tests.conftest import (
            StubLLM,
            StubSearchTool,
            StubReportWriter,
            StubCalendar,
            StubEmail,
        )

        stub = StubLLM()

        def _install_llm():
            mods = (
                coordinator_module, outline_module, researcher_module,
                analyst_module, executor_module,
            )
            originals = [(m, m.get_llm) for m in mods]
            for m in mods:
                m.get_llm = lambda: stub
            return originals

        def _install_tools(env):
            tools = env.tool_manager.tools
            search = StubSearchTool()
            for name in ("internet_search", "academic_search"):
                if name in tools:
                    tools[name] = search
            if "report_writer" in tools:
                tools["report_writer"] = StubReportWriter()
            if "mock_calendar" in tools:
                tools["mock_calendar"] = StubCalendar()
            if "mock_email" in tools:
                tools["mock_email"] = StubEmail()
            return env

        from environment.mas_environment import MASEnvironment

        originals = _install_llm()
        try:
            with tempfile.TemporaryDirectory() as d1, \
                 tempfile.TemporaryDirectory() as d2:

                # Run 1: no publisher.
                env_a = MASEnvironment(topology_name="shared_pool")
                _install_tools(env_a)
                result_a = env_a.execute_task(
                    "Write a proposal on AI in cyber security"
                )
                events_a = env_a.get_observable_events()
                obs_a = len(env_a.security_observations)

                # Run 2: with publisher.
                pub = _publisher(Path(d2))
                pub.start_run(task="T", topology="shared_pool")
                env_b = MASEnvironment(
                    topology_name="shared_pool", event_publisher=pub
                )
                _install_tools(env_b)
                result_b = env_b.execute_task(
                    "Write a proposal on AI in cyber security"
                )
                events_b = env_b.get_observable_events()
                obs_b = len(env_b.security_observations)
                pub.finish_run(status="completed")

            self.assertEqual(result_a, result_b)
            self.assertEqual(obs_a, obs_b)
            self.assertEqual(
                [e.get("event_type") for e in events_a],
                [e.get("event_type") for e in events_b],
            )
        finally:
            for module, original in originals:
                module.get_llm = original

    def test_stream_captured_the_running_episode(self):
        import agents.coordinator as coordinator_module
        import agents.outline as outline_module
        import agents.researcher as researcher_module
        import agents.analyst as analyst_module
        import agents.executor as executor_module
        from tests.conftest import (
            StubLLM,
            StubSearchTool,
            StubReportWriter,
            StubCalendar,
            StubEmail,
        )

        stub = StubLLM()
        mods = (
            coordinator_module, outline_module, researcher_module,
            analyst_module, executor_module,
        )
        originals = [(m, m.get_llm) for m in mods]
        for m in mods:
            m.get_llm = lambda: stub

        try:
            with tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                from environment.mas_environment import MASEnvironment

                pub = _publisher(tmp)
                pub.start_run(task="T", topology="shared_pool")
                env = MASEnvironment(
                    topology_name="shared_pool", event_publisher=pub
                )
                tools = env.tool_manager.tools
                search = StubSearchTool()
                for name in ("internet_search", "academic_search"):
                    if name in tools:
                        tools[name] = search
                if "report_writer" in tools:
                    tools["report_writer"] = StubReportWriter()
                if "mock_calendar" in tools:
                    tools["mock_calendar"] = StubCalendar()
                if "mock_email" in tools:
                    tools["mock_email"] = StubEmail()

                env.execute_task("Write a proposal on AI in cyber security")
                pub.finish_run(status="completed")

                events = read_all_events(pub.events_path)

            kinds = {e["event_type"] for e in events}
            # The lifecycle and the per-agent observations were streamed.
            self.assertIn("experiment_started", kinds)
            self.assertIn("episode_started", kinds)
            self.assertIn("episode_completed", kinds)
            self.assertIn("mas_event", kinds)
            self.assertIn("security_observation", kinds)
            self.assertTrue(
                all(e["run_id"] == pub.run_id for e in events)
            )
        finally:
            for module, original in originals:
                module.get_llm = original


if __name__ == "__main__":
    unittest.main()
