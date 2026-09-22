"""
Tests for the live event stream and the Streamlit dashboard (Task Q).

Covers the requirements:
  1. main.py execution still works with the observer enabled.
  2. The observer does not change experiment decisions or outcomes.
  3. Events become visible to the reader while the run is in progress.
  4. Refreshing does not duplicate events / start an experiment.
  5. The dashboard handles incomplete logs, missing files, completed runs.
  6. Existing tests continue to pass (full suite run separately).

No LLM experiments are launched: events are written by a real publisher
object, and the dashboard helpers are pure functions over events.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from security.live_events import (
    EventReader,
    LiveEventPublisher,
    read_all_events,
    read_run_marker,
)
import streamlit_app as app


def _tmp_paths():
    tmp = Path(tempfile.mkdtemp())
    return tmp / "events.jsonl", tmp / "run.json"


# =================================================================
# 1 + 5. PUBLISHER / READER BASICS
# =================================================================

class PublisherReaderTests(unittest.TestCase):

    def test_start_emit_finish_round_trip(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("task x", "shared_pool")
        pub.emit("mas_event", {"event_type": "tool_request", "sender": "researcher"})
        pub.finish_run("completed")

        events = read_all_events(events_path)
        kinds = [e["event_type"] for e in events]
        self.assertEqual(kinds[0], "experiment_started")
        self.assertIn("mas_event", kinds)
        self.assertEqual(kinds[-1], "experiment_completed")

        marker = read_run_marker(marker_path)
        self.assertEqual(marker["status"], "completed")
        self.assertEqual(marker["run_id"], pub.run_id)

    def test_every_event_carries_the_run_id(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("t", "shared_pool")
        pub.emit("mas_event", {"event_type": "tool_result"})

        for event in read_all_events(events_path):
            self.assertEqual(event["run_id"], pub.run_id)

    def test_missing_file_returns_empty_not_error(self):
        events_path, marker_path = _tmp_paths()
        self.assertEqual(read_all_events(events_path), [])
        self.assertIsNone(read_run_marker(marker_path))

    def test_reader_never_duplicates_events(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("t", "shared_pool")
        pub.emit("mas_event", {"event_type": "a"})
        pub.emit("mas_event", {"event_type": "b"})

        reader = EventReader(events_path)
        first = reader.read_new()
        second = reader.read_new()
        third = reader.read_new()

        self.assertEqual(len(first), 3)          # started + a + b
        self.assertEqual(second, [])
        self.assertEqual(third, [])

    def test_reader_does_not_return_partial_line(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("t", "shared_pool")

        reader = EventReader(events_path)
        reader.read_new()  # consume the started event

        # Append a half-written record (no trailing newline).
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write('{"run_id": "x", "event_type": "mas_event", "payl')

        self.assertEqual(reader.read_new(), [])  # partial is held back

        # Complete it; now it is delivered exactly once.
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write('oad": {}}\n')
        completed = reader.read_new()
        self.assertEqual(len(completed), 1)
        self.assertEqual(reader.read_new(), [])

    def test_new_run_truncation_resets_reader(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("t1", "shared_pool")
        pub.emit("mas_event", {"event_type": "a"})

        reader = EventReader(events_path)
        self.assertEqual(len(reader.read_new()), 2)

        # A second run truncates the file -> reader must not miss events.
        pub2 = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub2.start_run("t2", "shared_pool")
        pub2.emit("mas_event", {"event_type": "b"})

        fresh = reader.read_new()
        self.assertEqual(len(fresh), 2)
        for event in fresh:
            self.assertEqual(event["run_id"], pub2.run_id)

    def test_corrupt_line_is_skipped_not_fatal(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("t", "shared_pool")
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write("this is not json\n")
        pub.emit("mas_event", {"event_type": "ok"})

        events = read_all_events(events_path)
        kinds = [e["event_type"] for e in events]
        self.assertIn("mas_event", kinds)  # the good lines survive


# =================================================================
# NON-BLOCKING / FAILURE-ISOLATED
# =================================================================

class NonBlockingTests(unittest.TestCase):

    def test_publisher_never_raises_on_unwritable_path(self):
        # A directory that cannot be created as a file triggers errors
        # internally; the publisher must swallow them.
        bad = Path("Z:/nonexistent/does/not/exist/events.jsonl")
        pub = LiveEventPublisher(events_path=bad, marker_path=bad)
        # These must not raise.
        pub.start_run("t", "shared_pool")
        pub.emit("mas_event", {"event_type": "x"})
        pub.finish_run("completed")

    def test_emit_is_thread_safe(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("t", "shared_pool")

        def worker(n):
            for i in range(n):
                pub.emit("mas_event", {"event_type": "x", "i": i})

        threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = read_all_events(events_path)
        # No interleaved/corrupt lines: every line parsed.
        self.assertEqual(len(events), 1 + 200)


# =================================================================
# 3. EVENTS VISIBLE WHILE RUNNING
# =================================================================

class LiveVisibilityTests(unittest.TestCase):

    def test_marker_shows_running_before_completion(self):
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("in progress", "shared_pool")
        pub.emit("mas_event", {"event_type": "tool_request", "sender": "researcher"})

        # Mid-run: the marker says running and events are already visible.
        marker = read_run_marker(marker_path)
        self.assertEqual(marker["status"], "running")
        self.assertGreaterEqual(len(read_all_events(events_path)), 2)


# =================================================================
# 5 + DASHBOARD HELPERS (pure, no Streamlit runtime)
# =================================================================

class DashboardHelperTests(unittest.TestCase):

    def _events(self) -> list[dict]:
        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("task", "shared_pool")
        pub.emit("mas_event", {
            "event_type": "tool_request", "sender": "researcher",
            "receiver": "tool_manager", "tool_call": "internet_search",
            "content": "searching", "metadata": {}, "token_usage": 0,
        })
        pub.emit("mas_event", {
            "event_type": "investigation", "sender": "defender",
            "content": "tier activation", "metadata": {},
        })
        pub.emit("mas_event", {
            "event_type": "llm_usage", "sender": "coordinator", "token_usage": 120,
            "metadata": {
                "tokens_used": 120, "token_limit": 200000, "status": "consumed",
                "tokens_by_agent": {"coordinator": 120},
            },
        })
        pub.emit("security_observation", {
            "agent_id": "researcher", "security_score": 0.42,
            "investigation_required": False, "semantic_assessed": True,
            "deviation_score": 0.3, "task_similarity": 0.7,
            "subtask_similarity": 0.6,
            "detector_result": {"evidence_present": True},
            "response_preview": "About AI in cyber security...",
        })
        return read_all_events(events_path)

    def test_timeline_rows_include_kinds(self):
        rows = app.build_timeline_rows(self._events())
        kinds = {r["kind"] for r in rows}
        self.assertIn("tool_request", kinds)
        self.assertIn("security_observation", kinds)
        self.assertIn("llm_usage", kinds)

    def test_resource_state_from_events(self):
        res = app.latest_resource_state(self._events())
        self.assertEqual(res["tokens_used"], 120)
        self.assertEqual(res["token_limit"], 200000)
        self.assertEqual(res["llm_calls"], 1)
        self.assertEqual(res["tokens_by_agent"], {"coordinator": 120})

    def test_agent_statuses_and_security(self):
        agents = app.agent_statuses(self._events())
        self.assertIn("researcher", agents)
        self.assertIsNotNone(agents["researcher"]["security"])
        self.assertEqual(
            agents["researcher"]["security"]["security_score"], 0.42
        )

    def test_investigation_events_extracted(self):
        inv = app.investigation_events(self._events())
        self.assertEqual(len(inv), 1)
        self.assertEqual(app.mas_kind(inv[0]), "investigation")

    def test_status_summary_completed_and_waiting(self):
        # waiting (no marker)
        waiting = app._status_summary(None, [])
        self.assertEqual(waiting["status"], "waiting")

        events_path, marker_path = _tmp_paths()
        pub = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        pub.start_run("t", "shared_pool")
        pub.finish_run("completed")
        marker = read_run_marker(marker_path)
        done = app._status_summary(marker, read_all_events(events_path))
        self.assertEqual(done["status"], "completed")

    def test_old_events_not_shown_for_new_run(self):
        """
        Events from a previous run must not be attributed to the current
        run (the dashboard filters by marker run_id).
        """
        events_path, marker_path = _tmp_paths()
        old = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        old.start_run("old", "shared_pool")
        old.emit("mas_event", {"event_type": "old_event"})

        new = LiveEventPublisher(events_path=events_path, marker_path=marker_path)
        new.start_run("new", "shared_pool")
        new.emit("mas_event", {"event_type": "new_event"})

        marker = read_run_marker(marker_path)
        all_events = read_all_events(events_path)
        current = [e for e in all_events if e.get("run_id") == marker["run_id"]]
        kinds = {app.mas_kind(e) for e in current}
        self.assertIn("new_event", kinds)
        self.assertNotIn("old_event", kinds)


# =================================================================
# 2 + 4. OBSERVER DOES NOT CHANGE OUTCOMES / NO EPISODE FROM REFRESH
# =================================================================

class NoBehaviourChangeTests(unittest.TestCase):

    def test_publisher_attached_does_not_change_environment_events(self):
        """
        Running a stubbed episode with a publisher attached must produce
        the SAME observable events as the same episode without one.
        """

        from tests.conftest import (
            StubCalendar, StubEmail, StubLLM, StubReportWriter, StubSearchTool,
        )
        from attacks.scenarios import apply_attack
        from environment.mas_environment import MASEnvironment
        import agents.analyst as analyst_module
        import agents.coordinator as coordinator_module
        import agents.executor as executor_module
        import agents.outline as outline_module
        import agents.researcher as researcher_module

        def run(with_publisher: bool):
            stub = StubLLM()
            for module in (
                coordinator_module, outline_module, researcher_module,
                analyst_module, executor_module,
            ):
                module.get_llm = lambda: stub

            pub = None
            if with_publisher:
                events_path, marker_path = _tmp_paths()
                pub = LiveEventPublisher(
                    events_path=events_path, marker_path=marker_path
                )
                pub.start_run("task", "layered")

            env = MASEnvironment(
                topology_name="layered", event_publisher=pub
            )
            tools = env.tool_manager.tools
            for name in ("internet_search", "academic_search"):
                if name in tools:
                    tools[name] = StubSearchTool()
            if "report_writer" in tools:
                tools["report_writer"] = StubReportWriter()
            if "mock_calendar" in tools:
                tools["mock_calendar"] = StubCalendar()
            if "mock_calender" in tools:
                tools["mock_calender"] = StubCalendar()
            if "mock_email" in tools:
                tools["mock_email"] = StubEmail()

            apply_attack(env, "prompt_infection", task_id="q-1")
            env.execute_task("Write a proposal on AI in cyber security")

            # Compare the OBSERVABLE event signature (type, sender,
            # receiver) -- the decision-relevant trace.
            return [
                (e["event_type"], e.get("sender"), e.get("receiver"))
                for e in env.get_observable_events()
            ]

        without = run(False)
        with_pub = run(True)
        self.assertEqual(
            without,
            with_pub,
            "attaching the publisher changed the episode's observable events",
        )

    def test_dashboard_import_does_not_create_run_marker(self):
        """
        Importing the dashboard must not start an experiment or create a
        run marker on its own.
        """
        marker_path = Path("outputs/live_run.json")
        before = marker_path.exists()
        import importlib

        importlib.reload(app)
        after = marker_path.exists()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()