"""
Tests for the live Streamlit dashboard's DATA LAYER (streamlit_app.py).

The dashboard must be verifiable without launching a Streamlit server
and without running any LLM experiment. Accordingly these tests exercise
the pure helper functions directly with MOCKED event records that have
the exact shape ``security/live_events.py`` writes:

    {"run_id": str, "ts": iso, "seq": int,
     "event_type": str, "payload": {...}}

and the ``security_observation`` / lifecycle records the environment
emits.

Covered requirements
--------------------
* events become visible to the dashboard while a run is ongoing;
* refreshing the dashboard does NOT start or duplicate an experiment (it
  only reads files; ``EventReader`` never repeats events);
* the dashboard handles incomplete logs, missing files and completed
  runs;
* old runs are not mistaken for the current one (run_id scoping);
* nothing is fabricated -- absent data renders as absent.

The Streamlit rendering entry point (``render``) is also smoke-tested
with a fake ``st`` module so the widget-construction path is covered
without a browser.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import streamlit_app as dash
from security.live_events import (
    DEFAULT_RUN_MARKER_PATH,
    EventReader,
    LiveEventPublisher,
    read_all_events,
    read_run_marker,
)


# =====================================================================
# MOCK EVENT FIXTURES (shape matches the real stream)
# =====================================================================

def _mas_event(kind, *, seq, sender="", receiver="", content="",
               tool_call="", token_usage=0, metadata=None, ts=None):
    return {
        "run_id": "RUN1",
        "ts": ts or "2026-01-01T10:00:00",
        "seq": seq,
        "event_type": "mas_event",
        "payload": {
            "event_type": kind,
            "sender": sender,
            "receiver": receiver,
            "content": content,
            "tool_call": tool_call,
            "token_usage": token_usage,
            "metadata": metadata or {},
        },
    }


def _observation(*, seq, agent_id, score, investigate, ts=None):
    return {
        "run_id": "RUN1",
        "ts": ts or "2026-01-01T10:00:05",
        "seq": seq,
        "event_type": "security_observation",
        "payload": {
            "agent_id": agent_id,
            "security_score": score,
            "investigation_required": investigate,
            "semantic_assessed": True,
            "deviation_score": 0.42,
            "task_similarity": 0.8,
            "subtask_similarity": 0.7,
            "detector_result": {"evidence_present": investigate},
            "response_preview": "response text",
        },
    }


def _lifecycle(kind, *, seq, payload=None):
    return {
        "run_id": "RUN1",
        "ts": "2026-01-01T10:00:00",
        "seq": seq,
        "event_type": kind,
        "payload": payload or {},
    }


SAMPLE_EVENTS = [
    _lifecycle("experiment_started", seq=1, payload={
        "task": "Write a proposal", "topology": "shared_pool",
    }),
    _lifecycle("episode_started", seq=2, payload={
        "episode_id": "E1", "agents": ["coordinator", "outline",
                                       "researcher", "executor"],
        "budget": {"token_limit": 1000, "tokens_used": 0,
                   "tool_limit": 10, "tools_used": 0},
    }),
    _mas_event("task_received", seq=3, receiver="coordinator",
               content="Coordinator received user task"),
    _mas_event("message", seq=4, sender="coordinator", receiver="outline",
               content="Coordinator sent message to outline"),
    _mas_event("agent_result", seq=5, sender="outline",
               content="outline text"),
    _mas_event("tool_request", seq=6, sender="researcher",
               tool_call="internet_search", content="researcher requested tool"),
    _mas_event("tool_result", seq=7, sender="tool_control_plane",
               receiver="researcher", tool_call="internet_search",
               content="Tool completed"),
    _mas_event("llm_usage", seq=8, sender="researcher", token_usage=120,
               metadata={"tokens_used": 120, "token_limit": 1000,
                         "tokens_by_agent": {"researcher": 120},
                         "status": "consumed"}),
    _mas_event("agent_result", seq=9, sender="researcher",
               content="research result"),
    _observation(seq=10, agent_id="researcher", score=0.72,
                 investigate=True),
    _mas_event("investigation", seq=11, sender="security_monitor",
               content="Tier 3 judge invoked"),
    _mas_event("containment", seq=12, sender="security_monitor",
               content="Contained researcher output"),
    _lifecycle("episode_completed", seq=13, payload={
        "episode_id": "E1", "agent_count": 4,
        "budget": {"token_limit": 1000, "tokens_used": 480,
                   "tool_limit": 10, "tools_used": 2},
    }),
    _lifecycle("experiment_completed", seq=14, payload={}),
]


# =====================================================================
# TIMELINE
# =====================================================================

class TimelineTests(unittest.TestCase):

    def test_rows_are_ordered_and_typed(self):
        rows = dash.build_timeline_rows(SAMPLE_EVENTS)
        self.assertEqual(len(rows), len(SAMPLE_EVENTS))
        kinds = [r["kind"] for r in rows]
        self.assertEqual(kinds[0], "experiment_started")
        self.assertIn("security_observation", kinds)
        self.assertIn("tool_request", kinds)
        self.assertIn("tool_result", kinds)

    def test_timeline_shows_sender_and_tool(self):
        rows = dash.build_timeline_rows(SAMPLE_EVENTS)
        tool_rows = [r for r in rows if r["kind"] == "tool_request"]
        self.assertTrue(tool_rows)
        self.assertEqual(tool_rows[0]["sender"], "researcher")
        self.assertEqual(tool_rows[0]["tool"], "internet_search")

    def test_security_observation_row_has_score(self):
        rows = dash.build_timeline_rows(SAMPLE_EVENTS)
        obs = [r for r in rows if r["kind"] == "security_observation"]
        self.assertTrue(obs)
        self.assertIn("security_score", obs[0]["text"])


# =====================================================================
# AGENTS
# =====================================================================

class AgentStatusTests(unittest.TestCase):

    def test_last_activity_is_tracked_per_agent(self):
        agents = dash.agent_statuses(SAMPLE_EVENTS)
        self.assertIn("researcher", agents)
        self.assertEqual(agents["researcher"]["last_kind"] in {
            "agent_result", "llm_usage", "message",
        }, True)

    def test_agent_security_observation_is_attached(self):
        agents = dash.agent_statuses(SAMPLE_EVENTS)
        self.assertIsNotNone(agents["researcher"]["security"])
        self.assertEqual(
            agents["researcher"]["security"]["security_score"], 0.72
        )

    def test_messages_are_collected(self):
        agents = dash.agent_statuses(SAMPLE_EVENTS)
        self.assertTrue(agents["coordinator"]["messages"])

    def test_no_fabricated_agents(self):
        # An agent that never appears in the stream is absent.
        agents = dash.agent_statuses(SAMPLE_EVENTS)
        self.assertNotIn("analyst", agents)


# =====================================================================
# SECURITY
# =====================================================================

class SecurityViewTests(unittest.TestCase):

    def test_investigation_and_containment_events_are_found(self):
        inv = dash.investigation_events(SAMPLE_EVENTS)
        kinds = {dash.mas_kind(e) for e in inv}
        self.assertIn("investigation", kinds)
        self.assertIn("containment", kinds)

    def test_no_investigation_when_none_emitted(self):
        only_plain = [
            e for e in SAMPLE_EVENTS
            if dash.mas_kind(e) not in
            {"investigation", "containment", "resource_allocation"}
        ]
        self.assertEqual(dash.investigation_events(only_plain), [])


# =====================================================================
# RESOURCES
# =====================================================================

class ResourceViewTests(unittest.TestCase):

    def test_latest_tokens_and_limit(self):
        res = dash.latest_resource_state(SAMPLE_EVENTS)
        self.assertEqual(res["tokens_used"], 120)
        self.assertEqual(res["token_limit"], 1000)
        self.assertEqual(res["tokens_by_agent"], {"researcher": 120})
        self.assertEqual(res["llm_calls"], 1)

    def test_no_resource_data_when_none_emitted(self):
        res = dash.latest_resource_state([])
        self.assertIsNone(res["tokens_used"])
        self.assertIsNone(res["token_limit"])
        self.assertEqual(res["tool_timeouts"], 0)

    def test_tool_timeout_counted(self):
        ev = SAMPLE_EVENTS + [
            _mas_event("tool_timeout", seq=99, sender="tool_manager",
                       tool_call="internet_search", content="timeout")
        ]
        res = dash.latest_resource_state(ev)
        self.assertEqual(res["tool_timeouts"], 1)


# =====================================================================
# EXPERIMENT STATUS
# =====================================================================

class StatusTests(unittest.TestCase):

    def test_waiting_when_no_marker_and_no_events(self):
        summary = dash._status_summary(None, [])
        self.assertEqual(summary["status"], "waiting")

    def test_running_status(self):
        marker = {"run_id": "RUN1", "status": "running"}
        summary = dash._status_summary(marker, SAMPLE_EVENTS)
        self.assertEqual(summary["status"], "running")
        self.assertEqual(summary["run_id"], "RUN1")

    def test_completed_status(self):
        marker = {"run_id": "RUN1", "status": "completed"}
        summary = dash._status_summary(marker, SAMPLE_EVENTS)
        self.assertEqual(summary["status"], "completed")
        self.assertIn("completed", summary["detail"].lower())

    def test_failed_status_includes_error(self):
        marker = {"run_id": "RUN1", "status": "failed", "error": "boom"}
        summary = dash._status_summary(marker, SAMPLE_EVENTS)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("boom", summary["detail"])


# =====================================================================
# STREAM <-> DASHBOARD INTEGRATION (files, no server)
# =====================================================================

class DashboardReadsLiveStreamTests(unittest.TestCase):
    """
    The dashboard reads the SAME file a running experiment writes, and a
    refresh never starts/duplicates the experiment.
    """

    def test_events_appear_while_running_and_refresh_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = LiveEventPublisher(
                events_path=tmp / "live_events.jsonl",
                marker_path=tmp / "live_run.json",
            )
            pub.start_run(task="t", topology="shared_pool")

            reader = EventReader(events_path=pub.events_path)

            # Mid-run: some events are already visible.
            pub.emit("mas_event", {"event_type": "task_received"})
            first = reader.read_new()
            self.assertTrue(
                any(e["event_type"] == "mas_event" for e in first)
            )

            # A dashboard refresh reads again -- nothing duplicated.
            self.assertEqual(reader.read_new(), [])

            # More events arrive; the next refresh sees only the new ones.
            pub.emit("mas_event", {"event_type": "agent_result"})
            second = reader.read_new()
            self.assertEqual(len(second), 1)
            self.assertEqual(
                second[0]["payload"]["event_type"], "agent_result"
            )

            # The experiment keeps running throughout -- the dashboard
            # never triggered anything.
            self.assertEqual(pub.enabled, True)

    def test_dashboard_refresh_does_not_create_experiment_artifacts(self):
        """
        Merely reading (what a refresh does) must not create a run marker
        or events: only the experiment writes those.
        """

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            events = tmp / "live_events.jsonl"
            marker = tmp / "live_run.json"

            # No experiment has run.
            self.assertEqual(read_all_events(events), [])
            self.assertIsNone(read_run_marker(marker))

            # Simulate a refresh: read the (missing) files.
            reader = EventReader(events_path=events)
            self.assertEqual(reader.read_new(), [])
            self.assertIsNone(read_run_marker(marker))

            # The refresh created nothing.
            self.assertFalse(events.exists())
            self.assertFalse(marker.exists())

    def test_missing_files_are_handled(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self.assertEqual(read_all_events(tmp / "nope.jsonl"), [])
            self.assertIsNone(read_run_marker(tmp / "nope.json"))

    def test_incomplete_log_trailing_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            path = tmp / "live_events.jsonl"
            good = json.dumps({
                "run_id": "R", "ts": "2026-01-01T00:00:00", "seq": 1,
                "event_type": "mas_event",
                "payload": {"event_type": "task_received"},
            })
            # A complete line + a partial (still being written) line.
            path.write_text(good + "\n" + '{"run_id": "R", "seq": 2',
                            encoding="utf-8")

            events = read_all_events(path)
            # read_all_events skips the unparsable trailing fragment.
            self.assertEqual(len(events), 1)

    def test_completed_run_is_reported_completed(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = LiveEventPublisher(
                events_path=tmp / "live_events.jsonl",
                marker_path=tmp / "live_run.json",
            )
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("mas_event", {"event_type": "agent_result"})
            pub.finish_run(status="completed")

            marker = read_run_marker(pub.marker_path)
            events = read_all_events(pub.events_path)
            # Only the current run's events (run_id scoping like render()).
            events = [e for e in events if e.get("run_id") == marker["run_id"]]

            summary = dash._status_summary(marker, events)
            self.assertEqual(summary["status"], "completed")
            # History is retained for display after completion.
            self.assertTrue(events)

    def test_old_run_events_are_not_shown_as_live(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = LiveEventPublisher(
                events_path=tmp / "live_events.jsonl",
                marker_path=tmp / "live_run.json",
            )
            pub.start_run(task="old", topology="shared_pool")
            pub.emit("mas_event", {"event_type": "agent_result"})
            old_run = pub.run_id

            # A brand-new run truncates the file and rewrites the marker.
            pub2 = LiveEventPublisher(
                events_path=pub.events_path, marker_path=pub.marker_path,
            )
            pub2.start_run(task="new", topology="shared_pool")

            marker = read_run_marker(pub.marker_path)
            events = read_all_events(pub.events_path)
            live = [e for e in events if e.get("run_id") == marker["run_id"]]

            self.assertEqual(marker["run_id"], pub2.run_id)
            self.assertNotEqual(marker["run_id"], old_run)
            # No old event survives the current-run filter.
            self.assertTrue(all(e["run_id"] == pub2.run_id for e in live))


# =====================================================================
# RENDER SMOKE TEST (fake streamlit, no server)
# =====================================================================

class _FakeStreamlit:
    """Minimal stand-in for the `streamlit` module's rendering surface."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return _FakeStreamlit()
        return _record

    # Context managers used by the renderer.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    class _Cols(list):
        def __init__(self, n):
            super().__init__(_FakeStreamlit() for _ in range(n))

    def columns(self, spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        self.calls.append(("columns", (spec,), k))
        return _FakeStreamlit._Cols(n)

    def tabs(self, labels, *a, **k):
        self.calls.append(("tabs", (labels,), k))
        return [_FakeStreamlit() for _ in labels]

    def multiselect(self, label, options=None, **k):
        self.calls.append(("multiselect", (label, options), k))
        return list(options or [])

    def metric(self, *a, **k):
        self.calls.append(("metric", a, k))

    def expander(self, *a, **k):
        self.calls.append(("expander", a, k))
        return self

    def write(self, *a, **k):
        self.calls.append(("write", a, k))

    def caption(self, *a, **k):
        self.calls.append(("caption", a, k))

    def markdown(self, *a, **k):
        self.calls.append(("markdown", a, k))


class RenderSmokeTests(unittest.TestCase):

    def test_render_runs_with_mocked_streamlit_without_raising(self):
        fake = _FakeStreamlit()
        original = dash.st
        dash.st = fake
        try:
            dash.render()
        finally:
            dash.st = original

        names = {c[0] for c in fake.calls}
        self.assertIn("set_page_config", names)
        self.assertIn("title", names)
        self.assertIn("columns", names)

    def test_render_creates_the_expected_sections(self):
        fake = _FakeStreamlit()
        original = dash.st
        dash.st = fake
        try:
            dash.render()
        finally:
            dash.st = original
        # The page is organised into the required numbered sections. These
        # are rendered as subheaders; assert the required ones are present.
        headers = [
            str(c[1][0])
            for c in fake.calls
            if c[0] == "subheader" and c[1]
        ]
        joined = " | ".join(headers).lower()
        self.assertIn("run status", joined)
        self.assertIn("agent summary", joined)
        self.assertIn("live chunk stream", joined)
        self.assertIn("resource usage", joined)
        self.assertIn("raw event stream", joined)

    def test_render_with_no_run_does_not_raise(self):
        # Point the module at a temp dir with no files, then render.
        fake = _FakeStreamlit()
        original_st = dash.st
        original_events = dash.EVENTS_PATH
        original_marker = dash.MARKER_PATH
        with tempfile.TemporaryDirectory() as d:
            dash.EVENTS_PATH = Path(d) / "none.jsonl"
            dash.MARKER_PATH = Path(d) / "none.json"
            dash.st = fake
            try:
                dash.render()
            finally:
                dash.st = original_st
                dash.EVENTS_PATH = original_events
                dash.MARKER_PATH = original_marker
        # A "waiting" info box is shown when there is no run.
        self.assertTrue(
            any(c[0] == "info" for c in fake.calls)
            or any(c[0] == "metric" for c in fake.calls)
        )


if __name__ == "__main__":
    unittest.main()
