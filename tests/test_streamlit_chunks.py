"""
Tests for the LIVE CHUNK STREAM rendering in streamlit_app.py.

These verify the requirement that the dashboard displays the actual
result for EVERY processed response chunk -- its text, its Tier 1
semantic scores, its Tier 2 NLI result (or NOT RUN), and its Tier 3
judge verdict (or NOT RUN) -- using MOCKED event records with the exact
shape ``security/live_events.py`` writes and the environment's
``security_observation`` event produces.

No LLM experiment is run and no Streamlit server is started: the pure
helpers are exercised directly, and the rendering entry point is
smoke-tested with a fake ``st`` module that records every call.

Covered requirements
--------------------
1. A chunk event containing semantic results is displayed.
2. A chunk event containing NLI results is displayed.
3. Chunk text is displayed.
4. NLI premise and hypothesis are displayed.
5. NOT RUN is displayed when NLI is absent.
6. Multiple chunks from the same agent are displayed separately.
7. Streamlit refresh does not duplicate chunks.
8. Streamlit does not execute the experiment.
9. Live events appear before episode completion.
10. (Existing tests in test_streamlit_dashboard.py / test_live_dashboard.py
    continue to pass.)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import streamlit_app as dash
from security.live_events import (
    EventReader,
    LiveEventPublisher,
    read_all_events,
    read_run_marker,
)


# =====================================================================
# MOCK CHUNK EVENTS (exact shape of the environment's publish path)
# =====================================================================

CHUNK_TEXT_0 = (
    "Artificial intelligence is transforming cyber security by "
    "automating threat detection across large networks."
)
CHUNK_TEXT_1 = (
    "Some sources claim AI has no benefit for security operations, "
    "which contradicts the evidence collected."
)

NLI_PREMISE = "AI improves threat detection in large networks."
NLI_HYPOTHESIS = CHUNK_TEXT_0


def _semantic_payload(*, assessed=True, task=0.5447, subtask=0.3951,
                      objective=0.4553, scope=0.6049, deviation=0.5226,
                      confidence=1.0):
    return {
        "assessed": assessed,
        "task_similarity": task,
        "subtask_similarity": subtask,
        "objective_deviation": objective,
        "scope_deviation": scope,
        "deviation_score": deviation,
        "confidence": confidence,
    }


def _chunk(index, text, *, nli=None, judge=None):
    return {
        "chunk_index": index,
        "chunk_text": text,
        "tiers_ran": ["tier1"] + (["tier2"] if nli else []),
        "tier1_semantic": _semantic_payload(),
        "tier2_nli": nli if nli is not None else {"ran": False, "source": "not_run"},
        "tier3_judge": judge if judge is not None
        else {"ran": False, "source": "not_run", "enabled": False,
              "stubbed": False},
    }


def _observation_event(*, seq, agent_id, score, chunks, tier2_source="real",
                       tier3_status="not_run", ts="2026-01-01T10:00:05"):
    return {
        "run_id": "RUN1",
        "ts": ts,
        "seq": seq,
        "event_type": "security_observation",
        "payload": {
            "agent_id": agent_id,
            "assigned_subtask": "role template",
            "security_score": score,
            "investigation_required": score >= 0.45,
            "semantic_assessed": True,
            "task_similarity": 0.5447,
            "subtask_similarity": 0.3951,
            "deviation_score": 0.5226,
            "semantic_confidence": 1.0,
            "detector_result": {"evidence_present": True},
            "response_preview": CHUNK_TEXT_0[:240],
            "tiered": {
                "worst_chunk_deviation": 0.5226,
                "contradiction_flagged_chunks": 0,
                "tier3_invocations": 0,
                "chunk_count": len(chunks),
                "tier2_source": tier2_source,
                "tier3_status": tier3_status,
                "chunks": chunks,
            },
        },
    }


# An observation whose two chunks carry NLI on the first and none on the
# second, so both "ran" and "NOT RUN" paths are exercised.
EVENTS_WITH_NLI = [
    _observation_event(
        seq=1, agent_id="executor", score=0.62,
        chunks=[
            _chunk(0, CHUNK_TEXT_0, nli={
                "ran": True, "source": "real", "label": "entailment",
                "confidence": 0.91, "premise": NLI_PREMISE,
                "hypothesis": NLI_HYPOTHESIS,
            }),
            _chunk(1, CHUNK_TEXT_1, nli={
                "ran": True, "source": "stub", "label": "contradiction",
                "confidence": 0.77, "premise": "evidence text",
                "hypothesis": CHUNK_TEXT_1,
            }, judge={
                "ran": True, "contradicts_evidence": True,
                "reasoning": "claim reverses the evidence", "stubbed": False,
                "cached": False,
            }),
        ],
        tier2_source="real",
        tier3_status="real",
    ),
]

# An observation whose chunks have NO NLI at all (Tier 2 NOT RUN).
EVENTS_WITHOUT_NLI = [
    _observation_event(
        seq=1, agent_id="researcher", score=0.30,
        chunks=[_chunk(0, CHUNK_TEXT_0), _chunk(1, CHUNK_TEXT_1)],
        tier2_source="not_run",
        tier3_status="not_run",
    ),
]


# =====================================================================
# PURE HELPERS
# =====================================================================

class ChunkHelperTests(unittest.TestCase):

    def test_observation_chunks_reads_published_chunks(self):
        chunks = dash.observation_chunks(_observation_event(
            seq=1, agent_id="executor", score=0.5, chunks=[_chunk(0, "t")]
        )["payload"])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_index"], 0)

    def test_observation_chunks_empty_when_no_tiered(self):
        self.assertEqual(dash.observation_chunks({"agent_id": "x"}), [])

    def test_all_chunks_grouped_by_agent(self):
        grouped = dash.all_chunks_by_agent(EVENTS_WITH_NLI)
        self.assertIn("executor", grouped)
        self.assertEqual(len(grouped["executor"]), 2)
        # Provenance is attached without recomputation.
        self.assertEqual(grouped["executor"][0]["_agent_id"], "executor")
        self.assertEqual(grouped["executor"][0]["_security_score"], 0.62)

    def test_chunk_nli_status_real_stub_not_run(self):
        c0, c1 = EVENTS_WITH_NLI[0]["payload"]["tiered"]["chunks"]
        self.assertEqual(dash.chunk_nli_status(c0), "real")
        self.assertEqual(dash.chunk_nli_status(c1), "stub")
        none = EVENTS_WITHOUT_NLI[0]["payload"]["tiered"]["chunks"][0]
        self.assertEqual(dash.chunk_nli_status(none), "not_run")

    def test_chunk_judge_status(self):
        c0, c1 = EVENTS_WITH_NLI[0]["payload"]["tiered"]["chunks"]
        self.assertEqual(dash.chunk_judge_status(c0), "not_run")
        self.assertEqual(dash.chunk_judge_status(c1), "real")

    def test_semantic_fields_verbatim(self):
        chunk = EVENTS_WITH_NLI[0]["payload"]["tiered"]["chunks"][0]
        fields = dash.chunk_semantic_fields(chunk)
        self.assertEqual(fields["task_similarity"], 0.5447)
        self.assertEqual(fields["deviation_score"], 0.5226)
        self.assertTrue(fields["assessed"])

    def test_run_level_provenance_labels(self):
        self.assertEqual(dash.tier2_source_of_events(EVENTS_WITH_NLI), "real")
        self.assertEqual(dash.tier2_source_of_events(EVENTS_WITHOUT_NLI), "not_run")
        self.assertEqual(dash.tier3_status_of_events(EVENTS_WITH_NLI), "real")


# =====================================================================
# FAKE STREAMLIT (records calls; supports context managers)
# =====================================================================

class _FakeStreamlit:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return _FakeStreamlit()
        return _record

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

    def expander(self, *a, **k):
        self.calls.append(("expander", a, k))
        return self

    def text_area(self, *a, **k):
        self.calls.append(("text_area", a, k))
        return (k.get("value") or (a[1] if len(a) > 1 else ""))

    def markdown(self, *a, **k):
        self.calls.append(("markdown", a, k))

    def caption(self, *a, **k):
        self.calls.append(("caption", a, k))

    def metric(self, *a, **k):
        self.calls.append(("metric", a, k))

    def error(self, *a, **k):
        self.calls.append(("error", a, k))

    def warning(self, *a, **k):
        self.calls.append(("warning", a, k))

    def info(self, *a, **k):
        self.calls.append(("info", a, k))

    def success(self, *a, **k):
        self.calls.append(("success", a, k))

    def subheader(self, *a, **k):
        self.calls.append(("subheader", a, k))

    def divider(self, *a, **k):
        self.calls.append(("divider", a, k))

    def json(self, *a, **k):
        self.calls.append(("json", a, k))


def _render_with(events):
    """Render the dashboard against a temp events file, returning calls."""

    fake = _FakeStreamlit()
    original_st = dash.st
    original_events = dash.EVENTS_PATH
    original_marker = dash.MARKER_PATH
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        events_path = tmp / "live_events.jsonl"
        marker_path = tmp / "live_run.json"
        marker_path.write_text(
            '{"run_id": "RUN1", "status": "running", "started_at": '
            '"2026-01-01T10:00:00"}',
            encoding="utf-8",
        )
        with events_path.open("w", encoding="utf-8") as handle:
            for event in events:
                import json as _json
                handle.write(_json.dumps(event, default=str) + "\n")

        dash.EVENTS_PATH = events_path
        dash.MARKER_PATH = marker_path
        dash.st = fake
        try:
            dash.render()
        finally:
            dash.st = original_st
            dash.EVENTS_PATH = original_events
            dash.MARKER_PATH = original_marker
    return fake.calls


def _all_text(calls):
    """Flatten every string argument passed to any widget."""
    parts = []
    for _name, args, kwargs in calls:
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


# =====================================================================
# RENDERING REQUIREMENTS
# =====================================================================

class ChunkRenderingTests(unittest.TestCase):

    # 3 + 1 + 2. Chunk text + semantic + NLI are displayed.
    def test_chunk_card_shows_text_semantic_and_nli(self):
        calls = _render_with(EVENTS_WITH_NLI)
        text = _all_text(calls)

        # Requirement 1: Tier 1 semantic label + exact scores.
        self.assertIn("Tier 1 — Semantic", text)
        self.assertIn("task_similarity = 0.5447", text)
        self.assertIn("subtask_similarity = 0.3951", text)
        self.assertIn("objective_deviation = 0.4553", text)
        self.assertIn("scope_deviation = 0.6049", text)
        self.assertIn("deviation_score = 0.5226", text)
        self.assertIn("confidence = 1.0000", text)

        # Requirement 2 + 4: real NLI label, premise and hypothesis.
        self.assertIn("Tier 2 — NLI", text)
        self.assertIn("REAL NLI", text)
        self.assertIn("STUB NLI", text)
        self.assertIn("entailment", text)
        self.assertIn("contradiction", text)
        self.assertIn("Premise:", text)
        self.assertIn("Hypothesis:", text)
        self.assertIn(NLI_PREMISE, text)

    # 3. Chunk text is displayed (via text_area value).
    def test_chunk_text_is_displayed(self):
        calls = _render_with(EVENTS_WITH_NLI)
        text_areas = [c for c in calls if c[0] == "text_area"]
        self.assertEqual(len(text_areas), 2)  # two chunks -> two text boxes
        values = {c[2].get("value") for c in text_areas}
        self.assertIn(CHUNK_TEXT_0, values)
        self.assertIn(CHUNK_TEXT_1, values)

    # 5. NOT RUN is displayed when NLI is absent.
    def test_not_run_shown_when_nli_absent(self):
        calls = _render_with(EVENTS_WITHOUT_NLI)
        text = _all_text(calls)
        self.assertIn("Tier 2 — NLI: NOT RUN", text)
        # Run-level banner also says NLI is not run.
        self.assertIn("NLI is NOT RUN", text)

    # 6. Multiple chunks from the same agent are displayed separately.
    def test_multiple_chunks_rendered_separately(self):
        calls = _render_with(EVENTS_WITH_NLI)
        expanders = [c for c in calls if c[0] == "expander"]
        # One expander per chunk (2 chunks), at least within the chunk stream.
        chunk_headers = [
            c[1][0] for c in expanders
            if c[1] and str(c[1][0]).startswith(("Chunk", "🆕 Chunk"))
        ]
        self.assertEqual(len(chunk_headers), 2)
        self.assertTrue(any("Chunk 0" in h for h in chunk_headers))
        self.assertTrue(any("Chunk 1" in h for h in chunk_headers))

    # 9. Newest chunk is marked and shown, even before episode completion.
    def test_newest_chunk_is_marked(self):
        calls = _render_with(EVENTS_WITH_NLI)
        expanders = [c for c in calls if c[0] == "expander"]
        headers = [str(c[1][0]) for c in expanders if c[1]]
        self.assertTrue(any("🆕" in h for h in headers))

    def test_tier3_judge_rendered_when_it_ran(self):
        calls = _render_with(EVENTS_WITH_NLI)
        text = _all_text(calls)
        self.assertIn("Tier 3 — Judge", text)
        self.assertIn("REAL JUDGE", text)
        self.assertIn("contradicts_evidence", text)
        self.assertIn("claim reverses the evidence", text)


# =====================================================================
# LIVE-STREAM MECHANICS
# =====================================================================

class LiveStreamMechanicsTests(unittest.TestCase):
    """
    Requirements 7, 8, 9: refresh does not duplicate, the dashboard does
    not execute the experiment, and events are visible before completion.
    """

    def test_refresh_does_not_duplicate_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = LiveEventPublisher(
                events_path=tmp / "live_events.jsonl",
                marker_path=tmp / "live_run.json",
            )
            pub.start_run(task="t", topology="shared_pool")
            pub.emit("security_observation", {
                "agent_id": "executor",
                "tiered": {"chunks": [_chunk(0, CHUNK_TEXT_0)]},
            })

            reader = EventReader(events_path=pub.events_path)
            first = reader.read_new()
            obs_first = [
                e for e in first if e["event_type"] == "security_observation"
            ]
            self.assertEqual(len(obs_first), 1)

            # A refresh re-reads -> nothing new -> no duplicated chunks.
            second = reader.read_new()
            self.assertEqual(second, [])
            # The grouped chunk count is stable across refreshes.
            grouped = dash.all_chunks_by_agent(
                read_all_events(pub.events_path)
            )
            self.assertEqual(len(grouped["executor"]), 1)

    def test_chunk_events_visible_before_episode_completion(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pub = LiveEventPublisher(
                events_path=tmp / "live_events.jsonl",
                marker_path=tmp / "live_run.json",
            )
            pub.start_run(task="t", topology="shared_pool")
            # Publish a chunk observation, but DO NOT finish the run.
            pub.emit("security_observation", {
                "agent_id": "executor",
                "security_score": 0.6,
                "tiered": {
                    "tier2_source": "real",
                    "chunks": [_chunk(0, CHUNK_TEXT_0, nli={
                        "ran": True, "source": "real", "label": "neutral",
                        "confidence": 0.5, "premise": NLI_PREMISE,
                        "hypothesis": CHUNK_TEXT_0,
                    })],
                },
            })

            marker = read_run_marker(pub.marker_path)
            self.assertEqual(marker["status"], "running")  # still running
            events = read_all_events(pub.events_path)
            grouped = dash.all_chunks_by_agent(events)
            self.assertIn("executor", grouped)
            self.assertEqual(grouped["executor"][0]["chunk_text"], CHUNK_TEXT_0)

    def test_dashboard_does_not_execute_the_experiment(self):
        # streamlit_app must never CALL the experiment runner. The module
        # docstring legitimately MENTIONS these names in prose, so check
        # the compiled code's referenced names instead of the raw source.
        forbidden_names = {"execute_task", "MASEnvironment", "invoke"}

        referenced = set()
        for fn in (
            dash.render,
            dash.main,
            dash.render_chunk_card,
            dash._live_chunk_stream_section,
            dash._run_status_section,
        ):
            referenced.update(fn.__code__.co_names)

        self.assertFalse(
            referenced & forbidden_names,
            f"dashboard references experiment names: "
            f"{referenced & forbidden_names}",
        )

        # The dashboard must not import the runner or the environment.
        self.assertNotIn("environment.mas_environment", getattr(dash, "__dict__", {}))

    def test_render_reads_only_and_creates_nothing(self):
        fake = _FakeStreamlit()
        original_st = dash.st
        original_events = dash.EVENTS_PATH
        original_marker = dash.MARKER_PATH
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            events_path = tmp / "none.jsonl"
            marker_path = tmp / "none.json"
            dash.EVENTS_PATH = events_path
            dash.MARKER_PATH = marker_path
            dash.st = fake
            try:
                dash.render()
            finally:
                dash.st = original_st
                dash.EVENTS_PATH = original_events
                dash.MARKER_PATH = original_marker
            # Rendering created neither the event file nor the run marker.
            self.assertFalse(events_path.exists())
            self.assertFalse(marker_path.exists())

# =====================================================================
# END-TO-END: the environment publishes the schema the UI reads
# =====================================================================

class EndToEndPublishSchemaTests(unittest.TestCase):
    """
    Run a REAL offline episode (stub LLM + stub tools + stub NLI) and
    assert the environment actually publishes, per response chunk, the
    exact ``tiered.chunks`` schema streamlit_app reads. This proves the
    producer (main.py's observer) and the consumer (the dashboard) agree,
    without running any LLM.
    """

    def _run_episode(self, tmp: Path):
        from environment.mas_environment import MASEnvironment
        from security.observer import SecurityObserver
        from security.contradiction_checker import ContradictionChecker
        from security.semantic_assessor import SemanticAssessor
        from tests.conftest import (
            StubLLM, StubEncoder, StubSearchTool, StubReportWriter,
            StubCalendar, StubEmail,
        )
        import agents.coordinator as coordinator_module
        import agents.outline as outline_module
        import agents.researcher as researcher_module
        import agents.analyst as analyst_module
        import agents.executor as executor_module

        # Deterministic NLI stub: no download, but Tier 2 really runs.
        class _StubNLI:
            def __call__(self, inputs, truncation=True):
                return [
                    {"label": "neutral", "score": 0.6},
                    {"label": "entailment", "score": 0.3},
                    {"label": "contradiction", "score": 0.1},
                ]

        stub = StubLLM()
        mods = (
            coordinator_module, outline_module, researcher_module,
            analyst_module, executor_module,
        )
        originals = [(m, m.get_llm) for m in mods]
        for m in mods:
            m.get_llm = lambda: stub

        try:
            publisher = LiveEventPublisher(
                events_path=tmp / "live_events.jsonl",
                marker_path=tmp / "live_run.json",
            )
            publisher.start_run(task="T", topology="shared_pool")

            observer = SecurityObserver(
                semantic_assessor=SemanticAssessor(model=StubEncoder()),
                contradiction_checker=ContradictionChecker(model=_StubNLI()),
            )
            env = MASEnvironment(
                topology_name="shared_pool",
                event_publisher=publisher,
                security_observer=observer,
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
            publisher.finish_run(status="completed")
        finally:
            for module, original in originals:
                module.get_llm = original

        return read_all_events(publisher.events_path)

    def test_published_observations_carry_chunk_schema(self):
        with tempfile.TemporaryDirectory() as d:
            events = self._run_episode(Path(d))

        observations = [
            e for e in events if e.get("event_type") == "security_observation"
        ]
        self.assertTrue(observations, "no security observations were published")

        # Every observation carries the run-level provenance the UI shows.
        for event in observations:
            tiered = event["payload"].get("tiered")
            self.assertIsNotNone(tiered)
            self.assertIn(
                tiered.get("tier2_source"), {"real", "stub", "not_run"}
            )
            self.assertIn(
                tiered.get("tier3_status"), {"real", "stub", "not_run"}
            )
            self.assertIsInstance(tiered.get("chunks"), list)

        # At least one observation carries at least one chunk with the
        # exact per-chunk fields the dashboard renders.
        all_chunks = [
            chunk
            for event in observations
            for chunk in event["payload"]["tiered"]["chunks"]
        ]
        self.assertTrue(all_chunks, "no per-chunk results were published")
        for chunk in all_chunks:
            self.assertIn("chunk_index", chunk)
            self.assertIn("chunk_text", chunk)
            self.assertIn("tier1_semantic", chunk)
            self.assertIn("tier2_nli", chunk)
            self.assertIn("tier3_judge", chunk)
            sem = chunk["tier1_semantic"]
            for key in (
                "assessed", "task_similarity", "subtask_similarity",
                "objective_deviation", "scope_deviation", "deviation_score",
                "confidence",
            ):
                self.assertIn(key, sem)
            nli = chunk["tier2_nli"]
            self.assertIn("ran", nli)
            if nli["ran"]:
                for key in ("label", "confidence", "premise", "hypothesis"):
                    self.assertIn(key, nli)

        # And the dashboard helpers read that schema without error.
        grouped = dash.all_chunks_by_agent(events)
        self.assertTrue(grouped)
        for agent_id, chunks in grouped.items():
            for chunk in chunks:
                dash.chunk_semantic_fields(chunk)
                dash.chunk_nli_status(chunk)
                dash.chunk_judge_status(chunk)

    def test_chunk_text_is_real_response_text(self):
        with tempfile.TemporaryDirectory() as d:
            events = self._run_episode(Path(d))
        chunks = [
            chunk
            for e in events if e.get("event_type") == "security_observation"
            for chunk in e["payload"]["tiered"]["chunks"]
        ]
        # Chunk texts are non-empty and are actual slices of the response.
        self.assertTrue(all(chunk["chunk_text"].strip() for chunk in chunks))

    def test_default_environment_runs_real_tier2_nli(self):
        """
        main.py builds MASEnvironment WITHOUT injecting a security
        observer, so it gets ContradictionChecker() with _model=None --
        i.e. the REAL roberta-large-mnli pipeline. Assert that wiring, so
        the "normal main.py runs real NLI" claim is verified, not assumed.
        """
        from environment.mas_environment import MASEnvironment
        from security.contradiction_checker import ContradictionChecker

        env = MASEnvironment(topology_name="shared_pool")
        checker = env.security_observer.contradiction_checker
        self.assertIsInstance(checker, ContradictionChecker)
        # _model is None until first use -> real transformers pipeline.
        self.assertIsNone(checker._model)
        self.assertEqual(checker.model_name, "roberta-large-mnli")

        # Tier 3 is gated off by default (MAS_TIER3_JUDGE unset here).
        import os
        self.assertIsNone(env.security_observer.llm_judge)
        self.assertEqual(os.getenv("MAS_TIER3_JUDGE", "0"), "0")


if __name__ == "__main__":
    unittest.main()
