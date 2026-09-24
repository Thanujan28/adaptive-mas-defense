"""
Tests for the root-cause tracer (Task S, containment layer).

No model download: the contradiction checker is injected with a
deterministic stub NLI model, same pattern as
tests/test_contradiction_checker.py.
"""

from __future__ import annotations

import unittest

from security.contradiction_checker import ContradictionChecker
from security.root_cause_tracer import (
    ArtifactView,
    RootCauseTracer,
    REASON_HOP_LIMIT,
    REASON_NO_UPSTREAM,
    REASON_ROOT_FOUND,
    REASON_TRUSTED_ROOT,
    REASON_VISITED_CYCLE,
    TRUSTED_ROOT_LABEL,
    MAX_TRACE_HOPS,
)


class _StubNLI:
    """
    Deterministic NLI stub.

    A pair contradicts iff the HYPOTHESIS contains a marker word that
    the PREMISE does not (a simple, offline rule) -- enough to drive the
    tracer's one-hop contradiction check deterministically.
    """

    def __call__(self, inputs, truncation=True):
        premise = str(inputs.get("text", "")).lower()
        hypothesis = str(inputs.get("text_pair", "")).lower()
        contradicts = ("corrupt" in hypothesis) and (
            "corrupt" not in premise
        )
        if contradicts:
            return [
                {"label": "contradiction", "score": 0.95},
                {"label": "neutral", "score": 0.03},
                {"label": "entailment", "score": 0.02},
            ]
        return [
            {"label": "neutral", "score": 0.85},
            {"label": "entailment", "score": 0.10},
            {"label": "contradiction", "score": 0.05},
        ]


def _checker() -> ContradictionChecker:
    return ContradictionChecker(model=_StubNLI())


class _Chain:
    """
    A simple in-memory evidence chain for the tracer to walk.

    ``views`` maps artifact_id -> ArtifactView.
    """

    def __init__(self, views):
        self.views = views
        self.lookups: list[str] = []

    def lookup(self, artifact_id):
        self.lookups.append(artifact_id)
        return self.views.get(artifact_id)


class RootCauseTracerTests(unittest.TestCase):

    def test_root_found_within_limit(self):
        """
        detect -> mid (corrupt) -> raw tool result (clean = root).
        The clean raw tool result is the root cause.
        """

        chain = _Chain(
            {
                "req:detect": ArtifactView(
                    artifact_id="req:detect",
                    text="the summary corrupts the numbers",
                    upstream_ids=["req:mid"],
                    upstream_evidence=["mid says the numbers are fine"],
                ),
                "req:mid": ArtifactView(
                    artifact_id="req:mid",
                    text="the intermediate corrupts the raw result",
                    upstream_ids=["req:raw"],
                    upstream_evidence=["raw result: untainted data"],
                ),
                "req:raw": ArtifactView(
                    artifact_id="req:raw",
                    text="raw result: untainted data",
                    upstream_ids=["trusted_root"],
                    upstream_evidence=["the original task"],
                ),
            }
        )
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("req:detect")

        self.assertTrue(result.root_found)
        self.assertEqual(result.root_artifact_id, "req:raw")
        self.assertEqual(result.stopped_reason, REASON_ROOT_FOUND)
        # detect(hop0) -> mid(hop1) -> raw(hop2).
        self.assertEqual(result.hops_taken, 2)

    def test_hop_limit_exceeded(self):
        """
        Every hop is corrupt and there is always another upstream -> the
        walk must stop at MAX_TRACE_HOPS without finding a root.
        """

        views = {}
        # A long chain of corrupt artifacts, each contradicting its
        # upstream evidence, ending nowhere clean.
        for i in range(MAX_TRACE_HOPS + 3):
            views[f"req:{i}"] = ArtifactView(
                artifact_id=f"req:{i}",
                text="this artifact corrupts its source",
                upstream_ids=[f"req:{i + 1}"],
                upstream_evidence=["upstream is clean here"],
            )
        chain = _Chain(views)
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("req:0")

        self.assertFalse(result.root_found)
        self.assertIsNone(result.root_artifact_id)
        self.assertEqual(result.stopped_reason, REASON_HOP_LIMIT)
        self.assertEqual(result.hops_taken, MAX_TRACE_HOPS)

    def test_trusted_root_hit_first(self):
        """
        The very first hop is the trusted root -> stop, never trace into
        the task/plan.
        """

        chain = _Chain(
            {
                "req:detect": ArtifactView(
                    artifact_id="req:detect",
                    text="corrupt claim",
                    upstream_ids=["trusted_root"],
                    upstream_evidence=["the plan"],
                ),
                "trusted_root": ArtifactView(
                    artifact_id=TRUSTED_ROOT_LABEL,
                    text="the coordinator plan",
                    is_trusted_root=True,
                ),
            }
        )
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("trusted_root")

        self.assertFalse(result.root_found)
        self.assertEqual(result.stopped_reason, REASON_TRUSTED_ROOT)

    def test_trusted_root_reached_after_hops(self):
        """
        A corrupt chain that leads to the trusted root must stop there
        (report trusted-root), NOT treat the plan as the root cause.
        """

        chain = _Chain(
            {
                "req:detect": ArtifactView(
                    artifact_id="req:detect",
                    text="corrupt at detection",
                    upstream_ids=["req:mid"],
                    upstream_evidence=["mid is clean"],
                ),
                "req:mid": ArtifactView(
                    artifact_id="req:mid",
                    text="corrupt at mid",
                    upstream_ids=["agent:coordinator:plan"],
                    upstream_evidence=["plan is clean"],
                ),
                "agent:coordinator:plan": ArtifactView(
                    artifact_id="agent:coordinator:plan",
                    text="the coordinator plan",
                    is_trusted_root=True,
                ),
            }
        )
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("req:detect")

        self.assertFalse(result.root_found)
        self.assertEqual(result.stopped_reason, REASON_TRUSTED_ROOT)
        # detect -> mid -> plan(trusted).
        self.assertEqual(result.hops_taken, 2)

    def test_branching_shared_artifact_does_not_loop(self):
        """
        Two paths converge on the SAME artifact; the visited set must
        stop the walk instead of looping forever.
        """

        chain = _Chain(
            {
                "req:a": ArtifactView(
                    artifact_id="req:a",
                    text="corrupt a",
                    upstream_ids=["req:shared"],
                    upstream_evidence=["clean upstream"],
                ),
                # req:shared is corrupt and points back to req:a, forming
                # a cycle a -> shared -> a -> ...
                "req:shared": ArtifactView(
                    artifact_id="req:shared",
                    text="corrupt shared",
                    upstream_ids=["req:a"],
                    upstream_evidence=["clean upstream"],
                ),
            }
        )
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("req:a")

        self.assertFalse(result.root_found)
        self.assertEqual(result.stopped_reason, REASON_VISITED_CYCLE)

    def test_shared_artifact_converges_from_two_consumers(self):
        """
        A shared artifact reached from two different starting points must
        each terminate (and converge on the same clean root).
        """

        # shared is clean -> it is the root for both starts.
        root_views = {
            "req:x": ArtifactView(
                artifact_id="req:x",
                text="corrupt x",
                upstream_ids=["req:shared"],
                upstream_evidence=["clean"],
            ),
            "req:y": ArtifactView(
                artifact_id="req:y",
                text="corrupt y",
                upstream_ids=["req:shared"],
                upstream_evidence=["clean"],
            ),
            "req:shared": ArtifactView(
                artifact_id="req:shared",
                text="clean shared artifact",
                upstream_ids=["trusted_root"],
                upstream_evidence=["clean"],
            ),
        }
        chain = _Chain(root_views)
        tracer = RootCauseTracer(_checker(), chain.lookup)

        r1 = tracer.trace("req:x")
        r2 = tracer.trace("req:y")

        self.assertTrue(r1.root_found)
        self.assertTrue(r2.root_found)
        self.assertEqual(r1.root_artifact_id, "req:shared")
        self.assertEqual(r2.root_artifact_id, "req:shared")

    def test_no_upstream_evidence_is_clean_root(self):
        """
        An artifact with no upstream evidence cannot be shown to
        contradict anything, so it is treated as a CLEAN root (the walk
        ends here) rather than chased further.
        """

        chain = _Chain(
            {
                "req:detect": ArtifactView(
                    artifact_id="req:detect",
                    text="corrupt",
                    upstream_ids=[],
                    upstream_evidence=[],
                ),
            }
        )
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("req:detect")

        self.assertTrue(result.root_found)
        self.assertEqual(result.root_artifact_id, "req:detect")
        self.assertEqual(result.stopped_reason, REASON_ROOT_FOUND)

    def test_corrupt_artifact_without_upstream_stops(self):
        """
        A CORRUPT artifact that has no upstream id to follow stops with
        'no upstream evidence' (it is itself corrupt but there is
        nothing behind it to blame).
        """

        chain = _Chain(
            {
                "req:detect": ArtifactView(
                    artifact_id="req:detect",
                    text="this artifact corrupts its source",
                    upstream_ids=[],
                    # Contradictory against its OWN evidence, but there
                    # is no upstream id to walk to.
                    upstream_evidence=["upstream is clean here"],
                ),
            }
        )
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("req:detect")

        self.assertFalse(result.root_found)
        self.assertEqual(result.stopped_reason, REASON_NO_UPSTREAM)

    def test_unknown_start_id(self):
        chain = _Chain({})
        tracer = RootCauseTracer(_checker(), chain.lookup)

        result = tracer.trace("req:missing")

        self.assertFalse(result.root_found)
        self.assertEqual(result.stopped_reason, REASON_NO_UPSTREAM)

    def test_max_hops_default_is_named_constant(self):
        tracer = RootCauseTracer(_checker(), _Chain({}).lookup)
        self.assertEqual(tracer.max_hops, MAX_TRACE_HOPS)
        self.assertEqual(MAX_TRACE_HOPS, 3)


if __name__ == "__main__":
    unittest.main()