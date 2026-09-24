"""
Tests for the remediation executor (Task S, containment layer).

Pure policy: no environment, no model, no LLM.
"""

from __future__ import annotations

import unittest

from security.remediation import (
    MAX_REMEDIATION_ATTEMPTS,
    RemediationExecutor,
    RemediationPlan,
)


PIPELINE = ["coordinator", "outline", "researcher", "executor", "coordinator_final"]


class RemediationExecutorTests(unittest.TestCase):

    def _executor(self):
        # researcher consumes req-tool-1; the pipeline feeds forward.
        consumer_index = {
            "req-tool-1": ["researcher"],
        }
        consumer_graph = {
            "researcher": ["executor"],
            "executor": ["coordinator_final"],
        }
        return RemediationExecutor(
            consumer_index=consumer_index,
            consumer_graph=consumer_graph,
            pipeline_order=PIPELINE,
        )

    def test_single_agent_remediation(self):
        """An artifact consumed only by a leaf agent -> only that agent."""

        executor = RemediationExecutor(
            consumer_index={"req-x": ["executor"]},
            consumer_graph={},
            pipeline_order=PIPELINE,
        )
        plan, updated = executor.plan(
            root_artifact_id="req-x", attempts={}
        )

        self.assertIsInstance(plan, RemediationPlan)
        self.assertFalse(plan.over_attempt_cap)
        self.assertEqual(plan.agents_to_rerun, ["executor"])
        self.assertEqual(plan.discarded_artifact_id, "req-x")
        self.assertEqual(plan.attempt, 1)
        self.assertEqual(updated["req-x"], 1)

    def test_cascading_two_agent_remediation(self):
        """
        researcher consumed the artifact; executor is downstream of
        researcher -> BOTH re-run, in dependency order (researcher then
        executor).
        """

        executor = self._executor()
        plan, updated = executor.plan(
            root_artifact_id="req-tool-1", attempts={}
        )

        self.assertEqual(
            plan.agents_to_rerun,
            ["researcher", "executor", "coordinator_final"],
        )
        self.assertEqual(updated["req-tool-1"], 1)

    def test_cascade_includes_transitive_downstream(self):
        """The cascade is transitive, not just one hop."""

        executor = self._executor()
        plan, _ = executor.plan(root_artifact_id="req-tool-1", attempts={})

        # researcher -> executor -> coordinator_final, ordered by the
        # canonical pipeline order.
        self.assertEqual(
            plan.agents_to_rerun,
            ["researcher", "executor", "coordinator_final"],
        )

    def test_dependency_order_follows_pipeline(self):
        """Ordering is the pipeline order, not insertion/alphabetical."""

        executor = RemediationExecutor(
            consumer_index={"req-x": ["executor", "researcher"]},
            consumer_graph={},
            pipeline_order=PIPELINE,
        )
        plan, _ = executor.plan(root_artifact_id="req-x", attempts={})

        # researcher appears before executor in the pipeline.
        self.assertEqual(plan.agents_to_rerun, ["researcher", "executor"])

    def test_attempt_cap_is_hit_and_reported(self):
        """
        After MAX_REMEDIATION_ATTEMPTS attempts for the same artifact,
        the plan is over-cap with NO agents and the chunk is unresolved.
        """

        executor = self._executor()
        attempts = {}

        # First two attempts are allowed.
        for expected in range(1, MAX_REMEDIATION_ATTEMPTS + 1):
            plan, attempts = executor.plan(
                root_artifact_id="req-tool-1", attempts=attempts
            )
            self.assertFalse(plan.over_attempt_cap)
            self.assertTrue(plan.agents_to_rerun)
            self.assertEqual(plan.attempt, expected)

        # The next attempt is over the cap.
        plan, attempts = executor.plan(
            root_artifact_id="req-tool-1", attempts=attempts
        )
        self.assertTrue(plan.over_attempt_cap)
        self.assertEqual(plan.agents_to_rerun, [])
        # The counter is not advanced past the cap.
        self.assertEqual(
            attempts["req-tool-1"], MAX_REMEDIATION_ATTEMPTS
        )

    def test_counter_is_not_mutated_in_place(self):
        """plan() must RETURN a new counter, never mutate the input dict."""

        executor = self._executor()
        original = {"req-tool-1": 0}
        _, updated = executor.plan(
            root_artifact_id="req-tool-1", attempts=original
        )
        self.assertEqual(original, {"req-tool-1": 0})
        self.assertEqual(updated["req-tool-1"], 1)

    def test_unknown_artifact_yields_empty_plan(self):
        """An artifact nobody consumed -> nothing to re-run."""

        executor = self._executor()
        plan, updated = executor.plan(
            root_artifact_id="req-unknown", attempts={}
        )
        self.assertEqual(plan.agents_to_rerun, [])
        self.assertFalse(plan.over_attempt_cap)
        self.assertEqual(updated["req-unknown"], 1)

    def test_branching_shared_consumer_does_not_loop(self):
        """A cyclic consumer graph must terminate (visited-set)."""

        executor = RemediationExecutor(
            consumer_index={"req-x": ["researcher"]},
            consumer_graph={
                "researcher": ["executor"],
                "executor": ["researcher"],  # cycle back
            },
            pipeline_order=PIPELINE,
        )
        plan, _ = executor.plan(root_artifact_id="req-x", attempts={})
        self.assertEqual(plan.agents_to_rerun, ["researcher", "executor"])

    def test_max_attempts_default_is_named_constant(self):
        executor = self._executor()
        self.assertEqual(executor.max_attempts, MAX_REMEDIATION_ATTEMPTS)
        self.assertEqual(MAX_REMEDIATION_ATTEMPTS, 2)


if __name__ == "__main__":
    unittest.main()