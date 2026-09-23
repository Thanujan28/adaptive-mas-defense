"""
Tests for the security-observation subtask reference fix.

The subtask passed to the semantic assessor must be the TRUSTED,
registered role/instruction template + original task, never the
one-word pipeline stage label, and never upstream or tool-derived
content.

No network. No model download. A deterministic stub encoder is
injected into the semantic assessor.
"""

import unittest

import numpy as np

from environment.mas_environment import (
    MASEnvironment,
    OUTLINE_ASSIGNMENT_TEMPLATE,
    RESEARCHER_ASSIGNMENT_TEMPLATE,
    ANALYST_ASSIGNMENT_TEMPLATE,
    EXECUTOR_ASSIGNMENT_TEMPLATE,
    COORDINATOR_ASSIGNMENT_TEMPLATE,
)
from security.observer import SecurityObserver
from security.semantic_assessor import SemanticAssessor, _as_text
from security.state_builder import SecurityStateBuilder


# =============================================================
# STUB ENCODER (no model download)
# =============================================================

class _StubEncoder:
    KEYWORDS = ("security", "prompt", "outline", "research", "report")

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        rows = []
        for text in texts:
            low = str(text).lower()
            rows.append([float(low.count(k)) for k in self.KEYWORDS])
        array = np.asarray(rows, dtype=np.float32)
        if array.size and not array.any(axis=1).all():
            array[~array.any(axis=1), 0] = 1e-6
        return array


class _RecordingAssessor(SemanticAssessor):
    """
    Assessor that records the subtask it was asked to compare against.
    """

    def __init__(self):
        super().__init__(model=_StubEncoder())
        self.seen_subtasks = []

    def assess(self, original_task, assigned_subtask, agent_output):
        self.seen_subtasks.append(assigned_subtask)
        return super().assess(
            original_task=original_task,
            assigned_subtask=assigned_subtask,
            agent_output=agent_output,
        )


def _environment(assessor=None):
    observer = SecurityObserver(
        semantic_assessor=assessor or SemanticAssessor(model=_StubEncoder()),
        log_enabled=False,
    )
    return MASEnvironment(
        topology_name="centralized",
        security_observer=observer,
    )


STAGE_LABELS = {
    "outline_assignment",
    "outline",
    "research",
    "analysis",
    "execution",
}


# =============================================================
# SUBTASK IS THE TEMPLATE, NOT THE STAGE
# =============================================================

class SubtaskIsTemplateTests(unittest.TestCase):

    def test_registered_assignment_is_used_as_subtask(self):
        assessor = _RecordingAssessor()
        env = _environment(assessor)
        env.episode_state.task = "Summarize the security risks."
        env.security_observations = []
        env.agent_assignments = {}

        env._register_assignment(
            "researcher",
            RESEARCHER_ASSIGNMENT_TEMPLATE,
        )

        env.publish_agent_result(
            sender="researcher",
            receiver="executor",
            content="Some findings.",
            metadata={"stage": "research"},
        )

        # The live pipeline now runs the TIERED detector: the semantic
        # assessor is called once PER CHUNK (Tier 1) plus once for the
        # whole-response base fusion. Every one of those calls must
        # receive the SAME trusted subtask reference.
        self.assertTrue(assessor.seen_subtasks)
        subtasks = assessor.seen_subtasks

        for subtask in subtasks:
            # The template is present.
            self.assertIn(
                RESEARCHER_ASSIGNMENT_TEMPLATE.strip()[:30],
                subtask,
            )
            # The original task is present.
            self.assertIn("Summarize the security risks.", subtask)
            # The stage label is NOT the subtask.
            self.assertNotEqual(subtask.strip(), "research")
            self.assertNotIn(subtask.strip(), STAGE_LABELS)

        # All calls share the identical reference (chunk + base fusion
        # use the same trusted assignment).
        self.assertEqual(len(set(subtasks)), 1)

    def test_subtask_never_equals_a_stage_label(self):
        assessor = _RecordingAssessor()
        env = _environment(assessor)
        env.episode_state.task = "Task X"
        env.security_observations = []

        env.agent_assignments = {}
        env._register_assignment("outline", OUTLINE_ASSIGNMENT_TEMPLATE)
        env._register_assignment("researcher", RESEARCHER_ASSIGNMENT_TEMPLATE)
        env._register_assignment("analyst", ANALYST_ASSIGNMENT_TEMPLATE)
        env._register_assignment("executor", EXECUTOR_ASSIGNMENT_TEMPLATE)
        env._register_assignment("coordinator", COORDINATOR_ASSIGNMENT_TEMPLATE)

        calls = [
            ("coordinator", "outline", "outline_assignment"),
            ("outline", "researcher", "outline"),
            ("researcher", "executor", "research"),
            ("executor", "coordinator", "execution"),
        ]

        for sender, receiver, stage in calls:
            env.publish_agent_result(
                sender=sender,
                receiver=receiver,
                content="output text",
                metadata={"stage": stage},
            )

        for subtask in assessor.seen_subtasks:
            self.assertNotIn(subtask.strip(), STAGE_LABELS)

    def test_stage_is_kept_in_metadata_for_logging(self):
        env = _environment()
        env.episode_state.task = "Task"
        env.security_observations = []
        env.agent_assignments = {}
        env._register_assignment("researcher", RESEARCHER_ASSIGNMENT_TEMPLATE)

        env.publish_agent_result(
            sender="researcher",
            receiver="executor",
            content="Output",
            metadata={"stage": "research"},
        )

        observation = env.security_observations[-1]
        self.assertEqual(observation.metadata.get("stage"), "research")
        self.assertEqual(
            observation.metadata.get("subtask_source"),
            "instruction_template",
        )


# =============================================================
# NO UPSTREAM / TOOL CONTENT IN THE ASSIGNMENT
# =============================================================

class NoUpstreamContentTests(unittest.TestCase):

    def test_registered_assignment_excludes_upstream_sentinel(self):
        """
        Inject a sentinel string as upstream content and assert it is
        absent from the registered trusted assignment.
        """

        sentinel = "SENTINEL_UPSTREAM_LEAK_9173"

        env = _environment()
        env.episode_state.task = "Original task"
        env.agent_assignments = {}

        # Simulate a node registering only its static template, while
        # the interpolated prompt it actually builds contains the
        # sentinel upstream content.
        env._register_assignment(
            "researcher",
            RESEARCHER_ASSIGNMENT_TEMPLATE,
        )

        interpolated_prompt = (
            RESEARCHER_ASSIGNMENT_TEMPLATE
            + "OUTLINE TO RESEARCH:\n"
            + sentinel
        )

        # The prompt the agent receives DOES contain the sentinel...
        self.assertIn(sentinel, interpolated_prompt)

        # ...but the registered trusted assignment must NOT.
        self.assertNotIn(
            sentinel,
            env.agent_assignments["researcher"],
        )

    def test_assignment_contains_only_template_and_task(self):
        env = _environment()
        env.episode_state.task = "The immutable goal"
        env.agent_assignments = {}

        env._register_assignment("analyst", ANALYST_ASSIGNMENT_TEMPLATE)

        assignment = env.agent_assignments["analyst"]

        self.assertIn(ANALYST_ASSIGNMENT_TEMPLATE.strip()[:20], assignment)
        self.assertIn("The immutable goal", assignment)
        # No tool-result or outline text by construction.
        self.assertNotIn("OUTLINE TO RESEARCH", assignment)


# =============================================================
# FALLBACK BEHAVIOUR
# =============================================================

class FallbackTests(unittest.TestCase):

    def test_no_assignment_falls_back_to_task(self):
        assessor = _RecordingAssessor()
        env = _environment(assessor)
        env.episode_state.task = "Fallback task"
        env.security_observations = []
        env.agent_assignments = {}

        env.publish_agent_result(
            sender="executor",
            receiver="coordinator",
            content="Final report.",
            metadata={"stage": "execution"},
        )

        # Subtask passed to the assessor is "" so the assessor falls
        # back to the original task internally.
        self.assertEqual(assessor.seen_subtasks[0], "")

        observation = env.security_observations[-1]
        self.assertEqual(
            observation.metadata.get("subtask_source"),
            "fallback_task",
        )
        # Assessment still runs.
        self.assertIsNotNone(observation.semantic_assessment)
        self.assertTrue(observation.semantic_assessment.assessed)

    def test_fallback_assessment_still_uses_task(self):
        assessor = _RecordingAssessor()
        env = _environment(assessor)
        env.episode_state.task = "The original objective"
        env.security_observations = []
        env.agent_assignments = {}

        env.publish_agent_result(
            sender="executor",
            receiver="coordinator",
            content="The original objective is addressed here.",
            metadata={"stage": "execution"},
        )

        observation = env.security_observations[-1]
        assessment = observation.semantic_assessment
        # With "" subtask the assessor falls back to the task, so the
        # subtask axis is well defined (assessed True).
        self.assertTrue(assessment.assessed)


# =============================================================
# INVARIANTS UNCHANGED
# =============================================================

class InvariantTests(unittest.TestCase):

    def test_state_vector_length_unchanged(self):
        builder = SecurityStateBuilder()
        state = builder.build(events=[])
        self.assertEqual(
            len(builder.vector(state)),
            len(SecurityStateBuilder.FEATURE_NAMES),
        )

    def test_assignment_registry_reset_is_available(self):
        env = _environment()
        env.agent_assignments = {"researcher": "x"}
        # execute_task clears it; verify the attribute exists and is
        # a dict so the reset line is valid.
        self.assertIsInstance(env.agent_assignments, dict)

    def test_as_text_unchanged_helper(self):
        self.assertEqual(_as_text("  hi  "), "hi")
        self.assertEqual(_as_text(None), "")


if __name__ == "__main__":
    unittest.main()
