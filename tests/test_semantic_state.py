"""
Tests for the semantic-assessment wiring in the security state estimator.

These tests never touch the network and never download model weights.
A deterministic stub encoder is injected through
``SemanticAssessor(model=...)``.
"""

import unittest

import numpy as np

from security.semantic_assessor import (
    SemanticAssessor,
    SemanticAssessment,
)
from security.state_builder import SecurityStateBuilder


# =============================================================
# DETERMINISTIC STUB ENCODER
# =============================================================

class _StubEncoder:
    """
    Deterministic bag-of-keywords encoder.

    Each text is mapped to a small fixed-dimensional vector built
    from a keyword lexicon, so semantically related texts land close
    together and unrelated texts land far apart -- without any model
    download. The exact geometry is irrelevant; only the ordering of
    (on-topic > unrelated) similarity matters.
    """

    KEYWORDS = (
        "security",
        "prompt",
        "injection",
        "agent",
        "risk",
        "cookie",
        "recipe",
        "butter",
    )

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ):
        vectors = []

        for text in texts:
            lowered = text.lower()
            vector = [
                float(lowered.count(keyword))
                for keyword in self.KEYWORDS
            ]
            vectors.append(vector)

        array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        # Avoid all-zero vectors for texts with none of the keywords
        # by adding a tiny constant so cosine similarity is defined.
        if array.size and not array.any(axis=1).all():
            mask = ~array.any(axis=1)
            array[mask, 0] = 1e-6

        if normalize_embeddings:
            norms = np.linalg.norm(
                array,
                axis=1,
                keepdims=True,
            )
            norms[norms == 0.0] = 1.0
            array = array / norms

        return array


ON_TOPIC = (
    "Prompt injection is a security risk: an attacker can hijack "
    "an agent's instructions and spread malicious directives to "
    "peer agents."
)

OFF_TOPIC = (
    "Chocolate chip cookies: mix butter and sugar, then bake. "
    "This recipe has nothing to do with security."
)

TASK = (
    "Analyze the security risk of prompt injection between agents."
)


def _stub_assessor() -> SemanticAssessor:
    return SemanticAssessor(model=_StubEncoder())


# =============================================================
# 1. BUILD CALL PATTERN
# =============================================================

class BuildCallPatternTests(unittest.TestCase):

    def test_build_accepts_keyword_call_pattern(self):
        """
        The exact call pattern used by both production callers must
        not raise.
        """

        builder = SecurityStateBuilder()

        state = builder.build(
            events=[],
            semantic=SemanticAssessment(),
            resource_state={"tokens_used": 1, "tools_used": 2},
            memory_counts={"coordinator": 0},
        )

        self.assertIn("semantic_deviation", state)
        self.assertEqual(
            len(state),
            SecurityStateBuilder.feature_count(),
        )

    def test_semantic_is_keyword_only(self):
        """
        Passing semantic positionally must raise TypeError, so the
        positional-argument bug class cannot recur.
        """

        builder = SecurityStateBuilder()

        with self.assertRaises(TypeError):
            builder.build(
                [],
                SemanticAssessment(),
            )

    def test_default_semantic_when_none(self):
        """
        Without a semantic assessment, features fall back to the
        neutral defaults: deviation 0.5, confidence 0.0.
        """

        state = SecurityStateBuilder().build(events=[])

        self.assertEqual(state["semantic_deviation"], 0.5)
        self.assertEqual(state["semantic_confidence"], 0.0)


# =============================================================
# 2. SEMANTIC DISCRIMINATION
# =============================================================

class SemanticDiscriminationTests(unittest.TestCase):

    def test_off_topic_has_higher_deviation_than_on_topic(self):
        assessor = _stub_assessor()

        on_topic = assessor.assess(
            original_task=TASK,
            assigned_subtask="research",
            agent_output=ON_TOPIC,
        )

        off_topic = assessor.assess(
            original_task=TASK,
            assigned_subtask="research",
            agent_output=OFF_TOPIC,
        )

        self.assertTrue(on_topic.assessed)
        self.assertTrue(off_topic.assessed)

        self.assertGreater(
            off_topic.deviation_score,
            on_topic.deviation_score,
        )

        # Both must differ from the neutral 0.5 default.
        self.assertNotEqual(on_topic.deviation_score, 0.5)
        self.assertNotEqual(off_topic.deviation_score, 0.5)

    def test_deviation_flows_into_state_builder(self):
        assessor = _stub_assessor()
        builder = SecurityStateBuilder()

        on_topic = assessor.assess(
            original_task=TASK,
            assigned_subtask="research",
            agent_output=ON_TOPIC,
        )
        off_topic = assessor.assess(
            original_task=TASK,
            assigned_subtask="research",
            agent_output=OFF_TOPIC,
        )

        on_state = builder.build(
            events=[],
            semantic=on_topic,
        )
        off_state = builder.build(
            events=[],
            semantic=off_topic,
        )

        self.assertGreater(
            off_state["semantic_deviation"],
            on_state["semantic_deviation"],
        )
        self.assertGreater(
            on_state["semantic_confidence"],
            0.0,
        )


# =============================================================
# 3. ENVIRONMENT WIRING
# =============================================================

class EnvironmentWiringTests(unittest.TestCase):

    def _build_environment(self, **kwargs):
        from environment.mas_environment import MASEnvironment

        return MASEnvironment(
            topology_name="centralized",
            **kwargs,
        )

    def _observer_with_stub(self):
        from security.observer import SecurityObserver

        return SecurityObserver(
            semantic_assessor=_stub_assessor(),
            log_enabled=False,
        )

    def test_malicious_response_reaches_security_state(self):
        """
        After a malicious response is forwarded, the semantic
        confidence in the security state must be > 0, proving the
        assessor actually ran through the observer hook.
        """

        environment = self._build_environment(
            security_observer=self._observer_with_stub(),
        )

        # Simulate an episode task being set.
        environment.episode_state.task = TASK
        environment.security_observations = []

        # Forward a malicious (off-topic) agent response through the
        # observation hook.
        environment.publish_agent_result(
            sender="researcher",
            receiver="executor",
            content=OFF_TOPIC,
            metadata={"stage": "research"},
        )

        self.assertEqual(
            len(environment.security_observations),
            1,
        )

        state = environment.get_security_state()

        self.assertGreater(
            state["semantic_confidence"],
            0.0,
        )
        self.assertNotEqual(
            state["semantic_deviation"],
            0.5,
        )

    def test_aggregation_uses_worst_deviation(self):
        environment = self._build_environment(
            security_observer=self._observer_with_stub(),
        )

        environment.episode_state.task = TASK
        environment.security_observations = []

        environment.publish_agent_result(
            sender="outline",
            receiver="researcher",
            content=ON_TOPIC,
            metadata={"stage": "outline"},
        )
        environment.publish_agent_result(
            sender="researcher",
            receiver="executor",
            content=OFF_TOPIC,
            metadata={"stage": "research"},
        )

        aggregate = environment.get_semantic_assessment()

        # The aggregate must equal the worst (off-topic) assessment.
        worst = max(
            (
                observation.semantic_assessment
                for observation in environment.security_observations
                if observation.semantic_assessment is not None
            ),
            key=lambda assessment: assessment.deviation_score,
        )

        self.assertAlmostEqual(
            aggregate.deviation_score,
            worst.deviation_score,
        )
        self.assertGreater(
            aggregate.deviation_score,
            0.0,
        )

    def test_reset_clears_observations(self):
        environment = self._build_environment(
            security_observer=self._observer_with_stub(),
        )

        environment.episode_state.task = TASK
        environment.security_observations = []

        environment.publish_agent_result(
            sender="researcher",
            receiver="executor",
            content=OFF_TOPIC,
            metadata={"stage": "research"},
        )

        self.assertEqual(
            len(environment.security_observations),
            1,
        )

        # execute_task resets per-episode observations. We avoid
        # running the graph (no LLM available) by exercising the
        # same reset line directly.
        environment.security_observations = []
        environment.episode_state.task = TASK

        aggregate = environment.get_semantic_assessment()

        self.assertFalse(aggregate.assessed)
        self.assertEqual(aggregate.deviation_score, 0.5)

    def test_no_observations_returns_unassessed(self):
        environment = self._build_environment(
            security_observer=self._observer_with_stub(),
        )

        environment.security_observations = []

        aggregate = environment.get_semantic_assessment()

        self.assertFalse(aggregate.assessed)


# =============================================================
# 4. SEMANTIC DISABLED
# =============================================================

class SemanticDisabledTests(unittest.TestCase):

    def test_disabled_leaves_defaults_without_raising(self):
        from environment.mas_environment import MASEnvironment
        from security.observer import SecurityObserver

        observer = SecurityObserver(
            semantic_assessor=_stub_assessor(),
            semantic_enabled=False,
            log_enabled=False,
        )

        environment = MASEnvironment(
            topology_name="centralized",
            security_observer=observer,
            semantic_enabled=False,
        )

        environment.episode_state.task = TASK
        environment.security_observations = []

        # Must not raise even with a response present.
        environment.publish_agent_result(
            sender="researcher",
            receiver="executor",
            content=OFF_TOPIC,
            metadata={"stage": "research"},
        )

        state = environment.get_security_state()

        self.assertEqual(state["semantic_deviation"], 0.5)
        self.assertEqual(state["semantic_confidence"], 0.0)

        aggregate = environment.get_semantic_assessment()
        self.assertFalse(aggregate.assessed)


if __name__ == "__main__":
    unittest.main()
