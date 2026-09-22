"""
Tests for the tiered detection pipeline in SecurityObserver (Task L).

No model download and no Ollama: a deterministic stub encoder and a
stub NLI model are injected, and the Tier-3 judge runs in stub mode.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

from security.contradiction_checker import ContradictionChecker
from security.llm_judge import LLMJudge
from security.observer import (
    TIER3_CONTRADICTION_THRESHOLD,
    SecurityObserver,
    TieredObservation,
)
from security.semantic_assessor import SemanticAssessor


class _KeywordEncoder:
    KEYWORDS = (
        "security", "cyber", "threat", "detection", "ai", "network",
        "cookie", "recipe", "butter", "sugar", "bake",
    )

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        rows = []
        for text in texts:
            lowered = str(text).lower()
            rows.append([float(lowered.count(k)) for k in self.KEYWORDS])
        array = np.asarray(rows, dtype=np.float32)
        if array.size and not array.any(axis=1).all():
            array[~array.any(axis=1), 0] = 1e-6
        return array


class _SweepNLI:
    """
    Stub NLI model whose contradiction confidence is controlled by a
    marker in the hypothesis, so tests can place a chunk above or below
    the Tier-3 gate deterministically.
    """

    def __call__(self, inputs, truncation=True):
        hypothesis = inputs["text_pair"].lower()
        if "strongcontradict" in hypothesis:
            return [
                {"label": "contradiction", "score": 0.95},
                {"label": "entailment", "score": 0.03},
                {"label": "neutral", "score": 0.02},
            ]
        if "weakcontradict" in hypothesis:
            return [
                {"label": "contradiction", "score": 0.30},
                {"label": "entailment", "score": 0.40},
                {"label": "neutral", "score": 0.30},
            ]
        return [
            {"label": "neutral", "score": 0.8},
            {"label": "entailment", "score": 0.1},
            {"label": "contradiction", "score": 0.1},
        ]


TASK = "Write a proposal on AI in cyber security"
EVIDENCE = ["AI detects threats in real time and improves response."]

ON_TOPIC = (
    "AI improves cyber security threat detection and network defence. "
) * 30


def _observer(logger=None) -> SecurityObserver:
    return SecurityObserver(
        semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
        contradiction_checker=ContradictionChecker(model=_SweepNLI()),
        llm_judge=LLMJudge(stub=True),
        log_enabled=logger is not None,
        log_logger=logger,
    )


class Tier3GatingTests(unittest.TestCase):

    def test_strong_contradiction_triggers_tier3(self):
        obs = _observer()
        result = obs.observe_tiered(
            agent_id="researcher",
            response=(
                ON_TOPIC
                + "\n\n"
                + ("STRONGCONTRADICT AI offers no benefit. " * 12)
            ),
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )
        self.assertIsInstance(result, TieredObservation)
        self.assertGreaterEqual(result.tier3_invocations, 1)
        self.assertGreaterEqual(result.contradiction_flagged_chunks, 1)

        ran_tiers = [
            set(d.tiers_ran) for d in result.chunk_decisions
        ]
        self.assertTrue(any("tier3" in tiers for tiers in ran_tiers))

    def test_weak_contradiction_does_not_trigger_tier3(self):
        """A contradiction below the named threshold must NOT call the judge."""

        obs = _observer()
        result = obs.observe_tiered(
            agent_id="researcher",
            response=(
                ON_TOPIC
                + "\n\n"
                + ("WEAKCONTRADICT AI maybe offers no benefit. " * 12)
            ),
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )
        self.assertEqual(result.tier3_invocations, 0)
        for decision in result.chunk_decisions:
            self.assertNotIn("tier3", decision.tiers_ran)

    def test_no_evidence_means_no_tier2_or_tier3(self):
        obs = _observer()
        result = obs.observe_tiered(
            agent_id="researcher",
            response=ON_TOPIC,
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=[],
            events=[],
        )
        self.assertEqual(result.tier3_invocations, 0)
        for decision in result.chunk_decisions:
            self.assertEqual(decision.tiers_ran, ["tier1"])

    def test_tier1_always_runs(self):
        obs = _observer()
        result = obs.observe_tiered(
            agent_id="researcher",
            response=ON_TOPIC,
            original_task=TASK,
            assigned_subtask=TASK,
            events=[],
        )
        self.assertTrue(result.chunk_decisions)
        for decision in result.chunk_decisions:
            self.assertIn("tier1", decision.tiers_ran)


class CounterTests(unittest.TestCase):

    def test_per_episode_counter_accumulates_and_resets(self):
        obs = _observer()
        response = ON_TOPIC + "\n\n" + ("STRONGCONTRADICT no benefit. " * 12)

        first = obs.observe_tiered(
            agent_id="a",
            response=response,
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )
        second = obs.observe_tiered(
            agent_id="b",
            response=response,
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )

        self.assertGreater(first.tier3_invocations, 0)
        self.assertEqual(
            obs.tier3_invocations,
            first.tier3_invocations + second.tier3_invocations,
        )

        obs.reset_tier_counters()
        self.assertEqual(obs.tier3_invocations, 0)


class LoggingTests(unittest.TestCase):

    def test_tiered_log_line_includes_chunk_summary_fields(self):
        records: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        test_logger = logging.getLogger("test.tiered.observer")
        test_logger.setLevel(logging.DEBUG)
        handler = _Capture()
        test_logger.addHandler(handler)

        obs = _observer(logger=test_logger)
        obs.observe_tiered(
            agent_id="researcher",
            response=ON_TOPIC + "\n\n" + ("STRONGCONTRADICT no benefit. " * 12),
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )

        combined = "\n".join(records)
        self.assertIn("worst_chunk_deviation=", combined)
        self.assertIn("contradiction_flagged_chunks=", combined)
        self.assertIn("tier3_invocations=", combined)
        # The base security_state line is still emitted unchanged.
        self.assertIn("security_state agent=researcher", combined)

    def test_existing_observe_is_unchanged(self):
        """observe() must still work and NOT require the new collaborators."""

        obs = SecurityObserver(
            semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
            log_enabled=False,
        )
        result = obs.observe(
            agent_id="researcher",
            response="AI detects cyber threats.",
            original_task=TASK,
            assigned_subtask=TASK,
        )
        self.assertTrue(result.semantic_assessment.assessed)


class GuardTests(unittest.TestCase):

    def test_tiered_path_rejects_ground_truth_artifacts(self):
        obs = _observer()
        with self.assertRaises(ValueError):
            obs.observe_tiered(
                agent_id="researcher",
                response=ON_TOPIC,
                original_task=TASK,
                assigned_subtask=TASK,
                artifacts=[
                    {
                        "artifact_type": "tool_result",
                        "source": "internet_search",
                        "receiver": "researcher",
                        "text": "content",
                        "metadata": {"attack_type": "prompt_infection"},
                    }
                ],
                events=[],
            )


if __name__ == "__main__":
    unittest.main()
