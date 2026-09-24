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


class SharedChunkTests(unittest.TestCase):
    """
    The EXACT SAME response chunk must be used by Tier 1 (semantic) and
    Tier 2 (NLI).

    The authoritative chunks are produced once, by the existing
    ``SemanticAssessor`` (``assess_chunked`` -> ``chunked.chunk_texts``);
    Tier 2 must consume those chunk TEXTS, not a second chunking of the
    response. These tests record the actual text handed to each tier and
    compare it, so a second/re-split chunking mechanism would fail them.
    """

    def test_same_chunk_text_reaches_both_tiers(self):
        """
        Every chunk text used by Tier 1 == the chunk text used by Tier 2,
        for every index -- verified on the TEXT, not just the index.
        """

        # The NLI stub records the exact hypothesis (the agent chunk it
        # was asked about), so we can compare it against the chunks the
        # semantic assessor produced.
        seen_hypotheses: list[str] = []

        class _RecordingNLI:
            def __call__(self, inputs, truncation=True):
                seen_hypotheses.append(str(inputs["text_pair"]))
                return [
                    {"label": "neutral", "score": 0.8},
                    {"label": "entailment", "score": 0.1},
                    {"label": "contradiction", "score": 0.1},
                ]

        assessor = SemanticAssessor(model=_KeywordEncoder())
        observer = SecurityObserver(
            semantic_assessor=assessor,
            contradiction_checker=ContradictionChecker(
                model=_RecordingNLI()
            ),
            log_enabled=False,
        )

        response = (
            ("AI improves cyber security threat detection and network "
             "defence. ") * 40
            + "\n\n"
            + ("Threat detection uses machine learning on network traffic. ")
            * 40
            + "\n\n"
            + ("Cookie recipe with butter and sugar: bake until golden. ")
            * 40
        )

        # The authoritative chunks: exactly what the semantic assessor
        # (Tier 1) split the response into.
        expected = assessor.assess_chunked(
            original_task=TASK,
            assigned_subtask=TASK,
            agent_output=response,
        ).chunk_texts
        self.assertGreaterEqual(len(expected), 2, "need multiple chunks")

        result = observer.observe_tiered(
            agent_id="researcher",
            response=response,
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )

        # Tier 1 chunk (ChunkDecision.chunk_text) == semantic chunk.
        semantic_chunks = [d.chunk_text for d in result.chunk_decisions]
        self.assertEqual(semantic_chunks, expected)

        # Tier 2 ran for every chunk (evidence was supplied).
        for decision in result.chunk_decisions:
            self.assertIn("tier2", decision.tiers_ran)
            self.assertIsNotNone(decision.contradiction)

            nli_chunk = decision.contradiction.hypothesis
            # NLI internally splits the chunk into sentences, so the
            # hypothesis is a SENTENCE of the SAME chunk -- never some
            # other chunk, and never a re-chunking of the response.
            self.assertTrue(
                nli_chunk in decision.chunk_text,
                "Tier 2 hypothesis is not part of the Tier 1 chunk",
            )
            self.assertIn(
                decision.chunk_text,
                expected,
                "Tier 2 consumed a chunk Tier 1 never produced",
            )

        # Chunk-by-chunk: Chunk i used by Semantic == Chunk i used by NLI.
        for index, decision in enumerate(result.chunk_decisions):
            self.assertEqual(
                decision.chunk_text,
                expected[index],
                f"chunk {index} differs between tiers",
            )
            # Tier 2's hypothesis for chunk i comes from chunk i itself.
            self.assertIn(
                decision.contradiction.hypothesis,
                expected[index],
                f"Tier 2 read a different chunk at index {index}",
            )

        # Every recorded NLI hypothesis belongs to some Tier-1 chunk,
        # i.e. no chunk was invented by the NLI path.
        self.assertTrue(seen_hypotheses, "NLI was never called")
        for hypothesis in seen_hypotheses:
            self.assertTrue(
                any(hypothesis in chunk for chunk in expected),
                "NLI was asked about text outside the Tier-1 chunks",
            )

    def test_evidence_is_premise_and_chunk_is_hypothesis(self):
        """Direction must be evidence -> premise, chunk -> hypothesis."""

        observer = SecurityObserver(
            semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
            contradiction_checker=ContradictionChecker(model=_SweepNLI()),
            log_enabled=False,
        )

        evidence = "AI detects threats in real time."
        result = observer.observe_tiered(
            agent_id="researcher",
            response=(
                ("STRONGCONTRADICT AI offers no benefit. ") * 40
            ),
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=[evidence],
            events=[],
        )

        decision = result.chunk_decisions[0]
        self.assertEqual(decision.contradiction.premise, evidence)
        # The hypothesis is a sentence of the chunk used by Tier 1.
        self.assertIn(
            decision.contradiction.hypothesis,
            decision.chunk_text,
        )
        self.assertNotEqual(
            decision.contradiction.premise,
            decision.contradiction.hypothesis,
        )

    def test_nli_is_not_called_when_no_evidence(self):
        """No evidence -> no Tier 2, and no chunking work for NLI at all."""

        calls: list[str] = []

        class _RecordingNLI:
            def __call__(self, inputs, truncation=True):
                calls.append(str(inputs["text_pair"]))
                return [{"label": "neutral", "score": 1.0}]

        observer = SecurityObserver(
            semantic_assessor=SemanticAssessor(model=_KeywordEncoder()),
            contradiction_checker=ContradictionChecker(model=_RecordingNLI()),
            log_enabled=False,
        )

        result = observer.observe_tiered(
            agent_id="researcher",
            response=ON_TOPIC,
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=[],
            events=[],
        )

        self.assertEqual(calls, [])
        for decision in result.chunk_decisions:
            self.assertEqual(decision.tiers_ran, ["tier1"])
            self.assertIsNone(decision.contradiction)

    def test_chunk_decision_retains_index_text_semantic_and_nli(self):
        """Each decision keeps chunk_index, chunk_text, semantic + NLI."""

        observer = _observer()
        result = observer.observe_tiered(
            agent_id="researcher",
            response=(
                ON_TOPIC
                + "\n\n"
                + ("STRONGCONTRADICT AI offers no benefit. " * 24)
            ),
            original_task=TASK,
            assigned_subtask=TASK,
            evidence_chunks=EVIDENCE,
            events=[],
        )

        for index, decision in enumerate(result.chunk_decisions):
            self.assertEqual(decision.chunk_index, index)
            self.assertTrue(decision.chunk_text.strip())
            self.assertIsNotNone(decision.semantic)
            self.assertIsNotNone(decision.contradiction)
            self.assertIn("tier1", decision.tiers_ran)
            self.assertIn("tier2", decision.tiers_ran)


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
