"""
Tests for evidence linking (Task J).

No model download: the shared encoder is injected through
``SemanticAssessor(model=...)`` and reused by ``EvidenceLinker``.
"""

from __future__ import annotations

import unittest

import numpy as np

from security.evidence_linker import (
    EvidenceCandidate,
    EvidenceLinker,
)
from security.semantic_assessor import SemanticAssessor


class _KeywordEncoder:
    """
    Deterministic bag-of-keywords encoder.

    Texts sharing the relevant lexicon land close together; unrelated
    decoys land far apart -- no model download.
    """

    KEYWORDS = (
        "ransomware", "encryption", "threat", "detection", "ai",
        "security", "network", "telemetry",
        # decoy vocabularies
        "recipe", "cookie", "butter", "sugar",
        "weather", "rain", "temperature",
        "traffic", "commute", "bicycle",
        "coffee", "roast", "espresso",
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


RELEVANT = (
    "AI-powered ransomware detection uses network telemetry to spot "
    "the encryption of files before the threat spreads across the "
    "network. Detection models watch network telemetry for the "
    "encryption behaviour of ransomware and raise a threat alert for "
    "the security team to act on quickly."
)

DECOYS = [
    "Chocolate chip cookie recipe: cream the butter with sugar and "
    "bake until golden. The recipe uses butter, sugar and flour, and "
    "the oven temperature matters.",

    "Local weather: rain is expected, with a cool temperature and "
    "light wind across the region through the afternoon, so carry an "
    "umbrella if you travel.",

    "City traffic report: the bicycle commute is slow this morning, "
    "with heavy traffic on the bridge and a long queue for cyclists "
    "heading downtown.",

    "Coffee guide: a light roast espresso needs a fine grind and a "
    "short extraction; the roast and the coffee beans decide the "
    "flavour of the espresso.",
]

QUERY = (
    "Ransomware detection via AI and network telemetry."
)


def _linker() -> EvidenceLinker:
    return EvidenceLinker(SemanticAssessor(model=_KeywordEncoder()))


class EvidenceLinkerTests(unittest.TestCase):

    def test_relevant_chunk_ranks_first_among_decoys(self):
        linker = _linker()

        candidates = [
            EvidenceCandidate(text=RELEVANT, source="artifact:tool_result", index=0),
        ] + [
            EvidenceCandidate(
                text=decoy, source="artifact:tool_result", index=i + 1
            )
            for i, decoy in enumerate(DECOYS)
        ]

        links = linker.link(QUERY, candidates, top_k=1)

        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].candidate.text, RELEVANT)

    def test_relevant_first_even_when_decoys_come_first(self):
        """Ordering of candidates must not decide the ranking."""

        linker = _linker()
        candidates = [
            EvidenceCandidate(text=decoy, source="artifact:tool_result", index=i)
            for i, decoy in enumerate(DECOYS)
        ] + [
            EvidenceCandidate(text=RELEVANT, source="artifact:tool_result", index=99),
        ]

        links = linker.link(QUERY, candidates, top_k=3)
        self.assertEqual(links[0].candidate.text, RELEVANT)

    def test_top_k_returns_descending_similarity(self):
        linker = _linker()
        candidates = [
            EvidenceCandidate(text=RELEVANT, source="artifact:tool_result", index=0),
            *[
                EvidenceCandidate(
                    text=decoy, source="artifact:tool_result", index=i + 1
                )
                for i, decoy in enumerate(DECOYS)
            ],
        ]

        links = linker.link(QUERY, candidates, top_k=3)
        self.assertEqual(len(links), 3)
        scores = [link.similarity for link in links]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_link_over_artifacts_from_environment_shape(self):
        linker = _linker()
        artifacts = [
            {
                "artifact_type": "tool_result",
                "source": "internet_search",
                "receiver": "researcher",
                "text": decoy,
                "request_id": f"req-{i}",
                "message_id": None,
            }
            for i, decoy in enumerate(DECOYS)
        ]
        artifacts.append(
            {
                "artifact_type": "tool_result",
                "source": "internet_search",
                "receiver": "researcher",
                "text": RELEVANT,
                "request_id": "req-relevant",
                "message_id": None,
            }
        )

        links = linker.link_to_artifacts(QUERY, artifacts, top_k=1)
        self.assertEqual(len(links), 1)
        self.assertIn("ransomware", links[0].candidate.text.lower())
        self.assertTrue(links[0].candidate.is_artifact)

    def test_link_over_earlier_agent_responses(self):
        linker = _linker()
        responses = {
            "researcher": RELEVANT,
            "analyst": DECOYS[0],
        }
        links = linker.link_to_agent_responses(QUERY, responses, top_k=1)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].candidate.source, "agent:researcher")

    def test_ground_truth_artifact_is_rejected(self):
        """The linker must refuse a forbidden ground-truth key."""

        linker = _linker()
        artifacts = [
            {
                "artifact_type": "tool_result",
                "source": "internet_search",
                "receiver": "researcher",
                "text": RELEVANT,
                # Forbidden ground-truth key.
                "metadata": {"attack_type": "prompt_infection"},
            }
        ]
        with self.assertRaises(ValueError):
            linker.link_to_artifacts(QUERY, artifacts, top_k=1)

    def test_empty_inputs_return_no_links(self):
        linker = _linker()
        self.assertEqual(linker.link("", [], top_k=1), [])
        self.assertEqual(
            linker.link(
                QUERY,
                [EvidenceCandidate(text="", source="x", index=0)],
                top_k=1,
            ),
            [],
        )
        self.assertEqual(
            linker.link(
                QUERY,
                [EvidenceCandidate(text=RELEVANT, source="x", index=0)],
                top_k=0,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
