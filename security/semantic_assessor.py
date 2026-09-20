from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Cosine similarity between two 1-D embedding vectors.

    Returns a value in [-1, 1]; a zero vector yields 0.0.
    """

    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_text(value) -> str:
    # Normalise an arbitrary value into stripped text.
    #
    # Strings are stripped directly; other values (e.g. dict/list
    # structured outputs) are rendered with str() so the semantic
    # assessor always operates on text.

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


@dataclass
class SemanticAssessment:
    """
    Semantic assessment of an agent output.

    All scores are normalized to [0, 1].
    """

    task_similarity: float = 0.0
    subtask_similarity: float = 0.0
    objective_deviation: float = 0.0
    scope_deviation: float = 0.0
    confidence: float = 0.0
    assessed: bool = False

    @property
    def deviation_score(self) -> float:
        """
        Combined semantic deviation.

        Higher value means greater deviation from the intended task.
        """
        similarity = (
            0.5 * self.task_similarity
            + 0.5 * self.subtask_similarity
        )

        semantic_distance = 1.0 - similarity

        return max(
            0.0,
            min(
                1.0,
                0.5 * semantic_distance
                + 0.3 * self.objective_deviation
                + 0.2 * self.scope_deviation,
            ),
        )


class SemanticAssessor:
    """
    Semantic assessment backed by sentence-transformers embeddings.

    Agent outputs are embedded together with the original user task
    and the agent's assigned subtask. Cosine similarity between the
    embeddings drives the semantic deviation score consumed by the
    security observer and the PPO state builder.

    The model is loaded lazily on first use, so importing this module
    (or constructing an assessor) never requires the model to be
    downloaded until an assessment is actually requested.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        deviation_threshold: float = 0.45,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Optional[object] = None,
        min_confidence: float = 0.25,
        similarity_floor: float = 0.0,
        similarity_ceiling: float = 1.0,
    ) -> None:
        self.deviation_threshold = deviation_threshold
        self.model_name = model_name

        # An explicitly provided model (any object exposing ``encode``
        # that returns a sequence of vectors) bypasses the lazy
        # sentence-transformers load. This keeps the class testable
        # without downloading weights.
        self._model = model

        # Confidence scaling: very short outputs carry less evidence,
        # so they are down-weighted.
        self.min_confidence = min_confidence

        # Raw cosine similarity is in [-1, 1]. These bounds map it onto
        # the [0, 1] similarity range the deviation formula expects. By
        # default negative/orthogonal similarity becomes 0.0 similarity
        # (maximum distance) and a perfect match becomes 1.0.
        self.similarity_floor = similarity_floor
        self.similarity_ceiling = similarity_ceiling

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _get_model(self):
        """
        Lazily load and cache the sentence-transformers model.
        """

        if self._model is None:

            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )

            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for semantic "
                    "assessment. Install it with "
                    "`pip install sentence-transformers`."
                ) from exc

            self._model = SentenceTransformer(
                self.model_name
            )

        return self._model

    def _embed(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Embed a batch of texts into a (n, dim) float array.
        """

        model = self._get_model()

        embeddings = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    # =========================================================
    # SIMILARITY HELPERS
    # =========================================================

    def _normalize_similarity(
        self,
        similarity: float,
    ) -> float:
        """
        Map a raw cosine similarity onto [0, 1].
        """

        span = (
            self.similarity_ceiling
            - self.similarity_floor
        )

        if span <= 0.0:
            return _clamp01(similarity)

        return _clamp01(
            (similarity - self.similarity_floor)
            / span
        )

    @staticmethod
    def _confidence_for(
        text: str,
        min_confidence: float,
    ) -> float:
        """
        Heuristic confidence for one assessment.

        Longer, content-bearing outputs yield more reliable similarity
        estimates than near-empty strings.
        """

        token_count = len(text.split())

        # Saturate confidence at ~20 tokens.
        length_factor = min(
            1.0,
            token_count / 20.0,
        )

        return _clamp01(
            min_confidence
            + (1.0 - min_confidence) * length_factor
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def assess(
        self,
        original_task: str,
        assigned_subtask: str,
        agent_output: str,
    ) -> SemanticAssessment:
        """
        Assess an agent output against the original task and the
        subtask the agent was assigned.

        Similarity is computed with real sentence embeddings. A
        missing/empty subtask falls back to the original task so that
        the "subtask" axis stays well defined.
        """

        # Agent outputs may arrive as non-string content (e.g. a
        # structured plan dict). Coerce to text so the embedding call
        # never receives a non-string.
        original_task = _as_text(original_task)
        assigned_subtask = _as_text(assigned_subtask)
        agent_output = _as_text(agent_output)

        # -----------------------------------------------------
        # Empty output is a definite deviation.
        # -----------------------------------------------------

        if not agent_output:
            return SemanticAssessment(
                task_similarity=0.0,
                subtask_similarity=0.0,
                objective_deviation=1.0,
                scope_deviation=1.0,
                confidence=1.0,
                assessed=True,
            )

        # -----------------------------------------------------
        # Without a reference task there is nothing to anchor to.
        # -----------------------------------------------------

        if not original_task and not assigned_subtask:
            return SemanticAssessment(
                confidence=0.0,
                assessed=False,
            )

        reference_subtask = (
            assigned_subtask or original_task
        )

        embeddings = self._embed(
            [
                original_task or assigned_subtask,
                reference_subtask,
                agent_output,
            ]
        )

        task_embedding = embeddings[0]
        subtask_embedding = embeddings[1]
        output_embedding = embeddings[2]

        task_similarity = self._normalize_similarity(
            _cosine_similarity(
                task_embedding,
                output_embedding,
            )
        )

        subtask_similarity = self._normalize_similarity(
            _cosine_similarity(
                subtask_embedding,
                output_embedding,
            )
        )

        # Objective deviation: how far the output drifts from the
        # immutable user task.
        objective_deviation = _clamp01(
            1.0 - task_similarity
        )

        # Scope deviation: how far the output drifts from the specific
        # subtask this agent was asked to perform.
        scope_deviation = _clamp01(
            1.0 - subtask_similarity
        )

        confidence = self._confidence_for(
            agent_output,
            self.min_confidence,
        )

        return SemanticAssessment(
            task_similarity=task_similarity,
            subtask_similarity=subtask_similarity,
            objective_deviation=objective_deviation,
            scope_deviation=scope_deviation,
            confidence=confidence,
            assessed=True,
        )
