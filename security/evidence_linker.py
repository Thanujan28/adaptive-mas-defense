"""
security/evidence_linker.py

One-hop-back evidence linking for the defender (Task J).

Given a CHUNK from one agent's output, retrieve the most relevant
chunk(s) from an EARLIER-STAGE agent's response, or from the observable
artifact channel (``MASEnvironment.get_observable_artifacts()``). This
supports tracing an agent's claim back to the evidence it received --
e.g. a researcher's summary chunk back to the raw search-result
artifact that fed it.

Design constraints
------------------
* Reuses the EXISTING embedding path (``SemanticAssessor``'s injected
  encoder / ``_embed``) -- it does not create a second embedding model.
  Cosine similarity is the same ``_cosine_similarity`` used by the
  semantic assessor.
* Scoped to the CURRENT EPISODE: only candidates the caller passes in
  are considered. The linker never reaches into environment globals.
* Respects ``assert_no_ground_truth``: events/artifacts used as
  candidate sources are validated, so a caller cannot accidentally link
  evidence back to a ground-truth channel.
* Read-only and deterministic given the same encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from environment.visibility import assert_no_ground_truth

from .semantic_assessor import (
    SemanticAssessor,
    _cosine_similarity,
    _as_text,
    split_into_chunks,
)


@dataclass(frozen=True)
class EvidenceCandidate:
    """
    One linkable candidate passage.

    ``source`` is a short provenance tag: ``"agent:<name>"`` for another
    agent's response, or ``"artifact:<artifact_type>"`` for an observable
    artifact. ``index`` is the chunk index within that source (or the
    artifact's position when ``is_artifact`` is True).
    """

    text: str
    source: str
    index: int
    is_artifact: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceLink:
    """A ranked link from a query chunk to a candidate."""

    candidate: EvidenceCandidate
    similarity: float


class EvidenceLinker:
    """
    Top-k cosine evidence linker over an episode's observable sources.

    The embedding model is shared with the semantic assessor (injected
    via its ``model=``), so no extra model is loaded.
    """

    def __init__(
        self,
        assessor: SemanticAssessor,
        chunk_min_words: int = 150,
        chunk_max_words: int = 200,
    ) -> None:
        self.assessor = assessor
        self.chunk_min_words = chunk_min_words
        self.chunk_max_words = chunk_max_words

    # =========================================================
    # CANDIDATE BUILDING
    # =========================================================

    def _chunks_for_text(self, text: str) -> list[str]:
        return split_into_chunks(
            text,
            min_words=self.chunk_min_words,
            max_words=self.chunk_max_words,
        )

    def candidates_from_agent_responses(
        self,
        responses: Mapping[str, str],
    ) -> list[EvidenceCandidate]:
        """
        Build candidates from ``{agent_name: response_text}`` (earlier
        stage agents' outputs). Each response is chunked the same way
        agent output is chunked elsewhere.
        """

        candidates: list[EvidenceCandidate] = []
        for agent_name, response in (responses or {}).items():
            for index, chunk in enumerate(self._chunks_for_text(response)):
                candidates.append(
                    EvidenceCandidate(
                        text=chunk,
                        source=f"agent:{agent_name}",
                        index=index,
                    )
                )
        return candidates

    def candidates_from_artifacts(
        self,
        artifacts: Sequence[Mapping[str, Any]],
    ) -> list[EvidenceCandidate]:
        """
        Build candidates from ``get_observable_artifacts()`` output.

        Any artifact carrying a forbidden ground-truth key is rejected
        by ``assert_no_ground_truth`` before use.
        """

        artifacts = list(artifacts or [])

        # Strict guard: evidence linking is a defender-side path and
        # must never consume a ground-truth channel.
        assert_no_ground_truth([], artifacts=artifacts)

        candidates: list[EvidenceCandidate] = []
        for index, artifact in enumerate(artifacts):
            text = _as_text(artifact.get("text"))
            for chunk_index, chunk in enumerate(self._chunks_for_text(text)):
                candidates.append(
                    EvidenceCandidate(
                        text=chunk,
                        source=f"artifact:{artifact.get('artifact_type')}",
                        index=chunk_index,
                        is_artifact=True,
                        metadata={
                            "artifact_index": index,
                            "receiver": artifact.get("receiver"),
                            "source": artifact.get("source"),
                            "request_id": artifact.get("request_id"),
                            "message_id": artifact.get("message_id"),
                        },
                    )
                )
        return candidates

    # =========================================================
    # QUERY
    # =========================================================

    def link(
        self,
        query_chunk: str,
        candidates: Sequence[EvidenceCandidate],
        top_k: int = 1,
    ) -> list[EvidenceLink]:
        """
        Return the ``top_k`` candidates most similar to
        ``query_chunk``, ranked by cosine similarity descending.

        ``candidates`` MUST be scoped by the caller to the current
        episode; the linker does not fetch anything itself.
        """

        query_chunk = _as_text(query_chunk)
        candidates = [c for c in (candidates or []) if _as_text(c.text)]

        if not query_chunk or not candidates or top_k <= 0:
            return []

        texts = [query_chunk] + [c.text for c in candidates]

        # One batched embed call, using the assessor's shared encoder.
        embeddings = np.asarray(
            self.assessor._embed(texts),
            dtype=np.float32,
        )

        query_vec = embeddings[0]
        scored: list[EvidenceLink] = []
        for offset, candidate in enumerate(candidates, start=1):
            similarity = _cosine_similarity(query_vec, embeddings[offset])
            scored.append(
                EvidenceLink(candidate=candidate, similarity=float(similarity))
            )

        scored.sort(key=lambda link: link.similarity, reverse=True)
        return scored[:top_k]

    def link_to_artifacts(
        self,
        query_chunk: str,
        artifacts: Sequence[Mapping[str, Any]],
        top_k: int = 1,
    ) -> list[EvidenceLink]:
        """Convenience wrapper: ``link`` over artifact candidates."""

        return self.link(
            query_chunk,
            self.candidates_from_artifacts(artifacts),
            top_k=top_k,
        )

    def link_to_agent_responses(
        self,
        query_chunk: str,
        responses: Mapping[str, str],
        top_k: int = 1,
    ) -> list[EvidenceLink]:
        """Convenience wrapper: ``link`` over agent-response candidates."""

        return self.link(
            query_chunk,
            self.candidates_from_agent_responses(responses),
            top_k=top_k,
        )
