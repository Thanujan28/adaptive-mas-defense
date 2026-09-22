"""
security/contradiction_checker.py

Sentence-level Natural Language Inference (NLI) contradiction checking
for the defender's evidence chain.

WHAT THIS IS
------------
Given an EVIDENCE chunk (a passage an agent received, e.g. a search
result) and an agent's CLAIM, this module asks a local NLI model
whether the claim CONTRADICTS, is ENTAILED BY, or is NEUTRAL with
respect to the evidence.

DIRECTION MATTERS (documented explicitly)
-----------------------------------------
NLI is directional. ``check_contradiction(premise, hypothesis)`` is NOT
symmetric: the EVIDENCE is ALWAYS the PREMISE and the AGENT CLAIM is
ALWAYS the HYPOTHESIS. Swapping them changes the verdict (entailment in
one direction is often neutrality or contradiction in the other), so
this ordering is enforced everywhere, including
``check_chunk_against_evidence`` where the evidence chunk is the
premise and each sentence of the agent's chunk is the hypothesis.

KNOWN LIMITATIONS (documented explicitly)
-----------------------------------------
* NLI is WEAK on unsupported-but-not-contradicted claims. A fabricated
  claim that the evidence simply says nothing about is usually labelled
  NEUTRAL, not CONTRADICTION. Those cases are the job of the Tier-3 LLM
  judge (security/llm_judge.py), which is prompted to reason about
  unsupported claims; this module deliberately does not pretend to
  cover them.
* This is a SENTENCE-LEVEL check, not a paragraph-level one. A
  multi-sentence agent claim is split into sentences and each is
  checked independently; a claim whose contradiction only appears
  across sentence boundaries can be missed.
* The model is loaded lazily; constructing a checker never downloads
  weights until it is actually used. Tests inject a stub via
  ``model=`` (same pattern as ``SemanticAssessor(model=...)``) so no
  download is required.

DEFAULT MODEL
-------------
``roberta-large-mnli`` (a standard 3-way MNLI model) is the default.
``microsoft/deberta-v3-base-mnli`` is a lighter alternative; pass it as
``model_name=``. ``transformers`` is already a project dependency
(requirements.txt), so no new dependency is introduced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

# Labels emitted by MNLI-style models, in the conventional order.
LABEL_ENTAILMENT = "entailment"
LABEL_CONTRADICTION = "contradiction"
LABEL_NEUTRAL = "neutral"

# Default local NLI model. roberta-large-mnli is a well-known 3-way MNLI
# checkpoint; tests never load it because they inject a stub model.
DEFAULT_NLI_MODEL = "roberta-large-mnli"

# Sentence boundary for splitting an agent chunk into sentence-level
# hypotheses. Conservative: splits on . ! ? followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ContradictionResult:
    """
    Result of one NLI check.

    ``label`` is one of ``entailment`` / ``contradiction`` / ``neutral``.
    ``confidence`` is the model's probability for the returned label, in
    [0, 1]. ``premise`` / ``hypothesis`` record the direction actually
    used, so a caller can verify evidence was the premise.
    """

    label: str
    confidence: float
    premise: str = ""
    hypothesis: str = ""

    @property
    def is_contradiction(self) -> bool:
        return self.label == LABEL_CONTRADICTION

    @property
    def is_entailment(self) -> bool:
        return self.label == LABEL_ENTAILMENT


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return sentences or [text]


class ContradictionChecker:
    """
    Local NLI-based contradiction checker.

    The NLI model is injectable via ``model=`` (same pattern as
    ``SemanticAssessor(model=...)``): any object exposing a
    ``__call__(text, text_pair, ...)`` or an ``__call__`` accepting a
    single ``{"text","text_pair"}``/list argument that returns
    ``{"label": ..., "score": ...}`` (or ``[{"label","score"}, ...]``)
    works, so tests use a deterministic stub with no download.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        model: Optional[object] = None,
        contradiction_floor: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self._model = model
        # Minimum confidence for a contradiction to be reported as such
        # rather than falling back to the next-strongest label. Default
        # 0.0 keeps the raw argmax verdict.
        self.contradiction_floor = contradiction_floor

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _get_model(self):
        if self._model is None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "transformers is required for NLI contradiction "
                    "checking. Install it with "
                    "`pip install transformers`."
                ) from exc

            self._model = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )
        return self._model

    # =========================================================
    # RAW CALL NORMALISATION
    # =========================================================

    @staticmethod
    def _normalise_output(raw) -> list[dict]:
        """
        Normalise a HF pipeline result into a list of
        ``{"label": str, "score": float}`` dicts.
        """

        # A pipeline with top_k=None returns [[{...}, {...}, {...}]] for
        # a single input, or [{...}] for single-label pipelines.
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        if isinstance(raw, dict):
            raw = [raw]
        return [
            {
                "label": str(item.get("label", "")).lower(),
                "score": float(item.get("score", 0.0)),
            }
            for item in raw
        ]

    @staticmethod
    def _canonical_label(label: str) -> str:
        """
        Map model-specific label spellings onto the canonical three.
        """

        label = label.lower()
        if "contradict" in label:
            return LABEL_CONTRADICTION
        if "entail" in label:
            return LABEL_ENTAILMENT
        if "neutral" in label:
            return LABEL_NEUTRAL
        return label

    # =========================================================
    # PUBLIC API
    # =========================================================

    def check_contradiction(
        self,
        premise: str,
        hypothesis: str,
    ) -> ContradictionResult:
        """
        Run one directional NLI check.

        THE EVIDENCE IS THE PREMISE; THE AGENT CLAIM IS THE HYPOTHESIS.
        This ordering is not symmetric and must not be swapped.
        """

        premise = (premise or "").strip()
        hypothesis = (hypothesis or "").strip()

        if not premise or not hypothesis:
            return ContradictionResult(
                label=LABEL_NEUTRAL,
                confidence=0.0,
                premise=premise,
                hypothesis=hypothesis,
            )

        model = self._get_model()
        raw = model(
            {"text": premise, "text_pair": hypothesis},
            truncation=True,
        )
        scored = self._normalise_output(raw)

        if not scored:
            return ContradictionResult(
                label=LABEL_NEUTRAL,
                confidence=0.0,
                premise=premise,
                hypothesis=hypothesis,
            )

        canonical = [
            {
                "label": self._canonical_label(item["label"]),
                "score": item["score"],
            }
            for item in scored
        ]

        # Pick the highest-scoring label, but do not report a
        # contradiction below the configured floor.
        best = max(canonical, key=lambda item: item["score"])

        if (
            best["label"] == LABEL_CONTRADICTION
            and best["score"] < self.contradiction_floor
        ):
            non_contradictions = [
                item for item in canonical
                if item["label"] != LABEL_CONTRADICTION
            ]
            if non_contradictions:
                best = max(
                    non_contradictions, key=lambda item: item["score"]
                )

        return ContradictionResult(
            label=best["label"],
            confidence=float(best["score"]),
            premise=premise,
            hypothesis=hypothesis,
        )

    def check_chunk_against_evidence(
        self,
        chunk_text: str,
        evidence_chunks: Sequence[str],
    ) -> ContradictionResult:
        """
        Check an agent's CHUNK against a list of EVIDENCE chunks.

        The chunk is split into sentences; each sentence is checked as
        the HYPOTHESIS against every evidence chunk as the PREMISE
        (evidence is ALWAYS the premise -- see the module docstring).

        Returns the STRONGEST CONTRADICTION found (highest-confidence
        contradiction across all sentence x evidence pairs). If there is
        no contradiction at all, returns the BEST ENTAILMENT found, or a
        neutral result if the evidence entails nothing either. The
        returned ``premise``/``hypothesis`` identify the winning pair.

        This is sentence-level, not paragraph-level: a contradiction
        that only emerges across the chunk's sentences as a whole can be
        missed, and an unsupported-but-not-contradicted sentence yields
        NEUTRAL (deferred to the Tier-3 judge).
        """

        evidence_list = [
            (e or "").strip() for e in (evidence_chunks or []) if (e or "").strip()
        ]
        sentences = _split_sentences(chunk_text)

        if not evidence_list or not sentences:
            return ContradictionResult(
                label=LABEL_NEUTRAL,
                confidence=0.0,
            )

        contradictions: list[ContradictionResult] = []
        entailments: list[ContradictionResult] = []

        for sentence in sentences:
            for evidence in evidence_list:
                result = self.check_contradiction(
                    premise=evidence,
                    hypothesis=sentence,
                )
                if result.is_contradiction:
                    contradictions.append(result)
                elif result.is_entailment:
                    entailments.append(result)

        if contradictions:
            return max(contradictions, key=lambda r: r.confidence)
        if entailments:
            return max(entailments, key=lambda r: r.confidence)

        return ContradictionResult(
            label=LABEL_NEUTRAL,
            confidence=0.0,
        )
