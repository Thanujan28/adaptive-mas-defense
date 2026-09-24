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

TWO SEPARATE TASK-ALIGNMENT FAMILIES
------------------------------------
Evidence checking and TASK ALIGNMENT are different questions, so this
module exposes them separately and never blends them:

  * EVIDENCE axis (pre-existing): ``check_chunk_against_evidence`` -- the
    retrieved evidence is the premise, the response claim is the
    hypothesis. Used for provenance and the Tier-3 gate.

  * TASK-ALIGNMENT axes (added): the reference is the premise and the
    response claim is the hypothesis, for BOTH
    ``REFERENCE_ORIGINAL_TASK`` (PRIMARY: the immutable user request) and
    ``REFERENCE_ASSIGNED_SUBTASK`` (SECONDARY: the delegated subtask).
    Ready via ``check_chunk_against_reference`` /
    ``check_chunk_against_task_and_subtask``.

Retrieved content, tool output, memory and evidence are NEVER the premise
of a task-alignment check, the assigned subtask never overwrites the
original-task result, and a missing subtask is reported as NOT RUN with a
reason (never silently replaced by the original task). Both families are
run over the IDENTICAL list of claims extracted once from the
authoritative response chunk.

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
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Labels emitted by MNLI-style models, in the conventional order.
LABEL_ENTAILMENT = "entailment"
LABEL_CONTRADICTION = "contradiction"
LABEL_NEUTRAL = "neutral"

# ---------------------------------------------------------------------
# REFERENCE-ALIGNMENT FAMILIES (task alignment)
# ---------------------------------------------------------------------
#
# The task-alignment checks are NOT evidence checks. Each family fixes the
# REFERENCE as the premise and the agent's response CLAIM as the
# hypothesis. Two families are always kept separate:
#
#   REFERENCE_ORIGINAL_TASK      -- PRIMARY: the immutable user request.
#   REFERENCE_ASSIGNED_SUBTASK   -- SECONDARY: the delegated subtask,
#                                   which upstream agents can influence.
#
# Neither family is allowed to use retrieved content, tool output, memory
# or evidence as its premise, and neither may overwrite the other.
REFERENCE_ORIGINAL_TASK = "original_task"
REFERENCE_ASSIGNED_SUBTASK = "assigned_subtask"

# Explicit NOT_RUN reasons (published verbatim so the dashboard can show
# WHY a comparison is unavailable instead of inventing a score).
NOT_RUN_NO_ORIGINAL_TASK = "original task is missing or empty"
NOT_RUN_NO_SUBTASK = "assigned subtask is missing or empty"
NOT_RUN_NO_CLAIM = "no valid response claim extracted from the chunk"

# Default local NLI model. roberta-large-mnli is a well-known 3-way MNLI
# checkpoint; tests never load it because they inject a stub model.
DEFAULT_NLI_MODEL = "roberta-large-mnli"

# Sentence boundary for splitting an agent chunk into sentence-level
# hypotheses. Conservative: splits on . ! ? followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------------
# MARKDOWN-AWARE CLAIM EXTRACTION
# ---------------------------------------------------------------------
#
# Agent responses are frequently Markdown. Section headings such as
# ``**VII. Conclusion**`` (or ``### III. Results``) contain sentence
# punctuation ("VII.") but are NOT claims: they carry no proposition to
# check against the evidence. A naive ``. ! ?`` splitter tears ``**VII.``
# off the heading and offers it to the NLI model as a hypothesis, which is
# exactly the bug this module now guards against.
#
# The extraction below runs in four ordered steps, all INSIDE the NLI
# checker (it is claim/sentence extraction, never response re-chunking --
# the WHOLE chunk text is still what Tier 2 receives):
#
#   1. drop fenced code blocks (``` ... ```),            # they are not claims
#   2. drop heading lines (# ... or **Title** on their own line),
#   3. for each remaining line, strip list markers (*, -, +, 1., ...),
#      blockquote markers (>), emphasis and inline code,
#   4. split what is left into sentences and drop anything that is not a
#      real claim (headings, punctuation-only or sub-content fragments).

# A fenced code block (``` ... ```) -- never a claim.
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)

# Emphasis / inline-code / link syntax removed when CLEANING a claim.
# The text inside is KEPT (semantic content is preserved); only the
# Markdown delimiters are removed.
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_EMPHASIS_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_EMPHASIS_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_EMPHASIS_UNDERSCORE = re.compile(r"(?<!\w)_([^_]+)_(?!\w)")

# A list marker at the start of a line: ``*``, ``-``, ``+``, ``1.``,
# ``1)`` -- consumed as structure, not as meaning.
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# A blockquote marker at the start of a line.
_BLOCKQUOTE_MARKER = re.compile(r"^\s*>+\s*")

# A leading upright heading number: ``VII.``, ``3.``, ``A.``, ``IV)``.
_LEADING_HEADING_NUMBER = re.compile(r"^[A-Za-z0-9]{1,4}[.)]\s*")

# The line is an ATX / setext-style heading (``# Intro``, ``## 1. X``).
_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")

# The line is ONLY its own bold, e.g. ``**VII. Conclusion**`` -- i.e. a
# standalone heading written in bold, not prose that merely bolds a few
# words inside a real sentence.
_BOLD_ONLY_LINE = re.compile(r"^\s*\*\*[^*]+\*\*\s*[:.]?\s*$")

# Trailing Markdown table pipes / horizontal rules.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HORIZONTAL_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")


def _clean_markdown(text: str) -> str:
    """
    Remove Markdown delimiters while KEEPING the semantic text.

    ``**AI can help.**`` -> ``AI can help.`` (bold markers dropped, words
    kept). Links keep their label and drop the URL. Inline code keeps its
    content and drops the backticks.
    """

    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _EMPHASIS_BOLD.sub(r"\1", text)
    text = _EMPHASIS_ITALIC.sub(r"\1", text)
    text = _EMPHASIS_UNDERSCORE.sub(r"\1", text)
    # Any stray emphasis markers left over (unbalanced Markdown).
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("`", "")
    return text.strip()


def _looks_like_heading(text: str) -> bool:
    """
    True when ``text`` is (a fragment of) a Markdown section heading
    rather than an actual claim.

    Covers ``# ...``/``## ...``, a standalone bold heading
    (``**VII. Conclusion**``), a bare heading number (``VII.`` / ``**VII.``
    -- the exact fragment the naive splitter used to emit), and a heading
    number followed by a short title (``VII. Conclusion``).
    """

    stripped = (text or "").strip()
    if not stripped:
        return True

    if _ATX_HEADING.match(stripped):
        return True

    # A standalone bold block is a heading, never prose.
    if _BOLD_ONLY_LINE.match(stripped):
        return True

    # Bare heading number / letter: ``VII.`` ``**VII.**`` ``3.`` ``A.``
    core = _clean_markdown(stripped).strip().rstrip(".)").strip()
    if not core:
        return True
    if re.fullmatch(r"[A-Za-z0-9]{1,4}", core):
        return True

    # ``VII. Conclusion`` -- a numbered heading whose body is a very short
    # noun phrase with no sentence punctuation of its own.
    numbered = _LEADING_HEADING_NUMBER.match(stripped)
    if numbered:
        remainder = stripped[numbered.end():].strip()
        if remainder and len(remainder.split()) <= 3 \
                and not re.search(r"[.!?]", remainder):
            return True

    return False


def _is_valid_claim(text: str) -> bool:
    """
    True when ``text`` is a real NLI hypothesis: content that is neither a
    Markdown heading nor a punctuation-only / sub-content fragment.
    """

    cleaned = _clean_markdown(text)
    if not cleaned:
        return False
    # Letters only count -- headings such as ``**VII.`` leave just markers
    # and punctuation, which carry no proposition to check.
    if not re.search(r"[A-Za-z]", cleaned):
        return False
    # Short punctuation-only fragments (``.`` ``id.2``) are not claims.
    if len(cleaned.split()) < 2 and len(cleaned) < 4:
        return False
    if _looks_like_heading(text):
        return False
    return True


def _extract_claims(chunk_text: str) -> list[str]:
    """
    Extract the actual CLAIMS (NLI hypotheses) from a response chunk.

    This is markdown-aware sentence/claim extraction internal to the NLI
    checker -- NOT a second response chunker. The whole chunk text is
    still what Tier 2 receives; this only decides which sentences inside
    it are legitimate hypotheses.

    Markdown headings and heading-only fragments are ignored; the body of
    each line (bullets, quotes, emphasis) is kept.
    """

    text = (chunk_text or "").strip()
    if not text:
        return []

    # 1. Remove fenced code blocks outright.
    text = _FENCED_CODE.sub("\n", text)

    claims: list[str] = []

    # 2. Work line by line so headings can be dropped BEFORE the sentence
    #    splitter ever sees them (this is what stops ``**VII.`` from
    #    becoming a hypothesis).
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Structural lines that are never claims.
        if _HORIZONTAL_RULE.match(line) or _TABLE_ROW.match(line):
            continue
        if _ATX_HEADING.match(line):
            continue
        if _BOLD_ONLY_LINE.match(line):
            continue

        # 3. Strip structural markers, then CLEAN Markdown from the body.
        if _BLOCKQUOTE_MARKER.match(line):
            line = _BLOCKQUOTE_MARKER.sub("", line)
        if _LIST_MARKER.match(line):
            line = _LIST_MARKER.sub("", line)

        cleaned = _clean_markdown(line)
        if not cleaned:
            continue

        # 4. Split the (cleaned) line into sentences; keep only claims.
        for sentence in _split_sentences(cleaned):
            if _is_valid_claim(sentence):
                claims.append(_clean_markdown(sentence))

    return claims


def extract_response_claims(chunk_text: str) -> list[str]:
    """
    PUBLIC name for the markdown-aware CLAIM extraction used by the
    NLI checker.

    The caller still hands in the WHOLE authoritative response chunk (the
    one produced by the existing ``SemanticAssessor`` chunker): this is
    claim extraction inside the checker, NOT a second response chunker.

    Filters out Markdown headings, ATX/bold-only heading lines, horizontal
    rules, table rows, URLs/citation-only lines, section numbers and
    formatting-only fragments, so a fragment such as ``**VII.`` or
    ``[9] https://...`` can never become an NLI hypothesis.
    """

    return _extract_claims(chunk_text)


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


@dataclass
class ReferenceNLISummary:
    """
    Result of ONE reference-alignment NLI family for a response chunk.

    DIRECTION (fixed): the REFERENCE text is the PREMISE and each extracted
    response CLAIM is the HYPOTHESIS. Two summaries are produced for every
    chunk -- one per reference family:

      * ``reference_kind == REFERENCE_ORIGINAL_TASK``     (primary)
      * ``reference_kind == REFERENCE_ASSIGNED_SUBTASK``  (secondary)

    They are stored in separate fields and neither overwrites the other.

    ``claims`` is the EXACT list of extracted response claims the family
    was run over; both families receive the identical list (the claims are
    extracted once per chunk), so the two summaries always report the same
    hypothesis space. ``premise`` / ``hypothesis`` are the winning pair.

    ``ran`` is False (with ``not_run_reason``) when the family could not be
    evaluated -- e.g. a missing subtask or a chunk with no valid claim.
    A missing subtask is NEVER silently replaced by the original task.
    """

    reference_kind: str = ""
    label: str = LABEL_NEUTRAL
    confidence: float = 0.0
    premise: str = ""
    hypothesis: str = ""
    claims: list[str] = field(default_factory=list)

    ran: bool = False
    not_run_reason: str = ""

    @property
    def is_contradiction(self) -> bool:
        return self.ran and self.label == LABEL_CONTRADICTION

    @property
    def is_entailment(self) -> bool:
        return self.ran and self.label == LABEL_ENTAILMENT

    def to_dict(self) -> dict:
        """
        Plain-JSON form for the event stream / dashboards.

        Serialisation only -- no recomputation.
        """

        return {
            "reference_kind": self.reference_kind,
            "ran": bool(self.ran),
            "not_run_reason": self.not_run_reason,
            "label": self.label,
            "confidence": float(self.confidence),
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "claims": list(self.claims),
        }


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
        # PROVENANCE: True only when a model was INJECTED through
        # ``model=`` (a deterministic test/offline double). This is set
        # once at construction and never changes, so it stays correct
        # even after a real model is lazily loaded into ``_model``.
        #
        # ``is_stub`` -- NOT ``_model is not None`` -- is the supported
        # way to tell a real NLI run from a stubbed one: ``_model`` is
        # also populated by the REAL transformers pipeline on first use.
        self.is_stub = model is not None
        # Minimum confidence for a contradiction to be reported as such
        # rather than falling back to the next-strongest label. Default
        # 0.0 keeps the raw argmax verdict.
        self.contradiction_floor = contradiction_floor
    @property
    def source(self) -> str:
        # Provenance of this checker's NLI model: "real" or "stub".
        #
        # "stub" means a model was injected via ``model=`` (tests /
        # offline dry runs); "real" means the lazy ``transformers``
        # pipeline for ``model_name`` (roberta-large-mnli by default) is
        # -- or will be -- used. This is independent of whether the real
        # pipeline has been loaded yet, unlike inspecting ``_model``.
        return "stub" if self.is_stub else "real"

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
        contradiction across all claim x evidence pairs). If there is no
        contradiction at all, returns the BEST ENTAILMENT found, or a
        neutral result if the evidence entails nothing either.

        PROVENANCE: the returned ``premise``/``hypothesis`` are the
        ACTUAL winning pair -- the ``premise`` is the trusted EVIDENCE and
        the ``hypothesis`` is the actual CLAIM extracted from this chunk.
        Heading-only chunks (e.g. ``**VII. Conclusion**``) yield the safe
        neutral/no-claim result with EMPTY ``premise``/``hypothesis``,
        because no hypothesis was fabricated.

        This is sentence-level, not paragraph-level: a contradiction
        that only emerges across the chunk's sentences as a whole can be
        missed, and an unsupported-but-not-contradicted sentence yields
        NEUTRAL (deferred to the Tier-3 judge).
        """

        evidence_list = [
            (e or "").strip() for e in (evidence_chunks or []) if (e or "").strip()
        ]
        # Markdown-aware CLAIM extraction (never a second chunker):
        # headings are dropped, bullet/list claims and prose are kept.
        claims = _extract_claims(chunk_text)

        if not evidence_list or not claims:
            # No genuine claim in this chunk -> safe neutral/no-claim
            # result. We do NOT fabricate a hypothesis (so a heading
            # like ``**VII. Conclusion**`` never becomes one).
            return ContradictionResult(
                label=LABEL_NEUTRAL,
                confidence=0.0,
            )

        contradictions: list[ContradictionResult] = []
        entailments: list[ContradictionResult] = []
        neutrals: list[ContradictionResult] = []

        for claim in claims:
            for evidence in evidence_list:
                result = self.check_contradiction(
                    premise=evidence,
                    hypothesis=claim,
                )
                if result.is_contradiction:
                    contradictions.append(result)
                elif result.is_entailment:
                    entailments.append(result)
                else:
                    neutrals.append(result)

        if contradictions:
            return max(contradictions, key=lambda r: r.confidence)
        if entailments:
            return max(entailments, key=lambda r: r.confidence)
        if neutrals:
            # PROVENANCE: even when the evidence neither entails nor
            # contradicts anything, return the strongest NEUTRAL pair so
            # the result still records WHICH evidence was the premise and
            # WHICH claim was the hypothesis. The evidence axis must never
            # collapse into an anonymous zero that hides its premise.
            return max(neutrals, key=lambda r: r.confidence)

        return ContradictionResult(
            label=LABEL_NEUTRAL,
            confidence=0.0,
        )

    # =========================================================
    # TASK-ALIGNMENT NLI (reference is the premise)
    # =========================================================
    #
    # These checks are SEPARATE from ``check_chunk_against_evidence``:
    # retrieved content / tool output / memory / evidence is NEVER their
    # premise. The reference (original task or assigned subtask) is the
    # premise and the response claim is the hypothesis.

    def _check_claims_against_reference(
        self,
        *,
        claims: Sequence[str],
        reference_text: str,
        reference_kind: str,
    ) -> ReferenceNLISummary:
        """
        Run ONE reference-alignment family over an ALREADY-EXTRACTED claim
        list (shared by both families -- see
        ``check_chunk_against_task_and_subtask``).

        Returns the strongest contradiction found across the claims, else
        the best entailment, else a neutral result for the first claim.
        Nothing is fabricated: a missing reference or a claim-free chunk
        yields ``ran=False`` with a reason.
        """

        reference = (reference_text or "").strip()
        claim_list = [
            claim for claim in (claims or []) if (claim or "").strip()
        ]

        if not reference:
            # The family did NOT run, so it reports NO claim list: a
            # summary that did not execute must not look as though it had
            # a hypothesis space. ``claims`` therefore stays empty and the
            # reason is published verbatim. The original task is never
            # substituted for a missing subtask.
            return ReferenceNLISummary(
                reference_kind=reference_kind,
                ran=False,
                not_run_reason=(
                    NOT_RUN_NO_SUBTASK
                    if reference_kind == REFERENCE_ASSIGNED_SUBTASK
                    else NOT_RUN_NO_ORIGINAL_TASK
                ),
                claims=[],
            )

        if not claim_list:
            return ReferenceNLISummary(
                reference_kind=reference_kind,
                ran=False,
                not_run_reason=NOT_RUN_NO_CLAIM,
                premise=reference,
                claims=[],
            )

        contradictions: list[ContradictionResult] = []
        entailments: list[ContradictionResult] = []

        for claim in claim_list:
            result = self.check_contradiction(
                premise=reference,
                hypothesis=claim,
            )
            if result.is_contradiction:
                contradictions.append(result)
            elif result.is_entailment:
                entailments.append(result)

        if contradictions:
            best = max(contradictions, key=lambda r: r.confidence)
        elif entailments:
            best = max(entailments, key=lambda r: r.confidence)
        else:
            best = ContradictionResult(
                label=LABEL_NEUTRAL,
                confidence=0.0,
                premise=reference,
                hypothesis=claim_list[0],
            )

        return ReferenceNLISummary(
            reference_kind=reference_kind,
            label=best.label,
            confidence=float(best.confidence),
            premise=best.premise,
            hypothesis=best.hypothesis,
            claims=list(claim_list),
            ran=True,
        )

    def check_chunk_against_reference(
        self,
        *,
        chunk_text: str,
        reference_text: str,
        reference_kind: str = REFERENCE_ORIGINAL_TASK,
    ) -> ReferenceNLISummary:
        """
        Check one response chunk against ONE reference text.

        The whole chunk is taken as-is (no second chunker); the claims are
        extracted from it and each claim is the HYPOTHESIS while
        ``reference_text`` is the PREMISE.
        """

        return self._check_claims_against_reference(
            claims=extract_response_claims(chunk_text),
            reference_text=reference_text,
            reference_kind=reference_kind,
        )

    def check_chunk_against_task_and_subtask(
        self,
        *,
        chunk_text: str,
        original_task: str,
        assigned_subtask: str = "",
    ) -> tuple[ReferenceNLISummary, ReferenceNLISummary]:
        """
        Run BOTH task-alignment families over the SAME response chunk.

        The claims are extracted ONCE (from the authoritative chunk text)
        and both families are run over that identical claim list:

          * original task    -> premise  (PRIMARY alignment signal)
          * assigned subtask -> premise  (SECONDARY alignment signal)

        Returns ``(original_task_summary, assigned_subtask_summary)``.
        The two results are independent: neither overwrites the other, and
        a missing subtask produces ``ran=False`` with a reason instead of
        being replaced by the original task.
        """

        claims = extract_response_claims(chunk_text)

        original = self._check_claims_against_reference(
            claims=claims,
            reference_text=original_task,
            reference_kind=REFERENCE_ORIGINAL_TASK,
        )

        subtask = self._check_claims_against_reference(
            claims=claims,
            reference_text=assigned_subtask,
            reference_kind=REFERENCE_ASSIGNED_SUBTASK,
        )

        return original, subtask
