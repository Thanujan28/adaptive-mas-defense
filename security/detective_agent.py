"""
security/detective_agent.py

Detective agent for the remediation/containment layer (Task S).

ROLE
----
This is the FIRST acting component of the new containment layer. It is
spawned ONLY for a chunk that Tier 3 (the LLM judge) has already
CONFIRMED contradicts its linked evidence (see security/observer.py's
``ChunkDecision.tier3_verdict``). Its job is narrow:

  Given a judge-confirmed contradicting CHUNK and the EVIDENCE it was
  computed against, decide whether the chunk's problem is attributable
  to a SPECIFIC upstream artifact (a tool result, or a prior agent's
  output), or whether it is NOT clearly attributable.

It does NOT remediate anything and it does NOT walk the evidence chain
backward -- that is the root-cause tracer's job
(security/root_cause_tracer.py). The detective only produces a
structured ``DetectiveFinding``.

BUDGET
------
The observer PROPOSES a token ceiling for investigating one chunk
(``DETECTIVE_TOKEN_BUDGET``, a named constant). That is a proposal, not
a spend. The detective ENFORCES it: if its own LLM usage would exceed
the ceiling it stops and returns PARTIAL findings
(``budget_exhausted=True``, ``attributable=False``) rather than
overspending. Budget accounting is done with the shared token counter
when one is injected; otherwise a simple word-count estimate is used so
the class is testable without loading a tokenizer.

REUSE
-----
Evidence ranking reuses ``security/evidence_linker.py`` (no second
embedding model): the linker's shared encoder is the assessor's encoder.

This module is read-only with respect to the pipeline: it never mutates
an agent response or the environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .evidence_linker import EvidenceCandidate, EvidenceLinker
from .semantic_assessor import SemanticAssessor


logger = logging.getLogger(__name__)


# =================================================================
# NAMED CONSTANTS (no inline magic numbers)
# =================================================================

# Per-chunk token ceiling PROPOSED for one investigation. The detective
# is hard-capped at this; it never exceeds it (it returns partial
# findings instead). Deliberately a named constant so the containment
# layer's cost is one number to change and to audit.
DETECTIVE_TOKEN_BUDGET = 1500

# How many linked-evidence candidates to inspect per investigation.
DETECTIVE_LINK_TOP_K = 3


# =================================================================
# STRUCTURED FINDING
# =================================================================

@dataclass
class DetectiveFinding:
    """
    Structured result of one detective investigation.

    ``source_artifact_id`` is a stable identifier for the upstream
    artifact the chunk's problem is attributed to:

      * ``"req:<request_id>"``  - a specific tool result, keyed by the
                                  request id it was delivered under;
      * ``"agent:<name>"``      - a specific prior agent's output;
      * ``None``                - not attributable.

    ``requests_tracer`` is the detective's OWN decision (not automatic)
    about whether walking further back is warranted. ``tokens_used`` is
    charged against the proposed budget; ``budget_exhausted`` records
    that the ceiling was hit and the findings are partial.
    """

    attributable: bool
    source_artifact_id: Optional[str]
    requests_tracer: bool
    reasoning: str
    tokens_used: int
    budget_exhausted: bool = False


# =================================================================
# DETECTIVE AGENT
# =================================================================

class DetectiveAgent:
    """
    Attributes a judge-confirmed contradicting chunk to an upstream
    artifact, within a proposed token budget.

    The LLM is injectable via ``llm=`` (same pattern as
    ``LLMJudge(llm=...)``): any object exposing ``invoke(prompt)`` whose
    response has a ``.content`` attribute (or is itself a string) works,
    so tests use a deterministic stub with no backend.
    """

    def __init__(
        self,
        assessor: SemanticAssessor,
        llm: Optional[Any] = None,
        token_budget: int = DETECTIVE_TOKEN_BUDGET,
        linker: Optional[EvidenceLinker] = None,
        token_counter: Optional[Any] = None,
        link_top_k: int = DETECTIVE_LINK_TOP_K,
    ) -> None:
        self.assessor = assessor
        self._llm = llm
        self.token_budget = int(token_budget)
        self.link_top_k = int(link_top_k)

        # Reuse the shared encoder via the linker (no second model).
        self.linker = linker or EvidenceLinker(assessor)

        # Optional shared token counter (environment.resource_accounting
        # .LlamaTokenCounter). When absent, a deterministic word-count
        # estimate is used so the class is unit-testable offline.
        self.token_counter = token_counter

    # =========================================================
    # TOKEN ACCOUNTING (budget enforcement)
    # =========================================================

    def _count_tokens(self, text: str) -> int:
        if self.token_counter is not None:
            try:
                return int(self.token_counter.count(str(text)))
            except Exception:
                # Never let accounting break detection; fall back.
                pass
        # Deterministic offline estimate: ~1 token per whitespace word
        # (a deliberate over-estimate, so the budget is enforced
        # conservatively rather than under-counted).
        return len(str(text or "").split())

    # =========================================================
    # PUBLIC API
    # =========================================================

    def investigate(
        self,
        *,
        chunk_text: str,
        evidence_chunks: Sequence[str],
        evidence_artifacts: Sequence[Mapping[str, Any]],
        upstream_responses: Mapping[str, str],
        task: str,
    ) -> DetectiveFinding:
        """
        Investigate one judge-confirmed contradicting chunk.

        Returns a ``DetectiveFinding``. Never raises for budget reasons:
        exceeding the proposed budget yields PARTIAL findings with
        ``budget_exhausted=True``. Returns ``attributable=False`` when
        there is nothing to attribute the chunk to.
        """

        chunk_text = str(chunk_text or "")

        # ---- Build ranked evidence candidates (reuse the linker) ----
        #
        # Prefer the artifacts delivered TO the agent (a tool result or
        # a message, keyed by request_id / message_id); fall back to the
        # evidence text chunks when no structured artifacts are supplied.
        candidates: list[EvidenceCandidate] = self.linker.candidates_from_artifacts(
            evidence_artifacts
        )
        if not candidates:
            candidates = [
                EvidenceCandidate(text=str(text), source="artifact:evidence", index=i)
                for i, text in enumerate(evidence_chunks or [])
                if str(text).strip()
            ]
        # Also consider earlier-stage agents' outputs as attributions.
        candidates += self.linker.candidates_from_agent_responses(
            upstream_responses or {}
        )

        if not candidates:
            return DetectiveFinding(
                attributable=False,
                source_artifact_id=None,
                requests_tracer=False,
                reasoning="no linked evidence to attribute the chunk to",
                tokens_used=0,
            )

        links = self.linker.link(chunk_text, candidates, top_k=self.link_top_k)
        if not links:
            return DetectiveFinding(
                attributable=False,
                source_artifact_id=None,
                requests_tracer=False,
                reasoning="no linked evidence similarity to attribute the chunk to",
                tokens_used=0,
            )

        # ---- Budget check BEFORE the expensive LLM call ----
        prompt = self._build_prompt(
            chunk_text=chunk_text,
            task=task,
            links=links,
        )
        prompt_tokens = self._count_tokens(prompt)

        if prompt_tokens > self.token_budget:
            # Stop and report partial findings rather than overspend.
            logger.warning(
                "detective_budget_exhausted prompt_tokens=%s budget=%s",
                prompt_tokens,
                self.token_budget,
            )
            return DetectiveFinding(
                attributable=False,
                source_artifact_id=None,
                requests_tracer=False,
                reasoning=(
                    "budget exhausted before attribution: prompt needs "
                    f"{prompt_tokens} tokens, budget is {self.token_budget}"
                ),
                tokens_used=prompt_tokens,
                budget_exhausted=True,
            )

        # ---- Attribution via the (possibly stubbed) LLM ----
        raw = self._invoke_llm(prompt)
        completion_tokens = self._count_tokens(raw)
        total_tokens = prompt_tokens + completion_tokens

        # Enforce the ceiling on the TOTAL usage too (a verbose model
        # must not push us over by answering). Report partial findings.
        if total_tokens > self.token_budget:
            logger.warning(
                "detective_budget_exhausted total_tokens=%s budget=%s",
                total_tokens,
                self.token_budget,
            )
            return DetectiveFinding(
                attributable=False,
                source_artifact_id=None,
                requests_tracer=False,
                reasoning=(
                    "budget exhausted during attribution: used "
                    f"{total_tokens} tokens, budget is {self.token_budget}"
                ),
                tokens_used=total_tokens,
                budget_exhausted=True,
            )

        return self._parse_finding(raw, links=links, tokens_used=total_tokens)

    # =========================================================
    # LLM PROMPT / CALL
    # =========================================================

    def _build_prompt(
        self,
        *,
        chunk_text: str,
        task: str,
        links: Sequence[Any],
    ) -> str:
        """
        Build the narrow, structured attribution prompt.

        Deliberately asks for a small fixed set of fields (SOURCE /
        NONCE / TRACE / REASONING) rather than an open-ended
        "investigate this", so the reply stays parseable.
        """

        lines = []
        for link in links:
            candidate = link.candidate
            cid = self._candidate_id(candidate)
            lines.append(f"[{cid}] {candidate.text[:400]}")
        evidence_block = "\n\n".join(lines) if lines else "(none)"

        return (
            "You are a detective agent in a defensive multi-agent system.\n"
            "A chunk of an agent's output has been CONFIRMED to "
            "contradict its evidence. Your job: decide whether that "
            "problem is attributable to ONE of the upstream items below.\n\n"
            f"TASK:\n{task}\n\n"
            f"CONFIRMED CHUNK:\n{chunk_text}\n\n"
            f"UPSTREAM ITEMS (candidate sources):\n{evidence_block}\n\n"
            "Reply with EXACTLY these four lines and nothing else:\n"
            "SOURCE: <the [id] of the upstream item responsible, or NONE>\n"
            "TRACE: YES or NO (should a root-cause tracer walk further "
            "back from that item?)\n"
            "REASONING: <one short sentence>"
        )

    def _invoke_llm(self, prompt: str) -> str:
        if self._llm is None:
            # No LLM configured: return a conservative NONE so the
            # layer fails closed (never fabricates an attribution).
            return "SOURCE: NONE\nTRACE: NO\nREASONING: no llm configured"
        response = self._llm.invoke(prompt)
        return getattr(response, "content", str(response))

    @staticmethod
    def _candidate_id(candidate: EvidenceCandidate) -> str:
        """
        Stable identifier for one candidate, matching the finding's
        ``source_artifact_id`` convention:

          * ``"req:<request_id>"``  for a tool result with a request id;
          * ``"agent:<name>"``      for a prior agent's response;
          * ``"artifact:<type>#<i>"`` otherwise (artifact without a
                                      request id), a non-attributable
                                      but stable handle.
        """

        metadata = candidate.metadata or {}
        request_id = metadata.get("request_id")
        if candidate.is_artifact and request_id:
            return f"req:{request_id}"
        if candidate.source.startswith("agent:"):
            return candidate.source
        artifact_type = candidate.source.split(":", 1)[-1]
        return f"artifact:{artifact_type}#{candidate.index}"

    # =========================================================
    # REPLY PARSING
    # =========================================================

    def _parse_finding(
        self,
        raw: str,
        *,
        links: Sequence[Any],
        tokens_used: int,
    ) -> DetectiveFinding:
        """
        Parse the structured reply. Conservative: an unparseable or
        NONE source yields ``attributable=False`` (never fabricate an
        attribution).
        """

        source_value = ""
        trace_value = ""
        reasoning = ""

        for line in (raw or "").splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("SOURCE:"):
                source_value = stripped.split(":", 1)[1].strip()
            elif upper.startswith("TRACE:"):
                trace_value = stripped.split(":", 1)[1].strip().upper()
            elif upper.startswith("REASONING:"):
                reasoning = stripped.split(":", 1)[1].strip()

        if not source_value or source_value.upper() == "NONE":
            return DetectiveFinding(
                attributable=False,
                source_artifact_id=None,
                requests_tracer=False,
                reasoning=reasoning or "no upstream source identified",
                tokens_used=tokens_used,
            )

        # Map the chosen [id] back to a candidate id; keep the raw value
        # if it does not match (the model may echo an id verbatim).
        valid_ids = {self._candidate_id(link.candidate) for link in links}
        clean = source_value.strip("[]").strip()
        # Accept an exact id, or an id the model wrapped in [..].
        resolved = clean if clean in valid_ids else None
        if resolved is None:
            # Try matching a bracketed occurrence inside the raw value.
            for candidate_id in valid_ids:
                if candidate_id in source_value:
                    resolved = candidate_id
                    break

        if resolved is None:
            return DetectiveFinding(
                attributable=False,
                source_artifact_id=None,
                requests_tracer=False,
                reasoning=(
                    reasoning
                    or f"cited source {source_value!r} is not a known "
                    "upstream item"
                ),
                tokens_used=tokens_used,
            )

        return DetectiveFinding(
            attributable=True,
            source_artifact_id=resolved,
            requests_tracer=trace_value.startswith("YES"),
            reasoning=reasoning or "attributed to a specific upstream item",
            tokens_used=tokens_used,
        )