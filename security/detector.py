"""
Observable-evidence detector for the MAS security state.

This detector does NOT read simulator attack labels. It produces
evidence from observable content and behaviour only:

  * content evidence (prompt-injection style phrasing, unexpected
    URLs/emails/commands) via security.content_detector,
  * behavioural signals (tool timeouts, tool-call volume, message
    relay fan-out, memory writes from untrusted sources, per-agent
    token spikes).

``evidence_present`` is True when content/behaviour evidence is
present. It is NOT a claim that the system is compromised.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .content_detector import (
    STRICT_CATEGORIES,
    ContentEvidence,
    detect_artifact_evidence,
    detect_content_evidence,
    introduced_urls,
    scan_text,
)

# Default location of the per-agent clean-run token baseline
# (mean/std), produced by experiments/calibrate_detector.py. Until
# this file exists, the token-spike feature reports 0 and is marked
# "uncalibrated" rather than firing on an arbitrary rule of thumb
# (P4: the old mean+2std-over-current-episode rule could not
# mathematically fire with only 5 agents).
DEFAULT_TOKEN_BASELINE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "token_baseline.json"
)


class SecurityDetector:
    """
    Lightweight observable-evidence detector for the MAS security
    state.

    Evidence is derived from content and behaviour, never from
    simulator ground truth.

    By default (``use_template_signatures=False``) only the
    attack-agnostic GENERIC_PATTERNS are scanned (see
    ``security.content_detector``). Passing
    ``use_template_signatures=True`` additionally enables the
    verbatim phrases copied from the prompt-infection attack payload;
    this is a signature-based, template-aware detector and results
    produced with it MUST be labelled as such (P1: grading a detector
    against ground truth built from its own phrase list is circular).
    """

    def __init__(
        self,
        use_template_signatures: bool = False,
        token_baseline_path: str | Path | None = None,
        token_zscore_threshold: float = 3.0,
        count_alert_events: bool = False,
    ) -> None:
        self.use_template_signatures = use_template_signatures
        self.token_zscore_threshold = token_zscore_threshold
        self.token_baseline = self._load_token_baseline(
            token_baseline_path or DEFAULT_TOKEN_BASELINE_PATH
        )
        # AlertSensor (security.alert_sensor.AlertSensor) emits
        # observable `alert` events for ablation studies, but reads
        # ground truth internally to decide when to fire (P4). The
        # default detector therefore ignores `alert` events entirely;
        # this flag opts a detector instance into counting them as a
        # separate evidence source, for ablations only (see
        # experiments/ablate_signals.py's "fused+alert" condition).
        self.count_alert_events = count_alert_events

    @staticmethod
    def _load_token_baseline(path: str | Path) -> dict[str, dict[str, float]]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    @property
    def mode_label(self) -> str:
        return (
            "signature-based, template-aware"
            if self.use_template_signatures
            else "generic"
        )

    MESSAGE_EVENT_TYPES = {
        "message",
        "message_relay",
    }

    # Tool-event types that count toward tool-call volume.
    #
    # P4: request_tool() logs exactly one "tool_request" event per
    # call, regardless of routing branch; "tool_execution" was ALSO
    # counted previously, double-counting every call that reached
    # execution. Only "tool_request" is counted now.
    TOOL_CALL_EVENT_TYPES = {
        "tool_request",
    }

    TIMEOUT_EVENT_TYPES = {
        "tool_timeout",
        "timeout",
    }

    # Trusted local sources. Memory writes from anything else are
    # treated as coming from an untrusted/external source.
    TRUSTED_SOURCES = {
        "coordinator",
        "outline",
        "researcher",
        "analyst",
        "executor",
        "environment",
    }

    def detect(
        self,
        events: Iterable[Mapping[str, Any]],
        tool_limit: int | None = None,
        artifacts: Iterable[Mapping[str, Any]] | None = None,
        response: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute observable content and behaviour evidence.

        ``artifacts`` (see
        ``MASEnvironment.get_observable_artifacts``), when given, are
        the preferred evidence source (P2): they carry the actual
        text an agent received, rather than the compact summaries
        stored in ``events``. When omitted, falls back to scanning
        the ``content``/``memory_update`` fields of ``events``
        (legacy path, mostly a no-op since those fields are summaries).

        ``response`` (P5), when given, is scanned directly as the
        agent's own output. It is agent-generated prose, so only
        ``STRICT_CATEGORIES`` (instruction_override, role_reassignment)
        are applied to it -- the stronger exfiltration/imperative
        categories are reserved for untrusted artifacts (P4). To avoid
        double counting, ``response`` is skipped when its text is
        byte-identical to one of the supplied ``artifacts`` (e.g. the
        response was itself logged as a memory-write artifact).
        ``unexpected_url_count`` becomes the number of URLs in
        ``response`` that do not appear in any supplied artifact
        (introduced URLs, P4) when ``response`` is given.

        Returns counts plus ``evidence_present`` (with a deprecated
        ``detected`` alias) and the agent ids that triggered content
        evidence (``affected_agents``).
        """

        events = list(events)
        artifacts = list(artifacts) if artifacts is not None else None

        event_types = Counter(
            event.get("event_type")
            for event in events
        )

        # -----------------------------------------------------
        # Content evidence
        # -----------------------------------------------------

        evidence_by_receiver: dict[str, ContentEvidence] = {}

        if artifacts is not None:
            (
                content,
                untrusted_source_evidence_count,
                evidence_by_receiver,
            ) = detect_artifact_evidence(
                artifacts,
                use_template_signatures=self.use_template_signatures,
            )
        else:
            content = detect_content_evidence(
                events,
                use_template_signatures=self.use_template_signatures,
            )
            # Legacy fallback: memory writes from a sender outside
            # the trusted agent set. In practice always 0, since
            # every memory_write caller uses a trusted agent name
            # (P2) -- superseded by the artifact path above.
            untrusted_source_evidence_count = sum(
                1
                for event in events
                if event.get("event_type") == "memory_write"
                and event.get("sender") not in self.TRUSTED_SOURCES
            )

        # -----------------------------------------------------
        # Response evidence (P5): the agent's own output, scanned
        # directly rather than relying on it appearing in `events`.
        # -----------------------------------------------------

        artifact_texts = [
            artifact.get("text")
            for artifact in (artifacts or [])
            if isinstance(artifact.get("text"), str)
        ]

        if response and isinstance(response, str):

            artifact_hashes = {
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in artifact_texts
            }
            response_hash = hashlib.sha256(
                response.encode("utf-8")
            ).hexdigest()

            if response_hash not in artifact_hashes:

                response_evidence = scan_text(
                    response,
                    use_template_signatures=self.use_template_signatures,
                    categories=STRICT_CATEGORIES,
                )

                for category, count in response_evidence.category_counts.items():
                    content.category_counts[category] = (
                        content.category_counts.get(category, 0) + count
                    )
                content.high_confidence += response_evidence.high_confidence

            unexpected_url_count = len(
                introduced_urls(response, artifact_texts)
            )
        else:
            unexpected_url_count = content.url_count

        injection_evidence_count = content.category_counts.get(
            "instruction_override",
            0,
        )

        # -----------------------------------------------------
        # Behavioural evidence
        # -----------------------------------------------------

        timeout_count = sum(
            event_types[event_type]
            for event_type in self.TIMEOUT_EVENT_TYPES
        )

        tool_call_count = sum(
            event_types[event_type]
            for event_type in self.TOOL_CALL_EVENT_TYPES
        )

        relay_count = event_types["message_relay"]

        message_count = sum(
            event_types[event_type]
            for event_type in self.MESSAGE_EVENT_TYPES
        )

        # Relay fan-out (P4 redefinition): relay events whose
        # (sender, receiver) hop is off the topology's expected
        # route. A relay's own metadata records the coordinator-
        # computed path it should follow (see
        # MASEnvironment.publish_agent_result); a relay whose
        # receiver is not on its own declared path is off-route.
        # Previously this counted `relay_count - message_count`,
        # which is always <= 0 since message_count already includes
        # every message_relay event (always reported 0, P4).
        relay_fanout = sum(
            1
            for event in events
            if self._is_off_route_relay(event)
        )

        # Memory writes from untrusted / external sources.
        # (superseded by untrusted_source_evidence_count computed
        # above when artifacts are supplied; recomputed here only for
        # the legacy, artifact-less path so the variable always
        # exists.)

        # Per-agent token spikes (P4): per-agent z-score against a
        # clean-run baseline (mean/std) loaded from configs/, instead
        # of a mean+2std rule over the current episode's handful of
        # agents (which could not mathematically fire with 5 agents,
        # since max z = sqrt(n-1) < 2 for n <= 4).
        tokens_by_agent: dict[str, int] = {}
        for event in events:

            usage = event.get("token_usage") or 0
            agent = event.get("sender") or event.get("agent_id")

            if agent and usage:
                tokens_by_agent[agent] = (
                    tokens_by_agent.get(agent, 0) + int(usage)
                )

        token_spike_count, token_spike_status = self._count_token_spikes(
            tokens_by_agent
        )

        # Tool-call volume spike vs budget.
        tool_volume_spike = 0
        if tool_limit:
            tool_volume_spike = int(
                tool_call_count > tool_limit
            )

        # Alert-sensor evidence (opt-in only, see __init__; the
        # sensor is ground-truth-derived internally and must never
        # be part of the default, non-ablation detector).
        alert_evidence_count = event_types.get("alert", 0)

        # -----------------------------------------------------
        # Agents that triggered content evidence
        # -----------------------------------------------------

        if artifacts is not None:
            affected_agents = set(evidence_by_receiver.keys())
        else:
            affected_agents = {
                event.get("sender") or event.get("agent_id")
                for event in events
                if self._event_has_content_evidence(
                    event,
                    use_template_signatures=self.use_template_signatures,
                )
            }
            affected_agents.discard(None)

        # -----------------------------------------------------
        # evidence_present
        # -----------------------------------------------------

        evidence_present = bool(
            content.total
            or timeout_count
            or untrusted_source_evidence_count
            or token_spike_count
            or tool_volume_spike
            or (self.count_alert_events and alert_evidence_count)
        )

        return {
            # Evidence counts (renamed from label-based features)
            "injection_evidence_count": injection_evidence_count,
            "high_confidence_evidence_count": content.high_confidence,
            "untrusted_source_evidence_count": (
                untrusted_source_evidence_count
            ),
            "content_evidence_count": content.total,
            "content_category_counts": dict(
                content.category_counts
            ),

            # Behavioural evidence
            "tool_timeout_count": timeout_count,
            "tool_call_count": tool_call_count,
            "relay_fanout": relay_fanout,
            "token_spike_count": token_spike_count,
            "token_spike_status": token_spike_status,
            "tool_volume_spike": tool_volume_spike,

            # Alert-sensor evidence: always reported for visibility,
            # but only folded into `evidence_present` when
            # `count_alert_events=True` (ablation only, off by
            # default -- see __init__).
            "alert_evidence_count": alert_evidence_count,
            "alert_evidence_enabled": self.count_alert_events,

            # Unexpected artefacts
            "unexpected_url_count": unexpected_url_count,
            "unexpected_email_count": content.email_count,
            "unexpected_command_count": content.command_count,

            # Propagation / messaging
            "message_count": message_count,
            "relay_count": relay_count,

            # Decision
            "evidence_present": evidence_present,
            "affected_agents": sorted(affected_agents),

            # Labels the pattern set used, so reports cannot silently
            # mix generic and signature-based results (P1).
            "detector_mode": self.mode_label,

            # Deprecated alias kept for backward compatibility.
            "detected": evidence_present,
        }

    @staticmethod
    def _is_off_route_relay(event: Mapping[str, Any]) -> bool:
        """
        True when a ``message_relay`` event's receiver is off its own
        declared route (P4).

        Each relay hop records the coordinator-computed path it is
        part of in ``metadata["path"]`` (see
        ``MASEnvironment.publish_agent_result``). A relay is off-route
        when its receiver does not appear on that path at all --
        which never happens for a normal, coordinator-routed relay,
        but is exactly the anomaly this feature is meant to catch.
        """

        if event.get("event_type") != "message_relay":
            return False

        receiver = event.get("receiver")
        if not receiver:
            return False

        path = event.get("metadata", {}).get("path") or []
        return receiver not in path

    def _count_token_spikes(
        self,
        tokens_by_agent: Mapping[str, int],
    ) -> tuple[int, str]:
        """
        Count agents whose token usage is a strong outlier against a
        clean-run baseline (P4).

        Uses a per-agent z-score against ``self.token_baseline``
        (mean/std per agent, produced by
        ``experiments/calibrate_detector.py`` from clean episodes and
        stored in configs/token_baseline.json). Until that baseline
        exists, this always returns ``(0, "uncalibrated")`` rather
        than firing on an ad hoc rule -- the previous mean+2std rule
        over the current episode's handful of agents could not
        mathematically exceed threshold with n <= 5 (max z =
        sqrt(n-1)).
        """

        if not self.token_baseline:
            return 0, "uncalibrated"

        count = 0
        for agent, total in tokens_by_agent.items():

            stats = self.token_baseline.get(agent)
            if not stats or not stats.get("std"):
                continue

            z_score = (total - stats["mean"]) / stats["std"]
            if z_score > self.token_zscore_threshold:
                count += 1

        return count, "calibrated"

    @staticmethod
    def _event_has_content_evidence(
        event: Mapping[str, Any],
        use_template_signatures: bool = False,
    ) -> bool:
        """
        True when this single event's content triggers evidence.
        """

        content = event.get("content")

        if isinstance(content, Mapping):
            content = content.get("content")

        if not isinstance(content, str) or not content.strip():
            content = event.get("memory_update")

        if not isinstance(content, str) or not content.strip():
            return False

        return bool(
            detect_content_evidence(
                [{"content": content}],
                use_template_signatures=use_template_signatures,
            ).total
        )
