"""
streamlit_app.py

LIVE monitoring dashboard for the adaptive-mas-defense experiment.

THIS FILE ONLY READS. It never imports the environment, never calls
``execute_task`` / ``graph.invoke`` / ``main``, and never starts or
restarts an episode. It tails the append-only event file written by the
running experiment (security/live_events.py) and renders it.

Start the dashboard FIRST, then run the experiment in another terminal:

    # terminal 1 (dashboard)
    streamlit run streamlit_app.py

    # terminal 2 (experiment)
    python main.py

Refreshing the dashboard re-reads the file from a tracked byte offset, so
it never duplicates events and never touches the experiment.

Every field shown here comes from an event the experiment actually
emitted. Nothing is fabricated and nothing is recomputed here: the
per-chunk Tier 1/2/3 results are produced by the existing
``SecurityObserver`` inside main.py and simply DISPLAYED.

Chunk-level rendering
---------------------
The experiment publishes each agent response's already-computed
per-chunk results inside the ``security_observation`` event, under
``payload.tiered.chunks`` (see
``MASEnvironment._serialize_tiered_chunks``). Each chunk entry carries

  * its Tier 1 semantic results for TWO separate reference axes
    (``original_task_*`` -- primary -- and ``assigned_subtask_*`` --
    secondary),
  * its Tier 2 evidence-NLI result (or an explicit "not run" marker),
  * its two Tier 2 TASK-ALIGNMENT NLI results, kept in separate fields:
    ``tier2_original_task_nli`` (premise = original user prompt, primary)
    and ``tier2_assigned_subtask_nli`` (premise = assigned subtask,
    secondary; NOT RUN with a reason when no subtask exists),
  * its Tier 3 judge verdict (or "not run").

This module renders exactly those values -- it never recomputes semantic
similarity or NLI, and it never presents one comparison as if it were
the other.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from security.live_events import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_RUN_MARKER_PATH,
    read_all_events,
    read_run_marker,
)

EVENTS_PATH = Path(DEFAULT_EVENTS_PATH)
MARKER_PATH = Path(DEFAULT_RUN_MARKER_PATH)

# Event types that belong in the "investigation / defense" view.
INVESTIGATION_EVENT_TYPES = {
    "investigation",
    "containment",
    "resource_allocation",
}
# Event types that indicate a tool interaction.
TOOL_EVENT_TYPES = {
    "tool_request",
    "tool_execution",
    "tool_result",
    "tool_result_delivery",
    "tool_error",
    "tool_timeout",
    "tool_denied",
    "tool_forward",
    "tool_rejected",
    "tool_usage",
}
# Event types that indicate an agent lifecycle / activity signal.
AGENT_EVENT_TYPES = {
    "agent_result",
    "task_decomposition",
    "memory_write",
    "memory_read",
    "message",
    "message_receive",
    "message_relay",
    "llm_usage",
    "final_result",
    "report_created",
}
# Resource-relevant event types.
RESOURCE_EVENT_TYPES = {
    "llm_usage",
    "resource_reset",
    "tool_usage",
    "tool_timeout",
    "llm_rejected",
}


# =================================================================
# EVENT HELPERS (pure)
# =================================================================

def _payload_of(event: dict) -> dict:
    return event.get("payload") or {}


def _event_ts(event: dict) -> str:
    return str(event.get("ts") or _payload_of(event).get("timestamp") or "")


def is_mas_event(event: dict) -> bool:
    return event.get("event_type") == "mas_event"


def mas_kind(event: dict) -> str:
    """The MAS event_type for a mas_event wrapper, else the wrapper type."""
    if is_mas_event(event):
        return str(_payload_of(event).get("event_type") or "mas_event")
    return str(event.get("event_type") or "")


def _fmt(value, digits: int = 4) -> str:
    """Format a float for display, or '—' when absent (never fabricated)."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _axis_value(sem: dict, axis: str, suffix: str) -> str:
    """
    One Tier-1 axis value for the reference-axes table.

    Shows the published number when that axis actually ran, else an
    explicit ``NOT_RUN``/``N/A`` marker -- never the other axis' score.
    """

    if not sem.get(f"{axis}_assessed"):
        return "N/A (NOT_RUN)"
    value = sem.get(f"{axis}_{suffix}")
    if value is None:
        return "N/A (NOT_RUN)"
    return _fmt(value)


def _render_reference_nli(
    title: str,
    block: dict,
    *,
    premise_label: str,
) -> None:
    """
    Render ONE task-alignment NLI family (premise/hypothesis/label/conf).

    Every value comes verbatim from the published event; a family that did
    not run is shown as NOT RUN with the recorded reason instead of a
    fabricated label.
    """

    st.markdown(f"**{title}**")

    status = alignment_nli_status(block)
    if status == "not_run":
        reason = block.get("not_run_reason") or "reason not recorded"
        st.info(f"{title}: NOT RUN — {reason}")
        return

    badge = "REAL NLI" if status == "real" else "STUB NLI"
    st.markdown(f"`{badge}`")
    st.markdown(f"**Label:** {block.get('label')}")
    st.markdown(f"**Confidence:** {_fmt(block.get('confidence'))}")
    st.markdown(f"**Premise ({premise_label}):**")
    st.caption(str(block.get("premise") or "")[:800])
    st.markdown("**Hypothesis (extracted response claim):**")
    st.caption(str(block.get("hypothesis") or "")[:800])


def build_timeline_rows(events: list[dict]) -> list[dict]:
    """Flatten events into display rows (timestamp, kind, actors, text)."""

    rows: list[dict] = []
    for event in events:
        wrapper = event.get("event_type")
        payload = _payload_of(event)

        if wrapper == "mas_event":
            kind = str(payload.get("event_type") or "mas_event")
            rows.append(
                {
                    "timestamp": _event_ts(event),
                    "kind": kind,
                    "sender": payload.get("sender") or "",
                    "receiver": payload.get("receiver") or "",
                    "tool": payload.get("tool_call") or "",
                    "tokens": payload.get("token_usage") or 0,
                    "text": payload.get("content") or "",
                }
            )
        elif wrapper == "security_observation":
            rows.append(
                {
                    "timestamp": _event_ts(event),
                    "kind": "security_observation",
                    "sender": payload.get("agent_id") or "",
                    "receiver": "",
                    "tool": "",
                    "tokens": 0,
                    "text": (
                        f"security_score="
                        f"{payload.get('security_score')} "
                        f"investigate={payload.get('investigation_required')}"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "timestamp": _event_ts(event),
                    "kind": str(wrapper or ""),
                    "sender": "",
                    "receiver": "",
                    "tool": "",
                    "tokens": 0,
                    "text": _summarise_wrapper(wrapper, payload),
                }
            )
    return rows


def _summarise_wrapper(wrapper: str, payload: dict) -> str:
    if wrapper == "experiment_started":
        return (
            f"task={payload.get('task')!r} "
            f"topology={payload.get('topology')}"
        )
    if wrapper == "episode_started":
        return (
            f"episode={payload.get('episode_id')} "
            f"agents={payload.get('agents')}"
        )
    if wrapper == "episode_completed":
        return (
            f"episode={payload.get('episode_id')} "
            f"observed={payload.get('agent_count')}"
        )
    if wrapper == "episode_failed":
        return f"episode={payload.get('episode_id')} error={payload.get('error')}"
    if wrapper == "experiment_completed":
        return "run completed"
    if wrapper == "experiment_failed":
        return f"run failed: {payload.get('error')}"
    return str(payload)


def latest_resource_state(events: list[dict]) -> dict:
    """
    Derive the latest resource picture from the events the environment
    actually emits. Uses llm_usage / resource_reset metadata only.
    """

    state = {
        "tokens_used": None,
        "token_limit": None,
        "tokens_by_agent": {},
        "status": None,
        "tool_denied": 0,
        "tool_timeouts": 0,
        "tool_usage": 0,
        "llm_calls": 0,
    }

    for event in events:
        wrapper = event.get("event_type")
        payload = _payload_of(event)
        kind = mas_kind(event)

        if kind == "llm_usage" or wrapper == "mas_event" and kind == "llm_usage":
            meta = payload.get("metadata") or {}
            if meta.get("tokens_used") is not None:
                state["tokens_used"] = meta.get("tokens_used")
                state["token_limit"] = meta.get("token_limit")
                state["status"] = meta.get("status")
                state["tokens_by_agent"] = dict(
                    meta.get("tokens_by_agent") or {}
                )
            state["llm_calls"] += 1
        elif kind == "resource_reset":
            meta = payload.get("metadata") or {}
            if meta.get("token_limit") is not None:
                state["token_limit"] = meta.get("token_limit")
        elif kind == "tool_timeout":
            state["tool_timeouts"] += 1
        elif kind == "tool_denied":
            state["tool_denied"] += 1
        elif kind == "tool_usage":
            state["tool_usage"] += 1

    return state


def agent_statuses(events: list[dict]) -> dict:
    """
    Derive a per-agent status from the event stream: the last activity
    kind and its timestamp, plus the latest security observation.
    """

    agents: dict[str, dict] = {}

    for event in events:
        wrapper = event.get("event_type")
        payload = _payload_of(event)

        if wrapper == "mas_event":
            kind = str(payload.get("event_type") or "")
            sender = payload.get("sender")
            if sender and (kind in AGENT_EVENT_TYPES or kind in TOOL_EVENT_TYPES
                           or kind in INVESTIGATION_EVENT_TYPES):
                entry = agents.setdefault(
                    sender, {"last_kind": "", "last_ts": "", "last_text": "",
                             "messages": [], "security": None}
                )
                entry["last_kind"] = kind
                entry["last_ts"] = _event_ts(event)
                text = payload.get("content")
                if text:
                    entry["last_text"] = text
                if kind in {"message", "message_receive", "message_relay"}:
                    entry["messages"].append(
                        {
                            "ts": _event_ts(event),
                            "to": payload.get("receiver") or "",
                            "text": text or "",
                        }
                    )
                    # keep the last few
                    entry["messages"] = entry["messages"][-8:]

        elif wrapper == "security_observation":
            agent_id = payload.get("agent_id")
            if agent_id:
                entry = agents.setdefault(
                    agent_id, {"last_kind": "", "last_ts": "", "last_text": "",
                               "messages": [], "security": None}
                )
                entry["security"] = payload

    return agents


def investigation_events(events: list[dict]) -> list[dict]:
    """Return investigation/containment/resource-allocation events."""

    out = []
    for event in events:
        if is_mas_event(event) and mas_kind(event) in INVESTIGATION_EVENT_TYPES:
            out.append(event)
    return out


# =================================================================
# CHUNK-LEVEL HELPERS (pure; display only, never recompute)
# =================================================================

def observation_chunks(observation: dict) -> list[dict]:
    """
    Return the already-computed chunk list for a security observation.

    Reads ``payload.tiered.chunks`` -- the per-chunk Tier 1/2/3 results
    the experiment published. Returns [] when the observation carries no
    chunk data (e.g. a run with the base observer only). No computation
    happens here.
    """

    tiered = (observation or {}).get("tiered") or {}
    chunks = tiered.get("chunks") or []
    return [c for c in chunks if isinstance(c, dict)]


def tier2_source_label(observation: dict) -> str:
    """
    One of: 'real', 'stub', 'not_run' -- how Tier 2 (NLI) was produced.

    Prefers the run-level ``tier2_source`` published on the observation;
    falls back to inspecting the individual chunk NLI entries.
    """

    tiered = (observation or {}).get("tiered") or {}
    source = tiered.get("tier2_source")
    if source in {"real", "stub", "not_run"}:
        return source

    chunk_sources = {
        (c.get("tier2_nli") or {}).get("source")
        for c in observation_chunks(observation)
    }
    chunk_sources.discard(None)
    if not chunk_sources:
        return "not_run"
    if "real" in chunk_sources:
        return "real"
    if "stub" in chunk_sources:
        return "stub"
    return "not_run"


def chunk_nli_status(chunk: dict) -> str:
    """'real' | 'stub' | 'not_run' for a single chunk's Tier 2 NLI."""

    nli = chunk.get("tier2_nli") or {}
    if not nli.get("ran"):
        return "not_run"
    return str(nli.get("source") or "not_run")


def chunk_judge_status(chunk: dict) -> str:
    """'real' | 'stub' | 'not_run' for a single chunk's Tier 3 judge."""

    judge = chunk.get("tier3_judge") or {}
    if not judge.get("ran"):
        return "not_run"
    return "stub" if judge.get("stubbed") else "real"


def all_chunks_by_agent(events: list[dict]) -> dict[str, list[dict]]:
    """
    Group every chunk from every security observation by agent, in the
    order the experiment emitted them. Each returned chunk dict is the
    published chunk plus a ``_observation`` reference (its agent id and
    the observation's timestamp) so the UI can show provenance without
    recomputing anything.
    """

    grouped: dict[str, list[dict]] = {}
    for event in events:
        if event.get("event_type") != "security_observation":
            continue
        payload = _payload_of(event)
        agent_id = payload.get("agent_id")
        if not agent_id:
            continue
        for chunk in observation_chunks(payload):
            entry = dict(chunk)
            entry["_agent_id"] = agent_id
            entry["_ts"] = _event_ts(event)
            entry["_security_score"] = payload.get("security_score")
            entry["_investigation_required"] = payload.get(
                "investigation_required"
            )
            grouped.setdefault(agent_id, []).append(entry)
    return grouped


def tier2_source_of_events(events: list[dict]) -> str:
    """
    Run-level Tier-2 (NLI) provenance across every observation in a run:
    'real' | 'stub' | 'not_run'.

    Used to show a single banner telling the user whether the run's NLI
    results are REAL, STUB, or NOT RUN -- without recomputing anything.
    """

    labels = set()
    for event in events:
        if event.get("event_type") != "security_observation":
            continue
        labels.add(tier2_source_label(_payload_of(event)))
    if not labels:
        return "not_run"
    if "real" in labels:
        return "real"
    if "stub" in labels:
        return "stub"
    return "not_run"


def tier3_status_of_events(events: list[dict]) -> str:
    """
    Run-level Tier-3 (LLM judge) status across every observation:
    'real' | 'stub' | 'not_run'.
    """

    statuses = set()
    for event in events:
        if event.get("event_type") != "security_observation":
            continue
        tiered = _payload_of(event).get("tiered") or {}
        status = tiered.get("tier3_status")
        if status in {"real", "stub", "not_run"}:
            statuses.add(status)
    if not statuses:
        return "not_run"
    if "real" in statuses:
        return "real"
    if "stub" in statuses:
        return "stub"
    return "not_run"


def _round6(value):
    """
    Round a published numeric field to 6 dp for DISPLAY.

    The event stream carries full float precision; the dashboard shows a
    stable, human-comparable value. ``None`` (the axis did NOT run) passes
    through unchanged, so a missing axis can never be rendered as 0.0 and
    mistaken for a real score.
    """

    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


def chunk_semantic_fields(chunk: dict) -> dict:
    """
    The Tier-1 semantic fields to display for a chunk, verbatim from the
    published event (never recomputed).

    TWO INDEPENDENT REFERENCE AXES are returned side by side:

      * original-task  (primary,  trusted reference)
      * assigned-subtask (secondary reference)

    The legacy fused keys are returned too, unchanged, so older events and
    older callers keep working.
    """

    sem = chunk.get("tier1_semantic") or {}

    return {
        "assessed": bool(sem.get("assessed")),
        # ---- PRIMARY axis: original user task ----
        "original_task_assessed": bool(sem.get("original_task_assessed")),
        "original_task_similarity": _round6(
            sem.get("original_task_similarity")
        ),
        "original_task_deviation": _round6(
            sem.get("original_task_deviation")
        ),
        "original_task_not_run_reason": (
            sem.get("original_task_not_run_reason") or ""
        ),
        # ---- SECONDARY axis: assigned subtask ----
        "assigned_subtask_assessed": bool(
            sem.get("assigned_subtask_assessed")
        ),
        "assigned_subtask_similarity": _round6(
            sem.get("assigned_subtask_similarity")
        ),
        "assigned_subtask_deviation": _round6(
            sem.get("assigned_subtask_deviation")
        ),
        "assigned_subtask_not_run_reason": (
            sem.get("assigned_subtask_not_run_reason") or ""
        ),
        # ---- legacy aliases (kept for older events/readers) ----
        "task_similarity": sem.get("task_similarity"),
        "subtask_similarity": sem.get("subtask_similarity"),
        "objective_deviation": sem.get("objective_deviation"),
        "scope_deviation": sem.get("scope_deviation"),
        "deviation_score": sem.get("deviation_score"),
        "confidence": sem.get("confidence"),
    }


# ---------------------------------------------------------------------
# TASK-ALIGNMENT NLI (Tier 2): TWO separate reference families
# ---------------------------------------------------------------------
#
# The published chunk carries two explicit, independent blocks:
#
#   tier2_original_task_nli      premise = original user prompt (PRIMARY)
#   tier2_assigned_subtask_nli   premise = assigned subtask    (SECONDARY)
#
# Neither is read from the other, and neither falls back to the evidence
# axis (``tier2_nli``), so the dashboard can never present one comparison
# as if it were the other.

_LEGACY_ALIGNMENT_REASON = (
    "not published for this chunk (older event schema)"
)


def _alignment_block(chunk: dict, key: str) -> dict:
    """
    Return one published task-alignment NLI block, or an explicit NOT RUN
    block when the event does not carry it. Never computes anything.
    """

    block = chunk.get(key)
    if not isinstance(block, dict):
        return {
            "ran": False,
            "source": "not_run",
            "not_run_reason": _LEGACY_ALIGNMENT_REASON,
            "claims": [],
        }
    return block


def chunk_original_task_nli(chunk: dict) -> dict:
    """
    Tier-2 ORIGINAL-TASK alignment NLI for a chunk (PRIMARY signal).

    Premise is the original user prompt; hypothesis is an extracted
    response claim of the same chunk.
    """

    return _alignment_block(chunk, "tier2_original_task_nli")


def chunk_assigned_subtask_nli(chunk: dict) -> dict:
    """
    Tier-2 ASSIGNED-SUBTASK alignment NLI for a chunk (SECONDARY signal).

    Premise is the assigned subtask; hypothesis is the SAME extracted
    response claim. Reads NOT RUN (with a reason) when no subtask existed.
    """

    return _alignment_block(chunk, "tier2_assigned_subtask_nli")


def alignment_nli_status(block: dict) -> str:
    """'real' | 'stub' | 'not_run' for one task-alignment NLI block."""

    if not block.get("ran"):
        return "not_run"
    return str(block.get("source") or "not_run")



def latest_event_time(events: list[dict]) -> str:
    """ISO timestamp of the most recent event, or '' when there are none."""

    return _event_ts(events[-1]) if events else ""


# =================================================================
# UI
# ================================================================

def _status_summary(marker: dict | None, events: list[dict]) -> dict:
    if marker is None:
        return {
            "status": "waiting",
            "detail": "No run marker yet -- start main.py to begin.",
            "last_ts": _event_ts(events[-1]) if events else "",
            "run_id": "",
            "started_at": "",
        }
    status = marker.get("status", "unknown")
    detail = {
        "running": "Experiment running.",
        "completed": "Experiment completed.",
        "failed": f"Experiment failed: {marker.get('error')}",
    }.get(status, f"Status: {status}")
    return {
        "status": status,
        "detail": detail,
        "last_ts": _event_ts(events[-1]) if events else "",
        "run_id": marker.get("run_id", ""),
        "started_at": marker.get("started_at", ""),
    }


def render_chunk_card(chunk: dict, *, newest: bool = False) -> None:
    """
    Render ONE processed response chunk and its already-computed
    Tier 1/2/3 results. Every value shown comes verbatim from the
    published event; nothing is recomputed here.
    """

    index = chunk.get("chunk_index")
    ts = str(chunk.get("_ts") or "")
    ts_short = ts[11:19] if ts else ""
    score = chunk.get("_security_score")

    header = f"Chunk {index}"
    if newest:
        header = f"🆕 {header}"
    if ts_short:
        header = f"{header} — {ts_short}"
    if score is not None:
        header = f"{header} (security_score={_fmt(score)})"

    with st.expander(header, expanded=newest):

        # -------- CHUNK / SECTION --------
        st.markdown("**Chunk / section**")
        if ts:
            st.caption(f"chunk timestamp: {ts}")
        text = chunk.get("chunk_text") or ""
        st.text_area(
            "Chunk text",
            value=str(text),
            height=160,
            key=f"chunk_text_{chunk.get('_agent_id')}_{index}_{ts}",
            disabled=True,
        )

        col1, col2, col3 = st.columns(3)

        # -------- TIER 1 — SEMANTIC ALIGNMENT --------
        with col1:
            st.markdown("**Tier 1 — Semantic Alignment**")
            sem = chunk_semantic_fields(chunk)
            if not sem["assessed"]:
                st.caption("semantic assessment did not run for this chunk")
            else:
                # Two INDEPENDENT reference axes, rendered as separate rows
                # so neither can be read as the other.
                st.markdown(
                    "**Reference axes** — original task (primary), "
                    "assigned subtask (secondary)"
                )
                st.markdown(
                    "| Metric | Value |\n"
                    "| --- | --- |\n"
                    f"| Original-task similarity | "
                    f"{_axis_value(sem, 'original_task', 'similarity')} |\n"
                    f"| Original-task deviation | "
                    f"{_axis_value(sem, 'original_task', 'deviation')} |\n"
                    f"| Assigned-subtask similarity | "
                    f"{_axis_value(sem, 'assigned_subtask', 'similarity')} |\n"
                    f"| Assigned-subtask deviation | "
                    f"{_axis_value(sem, 'assigned_subtask', 'deviation')} |"
                )
                if not sem["original_task_assessed"]:
                    st.caption(
                        "Original-task axis NOT RUN: "
                        + (
                            sem["original_task_not_run_reason"]
                            or "reason unavailable"
                        )
                    )
                if not sem["assigned_subtask_assessed"]:
                    st.caption(
                        "Assigned-subtask axis NOT RUN: "
                        + (
                            sem["assigned_subtask_not_run_reason"]
                            or "reason unavailable"
                        )
                    )
                # Legacy fused values, unchanged field names/format.
                st.caption(
                    f"task_similarity = {_fmt(sem['task_similarity'])}\n\n"
                    f"subtask_similarity = {_fmt(sem['subtask_similarity'])}\n\n"
                    f"objective_deviation = {_fmt(sem['objective_deviation'])}\n\n"
                    f"scope_deviation = {_fmt(sem['scope_deviation'])}\n\n"
                    f"deviation_score = {_fmt(sem['deviation_score'])}\n\n"
                    f"confidence = {_fmt(sem['confidence'])}"
                )

        # -------- TIER 2 — NLI TASK ALIGNMENT --------
        with col2:
            st.markdown("**Tier 2 — NLI Task Alignment**")

            # -- evidence axis (pre-existing; NOT a task-alignment premise).
            st.markdown("**Evidence contradiction (pre-existing axis)**")
            nli_status = chunk_nli_status(chunk)
            nli = chunk.get("tier2_nli") or {}
            if nli_status == "not_run":
                st.error("Tier 2 — NLI: NOT RUN")
            else:
                badge = "REAL NLI" if nli_status == "real" else "STUB NLI"
                st.markdown(f"`{badge}`")
                st.markdown(f"**NLI:** {nli.get('label')}")
                st.markdown(f"**Confidence:** {_fmt(nli.get('confidence'))}")
                st.markdown("**Premise:**")
                st.caption(str(nli.get("premise") or "")[:800])
                st.markdown("**Hypothesis:**")
                st.caption(str(nli.get("hypothesis") or "")[:800])

            st.divider()

            # -- the two SEPARATE task-alignment families.
            _render_reference_nli(
                "Original-Task NLI (Primary)",
                chunk_original_task_nli(chunk),
                premise_label="original user prompt",
            )
            _render_reference_nli(
                "Assigned-Subtask NLI (Secondary)",
                chunk_assigned_subtask_nli(chunk),
                premise_label="assigned subtask",
            )


        # -------- TIER 3 — JUDGE --------
        with col3:
            st.markdown("**Tier 3 — Judge**")
            judge_status = chunk_judge_status(chunk)
            judge = chunk.get("tier3_judge") or {}
            if judge_status == "not_run":
                if judge.get("enabled"):
                    st.info("Tier 3 — Judge: NOT RUN (enabled, gate not met)")
                else:
                    st.info("Tier 3 — Judge: NOT RUN (judge disabled)")
            else:
                badge = "REAL JUDGE" if judge_status == "real" else "STUB JUDGE"
                st.markdown(f"`{badge}`")
                st.markdown(
                    "**contradicts_evidence:** "
                    f"{judge.get('contradicts_evidence')}"
                )
                if judge.get("cached"):
                    st.caption("cached verdict")
                st.markdown("**reasoning:**")
                st.caption(str(judge.get("reasoning") or "")[:800])


def _run_status_section(summary: dict, events: list[dict]) -> None:
    st.subheader("1. Run status")
    cols = st.columns(4)
    status_colour = {
        "running": "🟢",
        "completed": "✅",
        "failed": "🔴",
        "waiting": "⚪",
    }.get(summary["status"], "⚪")
    cols[0].metric("Status", f"{status_colour} {summary['status']}")
    cols[1].metric("Run ID", summary["run_id"] or "—")
    cols[2].metric(
        "Started at",
        summary["started_at"][11:19] if summary["started_at"] else "—",
    )
    cols[3].metric(
        "Latest event",
        summary["last_ts"][11:19] if summary["last_ts"] else "—",
    )
    st.caption(summary["detail"])

    # Run-level NLI provenance banner: tells the user plainly whether the
    # NLI results they see are REAL, STUB, or NOT RUN, and whether the
    # Tier-3 judge is enabled for this run.
    nli_source = tier2_source_of_events(events)
    judge_status = tier3_status_of_events(events)
    if nli_source == "not_run":
        st.warning(
            "Tier 2 — NLI is NOT RUN for this run. Either the observer was "
            "built without a contradiction checker, or no linked evidence "
            "was delivered to any agent, so NLI had nothing to check "
            "against. Per-chunk NLI will read 'NOT RUN' below."
        )
    elif nli_source == "stub":
        st.info("Tier 2 — NLI is running with a STUB model (not real NLI).")
    else:
        st.success("Tier 2 — NLI is running with the REAL model.")

    if judge_status == "not_run":
        st.caption(
            "Tier 3 — Judge: NOT RUN for this run. The judge is gated; "
            "set MAS_TIER3_JUDGE=1 to enable it."
        )
    else:
        st.caption(f"Tier 3 — Judge status: {judge_status.upper()}")


def _agent_summary_section(events: list[dict]) -> None:
    st.subheader("2. Agent summary")
    agents = agent_statuses(events)
    if not agents:
        st.write("No agent activity yet.")
        return
    for name, entry in sorted(agents.items()):
        sec = entry.get("security") or {}
        last_ts = entry["last_ts"][11:19] if entry["last_ts"] else "—"
        st.markdown(
            f"**{name}** — activity: `{entry['last_kind'] or '—'}` @ {last_ts} "
            f"— security_score: `{_fmt(sec.get('security_score'))}` "
            f"— investigation_required: `{sec.get('investigation_required')}`"
        )
        if entry["last_text"]:
            st.caption(str(entry["last_text"])[:300])


def _live_chunk_stream_section(events: list[dict]) -> None:
    st.subheader("3. Live chunk stream")
    st.caption(
        "Each card is one processed response chunk with its already-computed "
        "Tier 1 / Tier 2 / Tier 3 results. Newest chunks are shown first and "
        "marked 🆕. Nothing here is recomputed by the dashboard."
    )
    grouped = all_chunks_by_agent(events)
    if not grouped:
        st.write(
            "No chunk-level results yet. They appear as soon as main.py "
            "publishes a security observation for a processed response."
        )
        return
    for agent_id, chunks in sorted(grouped.items()):
        st.markdown(f"### Agent: {agent_id}")
        # Newest processed chunk first, so it is easy to identify live.
        for position, chunk in enumerate(reversed(chunks)):
            render_chunk_card(chunk, newest=(position == 0))


def _resources_section(events: list[dict]) -> None:
    st.subheader("4. Resource usage")
    res = latest_resource_state(events)
    c = st.columns(3)
    tokens_used = res["tokens_used"]
    token_limit = res["token_limit"]
    remaining = (
        max(0, token_limit - tokens_used)
        if (token_limit is not None and tokens_used is not None)
        else None
    )
    c[0].metric("Tokens used", tokens_used if tokens_used is not None else "—")
    c[1].metric("Token limit", token_limit if token_limit is not None else "—")
    c[2].metric("Remaining budget", remaining if remaining is not None else "—")

    c2 = st.columns(3)
    c2[0].metric("LLM calls", res["llm_calls"])
    c2[1].metric("Tool usage events", res["tool_usage"])
    c2[2].metric(
        "Timeouts / denied", f"{res['tool_timeouts']} / {res['tool_denied']}"
    )

    if res["tokens_by_agent"]:
        st.write("**Tokens by agent**")
        for agent, tokens in sorted(res["tokens_by_agent"].items()):
            st.caption(f"{agent}: {tokens}")

    st.divider()
    st.write("**Investigation / containment / allocation events**")
    inv = investigation_events(events)
    if not inv:
        st.write(
            "No investigation/containment/allocation events emitted in this "
            "run. (These appear only if the environment emits them.)"
        )
    else:
        for event in inv:
            payload = _payload_of(event)
            st.markdown(
                f"`{_event_ts(event)[11:19]}` **{mas_kind(event)}** "
                f"{payload.get('sender') or ''}"
            )
            st.caption(
                str(payload.get("content") or payload.get("metadata") or "")
            )


def _raw_event_stream_section(events: list[dict]) -> None:
    st.subheader("5. Raw event stream")
    st.caption(
        "The actual emitted event JSON, for debugging. "
        f"{len(events)} event(s) in this run."
    )
    with st.expander(f"Raw events ({len(events)})"):
        for event in reversed(events[-200:]):
            st.json(event)


def render() -> None:
    st.set_page_config(
        page_title="MAS Defense - Live Monitor",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("🛡️ Adaptive MAS Defense — Live Monitor")
    st.caption(
        "Read-only view of the experiment started by main.py. "
        "Refreshing never starts or restarts an episode."
    )

    marker = read_run_marker(MARKER_PATH)
    events = read_all_events(EVENTS_PATH)

    # Only events from the CURRENT run are shown as live.
    if marker is not None and marker.get("run_id"):
        events = [e for e in events if e.get("run_id") == marker["run_id"]]

    summary = _status_summary(marker, events)

    if summary["status"] == "waiting":
        st.info(
            "Waiting for an experiment. Start it in another terminal:\n\n"
            "`python main.py`"
        )

    _run_status_section(summary, events)
    st.divider()
    _agent_summary_section(events)
    st.divider()
    _live_chunk_stream_section(events)
    st.divider()
    _resources_section(events)
    st.divider()
    _raw_event_stream_section(events)

    # ---------------- Legacy detail views (kept) ----------------
    with st.expander("📜 Timeline (all events)"):
        rows = build_timeline_rows(events)
        if not rows:
            st.write("No events yet.")
        else:
            for row in reversed(rows[-400:]):
                ts = row["timestamp"][11:19] if row["timestamp"] else ""
                actors = (
                    f"{row['sender']}→{row['receiver']}"
                    if row["receiver"]
                    else row["sender"]
                )
                st.markdown(f"`{ts}` **{row['kind']}** {actors}")
                if row["tool"]:
                    st.caption(f"tool: {row['tool']}")
                if row["text"]:
                    st.caption(str(row["text"])[:400])

    with st.expander("🤖 Agent detail (messages / security observation)"):
        agents = agent_statuses(events)
        if not agents:
            st.write("No agent activity yet.")
        else:
            for name, entry in sorted(agents.items()):
                if entry["messages"]:
                    st.write(f"**{name} — recent messages**")
                    for msg in entry["messages"]:
                        st.caption(
                            f"`{msg['ts'][11:19]}` → {msg['to']}: "
                            f"{msg['text'][:200]}"
                        )
                sec = entry["security"]
                if sec:
                    st.write(f"**{name} — security observation**")
                    st.json(
                        {
                            "security_score": sec.get("security_score"),
                            "investigation_required": sec.get(
                                "investigation_required"
                            ),
                            "semantic_assessed": sec.get("semantic_assessed"),
                            # Two independent reference axes.
                            "original_task_assessed": sec.get(
                                "original_task_assessed"
                            ),
                            "original_task_similarity": sec.get(
                                "original_task_similarity"
                            ),
                            "original_task_deviation": sec.get(
                                "original_task_deviation"
                            ),
                            "assigned_subtask_assessed": sec.get(
                                "assigned_subtask_assessed"
                            ),
                            "assigned_subtask_similarity": sec.get(
                                "assigned_subtask_similarity"
                            ),
                            "assigned_subtask_deviation": sec.get(
                                "assigned_subtask_deviation"
                            ),
                            "deviation_score": sec.get("deviation_score"),
                            "detector_result": sec.get("detector_result"),
                        }
                    )


def main() -> None:
    # Auto-refresh only the dashboard; never touches the experiment.
    # st.fragment(run_every=...) re-runs ONLY `render`, and `render`
    # only reads files.
    try:
        refresh = st.fragment(run_every=2.0)(render)
        refresh()
    except Exception:
        # Older Streamlit without fragments: single render + manual rerun.
        render()
        if st.button("Refresh"):
            st.rerun()


if __name__ == "__main__":
    main()
