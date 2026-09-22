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
emitted. Nothing is fabricated.
"""

from __future__ import annotations

from datetime import datetime
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
# UI
# =================================================================

def _status_summary(marker: dict | None, events: list[dict]) -> dict:
    if marker is None:
        return {
            "status": "waiting",
            "detail": "No run marker yet -- start main.py to begin.",
            "last_ts": _event_ts(events[-1]) if events else "",
            "run_id": "",
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
    }


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

    # ---------------- Experiment status ----------------
    cols = st.columns(4)
    status_colour = {
        "running": "🟢",
        "completed": "✅",
        "failed": "🔴",
        "waiting": "⚪",
    }.get(summary["status"], "⚪")
    cols[0].metric("Status", f"{status_colour} {summary['status']}")
    cols[1].metric("Events", len(events))
    cols[2].metric("Run ID", summary["run_id"] or "—")
    cols[3].metric("Last event", summary["last_ts"][11:19] if summary["last_ts"] else "—")

    if summary["status"] == "waiting":
        st.info(
            "Waiting for an experiment. Start it in another terminal:\n\n"
            "`python main.py`"
        )

    st.divider()

    tab_timeline, tab_agents, tab_security, tab_resources = st.tabs(
        ["📜 Timeline", "🤖 Agents", "🔐 Security", "📊 Resources"]
    )

    # ---------------- Timeline ----------------
    with tab_timeline:
        rows = build_timeline_rows(events)
        if not rows:
            st.write("No events yet.")
        else:
            filter_kind = st.multiselect(
                "Filter by type",
                options=sorted({r["kind"] for r in rows}),
                default=[],
            )
            shown = (
                [r for r in rows if r["kind"] in filter_kind]
                if filter_kind
                else rows
            )
            for row in reversed(shown[-400:]):
                ts = row["timestamp"][11:19] if row["timestamp"] else ""
                actors = (
                    f"{row['sender']}→{row['receiver']}"
                    if row["receiver"]
                    else row["sender"]
                )
                line = f"`{ts}` **{row['kind']}** {actors}"
                st.markdown(line)
                if row["tool"]:
                    st.caption(f"tool: {row['tool']}")
                if row["text"]:
                    st.caption(str(row["text"])[:400])

    # ---------------- Agents ----------------
    with tab_agents:
        agents = agent_statuses(events)
        if not agents:
            st.write("No agent activity yet.")
        else:
            for name, entry in sorted(agents.items()):
                with st.expander(
                    f"{name} — last: {entry['last_kind'] or '—'} "
                    f"@ {entry['last_ts'][11:19] if entry['last_ts'] else '—'}"
                ):
                    if entry["last_text"]:
                        st.write("**Latest activity**")
                        st.caption(str(entry["last_text"])[:500])
                    if entry["messages"]:
                        st.write("**Recent messages**")
                        for msg in entry["messages"]:
                            st.caption(
                                f"`{msg['ts'][11:19]}` → {msg['to']}: "
                                f"{msg['text'][:200]}"
                            )
                    sec = entry["security"]
                    if sec:
                        st.write("**Security observation**")
                        st.json(
                            {
                                "security_score": sec.get("security_score"),
                                "investigation_required": sec.get(
                                    "investigation_required"
                                ),
                                "semantic_assessed": sec.get(
                                    "semantic_assessed"
                                ),
                                "deviation_score": sec.get("deviation_score"),
                                "detector_result": sec.get("detector_result"),
                            }
                        )

    # ---------------- Security ----------------
    with tab_security:
        observations = [
            _payload_of(e)
            for e in events
            if e.get("event_type") == "security_observation"
        ]
        st.subheader("Security state per agent (from emitted observations)")
        if not observations:
            st.write("No security observations yet.")
        else:
            for obs in observations:
                st.markdown(
                    f"**{obs.get('agent_id')}** — "
                    f"score={obs.get('security_score')} "
                    f"investigate={obs.get('investigation_required')}"
                )
                if obs.get("semantic_assessed"):
                    st.caption(
                        f"Tier 1 semantic: deviation="
                        f"{obs.get('deviation_score')} "
                        f"task_sim={obs.get('task_similarity')} "
                        f"subtask_sim={obs.get('subtask_similarity')}"
                    )
                st.caption(f"detector: {obs.get('detector_result')}")
                if obs.get("response_preview"):
                    st.caption(f"response: {str(obs['response_preview'])[:240]}")

        st.divider()
        st.subheader("Investigation / containment / allocation events")
        inv = investigation_events(events)
        if not inv:
            st.write(
                "No investigation/containment events emitted in this run. "
                "(These appear only if the environment emits them.)"
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

    # ---------------- Resources ----------------
    with tab_resources:
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
        c[2].metric("Remaining", remaining if remaining is not None else "—")

        c2 = st.columns(3)
        c2[0].metric("LLM calls", res["llm_calls"])
        c2[1].metric("Tool usage events", res["tool_usage"])
        c2[2].metric("Timeouts / denied", f"{res['tool_timeouts']} / {res['tool_denied']}")

        if res["tokens_by_agent"]:
            st.write("**Tokens by agent**")
            for agent, tokens in sorted(res["tokens_by_agent"].items()):
                st.caption(f"{agent}: {tokens}")


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
