"""
security/live_events.py

A lightweight, non-blocking live event stream for the LLM multi-agent
security experiment.

WHY THIS EXISTS
---------------
``main.py`` runs the experiment; a separate Streamlit process watches it.
The two processes need a simple, dependency-free way to share events
WITHOUT the dashboard ever starting, restarting or perturbing the
experiment.

DESIGN (deliberately minimal)
-----------------------------
* One producer (the experiment process) APPENDS JSON lines to a file.
* One consumer (Streamlit) READS the file from a tracked byte offset.

A file is the right tool here: it is a simple local IPC mechanism, it
survives dashboard refreshes, it needs no Redis/Kafka/database, and the
consumer can never execute anything in the producer.

The producer is NON-BLOCKING and FAILURE-ISOLATED: every write is
wrapped so that a missing/unwritable directory or a file error can NEVER
propagate into the experiment. If streaming is off or broken, ``main.py``
runs exactly as before.

The consumer never reads a partially written line: it only advances its
offset past a line that ends in a newline, so a record still being
appended is skipped until it is complete.

Old runs cannot be mistaken for the live one: every event carries the
``run_id`` of the run that produced it, and a ``run.json`` marker records
the CURRENT run id, its start time and its status
(running/completed/failed). The dashboard keys off the marker.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default locations (both under outputs/, which is gitignored).
DEFAULT_EVENTS_PATH = Path("outputs") / "live_events.jsonl"
DEFAULT_RUN_MARKER_PATH = Path("outputs") / "live_run.json"


# =================================================================
# PRODUCER
# =================================================================

@dataclass
class LiveEventPublisher:
    """
    Appends experiment events to an append-only JSONL file.

    Thread-safe (a lock guards the file handle) and NON-BLOCKING: any
    I/O error is swallowed so the experiment is never affected.

    Construct once per run and hand it to ``MASEnvironment``; a fresh
    ``run_id`` isolates this run's events from any previous run's.
    """

    events_path: Path = DEFAULT_EVENTS_PATH
    marker_path: Path = DEFAULT_RUN_MARKER_PATH
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    enabled: bool = True
    flush_every: int = 1

    def __post_init__(self) -> None:
        self.events_path = Path(self.events_path)
        self.marker_path = Path(self.marker_path)
        self._lock = threading.Lock()
        self._writes_since_flush = 0
        self._seq = 0

    # ------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------

    def start_run(self, task: str, topology: str, metadata: Optional[dict] = None) -> None:
        """Mark the start of a run: truncate the event file, write the marker."""

        if not self.enabled:
            return
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)

            # Fresh file per run so old events cannot be replayed as live.
            self.events_path.write_text("", encoding="utf-8")

            self._write_marker(
                {
                    "run_id": self.run_id,
                    "status": "running",
                    "task": task,
                    "topology": topology,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": None,
                    "error": None,
                    "pid": os.getpid(),
                    "metadata": metadata or {},
                }
            )
            self.emit(
                "experiment_started",
                {"task": task, "topology": topology, **(metadata or {})},
            )
        except Exception:
            # Never let streaming break the experiment.
            self.enabled = False

    def finish_run(
        self,
        status: str = "completed",
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Mark the run completed/failed and emit a terminal event."""

        if not self.enabled:
            return
        try:
            self.emit(
                f"experiment_{status}",
                {"error": error, **(metadata or {})},
            )
            marker = self._read_marker() or {"run_id": self.run_id}
            marker.update(
                {
                    "run_id": self.run_id,
                    "status": status,
                    "completed_at": datetime.now().isoformat(),
                    "error": error,
                }
            )
            self._write_marker(marker)
        except Exception:
            self.enabled = False

    # ------------------------------------------------------------
    # EVENT EMISSION
    # ------------------------------------------------------------

    def emit(self, event_type: str, payload: Optional[dict] = None) -> None:
        """
        Append one event. Non-blocking and failure-isolated.

        The record is a single JSON object on one line, terminated by
        ``\\n``, so the consumer can detect completeness.
        """

        if not self.enabled:
            return

        record = {
            "run_id": self.run_id,
            "ts": datetime.now().isoformat(),
            "seq": self._next_seq(),
            "event_type": event_type,
            "payload": payload or {},
        }

        try:
            line = json.dumps(record, default=str)
        except Exception:
            # Unserialisable payload: skip it rather than crash the run.
            return

        try:
            with self._lock:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    self._writes_since_flush += 1
                    if self._writes_since_flush >= max(1, self.flush_every):
                        handle.flush()
                        self._writes_since_flush = 0
        except Exception:
            # Any file error disables streaming; the experiment continues.
            self.enabled = False

    # ------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _write_marker(self, marker: dict) -> None:
        # Atomic-ish: write to a temp file then replace, so a consumer
        # never sees a half-written marker.
        tmp = self.marker_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(marker, default=str, indent=2), encoding="utf-8")
        os.replace(tmp, self.marker_path)

    def _read_marker(self) -> Optional[dict]:
        try:
            if self.marker_path.exists():
                return json.loads(self.marker_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None


# =================================================================
# CONSUMER
# =================================================================

@dataclass
class EventReader:
    """
    Incremental reader for the append-only event file.

    Keeps a byte offset and only returns lines that are COMPLETE (end in
    a newline), so a record still being written is never parsed
    partially. Duplicate refreshes return no new events because the
    offset only moves forward.

    If the file is REPLACED by a new run (a new ``run_id`` at the top of
    the file), the reader resets to the start so a new run is never
    missed -- even when the new file is already larger than the old
    offset.
    """

    events_path: Path = DEFAULT_EVENTS_PATH

    def __post_init__(self) -> None:
        self.events_path = Path(self.events_path)
        self._offset = 0
        self._carry = ""
        self._run_id: Optional[str] = None

    def reset(self) -> None:
        self._offset = 0
        self._carry = ""
        self._run_id = None

    def _first_run_id(self) -> Optional[str]:
        """Read the run_id of the FIRST complete line in the file."""

        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                first = handle.readline()
        except OSError:
            return None
        first = first.strip()
        if not first:
            return None
        try:
            return json.loads(first).get("run_id")
        except json.JSONDecodeError:
            return None

    def read_new(self) -> list[dict]:
        """Return events appended since the last call (never duplicates)."""

        if not self.events_path.exists():
            return []

        try:
            size = self.events_path.stat().st_size
        except OSError:
            return []

        current_run = self._first_run_id()

        # Detect a NEW run two ways: the file shrank below our offset
        # (truncation), or the run_id at the top changed. Either means a
        # different experiment owns the file now, so reset.
        if size < self._offset or (
            current_run is not None
            and self._run_id is not None
            and current_run != self._run_id
        ):
            self._offset = 0
            self._carry = ""

        if current_run is not None:
            self._run_id = current_run

        if size == self._offset:
            return []

        events: list[dict] = []
        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        except OSError:
            return []

        data = self._carry + chunk
        # Keep any trailing partial line in the carry buffer.
        if data.endswith("\n"):
            complete, self._carry = data, ""
        else:
            last_newline = data.rfind("\n")
            if last_newline == -1:
                self._carry = data
                return []
            complete, self._carry = data[: last_newline + 1], data[last_newline + 1 :]

        for line in complete.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip a corrupt line rather than fail the whole refresh.
                continue

        self._offset += len(chunk.encode("utf-8"))
        return events


def read_run_marker(marker_path: Path = DEFAULT_RUN_MARKER_PATH) -> Optional[dict]:
    """Read the current-run marker, or None if it does not exist yet."""

    marker_path = Path(marker_path)
    try:
        if marker_path.exists():
            return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def read_all_events(events_path: Path = DEFAULT_EVENTS_PATH) -> list[dict]:
    """Read every complete event currently in the file (history)."""

    events_path = Path(events_path)
    if not events_path.exists():
        return []
    out: list[dict] = []
    try:
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out
