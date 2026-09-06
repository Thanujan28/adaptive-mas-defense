import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


def calendar_request_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Build a calendar tool request from explicit scheduling language."""
    normalized = str(text or "").strip()
    normalized = normalized.replace("‘", '"').replace("’", '"')
    lowered = normalized.lower()
    if not any(word in lowered for word in ("calendar", "schedule", "meeting", "appointment")):
        return None

    event_match = re.search(r"\bevent[- ]?(\d+)\b", lowered)
    if event_match and any(word in lowered for word in ("delete", "remove", "cancel")):
        return {
            "tool_name": "mock_calendar",
            "arguments": {
                "operation": "delete",
                "event_id": f"event-{int(event_match.group(1)):04d}",
            },
        }

    if any(phrase in lowered for phrase in ("list events", "show events", "view events", "calendar events")):
        return {
            "tool_name": "mock_calendar",
            "arguments": {"operation": "list"},
        }

    operation = "get" if event_match and any(
        word in lowered for word in ("get", "retrieve", "view")
    ) else "create"
    if operation == "get":
        return {
            "tool_name": "mock_calendar",
            "arguments": {
                "operation": "get",
                "event_id": f"event-{int(event_match.group(1)):04d}",
            },
        }

    start_match = re.search(
        r"\b(20\d{2}-\d{2}-\d{2}(?:[tT ][0-9]{2}:[0-9]{2}(?::[0-9]{2})?(?:Z)?)?)\b",
        normalized,
    )
    if not start_match:
        relative_match = re.search(
            r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
            r"(?:\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?)?",
            normalized,
            re.IGNORECASE,
        )
        if relative_match:
            weekdays = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5,
                "sunday": 6,
            }
            now = datetime.now()
            days_ahead = (
                weekdays[relative_match.group(1).lower()]
                - now.weekday()
            ) % 7 or 7
            event_date = now.date() + timedelta(days=days_ahead)
            hour = int(relative_match.group(2) or 9)
            minute = int(relative_match.group(3) or 0)
            meridiem = (relative_match.group(4) or "").upper()
            if meridiem == "PM" and hour != 12:
                hour += 12
            if meridiem == "AM" and hour == 12:
                hour = 0
            start_value = datetime.combine(
                event_date,
                datetime.min.time().replace(hour=hour, minute=minute),
            ).isoformat()
            start_match = None
        else:
            start_value = None
    else:
        start_value = start_match.group(1)
    if not start_match and not start_value:
        return None

    title_match = re.search(
        r"\b(?:titled|called|named|title)\b\s*[:=]?\s*[\"']?(.+?)(?:\s+on\s+(?:20\d{2}-\d{2}-\d{2}|next\b)|\s+with\s+the\s+selected|[\"';\n]|$)",
        normalized,
        re.IGNORECASE,
    )
    title = title_match.group(1).strip() if title_match else "Scheduled event"
    title = re.sub(r"\s+(?:on|at)\s*$", "", title, flags=re.IGNORECASE).strip()
    participants = re.findall(
        r"\b(?:Coordinator|Researcher-[12]|Analyst-[12])\b",
        normalized,
    )
    duration_match = re.search(r"(\d+)\s*[- ]?minute", lowered)
    return {
        "tool_name": "mock_calendar",
        "arguments": {
            "operation": "create",
            "title": title,
            "start": start_value,
            **({"duration_minutes": int(duration_match.group(1))} if duration_match else {}),
            **({"participants": participants} if participants else {}),
        },
    }


class MockCalendarTool:
    """Deterministic JSON-backed calendar used by MAS experiments."""

    def __init__(self, database_path: str = "configs/mock_calendar.json"):
        self.database_path = Path(database_path)
        self._events: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1
        self._load()

    def create_event(
        self,
        title: str,
        start: str,
        end: Optional[str] = None,
        description: str = "",
        location: str = "",
        duration_minutes: Optional[int] = None,
        participants: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not title or not str(title).strip():
            raise ValueError("Calendar event title cannot be empty.")
        if not start or not str(start).strip():
            raise ValueError("Calendar event start cannot be empty.")

        event_id = f"event-{self._next_id:04d}"
        self._next_id += 1
        event = {
            "id": event_id,
            "title": str(title).strip(),
            "start": str(start).strip(),
            "end": str(end).strip() if end else "",
            "description": str(description).strip(),
            "location": str(location).strip(),
            "duration_minutes": duration_minutes,
            "participants": list(participants or []),
        }
        self._events[event_id] = event
        self._save()
        return dict(event)

    def list_events(self) -> List[Dict[str, Any]]:
        return [
            dict(self._events[event_id])
            for event_id in sorted(self._events)
        ]

    def get_event(self, event_id: str) -> Dict[str, Any]:
        event = self._events.get(str(event_id).strip())
        if event is None:
            raise KeyError(f"Calendar event not found: {event_id}")
        return dict(event)

    def delete_event(self, event_id: str) -> Dict[str, Any]:
        normalized_id = str(event_id).strip()
        if normalized_id not in self._events:
            raise KeyError(f"Calendar event not found: {event_id}")
        del self._events[normalized_id]
        self._save()
        return {"status": "deleted", "id": normalized_id}

    def _load(self):
        if not self.database_path.exists():
            self._save()
            return

        try:
            with self.database_path.open("r", encoding="utf-8") as database:
                raw_data = database.read().strip()
            if not raw_data:
                self._save()
                return
            data = json.loads(raw_data)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Calendar database could not be read: {self.database_path}"
            ) from exc

        events = data.get("events", []) if isinstance(data, dict) else []
        if not isinstance(events, list):
            raise ValueError("Calendar database events must be a list.")

        self._events = {
            event["id"]: dict(event)
            for event in events
            if isinstance(event, dict) and event.get("id")
        }
        stored_next_id = data.get("next_id", 1) if isinstance(data, dict) else 1
        self._next_id = max(1, int(stored_next_id))

    def _save(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "events": self.list_events(),
        }
        temporary_path = self.database_path.with_suffix(
            self.database_path.suffix + ".tmp"
        )
        with temporary_path.open("w", encoding="utf-8") as database:
            json.dump(data, database, indent=2, sort_keys=True)
            database.write("\n")
        temporary_path.replace(self.database_path)