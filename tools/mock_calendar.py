import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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
                data = json.load(database)
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