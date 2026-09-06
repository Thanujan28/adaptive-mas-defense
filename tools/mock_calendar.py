from typing import Any, Dict, List, Optional


class MockCalendarTool:
    """Deterministic in-memory calendar used by MAS experiments."""

    def __init__(self):
        self._events: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1

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
        return {"status": "deleted", "id": normalized_id}