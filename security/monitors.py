from typing import Iterable, Mapping


class SecurityMonitor:
	def summarize(self, events: Iterable[Mapping]) -> dict:
		events = list(events)
		return {
			"investigations": sum(event.get("event_type") == "investigation" for event in events),
			"containments": sum(event.get("event_type") == "containment" for event in events),
			"resource_events": sum(event.get("event_type") in {"llm_usage", "tool_execution"} for event in events),
		}
