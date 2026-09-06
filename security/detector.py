from typing import Iterable, Mapping


class SecurityDetector:
	"""Deterministic event-based detector used by PPO state/reward code."""

	def detect(self, events: Iterable[Mapping]) -> dict:
		attacks = [event for event in events if event.get("event_type") == "attack"]
		return {
			"detected": bool(attacks),
			"attack_count": len(attacks),
			"attack_ids": [event.get("request_id") for event in attacks],
		}
