from typing import Dict, Iterable, Mapping, Optional


class SecurityStateBuilder:
	"""Build a fixed, measured security state from MAS events."""

	FEATURE_NAMES = (
		"tokens_used",
		"tools_used",
		"tool_timeouts",
		"message_count",
		"relay_count",
		"memory_count",
		"attack_count",
		"investigation_count",
		"containment_count",
		"resource_allocation_count",
	)

	def build(
		self,
		events: Iterable[Mapping],
		resource_state: Optional[Mapping] = None,
		memory_counts: Optional[Mapping[str, int]] = None,
	) -> Dict[str, float]:
		events = list(events)
		resource_state = resource_state or {}
		memory_counts = memory_counts or {}
		event_counts = {
			name: sum(1 for event in events if event.get("event_type") == name)
			for name in (
				"message",
				"message_relay",
				"attack",
				"investigation",
				"containment",
				"resource_allocation",
			)
		}
		return {
			"tokens_used": float(resource_state.get("tokens_used", 0)),
			"tools_used": float(resource_state.get("tools_used", 0)),
			"tool_timeouts": float(resource_state.get("timed_out_tools", 0)),
			"message_count": float(event_counts["message"]),
			"relay_count": float(event_counts["message_relay"]),
			"memory_count": float(sum(memory_counts.values())),
			"attack_count": float(event_counts["attack"]),
			"investigation_count": float(event_counts["investigation"]),
			"containment_count": float(event_counts["containment"]),
			"resource_allocation_count": float(event_counts["resource_allocation"]),
		}

	def vector(self, state: Mapping[str, float]):
		return [float(state.get(name, 0.0)) for name in self.FEATURE_NAMES]
