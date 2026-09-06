class PPOReward:
	"""Measured reward components for detection, containment, completion, cost."""

	def compute_components(self, events, resource_state):
		event_types = [event.get("event_type") for event in events]
		return {
			"detection": 1.0 if "attack" in event_types and "investigation" in event_types else 0.0,
			"containment": 1.0 if "containment" in event_types else 0.0,
			"task_completion": 1.0 if "final_result" in event_types else 0.0,
			"resource_cost": -(
				resource_state.get("tokens_used", 0) / max(resource_state.get("token_limit", 1), 1)
				+ resource_state.get("tools_used", 0) / max(resource_state.get("tool_limit", 1), 1)
			),
		}

	def compute(self, events, resource_state):
		components = self.compute_components(events, resource_state)
		return sum(components.values())
