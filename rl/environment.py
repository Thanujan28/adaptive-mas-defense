from typing import Any, Dict, Optional

from security.state_builder import SecurityStateBuilder


class PPOEnvironment:
	"""Small adapter exposing MAS episodes as PPO transitions."""

	def __init__(self, mas_environment, reward_calculator=None):
		self.mas_environment = mas_environment
		self.state_builder = SecurityStateBuilder()
		self.reward_calculator = reward_calculator
		self.state: Dict[str, float] = {}
		self.transitions = []
		self.step_count = 0

	def reset(self, task: Optional[str] = None) -> Dict[str, float]:
		self.transitions = []
		self.step_count = 0
		if task is not None:
			self.mas_environment.execute_task(task)
		self.state = self._state()
		return dict(self.state)

	def step(self, action: Any = None):
		previous = dict(self.state)
		self.step_count += 1
		self.state = self._state()
		reward = self.reward_calculator.compute(
			self.mas_environment.get_events(),
			self.mas_environment.get_resource_state(),
		) if self.reward_calculator else 0.0
		transition = (previous, action, reward, dict(self.state))
		self.transitions.append(transition)
		return dict(self.state), reward, True, {"transition": transition}

	def _state(self):
		memory_counts = {
			name: self.mas_environment.memory.get_memory(name).count()
			for name in self.mas_environment.agent_names
		}
		return self.state_builder.build(
			self.mas_environment.get_events(),
			self.mas_environment.get_resource_state(),
			memory_counts,
		)
