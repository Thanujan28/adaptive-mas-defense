from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeState:
    """Authoritative mutable state for one isolated MAS episode."""

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: Optional[str] = None
    result: Optional[dict] = None
    events: List[Any] = field(default_factory=list)
    mailboxes: Dict[str, list] = field(default_factory=dict)
    shared_pool: list = field(default_factory=list)
    temporary_tools: Dict[str, Any] = field(default_factory=dict)

    def reset(self, agent_names, memory, resource_budget):
        self.episode_id = str(uuid.uuid4())
        self.task = None
        self.result = None
        self.events.clear()
        self.mailboxes = {agent: [] for agent in agent_names}
        self.shared_pool.clear()
        self.temporary_tools.clear()
        memory.clear()
        resource_budget.reset()
        return self.episode_id
