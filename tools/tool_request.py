from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ToolRequest:

    agent: str

    tool_name: str

    arguments: Dict[str, Any] = field(
        default_factory=dict
    )

    request_id: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def requester(self) -> str:
        return self.agent

    def to_dict(self):

        return {
            "request_id": self.request_id,
            "agent": self.agent,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "metadata": self.metadata,
        }