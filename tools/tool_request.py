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

    def to_dict(self):

        return {
            "request_id": self.request_id,
            "agent": self.agent,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }