from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid


@dataclass
class MASEvent:
    """
    Represents an event occurring inside the multi-agent system.

    Events are used to construct the security state of the MAS.

    The event model captures:
        - agent communication
        - task execution
        - memory operations
        - tool requests and executions
        - tool results
        - token usage
        - security-related metadata

    Full tool results are NOT stored in the event.
    Only compact metadata such as result count, status,
    and tool information should be recorded.

    A request_id allows all events belonging to the same
    tool request or interaction to be correlated.
    """

    timestamp: str
    event_type: str

    sender: Optional[str] = None
    receiver: Optional[str] = None

    content: Optional[str] = None

    tool_call: Optional[str] = None

    memory_update: Optional[str] = None

    token_usage: int = 0

    request_id: Optional[str] = None

    result_count: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        content: Optional[str] = None,
        tool_call: Optional[str] = None,
        memory_update: Optional[str] = None,
        token_usage: int = 0,
        request_id: Optional[str] = None,
        result_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a new MAS event.

        If request_id is not provided, a unique ID is generated.

        Only compact information should be stored in the event.
        Full tool results should remain in the MAS state.
        """

        return cls(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            sender=sender,
            receiver=receiver,
            content=content,
            tool_call=tool_call,
            memory_update=memory_update,
            token_usage=token_usage,
            request_id=request_id or str(uuid.uuid4()),
            result_count=result_count,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event into a dictionary.
        """

        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "tool_call": self.tool_call,
            "memory_update": self.memory_update,
            "token_usage": self.token_usage,
            "request_id": self.request_id,
            "result_count": self.result_count,
            "metadata": self.metadata,
        }