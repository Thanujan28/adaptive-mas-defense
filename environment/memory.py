from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class MemoryRecord:
    """
    Represents one memory stored by an individual agent.
    """

    memory_id: str
    agent: str
    content: str
    timestamp: str
    importance: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "agent": self.agent,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata,
        }


class AgentMemory:
    """
    Simple per-agent episodic memory.

    Each agent has its own independent memory store.

    This is intentionally implemented without embeddings or
    vector databases. It provides a clean baseline for later
    experiments involving memory poisoning and memory attacks.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.memories: List[MemoryRecord] = []

    # =========================================================
    # WRITE MEMORY
    # =========================================================

    def add(
        self,
        content: str,
        importance: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryRecord:

        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty.")

        importance = max(1, min(10, importance))

        memory = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            agent=self.agent_name,
            content=content.strip(),
            timestamp=datetime.now().isoformat(),
            importance=importance,
            metadata=metadata or {},
        )

        self.memories.append(memory)

        return memory

    # =========================================================
    # READ MEMORY
    # =========================================================

    def retrieve(
        self,
        query: Optional[str] = None,
        top_k: int = 3
    ) -> List[MemoryRecord]:

        if not self.memories:
            return []

        # -----------------------------------------------------
        # Baseline retrieval:
        #
        # 1. Higher importance
        # 2. More recent memories
        #
        # Query-based semantic retrieval will be added later.
        # -----------------------------------------------------

        memories = sorted(
            self.memories,
            key=lambda memory: (
                memory.importance,
                memory.timestamp
            ),
            reverse=True
        )

        return memories[:top_k]

    # =========================================================
    # ALL MEMORIES
    # =========================================================

    def get_all(self) -> List[MemoryRecord]:

        return list(self.memories)

    # =========================================================
    # MEMORY COUNT
    # =========================================================

    def count(self) -> int:

        return len(self.memories)

    # =========================================================
    # CLEAR MEMORY
    # =========================================================

    def clear(self):

        self.memories.clear()


class MemoryManager:
    """
    Manages independent memory stores for all MAS agents.
    """

    def __init__(self, agent_names: List[str]):

        self.memories: Dict[str, AgentMemory] = {
            agent_name: AgentMemory(agent_name)
            for agent_name in agent_names
        }

    # =========================================================
    # GET AGENT MEMORY
    # =========================================================

    def get_memory(self, agent_name: str) -> AgentMemory:

        if agent_name not in self.memories:
            raise ValueError(
                f"No memory store exists for agent: {agent_name}"
            )

        return self.memories[agent_name]

    # =========================================================
    # WRITE
    # =========================================================

    def add(
        self,
        agent_name: str,
        content: str,
        importance: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryRecord:

        return self.get_memory(agent_name).add(
            content=content,
            importance=importance,
            metadata=metadata,
        )

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        agent_name: str,
        query: Optional[str] = None,
        top_k: int = 3
    ) -> List[MemoryRecord]:

        return self.get_memory(agent_name).retrieve(
            query=query,
            top_k=top_k,
        )

    # =========================================================
    # GET ALL
    # =========================================================

    def get_all(self, agent_name: str) -> List[MemoryRecord]:

        return self.get_memory(agent_name).get_all()

    # =========================================================
    # CLEAR ALL
    # =========================================================

    def clear(self):

        for memory in self.memories.values():
            memory.clear()