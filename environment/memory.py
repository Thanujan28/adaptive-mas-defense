from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
import re


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
    Importance-weighted per-agent episodic memory.

    Each agent has its own independent memory store.

    Retrieval ranks lexical query overlap first, then importance,
    then recency. The store is capped at 100 entries; when full,
    the lowest-ranked record (importance, then oldest) is evicted.
    This keeps the experiment deterministic without requiring an
    embedding service.
    """

    CAPACITY = 100

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

        if len(self.memories) > self.CAPACITY:
            eviction_index = min(
                range(len(self.memories)),
                key=lambda index: (
                    self.memories[index].importance,
                    self.memories[index].timestamp,
                ),
            )
            self.memories.pop(eviction_index)

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

        query_terms = set(
            re.findall(r"[a-z0-9]+", (query or "").lower())
        )

        def retrieval_score(memory: MemoryRecord):
            content_terms = set(
                re.findall(r"[a-z0-9]+", memory.content.lower())
            )
            overlap = len(query_terms & content_terms)
            return overlap, memory.importance, memory.timestamp

        memories = sorted(
            self.memories,
            key=retrieval_score,
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