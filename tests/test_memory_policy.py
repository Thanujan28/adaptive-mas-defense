import unittest

from environment.memory import AgentMemory


class MemoryPolicyTests(unittest.TestCase):

    def test_retrieval_uses_query_overlap_then_importance(self):
        memory = AgentMemory("researcher-1")
        memory.add("calendar scheduling policy", importance=2)
        memory.add("calendar security policy", importance=8)
        memory.add("unrelated result", importance=10)

        results = memory.retrieve("calendar", top_k=2)

        self.assertEqual(
            [record.content for record in results],
            ["calendar security policy", "calendar scheduling policy"],
        )

    def test_capacity_evicts_lowest_importance(self):
        memory = AgentMemory("researcher-1")
        for index in range(AgentMemory.CAPACITY):
            memory.add(f"memory {index}", importance=5)

        memory.add("important memory", importance=10)

        self.assertEqual(memory.count(), AgentMemory.CAPACITY)
        self.assertNotIn(
            "memory 0",
            [record.content for record in memory.get_all()],
        )
        self.assertIn(
            "important memory",
            [record.content for record in memory.get_all()],
        )


if __name__ == "__main__":
    unittest.main()