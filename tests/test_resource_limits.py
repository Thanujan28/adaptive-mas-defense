import unittest
import time

from environment.mas_environment import MASEnvironment
from environment.resource_accounting import (
    ResourceBudget,
    truncate_tool_result,
)
from tools.tool_manager import ToolManager


class _Response:
    content = "ok"
    usage_metadata = {"total_tokens": 3}


class _LLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _Response()


class _SlowCalendar:
    def list_events(self):
        time.sleep(0.05)
        return []


class ResourceLimitTests(unittest.TestCase):

    def test_environment_loads_configured_limits(self):
        environment = MASEnvironment("shared_pool")
        budget = environment.resource_budget
        self.assertEqual(budget.token_limit, 200000)
        self.assertEqual(budget.tool_limit, 50)
        self.assertEqual(budget.tool_timeout_seconds, 30)
        self.assertEqual(budget.tool_result_tokens, 2000)
        self.assertEqual(budget.memory_capacity, 100)
        self.assertEqual(
            environment.memory.get_memory("coordinator").capacity,
            100,
        )

    def test_budget_rejects_tokens_before_next_call(self):
        budget = ResourceBudget(token_limit=3)
        llm = _LLM()
        from environment.resource_accounting import TrackedLLM

        tracked = TrackedLLM(llm, "coordinator", budget)
        tracked.invoke("first")
        with self.assertRaises(RuntimeError):
            tracked.invoke("second")
        self.assertEqual(llm.calls, 1)

    def test_tool_result_truncation_uses_requested_limit(self):
        result = truncate_tool_result(" ".join(["token"] * 11), max_tokens=10)
        self.assertEqual(len(result.split()), 10)
        nested = truncate_tool_result(
            [{"content": "one two three"}, {"content": "four five six"}],
            max_tokens=5,
        )
        self.assertEqual(
            sum(len(item["content"].split()) for item in nested),
            5,
        )

    def test_tool_budget_rejects_after_limit(self):
        budget = ResourceBudget(tool_limit=1)
        manager = ToolManager(resource_budget=budget)
        manager.tools["mock_calendar"] = type(
            "Calendar", (), {"list_events": lambda self: []}
        )()
        manager.execute("coordinator", "mock_calendar", {"operation": "list"})
        with self.assertRaises(RuntimeError):
            manager.execute("coordinator", "mock_calendar", {"operation": "list"})
        self.assertEqual(budget.tools_used, 1)

    def test_tool_timeout_is_recorded(self):
        events = []
        budget = ResourceBudget(tool_timeout_seconds=0.01)
        manager = ToolManager(
            resource_budget=budget,
            event_callback=lambda *event: events.append(event),
        )
        manager.tools["mock_calendar"] = _SlowCalendar()
        with self.assertRaises(TimeoutError):
            manager.execute("coordinator", "mock_calendar", {"operation": "list"})
        self.assertEqual(budget.timed_out_tools, 1)
        self.assertTrue(any(event[0] == "tool_timeout" for event in events))

    def test_episode_reset_clears_counters(self):
        environment = MASEnvironment("shared_pool")
        environment.memory.add(
            agent_name="coordinator",
            content="old episode memory",
        )
        environment.resource_budget.tokens_used = 12
        environment.resource_budget.tools_used = 4
        environment.graph = type(
            "GraphStub",
            (),
            {"invoke": lambda self, state: {"final_result": "ok"}},
        )()
        environment.execute_task("new episode")
        self.assertEqual(environment.get_resource_state()["tokens_used"], 0)
        self.assertEqual(environment.get_resource_state()["tools_used"], 0)
        self.assertEqual(environment.memory.get_memory("coordinator").count(), 0)


if __name__ == "__main__":
    unittest.main()