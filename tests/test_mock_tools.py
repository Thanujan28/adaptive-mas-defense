import json
import tempfile
import unittest
from pathlib import Path

from environment.mas_environment import MASEnvironment
from tools.mock_email import MockEmailTool, mail_request_from_text
from tools.mock_calendar import MockCalendarTool
from tools.mock_calendar import calendar_request_from_text
from tools.tool_manager import ToolManager


class _FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeMailHog:
    def __init__(self):
        self.posts = []

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        return _FakeResponse()

    def get(self, url, timeout):
        return _FakeResponse({
            "items": [{
                "ID": "mail-1",
                "Content": {
                    "Headers": {
                        "From": ["sender@example.test"],
                        "To": ["recipient@example.test"],
                        "Subject": ["Status"],
                    },
                    "Body": "Complete",
                },
            }]
        })


class MockToolTests(unittest.TestCase):

    def test_calendar_intent_becomes_structured_tool_request(self):
        request = calendar_request_from_text(
            "Schedule a calendar event titled Research review on 2026-09-07T09:00:00Z."
        )
        self.assertEqual(
            request,
            {
                "tool_name": "mock_calendar",
                "arguments": {
                    "operation": "create",
                    "title": "Research review",
                    "start": "2026-09-07T09:00:00Z",
                },
            },
        )

    def test_executor_meeting_instruction_becomes_calendar_request(self):
        from agents.executor import ExecutorAgent, _calendar_description_from_analysis

        request = ExecutorAgent(name="executor-1").create_tool_request(
            "Schedule a 60-minute research meeting for next Wednesday at 10:00 AM, "
            "create a calendar event titled ‘AI Agent Security Research Meeting’ "
            "and invite the Coordinator, Researcher-1, Researcher-2, Analyst-1, "
            "and Analyst-2 as participants",
            "selected topic",
        )
        self.assertEqual(request["tool_name"], "mock_calendar")
        self.assertEqual(request["arguments"]["duration_minutes"], 60)
        self.assertEqual(request["arguments"]["start"], "2026-09-09T10:00:00")
        self.assertIn("Coordinator", request["arguments"]["participants"])
        self.assertEqual(
            _calendar_description_from_analysis(
                "Selected topic: Agent memory poisoning in multi-agent systems.\n"
                "Additional analysis: this is a long report."
            ),
            "Agent memory poisoning in multi-agent systems.",
        )
        self.assertLess(
            len(_calendar_description_from_analysis("A long report without a label.")),
            100,
        )
    def test_calendar_operations_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "calendar.json"
            calendar = MockCalendarTool(str(database_path))
            event = calendar.create_event(
                title="Research review",
                start="2026-09-07T09:00:00Z",
            )

            self.assertEqual(event["id"], "event-0001")
            with database_path.open("r", encoding="utf-8") as database:
                stored_data = json.load(database)
            self.assertEqual(stored_data["events"], [event])

            reloaded_calendar = MockCalendarTool(str(database_path))
            self.assertEqual(reloaded_calendar.list_events(), [event])
            self.assertEqual(
                reloaded_calendar.delete_event("event-0001"),
                {"status": "deleted", "id": "event-0001"},
            )

    def test_mail_intent_becomes_structured_tool_request(self):
        request = mail_request_from_text(
            "Send an email to recipient@example.test "
            "subject: Status body: Complete"
        )
        self.assertEqual(
            request,
            {
                "tool_name": "mock_mail",
                "arguments": {
                    "operation": "send",
                    "to": "recipient@example.test",
                    "subject": "Status",
                    "body": "Complete",
                },
            },
        )

    def test_email_uses_mailhog_api(self):
        fake_mailhog = _FakeMailHog()
        email = MockEmailTool(http_client=fake_mailhog)
        self.assertEqual(
            email.send_email(
                to="recipient@example.test",
                subject="Status",
                body="Complete",
            ),
            {
                "status": "sent",
                "to": ["recipient@example.test"],
                "subject": "Status",
            },
        )
        self.assertEqual(fake_mailhog.posts[0][0], "http://localhost:8025/api/v1/send")
        self.assertEqual(
            email.list_messages()[0]["subject"],
            "Status",
        )

    def test_agent_can_request_both_mock_tool_names(self):
        manager = ToolManager()
        self.assertTrue(manager.is_allowed("researcher-1", "mock_calendar"))
        self.assertTrue(manager.is_allowed("researcher-1", "mock_mail"))
        self.assertIs(manager.tools["mock_mail"], manager.tools["mock_email"])
        self.assertIs(
            manager.tools["mock_calender"],
            manager.tools["mock_calendar"],
        )
        self.assertTrue(manager.is_allowed("researcher-1", "mock_calender"))

        environment = MASEnvironment(topology_name="layered")
        environment.request_tool(
            requesting_agent="researcher-1",
            tool_name="mock_calendar",
            arguments={"operation": "list"},
        )
        self.assertEqual(
            environment.get_events()[0]["metadata"]["authorization_result"],
            "allowed",
        )

    def test_permissions_are_independent_of_topology(self):
        manager = ToolManager()
        for topology_name in (
            "centralized",
            "layered",
            "fully_connected",
            "shared_pool",
        ):
            manager.set_topology(topology_name)
            self.assertTrue(manager.is_allowed("researcher-1", "mock_calendar"))
            self.assertTrue(manager.is_allowed("analyst-1", "mock_email"))
            self.assertTrue(manager.is_allowed("executor-1", "mock_email"))
            self.assertTrue(manager.is_allowed("executor-1", "mock_calendar"))
            self.assertFalse(manager.is_allowed("executor-1", "internet_search"))

    def test_tool_events_preserve_route_identity_and_topology(self):
        for topology_name, requesting_agent in (
            ("centralized", "researcher-1"),
            ("layered", "researcher-1"),
            ("fully_connected", "researcher-1"),
            ("shared_pool", "researcher-1"),
        ):
            with self.subTest(topology=topology_name):
                environment = MASEnvironment(topology_name=topology_name)
                environment.request_tool(
                    requesting_agent=requesting_agent,
                    tool_name="mock_calendar",
                    arguments={
                        "operation": "create",
                        "title": "Topology check",
                        "start": "2026-09-07T09:00:00Z",
                    },
                )
                tool_events = [
                    event for event in environment.get_events()
                    if event["event_type"].startswith("tool_")
                ]
                self.assertEqual(
                    [event["event_type"] for event in tool_events],
                    [
                        "tool_request",
                        "tool_forward",
                        "tool_execution",
                        "tool_usage",
                        "tool_result",
                        "tool_result_delivery",
                    ],
                )
                request_ids = {
                    event["request_id"]
                    for event in tool_events
                    if event["event_type"] != "tool_usage"
                }
                self.assertEqual(len(request_ids), 1)
                for event in tool_events:
                    if event["event_type"] == "tool_usage":
                        continue
                    self.assertEqual(
                        event["metadata"]["requesting_agent"],
                        requesting_agent,
                    )
                    self.assertEqual(
                        event["metadata"]["topology"],
                        topology_name,
                    )


if __name__ == "__main__":
    unittest.main()