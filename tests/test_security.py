import unittest

from environment.topology import CommunicationTopology
from environment.mas_environment import MASEnvironment
from tools.tool_control_plane import ToolControlPlane
from tools.tool_manager import ToolManager
from tools.tool_request import ToolRequest


class _EmptySearchTool:

	def search(self, query, max_results):
		return []


class ToolAuthorizationTests(unittest.TestCase):

	def setUp(self):
		self.manager = ToolManager()

	def test_centralized_coordinator_can_search(self):
		self.manager.set_topology("centralized")
		self.manager.tools["internet_search"] = _EmptySearchTool()
		self.assertEqual(
			self.manager.execute(
				"coordinator",
				"internet_search",
				{"query": "security"}
			),
			[]
		)

	def test_centralized_researcher_is_denied(self):
		self.manager.set_topology("centralized")
		with self.assertRaises(PermissionError):
			self.manager.execute(
				"researcher",
				"internet_search",
				{"query": "security"}
			)

	def test_centralized_analyst_is_denied(self):
		self.manager.set_topology("centralized")
		with self.assertRaises(PermissionError):
			self.manager.execute(
				"analyst",
				"academic_search",
				{"query": "security"}
			)

	def test_centralized_executor_is_denied(self):
		self.manager.set_topology("centralized")
		with self.assertRaises(PermissionError):
			self.manager.execute(
				"executor",
				"internet_search",
				{"query": "security"}
			)

	def test_fully_connected_p2p_researcher_can_search(self):
		self.manager.set_topology("fully_connected_p2p")
		self.manager.tools["internet_search"] = _EmptySearchTool()
		self.assertEqual(
			self.manager.execute(
				"researcher",
				"internet_search",
				{"query": "security"}
			),
			[]
		)

	def test_fully_connected_p2p_analyst_can_search(self):
		self.manager.set_topology("fully_connected_p2p")
		self.manager.tools["academic_search"] = _EmptySearchTool()
		self.assertEqual(
			self.manager.execute(
				"analyst",
				"academic_search",
				{"query": "security"}
			),
			[]
		)

	def test_centralized_coordinator_can_write_reports(self):
		self.manager.set_topology("centralized")
		self.assertTrue(
			self.manager.is_allowed(
				"coordinator",
				"report_writer"
			)
		)

	def test_complete_authorization_matrix(self):
		expected = {
			"layered": {
				"coordinator": {
					"internet_search": True,
					"academic_search": True,
					"report_writer": True,
				},
				"researcher": {
					"internet_search": True,
					"academic_search": True,
				},
				"analyst": {
					"internet_search": True,
					"academic_search": True,
				},
				"executor": {
					"internet_search": False,
				},
			},
			"centralized": {
				"coordinator": {
					"internet_search": True,
					"academic_search": True,
					"report_writer": True,
				},
				"researcher": {
					"internet_search": False,
				},
				"analyst": {
					"internet_search": False,
				},
				"executor": {
					"internet_search": False,
				},
			},
			"fully_connected_p2p": {
				"coordinator": {
					"internet_search": True,
					"academic_search": True,
					"report_writer": True,
				},
				"researcher": {
					"internet_search": True,
					"academic_search": True,
				},
				"analyst": {
					"internet_search": True,
					"academic_search": True,
				},
				"executor": {
					"internet_search": False,
				},
			},
			"shared_pool": {
				"coordinator": {
					"internet_search": True,
					"academic_search": True,
					"report_writer": True,
				},
				"researcher": {
					"internet_search": True,
					"academic_search": True,
				},
				"analyst": {
					"internet_search": True,
					"academic_search": True,
				},
				"executor": {
					"internet_search": False,
				},
			},
		}

		for topology_name, agents in expected.items():
			with self.subTest(topology=topology_name):
				self.manager.set_topology(topology_name)
				for agent, tools in agents.items():
					for tool_name, allowed in tools.items():
						with self.subTest(
							agent=agent,
							tool=tool_name
						):
							self.assertEqual(
								self.manager.is_allowed(
									agent,
									tool_name
								),
								allowed
							)

	def test_unknown_inputs_fail_safely(self):
		self.manager.set_topology("layered")
		self.assertFalse(
			self.manager.is_allowed(
				"unknown_agent",
				"internet_search"
			)
		)
		self.assertFalse(
			self.manager.is_allowed(
				"coordinator",
				"unknown_tool"
			)
		)
		with self.assertRaises(ValueError):
			self.manager.set_topology("unknown_topology")

	def test_centralized_control_plane_requires_coordinator_submitter(self):
		control_plane = ToolControlPlane(self.manager)
		request = ToolRequest(
			agent="researcher",
			tool_name="internet_search",
			arguments={"query": "security"},
		)

		with self.assertRaises(PermissionError):
			control_plane.submit(
				request,
				submitted_by="researcher"
			)

	def test_centralized_researcher_request_is_forwarded_by_coordinator(self):
		environment = MASEnvironment(
			topology_name="centralized"
		)
		environment.tool_manager.tools[
			"internet_search"
		] = _EmptySearchTool()

		result = environment.request_tool(
			requesting_agent="researcher",
			tool_name="internet_search",
			arguments={"query": "security"},
		)

		self.assertEqual(result, [])
		events = environment.get_events()
		self.assertEqual(
			events[0]["metadata"]["authorization_result"],
			"allowed"
		)
		self.assertEqual(
			events[1]["sender"],
			"coordinator"
		)
		self.assertEqual(
			events[1]["receiver"],
			"tool_manager"
		)

	def test_centralized_researcher_cannot_access_control_plane_directly(self):
		self.manager.set_topology("centralized")
		control_plane = ToolControlPlane(self.manager)
		request = ToolRequest(
			agent="researcher",
			tool_name="internet_search",
			arguments={"query": "security"},
		)

		with self.assertRaises(PermissionError):
			control_plane.submit(
				request,
				submitted_by="researcher"
			)

	def test_layered_control_plane_accepts_researcher_directly(self):
		self.manager.set_topology("layered")
		self.manager.tools["internet_search"] = _EmptySearchTool()
		control_plane = ToolControlPlane(self.manager)
		request = ToolRequest(
			agent="researcher",
			tool_name="internet_search",
			arguments={"query": "security"},
		)

		self.assertEqual(
			control_plane.submit(
				request,
				submitted_by="researcher"
			),
			[]
		)

	def test_shared_pool_control_plane_accepts_analyst_directly(self):
		self.manager.set_topology("shared_pool")
		self.manager.tools["academic_search"] = _EmptySearchTool()
		control_plane = ToolControlPlane(self.manager)
		request = ToolRequest(
			agent="analyst",
			tool_name="academic_search",
			arguments={"query": "security"},
		)

		self.assertEqual(
			control_plane.submit(
				request,
				submitted_by="analyst"
			),
			[]
		)

	def test_executor_cannot_submit_directly_in_any_topology(self):
		control_plane = ToolControlPlane(self.manager)
		request = ToolRequest(
			agent="executor",
			tool_name="internet_search",
			arguments={"query": "security"},
		)

		for topology_name in (
			"layered",
			"centralized",
			"fully_connected_p2p",
			"shared_pool",
		):
			with self.subTest(topology=topology_name):
				self.manager.set_topology(topology_name)
				with self.assertRaises(PermissionError):
					control_plane.submit(
						request,
						submitted_by="executor"
					)

	def test_environment_logs_executor_denial_without_execution(self):
		environment = MASEnvironment(
			topology_name="centralized"
		)

		with self.assertRaises(PermissionError):
			environment.request_tool(
				requesting_agent="executor",
				tool_name="internet_search",
				arguments={"query": "security"},
			)

		event_types = [
			event["event_type"]
			for event in environment.get_events()
		]

		self.assertEqual(event_types[-1], "tool_denied")
		self.assertNotIn("tool_execution", event_types)

	def test_layered_request_uses_direct_control_plane_route(self):
		environment = MASEnvironment(
			topology_name="layered"
		)
		environment.tool_manager.tools[
			"internet_search"
		] = _EmptySearchTool()

		environment.request_tool(
			requesting_agent="analyst",
			tool_name="internet_search",
			arguments={"query": "security"},
		)

		events = environment.get_events()
		self.assertEqual(
			[
				event["event_type"]
				for event in events
			],
			[
				"tool_request",
				"tool_forward",
				"tool_execution",
				"tool_result",
				"tool_result_delivery",
			]
		)
		self.assertEqual(
			events[0]["metadata"]["authorization_result"],
			"allowed"
		)
		self.assertEqual(
			events[1]["sender"],
			"analyst"
		)
		self.assertEqual(
			events[1]["receiver"],
			"tool_control_plane"
		)
		self.assertEqual(
			events[0]["receiver"],
			"tool_control_plane"
		)
		self.assertEqual(
			events[4]["receiver"],
			"analyst"
		)


class TopologyTests(unittest.TestCase):

	def test_centralized_topology_contains_only_agents(self):
		agents = [
			"coordinator",
			"researcher",
			"analyst",
			"executor",
		]
		topology = CommunicationTopology.centralized(agents)

		self.assertEqual(set(topology.get_nodes()), set(agents))
		self.assertEqual(
			set(topology.get_edges()),
			{
				("coordinator", "researcher"),
				("researcher", "coordinator"),
				("coordinator", "analyst"),
				("analyst", "coordinator"),
				("coordinator", "executor"),
				("executor", "coordinator"),
			}
		)

	def test_all_topology_sizes_and_nodes(self):
		agents = [
			"coordinator",
			"researcher",
			"analyst",
			"executor",
		]
		expected_sizes = {
			"layered": (4, 6),
			"centralized": (4, 6),
			"fully_connected_p2p": (4, 12),
			"shared_pool": (5, 8),
		}

		for topology_name, expected in expected_sizes.items():
			with self.subTest(topology=topology_name):
				topology = CommunicationTopology.create(
					topology_name,
					agents
				)
				self.assertEqual(
					topology.number_of_agents(),
					expected[0]
				)
				self.assertEqual(
					topology.number_of_connections(),
					expected[1]
				)
				self.assertNotIn(
					"tool_manager",
					topology.get_nodes()
				)
				self.assertNotIn(
					"tool_control_plane",
					topology.get_nodes()
				)


if __name__ == "__main__":
	unittest.main()
