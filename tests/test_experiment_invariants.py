import unittest

from environment.topology import CommunicationTopology


AGENTS = [
    "coordinator",
    "planner",
    "researcher-1",
    "researcher-2",
    "analyst-1",
    "analyst-2",
    "executor-1",
    "executor-2",
]


class TopologyInvariantTests(unittest.TestCase):

    def test_definitions_are_immutable_and_complete(self):
        for name in ("centralized", "layered", "fully_connected", "shared_pool"):
            topology = CommunicationTopology.create(name, AGENTS)
            definition = topology.definition
            self.assertEqual(set(AGENTS), set(definition.nodes) - {"shared_pool"})
            with self.assertRaises((AttributeError, TypeError)):
                definition.name = "changed"

    def test_centralized_forbids_direct_downstream_edges(self):
        topology = CommunicationTopology.centralized(AGENTS)
        self.assertFalse(topology.can_communicate("researcher-1", "analyst-1"))
        self.assertEqual(
            topology.shortest_path("researcher-1", "analyst-1"),
            ("researcher-1", "coordinator", "analyst-1"),
        )

    def test_shared_pool_has_delivery_invariants(self):
        topology = CommunicationTopology.shared_pool(AGENTS)
        self.assertTrue(topology.can_communicate("researcher-1", "shared_pool"))
        self.assertTrue(topology.can_communicate("shared_pool", "analyst-1"))
        self.assertFalse(topology.can_communicate("researcher-1", "analyst-1"))


if __name__ == "__main__":
    unittest.main()