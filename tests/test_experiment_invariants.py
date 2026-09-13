import unittest

from environment.topology import CommunicationTopology


AGENTS = [
    "coordinator",
    "researcher",
    "analyst",
    "executor",
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
        self.assertFalse(topology.can_communicate("researcher", "analyst"))
        self.assertEqual(
            topology.shortest_path("researcher", "analyst"),
            ("researcher", "coordinator", "analyst"),
        )

    def test_shared_pool_has_delivery_invariants(self):
        topology = CommunicationTopology.shared_pool(AGENTS)
        self.assertTrue(topology.can_communicate("researcher", "shared_pool"))
        self.assertTrue(topology.can_communicate("shared_pool", "analyst"))
        self.assertFalse(topology.can_communicate("researcher", "analyst"))


if __name__ == "__main__":
    unittest.main()