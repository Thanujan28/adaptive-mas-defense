import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TopologyDefinition:
    """Immutable topology contract used for experiment comparison."""

    name: str
    nodes: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]


class CommunicationTopology:
    """
    Represents the communication topology of the multi-agent system.

    NetworkX is used to maintain a directed communication graph:
        Nodes  -> MAS agents
        Edges  -> Allowed communication relationships

    Example:
        Coordinator -> Analyst
        Analyst -> Researcher
        Researcher -> Executor
    """

    def __init__(
        self,
        agents: List[str],
        topology_name: str = "custom"
    ):
        self.agents = list(agents)
        self.topology_name = topology_name

        self.graph = nx.DiGraph()

        self.graph.add_nodes_from(self.agents)

    def add_connection(
        self,
        sender: str,
        receiver: str
    ):
        """
        Add a directed communication edge.

        sender -> receiver
        """

        if sender not in self.graph:
            self.graph.add_node(sender)

        if receiver not in self.graph:
            self.graph.add_node(receiver)

        self.graph.add_edge(sender, receiver)

    def add_bidirectional_connection(
        self,
        agent_a: str,
        agent_b: str
    ):
        """
        Add communication in both directions.

        agent_a <-> agent_b
        """

        self.add_connection(agent_a, agent_b)
        self.add_connection(agent_b, agent_a)

    def can_communicate(
        self,
        sender: str,
        receiver: str
    ) -> bool:
        """
        Check whether sender is allowed to communicate
        directly with receiver.
        """

        return self.graph.has_edge(sender, receiver)

    def get_receivers(
        self,
        sender: str
    ) -> List[str]:
        """
        Return agents that can receive messages
        directly from the sender.
        """

        return list(self.graph.successors(sender))

    def get_senders(
        self,
        receiver: str
    ) -> List[str]:
        """
        Return agents that can send messages
        directly to the receiver.
        """

        return list(self.graph.predecessors(receiver))

    def get_graph(self) -> nx.DiGraph:
        """
        Return the underlying NetworkX directed graph.
        """

        return self.graph

    @property
    def definition(self) -> TopologyDefinition:
        return TopologyDefinition(
            name=self.topology_name,
            nodes=tuple(self.get_nodes()),
            edges=tuple(self.get_edges()),
        )

    def is_reachable(self, sender: str, receiver: str) -> bool:
        return nx.has_path(self.graph, sender, receiver)

    def shortest_path(self, sender: str, receiver: str) -> Tuple[str, ...]:
        return tuple(nx.shortest_path(self.graph, sender, receiver))

    def get_nodes(self) -> List[str]:
        """
        Return all agents in the topology.
        """

        return list(self.graph.nodes)

    def get_edges(self):
        """
        Return all directed communication edges.
        """

        return list(self.graph.edges)

    def number_of_agents(self) -> int:
        """
        Return the number of agents.
        """

        return self.graph.number_of_nodes()

    def number_of_connections(self) -> int:
        """
        Return the number of communication connections.
        """

        return self.graph.number_of_edges()

    # ---------------------------------------------------------
    # Topology configurations
    # ---------------------------------------------------------

    @staticmethod
    def layered(agents: List[str]):
        """
        True layered / hierarchical topology.

        Communication structure:

            Coordinator
                ↕
            Researcher
                ↕
            Analyst
                ↕
            Executor

        Only adjacent layers can communicate directly.
        """

        topology = CommunicationTopology(
            agents,
            topology_name="layered"
        )

        legacy_agents = {
            "coordinator",
            "researcher",
            "analyst",
            "executor",
        }
        if set(agents) == legacy_agents:
            topology = CommunicationTopology(agents, topology_name="layered")
            for first, second in (
                ("coordinator", "researcher"),
                ("researcher", "analyst"),
                ("analyst", "executor"),
            ):
                topology.add_bidirectional_connection(first, second)
            return topology

        required_agents = {
            "coordinator",
            "planner",
            "researcher-1",
            "researcher-2",
            "analyst-1",
            "analyst-2",
            "executor-1",
            "executor-2",
        }

        missing = required_agents - set(agents)

        if missing:
            raise ValueError(
                "Layered topology requires agents: "
                f"{sorted(required_agents)}. "
                f"Missing: {sorted(missing)}"
            )

        layers = [
            ["coordinator"],
            ["planner"],
            ["researcher-1", "researcher-2"],
            ["analyst-1", "analyst-2"],
            ["executor-1", "executor-2"],
        ]

        for layer in layers:
            for sender in layer:
                for receiver in layer:
                    if sender != receiver:
                        topology.add_connection(sender, receiver)

        for current_layer, next_layer in zip(layers, layers[1:]):
            for sender in current_layer:
                for receiver in next_layer:
                    topology.add_bidirectional_connection(sender, receiver)

        return topology


    @staticmethod
    def fully_connected(agents: List[str]):
        """
        fully_connected_p2p peer-to-peer topology.

        Every agent can communicate directly
        with every other agent.
        """

        topology = CommunicationTopology(
            agents,
            topology_name="fully_connected"
        )

        for sender in agents:
            for receiver in agents:

                if sender != receiver:
                    topology.add_connection(
                        sender,
                        receiver
                    )

        return topology

    @staticmethod
    def fully_connected_p2p(agents: List[str]):
        """Compatibility alias for the fully connected peer-to-peer topology."""
        return CommunicationTopology.fully_connected(agents)

    @staticmethod
    def centralized(agents: List[str]):
        """
        Centralized topology.

        All agents communicate through
        the Coordinator.
        """

        topology = CommunicationTopology(
            agents,
            topology_name="centralized"
        )

        coordinator = "coordinator"

        if coordinator not in agents:
            raise ValueError(
                "Centralized topology requires "
                "'coordinator' agent."
            )

        for agent in agents:

            if agent == coordinator:
                continue

            topology.add_bidirectional_connection(
                coordinator,
                agent
            )

        return topology

    def visualize(self):
        import matplotlib.pyplot as plt

        pos = {
            agent: (index % 3, -(index // 3))
            for index, agent in enumerate(self.agents)
        }

        nx.draw_networkx(
            self.graph,
            pos=pos,
            with_labels=True,
            arrows=True,
            node_size=2500,
            font_size=10,
            arrowsize=20,
        )

        plt.title(f"{self.__class__.__name__}: Communication Topology")
        plt.axis("off")
        plt.show()

    @staticmethod
    def shared_pool(agents: List[str]):
        """
        Shared-pool communication configuration.

        Agents do not communicate directly with one another.
        Communication is performed through the shared pool.

        The pool is represented as an infrastructure node in the
        formal graph so the topology can be inspected and analyzed.
        Runtime delivery still uses the environment's shared-pool
        storage and targeted mailbox semantics.
        """

        topology = CommunicationTopology(
            agents,
            topology_name="shared_pool"
        )

        pool = "shared_pool"
        topology.graph.add_node(pool)

        for agent in agents:
            topology.add_connection(agent, pool)
            topology.add_connection(pool, agent)

        return topology

    # ---------------------------------------------------------
    # Factory
    # ---------------------------------------------------------

    @staticmethod
    def create(
        name: str,
        agents: List[str]
    ):
        """
        Create a communication topology
        using the specified topology name.
        """

        name = name.lower()

        if name == "layered":
            return CommunicationTopology.layered(agents)

        if name in (
            "fully_connected",
            "peer_to_peer",
            "fully_connected_p2p",
            "fully_connect_p2p",
        ):
            return CommunicationTopology.fully_connected(agents)

        if name == "centralized":
            return CommunicationTopology.centralized(agents)

        if name == "shared_pool":
            return CommunicationTopology.shared_pool(agents)

        raise ValueError(
            f"Unknown topology: {name}. "
            f"Use layered, fully_connected_p2p, centralized, or shared_pool."
        )

        