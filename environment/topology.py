import networkx as nx
from typing import List


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

        required_agents = {
            "coordinator",
            "researcher",
            "analyst",
            "executor"
        }

        missing = required_agents - set(agents)

        if missing:
            raise ValueError(
                "Layered topology requires agents: "
                f"{sorted(required_agents)}. "
                f"Missing: {sorted(missing)}"
            )

        # Layer 1 <-> Layer 2
        topology.add_bidirectional_connection(
            "coordinator",
            "researcher"
        )

        # Layer 2 <-> Layer 3
        topology.add_bidirectional_connection(
            "researcher",
            "analyst"
        )

        # Layer 3 <-> Layer 4
        topology.add_bidirectional_connection(
            "analyst",
            "executor"
        )

        return topology


    @staticmethod
    def decentralized(agents: List[str]):
        """
        Decentralized peer-to-peer topology.

        Every agent can communicate directly
        with every other agent.
        """

        topology = CommunicationTopology(agents)

        for sender in agents:
            for receiver in agents:

                if sender != receiver:
                    topology.add_connection(
                        sender,
                        receiver
                    )

        return topology

    @staticmethod
    def centralized(agents: List[str]):
        """
        Centralized topology.

        All agents communicate through
        the Coordinator.
        """

        topology = CommunicationTopology(agents)

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
            "coordinator": (0, 3),
            "analyst": (0, 2),
            "researcher": (0, 1),
            "executor": (0, 0),
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
        Communication is performed through the shared pool,
        which is infrastructure external to the agent topology.
        """

        topology = CommunicationTopology(
            agents,
            topology_name="shared_pool"
        )

        # No agent-to-agent edges are added.
        # The shared pool is infrastructure and is therefore
        # not represented as an agent-topology node.

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

        if name == "decentralized":
            return CommunicationTopology.decentralized(agents)

        if name == "centralized":
            return CommunicationTopology.centralized(agents)

        if name == "shared_pool":
            return CommunicationTopology.shared_pool(agents)

        raise ValueError(
            f"Unknown topology: {name}. "
            f"Use layered, decentralized, "
            f"centralized, or shared_pool."
        )

        