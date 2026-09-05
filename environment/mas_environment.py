import uuid
from typing import TypedDict, Optional

import networkx as nx

from langgraph.graph import StateGraph, START, END

from agents.coordinator import CoordinatorAgent
from agents.researcher import ResearcherAgent
from agents.analyst import AnalystAgent
from agents.executor import ExecutorAgent

from tools.tool_manager import ToolManager
from tools.tool_request import ToolRequest
from tools.tool_control_plane import ToolControlPlane

from environment.events import MASEvent
from environment.topology import CommunicationTopology
from environment.memory import MemoryManager


# =============================================================
# LANGGRAPH STATE
# =============================================================

class MASState(TypedDict):

    # =====================================================
    # WORKFLOW / ORCHESTRATION STATE
    # =====================================================

    task: str

    plan: Optional[dict]

    # Agent outputs are NOT used as unrestricted
    # communication channels.
    #
    # Information that crosses agents must pass
    # through the communication topology.

    error: Optional[str]

    # =====================================================
    # FINAL SYSTEM OUTPUT
    # =====================================================

    final_result: Optional[str]

    report_file: Optional[dict]


# =============================================================
# MAS ENVIRONMENT
# =============================================================

class MASEnvironment:
    """
    LLM-based Multi-Agent System environment.

    Components:

        LangGraph
            -> workflow execution

        NetworkX
            -> communication topology

        AgentMemory / MemoryManager
            -> per-agent episodic memory

        ToolManager
            -> controlled tool execution

        MASEvent
            -> event logging


    Supported communication topologies:

        1. Layered
        2. Centralized
        3. Decentralized
        4. Shared Pool


    Layered:

        Coordinator
             |
             v
        Researcher
             |
             v
          Analyst
             |
             v
         Executor


    Centralized:

              Coordinator
             /     |      \
            v      v       v
        Researcher Analyst Executor


    Decentralized:

        Any agent <-> Any other agent


    Shared Pool:

        Coordinator ──┐
        Researcher  ──┤
        Analyst     ──┼──> Shared Pool
        Executor    ──┘


    Tool flow:

        Centralized:
            Agent
                ↓
            Coordinator
                ↓
            Tool Control Plane

        Layered, decentralized, shared_pool:
            Agent
                ↓
            Tool Control Plane

        Both routes continue through ToolManager and the
        external tool before returning to the requesting agent.
    """

    def __init__(
        self,
        topology_name="centralized"
    ):

        # =====================================================
        # AGENT DEFINITIONS
        # =====================================================

        self.agent_names = [
            "coordinator",
            "analyst",
            "researcher",
            "executor",
        ]

        self.agent_mailboxes = {
            agent: []
            for agent in self.agent_names
        }

        # =====================================================
        # MEMORY SYSTEM
        # =====================================================

        self.memory = MemoryManager(
            self.agent_names
        )

        # =====================================================
        # TOOL SYSTEM
        # =====================================================

        self.tool_manager = ToolManager()
        self.tool_control_plane = ToolControlPlane(
            self.tool_manager
        )

        # =====================================================
        # AGENTS
        # =====================================================

        self.coordinator = CoordinatorAgent(
            name="coordinator",
            memory=self.memory.get_memory(
                "coordinator"
            ),
            tool_manager=self.tool_manager,
            tool_control_plane=self.tool_control_plane
        )

        self.analyst = AnalystAgent(
            name="analyst",
            memory=self.memory.get_memory(
                "analyst"
            )
        )

        self.researcher = ResearcherAgent(
            name="researcher",
            memory=self.memory.get_memory(
                "researcher"
            )
        )

        self.executor = ExecutorAgent(
            name="executor",
            memory=self.memory.get_memory(
                "executor"
            )
        )

        self.agents = {
            "coordinator": self.coordinator,
            "analyst": self.analyst,
            "researcher": self.researcher,
            "executor": self.executor,
        }

        # =====================================================
        # COMMUNICATION TOPOLOGY
        # =====================================================

        self.topology_name = topology_name.lower()

        self.topology = CommunicationTopology.create(
            self.topology_name,
            self.agent_names
        )

        self.tool_manager.set_topology(
            self.topology_name
        )

        print(
            "\nCommunication topology:"
            + self.topology_name
        )

        print(
            self.topology.get_edges()
        )

        # =====================================================
        # EVENT LOG
        # =====================================================

        self.events = []

        # =====================================================
        # SHARED COMMUNICATION POOL
        # =====================================================

        self.shared_pool = []

        # =====================================================
        # BUILD LANGGRAPH WORKFLOW
        # =====================================================

        self.graph = self._build_graph()

    # =========================================================
    # EVENT LOGGING HELPERS
    # =========================================================

    @staticmethod
    def _content_length(content) -> int:
        """
        Return content length without storing the actual
        content inside the event log.
        """

        if content is None:
            return 0

        return len(str(content))

    @staticmethod
    def _argument_keys(arguments: dict) -> list:
        """
        Return only argument names.

        Actual argument values are deliberately excluded
        from the event log.
        """

        if not isinstance(arguments, dict):
            return []

        return list(arguments.keys())

    def log_event(
        self,
        event: MASEvent
    ):

        self.events.append(
            event
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"[{event.timestamp}] "
            f"{event.event_type.upper()}"
        )

        if event.sender:
            print(
                f"Sender   : {event.sender}"
            )

        if event.receiver:
            print(
                f"Receiver : {event.receiver}"
            )

        if event.content:
            print(
                f"Content  : {event.content}"
            )

        if event.tool_call:
            print(
                f"Tool     : {event.tool_call}"
            )

        if event.memory_update:
            print(
                f"Memory   : {event.memory_update}"
            )

        if event.token_usage:
            print(
                f"Tokens   : {event.token_usage}"
            )

        if event.metadata:
            print(
                f"Metadata : {event.metadata}"
            )

        print(
            "-" * 70
        )

    # =========================================================
    # COMMUNICATION
    # =========================================================

    # =========================================================
# COMMUNICATION
# =========================================================

    def send_message(
        self,
        sender: str,
        receiver: str,
        content: str,
        metadata=None
    ):
        """
        Deliver a message through the active communication topology.

        Normal topologies:
            Delivery is permitted only when NetworkX contains
            the corresponding directed edge.

        Shared pool:
            Message is written to the shared pool and becomes
            readable only by the intended receiver.
        """

        if sender not in self.agent_names:
            raise ValueError(
                f"Unknown sender: {sender}"
            )

        if receiver not in self.agent_names:
            raise ValueError(
                f"Unknown receiver: {receiver}"
            )

        if content is None:
            raise ValueError(
                "Message content cannot be None."
            )

        message_id = str(
            uuid.uuid4()
        )

        message = {
            "message_id": message_id,
            "sender": sender,
            "receiver": receiver,
            "content": content,
            "topology": self.topology_name,
            "metadata": metadata or {},
        }

        # =====================================================
        # SHARED POOL
        # =====================================================

        if self.topology_name == "shared_pool":

            self.shared_pool.append(
                message
            )

            self.log_event(
                MASEvent.create(
                    event_type="pool_write",

                    sender=sender,

                    receiver="shared_pool",

                    content=(
                        f"{sender} published a message "
                        f"for {receiver}"
                    ),

                    metadata={
                        "message_id":
                            message_id,

                        "target_agent":
                            receiver,

                        "topology":
                            "shared_pool",

                        "content_length":
                            self._content_length(
                                content
                            ),

                        "pool_size":
                            len(self.shared_pool),
                    }
                )
            )

            return message

        # =====================================================
        # DIRECT AGENT COMMUNICATION
        # =====================================================

        if not self.topology.can_communicate(
            sender,
            receiver
        ):
            raise ValueError(
                f"Communication not allowed under "
                f"{self.topology_name}: "
                f"{sender} -> {receiver}"
            )

        self.agent_mailboxes[
            receiver
        ].append(message)

        self.log_event(
            MASEvent.create(
                event_type="message",

                sender=sender,

                receiver=receiver,

                content=(
                    f"{sender} sent message "
                    f"to {receiver}"
                ),

                metadata={
                    "message_id":
                        message_id,

                    "topology":
                        self.topology_name,

                    "content_length":
                        self._content_length(
                            content
                        ),

                    "mailbox_size":
                        len(
                            self.agent_mailboxes[
                                receiver
                            ]
                        ),

                    **(
                        metadata
                        or {}
                    )
                }
            )
        )

        return message


    # =========================================================
    # RECEIVE AGENT MESSAGE
    # =========================================================

    def receive_agent_message(
        self,
        receiver: str,
        expected_sender=None
    ):
        """
        Retrieve the oldest unread topology-delivered message.

        Communication is consumed from the receiving agent's
        mailbox. Agent outputs are therefore never obtained
        from LangGraph state.
        """

        if receiver not in self.agent_names:
            raise ValueError(
                f"Unknown receiver: {receiver}"
            )

        # =====================================================
        # SHARED POOL
        # =====================================================

        if self.topology_name == "shared_pool":

            messages = [
                message
                for message in self.shared_pool
                if message["receiver"] == receiver
                and (
                    expected_sender is None
                    or message["sender"]
                    == expected_sender
                )
            ]

            if not messages:
                return None

            message = messages[0]

            self.shared_pool.remove(
                message
            )

            self.log_event(
                MASEvent.create(
                    event_type="pool_read",

                    receiver=receiver,

                    content=(
                        f"{receiver} retrieved "
                        f"a message from "
                        f"{message['sender']}"
                    ),

                    metadata={
                        "message_id":
                            message["message_id"],

                        "sender":
                            message["sender"],

                        "topology":
                            "shared_pool",

                        "remaining_pool_size":
                            len(self.shared_pool),
                    }
                )
            )

            return message["content"]

        # =====================================================
        # NORMAL TOPOLOGIES
        # =====================================================

        mailbox = self.agent_mailboxes[
            receiver
        ]

        for index, message in enumerate(
            mailbox
        ):

            if (
                expected_sender is None
                or message["sender"]
                == expected_sender
            ):

                message = mailbox.pop(
                    index
                )

                self.log_event(
                    MASEvent.create(
                        event_type="message_receive",

                        sender=message[
                            "sender"
                        ],

                        receiver=receiver,

                        content=(
                            f"{receiver} received "
                            f"a message from "
                            f"{message['sender']}"
                        ),

                        metadata={
                            "message_id":
                                message[
                                    "message_id"
                                ],

                            "topology":
                                self.topology_name,

                            "content_length":
                                self._content_length(
                                    message[
                                        "content"
                                    ]
                                ),

                            "remaining_mailbox_size":
                                len(mailbox),
                        }
                    )
                )

                return message[
                    "content"
                ]

        return None



    

    # =========================================================
    # SHARED POOL READ
    # =========================================================

    def read_shared_pool(self, receiver: str):

        if self.topology_name != "shared_pool":
            raise ValueError(
                "Shared pool is only available "
                "under shared_pool topology."
            )

        messages = [
            message
            for message in self.shared_pool
            if message["receiver"] == receiver
        ]

        self.log_event(
            MASEvent.create(
                event_type="pool_read",
                receiver=receiver,
                content=(
                    f"{receiver} retrieved "
                    f"{len(messages)} messages "
                    f"from shared pool"
                ),
                metadata={
                    "topology": "shared_pool",
                    "message_count": len(messages),
                    "pool_size": len(self.shared_pool),
                }
            )
        )

        return messages

    # =========================================================
    # TOOL REQUEST ROUTING
    # =========================================================

    def request_tool(
        self,
        requesting_agent: str,
        tool_name: str,
        arguments: dict
    ):
        """
        Submit a tool request through the Tool Control Plane.

        Tool authorization is independent of communication topology.

        Authorized:
            coordinator
            researcher
            analyst

        Unauthorized:
            executor
        """

        request_id = str(uuid.uuid4())

        argument_keys = self._argument_keys(
            arguments
        )

        # =====================================================
        # AUTHORIZATION
        # =====================================================

        authorized = self.tool_manager.is_allowed(
            requesting_agent,
            tool_name
        )

        # =====================================================
        # TOOL REQUEST EVENT
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="tool_request",

                sender=requesting_agent,

                receiver="tool_control_plane",

                content=(
                    f"{requesting_agent} requested "
                    f"tool '{tool_name}'"
                ),

                tool_call=tool_name,

                request_id=request_id,

                metadata={
                    "argument_keys": argument_keys,
                    "argument_count": len(argument_keys),
                    "topology": self.topology_name,
                    "requesting_agent": requesting_agent,
                    "tool_name": tool_name,
                    "authorization_result": (
                        "allowed"
                        if authorized
                        else "denied"
                    ),
                },
            )
        )

        # =====================================================
        # DENIED
        # =====================================================

        if not authorized:

            self.log_event(
                MASEvent.create(
                    event_type="tool_denied",

                    sender=requesting_agent,

                    receiver="tool_control_plane",

                    tool_call=tool_name,

                    request_id=request_id,

                    metadata={
                        "requesting_agent":
                            requesting_agent,

                        "tool_name":
                            tool_name,

                        "topology":
                            self.topology_name,

                        "authorization_result":
                            "denied",

                        "reason":
                            "Agent is not authorized "
                            "for this tool."
                    },
                )
            )

            raise PermissionError(
                f"Agent '{requesting_agent}' is not "
                f"authorized to use tool "
                f"'{tool_name}'."
            )

        # =====================================================
        # TOOL CONTROL PLANE
        # =====================================================

        if (
            self.topology_name == "centralized"
            and requesting_agent != "coordinator"
        ):

            self.publish_agent_result(
                sender=requesting_agent,
                receiver="coordinator",
                content={
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                metadata={
                    "message_type": "tool_request",
                    "request_id": request_id,
                },
            )

            routed_request = self.receive_agent_message(
                receiver="coordinator",
                expected_sender=requesting_agent,
            )

            result = self.coordinator.handle_tool_request(
                agent=requesting_agent,
                tool_name=routed_request["tool_name"],
                arguments=routed_request["arguments"],
            )

            self.publish_agent_result(
                sender="coordinator",
                receiver=requesting_agent,
                content=result,
                metadata={
                    "message_type": "tool_result",
                    "request_id": request_id,
                },
            )

            result = self.receive_agent_message(
                receiver=requesting_agent,
                expected_sender="coordinator",
            )

            result_count = (
                len(result)
                if isinstance(result, list)
                else 1
            )

            self.log_event(
                MASEvent.create(
                    event_type="tool_result",

                    sender="coordinator",

                    receiver=requesting_agent,

                    content=(
                        f"Tool '{tool_name}' completed "
                        f"successfully"
                    ),

                    tool_call=tool_name,

                    request_id=request_id,

                    result_count=result_count,

                    metadata={
                        "requesting_agent":
                            requesting_agent,

                        "result_type":
                            type(result).__name__,

                        "result_count":
                            result_count,

                        "status":
                            "success",

                        "routed_by":
                            "coordinator"
                    }
                )
            )

            return result

        self.log_event(
            MASEvent.create(
                event_type="tool_forward",

                sender=requesting_agent,

                receiver="tool_control_plane",

                content=(
                    f"{requesting_agent} submitted "
                    f"'{tool_name}' request"
                ),

                tool_call=tool_name,

                request_id=request_id,

                metadata={
                    "requesting_agent":
                        requesting_agent,

                    "argument_keys":
                        argument_keys,

                    "argument_count":
                        len(argument_keys),

                    "topology":
                        self.topology_name
                }
            )
        )

        # =====================================================
        # TOOL EXECUTION
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="tool_execution",

                sender="tool_control_plane",

                receiver=tool_name,

                content=(
                    f"Executing tool '{tool_name}'"
                ),

                tool_call=tool_name,

                request_id=request_id,

                metadata={
                    "requesting_agent":
                        requesting_agent
                }
            )
        )

        # =====================================================
        # EXECUTE
        # =====================================================

        try:

            request = ToolRequest(
                agent=requesting_agent,
                tool_name=tool_name,
                arguments=arguments,
                request_id=request_id,
            )

            result = self.tool_control_plane.submit(
                request,
                submitted_by=requesting_agent,
            )

        except Exception as exc:

            self.log_event(
                MASEvent.create(
                    event_type="tool_error",

                    sender="tool_control_plane",

                    receiver=requesting_agent,

                    content=(
                        f"Tool '{tool_name}' failed"
                    ),

                    tool_call=tool_name,

                    request_id=request_id,

                    metadata={
                        "requesting_agent":
                            requesting_agent,

                        "error_type":
                            type(exc).__name__,

                        "error_message":
                            str(exc),

                        "status":
                            "failed"
                    }
                )
            )

            raise

        # =====================================================
        # TOOL RESULT
        # =====================================================

        result_count = (
            len(result)
            if isinstance(result, list)
            else 1
        )

        self.log_event(
            MASEvent.create(
                event_type="tool_result",

                sender=tool_name,

                receiver=requesting_agent,

                content=(
                    f"Tool '{tool_name}' completed "
                    f"successfully"
                ),

                tool_call=tool_name,

                request_id=request_id,

                result_count=result_count,

                metadata={
                    "requesting_agent":
                        requesting_agent,

                    "result_type":
                        type(result).__name__,

                    "result_count":
                        result_count,

                    "status":
                        "success"
                }
            )
        )

        # =====================================================
        # RESULT DELIVERY
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="tool_result_delivery",

                sender="tool_control_plane",

                receiver=requesting_agent,

                content=(
                    f"Tool '{tool_name}' result "
                    f"delivered to {requesting_agent}"
                ),

                tool_call=tool_name,

                request_id=request_id,

                result_count=result_count,

                metadata={
                    "requesting_agent":
                        requesting_agent,

                    "result_type":
                        type(result).__name__,

                    "result_count":
                        result_count
                }
            )
        )

        return result
    # =========================================================
    # MEMORY WRITE EVENT
    # =========================================================

    def log_memory_write(
        self,
        agent_name: str,
        content: str,
        importance: int = 5,
        metadata=None
    ):

        memory = self.memory.add(
            agent_name=agent_name,
            content=content,
            importance=importance,
            metadata=metadata,
        )

        self.log_event(
            MASEvent.create(
                event_type="memory_write",

                sender=agent_name,

                content="Memory written",

                memory_update=memory.memory_id,

                metadata={
                    "importance":
                        importance,

                    "agent":
                        agent_name,

                    "content_length":
                        self._content_length(
                            content
                        ),

                    **(metadata or {})
                }
            )
        )

        return memory

    # =========================================================
    # MEMORY READ EVENT
    # =========================================================

    def read_memory(
        self,
        agent_name: str,
        query=None,
        top_k=3
    ):

        memories = self.memory.retrieve(
            agent_name=agent_name,
            query=query,
            top_k=top_k,
        )

        self.log_event(
            MASEvent.create(
                event_type="memory_read",

                receiver=agent_name,

                content=(
                    f"Retrieved "
                    f"{len(memories)} memories."
                ),

                metadata={
                    "query":
                        query,

                    "top_k":
                        top_k,

                    "memory_count":
                        len(memories),
                }
            )
        )

        return memories

    # =========================================================
    # COORDINATOR NODE
    # =========================================================

   # =========================================================
# COORDINATOR NODE
# =========================================================

    def coordinator_node(
        self,
        state: MASState
    ):
        """
        Coordinator receives the user task and creates the
        execution plan.

        The Coordinator does not perform research, analysis,
        or execution itself.
        """

        task = state["task"]

        # =====================================================
        # TASK RECEIVED
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="task_received",

                receiver="coordinator",

                content="Coordinator received user task",

                metadata={
                    "topology":
                        self.topology_name,

                    "task_length":
                        self._content_length(task)
                }
            )
        )

        # =====================================================
        # RECORD TASK IN MEMORY
        # =====================================================

        self.log_memory_write(
            agent_name="coordinator",

            content=(
                f"Received user task: {task}"
            ),

            importance=8,

            metadata={
                "event":
                    "task_received"
            }
        )

        # =====================================================
        # READ COORDINATOR MEMORY
        # =====================================================

        self.read_memory(
            agent_name="coordinator",

            query=task,

            top_k=3
        )

        # =====================================================
        # CREATE PLAN
        # =====================================================

        try:

            plan = self.coordinator.create_plan(
                task
            )

        except ValueError as e:

            self.log_event(
                MASEvent.create(
                    event_type="agent_failure",

                    sender="coordinator",

                    content=str(e),

                    metadata={
                        "stage":
                            "planning",

                        "failure_type":
                            "invalid_structured_output"
                    }
                )
            )

            return {
                "plan": None,
                "error": str(e)
            }

        # =====================================================
        # LOG PLAN
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="task_decomposition",

                sender="coordinator",

                content=(
                    "Coordinator created task plan"
                ),

                metadata={
                    "topology":
                        self.topology_name,

                    "plan_stages":
                        list(plan.keys())
                        if isinstance(plan, dict)
                        else [],

                    "stage_count":
                        len(plan)
                        if isinstance(plan, dict)
                        else 0
                }
            )
        )

        self.publish_agent_result(
            sender="coordinator",
            receiver="researcher",
            content=plan["research"],
            metadata={
                "stage": "research_assignment"
            }
        )

        return {
            "plan": plan,
            "error": None
        }
    # =========================================================
    # RESEARCHER NODE
    # =========================================================

   # =========================================================
# RESEARCHER NODE
# =========================================================

    def research_node(
        self,
        state: MASState
    ):
        """
        Execute the Researcher stage.

        Communication:

            Layered:
                Coordinator -> Researcher
                Researcher -> Analyst

            Centralized:
                Coordinator -> Researcher
                Researcher -> Coordinator -> Analyst

            Decentralized:
                Coordinator -> Researcher
                Researcher -> Analyst

            Shared Pool:
                Coordinator -> Pool -> Researcher
                Researcher -> Pool -> Analyst
        """

        # =====================================================
        # RECEIVE COORDINATOR INSTRUCTION
        # =====================================================

        research_instruction = self.receive_agent_message(
            receiver="researcher",
            expected_sender="coordinator"
        )

        if research_instruction is None:
            raise ValueError(
                "Researcher received no research instruction."
            )

        # =====================================================
        # SHARED POOL READ
        # =====================================================

        # NOTE:
        # receive_agent_message() should already perform the
        # pool read for shared_pool.
        #
        # Therefore DO NOT call read_shared_pool() separately
        # if receive_agent_message() handles it.

        # =====================================================
        # RESEARCHER MEMORY
        # =====================================================

        self.read_memory(
            agent_name="researcher",
            query=research_instruction,
            top_k=3
        )

        # =====================================================
        # RESEARCHER TOOL DECISION
        # =====================================================

        tool_request = (
            self.researcher.create_tool_request(
                research_instruction
            )
        )

        tool_results = []

        # =====================================================
        # TOOL REQUEST
        # =====================================================

        if tool_request:

            try:

                result = self.request_tool(
                    requesting_agent=tool_request["agent"],
                    tool_name=tool_request["tool_name"],
                    arguments=tool_request["arguments"]
                )

            except PermissionError:

                result = []

            if isinstance(result, list):

                tool_results.extend(result)

            else:

                tool_results.append(result)

        # =====================================================
        # RESEARCHER EXECUTION
        # =====================================================

        research_result = self.researcher.run(
            research_instruction,
            tool_results=tool_results
        )

        # =====================================================
        # LOG RESEARCH RESULT
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="agent_result",
                sender="researcher",
                content=(
                    "Researcher completed "
                    "research stage"
                ),
                metadata={
                    "stage": "research",
                    "topology": self.topology_name,
                    "used_external_tools": bool(tool_results),
                    "source_count": len(tool_results),
                    "content_length": self._content_length(
                        research_result
                    )
                }
            )
        )

        # =====================================================
        # RESEARCHER MEMORY
        # =====================================================

        self.log_memory_write(
            agent_name="researcher",
            content=(
                f"Research result for assignment:\n"
                f"{research_instruction}\n\n"
                f"{research_result}"
            ),
            importance=7,
            metadata={
                "stage": "research",
                "used_external_tools": bool(tool_results),
                "source_count": len(tool_results)
            }
        )

        # =====================================================
        # PUBLISH RESEARCH RESULT
        # =====================================================

        self.publish_agent_result(
            sender="researcher",
            receiver="analyst",
            content=research_result,
            metadata={
                "stage": "research",
                "source_count": len(tool_results)
            }
        )

        # =====================================================
        # RETURN
        # =====================================================
        #
        # IMPORTANT:
        # Do NOT return research_result.
        #
        # Communication happens through the topology layer.
        #
        # LangGraph controls execution order only.
        # =====================================================

        return {}
    # =========================================================
    # ANALYST NODE
    # =========================================================

    # =========================================================
# ANALYST NODE
# =========================================================

    def analysis_node(
        self,
        state: MASState
    ):
        """
        Execute the Analyst stage.

        Research information is obtained exclusively through
        the topology-controlled communication mechanism.
        """

        plan = state["plan"]

        if plan is None:
            raise ValueError(
                "Analysis node received no plan."
            )

        analysis_instruction = plan[
            "analysis"
        ]

        # =====================================================
        # RECEIVE RESEARCH RESULT
        # =====================================================

        research = self.receive_agent_message(
            receiver="analyst",
            expected_sender=(
                "coordinator"
                if self.topology_name == "centralized"
                else "researcher"
            )
        )

        if research is None:
            raise ValueError(
                "Analyst received no research message."
            )

        # =====================================================
        # RESEARCH SOURCE COUNT
        # =====================================================
        #
        # Source contents are no longer carried through
        # LangGraph state. The analyst therefore receives
        # only the published research result.
        #
        # If source metadata is required experimentally,
        # transmit it explicitly as message metadata rather
        # than through hidden state.
        # =====================================================

        research_sources = []

        # =====================================================
        # PREPARE ANALYSIS INPUT
        # =====================================================

        analysis_message = (
            f"Analysis assignment:\n"
            f"{analysis_instruction}\n\n"
            f"Research findings:\n"
            f"{research}"
        )

        # =====================================================
        # ANALYST MEMORY
        # =====================================================

        self.read_memory(
            agent_name="analyst",
            query=analysis_instruction,
            top_k=3
        )

        # =====================================================
        # ANALYST TOOL DECISION
        # =====================================================

        tool_request = (
            self.analyst.create_tool_request(
                analysis_instruction=
                    analysis_instruction,

                research_information=
                    research,

                research_sources=
                    research_sources
            )
        )

        tool_results = []

        if tool_request:

            try:

                result = self.request_tool(
                    requesting_agent=
                        tool_request["agent"],

                    tool_name=
                        tool_request["tool_name"],

                    arguments=
                        tool_request["arguments"]
                )

            except PermissionError:

                result = []

            if isinstance(result, list):

                tool_results.extend(
                    result
                )

            else:

                tool_results.append(
                    result
                )

        # =====================================================
        # ANALYST RUN
        # =====================================================

        analysis_result = self.analyst.run(
            analysis_instruction,

            research,

            research_sources=
                research_sources,

            tool_results=
                tool_results
        )

        # =====================================================
        # LOG ANALYSIS RESULT
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="agent_result",

                sender="analyst",

                content=(
                    "Analyst completed "
                    "analysis stage"
                ),

                metadata={
                    "stage":
                        "analysis",

                    "topology":
                        self.topology_name,

                    "used_external_tools":
                        bool(tool_results),

                    "content_length":
                        self._content_length(
                            analysis_result
                        )
                }
            )
        )

        # =====================================================
        # ANALYST MEMORY
        # =====================================================

        self.log_memory_write(
            agent_name="analyst",

            content=(
                f"Analysis assignment:\n"
                f"{analysis_instruction}\n\n"
                f"Analysis result:\n"
                f"{analysis_result}"
            ),

            importance=7,

            metadata={
                "stage":
                    "analysis",

                "used_external_tools":
                    bool(tool_results)
            }
        )

        # =====================================================
        # PUBLISH ANALYSIS RESULT
        # =====================================================

        self.publish_agent_result(
            sender="analyst",
            receiver="executor",
            content=analysis_result,
            metadata={
                "stage":
                    "analysis"
            }
        )

        # =====================================================
        # RETURN ORCHESTRATION STATE ONLY
        # =====================================================

        return {}
  
   # =========================================================
    # EXECUTOR NODE
    # =========================================================

    def execution_node(
        self,
        state: MASState
    ):
        """
        Execute the Executor stage.

        The Executor receives Analyst information exclusively
        through topology-controlled communication.

        The Executor has no direct tool authorization.
        """

        plan = state["plan"]

        if plan is None:
            raise ValueError(
                "Execution node received no plan."
            )

        execution_instruction = plan[
            "execution"
        ]

        # =====================================================
        # RECEIVE ANALYST RESULT
        # =====================================================

        analysis = self.receive_agent_message(
            receiver="executor",
            expected_sender=(
                "coordinator"
                if self.topology_name == "centralized"
                else "analyst"
            )
        )

        if analysis is None:
            raise ValueError(
                "Executor received no analysis message."
            )

        # =====================================================
        # EXECUTION INPUT
        # =====================================================

        execution_message = (
            f"Execution assignment:\n"
            f"{execution_instruction}\n\n"
            f"Analyst findings:\n"
            f"{analysis}"
        )

        # =====================================================
        # EXECUTOR MEMORY
        # =====================================================

        self.read_memory(
            agent_name="executor",
            query=execution_instruction,
            top_k=3
        )

        # =====================================================
        # IMPORTANT:
        # EXECUTOR HAS NO TOOL ACCESS
        # =====================================================
        #
        # Do NOT call:
        #
        #     self.executor.create_tool_request()
        #
        # and do NOT call request_tool() from the Executor.
        #
        # Tool authorization remains topology-independent.
        # =====================================================

        tool_results = []

        # =====================================================
        # EXECUTOR RUN
        # =====================================================

        execution_result = self.executor.run(
            execution_instruction,

            analysis,

            tool_results=tool_results
        )

        # =====================================================
        # LOG EXECUTION RESULT
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="agent_result",

                sender="executor",

                content=execution_result,

                metadata={
                    "stage":
                        "execution",

                    "topology":
                        self.topology_name,

                    "used_external_tools":
                        False,

                    "content_length":
                        self._content_length(
                            execution_result
                        )
                }
            )
        )

        # =====================================================
        # EXECUTOR MEMORY
        # =====================================================

        self.log_memory_write(
            agent_name="executor",

            content=(
                f"Execution assignment:\n"
                f"{execution_instruction}\n\n"
                f"Execution result:\n"
                f"{execution_result}"
            ),

            importance=8,

            metadata={
                "stage":
                    "execution",

                "used_external_tools":
                    False
            }
        )

        # =====================================================
        # RETURN RESULT THROUGH COMMUNICATION
        # =====================================================

        self.publish_agent_result(
            sender="executor",
            receiver="coordinator",
            content=execution_result,
            metadata={
                "stage":
                    "execution"
            }
        )

        # =====================================================
        # RETURN ORCHESTRATION STATE ONLY
        # =====================================================

        return {}
   
    # =========================================================
    # FINAL COORDINATOR NODE
    # =========================================================

    def final_node(
        self,
        state: MASState
    ):
        """
        Final Coordinator stage.

        The Coordinator receives the Executor result through
        the communication topology rather than LangGraph state.
        """

        # =====================================================
        # RECEIVE EXECUTION RESULT
        # =====================================================

        execution = self.receive_agent_message(
            receiver="coordinator",
            expected_sender=(
                "researcher"
                if self.topology_name == "layered"
                else "executor"
            )
        )

        if execution is None:
            raise ValueError(
                "Coordinator received no execution result."
            )

        # =====================================================
        # COORDINATOR MEMORY
        # =====================================================

        self.read_memory(
            agent_name="coordinator",

            query=state["task"],

            top_k=3
        )

        # =====================================================
        # FINAL AGGREGATION
        # =====================================================

        final_result = self.coordinator.aggregate(
            state["task"],
            execution
        )

        # =====================================================
        # LOG FINAL RESULT
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="final_result",

                sender="coordinator",

                content=(
                    "Coordinator generated "
                    "final result"
                ),

                metadata={
                    "topology":
                        self.topology_name,

                    "content_length":
                        self._content_length(
                            final_result
                        )
                }
            )
        )

        return {
            "final_result":
                final_result
        }
    # =========================================================
    # REPORT WRITER NODE
    # =========================================================

    def report_writer_node(
        self,
        state: MASState
    ):

        final_report = state.get(
            "final_result"
        )

        if not final_report:

            raise ValueError(
                "Cannot create report: "
                "final result is empty."
            )

        # =====================================================
        # CREATE REPORT TITLE
        # =====================================================

        task = state.get(
            "task",
            "Multi-Agent System Analysis"
        )

        report_title = (
            "Multi-Agent System Analysis Report"
        )

        # =====================================================
        # CREATE REPORT WRITER REQUEST
        # =====================================================

        report_request = ToolRequest(
            agent="coordinator",

            tool_name="report_writer",

            arguments={
                "title":
                    report_title,

                "filename":
                    None,

                "content":
                    final_report,

                "task":
                    task
            }
        )

        # =====================================================
        # COORDINATOR -> REPORT WRITER
        #
        # Routed through ToolManager.
        # =====================================================

        report_result = self.request_tool(
            requesting_agent=
                report_request.agent,

            tool_name=
                report_request.tool_name,

            arguments=
                report_request.arguments
        )

        # =====================================================
        # LOG REPORT CREATION
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="report_created",

                sender="coordinator",

                receiver="report_writer",

                content=(
                    f"Report created: "
                    f"{report_result.get(
                        'filename',
                        'unknown'
                    )}"
                ),

                tool_call="report_writer",

                metadata={
                    "file_type":
                        "docx",

                    "path":
                        report_result.get(
                            "path"
                        ),

                    "status":
                        report_result.get(
                            "status"
                        )
                }
            )
        )

        return {
            "report_file":
                report_result
        }

    # =========================================================
    # BUILD LANGGRAPH
    # =========================================================

    def _build_graph(
        self
    ):

        workflow = StateGraph(
            MASState
        )

        # =====================================================
        # NODES
        # =====================================================

        workflow.add_node(
            "coordinator",
            self.coordinator_node
        )

        workflow.add_node(
            "researcher",
            self.research_node
        )

        workflow.add_node(
            "analyst",
            self.analysis_node
        )

        workflow.add_node(
            "executor",
            self.execution_node
        )

        workflow.add_node(
            "final",
            self.final_node
        )

        workflow.add_node(
            "report_writer",
            self.report_writer_node
        )

        # =====================================================
        # EXECUTION EDGES
        # =====================================================

        workflow.add_edge(
            START,
            "coordinator"
        )

        workflow.add_edge(
            "coordinator",
            "researcher"
        )

        workflow.add_edge(
            "researcher",
            "analyst"
        )

        workflow.add_edge(
            "analyst",
            "executor"
        )

        workflow.add_edge(
            "executor",
            "final"
        )

        workflow.add_edge(
            "final",
            "report_writer"
        )

        workflow.add_edge(
            "report_writer",
            END
        )

        # =====================================================
        # COMPILE
        # =====================================================

        return workflow.compile()

    # =========================================================
# PUBLISH AGENT RESULT
# =========================================================

    def publish_agent_result(
        self,
        sender,
        receiver,
        content,
        metadata=None
    ):
        """
        Publish an agent result according to the active communication topology.
        """

        if self.topology_name == "shared_pool":
            return self.send_message(
                sender=sender,
                receiver=receiver,
                content=content,
                metadata=metadata
            )

        elif self.topology_name == "centralized":

            # Sender sends result to Coordinator.
            if sender != "coordinator":
                self.send_message(
                    sender=sender,
                    receiver="coordinator",
                    content=content,
                    metadata=metadata
                )

            # Coordinator forwards the information to the destination agent.
            if receiver != "coordinator":
                self.send_message(
                    sender="coordinator",
                    receiver=receiver,
                    content=content,
                    metadata=metadata
                )

            return content

        elif self.topology_name == "layered":

            if self.topology.can_communicate(
                sender,
                receiver
            ):
                return self.send_message(
                    sender=sender,
                    receiver=receiver,
                    content=content,
                    metadata=metadata
                )

            path = nx.shortest_path(
                self.topology.get_graph(),
                sender,
                receiver
            )

            for current, next_agent in zip(
                path,
                path[1:]
            ):
                self.send_message(
                    sender=current,
                    receiver=next_agent,
                    content=content,
                    metadata=metadata
                )

                if next_agent != receiver:
                    content = self.receive_agent_message(
                        receiver=next_agent,
                        expected_sender=current
                    )

            return content

        else:
            # Decentralized
            return self.send_message(
                sender=sender,
                receiver=receiver,
                content=content,
                metadata=metadata
            )
    # =========================================================
    # EXECUTE TASK
    # =========================================================

    def execute_task(
        self,
        task: str
    ):
        self.agent_mailboxes = {
        agent: []
        for agent in self.agent_names
    }

        # =====================================================
        # CLEAR EVENT LOG
        # =====================================================

        self.events = []

        # =====================================================
        # CLEAR SHARED POOL
        #
        # Each execute_task() represents a new episode.
        # =====================================================

        self.shared_pool = []

        # =====================================================
        # INITIAL LANGGRAPH STATE
        # =====================================================

        initial_state: MASState = {

            "task":
                task,

            "plan":
                None,

            "research":
                None,

            "research_sources":
                [],

            "analysis":
                None,

            "execution":
                None,

            "final_result":
                None,

            "report_file":
                None,

            "error":
                None,
        }

        # =====================================================
        # EXECUTE GRAPH
        # =====================================================

        result = self.graph.invoke(
            initial_state
        )

        return result[
            "final_result"
        ]

    # =========================================================
    # EVENT ACCESS
    # =========================================================

    def get_events(self):

        return [
            event.to_dict()
            for event in self.events
        ]

    # =========================================================
    # TOPOLOGY ACCESS
    # =========================================================

    def get_topology(self):

        return self.topology.get_graph()

    # =========================================================
    # MEMORY ACCESS
    # =========================================================

    def get_agent_memories(
        self,
        agent_name
    ):

        return [
            memory.to_dict()
            for memory in self.memory.get_all(
                agent_name
            )
        ]

    def get_all_memories(self):

        return {
            agent_name: [
                memory.to_dict()
                for memory in self.memory.get_all(
                    agent_name
                )
            ]

            for agent_name in self.agent_names
        }

    # =========================================================
    # CLEAR MEMORIES
    # =========================================================

    def clear_memories(
        self
    ):

        self.memory.clear()