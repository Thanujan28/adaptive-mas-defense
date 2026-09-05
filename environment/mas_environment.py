import uuid
from typing import TypedDict, Optional

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
    """
    Shared state maintained by the LangGraph workflow.
    """

    task: str

    plan: Optional[dict]

    research: Optional[str]

    # Structured source evidence collected by Researcher.
    research_sources: Optional[list]

    analysis: Optional[str]

    execution: Optional[str]

    # Final result produced by Coordinator.
    final_result: Optional[str]

    # Result returned by ReportWriterTool.
    report_file: Optional[dict]

    error: Optional[str]


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

    def send_message(
        self,
        sender: str,
        receiver: str,
        content: str
    ):
        """
        Send a message according to the configured
        communication topology.

        For:

            layered
            centralized
            decentralized

        communication is validated directly against
        the NetworkX topology.

        For:

            shared_pool

        the message is published to the shared
        communication pool instead of creating a
        direct agent-to-agent communication path.
        """

        # =====================================================
        # VALIDATE AGENTS
        # =====================================================

        if sender not in self.agent_names:
            raise ValueError(
                f"Unknown sender: {sender}"
            )

        if receiver not in self.agent_names:
            raise ValueError(
                f"Unknown receiver: {receiver}"
            )

        # =====================================================
        # SHARED POOL COMMUNICATION
        # =====================================================

        if self.topology_name == "shared_pool":

            # -------------------------------------------------
            # Store actual message in internal pool.
            #
            # The complete message is NOT placed in the
            # security event log.
            # -------------------------------------------------

            self.shared_pool.append(
                {
                    "message_id":
                        str(uuid.uuid4()),

                    "sender":
                        sender,

                    "receiver":
                        receiver,

                    "content":
                        content,
                }
            )

            # -------------------------------------------------
            # Log pool write
            # -------------------------------------------------

            self.log_event(
                MASEvent.create(
                    event_type="pool_write",

                    sender=sender,

                    receiver="shared_pool",

                    content=(
                        f"{sender} published message "
                        f"to shared pool"
                    ),

                    metadata={
                        "topology":
                            self.topology_name,

                        "target_agent":
                            receiver,

                        "content_length":
                            self._content_length(
                                content
                            ),

                        "pool_size":
                            len(
                                self.shared_pool
                            )
                    }
                )
            )

            return content

        # =====================================================
        # DIRECT COMMUNICATION
        # =====================================================

        if not self.topology.can_communicate(
            sender,
            receiver
        ):

            raise ValueError(
                f"Communication not allowed: "
                f"{sender} -> {receiver}"
            )

        # =====================================================
        # LOG DIRECT MESSAGE
        # =====================================================

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
                    "topology":
                        self.topology_name,

                    "content_length":
                        self._content_length(
                            content
                        )
                }
            )
        )

        return content

    # =========================================================
    # SHARED POOL READ
    # =========================================================

    def read_shared_pool(
        self,
        receiver: str
    ):
        """
        Retrieve messages intended for an agent
        from the shared communication pool.

        Only metadata is logged.
        Actual message contents remain in memory.
        """

        if self.topology_name != "shared_pool":

            raise ValueError(
                "Shared pool can only be accessed "
                "when using shared_pool topology."
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
                    f"{receiver} retrieved messages "
                    f"from shared pool"
                ),

                metadata={
                    "topology":
                        self.topology_name,

                    "message_count":
                        len(messages),

                    "pool_size":
                        len(self.shared_pool)
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
        Route a tool request through the Coordinator.

        Full tool arguments are NOT stored in the event log.

        Only:

            argument_keys
            argument_count
            result_count
            result_type
            status

        are recorded.

        Full arguments are still passed internally
        to the ToolManager.
        """

        request_id = str(
            uuid.uuid4()
        )

        argument_keys = self._argument_keys(
            arguments
        )

        authorized = self.tool_manager.is_allowed(
            requesting_agent,
            tool_name
        )

        centralized_route = (
            self.topology_name == "centralized"
        )

        coordinator_forwardable = (
            centralized_route
            and requesting_agent in {
                "researcher",
                "analyst",
            }
            and tool_name in self.tool_manager.permissions[
                "centralized"
            ]["coordinator"]
        )

        request_authorized = (
            authorized
            or coordinator_forwardable
        )

        request_receiver = (
            "coordinator"
            if centralized_route
            else "tool_control_plane"
        )

        self.log_event(
            MASEvent.create(
                event_type="tool_request",
                sender=requesting_agent,
                receiver=request_receiver,
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
                    "agent": requesting_agent,
                    "tool_name": tool_name,
                    "authorization_result": (
                        "allowed"
                        if request_authorized
                        else "denied"
                    ),
                },
            )
        )

        if not request_authorized:
            self.log_event(
                MASEvent.create(
                    event_type="tool_denied",
                    sender=requesting_agent,
                    receiver="coordinator",
                    tool_call=tool_name,
                    request_id=request_id,
                    metadata={
                        "agent": requesting_agent,
                        "tool_name": tool_name,
                        "topology": self.topology_name,
                        "authorization_result": "denied",
                        "reason": (
                            "Agent is not authorized for this tool "
                            "in the active topology."
                        ),
                    },
                )
            )

            raise PermissionError(
                f"Agent '{requesting_agent}' is not authorized "
                f"to use tool '{tool_name}' "
                f"in topology '{self.topology_name}'."
            )

        route_sender = (
            "coordinator"
            if centralized_route
            else requesting_agent
        )

        route_receiver = (
            "tool_manager"
            if centralized_route
            else "tool_control_plane"
        )

        route_description = (
            f"Coordinator forwarded '{tool_name}' request"
            if centralized_route
            else f"{requesting_agent} submitted '{tool_name}' request"
        )

        # =====================================================
        # AGENT/COORDINATOR -> TOOL CONTROL PLANE
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="tool_forward",

                sender=route_sender,

                receiver=route_receiver,

                content=route_description,

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
        # TOOL CONTROL PLANE -> TOOL
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
        # EXECUTE TOOL
        # =====================================================

        try:

            request = ToolRequest(
                agent=requesting_agent,
                tool_name=tool_name,
                arguments=arguments,
                request_id=request_id,
            )

            if centralized_route:
                result = self.coordinator.handle_tool_request(
                    agent=requesting_agent,
                    tool_name=tool_name,
                    arguments=arguments
                )
            else:
                result = self.tool_control_plane.submit(
                    request,
                    submitted_by=requesting_agent,
                )

        except Exception as exc:

            self.log_event(
                MASEvent.create(
                    event_type="tool_error",

                    sender=tool_name,

                    receiver="coordinator",

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

                        "argument_keys":
                            argument_keys,

                        "argument_count":
                            len(argument_keys),

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

        result_receiver = (
            "coordinator"
            if centralized_route
            else requesting_agent
        )

        delivery_sender = (
            "coordinator"
            if centralized_route
            else "tool_control_plane"
        )

        self.log_event(
            MASEvent.create(
                event_type="tool_result",

                sender=tool_name,

                receiver=result_receiver,

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
        # COORDINATOR -> AGENT
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="tool_result_delivery",

                sender=delivery_sender,

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

    def coordinator_node(
        self,
        state: MASState
    ):

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
                        self._content_length(
                            task
                        )
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
                "plan":
                    None,

                "error":
                    str(e)
            }

        # =====================================================
        # LOG PLAN WITHOUT FULL CONTENT
        # =====================================================

        self.log_event(
            MASEvent.create(
                event_type="task_decomposition",

                sender="coordinator",

                content="Coordinator created task plan",

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

        return {
            "plan":
                plan,

            "error":
                None
        }

    # =========================================================
    # RESEARCHER NODE
    # =========================================================

    def research_node(
        self,
        state: MASState
    ):

        plan = state["plan"]

        if plan is None:

            raise ValueError(
                "Research node received no plan."
            )

        research_instruction = plan[
            "research"
        ]

        # =====================================================
        # COORDINATOR -> RESEARCHER
        # =====================================================

        self.send_message(
            sender="coordinator",

            receiver="researcher",

            content=research_instruction
        )

        # =====================================================
        # SHARED POOL READ
        # =====================================================

        if self.topology_name == "shared_pool":

            self.read_shared_pool(
                receiver="researcher"
            )

        # =====================================================
        # RESEARCHER MEMORY
        # =====================================================

        self.read_memory(
            agent_name="researcher",

            query=research_instruction,

            top_k=3
        )

        # =====================================================
        # RESEARCHER DECIDES WHETHER TOOL IS REQUIRED
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
                    requesting_agent=
                        tool_request["agent"],

                    tool_name=
                        tool_request["tool_name"],

                    arguments=
                        tool_request["arguments"]
                )
            except PermissionError:
                result = []

            if isinstance(
                result,
                list
            ):

                tool_results.extend(
                    result
                )

            else:

                tool_results.append(
                    result
                )

        # =====================================================
        # RESEARCHER RUN
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
                    "stage":
                        "research",

                    "topology":
                        self.topology_name,

                    "used_external_tools":
                        bool(tool_results),

                    "source_count":
                        len(tool_results),

                    "content_length":
                        self._content_length(
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
                "stage":
                    "research",

                "used_external_tools":
                    bool(tool_results),

                "source_count":
                    len(tool_results)
            }
        )

        # =====================================================
        # RETURN RESEARCH + ORIGINAL SOURCES
        # =====================================================

        return {
            "research":
                research_result,

            "research_sources":
                tool_results
        }

    # =========================================================
    # ANALYST NODE
    # =========================================================

    def analysis_node(
        self,
        state: MASState
    ):

        plan = state["plan"]

        research = state["research"]

        research_sources = state.get(
            "research_sources",
            []
        )

        if plan is None:

            raise ValueError(
                "Analysis node received no plan."
            )

        if research is None:

            raise ValueError(
                "Analysis node received no research result."
            )

        analysis_instruction = plan[
            "analysis"
        ]

        # =====================================================
        # PREPARE ANALYSIS MESSAGE
        # =====================================================

        analysis_message = (
            f"Analysis assignment:\n"
            f"{analysis_instruction}\n\n"
            f"Research findings:\n"
            f"{research}\n\n"
            f"Research sources available: "
            f"{len(research_sources)}"
        )

        # =====================================================
        # LAYERED
        #
        # Researcher -> Analyst
        # =====================================================

        if self.topology_name == "layered":

            self.send_message(
                sender="researcher",

                receiver="analyst",

                content=analysis_message
            )

        # =====================================================
        # SHARED POOL
        #
        # Coordinator publishes to pool.
        # Analyst reads from pool.
        # =====================================================

        elif self.topology_name == "shared_pool":

            self.send_message(
                sender="coordinator",

                receiver="analyst",

                content=analysis_message
            )

            self.read_shared_pool(
                receiver="analyst"
            )

        # =====================================================
        # CENTRALIZED / DECENTRALIZED
        # =====================================================

        else:

            self.send_message(
                sender="coordinator",

                receiver="analyst",

                content=analysis_message
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
        # ANALYST DECIDES WHETHER ADDITIONAL TOOL IS REQUIRED
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

        # =====================================================
        # OPTIONAL ADDITIONAL TOOL REQUEST
        # =====================================================

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

            if isinstance(
                result,
                list
            ):

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

                    "research_source_count":
                        len(research_sources),

                    "additional_source_count":
                        len(tool_results),

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
                    bool(tool_results),

                "research_source_count":
                    len(research_sources),

                "additional_source_count":
                    len(tool_results)
            }
        )

        return {
            "analysis":
                analysis_result
        }

    # =========================================================
    # EXECUTOR NODE
    # =========================================================

    def execution_node(
        self,
        state: MASState
    ):

        plan = state["plan"]

        analysis = state["analysis"]

        if plan is None:

            raise ValueError(
                "Execution node received no plan."
            )

        if analysis is None:

            raise ValueError(
                "Execution node received no analysis."
            )

        execution_instruction = plan[
            "execution"
        ]

        execution_message = (
            f"Execution assignment:\n"
            f"{execution_instruction}\n\n"
            f"Analyst findings:\n"
            f"{analysis}"
        )

        # =====================================================
        # LAYERED
        #
        # Analyst -> Executor
        # =====================================================

        if self.topology_name == "layered":

            self.send_message(
                sender="analyst",

                receiver="executor",

                content=execution_message
            )

        # =====================================================
        # SHARED POOL
        # =====================================================

        elif self.topology_name == "shared_pool":

            self.send_message(
                sender="coordinator",

                receiver="executor",

                content=execution_message
            )

            self.read_shared_pool(
                receiver="executor"
            )

        # =====================================================
        # CENTRALIZED / DECENTRALIZED
        # =====================================================

        else:

            self.send_message(
                sender="coordinator",

                receiver="executor",

                content=execution_message
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
        # EXECUTOR DECIDES WHETHER TOOL IS REQUIRED
        # =====================================================

        tool_request = (
            self.executor.create_tool_request(
                execution_instruction=
                    execution_instruction,

                analysis=
                    analysis
            )
        )

        tool_results = []

        # =====================================================
        # TOOL REQUEST
        # =====================================================

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

            if isinstance(
                result,
                list
            ):

                tool_results.extend(
                    result
                )

            else:

                tool_results.append(
                    result
                )

        # =====================================================
        # EXECUTOR RUN
        # =====================================================

        execution_result = self.executor.run(
            execution_instruction,

            analysis,

            tool_results=
                tool_results
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
                        bool(tool_results),

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
                    bool(tool_results)
            }
        )

        return {
            "execution":
                execution_result
        }

    # =========================================================
    # FINAL COORDINATOR NODE
    # =========================================================

    def final_node(
        self,
        state: MASState
    ):

        execution = state[
            "execution"
        ]

        if execution is None:

            raise ValueError(
                "Final node received no execution result."
            )

        # =====================================================
        # RETURN COMMUNICATION
        # =====================================================

        if self.topology_name == "layered":

            # -------------------------------------------------
            # Executor -> Analyst
            # -------------------------------------------------

            self.send_message(
                sender="executor",

                receiver="analyst",

                content=execution
            )

            # -------------------------------------------------
            # Analyst -> Researcher
            # -------------------------------------------------

            self.send_message(
                sender="analyst",

                receiver="researcher",

                content=execution
            )

            # -------------------------------------------------
            # Researcher -> Coordinator
            # -------------------------------------------------

            self.send_message(
                sender="researcher",

                receiver="coordinator",

                content=execution
            )

        elif self.topology_name == "shared_pool":

            # -------------------------------------------------
            # Executor -> Shared Pool
            # -------------------------------------------------

            self.send_message(
                sender="executor",

                receiver="coordinator",

                content=execution
            )

            # -------------------------------------------------
            # Coordinator reads result
            # -------------------------------------------------

            self.read_shared_pool(
                receiver="coordinator"
            )

        else:

            # -------------------------------------------------
            # Centralized / Decentralized
            # -------------------------------------------------

            self.send_message(
                sender="executor",

                receiver="coordinator",

                content=execution
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
    # EXECUTE TASK
    # =========================================================

    def execute_task(
        self,
        task: str
    ):

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