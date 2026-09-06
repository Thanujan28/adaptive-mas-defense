
from agents.llm import get_llm
from tools.tool_request import ToolRequest
import json


class CoordinatorAgent:

    def __init__(
        self,
        name="coordinator",
        memory=None,
        tool_manager=None,
        tool_control_plane=None
    ):

        self.name = name
        self.llm = get_llm()
        self.memory = memory

        # ToolManager is injected by the MAS environment.
        #
        # The Coordinator is a routing layer.
        # It does NOT decide which tool an agent needs.
        self.tool_manager = tool_manager
        self.tool_control_plane = tool_control_plane

    # ============================================================
    # MEMORY
    # ============================================================

    def set_memory(self, memory):

        self.memory = memory

    def remember(
        self,
        content: str,
        importance: int = 5,
        metadata=None
    ):

        if self.memory is None:
            return None

        return self.memory.add(
            content=content,
            importance=importance,
            metadata=metadata,
        )

    def recall(
        self,
        query=None,
        top_k=3
    ):

        if self.memory is None:
            return []

        return self.memory.retrieve(
            query=query,
            top_k=top_k,
        )

    def _format_memories(
        self,
        memories
    ):

        if not memories:

            return (
                "No previous memories available."
            )

        return "\n\n".join(
            f"- {memory.content}"
            for memory in memories
        )

    # ============================================================
    # TOOL MANAGER
    # ============================================================

    def set_tool_manager(
        self,
        tool_manager
    ):

        self.tool_manager = tool_manager

    def set_tool_control_plane(
        self,
        tool_control_plane
    ):

        self.tool_control_plane = tool_control_plane

    # ============================================================
    # TOOL REQUEST
    # ============================================================

    def handle_tool_request(
        self,
        agent: str,
        tool_name: str,
        arguments: dict,
        request_id: str = ""
    ):
        """
        Route a centralized tool request from an agent through the
        Tool Control Plane to the ToolManager.

        IMPORTANT:

        The Coordinator does NOT decide which tool is required.

        The requesting agent makes that decision.

        Centralized flow:

            Agent
              ↓
            Coordinator
              ↓
            ToolManager
              ↓
              Tool
              ↓
            ToolManager
              ↓
            Coordinator
              ↓
            Agent

        Security validation can later be inserted around this
        boundary without changing the agent architecture.
        """

        # --------------------------------------------------------
        # Validate ToolManager
        # --------------------------------------------------------

        if (
            self.tool_manager is None
            and self.tool_control_plane is None
        ):

            raise RuntimeError(
                "Coordinator has no ToolManager configured."
            )

        # --------------------------------------------------------
        # Validate requesting agent
        # --------------------------------------------------------

        if not agent or not str(agent).strip():

            raise ValueError(
                "Tool request must specify requesting agent."
            )

        agent = str(agent).strip()

        # --------------------------------------------------------
        # Validate tool name
        # --------------------------------------------------------

        if not tool_name or not str(tool_name).strip():

            raise ValueError(
                "Tool request must specify tool name."
            )

        tool_name = str(
            tool_name
        ).strip()

        # --------------------------------------------------------
        # Validate arguments
        # --------------------------------------------------------

        if arguments is None:

            arguments = {}

        if not isinstance(
            arguments,
            dict
        ):

            raise ValueError(
                "Tool arguments must be a dictionary."
            )

        # --------------------------------------------------------
        # Forward request to ToolManager
        # --------------------------------------------------------

        request = ToolRequest(
            agent=agent,
            tool_name=tool_name,
            arguments=arguments,
            request_id=request_id,
        )

        if self.tool_control_plane is not None:
            result = self.tool_control_plane.submit(
                request,
                submitted_by=self.name,
            )
        else:
            result = self.tool_manager.execute(
                agent=agent,
                tool_name=tool_name,
                arguments=arguments
            )

        # --------------------------------------------------------
        # Return result to requesting agent
        # --------------------------------------------------------

        return result

    # ============================================================
    # PLAN CREATION
    # ============================================================

    def create_plan(
        self,
        task
    ):

        if not task or not str(task).strip():

            raise ValueError(
                "Coordinator cannot create a plan "
                "for an empty task."
            )

        task = str(
            task
        ).strip()

        memories = self.recall(
            query=task,
            top_k=3
        )

        memory_context = self._format_memories(
            memories
        )

        prompt = f"""
You are the Coordinator agent in a multi-agent system.

Your responsibility is to decompose the user's task into
exactly three clear stages:

1. research
2. analysis
3. execution

The three stages must be distinct and logically connected.

The Research stage should identify what information needs
to be investigated.

The Analysis stage should explain what should be evaluated,
compared, interpreted, or verified based on the research.

The Execution stage should specify what final response,
recommendation, result, or action should be produced based
on the analysis.

Important rules:

- Return ONLY valid JSON.
- Do not include explanations before or after the JSON.
- Do not use Markdown code fences.
- Do not invent budgets, costs, timelines, staffing,
  resources, citations, or implementation estimates unless
  the user's task explicitly requests them.
- Base the plan primarily on the user's original task.
- Keep all three stages directly relevant to the task.
- Research, analysis, and execution must not be duplicates.
- Previous memory may provide context, but it must not
  override the user's current task.
- Do not perform the research yourself.
- Do not perform the analysis yourself.
- Do not produce the final answer.

Previous Coordinator memories:

{memory_context}

Required JSON structure:

{{
    "research": "Research task",
    "analysis": "Analysis task",
    "execution": "Execution task"
}}

User task:

{task}
"""

        response = self.llm.invoke(
            prompt
        )

        content = response.content.strip()

        # --------------------------------------------------------
        # First parsing attempt
        # --------------------------------------------------------

        try:

            plan = self._parse_plan(
                content
            )

            self.remember(
                content=(
                    f"Created execution plan for task: "
                    f"{task}\n"
                    f"Plan: {plan}"
                ),
                importance=7,
                metadata={
                    "event": "plan_created"
                }
            )

            return plan

        except (
            json.JSONDecodeError,
            ValueError
        ):

            print(
                "\n[Coordinator] Invalid structured output "
                "received. Retrying..."
            )

        # --------------------------------------------------------
        # Retry / structured-output repair
        # --------------------------------------------------------

        repair_prompt = f"""
You are the Coordinator agent.

Your previous response did not satisfy the required
JSON format.

Generate the task decomposition again.

You MUST return ONLY valid JSON.

Do not include Markdown.
Do not include explanations.
Do not include text before or after the JSON.

The JSON must contain exactly these three required
fields:

{{
    "research": "Research task",
    "analysis": "Analysis task",
    "execution": "Execution task"
}}

Rules:

- Research identifies information that needs to be
  investigated.
- Analysis evaluates or interprets the research.
- Execution produces the requested final result or action.
- Keep the three stages distinct.
- Do not invent requirements that are not present in
  the original task.

Original user task:

{task}

Previous invalid response:

{content}
"""

        response = self.llm.invoke(
            repair_prompt
        )

        retry_content = (
            response.content.strip()
        )

        try:

            plan = self._parse_plan(
                retry_content
            )

            self.remember(
                content=(
                    f"Created execution plan after retry "
                    f"for task: {task}\n"
                    f"Plan: {plan}"
                ),
                importance=7,
                metadata={
                    "event": "plan_created_after_retry"
                }
            )

            return plan

        except (
            json.JSONDecodeError,
            ValueError
        ):

            raise ValueError(
                "Coordinator failed to produce a valid "
                "plan after retry.\n\n"
                f"First response:\n{content}\n\n"
                f"Retry response:\n{retry_content}"
            )

    # ============================================================
    # PLAN PARSING
    # ============================================================

    def _parse_plan(
        self,
        content
    ):

        if not content:

            raise ValueError(
                "Coordinator returned an empty response."
            )

        cleaned = content.strip()

        # --------------------------------------------------------
        # Remove Markdown code fences if the model ignored
        # the instruction.
        # --------------------------------------------------------

        if cleaned.startswith("```"):

            cleaned = cleaned.replace(
                "```json",
                ""
            )

            cleaned = cleaned.replace(
                "```JSON",
                ""
            )

            cleaned = cleaned.replace(
                "```",
                ""
            )

            cleaned = cleaned.strip()

        # --------------------------------------------------------
        # Direct JSON parsing
        # --------------------------------------------------------

        try:

            plan = json.loads(
                cleaned
            )

        except json.JSONDecodeError:

            # ----------------------------------------------------
            # Attempt to extract the JSON object from surrounding
            # text.
            # ----------------------------------------------------

            start = cleaned.find(
                "{"
            )

            end = cleaned.rfind(
                "}"
            )

            if (
                start == -1
                or end == -1
                or end <= start
            ):

                raise ValueError(
                    "Coordinator returned invalid JSON:\n"
                    + content
                )

            json_content = cleaned[
                start:end + 1
            ]

            plan = json.loads(
                json_content
            )

        return self._validate_plan(
            plan
        )

    # ============================================================
    # PLAN VALIDATION
    # ============================================================

    def _validate_plan(
        self,
        plan
    ):

        if not isinstance(
            plan,
            dict
        ):

            raise ValueError(
                "Coordinator plan must be a JSON object."
            )

        required_fields = [
            "research",
            "analysis",
            "execution"
        ]

        # --------------------------------------------------------
        # Required fields
        # --------------------------------------------------------

        for field in required_fields:

            if field not in plan:

                raise ValueError(
                    f"Coordinator plan is missing "
                    f"required field: {field}"
                )

            if not isinstance(
                plan[field],
                str
            ):

                raise ValueError(
                    f"Coordinator field '{field}' "
                    f"must be a string."
                )

            if not plan[field].strip():

                raise ValueError(
                    f"Coordinator field '{field}' "
                    f"cannot be empty."
                )

        # --------------------------------------------------------
        # Ensure the stages are meaningfully different.
        # --------------------------------------------------------

        normalized = [
            plan["research"].strip().lower(),
            plan["analysis"].strip().lower(),
            plan["execution"].strip().lower()
        ]

        if (
            normalized[0]
            == normalized[1]
            == normalized[2]
        ):

            raise ValueError(
                "Coordinator produced identical "
                "research, analysis, and execution stages."
            )

        return plan

    # ============================================================
    # FINAL RESULT AGGREGATION
    # ============================================================

    def aggregate(
        self,
        task,
        result
    ):
        """
        Produce the final user-facing answer from the Executor
        result.

        The Coordinator does not perform new research here.

        It verifies that the Executor result is relevant to
        the original task and avoids introducing unsupported
        information.
        """

        if not task or not str(task).strip():

            raise ValueError(
                "Cannot aggregate result for an empty task."
            )

        if result is None:

            raise ValueError(
                "Cannot aggregate an empty Executor result."
            )

        task = str(
            task
        ).strip()

        result = str(
            result
        ).strip()

        memories = self.recall(
            query=task,
            top_k=3
        )

        memory_context = self._format_memories(
            memories
        )

        prompt = f"""
You are the Coordinator agent performing final verification.

Original user task:

{task}

Executor result:

{result}

Previous Coordinator memories:

{memory_context}

Your responsibility is to produce the final answer to the
user.

Rules:

1. Base the final answer on the Executor result and the
   original user task.

2. Do not perform new research.

3. Do not introduce facts that are absent from the Executor
   result unless they are directly stated in the original
   user task.

4. Do not invent citations.

5. Do not invent authors.

6. Do not invent papers.

7. Do not invent URLs.

8. Do not invent costs.

9. Do not invent budgets.

10. Do not invent timelines.

11. Do not invent staffing or resource requirements.

12. Do not invent numerical estimates.

13. Do not turn an unsupported claim from the Executor into
    an established fact.

14. If the Executor explicitly identifies insufficient
    evidence, preserve that limitation.

15. Do not claim that a system, recommendation, source,
    organization, or approach is secure unless the available
    evidence supports that conclusion.

16. Previous memory is supporting context only and must not
    override the current task or Executor result.

17. Do not describe the internal multi-agent workflow.

18. Return only the final answer.

Final answer:
"""

        response = self.llm.invoke(
            prompt
        )

        final_result = (
            response.content.strip()
        )

        if not final_result:

            raise ValueError(
                "Coordinator produced an empty final result."
            )

        self.remember(
            content=(
                f"Final result produced for task: "
                f"{task}\n"
                f"Result: {final_result}"
            ),
            importance=8,
            metadata={
                "event": "final_result"
            }
        )

        return final_result
