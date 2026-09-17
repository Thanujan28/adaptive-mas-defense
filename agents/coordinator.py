
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
            metadata={
                "requesting_agent": agent,
            },
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

    def create_plan(self, task):

        if not task or not str(task).strip():
            raise ValueError(
                "Coordinator cannot create a plan "
                "for an empty task."
            )

        task = str(task).strip()

        memories = self.recall(
            query=task,
            top_k=3
        )

        memory_context = self._format_memories(memories)

        prompt = f"""
    You are the Coordinator agent in a multi-agent system.

    Your responsibility is to intelligently decompose the user's
    task into exactly three role-specific assignments:

    1. Researcher
    2. Analyst
    3. Executor

    You must understand the semantic meaning of the user's task.
    Do NOT split the task using keywords, regular expressions,
    sentence positions, or arbitrary text segments.

    The three agents have different responsibilities.

    RESEARCHER:
    Investigates information required to answer the user's task.
    The Researcher should identify relevant evidence, sources,
    facts, concepts, methods, limitations, and open questions.

    ANALYST:
    Evaluates the Researcher's findings against the ORIGINAL USER
    TASK. The Analyst must determine whether the findings actually
    address the original objective, identify gaps, compare or
    interpret evidence, and develop the required conclusions.

    EXECUTOR:
    Uses the validated analysis to produce the final deliverable
    requested by the user.

    IMPORTANT:

    - Preserve the original user objective.
    - Preserve important domain terms, research topics,
    constraints, requested methods, and requested deliverables.
    - Do not replace the user's research topic with a related topic.
    - Do the task decomposition carefully while preserving the original context.
    - Ensure that each agent know for why they work on this sub task.
    - When you tell work on given topic/given object/given task/given person to any of agent, clearly state to each agent what is that given topic/object/task/person is.
    - Each agent should know what is the objective of the task they are working on.
    - Do not assume that a Researcher's output defines the task.
    - The original user task is the authoritative task definition.
    - Each downstream assignment must remain traceable to the
    original task.
    - The Analyst must explicitly evaluate whether Researcher
    findings remain aligned with the original task.
    - The Executor must preserve the original task objective when
    producing the final result.
    - Previous memory may provide supporting context but MUST NOT
    override the current user task.
    - Do not perform the research yourself.
    - Do not perform the analysis yourself.
    - Do not produce the final answer.

    For each stage, provide:

    - objective
    - tasks
    - required_output

    Return ONLY valid JSON.

    Required JSON structure:

    {{
        "research": {{
            "objective": "...",
            "tasks": ["...", "..."],
            "required_output": "..."
        }},
        "analysis": {{
            "objective": "...",
            "tasks": ["...", "..."],
            "required_output": "..."
        }},
        "execution": {{
            "objective": "...",
            "tasks": ["...", "..."],
            "required_output": "..."
        }}
    }}

    Previous Coordinator memories:

    {memory_context}

    ORIGINAL USER TASK:

    {task}
    """

        response = self.llm.invoke(prompt)

        content = response.content.strip()

        try:
            plan = self._parse_plan(content)

            self.remember(
                content=(
                    f"Created semantic execution plan for task: "
                    f"{task}\nPlan: {plan}"
                ),
                importance=7,
                metadata={
                    "event": "plan_created"
                }
            )

            return plan

        except (json.JSONDecodeError, ValueError):

            print(
                "\n[Coordinator] Invalid structured output "
                "received. Retrying..."
            )

            repair_prompt = f"""
    You are the Coordinator agent.

    Your previous response was invalid.

    Create a valid semantic task decomposition without losing context for the
ORIGINAL USER TASK below.

    Do not split the task using regex or text positions.
    Do the task decomposition carefully while preserving the original context.
    Ensure that each agent know for why they work on this sub task.
    When you tell work on given topic/given object/given task/given person to any of agent, clearly state to each agent what is that given topic/object/task/person is.
    Each agent should know what is the objective of the task they are working on.

    Return ONLY valid JSON using exactly this structure:

    {{
        "research": {{
            "objective": "...",
            "tasks": ["...", "..."],
            "required_output": "..."
        }},
        "analysis": {{
            "objective": "...",
            "tasks": ["...", "..."],
            "required_output": "..."
        }},
        "execution": {{
            "objective": "...",
            "tasks": ["...", "..."],
            "required_output": "..."
        }}
    }}

    The assignments must remain semantically aligned with the
    original user task.

    Original user task:

    {task}

    Previous invalid response:

    {content}
    """

            response = self.llm.invoke(repair_prompt)

            retry_content = response.content.strip()

            plan = self._parse_plan(retry_content)

            self.remember(
                content=(
                    f"Created semantic execution plan after retry "
                    f"for task: {task}\nPlan: {plan}"
                ),
                importance=7,
                metadata={
                    "event": "plan_created_after_retry"
                }
            )

            return plan

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

        required_stage_keys = [
            "objective",
            "tasks",
            "required_output"
        ]

        # --------------------------------------------------------
        # Required fields
        # --------------------------------------------------------
        # Each stage is a nested object with objective/tasks/
        # required_output, as requested by the prompt schema.

        for field in required_fields:

            if field not in plan:

                raise ValueError(
                    f"Coordinator plan is missing "
                    f"required field: {field}"
                )

            stage = plan[field]

            if not isinstance(
                stage,
                dict
            ):

                raise ValueError(
                    f"Coordinator field '{field}' "
                    f"must be an object."
                )

            for key in required_stage_keys:

                if key not in stage:

                    raise ValueError(
                        f"Coordinator field '{field}' "
                        f"is missing required key: {key}"
                    )

            if not isinstance(
                stage["objective"],
                str
            ) or not stage["objective"].strip():

                raise ValueError(
                    f"Coordinator field '{field}.objective' "
                    f"cannot be empty."
                )

            if (
                not isinstance(stage["tasks"], list)
                or not stage["tasks"]
            ):

                raise ValueError(
                    f"Coordinator field '{field}.tasks' "
                    f"must be a non-empty list."
                )

            if not isinstance(
                stage["required_output"],
                str
            ) or not stage["required_output"].strip():

                raise ValueError(
                    f"Coordinator field '{field}.required_output' "
                    f"cannot be empty."
                )

        # --------------------------------------------------------
        # Ensure the stages are meaningfully different.
        # --------------------------------------------------------

        normalized = [
            plan["research"]["objective"].strip().lower(),
            plan["analysis"]["objective"].strip().lower(),
            plan["execution"]["objective"].strip().lower()
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

        print(plan)
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
