import json

from agents.llm import get_llm


class OutlineAgent:
    """
    Outline agent in the multi-agent research workflow.

    Role
    ----
    Given the original user task, produce a structured outline for a
    research proposal / thesis:

        topic
          sub-topic 1
          sub-topic 2
          ...

    The outline is the plan that the Researcher consumes. For each
    sub-topic the Researcher gathers external evidence and reports the
    important key points; the Executor then compiles the final report.

    Pipeline
    --------
        Coordinator
             |
          Outline
             |
        Researcher  (one research pass per sub-topic)
             |
         Executor
             |
          Final

    The Outline agent does NOT search the Internet and does NOT write
    the final deliverable. It only structures the topic.
    """

    def __init__(self, name="outline", memory=None):

        self.name = name
        self.llm = get_llm()
        self.memory = memory

    # =========================================================
    # MEMORY
    # =========================================================

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

    # =========================================================
    # TASK NORMALIZATION
    # =========================================================

    def _task_to_sentence(
        self,
        task
    ) -> str:
        """
        Normalize the Coordinator's assignment into a readable
        sentence instead of raw JSON.

        The Coordinator may hand over the outline assignment as a
        structured object:

            {
                "objective": "...",
                "tasks": ["...", "..."],
                "required_output": "..."
            }

        A plain string is returned unchanged.
        """

        if isinstance(
            task,
            dict
        ):

            data = task

        else:

            try:

                data = json.loads(
                    str(task)
                )

            except (
                TypeError,
                ValueError,
            ):

                data = None

        if not isinstance(
            data,
            dict
        ) and isinstance(
            task,
            str
        ):

            return task.strip()

        if not isinstance(
            data,
            dict
        ):

            return str(task).strip()

        objective = str(
            data.get(
                "objective",
                ""
            )
        ).strip()

        tasks = data.get(
            "tasks",
            []
        )

        if isinstance(
            tasks,
            str
        ):

            tasks = [tasks]

        elif not isinstance(
            tasks,
            list
        ):

            tasks = []

        tasks = [
            str(item).strip()
            for item in tasks
            if str(item).strip()
        ]

        required_output = str(
            data.get(
                "required_output",
                ""
            )
        ).strip()

        sentences = []

        if objective:

            sentences.append(
                f"Your objective is to {objective}"
                if not objective.lower().startswith(
                    (
                        "to ",
                        "your objective",
                    )
                )
                else objective
            )

        if tasks:

            sentences.append(
                "You should: "
                + "; ".join(tasks)
                + "."
            )

        if required_output:

            sentences.append(
                f"The required output is: {required_output}"
            )

        if not sentences:

            return str(task).strip()

        return " ".join(
            sentence
            if sentence.endswith(".")
            else f"{sentence}."
            for sentence in sentences
        )

    # =========================================================
    # OUTLINE BUILDING
    # =========================================================

    def create_outline(
        self,
        task: str,
        outline_instruction: str = "",
    ) -> dict:
        """
        Build a structured outline for the ORIGINAL user task.

        Returns a dictionary:

            {
                "Research Topic": "...",
                "topic": "...",
                "sub_topics": [
                    {
                        "title": "...",
                        "focus": "...",
                        "guiding_questions": ["...", "..."]
                    },
                    ...
                ],
                "raw": "<human-readable outline>"
                "topic": "...",
                "sub_topics": [
                    {
                        "title": "...",
                        "focus": "...",
                        "guiding_questions": ["...", "..."]
                    },
                    ...
                ],
                "raw": "<human-readable outline>"
            }

        Only valid JSON is accepted from the model. On parse failure
        the method retries once with a repair prompt.
        """

        if not task or not str(task).strip():

            raise ValueError(
                "Outline agent received an empty task."
            )

        task = str(task).strip()

        formatted_instruction = (
            self._task_to_sentence(
                outline_instruction
            )
            if outline_instruction
            else ""
        )

        memories = self.recall(
            query=task,
            top_k=3
        )

        memory_context = self._format_memories(
            memories
        )

        prompt = f"""
You are the Outline agent in a multi-agent research system.

You are step 1 of 3. Your bounded responsibility is ONLY to create
the proposal outline for the original topic below. You do NOT gather
evidence and you do NOT write the proposal.

Your output goes to: the Researcher, who will gather evidence for
each sub-topic you define.
Your output must be reused later by: the Executor, who will write
the final proposal from this outline plus the research.

ORIGINAL USER TASK (AUTHORITATIVE - the goal you are outlining for):

{task}

Coordinator outline assignment (the bounded scope for your role):

{formatted_instruction}

Previous Outline memories (supporting context only):

{memory_context}

WHAT YOU PRODUCE: the outline of the work.

- one overall research topic (taken directly from the ORIGINAL USER TASK)
- a set of major sub-topics that must be covered
Rules:

1. Preserve the ORIGINAL USER TASK. Do not replace, narrow, or
   redefine the user's research topic. The "topic" you output must
   be the SAME topic the user asked about.

2. The sub-topics must be specific, non-overlapping, and directly
   relevant to the original topic. Every sub-topic must serve the
   ORIGINAL USER TASK.

3. Each sub-topic must be researchable: the Researcher will search
   the Internet for evidence on it and report key points. The
   "focus" line tells the Researcher exactly what evidence to find.

4. For each sub-topic, provide:
   - title: short name of the sub-topic
   - focus: what evidence the Researcher must gather for it
   - guiding_questions: 2-4 concrete questions to answer
5. Order the sub-topics in a logical sequence for a proposal or
   thesis (for example: background, problem, related work, method,
   evaluation, risks, timeline).

6. Produce between 4 and 8 sub-topics.

BOUNDARIES (do not cross them):

7. Do NOT gather evidence or search the Internet.

8. Do NOT write the proposal or the final report.

9. Do NOT invent citations, authors, papers, URLs, statistics,
   budgets, or timelines.

10. Return ONLY valid JSON. No prose, no markdown fences.

Required JSON structure:

{{
    "topic": "...",
    "sub_topics": [
        {{
            "title": "...",
            "focus": "...",
            "guiding_questions": ["...", "..."]
        }}
    ]
}}
"""

        response = self.llm.invoke(prompt)

        content = response.content.strip()

        try:

            outline = self._parse_outline(
                content
            )

        except (json.JSONDecodeError, ValueError):

            repair_prompt = f"""
You are the Outline agent. Your previous response was invalid.

Produce a valid proposal outline for the ORIGINAL USER TASK below.
The outline topic must be the SAME topic the user asked about.
You only define the topic and sub-topics; you do not gather
evidence and you do not write the proposal.

Return ONLY valid JSON using exactly this structure:

{{
    "topic": "...",
    "sub_topics": [
        {{
            "title": "...",
            "focus": "...",
            "guiding_questions": ["...", "..."]
        }}
    ]
}}

ORIGINAL USER TASK:

{task}

Previous invalid response:

{content}
"""

            response = self.llm.invoke(
                repair_prompt
            )

            outline = self._parse_outline(
                response.content.strip()
            )

        outline["raw"] = self.format_outline(
            outline
        )

        self.remember(
            content=(
                f"Created outline for task: {task}\n"
                f"{outline['raw']}"
            ),
            importance=7,
            metadata={
                "event": "outline_created",
                "sub_topic_count": len(
                    outline.get(
                        "sub_topics",
                        []
                    )
                ),
            },
        )

        return outline

    # =========================================================
    # OUTLINE PARSING
    # =========================================================

    def _parse_outline(
        self,
        content
    ) -> dict:

        if not content:

            raise ValueError(
                "Outline agent returned an empty response."
            )

        cleaned = content.strip()

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

        try:

            outline = json.loads(
                cleaned
            )

        except json.JSONDecodeError:

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
                    "Outline agent returned invalid JSON:\n"
                    + content
                )

            outline = json.loads(
                cleaned[start:end + 1]
            )

        return self._validate_outline(
            outline
        )

    # =========================================================
    # OUTLINE VALIDATION
    # =========================================================

    def _validate_outline(
        self,
        outline
    ) -> dict:

        if not isinstance(
            outline,
            dict
        ):

            raise ValueError(
                "Outline must be a JSON object."
            )

        topic = str(
            outline.get(
                "topic",
                ""
            )
        ).strip()

        if not topic:

            raise ValueError(
                "Outline is missing a topic."
            )

        sub_topics = outline.get(
            "sub_topics",
            []
        )

        if not isinstance(
            sub_topics,
            list
        ) or not sub_topics:

            raise ValueError(
                "Outline must contain a non-empty "
                "'sub_topics' list."
            )

        normalized = []

        for sub_topic in sub_topics:

            if isinstance(
                sub_topic,
                str
            ):

                title = sub_topic.strip()

                if not title:
                    continue

                normalized.append(
                    {
                        "title": title,
                        "focus": "",
                        "guiding_questions": [],
                    }
                )

                continue

            if not isinstance(
                sub_topic,
                dict
            ):
                continue

            title = str(
                sub_topic.get(
                    "title",
                    ""
                )
            ).strip()

            if not title:
                continue

            focus = str(
                sub_topic.get(
                    "focus",
                    ""
                )
            ).strip()

            questions = sub_topic.get(
                "guiding_questions",
                []
            )

            if isinstance(
                questions,
                str
            ):

                questions = [questions]

            elif not isinstance(
                questions,
                list
            ):

                questions = []

            questions = [
                str(question).strip()
                for question in questions
                if str(question).strip()
            ]

            normalized.append(
                {
                    "title": title,
                    "focus": focus,
                    "guiding_questions": questions,
                }
            )

        if not normalized:

            raise ValueError(
                "Outline contains no usable sub-topics."
            )

        return {
            "topic": topic,
            "sub_topics": normalized,
        }

    # =========================================================
    # OUTLINE FORMATTING
    # =========================================================

    def format_outline(
        self,
        outline
    ) -> str:
        """
        Render the structured outline as a readable text block.

        This is what is passed to the Researcher so it can perform
        one research pass per sub-topic.
        """

        if not isinstance(
            outline,
            dict
        ):

            return str(outline)

        topic = str(
            outline.get(
                "topic",
                ""
            )
        ).strip()

        sub_topics = outline.get(
            "sub_topics",
            []
        )

        if not isinstance(
            sub_topics,
            list
        ):

            sub_topics = []

        lines = []

        if topic:

            lines.append(
                f"Research topic: {topic}"
            )

        lines.append("")

        lines.append("Sub-topics:")

        for index, sub_topic in enumerate(
            sub_topics,
            start=1
        ):

            if not isinstance(
                sub_topic,
                dict
            ):

                lines.append(
                    f"{index}. {sub_topic}"
                )

                continue

            title = str(
                sub_topic.get(
                    "title",
                    ""
                )
            ).strip()

            lines.append(
                f"{index}. {title}"
            )

            focus = str(
                sub_topic.get(
                    "focus",
                    ""
                )
            ).strip()

            if focus:

                lines.append(
                    f"   Focus: {focus}"
                )

            questions = sub_topic.get(
                "guiding_questions",
                []
            )

            if isinstance(
                questions,
                list
            ) and questions:

                for question in questions:

                    question = str(
                        question
                    ).strip()

                    if question:

                        lines.append(
                            f"   - {question}"
                        )

        return "\n".join(lines).strip()
