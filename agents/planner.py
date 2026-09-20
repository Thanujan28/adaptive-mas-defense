class PlannerAgent:
    """Convert the Coordinator's task decomposition into an executable plan."""

    def __init__(self, name="planner", memory=None):
        self.name = name
        self.memory = memory

    def set_memory(self, memory):
        self.memory = memory

    def create_execution_plan(self, task, coordinator_plan):
        if not task or not str(task).strip():
            raise ValueError("Planner cannot plan an empty task.")

        if not isinstance(coordinator_plan, dict):
            raise ValueError("Planner requires a Coordinator plan.")

        required_stages = ("outline", "research", "execution")
        missing = [stage for stage in required_stages if not coordinator_plan.get(stage)]
        if missing:
            raise ValueError(
                "Coordinator plan is missing stages: "
                + ", ".join(missing)
            )

        return {
            "task": str(task).strip(),
            "outline": coordinator_plan["outline"],
            "research": coordinator_plan["research"],
            "execution": coordinator_plan["execution"],
            "assignments": {
                "outline-1": "topic and sub-topic structuring",
                "researcher-1": "evidence collection for each sub-topic",
                "executor-1": "final report compilation",
            },
        }