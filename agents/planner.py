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

        required_stages = ("research", "analysis", "execution")
        missing = [stage for stage in required_stages if not coordinator_plan.get(stage)]
        if missing:
            raise ValueError(
                "Coordinator plan is missing stages: "
                + ", ".join(missing)
            )

        return {
            "task": str(task).strip(),
            "research": coordinator_plan["research"],
            "analysis": coordinator_plan["analysis"],
            "execution": coordinator_plan["execution"],
            "assignments": {
                "researcher-1": "broad discovery",
                "researcher-2": "independent evidence collection and verification",
                "analyst-1": "primary analysis",
                "analyst-2": "finding verification and inconsistency detection",
                "executor-1": "downstream task execution",
                "executor-2": "output validation",
            },
        }