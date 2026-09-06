from tools.tool_request import ToolRequest


class ToolControlPlane:
    """System-level gateway between MAS agents and ToolManager."""

    def __init__(self, tool_manager):
        self.tool_manager = tool_manager

    def submit(
        self,
        request: ToolRequest,
        submitted_by: str = "coordinator"
    ):
        if not isinstance(request, ToolRequest):
            raise TypeError("Tool control plane requires a ToolRequest.")

        if (
            self.tool_manager.current_topology == "centralized"
            and submitted_by != "coordinator"
        ):
            raise PermissionError(
                "Only the Coordinator may submit tool requests "
                "in centralized topology."
            )

        if (
            submitted_by.startswith("executor")
            and request.tool_name not in (
                "mock_email",
                "mock_mail",
                "mock_calendar",
                "mock_calender",
            )
        ):
            raise PermissionError(
                "Executors are only authorized to submit mock email requests."
            )

        execution_agent = (
            "coordinator"
            if self.tool_manager.current_topology == "centralized"
            else request.agent
        )

        return self.tool_manager.execute(
            agent=execution_agent,
            tool_name=request.tool_name,
            arguments=request.arguments,
        )