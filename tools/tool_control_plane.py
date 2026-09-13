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

        if not self.tool_manager.is_allowed(request.agent, request.tool_name):
            raise PermissionError(
                f"Agent '{request.agent}' is not authorized "
                f"to use tool '{request.tool_name}'."
            )

        return self.tool_manager.execute(
            agent=request.agent,
            tool_name=request.tool_name,
            arguments=request.arguments,
            authorization_agent=request.requester or request.agent,
            request_id=request.request_id,
            metadata=request.metadata,
        )