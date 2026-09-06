from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Any, Optional, Callable

from tools.internet_search import InternetSearchTool
from tools.academic_search import AcademicSearchTool
from tools.source_collector import SourceCollector
from tools.report_writer import ReportWriterTool
from tools.mock_calendar import MockCalendarTool
from tools.mock_email import MockEmailTool
from environment.resource_accounting import ResourceBudget


class ToolManager:

    def __init__(
        self,
        tool_limit: int = 50,
        tool_timeout_seconds: float = 30.0,
        resource_budget: Optional[ResourceBudget] = None,
        event_callback: Optional[Callable[..., None]] = None,
    ):
        self.resource_budget = resource_budget or ResourceBudget(
            tool_limit=tool_limit,
            tool_timeout_seconds=tool_timeout_seconds,
        )
        self.event_callback = event_callback
        # The registry and fixed permission matrix are initialized below.
        self.timed_out_tools = 0

    @property
    def tool_limit(self):
        return self.resource_budget.tool_limit

    @property
    def tool_timeout_seconds(self):
        return self.resource_budget.tool_timeout_seconds

    @property
    def tools_used(self):
        return self.resource_budget.tools_used

    @tools_used.setter
    def tools_used(self, value):
        self.resource_budget.tools_used = int(value)

    @property
    def timed_out_tools(self):
        return self.resource_budget.timed_out_tools

    @timed_out_tools.setter
    def timed_out_tools(self, value):
        if getattr(self, "_registry_initialized", False):
            self.resource_budget.timed_out_tools = int(value)
            return
        self.resource_budget.timed_out_tools = int(value)

        # =====================================================
        # AVAILABLE TOOLS
        # =====================================================

        self.tools = {

            "internet_search":
                InternetSearchTool(),

            "academic_search":
                AcademicSearchTool(),

            "source_collector":
                SourceCollector(),

            "report_writer":
                ReportWriterTool(),

            "mock_calendar":
                MockCalendarTool(),

            "mock_email":
                MockEmailTool(),
        }

        # Keep both names available to agent prompts and older callers.
        self.tools["mock_mail"] = self.tools["mock_email"]
        self.tools["mock_calender"] = self.tools["mock_calendar"]

        # =====================================================
        # FIXED AGENT TOOL AUTHORIZATION
        # =====================================================
        #
        # IMPORTANT:
        # Tool authorization is independent of
        # communication topology.
        #
        # Coordinator -> authorized
        # Planner, Researchers, and Analysts -> authorized
        # Executors -> NOT authorized
        #
        # This policy is identical for every topology.
        # =====================================================

        self.permissions = {

            "coordinator": [
                "internet_search",
                "academic_search",
                "source_collector",
                "report_writer",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],

            "planner": [
                "internet_search",
                "academic_search",
                "source_collector",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],

            "researcher-1": [
                "internet_search",
                "academic_search",
                "source_collector",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],

            "researcher-2": [
                "internet_search",
                "academic_search",
                "source_collector",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],

            "analyst-1": [
                "internet_search",
                "academic_search",
                "source_collector",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],

            "analyst-2": [
                "internet_search",
                "academic_search",
                "source_collector",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],

            "executor-1": ["mock_email", "mock_mail", "mock_calendar", "mock_calender"],
            "executor-2": ["mock_email", "mock_mail"],

            # Compatibility aliases for the former four-agent API.
            "researcher": [
                "internet_search",
                "academic_search",
                "source_collector",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],
            "analyst": [
                "internet_search",
                "academic_search",
                "source_collector",
                "mock_calendar",
                "mock_email",
                "mock_mail",
                "mock_calender",
            ],
            "executor": ["mock_email", "mock_mail"],
        }

        self.current_topology = None
        self._registry_initialized = True

    # =========================================================
    # TOPOLOGY
    # =========================================================

    def set_topology(self, topology_name: str):

        self.current_topology = topology_name.lower()

    # =========================================================
    # PERMISSION CHECK
    # =========================================================

    def is_allowed(
        self,
        agent: str,
        tool_name: str
    ) -> bool:

        allowed_tools = self.permissions.get(
            agent,
            []
        )

        return tool_name in allowed_tools

    # =========================================================
    # TOOL EXECUTION
    # =========================================================

    def execute(
        self,
        agent: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ):
        try:
            invocation_number = self.resource_budget.record_tool()
        except RuntimeError:
            self._resource_event(
                "tool_rejected", tool_name, "tool_budget_exhausted"
            )
            raise
        self._resource_event(
            "tool_usage", tool_name, "consumed", invocation_number
        )
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._execute,
            agent,
            tool_name,
            arguments,
        )
        try:
            return future.result(timeout=self.tool_timeout_seconds)
        except TimeoutError as exc:
            self.resource_budget.timed_out_tools += 1
            self._resource_event("tool_timeout", tool_name, "timeout")
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(
                f"Tool '{tool_name}' exceeded the {self.tool_timeout_seconds}s timeout."
            ) from exc
        finally:
            if not future.running():
                executor.shutdown(wait=False, cancel_futures=True)

    def _resource_event(self, event_type, tool_name, status, count=None):
        if self.event_callback is not None:
            self.event_callback(event_type, tool_name, status, count)

    def _execute(
        self,
        agent: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ):

        # -----------------------------------------------------
        # Validate agent
        # -----------------------------------------------------

        if not agent or not str(agent).strip():
            raise ValueError(
                "Tool request must specify an agent."
            )

        agent = str(agent).strip()

        # -----------------------------------------------------
        # Validate tool name
        # -----------------------------------------------------

        if not tool_name or not str(tool_name).strip():
            raise ValueError(
                "Tool request must specify a tool name."
            )

        tool_name = str(tool_name).strip()

        # -----------------------------------------------------
        # Check tool
        # -----------------------------------------------------

        if tool_name not in self.tools:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        # -----------------------------------------------------
        # Authorization
        # -----------------------------------------------------

        if not self.is_allowed(
            agent,
            tool_name
        ):

            raise PermissionError(
                f"Agent '{agent}' is not authorized "
                f"to use tool '{tool_name}'."
            )

        # -----------------------------------------------------
        # Validate arguments
        # -----------------------------------------------------

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise ValueError(
                "Tool arguments must be a dictionary."
            )

        # =====================================================
        # INTERNET SEARCH
        # =====================================================

        if tool_name == "internet_search":

            return self._execute_search(
                search_tool=self.tools["internet_search"],
                arguments=arguments,
                search_type="internet"
            )

        # =====================================================
        # ACADEMIC SEARCH
        # =====================================================

        if tool_name == "academic_search":

            return self._execute_search(
                search_tool=self.tools["academic_search"],
                arguments=arguments,
                search_type="academic"
            )

        # =====================================================
        # SOURCE COLLECTOR
        # =====================================================

        if tool_name == "source_collector":

            url = arguments.get("url")

            if not url:
                raise ValueError(
                    "source_collector requires a URL."
                )

            return self.tools[
                "source_collector"
            ].collect(
                url=url
            )

        # =====================================================
        # REPORT WRITER
        # =====================================================

        if tool_name == "report_writer":

            title = arguments.get("title")
            content = arguments.get("content")
            filename = arguments.get("filename")

            return self.tools[
                "report_writer"
            ].write_report(
                title=title,
                content=content,
                filename=filename
            )

        if tool_name in ("mock_calendar", "mock_calender"):
            operation = arguments.get("operation", "list")
            calendar = self.tools["mock_calendar"]
            if operation == "create":
                return calendar.create_event(**{
                    key: arguments[key]
                    for key in (
                        "title", "start", "end", "description", "location",
                        "duration_minutes", "participants",
                    )
                    if key in arguments
                })
            if operation == "list":
                return calendar.list_events()
            if operation == "get":
                return calendar.get_event(arguments.get("event_id"))
            if operation == "delete":
                return calendar.delete_event(arguments.get("event_id"))
            raise ValueError(f"Unknown mock_calendar operation: {operation}")

        if tool_name in ("mock_email", "mock_mail"):
            operation = arguments.get("operation", "send")
            email = self.tools["mock_email"]
            if operation == "send":
                return email.send_email(
                    to=arguments.get("to"),
                    subject=arguments.get("subject"),
                    body=arguments.get("body"),
                    sender=arguments.get("sender", "mas@localhost"),
                )
            if operation == "list":
                return email.list_messages()
            if operation == "get":
                return email.get_message(arguments.get("message_id"))
            raise ValueError(f"Unknown mock_email operation: {operation}")

        raise ValueError(
            f"No execution handler for tool '{tool_name}'."
        )

    # =========================================================
    # SEARCH EXECUTION
    # =========================================================

    def _execute_search(
        self,
        search_tool,
        arguments: Dict[str, Any],
        search_type: str
    ):

        query = arguments.get("query")

        if not query or not str(query).strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        query = str(query).strip()

        max_results = arguments.get(
            "max_results",
            5
        )

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5

        max_results = max(
            1,
            min(max_results, 10)
        )

        search_results = search_tool.search(
            query=query,
            max_results=max_results
        )

        if not search_results:
            return []

        collected_results = []

        for source in search_results:

            if not isinstance(source, dict):
                continue

            enriched_source = dict(source)

            source_url = source.get("source_url")

            if not source_url:

                enriched_source["content_status"] = \
                    "no_source_url"

                enriched_source["content"] = None
                enriched_source["content_type"] = None
                enriched_source["content_length"] = 0

                enriched_source["content_error"] = \
                    "No source URL was provided."

                enriched_source["search_type"] = search_type

                collected_results.append(
                    enriched_source
                )

                continue

            try:

                collected = self.tools[
                    "source_collector"
                ].collect(
                    url=source_url
                )

                enriched_source["content"] = \
                    collected.get("content")

                enriched_source["content_type"] = \
                    collected.get("content_type")

                enriched_source["content_status"] = \
                    collected.get("content_status")

                enriched_source["content_length"] = \
                    collected.get("content_length", 0)

                enriched_source["content_error"] = \
                    collected.get("content_error")

                enriched_source["collected_url"] = \
                    collected.get(
                        "url",
                        source_url
                    )

            except Exception as exc:

                enriched_source["content"] = None
                enriched_source["content_type"] = None

                enriched_source["content_status"] = \
                    "collection_failed"

                enriched_source["content_length"] = 0

                enriched_source["content_error"] = \
                    str(exc)

            enriched_source["search_type"] = search_type

            collected_results.append(
                enriched_source
            )

        return collected_results