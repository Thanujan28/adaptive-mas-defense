from typing import Dict, Any

from tools.internet_search import InternetSearchTool
from tools.academic_search import AcademicSearchTool
from tools.source_collector import SourceCollector
from tools.report_writer import ReportWriterTool


class ToolManager:

    def __init__(self):

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
        }

        # =====================================================
        # AGENT TOOL PERMISSIONS
        # =====================================================

        self.permissions = {

            # -------------------------------------------------
            # Coordinator
            # -------------------------------------------------
            #
            # Coordinator can:
            #   - search the Internet
            #   - search academic sources
            #   - create final reports
            #
            "coordinator": [
                "internet_search",
                "academic_search",
                "report_writer",
            ],

            # -------------------------------------------------
            # Researcher
            # -------------------------------------------------
            #
            # Researcher is responsible for information
            # gathering.
            #
            "researcher": [
                "internet_search",
                "academic_search",
            ],

            # -------------------------------------------------
            # Analyst
            # -------------------------------------------------
            #
            # Analyst can obtain additional evidence when
            # required.
            #
            "analyst": [
                "internet_search",
                "academic_search",
            ],

            # -------------------------------------------------
            # Executor
            # -------------------------------------------------
            #
            # Executor does not directly access tools.
            # It produces the final report content.
            #
            "executor": [],
        }

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

        if (
            not tool_name
            or not str(tool_name).strip()
        ):

            raise ValueError(
                "Tool request must specify a tool name."
            )

        tool_name = str(
            tool_name
        ).strip()

        # -----------------------------------------------------
        # Check whether tool exists
        # -----------------------------------------------------

        if tool_name not in self.tools:

            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        # -----------------------------------------------------
        # Permission check
        # -----------------------------------------------------

        if not self.is_allowed(
            agent,
            tool_name
        ):

            raise PermissionError(
                f"Agent '{agent}' is not allowed "
                f"to use tool '{tool_name}'."
            )

        # -----------------------------------------------------
        # Validate arguments
        # -----------------------------------------------------

        if arguments is None:

            arguments = {}

        if not isinstance(
            arguments,
            dict
        ):

            raise ValueError(
                "Tool arguments must be a dictionary."
            )

        # =====================================================
        # INTERNET SEARCH
        # =====================================================

        if tool_name == "internet_search":

            return self._execute_search(
                search_tool=
                    self.tools["internet_search"],

                arguments=
                    arguments,

                search_type=
                    "internet"
            )

        # =====================================================
        # ACADEMIC SEARCH
        # =====================================================

        if tool_name == "academic_search":

            return self._execute_search(
                search_tool=
                    self.tools["academic_search"],

                arguments=
                    arguments,

                search_type=
                    "academic"
            )

        # =====================================================
        # SOURCE COLLECTOR
        # =====================================================

        if tool_name == "source_collector":

            url = arguments.get(
                "url"
            )

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

            title = arguments.get(
                "title"
            )

            content = arguments.get(
                "content"
            )

            filename = arguments.get(
                "filename"
            )

            return self.tools[
                "report_writer"
            ].write_report(
                title=title,
                content=content,
                filename=filename
            )

        # =====================================================
        # SAFETY FALLBACK
        # =====================================================

        raise ValueError(
            f"No execution handler for tool "
            f"'{tool_name}'."
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

        query = arguments.get(
            "query"
        )

        if (
            not query
            or not str(query).strip()
        ):

            raise ValueError(
                "Search query cannot be empty."
            )

        query = str(
            query
        ).strip()

        # -----------------------------------------------------
        # max_results
        # -----------------------------------------------------

        max_results = arguments.get(
            "max_results",
            5
        )

        try:

            max_results = int(
                max_results
            )

        except (
            TypeError,
            ValueError
        ):

            max_results = 5

        max_results = max(
            1,
            min(
                max_results,
                10
            )
        )

        # -----------------------------------------------------
        # Execute search
        # -----------------------------------------------------

        search_results = search_tool.search(
            query=query,
            max_results=max_results
        )

        if not search_results:

            return []

        # -----------------------------------------------------
        # Collect source content
        # -----------------------------------------------------

        collected_results = []

        for source in search_results:

            if not isinstance(
                source,
                dict
            ):

                continue

            enriched_source = dict(
                source
            )

            source_url = source.get(
                "source_url"
            )

            # -------------------------------------------------
            # No URL
            # -------------------------------------------------

            if not source_url:

                enriched_source[
                    "content_status"
                ] = "no_source_url"

                enriched_source[
                    "content"
                ] = None

                enriched_source[
                    "content_type"
                ] = None

                enriched_source[
                    "content_length"
                ] = 0

                enriched_source[
                    "content_error"
                ] = (
                    "No source URL was provided."
                )

                collected_results.append(
                    enriched_source
                )

                continue

            # -------------------------------------------------
            # Collect source
            # -------------------------------------------------

            try:

                collected = self.tools[
                    "source_collector"
                ].collect(
                    url=source_url
                )

                enriched_source[
                    "content"
                ] = collected.get(
                    "content"
                )

                enriched_source[
                    "content_type"
                ] = collected.get(
                    "content_type"
                )

                enriched_source[
                    "content_status"
                ] = collected.get(
                    "content_status"
                )

                enriched_source[
                    "content_length"
                ] = collected.get(
                    "content_length",
                    0
                )

                enriched_source[
                    "content_error"
                ] = collected.get(
                    "content_error"
                )

                enriched_source[
                    "collected_url"
                ] = collected.get(
                    "url",
                    source_url
                )

            except Exception as exc:

                enriched_source[
                    "content"
                ] = None

                enriched_source[
                    "content_type"
                ] = None

                enriched_source[
                    "content_status"
                ] = (
                    "collection_failed"
                )

                enriched_source[
                    "content_length"
                ] = 0

                enriched_source[
                    "content_error"
                ] = str(exc)

            # -------------------------------------------------
            # Search type
            # -------------------------------------------------

            enriched_source[
                "search_type"
            ] = search_type

            collected_results.append(
                enriched_source
            )

        return collected_results