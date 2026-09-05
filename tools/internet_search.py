import os
from typing import List, Dict

import requests


class InternetSearchTool:

    BASE_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 15
    ):
        """
        Tavily-based general Internet search.

        API key can be supplied directly or through:

            TAVILY_API_KEY

        Environment variable is preferred so that the API key
        is not stored in source code.
        """

        self.api_key = (
            api_key
            or os.getenv("TAVILY_API_KEY")
        )

        self.timeout = timeout

    # =========================================================
    # PUBLIC SEARCH METHOD
    # =========================================================

    def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict]:

        # -----------------------------------------------------
        # Validate API key
        # -----------------------------------------------------

        if not self.api_key:

            raise RuntimeError(
                "TAVILY_API_KEY is not configured."
            )

        # -----------------------------------------------------
        # Validate query
        # -----------------------------------------------------

        if not query or not query.strip():

            raise ValueError(
                "Internet search query cannot be empty."
            )

        # -----------------------------------------------------
        # Normalize result count
        # -----------------------------------------------------

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
        # Tavily request
        # -----------------------------------------------------

        payload = {
            "api_key": self.api_key,
            "query": query.strip(),
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        try:

            response = requests.post(
                self.BASE_URL,
                json=payload,
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": (
                        "AdaptiveMASResearchBot/1.0"
                    )
                }
            )

            response.raise_for_status()

            data = response.json()

        except requests.exceptions.Timeout as exc:

            raise ConnectionError(
                "Internet search timed out."
            ) from exc

        except requests.exceptions.HTTPError as exc:

            status_code = (
                exc.response.status_code
                if exc.response is not None
                else "unknown"
            )

            raise ConnectionError(
                f"Tavily search failed "
                f"(HTTP {status_code})."
            ) from exc

        except requests.exceptions.RequestException as exc:

            raise ConnectionError(
                f"Internet search failed: {exc}"
            ) from exc

        except ValueError as exc:

            raise ConnectionError(
                "Tavily returned invalid JSON."
            ) from exc

        # =====================================================
        # NORMALIZE RESULTS
        # =====================================================

        results = []

        for item in data.get(
            "results",
            []
        ):

            if not isinstance(
                item,
                dict
            ):

                continue

            title = item.get(
                "title"
            )

            url = item.get(
                "url"
            )

            content = item.get(
                "content"
            )

            score = item.get(
                "score"
            )

            published_date = item.get(
                "published_date"
            )

            # -------------------------------------------------
            # Skip malformed results
            # -------------------------------------------------

            if not title or not url:

                continue

            results.append({

                # ---------------------------------------------
                # Source identification
                # ---------------------------------------------

                "id": url,

                "title": title,

                "url": url,

                "source_url": url,

                # ---------------------------------------------
                # Search metadata
                # ---------------------------------------------

                "score": score,

                "published_date": published_date,

                "search_provider": "tavily",

                # ---------------------------------------------
                # Search snippet
                # ---------------------------------------------

                "snippet": content,

                # ---------------------------------------------
                # Content will be populated by SourceCollector
                # ---------------------------------------------

                "content": None,

                "content_type": None,

                "content_status": "not_collected",

                "content_length": 0,

                "content_error": None,
            })

        return results