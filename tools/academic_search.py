import requests
from typing import List, Dict


class AcademicSearchTool:

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict]:

        if not query or not query.strip():
            raise ValueError(
                "Academic search query cannot be empty."
            )

        max_results = max(
            1,
            min(int(max_results), 10)
        )

        params = {
            "search": query.strip(),
            "per-page": max_results
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as exc:
            raise ConnectionError(
                f"Academic search failed: {exc}"
            ) from exc

        results = []

        for work in data.get("results", []):

            primary_location = (
                work.get("primary_location") or {}
            )

            open_access = (
                work.get("open_access") or {}
            )

            landing_page_url = (
                primary_location.get(
                    "landing_page_url"
                )
            )

            pdf_url = (
                primary_location.get(
                    "pdf_url"
                )
            )

            oa_url = (
                open_access.get(
                    "oa_url"
                )
            )

            best_source_url = (
                pdf_url
                or oa_url
                or landing_page_url
                or work.get("doi")
            )

            authors = []

            for authorship in work.get(
                "authorships",
                []
            ):

                author = (
                    authorship.get("author")
                    or {}
                )

                author_name = (
                    author.get("display_name")
                )

                if author_name:
                    authors.append(
                        author_name
                    )

            results.append({

                "id": work.get("id"),

                "title": work.get("title"),

                "publication_year": work.get(
                    "publication_year"
                ),

                "doi": work.get("doi"),

                "type": work.get("type"),

                "cited_by_count": work.get(
                    "cited_by_count"
                ),

                "url": landing_page_url,

                "pdf_url": pdf_url,

                "oa_url": oa_url,

                "source_url": best_source_url,

                "is_open_access": open_access.get(
                    "is_oa"
                ),

                "authors": authors,

                "content": None,

                "content_type": None,

                "content_status": "not_collected",

                "content_error": None,
            })

        return results