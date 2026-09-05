import io
import ipaddress
import socket

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from urllib.parse import urlparse


class SourceCollector:

    def __init__(
        self,
        timeout: int = 15,
        max_content_length: int = 5_000_000
    ):

        self.timeout = timeout
        self.max_content_length = max_content_length

    # =========================================================
    # PUBLIC METHOD
    # =========================================================

    def collect(
        self,
        url: str
    ):

        if not url or not url.strip():

            raise ValueError(
                "Source URL cannot be empty."
            )

        url = url.strip()

        # -----------------------------------------------------
        # Validate original URL
        # -----------------------------------------------------

        self._validate_url(url)

        try:

            response = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": (
                        "AdaptiveMASResearchBot/1.0"
                    )
                },
                allow_redirects=True,
                stream=True
            )

            response.raise_for_status()

            # -------------------------------------------------
            # Validate final redirected URL
            # -------------------------------------------------

            final_url = response.url

            self._validate_url(
                final_url
            )

            # -------------------------------------------------
            # Check declared content length
            # -------------------------------------------------

            content_length = response.headers.get(
                "Content-Length"
            )

            if content_length:

                try:

                    declared_length = int(
                        content_length
                    )

                except ValueError:

                    declared_length = None

                if (
                    declared_length is not None
                    and declared_length
                    > self.max_content_length
                ):

                    raise ValueError(
                        "Source is too large to collect."
                    )

            # -------------------------------------------------
            # Download content
            # -------------------------------------------------

            raw_content = response.content

            if len(raw_content) > self.max_content_length:

                raise ValueError(
                    "Downloaded source exceeds size limit."
                )

            # -------------------------------------------------
            # Content type
            # -------------------------------------------------

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                )
                .lower()
                .split(";")[0]
                .strip()
            )

            # =================================================
            # PDF
            # =================================================

            if (
                content_type == "application/pdf"
                or final_url.lower().split("?")[0].endswith(
                    ".pdf"
                )
            ):

                text = self._extract_pdf(
                    raw_content
                )

                if not text.strip():

                    return {
                        "url": final_url,
                        "content_type": "pdf",
                        "content": None,
                        "content_status": "empty_content",
                        "content_length": 0,
                        "content_error": (
                            "PDF contains no extractable text."
                        ),
                    }

                return {
                    "url": final_url,
                    "content_type": "pdf",
                    "content": text,
                    "content_status": "collected",
                    "content_length": len(text),
                    "content_error": None,
                }

            # =================================================
            # HTML
            # =================================================

            if content_type in (
                "text/html",
                "application/xhtml+xml"
            ):

                text = self._extract_html(
                    raw_content
                )

                if not text.strip():

                    return {
                        "url": final_url,
                        "content_type": "html",
                        "content": None,
                        "content_status": "empty_content",
                        "content_length": 0,
                        "content_error": (
                            "No readable text was extracted "
                            "from the webpage."
                        ),
                    }

                return {
                    "url": final_url,
                    "content_type": "html",
                    "content": text,
                    "content_status": "collected",
                    "content_length": len(text),
                    "content_error": None,
                }

            # =================================================
            # Unsupported content
            # =================================================

            return {
                "url": final_url,
                "content_type": content_type,
                "content": None,
                "content_status": "unsupported_content_type",
                "content_length": 0,
                "content_error": (
                    f"Unsupported content type: "
                    f"{content_type or 'unknown'}"
                ),
            }

        except requests.exceptions.RequestException as exc:

            return {
                "url": url,
                "content_type": None,
                "content": None,
                "content_status": "collection_failed",
                "content_length": 0,
                "content_error": str(exc),
            }

        except Exception as exc:

            return {
                "url": url,
                "content_type": None,
                "content": None,
                "content_status": "collection_failed",
                "content_length": 0,
                "content_error": str(exc),
            }

    # =========================================================
    # HTML EXTRACTION
    # =========================================================

    def _extract_html(
        self,
        raw_content: bytes
    ) -> str:

        soup = BeautifulSoup(
            raw_content,
            "html.parser"
        )

        # -----------------------------------------------------
        # Remove irrelevant elements
        # -----------------------------------------------------

        for element in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "form",
            "svg"
        ]):

            element.decompose()

        text = soup.get_text(
            separator="\n"
        )

        lines = []

        for line in text.splitlines():

            line = " ".join(
                line.split()
            )

            if line:

                lines.append(
                    line
                )

        text = "\n".join(
            lines
        )

        return text[:100_000]

    # =========================================================
    # PDF EXTRACTION
    # =========================================================

    def _extract_pdf(
        self,
        raw_content: bytes
    ) -> str:

        reader = PdfReader(
            io.BytesIO(raw_content)
        )

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                page_text = (
                    page.extract_text()
                    or ""
                )

            except Exception:

                page_text = ""

            if page_text.strip():

                pages.append(
                    f"--- Page {page_number} ---\n"
                    f"{page_text.strip()}"
                )

        text = "\n\n".join(
            pages
        )

        return text[:100_000]

    # =========================================================
    # URL SECURITY
    # =========================================================

    def _validate_url(
        self,
        url: str
    ):

        parsed = urlparse(
            url
        )

        # -----------------------------------------------------
        # Scheme
        # -----------------------------------------------------

        if parsed.scheme not in (
            "http",
            "https"
        ):

            raise ValueError(
                "Only HTTP and HTTPS URLs are allowed."
            )

        # -----------------------------------------------------
        # Hostname
        # -----------------------------------------------------

        hostname = parsed.hostname

        if not hostname:

            raise ValueError(
                "URL does not contain a valid hostname."
            )

        hostname = hostname.lower()

        # -----------------------------------------------------
        # Explicit localhost blocking
        # -----------------------------------------------------

        if hostname in (
            "localhost",
            "localhost.localdomain"
        ):

            raise ValueError(
                "Localhost URLs are not allowed."
            )

        # -----------------------------------------------------
        # Resolve hostname
        # -----------------------------------------------------

        try:

            addresses = socket.getaddrinfo(
                hostname,
                None
            )

        except socket.gaierror as exc:

            raise ValueError(
                f"Unable to resolve hostname: {hostname}"
            ) from exc

        # -----------------------------------------------------
        # Block unsafe IP ranges
        # -----------------------------------------------------

        for address in addresses:

            ip_string = address[4][0]

            try:

                ip = ipaddress.ip_address(
                    ip_string
                )

            except ValueError:

                continue

            if ip.is_private:

                raise ValueError(
                    "Blocked private network URL."
                )

            if ip.is_loopback:

                raise ValueError(
                    "Blocked loopback URL."
                )

            if ip.is_link_local:

                raise ValueError(
                    "Blocked link-local URL."
                )

            if ip.is_reserved:

                raise ValueError(
                    "Blocked reserved IP URL."
                )

            if ip.is_multicast:

                raise ValueError(
                    "Blocked multicast URL."
                )