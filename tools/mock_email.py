import os
from typing import Any, Dict, List, Optional

import requests


class MockEmailTool:
    """MailHog-backed email tool for local MAS experiments."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 10,
        http_client=requests,
    ):
        self.base_url = (
            base_url or os.getenv("MAILHOG_API_URL", "http://localhost:8025")
        ).rstrip("/")
        self.timeout = timeout
        self.http_client = http_client

    def send_email(
        self,
        to: str | List[str],
        subject: str,
        body: str,
        sender: str = "mas@localhost",
    ) -> Dict[str, Any]:
        recipients = [to] if isinstance(to, str) else list(to or [])
        recipients = [str(address).strip() for address in recipients if str(address).strip()]
        if not recipients:
            raise ValueError("Email recipient cannot be empty.")
        if not subject or not str(subject).strip():
            raise ValueError("Email subject cannot be empty.")
        if not body or not str(body).strip():
            raise ValueError("Email body cannot be empty.")

        response = self.http_client.post(
            f"{self.base_url}/api/v1/send",
            json={
                "From": {"Mailbox": str(sender).split("@")[0], "Domain": str(sender).split("@")[-1]},
                "To": [
                    {
                        "Mailbox": address.split("@")[0],
                        "Domain": address.split("@")[-1],
                    }
                    for address in recipients
                ],
                "Subject": str(subject).strip(),
                "Content": str(body),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return {
            "status": "sent",
            "to": recipients,
            "subject": str(subject).strip(),
        }

    def list_messages(self) -> List[Dict[str, Any]]:
        response = self.http_client.get(
            f"{self.base_url}/api/v2/messages",
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        messages = data.get("items", []) if isinstance(data, dict) else []
        return [self._normalize_message(message) for message in messages]

    def get_message(self, message_id: str) -> Dict[str, Any]:
        if not message_id or not str(message_id).strip():
            raise ValueError("MailHog message ID cannot be empty.")
        response = self.http_client.get(
            f"{self.base_url}/api/v1/messages/{str(message_id).strip()}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._normalize_message(response.json())

    @staticmethod
    def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
        headers = message.get("Content", {}).get("Headers", {})
        return {
            "id": message.get("ID", message.get("id", "")),
            "from": headers.get("From", [""])[0],
            "to": headers.get("To", []),
            "subject": headers.get("Subject", [""])[0],
            "body": message.get("Content", {}).get("Body", ""),
        }