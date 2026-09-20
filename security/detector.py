from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Any


class SecurityDetector:
    """
    Lightweight rule-based detector for the MAS security state.

    This detector does not claim that an agent is compromised.
    It produces observable evidence that can be consumed by the
    semantic assessor and PPO state builder.
    """

    ATTACK_EVENT_TYPES = {
        "attack",
        "external_result_injection",
        "prompt_injection",
        "memory_poisoning",
        "suspicious_tool_result",
    }

    MESSAGE_EVENT_TYPES = {
        "message",
        "message_relay",
    }

    def detect(
        self,
        events: Iterable[Mapping[str, Any]],
    ) -> dict[str, float | int | bool | list[str]]:
        events = list(events)

        event_types = Counter(
            event.get("event_type")
            for event in events
        )

        attack_events = [
            event
            for event in events
            if event.get("event_type") in self.ATTACK_EVENT_TYPES
        ]

        timeout_count = sum(
            1
            for event in events
            if event.get("event_type") in {
                "tool_timeout",
                "timeout",
            }
        )

        suspicious_events = sum(
            1
            for event in events
            if event.get("suspicious") is True
        )

        return {
            "attack_count": len(attack_events),
            "explicit_attack_count": sum(
                1
                for event in attack_events
                if event.get("event_type") in {
                    "attack",
                    "prompt_injection",
                    "memory_poisoning",
                }
            ),
            "external_injection_count": event_types[
                "external_result_injection"
            ],
            "suspicious_event_count": suspicious_events,
            "message_count": sum(
                event_types[event_type]
                for event_type in self.MESSAGE_EVENT_TYPES
            ),
            "relay_count": event_types["message_relay"],
            "tool_timeout_count": timeout_count,
            "detected": bool(attack_events or suspicious_events),
            "attack_ids": [
                event.get("request_id")
                for event in attack_events
                if event.get("request_id") is not None
            ],
        }