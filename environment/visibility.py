"""
Observability boundary for MAS events.

The defender may only ever see *observable* evidence: content,
behaviour and provenance that a real deployment could actually
observe. Simulator ground truth (attack labels, infection markers,
attack ids) must never reach the defender's observation.

This module centralises:

  * the set of forbidden ground-truth keys,
  * the set of attack-type event types,
  * sanitisation of observable event dictionaries, and
  * a strict guard that raises when ground truth leaks into a
    defender input.
"""

from __future__ import annotations

from typing import Any, Mapping


# Ground-truth keys that must never appear in a defender input,
# whether at the top level of an event or nested inside metadata.
FORBIDDEN_GROUND_TRUTH_KEYS = frozenset(
    {
        "infected",
        "attack_type",
        "attack_id",
        "ground_truth",
        "suspicious",
        "infection_hop",
        "compromised",
        "propagated",
    }
)


# Attack-type event types that only the simulator/ground-truth
# channel may carry. A defender-visible event must not use these.
ATTACK_EVENT_TYPES = frozenset(
    {
        "attack",
        "external_result_injection",
        "prompt_injection",
        "memory_poisoning",
        "suspicious_tool_result",
        "external_result_poisoned",
        "researcher_received_poisoned_result",
        "researcher_output_infected",
        "analyst_received_infected_input",
        "analyst_output_infected",
        "executor_received_infected_input",
        "executor_output_infected",
    }
)


# Senders that only the ground-truth channel may use.
GROUND_TRUTH_SENDERS = frozenset(
    {
        "attack_simulator",
    }
)


def _strip_forbidden_keys(
    value: Any,
) -> Any:
    """
    Recursively remove forbidden ground-truth keys from a value.

    Works on mappings and sequences, returning sanitised copies.
    """

    if isinstance(value, Mapping):

        sanitised = {}

        for key, item in value.items():

            if key in FORBIDDEN_GROUND_TRUTH_KEYS:
                continue
            sanitised[key] = _strip_forbidden_keys(item)

        return sanitised

    if isinstance(value, (list, tuple)):

        return [
            _strip_forbidden_keys(item)
            for item in value
        ]

    return value


def sanitize_observable_event(
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return a sanitised, observable-only copy of an event dictionary.

    Forbidden ground-truth keys are removed from the top level and
    from nested metadata. The ``visibility`` marker is dropped so a
    defender consumer cannot accidentally branch on it.
    """

    sanitised = _strip_forbidden_keys(event)

    # Drop the visibility marker from defender-visible events.
    sanitised.pop("visibility", None)

    return sanitised


def _iter_violations(
    event: Mapping[str, Any],
) -> list[str]:
    """
    Return a list of human-readable ground-truth violations in an
    event, or an empty list when the event is clean.
    """

    violations: list[str] = []

    if event.get("visibility") == "ground_truth":
        violations.append(
            "event visibility is 'ground_truth'"
        )

    if event.get("sender") in GROUND_TRUTH_SENDERS:
        violations.append(
            f"ground-truth sender {event.get('sender')!r}"
        )

    if event.get("event_type") in ATTACK_EVENT_TYPES:
        violations.append(
            f"attack-type event_type {event.get('event_type')!r}"
        )

    for key in _find_forbidden_keys(event):
        violations.append(
            f"forbidden key {key!r}"
        )

    return violations


def _find_forbidden_keys(
    value: Any,
    path: str = "",
) -> list[str]:
    """
    Recursively collect forbidden keys found anywhere in a value.
    """

    found: list[str] = []

    if isinstance(value, Mapping):

        for key, item in value.items():

            if key in FORBIDDEN_GROUND_TRUTH_KEYS:
                found.append(
                    f"{path}{key}" if path else str(key)
                )
            found.extend(
                _find_forbidden_keys(
                    item,
                    path=f"{path}{key}.",
                )
            )

    elif isinstance(value, (list, tuple)):

        for index, item in enumerate(value):
            found.extend(
                _find_forbidden_keys(
                    item,
                    path=f"{path}{index}.",
                )
            )

    return found


def assert_no_ground_truth(
    events,
) -> None:
    """
    Raise ValueError if any event carries ground-truth leakage.

    Used as a strict, non-negotiable guard on every defender input
    (SecurityObserver.observe and SecurityStateBuilder.build).

    Leakage is any of:

      * visibility == "ground_truth"
      * sender in GROUND_TRUTH_SENDERS (e.g. "attack_simulator")
      * event_type in ATTACK_EVENT_TYPES
      * a forbidden ground-truth key anywhere in the event or its
        nested metadata
    """

    problems: list[str] = []

    for index, event in enumerate(events):

        if not isinstance(event, Mapping):
            continue

        for violation in _iter_violations(event):
            problems.append(
                f"event[{index}]: {violation}"
            )

    if problems:
        raise ValueError(
            "Ground-truth leakage detected in defender input:\n  "
            + "\n  ".join(problems)
        )
