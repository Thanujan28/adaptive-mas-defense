"""
Optional IDS-style noisy alert sensor.

ALERT SENSOR (ablation only, OFF by default).

A real defender often has an intrusion-detection system that emits
noisy alerts. This sensor simulates that: it reads ground truth
internally but its OUTPUT is an observable ``alert`` event, so it
can be fed to the defender like any other observable signal.

It exposes two knobs:

  * ``tpr`` (true positive rate): probability of raising an alert on
    a ground-truth-positive step,
  * ``fpr`` (false positive rate): probability of raising an alert on
    a ground-truth-negative step.

An alert is observable and carries NO ground-truth label: only the
alert itself. It is intended for ablation studies, never for the
default defender configuration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


@dataclass
class AlertSensor:
    """
    Noisy IDS-style alert generator.

    Reads ground truth internally; emits observable alert events.
    """

    tpr: float = 0.9
    fpr: float = 0.1
    seed: Optional[int] = None
    source: str = "ids_alert_sensor"

    def __post_init__(self) -> None:
        if not 0.0 <= self.tpr <= 1.0:
            raise ValueError("tpr must be in [0, 1]")
        if not 0.0 <= self.fpr <= 1.0:
            raise ValueError("fpr must be in [0, 1]")
        self._rng = random.Random(self.seed)

    def _ground_truth_positive(
        self,
        ground_truth_events: Iterable[Mapping[str, Any]],
    ) -> bool:
        """
        True when the ground-truth channel indicates an attack step.
        """

        for event in ground_truth_events:
            if event.get("event_type") in {
                "external_result_injection",
                "prompt_injection",
                "memory_poisoning",
                "attack",
            }:
                return True
        return False

    def observe(
        self,
        ground_truth_events: Iterable[Mapping[str, Any]],
        agent_id: str = "unknown",
    ) -> Optional[dict[str, Any]]:
        """
        Decide whether to raise an observable alert for this step.

        Returns an observable alert event dict, or None.

        The alert carries ONLY observable fields; it must never
        include the ground-truth label that caused it.
        """

        is_positive = self._ground_truth_positive(
            ground_truth_events
        )

        probability = self.tpr if is_positive else self.fpr

        if self._rng.random() >= probability:
            return None

        return {
            "event_type": "alert",
            "sender": self.source,
            "receiver": agent_id,
            "content": "IDS raised an alert",
            "visibility": "observable",
            "metadata": {
                "alert_source": self.source,
            },
        }
