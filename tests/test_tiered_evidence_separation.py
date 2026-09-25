"""Regression tests separating dispatch assignments from Tier-2 evidence."""

from __future__ import annotations

from types import SimpleNamespace

from environment.mas_environment import MASEnvironment


ASSIGNMENT_TEXT = "Execution assignment: Role: Executor (step 3 of 3)."
FINDINGS_TEXT = "Research finding: solar adoption increased by 12 percent in 2024."


class _CaptureObserver:
    """Capture the evidence chunks supplied to the tiered observer."""

    def __init__(self):
        self.evidence_chunks = None

    def observe_tiered(self, **kwargs):
        self.evidence_chunks = list(kwargs["evidence_chunks"])
        return SimpleNamespace(
            base=SimpleNamespace(metadata={}, semantic_assessment=None),
            metadata={},
        )


def test_dispatch_assignment_is_not_tier2_evidence():
    env = MASEnvironment(topology_name="centralized")
    observer = _CaptureObserver()
    env.security_observer = observer

    env.log_artifact(
        artifact_type="assignment",
        source="coordinator",
        receiver="executor",
        content=ASSIGNMENT_TEXT,
    )
    env.log_artifact(
        artifact_type="message",
        source="researcher",
        receiver="executor",
        content=FINDINGS_TEXT,
    )

    env._observe_agent_response(
        agent_id="executor",
        response="The final proposal will use the research finding.",
    )

    assert observer.evidence_chunks is not None
    assert ASSIGNMENT_TEXT not in observer.evidence_chunks
    assert FINDINGS_TEXT in observer.evidence_chunks
