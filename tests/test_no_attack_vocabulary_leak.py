"""
Lint test for P7 (Task 7): the observable channel must never leak
attack vocabulary through ENVIRONMENT-authored fields.

Why this test exists
--------------------
The "leak" we guard against is not the attacker's payload -- that
payload *must* reach the defender's observable channel, because
scanning delivered content for manipulation is the whole point of the
content detector. The leak is the *environment* telling the defender,
in its own words, that something was an attack: e.g.
``status="compromised"``, ``metadata={"stage": "poisoned"}`` or an
event named ``external_result_injection``.

The lint therefore draws an explicit line:

  * ENVIRONMENT-AUTHORED fields (event_type, sender, receiver,
    metadata keys AND values, artifact_type, source, request_id, and
    the ``content`` of every environment-authored event) MUST NOT
    contain attack vocabulary (infect, inject, poison, attack,
    compromise, malicious, payload).

  * AGENT/ATTACKER-DELIVERED text is EXEMPT, because it is the data
    the defender is meant to inspect: ``agent_result`` content, message
    bodies, artifact ``text`` and memory-write text.

This test runs a REAL episode through ``MASEnvironment.execute_task``
with a stubbed LLM and stubbed tools (see tests/conftest.py), so real
emitters (request_tool, _process_external_tool_result, the agent nodes,
log_memory_write, publish_agent_result) are exercised.

New emitters must be classified: an event type that is not in
ENVIRONMENT_AUTHORED_EVENT_TYPES or AGENT_AUTHORED_EVENT_TYPES makes
the test FAIL, forcing a conscious classification.
"""

from __future__ import annotations
from typing import Any
import pytest
from attacks.scenarios import IMPLEMENTED_CONDITIONS, apply_attack
from environment.mas_environment import MASEnvironment
from environment.visibility import FORBIDDEN_GROUND_TRUTH_KEYS
# =================================================================
# VOCABULARY
# =================================================================

ATTACK_VOCABULARY = (
    "infect",
    "inject",
    "poison",
    "attack",
    "compromise",
    "malicious",
    "payload",
)

# =================================================================
# EVENT CLASSIFICATION
#
# Every observable event type an episode can emit must appear in
# exactly one of these two sets. This is deliberate: the test fails on
# an unclassified event type so new emitters are classified rather
# than silently ignored.
# =================================================================

# Events whose CONTENT (and all other fields) are authored by the
# environment/simulator and must therefore be free of attack
# vocabulary.
ENVIRONMENT_AUTHORED_EVENT_TYPES = frozenset(
    {
        "resource_reset",
        "task_received",
        "task_decomposition",
        "agent_failure",
        "memory_write",
        "memory_read",
        "llm_usage",
        "message",
        "message_receive",
        "message_relay",
        "pool_write",
        "pool_read",
        "tool_request",
        "tool_denied",
        "tool_forward",
        "tool_execution",
        "tool_result",
        "tool_result_delivery",
        "tool_error",
        "tool_usage",
        "tool_timeout",
        "tool_rejected",
        "final_result",
        "report_created",
    }
)

# Events whose CONTENT is agent-authored (an agent's own output) and is
# therefore EXEMPT from the vocabulary lint. Every OTHER field of these
# events (event_type, sender, receiver, metadata) is still checked.
AGENT_AUTHORED_EVENT_TYPES = frozenset(
    {
        "agent_result",
    }
)

KNOWN_EVENT_TYPES = (
    ENVIRONMENT_AUTHORED_EVENT_TYPES | AGENT_AUTHORED_EVENT_TYPES
)

# =================================================================
# HELPERS
# =================================================================

def _vocabulary_hits(value: Any) -> list[str]:
    """Return every attack-vocabulary word found anywhere in value."""

    hits: list[str] = []

    if isinstance(value, str):
        lowered = value.lower()
        for word in ATTACK_VOCABULARY:
            if word in lowered:
                hits.append(word)

    elif isinstance(value, dict):
        for item in value.values():
            hits.extend(_vocabulary_hits(item))

    elif isinstance(value, (list, tuple, set)):
        for item in value:
            hits.extend(_vocabulary_hits(item))

    return hits

def _find_forbidden_keys(value: Any, path: str = "") -> list[str]:
    """Recursively collect any FORBIDDEN_GROUND_TRUTH_KEYS in value."""

    found: list[str] = []

    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_GROUND_TRUTH_KEYS:
                found.append(f"{path}{key}" if path else str(key))
            found.extend(_find_forbidden_keys(item, path=f"{path}{key}."))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, path=f"{path}{index}."))

    return found


# =================================================================
# FIXTURE: a real, stubbed episode
# =================================================================

def _run_episode(
    condition: str,
    topology: str = "layered",
    task: str = "Write a proposal on AI in cyber security",
) -> MASEnvironment:
    """
    Run a REAL episode (execute_task) under ``condition`` with a
    stubbed LLM and stubbed tools. The stub agent text deliberately
    contains the words "attack" and "malicious" (see tests/conftest
    .py) to prove legitimate cybersecurity-topic agent prose does not
    trip the lint.
    """

    from tests.conftest import (
        StubCalendar,
        StubEmail,
        StubLLM,
        StubReportWriter,
        StubSearchTool,
    )

    import agents.analyst as analyst_module
    import agents.coordinator as coordinator_module
    import agents.executor as executor_module
    import agents.outline as outline_module
    import agents.researcher as researcher_module
    stub = StubLLM()
    for module in (
        coordinator_module,
        outline_module,
        researcher_module,
        analyst_module,
        executor_module,
    ):
        module.get_llm = lambda: stub
    env = MASEnvironment(topology_name=topology)

    tools = env.tool_manager.tools
    for name in ("internet_search", "academic_search"):
        if name in tools:
            tools[name] = StubSearchTool()
    if "report_writer" in tools:
        tools["report_writer"] = StubReportWriter()
    if "mock_calendar" in tools:
        tools["mock_calendar"] = StubCalendar()
    if "mock_calender" in tools:
        tools["mock_calender"] = StubCalendar()
    if "mock_email" in tools:
        tools["mock_email"] = StubEmail()

    apply_attack(env, condition, task_id="lint-1")

    env.execute_task(task)

    return env
@pytest.fixture(scope="module")
def lint_episodes():
    """One real stubbed episode per implemented condition."""

    return {
        condition: _run_episode(condition)
        for condition in IMPLEMENTED_CONDITIONS
    }

# =================================================================
# THE LINT
# =================================================================

def _assert_environment_fields_are_neutral(env: MASEnvironment) -> None:
    """
    Assert every environment-authored field is free of attack
    vocabulary, and that every emitted event type is classified.
    """

    for event in env.get_observable_events():

        event_type = event.get("event_type")

        assert event_type in KNOWN_EVENT_TYPES, (
            "Unclassified observable event type "
            f"{event_type!r}: add it to "
            "ENVIRONMENT_AUTHORED_EVENT_TYPES or "
            "AGENT_AUTHORED_EVENT_TYPES after checking its fields."
        )

        # event_type / sender / receiver are always environment-set.
        for field in ("event_type", "sender", "receiver"):
            hits = _vocabulary_hits(event.get(field))
            assert not hits, (event_type, field, event, hits)

        # Metadata keys AND values are environment-authored.
        metadata = event.get("metadata", {})
        hits = _vocabulary_hits(metadata)
        assert not hits, (event_type, "metadata", event, hits)

        # content is environment-authored for every event EXCEPT the
        # agent-authored ones.
        if event_type in ENVIRONMENT_AUTHORED_EVENT_TYPES:
            hits = _vocabulary_hits(event.get("content"))
            assert not hits, (event_type, "content", event, hits)

    for artifact in env.get_observable_artifacts():

        # Artifact "text" is delivered content and is EXEMPT.
        for field in (
            "artifact_type",
            "source",
            "receiver",
            "request_id",
            "message_id",
        ):
            hits = _vocabulary_hits(artifact.get(field))
            assert not hits, (artifact.get("artifact_type"), field, artifact, hits)


def _assert_no_forbidden_ground_truth_keys(env: MASEnvironment) -> None:
    """No FORBIDDEN_GROUND_TRUTH_KEYS anywhere, recursively."""

    for event in env.get_observable_events():
        found = _find_forbidden_keys(event)
        assert not found, (event, found)

    for artifact in env.get_observable_artifacts():
        found = _find_forbidden_keys(artifact)
        assert not found, (artifact, found)

# =================================================================
# TESTS: real episodes, all implemented conditions
# =================================================================

@pytest.mark.parametrize("condition", IMPLEMENTED_CONDITIONS)
def test_real_episode_environment_fields_are_neutral(condition, lint_episodes):
    env = lint_episodes[condition]
    _assert_environment_fields_are_neutral(env)
    _assert_no_forbidden_ground_truth_keys(env)

@pytest.mark.parametrize("condition", IMPLEMENTED_CONDITIONS)
def test_legitimate_agent_cyber_prose_is_not_flagged(condition, lint_episodes):
    """
    The stub agent text discusses attacks and malicious traffic. That
    is EXEMPT text (agent-authored content / artifact text), so the
    lint must pass, whereas a naive "scan everything" lint would fail.
    """

    env = lint_episodes[condition]
    _assert_environment_fields_are_neutral(env)

    # Sanity check that the exemption is actually exercised: at least
    # one agent-authored field really does contain the vocabulary.
    vocabulary_seen = any(
        _vocabulary_hits(event.get("content"))
        for event in env.get_observable_events()
        if event.get("event_type") in AGENT_AUTHORED_EVENT_TYPES
    ) or any(
        _vocabulary_hits(artifact.get("text"))
        for artifact in env.get_observable_artifacts()
        if artifact.get("artifact_type") == "message"
    )

    assert vocabulary_seen, (
        "Expected the stubbed agent text to contain attack vocabulary "
        "so the exemption is meaningful; got none."
    )


def test_prompt_infection_payload_reaches_observable_artifact(lint_episodes):
    """
    The attacker payload MUST reach the observable artifact channel
    (that is what the content detector scans). This confirms the lint
    is not passing because the attack never happened.
    """

    env = lint_episodes["prompt_infection"]

    payload_artifacts = [
        artifact
        for artifact in env.get_observable_artifacts()
        if artifact.get("artifact_type") == "tool_result"
        and "ignore the previous instructions"
        in (artifact.get("text") or "").lower()
    ]

    assert payload_artifacts, "expected the injected payload in a tool_result artifact"


def test_clean_episode_has_no_payload_in_artifacts(lint_episodes):
    env = lint_episodes["clean"]

    for artifact in env.get_observable_artifacts():
        lowered = (artifact.get("text") or "").lower()
        assert "ignore the previous instructions" not in lowered
# =================================================================
# TESTS: unclassified event types must fail
# =================================================================

def test_unclassified_event_type_is_rejected():
    """
    A new/unknown event type must FAIL the lint until it is
    classified. We simulate one by adding a stray event.
    """

    from environment.events import MASEvent
    # Fresh episode: this test mutates the env, and the module-scoped
    # shared fixture must stay read-only.
    env = _run_episode("clean")
    env.log_event(
        MASEvent.create(
            event_type="brand_new_emitter",
            sender="environment",
            content="A new kind of event.",
        )
    )

    with pytest.raises(AssertionError):
        _assert_environment_fields_are_neutral(env)

# =================================================================
# TESTS: negative controls
# =================================================================

def test_negative_control_attack_vocabulary_in_metadata_fails():
    """
    An observable event carrying attack vocabulary in its METADATA
    must make the lint fail. This proves the lint has teeth.
    """

    from environment.events import MASEvent
    # Fresh episode: never mutate the shared read-only fixture.
    env = _run_episode("clean")
    env.log_event(
        MASEvent.create(
            event_type="message",
            sender="researcher",
            receiver="executor",
            content="Ordinary message.",
            metadata={"status": "poisoned_by_attacker"},
        )
    )

    with pytest.raises(AssertionError):
        _assert_environment_fields_are_neutral(env)


def test_negative_control_attack_vocabulary_in_environment_content_fails():
    """Attack vocabulary in an environment-authored event CONTENT fails."""

    from environment.events import MASEvent
    # Fresh episode: never mutate the shared read-only fixture.
    env = _run_episode("clean")
    env.log_event(
        MASEvent.create(
            event_type="tool_result",
            sender="tool_manager",
            receiver="researcher",
            content="Tool payload was injected into the agent.",
        )
    )

    with pytest.raises(AssertionError):
        _assert_environment_fields_are_neutral(env)


def test_negative_control_vocabulary_only_in_artifact_text_passes():
    """
    Attack vocabulary ONLY in an artifact's delivered text must PASS:
    that text is exactly the attacker-delivered evidence the defender
    is meant to scan.
    """

    env = _run_episode("clean")
    env.log_artifact(
        artifact_type="tool_result",
        source="internet_search",
        receiver="researcher",
        content="This document mentions an attack payload and malicious injection.",
        request_id="req-negative-control",
    )

    # No exception: artifact text is exempt.
    _assert_environment_fields_are_neutral(env)
    _assert_no_forbidden_ground_truth_keys(env)


def test_negative_control_forbidden_key_in_metadata_fails():
    """A forbidden ground-truth key nested in metadata must fail."""

    env = _run_episode("clean")
    env.log_artifact(
        artifact_type="message",
        source="researcher",
        receiver="executor",
        content="benign",
        message_id="m-forbidden",
    )
    env.observable_artifacts[-1]["metadata"] = {"nested": {"attack_type": "x"}}

    with pytest.raises(AssertionError):
        _assert_no_forbidden_ground_truth_keys(env)

# =================================================================
# EVENT-TYPE CLASSIFICATION IS EXHAUSTIVE FOR THE IMPLEMENTED SET
# =================================================================

def test_all_emitted_event_types_are_classified(lint_episodes):
    """
    Belt-and-braces: every event type any implemented condition emits
    must be classified. This is the explicit statement of the
    invariant the lint enforces entry-by-entry.
    """

    seen = set()
    for env in lint_episodes.values():
        for event in env.get_observable_events():
            seen.add(event.get("event_type"))

    unclassified = seen - KNOWN_EVENT_TYPES
    assert not unclassified, unclassified
