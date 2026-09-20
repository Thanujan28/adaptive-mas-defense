"""
Lint test for P7: the observable channel (events + artifacts) must
never carry attack vocabulary as EVENT METADATA -- only as content an
attacker actually delivered (artifact "text"), which is expected and
fine (that's what the defender needs to be able to scan for
evidence).

Runs a lightweight simulated episode (no live LLM required) for the
attack conditions actually implemented in this repository: "clean"
and "prompt_infection" (attacks/memory_poisoning.py and
attacks/resource_exhaustion.py are unimplemented stubs -- see
tests/README or the final report).
"""

from __future__ import annotations

from typing import Any

from attacks.prompt_infection import PromptInfectionAttack
from environment.mas_environment import MASEnvironment

ATTACK_VOCABULARY = (
    "infect",
    "inject",
    "poison",
    "attack",
    "compromise",
    "malicious",
    "payload",
)


def _contains_attack_vocabulary(value: Any) -> list[str]:
    hits: list[str] = []

    if isinstance(value, str):
        lowered = value.lower()
        for word in ATTACK_VOCABULARY:
            if word in lowered:
                hits.append(word)

    elif isinstance(value, dict):
        for item in value.values():
            hits.extend(_contains_attack_vocabulary(item))

    elif isinstance(value, (list, tuple)):
        for item in value:
            hits.extend(_contains_attack_vocabulary(item))

    return hits


def _run_clean_condition() -> MASEnvironment:
    env = MASEnvironment(topology_name="layered")

    env.log_artifact(
        artifact_type="tool_result",
        source="internet_search",
        receiver="researcher",
        content=[
            {"content": "Renewable energy adoption has risen steadily since 2020."}
        ],
        request_id="req-clean-1",
    )
    env.send_message(
        sender="researcher",
        receiver="executor",
        content="Key findings: solar adoption rose 12% in 2024.",
    )
    env.log_memory_write(
        agent_name="researcher",
        content="Stored finding about solar adoption trends.",
    )

    return env


def _run_prompt_infection_condition() -> MASEnvironment:
    env = MASEnvironment(topology_name="layered")

    attack = PromptInfectionAttack(target_agent="researcher")
    env.set_attack_simulator(attack)

    real_result = [
        {
            "content": "Renewable energy adoption has risen steadily since 2020.",
        }
    ]

    processed_result = env._process_external_tool_result(
        result=real_result,
        requesting_agent="researcher",
        tool_name="internet_search",
        request_id="req-attack-1",
    )

    # This is exactly what request_tool() does after processing: log
    # the artifact carrying whatever the agent actually received,
    # infected or not, with no ground-truth marker.
    env.log_artifact(
        artifact_type="tool_result",
        source="internet_search",
        receiver="researcher",
        content=processed_result,
        request_id="req-attack-1",
    )

    env.send_message(
        sender="researcher",
        receiver="executor",
        content="Research summary based on the retrieved sources.",
    )

    return env


def _assert_no_attack_vocabulary_in_metadata(env: MASEnvironment) -> None:
    for event in env.get_observable_events():
        # Event-level fields: content/sender/receiver/event_type and
        # metadata must never carry attack vocabulary. These are
        # environment/simulator metadata, not attacker-delivered
        # content (that lives only in artifacts).
        for field in ("event_type", "sender", "receiver", "content"):
            hits = _contains_attack_vocabulary(event.get(field))
            assert not hits, (event, hits)
        hits = _contains_attack_vocabulary(event.get("metadata", {}))
        assert not hits, (event, hits)

    for artifact in env.get_observable_artifacts():
        # Artifact metadata (everything except "text") must also be
        # free of attack vocabulary; "text" is delivered content and
        # MAY legitimately contain it (that's what the attacker put
        # there, and exactly what content-based detection scans).
        for field in ("artifact_type", "source", "receiver"):
            hits = _contains_attack_vocabulary(artifact.get(field))
            assert not hits, (artifact, hits)


def test_clean_condition_has_no_attack_vocabulary_anywhere():
    env = _run_clean_condition()
    _assert_no_attack_vocabulary_in_metadata(env)

    # Clean condition: artifact text is also benign (nothing to
    # exempt).
    for artifact in env.get_observable_artifacts():
        assert not _contains_attack_vocabulary(artifact.get("text"))


def test_prompt_infection_condition_metadata_is_clean():
    env = _run_prompt_infection_condition()
    _assert_no_attack_vocabulary_in_metadata(env)


def test_prompt_infection_artifact_text_may_carry_the_payload():
    # Sanity check that the exemption is meaningful: the delivered
    # artifact text DOES carry attacker content (this is expected and
    # required for the content detector to have anything to scan).
    env = _run_prompt_infection_condition()
    artifacts = env.get_observable_artifacts()
    assert any(
        "ignore the previous instructions" in (a.get("text") or "").lower()
        for a in artifacts
    )
