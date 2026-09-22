from environment.mas_environment import MASEnvironment
from attacks.prompt_infection import PromptInfectionAttack
from security.logging_config import enable_security_logging
import json
from pathlib import Path


def main():

    # Turn on the security_state logs. Without this the observer's
    # records are emitted but silently dropped, because nothing
    # configures the security.observer logger.
    enable_security_logging()

    print("=" * 60)
    print("Adaptive MAS Defense - Prompt Infection Test")
    print("=" * 60)

    environment = MASEnvironment(
        topology_name="shared_pool",
    )

    print("\nCommunication topology:")
    print(environment.get_topology())

    task = input("\nEnter your task: ").strip()

    if not task:
        print("Error: Task cannot be empty.")
        return

    # ---------------------------------------------------------
    # CREATE PROMPT INFECTION ATTACK
    # ---------------------------------------------------------

    target_agent = "researcher"

    attack = PromptInfectionAttack(
        environment=environment,
        target_agent=target_agent,
    )

    environment.set_attack_simulator(attack)

    print("\nPrompt Infection simulator configured for target agent:", target_agent)

    # ---------------------------------------------------------
    # EXECUTE MAS WITH ATTACK
    # ---------------------------------------------------------

    print("\nExecuting task...")
    print("-" * 60)

    result = environment.execute_task(
        task=task,
    )

    print("\nFinal result:")
    print(result)

    # ---------------------------------------------------------
    # SECURITY STATE ASSESSMENT PER AGENT
    #
    # Every agent response was observed before it was forwarded
    # (see publish_agent_result). Print the resulting assessment
    # for each agent, then the episode aggregate used by the PPO
    # security state.
    # ---------------------------------------------------------

    print_security_assessments(environment)

    # ---------------------------------------------------------
    # SHOW OBSERVABLE EVENTS
    # ---------------------------------------------------------

    print("\nRecorded observable events:")
    print("-" * 60)

    for event in environment.get_observable_events():
        print(event)

    # ---------------------------------------------------------
    # SAVE THIS RUN FOR THE DASHBOARD
    #
    # The dashboard (experiments/security_dashboard.py) runs the tiered
    # observer (Tier 1 chunked semantic + Tier 2 NLI + Tier 3 judge)
    # over records. main.py's own observer is the plain one, so we save
    # exactly what the dashboard needs -- each observed agent response,
    # its task/subtask, and the observable artifacts delivered to that
    # agent as evidence -- to a JSONL the dashboard can reopen.
    # ---------------------------------------------------------

    dump_path = Path("outputs/last_run.jsonl")
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = environment.get_observable_artifacts()
    written = 0
    with dump_path.open("w", encoding="utf-8") as handle:
        for observation in environment.security_observations:
            response = observation.response
            if not isinstance(response, str):
                response = str(response or "")
            evidence = [
                artifact.get("text") or ""
                for artifact in artifacts
                if artifact.get("receiver") == observation.agent_id
            ]
            handle.write(
                json.dumps(
                    {
                        "condition": observation.agent_id,
                        "task": observation.original_task,
                        "subtask": observation.assigned_subtask
                        or observation.agent_id,
                        "output": response,
                        "evidence": evidence,
                    }
                )
                + "\n"
            )
            written += 1

    print(f"\nSaved {written} agent response(s) to {dump_path}")
    print(
        "Open the dashboard on this run with:\n"
        "  python -m experiments.security_dashboard --stub --stub-nli "
        "--stub-judge --from-jsonl outputs/last_run.jsonl"
    )


def print_security_assessments(environment):
    # Print the per-agent security assessment and the episode aggregate.

    observations = environment.security_observations
    print("\n" + "=" * 60)
    print(f"SECURITY STATE ASSESSMENT ({len(observations)} agent responses)")
    print("=" * 60)

    if not observations:
        print("  (no agent responses were observed)")

    for index, observation in enumerate(observations, start=1):

        assessment = observation.semantic_assessment
        detector = observation.detector_result
        print(f"\n[{index}] agent={observation.agent_id} "
              f"subtask={observation.assigned_subtask!r}")

        if assessment is None or not assessment.assessed:
            print("    semantic   : (not assessed)")
        else:
            print(
                f"    semantic   : "
                f"task_sim={assessment.task_similarity:.4f} "
                f"subtask_sim={assessment.subtask_similarity:.4f} "
                f"deviation={assessment.deviation_score:.4f} "
                f"confidence={assessment.confidence:.4f}"
            )

        print(
            f"    evidence   : "
            f"present={detector.get('evidence_present')} "
            f"injection={detector.get('injection_evidence_count')} "
            f"high_conf={detector.get('high_confidence_evidence_count')} "
            f"untrusted={detector.get('untrusted_source_evidence_count')}"
        )

        print(
            f"    score      : "
            f"security_score={observation.security_score:.4f} "
            f"investigate={observation.investigation_required}"
        )

    aggregate = environment.get_semantic_assessment()

    print("\n" + "-" * 60)
    print("EPISODE SEMANTIC AGGREGATE (worst-deviation rule)")
    print("-" * 60)
    print(f"    assessed        = {aggregate.assessed}")
    print(f"    task_similarity = {aggregate.task_similarity:.4f}")
    print(f"    deviation_score = {aggregate.deviation_score:.4f}")
    print(f"    confidence      = {aggregate.confidence:.4f}")


if __name__ == "__main__":
    main()