from environment.mas_environment import MASEnvironment
from attacks.prompt_infection import PromptInfectionAttack


def main():

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

    attack = PromptInfectionAttack(
        environment=environment,
        target_agent="planner",
    )

    injection = attack.create_injection(
        task_id="001"
    )

    print("\nPrompt Infection injected into: planner")
    print("Infection ID:", injection["metadata"]["infection_id"])

    # ---------------------------------------------------------
    # EXECUTE MAS WITH ATTACK
    # ---------------------------------------------------------

    print("\nExecuting task...")
    print("-" * 60)

    result = environment.execute_task(
        task=task,
        attack_injections=[
            injection
        ],
    )

    print("\nFinal result:")
    print(result)

    # ---------------------------------------------------------
    # SHOW EVENTS
    # ---------------------------------------------------------

    print("\nRecorded MAS events:")
    print("-" * 60)

    for event in environment.get_events():
        print(event)


if __name__ == "__main__":
    main()