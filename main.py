from environment.mas_environment import MASEnvironment
from attacks.scenarios import apply_attack


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

    print("\nAttack condition:")
    print("1. clean")
    print("2. prompt_infection")

    choice = input("\nSelect attack: ").strip()

    if choice == "1":
        attack_condition = "clean"

    elif choice == "2":
        attack_condition = "prompt_infection"

    else:
        print("Invalid attack selection.")
        return

    target_agent = input(
        "\nTarget agent [planner]: "
    ).strip()

    if not target_agent:
        target_agent = "planner"

    if target_agent not in environment.agent_names:
        print(
            f"Invalid target agent: {target_agent}"
        )
        return

    task_id = "manual-test-001"

    # ---------------------------------------------------------
    # ATTACK INJECTION
    # ---------------------------------------------------------

    if attack_condition == "prompt_infection":

        print("\nInjecting Prompt Infection...")
        print(
            f"Target: {target_agent}"
        )

        apply_attack(
            environment=environment,
            attack_condition=attack_condition,
            task_id=task_id,
            target_agent=target_agent,
        )

        print("\nPrompt Infection injected.")

    # ---------------------------------------------------------
    # NORMAL MAS EXECUTION
    # ---------------------------------------------------------

    print("\nExecuting task...")
    print("-" * 60)

    result = environment.execute_task(task)

    print("\nFinal result:")
    print(result)

    # ---------------------------------------------------------
    # EVENTS
    # ---------------------------------------------------------

    print("\nRecorded MAS events:")
    print("-" * 60)

    for event in environment.get_events():
        print(event)


if __name__ == "__main__":
    main()