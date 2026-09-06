from environment.mas_environment import MASEnvironment


def main():

    print("=" * 60)
    print("Adaptive MAS Defense - Environment Test")
    print("=" * 60)

    environment = MASEnvironment(
        topology_name="shared_pool",
    )

    print("\nCommunication topology:")
    print(environment.get_topology())

    # Get task dynamically from the user
    task = input("\nEnter your task: ").strip()

    if not task:
        print("Error: Task cannot be empty.")
        return

    print("\nExecuting task...")
    print("-" * 60)

    result = environment.execute_task(task)

    print("\nFinal result:")
    print(result)

    print("\nRecorded MAS events:")
    print("-" * 60)

    for event in environment.get_events():
        print(event)


if __name__ == "__main__":
    main()