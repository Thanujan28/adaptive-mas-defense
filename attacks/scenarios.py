from typing import Any

from attacks.prompt_infection import PromptInfectionAttack


def apply_attack(
    environment: Any,
    attack_condition: str,
    task_id: str = "",
    target_agent: str = "researcher",
):
    """
    Apply the selected controlled attack scenario.
    """

    if attack_condition == "clean":
        return None

    if attack_condition == "prompt_infection":
        attack = PromptInfectionAttack(
            target_agent=target_agent,
            environment=environment,
        )
        environment.set_attack_simulator(attack)
        return attack

    raise ValueError(
        f"Unknown attack condition: {attack_condition}"
    )


def record_response(environment: Any, attack_condition: str, task_id: str = ""):
    """Record attack response if needed."""
    pass
