from typing import Any

from attacks.prompt_infection import PromptInfectionAttack


def apply_attack(
    environment: Any,
    attack_condition: str,
    task_id: str,
    target_agent: str = "researcher-1",
):
    """
    Apply the selected controlled attack scenario.
    """

    if attack_condition == "clean":
        return None

    if attack_condition == "prompt_infection":

        attack = PromptInfectionAttack(
            environment=environment,
            target_agent=target_agent,
        )

        return attack.inject(task_id)

    raise ValueError(
        f"Unknown attack condition: {attack_condition}"
    )