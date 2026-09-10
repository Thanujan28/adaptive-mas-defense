from typing import Any

from attacks.prompt_infection import PromptInfectionAttack


def apply_attack(
    environment: Any,
    attack_condition: str,
    task_id: str,
):

    if attack_condition == "clean":
        return None

    if attack_condition == "prompt_infection":

        attack = PromptInfectionAttack(
            environment=environment,
            source_agent="coordinator",
            target_agent="planner",
            max_hops=3,
        )

        return attack.inject(task_id)

    raise ValueError(
        f"Unknown attack condition: {attack_condition}"
    )