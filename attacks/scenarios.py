from typing import Any
from attacks.prompt_infection import PromptInfectionAttack
# =============================================================
# IMPLEMENTED ATTACK CONDITIONS (single source of truth)
#
# Only these conditions are actually implemented by
# ``apply_attack``. Anything else (memory_poisoning,
# resource_exhaustion, toma, ...) has no simulator in this
# repository: ``attacks/memory_poisoning.py``,
# ``attacks/resource_exhaustion.py`` and ``attacks/toma.py`` are
# empty stubs. Experiments MUST iterate over this tuple (not a
# hand-maintained list) so that no number is ever reported for an
# unimplemented condition.
# =============================================================

IMPLEMENTED_CONDITIONS: tuple[str, ...] = (
    "clean",
    "prompt_infection",
)

# Attack conditions that experiments may wish to evaluate but which
# are deliberately NOT implemented. Reported as
# "NOT IMPLEMENTED - not evaluated"; never assigned a number.
UNIMPLEMENTED_CONDITIONS: tuple[str, ...] = (
    "memory_poisoning",
    "resource_exhaustion",
)


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
