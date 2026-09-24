# =============================================================
# run_experiment.py -- PPO / RL reproducibility harness
#
# RELATIONSHIP TO eval_detector.py (read this before running either):
#
#   * experiments/eval_detector.py is the CURRENT tool for DETECTOR
#     evaluation: it measures the security detector's AUROC / TPR@5%FPR
#     and the Tier-3 investigation-strategy cost. It never touches the
#     RL stack.
#
#   * THIS script predates eval_detector.py and is NOT fully covered by
#     it. It drives the RL loop (rl.environment.PPOEnvironment /
#     rl.policy.PPOReward) over conditions x topologies x benchmark
#     tasks, and writes decision-step / transition counts, the PPO
#     security-state signatures and the episode reproducibility check to
#     outputs/experiment_results.json. eval_detector.py does not import
#     or exercise PPOEnvironment at all, so these RL numbers exist ONLY
#     here.
#
#   Keep both: this one is the PPO/reproducibility harness; eval_detector
#   is the detector-evaluation harness. They share the conditions x
#   topologies loop shape but measure different things.
#
# Nothing in the repo imports this module (main.py does not); it is run
# as a script.
# =============================================================

import argparse
import hashlib
import json
import random
import platform
import sys
from pathlib import Path

from benchmark import build_benchmark
from attacks.scenarios import (
    IMPLEMENTED_CONDITIONS,
    UNIMPLEMENTED_CONDITIONS,
    apply_attack,
    record_response,
)
from environment.mas_environment import MASEnvironment
from rl.environment import PPOEnvironment
from rl.policy import PPOReward
TOPOLOGIES = ("centralized", "layered", "fully_connected", "shared_pool")

# Single source of truth lives in attacks/scenarios.py. Only these
# conditions are evaluated; anything unimplemented is reported as
# "NOT IMPLEMENTED - not evaluated" instead of crashing the run or
# producing a fabricated number.
ATTACK_CONDITIONS = IMPLEMENTED_CONDITIONS


def run_experiment(
    output_path: str = "outputs/experiment_results.json",
    limit: int | None = None,
    episodes: int = 30,
    pilot_repetitions: int = 3,
    seed: int = 20260906,
):
    random.seed(seed)
    config_bytes = Path("configs/config.yaml").read_bytes()
    experiment_metadata = {
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model": "llama3.1:8b",
        "runner_version": "resource-controlled-v1",
    }
    results = []

    # Never silently skip or fabricate: state plainly which conditions
    # exist but are not evaluated.
    for condition in UNIMPLEMENTED_CONDITIONS:
        print(f"{condition}: NOT IMPLEMENTED - not evaluated")

    tasks = build_benchmark()[:limit] if limit is not None else build_benchmark()
    for topology in TOPOLOGIES:
        for attack_condition in ATTACK_CONDITIONS:
            for task_spec in tasks:
                environment = MASEnvironment(topology_name=topology)
                ppo_environment = PPOEnvironment(environment, PPOReward())
                ppo_result = _run_with_reproducibility(
                    environment,
                    ppo_environment,
                    task_spec["task"],
                    attack_condition=attack_condition,
                    task_id=task_spec["task_id"],
                    episodes=episodes,
                    pilot_repetitions=pilot_repetitions,
                )
                results.append({
                    "task_id": task_spec["task_id"],
                    "family": task_spec["family"],
                    "topology": topology,
                    "attack_condition": attack_condition,
                    "experiment_metadata": experiment_metadata,
                    **ppo_result,
                })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _run_with_reproducibility(
    environment,
    ppo_environment,
    task,
    attack_condition="clean",
    task_id="",
    episodes=30,
    pilot_repetitions=3,
):
    episode_results = []
    for _ in range(episodes):
        ppo_environment.reset(task)
        apply_attack(environment, attack_condition, task_id)
        state = ppo_environment._state()
        ppo_environment.state = state
        record_response(environment, attack_condition, task_id)
        ppo_environment.step()
        episode_results.append({
            "state": state,
            "decision_steps": ppo_environment.step_count,
            "transition_count": len(ppo_environment.transitions),
            "resource_state": environment.get_resource_state(),
        })

    pilot_signatures = []
    for _ in range(pilot_repetitions):
        ppo_environment.reset(task)
        apply_attack(environment, attack_condition, task_id)
        state = ppo_environment._state()
        ppo_environment.state = state
        ppo_environment.step()
        pilot_signatures.append((state, ppo_environment.step_count))

    first_signature = pilot_signatures[0]
    return {
        "evaluation_episodes": episodes,
        "decision_steps_per_episode": [
            result["decision_steps"] for result in episode_results
        ],
        "transitions_per_episode": [
            result["transition_count"] for result in episode_results
        ],
        "reproducible": all(
            signature == first_signature
            for signature in pilot_signatures[1:]
        ),
        "pilot_signatures": pilot_signatures,
        "resource_state": episode_results[-1]["resource_state"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="outputs/experiment_results.json")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--pilot-repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    run_experiment(
        output_path=args.output,
        limit=args.limit,
        episodes=args.episodes,
        pilot_repetitions=args.pilot_repetitions,
        seed=args.seed,
    )