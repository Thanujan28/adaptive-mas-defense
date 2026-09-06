def collect_episode_transitions(ppo_environment, task):
	ppo_environment.reset(task)
	ppo_environment.step(action=None)
	return {
		"transitions": list(ppo_environment.transitions),
		"decision_steps": ppo_environment.step_count,
	}
