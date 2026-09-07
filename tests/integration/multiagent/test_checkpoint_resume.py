"""Policy export runs actual weights; PPO resume retains optimizer and seed progress."""

from dataclasses import replace
import numpy as np
import pytest
import ray
import torch
from ray.rllib.algorithms.ppo import PPO

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.train import build_ppo_config
from examples.multiagent_imaging.checkpoints import (
    load_policy,
    save_checkpoint,
    resume_checkpoint,
    optimizer_digest,
    policy_callable,
)
from examples.multiagent_imaging.evaluate import run_rollout
from examples.multiagent_imaging.training_audit import parameter_change


def test_checkpoint_outputs_optimizer_resume_and_seed_progress(tmp_path):
    torch.set_num_threads(1)
    config = MultiAgentImagingConfig(
        n_sensors=3,
        n_targets=4,
        n_candidates=2,
        episode_duration_s=360,
        max_step_duration_s=120,
        imaging_duration_s=120,
        information_case="completion",
        communication_mode="directed",
        seed=18,
    )
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    algorithm = None
    try:
        algorithm = PPO(build_ppo_config(config, train_batch_size=16))
        algorithm.train()
        env = algorithm.env_runner.env.unwrapped.par_env
        probes = np.asarray(
            [s.get_obs() for s in env.sensing_satellites], dtype=np.float32
        )
        before = algorithm.get_weights(["imager"])["imager"]
        expected_optimizer = optimizer_digest(algorithm)
        next_episode = env._episode_seed_index
        save_checkpoint(algorithm, config, tmp_path, probes)
        module, evidence = load_policy(tmp_path, config)
        assert evidence["max_logit_error"] == 0
        assert evidence["matched_actions"]
        with pytest.raises(ValueError, match="schema mismatch"):
            load_policy(tmp_path, replace(config, n_candidates=3))
        # Run the exported policy through an actual simulator episode, with a held-
        # out seed. This path cannot silently fall back to the heuristic.
        calls = []
        actual_policy = policy_callable(module)

        def counted_policy(obs):
            result = actual_policy(obs)
            calls.append(result)
            return result

        rollout = run_rollout(replace(config, seed=19), policy=counted_policy)
        assert calls and len(calls) == sum(
            rollout["coordination"]["policy_decisions"].values()
        )
        assert rollout["sim_time_s"] == config.episode_duration_s
        algorithm.stop()
        algorithm = resume_checkpoint(tmp_path, config)
        assert algorithm.training_iteration == 1
        assert optimizer_digest(algorithm) == expected_optimizer
        assert (
            algorithm.env_runner.env.unwrapped.par_env._episode_seed_index
            == next_episode
        )
        result = algorithm.train()
        assert algorithm.training_iteration == 2
        change = parameter_change(before, algorithm.get_weights(["imager"])["imager"])
        assert change["l2"] > 0
        assert np.isfinite(result["learners"]["imager"]["total_loss"])
        assert (
            algorithm.env_runner.env.unwrapped.par_env._episode_seed_index
            > next_episode
        )
        assert optimizer_digest(algorithm) != expected_optimizer
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
