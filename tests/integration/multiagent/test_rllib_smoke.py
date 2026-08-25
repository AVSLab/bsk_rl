import ray
from ray.rllib.algorithms.ppo import PPO

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.train import build_ppo_config


def test_short_shared_policy_rllib_training():
    experiment = MultiAgentImagingConfig(
        n_sensors=2,
        n_targets=4,
        n_candidates=2,
        episode_duration_s=360.0,
        max_step_duration_s=120.0,
        imaging_duration_s=120.0,
        seed=9,
    )
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)
    algorithm = PPO(build_ppo_config(experiment, train_batch_size=16))
    try:
        result = algorithm.train()
        assert result["num_env_steps_sampled_lifetime"] >= 16
    finally:
        algorithm.stop()
        ray.shutdown()
