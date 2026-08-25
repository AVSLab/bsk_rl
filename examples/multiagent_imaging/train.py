"""Shared target-wise PPO training for sensing agents only."""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
except (ImportError, ModuleNotFoundError):
    from ray.rllib.core.rl_module.marl_module import (
        MultiAgentRLModuleSpec as MultiRLModuleSpec,
    )

from bsk_rl.utils.rllib.discounting import (
    CondenseMultiStepActions,
    ContinuePreviousAction,
    MakeAddedStepActionValid,
    TimeDiscountedGAEPPOTorchLearner,
)
from bsk_rl.utils.rllib.target_gnn_module import GNNModule

from examples.multiagent_imaging.config import (
    GLOBAL_FEATURES,
    MultiAgentImagingConfig,
    NON_IMAGING_ACTIONS,
)
from examples.multiagent_imaging.environment import build_environment


SHARED_POLICY_ID = "shared_sensor_policy"


def make_shared_policy_mapping(sensor_ids: set[str]):
    """Map only the explicitly configured sensing IDs to one shared module."""
    allowed = frozenset(sensor_ids)

    def mapping(agent_id, *args, **kwargs):
        if agent_id not in allowed:
            raise KeyError(f"Non-sensing agent {agent_id!r} reached policy mapping.")
        return SHARED_POLICY_ID

    return mapping


def target_attention_config(config: MultiAgentImagingConfig) -> dict:
    return {
        "n_targets": config.n_candidates,
        "obs_sat": GLOBAL_FEATURES,
        "width_f": 64,
        "depth_f": 2,
        "block_f": False,
        "width_g": 64,
        "depth_g": 2,
        "tgt_encoded_dim": 64,
        "attention_depth": 1,
        "num_heads": 2,
        "attention_dim": 64,
        "dropout": 0.0,
        "critic_tgt_encoded_dim": 64,
        "critic_width_f": 64,
        "critic_depth_f": 2,
        "critic_block_f": False,
        "critic_width_g": 64,
        "critic_depth_g": 2,
        "critic_pooling_std": False,
        "non_imaging_actions": NON_IMAGING_ACTIONS,
        "condition_on_spacecraft": True,
    }


def build_ppo_config(
    experiment: MultiAgentImagingConfig,
    *,
    train_batch_size: int,
) -> PPOConfig:
    env_name = "MultiAgentImaging-RLlib"
    register_env(
        env_name,
        lambda env_config: ParallelPettingZooEnv(
            build_environment(MultiAgentImagingConfig(**dict(env_config)))
        ),
    )
    sensor_ids = {f"sensor_{index}" for index in range(experiment.n_sensors)}
    policy_mapping = make_shared_policy_mapping(sensor_ids)

    config = (
        PPOConfig()
        .environment(env=env_name, env_config=experiment.to_dict())
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length="auto",
            module_to_env_connector=lambda env: (ContinuePreviousAction(),),
        )
        .resources(num_gpus=0)
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=policy_mapping,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={
                    SHARED_POLICY_ID: RLModuleSpec(
                        module_class=GNNModule,
                        model_config_dict=target_attention_config(experiment),
                    )
                }
            )
        )
        .training(
            gamma=0.999,
            lambda_=0.95,
            lr=3e-5,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=max(8, train_batch_size // 2),
            num_sgd_iter=1,
            learner_connector=lambda obs_space, act_space: (
                MakeAddedStepActionValid(expected_train_batch_size=train_batch_size),
                CondenseMultiStepActions(),
            ),
            learner_class=TimeDiscountedGAEPPOTorchLearner,
            learner_config_dict={"reward_time": "step_start"},
        )
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "smoke.json",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    experiment = MultiAgentImagingConfig.from_json(args.config)
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)
    algorithm = PPO(
        build_ppo_config(
            experiment,
            train_batch_size=args.train_batch_size,
        )
    )
    try:
        for iteration in range(args.iterations):
            result = algorithm.train()
            print(
                f"iteration={iteration} "
                f"sampled={result['num_env_steps_sampled_lifetime']} "
                f"return={result['env_runners'].get('episode_return_mean')}"
            )
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(algorithm.save(str(args.checkpoint_dir)))
    finally:
        algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
