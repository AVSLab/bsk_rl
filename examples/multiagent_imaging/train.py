"""Shared target-wise PPO training for sensing agents only."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    OBSERVATION_VERSION,
)
from examples.multiagent_imaging.environment import build_environment
from bsk_rl.obs.completion_observations import (
    candidate_snapshot,
    can_continue,
    peer_snapshot,
)


SHARED_POLICY_ID = "imager"


@dataclass(frozen=True)
class LauncherResources:
    """Single-node Ray layout; one Basilisk environment per rollout process."""

    num_env_runners: int = 0
    cpus_per_env_runner: int = 1
    num_learners: int = 0
    gpus_per_learner: float = 0.0
    ray_cpus: int = 2
    torch_threads: int = 1
    sample_timeout_s: float = 1800.0


def worker_seed(base_seed, worker_index=0, vector_index=0):
    """Stable independent 32-bit streams keyed by experiment, worker and vector IDs."""
    return int(
        np.random.SeedSequence(
            [int(base_seed), int(worker_index), int(vector_index), 731]
        ).generate_state(1)[0]
    )


def make_rllib_env(env_config):
    config = dict(env_config)
    config["seed"] = worker_seed(
        config["seed"],
        getattr(env_config, "worker_index", 0),
        getattr(env_config, "vector_index", 0),
    )
    return CompletionPettingZooEnv(build_environment(MultiAgentImagingConfig(**config)))


class CompletionPettingZooEnv(ParallelPettingZooEnv):
    """Let RLlib's API checker sample valid state-dependent discrete actions."""

    def action_space_sample(self, agent_ids=None):
        """Sample the same candidate/continue mask used by the policy actor."""
        selected = set(self.par_env.possible_agents if agent_ids is None else agent_ids)
        result = {}
        for sensor in self.par_env.sensing_satellites:
            if sensor.name not in selected:
                continue
            count = sensor.completion_n_candidates
            snapshot = candidate_snapshot(sensor, count)
            mask = np.asarray(
                [
                    1,
                    1,
                    1,
                    int(sensor.completion_communication_mode == "broadcast"),
                    int(can_continue(sensor)),
                    *(int(t is not None) for t in snapshot.targets),
                    *(
                        [int(p is not None) for p in peer_snapshot(sensor).peers]
                        if sensor.completion_communication_mode == "directed"
                        else []
                    ),
                ],
                dtype=np.int8,
            )
            result[sensor.name] = sensor.action_space.sample(mask=mask)
        return result


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
        "n_peers": config.n_peers,
        "communication_mode": config.communication_mode,
        "peer_features": 12,
        "information_case": config.information_case,
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
        "completion_mask": True,
        "observation_version": OBSERVATION_VERSION,
    }


def build_ppo_config(
    experiment: MultiAgentImagingConfig,
    *,
    train_batch_size: int,
    resources: LauncherResources | None = None,
) -> PPOConfig:
    env_name = "MultiAgentImaging-RLlib"
    resources = resources or LauncherResources()
    register_env(env_name, make_rllib_env)
    sensor_ids = {f"sensor_{index}" for index in range(experiment.n_sensors)}
    policy_mapping = make_shared_policy_mapping(sensor_ids)

    from examples.multiagent_imaging.training_audit import CompletionAuditCallbacks

    config = (
        PPOConfig()
        .callbacks(CompletionAuditCallbacks)
        .environment(env=env_name, env_config=experiment.to_dict())
        .framework("torch")
        .debugging(seed=experiment.seed)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=resources.num_env_runners,
            num_cpus_per_env_runner=resources.cpus_per_env_runner,
            num_envs_per_env_runner=1,
            sample_timeout_s=resources.sample_timeout_s,
            rollout_fragment_length="auto",
            # A physical task may span many peer events. Complete episodes avoid
            # dropping its start/log-probability across rollout-fragment boundaries.
            batch_mode="complete_episodes",
            module_to_env_connector=lambda env: (ContinuePreviousAction(),),
        )
        .resources(num_gpus=0)
        .learners(
            num_learners=resources.num_learners,
            num_gpus_per_learner=resources.gpus_per_learner,
        )
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
            gamma=experiment.discount_per_s,
            lambda_=math.exp(math.log(0.5) / experiment.gae_trace_half_life_s),
            lr=3e-5,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=max(8, train_batch_size // 2),
            num_sgd_iter=1,
            learner_connector=lambda obs_space, act_space: (
                MakeAddedStepActionValid(expected_train_batch_size=train_batch_size),
                CondenseMultiStepActions(gamma=experiment.discount_per_s),
            ),
            learner_class=TimeDiscountedGAEPPOTorchLearner,
            learner_config_dict={"reward_time": "step_start", "lambda_time": "second"},
        )
    )
    return config


def train_run(
    experiment,
    *,
    iterations=1,
    train_batch_size=64,
    output,
    resources=None,
    resume=None,
):
    """Bounded updates with actual sampler, optimizer, checkpoint and cost evidence."""
    import json
    import os
    import torch
    from dataclasses import asdict
    from examples.multiagent_imaging.checkpoints import (
        save_checkpoint,
        resume_checkpoint,
    )
    from examples.multiagent_imaging.readiness import (
        measure,
        provenance,
        write_json,
        save_reproduction_bundle,
    )
    from examples.multiagent_imaging.training_audit import take_audits, parameter_change

    resources = resources or LauncherResources()
    torch.set_num_threads(resources.torch_threads)
    output = Path(output).resolve()
    if (output / "updates.json").exists():
        raise FileExistsError(
            "Use a fresh output directory; existing update evidence must not be overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "provenance.json", provenance(experiment))
    write_json(output / "launcher.json", asdict(resources))
    save_reproduction_bundle(output)
    ray_options = {}
    if os.environ.get("BSK_RL_RAY_TMPDIR"):
        ray_options["_temp_dir"] = os.environ["BSK_RL_RAY_TMPDIR"]
    ray.init(
        ignore_reinit_error=True,
        num_cpus=resources.ray_cpus,
        include_dashboard=False,
        **ray_options,
    )
    algorithm = None
    records = []
    try:
        with measure() as startup:
            algorithm = (
                resume_checkpoint(resume, experiment, resources=resources)
                if resume
                else PPO(
                    build_ppo_config(
                        experiment,
                        train_batch_size=train_batch_size,
                        resources=resources,
                    )
                )
            )
        write_json(output / "startup.json", startup)
        for _ in range(iterations):
            before = algorithm.get_weights([SHARED_POLICY_ID])[SHARED_POLICY_ID]
            with measure() as measurement:
                result = algorithm.train()
            after = algorithm.get_weights([SHARED_POLICY_ID])[SHARED_POLICY_ID]
            change = parameter_change(before, after)
            audits = [
                item
                for worker in algorithm.env_runner_group.foreach_worker(take_audits)
                for item in worker
            ]
            if not audits:
                raise AssertionError("No complete episodes were collected.")
            learners = result["learners"]
            losses = {
                key: float(learners[SHARED_POLICY_ID][key])
                for key in ("total_loss", "policy_loss", "vf_loss")
            }
            if not all(np.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Nonfinite PPO losses.")
            gradients = learners["__all_modules__"]
            sim_seconds = sum(a["simulated_seconds"] for a in audits)
            measurement.update(
                simulated_seconds=sim_seconds,
                sim_seconds_per_wall_second=sim_seconds / measurement["wall_time_s"],
            )
            record = dict(
                iteration=algorithm.training_iteration,
                measurement=measurement,
                losses=losses,
                gradient_l2=float(gradients["gradient_l2"]),
                finite_gradients=float(gradients["finite_gradients"]),
                parameter_change=change,
                requested_minimum_batch=train_batch_size,
                actual_env_steps=sum(a["env_steps"] for a in audits),
                actual_agent_steps=sum(a["agent_steps"] for a in audits),
                actual_policy_decisions=sum(
                    sum(a["policy_decisions"].values()) for a in audits
                ),
                complete_episode_count=len(audits),
                episodes=audits,
            )
            probes = np.asarray(
                [o for a in audits for o in a["probe_observations"]], dtype=np.float32
            )
            checkpoint = output / f"checkpoint_{algorithm.training_iteration:04d}"
            record["restore_validation"] = save_checkpoint(
                algorithm, experiment, checkpoint, probes
            )
            record["checkpoint"] = str(checkpoint)
            records.append(record)
            write_json(output / "updates.json", records)
            print(
                json.dumps(
                    {key: value for key, value in record.items() if key != "episodes"}
                ),
                flush=True,
            )
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs" / "smoke.json"
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument(
        "--checkpoint-dir",
        "--output",
        dest="output",
        type=Path,
        default=Path("results/multiagent_imaging/training"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--cpus-per-env-runner", type=int, default=1)
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--gpus-per-learner", type=float, default=0)
    parser.add_argument("--ray-cpus", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--sample-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()
    if args.iterations < 1 or args.train_batch_size < 8:
        parser.error("Require at least one update and a minimum batch of eight.")
    resources = LauncherResources(
        args.num_env_runners,
        args.cpus_per_env_runner,
        args.num_learners,
        args.gpus_per_learner,
        args.ray_cpus,
        args.torch_threads,
        args.sample_timeout_s,
    )
    train_run(
        MultiAgentImagingConfig.from_json(args.config),
        iterations=args.iterations,
        train_batch_size=args.train_batch_size,
        output=args.output,
        resources=resources,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
