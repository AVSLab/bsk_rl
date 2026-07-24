import multiprocessing as mp
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ray
from Basilisk.architecture import bskLogging
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.logger import UnifiedLogger

from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import (
    CondenseMultiStepActions,
    ContinuePreviousAction,
    ContinuePreviousActionAppended,
    MakeAddedStepActionValid,
    TimeDiscountedGAEPPOTorchLearner,
)

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


@dataclass
class BenchmarkEnv:
    env_args: dict
    policies: set
    policy_mapping_fn: Callable
    module_specs: dict
    training_args: dict
    use_async_connectors: bool = False


def get_available_cores():
    """Returns the number of available CPU cores, accounting for SLURM environment variables if present."""
    try:
        processes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"].split("(")[0])
    except Exception:
        processes = mp.cpu_count()
    return processes


def sanitize_np(obj):
    """Make a nested dict with numpy arrays into a nested dict with lists."""
    if type(obj).__module__ == "numpy":
        if isinstance(obj, np.ndarray):
            return [sanitize_np(el) for el in obj]
        else:
            return obj.item()
    if isinstance(obj, dict):
        return {k: sanitize_np(v) for k, v in obj.items()}
    else:
        return obj


def create_new_model(
    output_directory,
    env_args,
    policies,
    policy_mapping_fn,
    module_specs,
    num_env_runners=1,
    use_async_connectors=False,
    training_args={},
    temp_dir="/tmp",
):
    """Configure a PPO model for training with sMDP discounting and asynchronous multiagent actions."""

    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=2_000_000_000,  # 2 GB
        _temp_dir=temp_dir,
    )

    # Add connectors for async multi-agent actions if needed
    module_to_env_connector = dict()
    if use_async_connectors:
        module_to_env_connector = dict(
            module_to_env_connector=lambda env: (ContinuePreviousAction(),)
        )

    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=num_env_runners,
            sample_timeout_s=50000.0,
            **module_to_env_connector,
        )
        .environment(
            env="ConstellationTasking-RLlib",
            env_config=env_args,
        )
        .callbacks(WrappedEpisodeDataCallbacks)
        .reporting(
            metrics_num_episodes_for_smoothing=1,
            metrics_episode_collection_timeout_s=180,
        )
        .checkpointing(export_native_model_files=True)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .resources(num_gpus=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs=module_specs,
            ),
        )
    )

    # Add connectors for async multi-agent actions if needed
    learning_connector = dict()
    if use_async_connectors:
        learning_connector = dict(
            learner_connector=lambda obs_space, act_space: (
                MakeAddedStepActionValid(
                    expected_train_batch_size=config.train_batch_size
                ),
                CondenseMultiStepActions(),
            ),
        )

    config.training(
        **learning_connector,
        learner_class=TimeDiscountedGAEPPOTorchLearner,
        learner_config_dict=dict(reward_time="step_end"),
    )
    config.logger_config = dict(type=UnifiedLogger, logdir=output_directory)

    # Add connectors for async multi-agent actions if needed
    if use_async_connectors:
        old_connector_builder = config.build_module_to_env_connector

        def new_connector_builder(env):
            pipeline = old_connector_builder(env)
            pipeline.insert_after(
                "NormalizeAndClipActions", ContinuePreviousActionAppended()
            )
            return pipeline

        config.build_module_to_env_connector = new_connector_builder

    ppo = PPO(config)

    return ppo


def find_latest_checkpoint(output_directory):
    checkpoint_numbers = []
    for folder_name in os.listdir(output_directory):
        if folder_name.startswith("checkpoint_"):
            num_str = folder_name.split("_")[1]
            if num_str.isdigit():
                checkpoint_numbers.append(int(num_str))

    if not checkpoint_numbers:
        print(f"No checkpoints found in {output_directory}")
        return 0

    return max(checkpoint_numbers)


def load_existing_model(
    output_directory,
    temp_dir="/tmp",
):
    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir

    iter = find_latest_checkpoint(output_directory)
    print(f"Starting training from iteration {iter}")

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=2_000_000_000,  # 2 GB
        _temp_dir=temp_dir,
    )
    checkpoint_path = output_directory / f"checkpoint_{str(iter).zfill(6)}"
    ppo = PPO.from_checkpoint(checkpoint_path)

    return ppo


def train(
    ppo: PPO,
    output_directory: Path,
    checkpoint_frequency=1,
    checkpoints_to_keep=2,
    total_timesteps=1_000_000,
):
    iter = find_latest_checkpoint(output_directory)
    step = 0

    # Track the best return
    current_best_return = -np.inf
    if (output_directory / "checkpoint_best" / "return.txt").exists():
        with open(output_directory / "checkpoint_best" / "return.txt", "r") as file:
            current_best_return = float(file.read().strip())

    while True:
        print(
            f"Starting iteration {iter} at step {step}, current best return: {current_best_return}"
        )

        # Train for one iteration and get the results
        results = ppo.train()
        step = results["num_env_steps_sampled_lifetime"]
        step_return = results["env_runners"].get("episode_return_mean", -np.inf)

        # Check if this is the best return we've seen and save a checkpoint if so
        if step_return > current_best_return:
            checkpoint_path = output_directory / "checkpoint_best"
            # if this directory exists, clear it
            try:
                shutil.rmtree(checkpoint_path)
            except FileNotFoundError:
                pass
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(str(checkpoint_path))
            with open(
                checkpoint_path / f"iteration_{str(iter).zfill(6)}.txt", "w"
            ) as file:
                file.write(f"iter: {iter}\n")
            current_best_return = step_return
            with open(checkpoint_path / "return.txt", "w") as file:
                file.write(f"{current_best_return}\n")

        # Save a checkpoint at the specified frequency
        if iter % checkpoint_frequency == 0:
            checkpoint_path = output_directory / f"checkpoint_{str(iter).zfill(6)}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(str(checkpoint_path))

        # Delete old checkpoints if we've exceeded the number to keep
        if iter > checkpoints_to_keep * checkpoint_frequency - 1:
            for i in range(checkpoint_frequency):
                remove_dir = (
                    output_directory
                    / f"checkpoint_{str(iter - checkpoints_to_keep * checkpoint_frequency - i).zfill(6)}"
                )
                try:
                    shutil.rmtree(remove_dir)
                except FileNotFoundError:
                    pass

        # Check if we've reached the total timesteps for training
        if step > total_timesteps:
            break

        iter += 1


if __name__ == "__main__":
    import argparse
    import sys

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="nadir_science:nadir_science",
        help="Environment to train on. Should be in the format 'module_name:benchmark_name'.",
    )
    parser.add_argument(
        "-j",
        "--num_env_runners",
        type=int,
        default=-1,
        help="Number of environments to train with. If -1, use all available cores minus one.",
    )
    parser.add_argument(
        "-r",
        "--restart",
        action="store_true",
        help="Restart run, deleting existing checkpoints.",
    )
    parser.add_argument(
        "-t",
        "--temp_dir",
        type=str,
        default="/tmp",
        help="Temporary directory for Ray and model checkpoints.",
    )
    parser.add_argument(
        "-s",
        "--total_timesteps",
        type=int,
        default=20_000_000,
        help="Total number of environment steps to train for.",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=1,
        help="Frequency of checkpointing.",
    )
    parser.add_argument(
        "--checkpoints_to_keep",
        type=int,
        default=2,
        help="Number of checkpoints to keep.",
    )
    args = parser.parse_args()

    # Process CLI
    output_dir = Path(args.output_dir)
    temp_dir = args.temp_dir
    if args.num_env_runners != -1:
        num_env_runners = args.num_env_runners
    else:
        num_env_runners = get_available_cores() - 1
    print(f"Tensorboard logging: tensorboard --logdir {output_dir}")

    # Dynamically import the specified benchmark environment
    module_name, benchmark_name = args.env.split(":")
    module = __import__(f"{module_name}", fromlist=[benchmark_name])
    benchmark_env = getattr(module, benchmark_name)

    # Restart run if specified
    if args.restart:
        print(f"Restarting run, deleting existing checkpoints in {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except FileNotFoundError:
            pass

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Make a new PPO model if no checkpoint exists
    if not (output_dir / "checkpoint_best").exists():
        # Record training parameters
        with open(output_dir / "params.txt", "w") as file:
            yaml.dump(
                sanitize_np(
                    {
                        k: getattr(benchmark_env, k)
                        for k in benchmark_env.__dataclass_fields__.keys()
                    }
                ),
                file,
            )
        ppo = create_new_model(
            output_directory=output_dir,
            **{
                k: getattr(benchmark_env, k)
                for k in benchmark_env.__dataclass_fields__.keys()
            },
            num_env_runners=num_env_runners,
            temp_dir=temp_dir,
        )
    # Otherwise, load the existing model and continue training
    else:
        ppo = load_existing_model(
            output_directory=output_dir,
            temp_dir=temp_dir,
        )

    train(
        ppo=ppo,
        output_directory=output_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        checkpoints_to_keep=args.checkpoints_to_keep,
        total_timesteps=args.total_timesteps,
    )

    sys.exit(0)
