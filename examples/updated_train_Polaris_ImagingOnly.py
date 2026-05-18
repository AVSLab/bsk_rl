import os
import shutil
from functools import partial
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dataclasses import asdict
from sim_config import SimConfig
from wandb_config import WandbLogger

EXAMPLES_DIR = Path(__file__).resolve().parent

_DEFAULT_TORCH_THREADS = "11"
_TORCH_THREADS = int(os.environ.get("BSK_RL_TORCH_THREADS", _DEFAULT_TORCH_THREADS))
torch.set_num_threads(_TORCH_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_TORCH_THREADS) # 11 on the cluster historically
PPO_MIN_TRAIN_BATCH_SIZE = 128
PRIORITY_NORM = 2.0  # Most baseline priorities land near [0, 2]; boosted priorities may intentionally exceed 1.
REL_VEL_NORM_MPS = 16_000.0  # [m/s], about twice circular LEO speed for worst-case retrograde encounters.


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    return int(default if raw_value is None else raw_value)


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    return float(default if raw_value is None else raw_value)


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return float(raw_value)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _wandb_key_path() -> Path:
    explicit = os.environ.get("BSK_RL_WANDB_KEY_PATH")
    if explicit:
        return Path(explicit).expanduser()
    local_key = EXAMPLES_DIR / "wandb_key.txt"
    if local_key.exists():
        return local_key
    return Path.cwd() / "wandb_key.txt"


def _maybe_init_wandb(run_name: str, config: dict[str, Any]):
    if not _env_bool("BSK_RL_USE_WANDB", True):
        print("W&B disabled via BSK_RL_USE_WANDB=0")
        return None

    require_wandb = _env_bool("BSK_RL_REQUIRE_WANDB", False)
    key_path = _wandb_key_path()
    if not key_path.exists():
        message = f"W&B key file not found at {key_path}"
        if require_wandb:
            raise FileNotFoundError(message)
        print(f"W&B disabled: {message}")
        return None

    try:
        return WandbLogger(
            project_name=os.environ.get("BSK_RL_WANDB_PROJECT", "amos2026-bsk-rl"),
            run_name=run_name,
            group=os.environ.get(
                "BSK_RL_WANDB_GROUP", "polaris-big-network-imaging-only-obs-v9"
            ),
            key_path=key_path,
            config=config,
        )
    except Exception as exc:
        if require_wandb:
            raise
        print(f"W&B disabled after initialization failure: {exc}")
        return None

import ray
from bsk_rl.utils.utils import get_available_cores
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from bsk_rl.sim import dyn, fsw, world

# from bsk_rl.data import FuelPenalty, RSOInspectionReward
# from bsk_rl.scene import FibonacciSphereRSOPoints
# from bsk_rl.utils.orbital import random_orbit, random_unit_vector, relative_to_chief
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import (  # EpisodeDataCallbacks,
    CondenseMultiStepActions,
    ContinuePreviousAction,
    MakeAddedStepActionValid,
    TimeDiscountedGAEPPOTorchLearner,
)

try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
except (ImportError, ModuleNotFoundError):  # Older versions of RLlib
    from ray.rllib.core.rl_module.marl_module import (
        MultiAgentRLModuleSpec as MultiRLModuleSpec,
    )
    from ray.rllib.core.rl_module.rl_module import (
        SingleAgentRLModuleSpec as RLModuleSpec,
    )

import warnings

try:
    from ray.util import RayDeprecationWarning

    warnings.filterwarnings(
        "ignore",
        category=RayDeprecationWarning,
        message=".*UnifiedLogger.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=RayDeprecationWarning,
        message=".*JsonLogger.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=RayDeprecationWarning,
        message=".*CSVLogger.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=RayDeprecationWarning,
        message=".*TBXLogger.*",
    )
except Exception:
    pass

# os.environ["RAY_DEDUP_LOGS"] = "0"


def train_model(
    model_name,
    output_directory,
    env_args={},
    n_envs=1,
    checkpoint_frequency=1,
    checkpoints_to_keep=2,
    reload_frequency=500_000,
    total_timesteps=1_000_000,
    training_args={},
    temp_dir="/tmp",
    wandb_logger=None,
):
    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if "target" in agent_id:
            return "rso"
        return "inspector"

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=2_000_000_000,  # 2 GB
        _temp_dir=temp_dir,
    )
    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=n_envs,
            sample_timeout_s=50000.0,
            # module_to_env_connector=lambda env: (ContinuePreviousAction(),),
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
            policies={"inspector", "rso"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={
                    "inspector": RLModuleSpec(
                        model_config_dict={
                            "use_lstm": False,
                            "fcnet_hiddens": [2048, 2048],
                            "vf_share_layers": False,
                        },
                    ),
                    "rso": RLModuleSpec(
                        model_config_dict={
                            "use_lstm": False,
                            "fcnet_hiddens": [2, 2],
                            "vf_share_layers": False,
                        },
                    ),
                }
            ),
        )
    )
    config.training(
        **training_args,
        learner_connector=lambda obs_space, act_space: (
            # MakeAddedStepActionValid(expected_train_batch_size=config.train_batch_size),
            # CondenseMultiStepActions(),
        ),
        learner_class=TimeDiscountedGAEPPOTorchLearner,
        learner_config_dict=dict(reward_time="step_start"),
    )
    config.logger_config = dict(
        type=UnifiedLogger, logdir=output_directory / model_name
    )

    ppo = PPO(config)

    iter = 0
    step = 0

    current_best_return = -np.inf

    try:
        while True:
            prev_step = step
            print(f"[train] starting iteration {iter} at sampled_steps={step}", flush=True)
            results = ppo.train()
            step = results["num_env_steps_sampled_lifetime"]
            step_return = results["env_runners"].get("episode_return_mean", -np.inf)
            print(
                "[train] finished iteration "
                f"{iter}: sampled_steps={step}, episode_return_mean={step_return}",
                flush=True,
            )

            if wandb_logger is not None:
                wandb_logger.log(results)

            if step_return > current_best_return:
                checkpoint_path = output_directory / model_name / "checkpoint_best"
                # if this directory exists, clear it
                try:
                    shutil.rmtree(checkpoint_path)
                except FileNotFoundError:
                    pass
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)
                with open(
                    checkpoint_path / f"iteration_{str(iter).zfill(6)}.txt", "w"
                ) as file:
                    file.write(f"iter: {iter}\n")
                current_best_return = step_return

            checkpoint_path = (
                output_directory / model_name / f"checkpoint_{str(iter).zfill(6)}"
            )
            if iter % checkpoint_frequency == 0:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)

            if step > total_timesteps:
                break

            if step % reload_frequency < prev_step % reload_frequency:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                ppo.save_checkpoint(checkpoint_path)
                ray.shutdown()
                ray.init(
                    ignore_reinit_error=True,
                    num_cpus=get_available_cores(),
                    object_store_memory=3_000_000_000,  # 2 GB
                    _temp_dir=temp_dir,
                )
                ppo = PPO.from_checkpoint(checkpoint_path)

            if iter > checkpoints_to_keep * checkpoint_frequency - 1:
                for i in range(checkpoint_frequency):
                    remove_dir = (
                        output_directory
                        / model_name
                        / f"checkpoint_{str(iter - checkpoints_to_keep * checkpoint_frequency - i).zfill(6)}"
                    )
                    try:
                        shutil.rmtree(remove_dir)
                    except FileNotFoundError:
                        pass

            iter += 1
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


def _finite_values(values):
    """Return finite floats for callback metrics without breaking RLlib logging."""
    finite = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            finite.append(value)
    return finite


def _safe_mean(values, default=-1.0):
    values = _finite_values(values)
    return float(np.mean(values)) if values else default


def _safe_median(values, default=-1.0):
    values = _finite_values(values)
    return float(np.median(values)) if values else default


def _safe_std(values, default=-1.0):
    values = _finite_values(values)
    return float(np.std(values)) if values else default


def _safe_max(values, default=-1.0):
    values = _finite_values(values)
    return float(np.max(values)) if values else default


def _safe_min(values, default=-1.0):
    values = _finite_values(values)
    return float(np.min(values)) if values else default


def env_metrics_callback(env):
    data = {}
    reward_data = env.rewarder.data

    # Number of unique targets successfully imaged
    num_imaged = len(reward_data.imaged)
    data["num_unique_targets_imaged"] = num_imaged

    # Episode duration
    episode_duration = env.simulator.sim_time
    data["episode_duration_sec"] = episode_duration

    aliveness = env.satellites[0].dynamics.is_alive()
    data["number of alive cases"] = aliveness

    battery_valid = env.satellites[0].dynamics.battery_valid()
    data["battery_valid"] = battery_valid

    rw_valid = env.satellites[0].dynamics.rw_speeds_valid()
    data["rw_valid"] = rw_valid

    # Legacy fixed-duration estimate kept for continuity with older TensorBoard plots.
    total_imaging_time = num_imaged * 300  # Each successful image takes 300s
    idle_time = episode_duration - total_imaging_time
    num_unproductive_actions = idle_time / 300
    data["non-imaging_action_count"] = int(round(num_unproductive_actions))

    # Legacy fixed-duration estimate kept for continuity with older TensorBoard plots.
    total_imaging_time = num_imaged * 300  # Each successful image takes 300s
    non_imaging_time = episode_duration - total_imaging_time
    data["non-imaging_time"] = int(round(non_imaging_time))

    data["cumulativeRewardSS1"]=env.rewarder.cum_reward['SS1']
    data["illuminated_images"] = len(env.rewarder.imaged_illuminated)

    SS1_actions_spec = env.satellites[0].action_builder.action_spec[0]
    target_priority_by_id = {}
    try:
        target_priority_by_id = {
            int(target.id): float(target.priority)
            for target in env.scenario.target_spacecrafts
        }
    except Exception:
        target_priority_by_id = {}

    all_target_priorities = list(target_priority_by_id.values())
    data["target_priority_sum"] = float(np.sum(all_target_priorities)) if all_target_priorities else -1
    data["target_priority_mean"] = _safe_mean(all_target_priorities)
    data["target_priority_std"] = _safe_std(all_target_priorities)
    data["target_priority_max"] = _safe_max(all_target_priorities)
    data["mean_target_priority"] = data["target_priority_mean"]
    data["std_target_priority"] = data["target_priority_std"]
    data["max_target_priority"] = data["target_priority_max"]

    scenario = getattr(env, "scenario", None)
    hio_ids = set(getattr(scenario, "hio_target_ids", []))
    shio_ids = set(getattr(scenario, "shio_target_ids", []))
    event_time = getattr(scenario, "priority_event_time", None)
    if event_time is None and scenario is not None:
        event_time = getattr(scenario, "dynamic_priority_event_time_sec", None)
    if event_time is None and scenario is not None:
        event_time = float(env.time_limit) * float(
            getattr(scenario, "dynamic_priority_event_fraction", 0.5)
        )
    event_time = float(event_time) if event_time is not None else -1.0
    event_applied_time = getattr(scenario, "priority_event_applied_time", None)
    event_targets = [
        target
        for target in getattr(scenario, "target_spacecrafts", [])
        if getattr(target, "priority_event_kind", "") in {"HIO", "SHIO"}
    ]
    hio_targets = [
        target for target in event_targets if getattr(target, "priority_event_kind", "") == "HIO"
    ]
    shio_targets = [
        target for target in event_targets if getattr(target, "priority_event_kind", "") == "SHIO"
    ]
    data["dynamic_priority_event_enabled"] = float(
        bool(getattr(scenario, "dynamic_priority_event_enabled", False))
    )
    data["dynamic_priority_event_applied"] = float(
        bool(getattr(scenario, "priority_event_applied", False))
    )
    data["dynamic_priority_event_time_sec"] = event_time
    data["dynamic_priority_event_applied_time_sec"] = (
        float(event_applied_time) if event_applied_time is not None else -1.0
    )
    data["hio_target_count"] = len(hio_ids)
    data["shio_target_count"] = len(shio_ids)
    data["hio_candidate_access_count"] = int(
        sum(getattr(target, "priority_event_candidate_count", 0) for target in hio_targets)
    )
    data["shio_candidate_access_count"] = int(
        sum(getattr(target, "priority_event_candidate_count", 0) for target in shio_targets)
    )
    data["time_to_first_hio_candidate_access_sec"] = _safe_min(
        getattr(target, "priority_event_first_candidate_time", None)
        for target in hio_targets
    )
    data["time_to_first_shio_candidate_access_sec"] = _safe_min(
        getattr(target, "priority_event_first_candidate_time", None)
        for target in shio_targets
    )

    # These records come from ImageRSO and capture the real variable-duration action
    # time, not the nominal 300 s maximum.
    imaging_attempt_records = list(
        getattr(SS1_actions_spec, "imaging_attempt_records", [])
    )
    imaging_action_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if record.get("start_time") is not None and record.get("end_time") is not None
    ]
    successful_imaging_action_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if record.get("success")
        and record.get("start_time") is not None
        and record.get("end_time") is not None
    ]
    unsuccessful_imaging_action_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if not record.get("success")
        and record.get("start_time") is not None
        and record.get("end_time") is not None
    ]
    imaging_slew_times = [
        record.get("slew_time_s") for record in imaging_attempt_records
    ]
    successful_imaging_slew_times = [
        record.get("slew_time_s")
        for record in imaging_attempt_records
        if record.get("success")
    ]
    unsuccessful_imaging_slew_times = [
        record.get("slew_time_s")
        for record in imaging_attempt_records
        if not record.get("success")
    ]
    imaging_success_flags = [
        bool(record.get("success")) for record in imaging_attempt_records
    ]
    attempted_priorities = [
        record.get(
            "target_priority",
            target_priority_by_id.get(int(record["target_id"])),
        )
        for record in imaging_attempt_records
        if record.get("target_id") is not None
    ]
    successful_capture_priorities = [
        record.get(
            "target_priority",
            target_priority_by_id.get(int(record["target_id"])),
        )
        for record in imaging_attempt_records
        if record.get("success") and record.get("target_id") is not None
    ]
    data["num_imaging_attempts"] = len(imaging_attempt_records)
    data["imaging_attempt_success_rate"] = _safe_mean(imaging_success_flags)
    data["actual_imaging_action_time_sec"] = float(np.sum(imaging_action_durations))
    data["actual_non_imaging_time_sec"] = (
        float(episode_duration) - data["actual_imaging_action_time_sec"]
    )
    data["mean_imaging_action_duration_sec"] = _safe_mean(imaging_action_durations)
    data["median_imaging_action_duration_sec"] = _safe_median(
        imaging_action_durations
    )
    data["mean_successful_imaging_action_duration_sec"] = _safe_mean(
        successful_imaging_action_durations
    )
    data["median_successful_imaging_action_duration_sec"] = _safe_median(
        successful_imaging_action_durations
    )
    data["mean_unsuccessful_imaging_action_duration_sec"] = _safe_mean(
        unsuccessful_imaging_action_durations
    )
    data["median_unsuccessful_imaging_action_duration_sec"] = _safe_median(
        unsuccessful_imaging_action_durations
    )
    data["mean_imaging_slew_time_sec"] = _safe_mean(imaging_slew_times)
    data["median_imaging_slew_time_sec"] = _safe_median(imaging_slew_times)
    data["mean_successful_imaging_slew_time_sec"] = _safe_mean(
        successful_imaging_slew_times
    )
    data["median_successful_imaging_slew_time_sec"] = _safe_median(
        successful_imaging_slew_times
    )
    data["mean_unsuccessful_imaging_slew_time_sec"] = _safe_mean(
        unsuccessful_imaging_slew_times
    )
    data["median_unsuccessful_imaging_slew_time_sec"] = _safe_median(
        unsuccessful_imaging_slew_times
    )
    data["mean_attempted_target_priority"] = _safe_mean(attempted_priorities)
    data["mean_successful_capture_priority"] = _safe_mean(
        successful_capture_priorities
    )
    chosen_pairs = [
        (int(target_id), float(command_time))
        for target_id, command_time in zip(
            getattr(SS1_actions_spec, "chosen_target_ids", []),
            getattr(SS1_actions_spec, "imaging_times", []),
        )
        if command_time is not None and float(command_time) >= event_time
    ]
    hio_command_times = [t for target_id, t in chosen_pairs if target_id in hio_ids]
    shio_command_times = [t for target_id, t in chosen_pairs if target_id in shio_ids]
    successful_event_records = [
        record
        for record in imaging_attempt_records
        if record.get("success")
        and record.get("target_id") is not None
        and float(record.get("first_capture_time") or record.get("end_time") or -1.0)
        >= event_time
    ]
    hio_success_times = [
        float(record.get("first_capture_time") or record.get("end_time"))
        for record in successful_event_records
        if int(record["target_id"]) in hio_ids
    ]
    shio_success_times = [
        float(record.get("first_capture_time") or record.get("end_time"))
        for record in successful_event_records
        if int(record["target_id"]) in shio_ids
    ]
    data["hio_command_count_after_event"] = len(hio_command_times)
    data["shio_command_count_after_event"] = len(shio_command_times)
    data["hio_successful_capture_count_after_event"] = len(hio_success_times)
    data["shio_successful_capture_count_after_event"] = len(shio_success_times)
    data["time_to_first_hio_command_sec"] = _safe_min(hio_command_times)
    data["time_to_first_shio_command_sec"] = _safe_min(shio_command_times)
    data["time_to_first_hio_success_sec"] = _safe_min(hio_success_times)
    data["time_to_first_shio_success_sec"] = _safe_min(shio_success_times)

    verified_records = list(
        getattr(env.rewarder.data, "verified_useful_records", [])
    )
    verified_priorities = [
        target_priority_by_id.get(int(record["target_id"]))
        for record in verified_records
        if record.get("target_id") is not None
    ]
    data["mean_verified_useful_priority"] = _safe_mean(verified_priorities)
    data["max_verified_useful_priority"] = _safe_max(verified_priorities)

    total_downlinks = int(getattr(env.rewarder, "total_downlinks", 0))
    useful_downlinks = int(getattr(env.rewarder, "useful_downlinks", 0))
    failed_downlinks = int(getattr(env.rewarder, "failed_downlinks", 0))
    data["total_downlinks"] = total_downlinks
    data["useful_downlinks"] = useful_downlinks
    data["failed_downlinks"] = failed_downlinks
    data["bad_downlinks"] = int(getattr(env.rewarder, "bad_downlinks", failed_downlinks))
    data["pending_images"] = int(getattr(env.rewarder, "pending_images", 0))
    data["verified_image_count"] = int(getattr(env.rewarder, "verified_image_count", 0))
    data["reimage_count"] = int(getattr(env.rewarder, "reimage_count", 0))
    data["useful_downlink_fraction"] = (
        useful_downlinks / total_downlinks if total_downlinks > 0 else 0.0
    )
    cooldown_until_by_id = getattr(reward_data, "cooldown_until_by_id", {})
    pending_by_id = getattr(reward_data, "pending_image_records_by_id", {})
    active_cooldown_ids = {
        int(target_id)
        for target_id, cooldown_until in cooldown_until_by_id.items()
        if float(episode_duration) < float(cooldown_until)
    }
    pending_target_ids = {
        int(target_id) for target_id, records in pending_by_id.items() if records
    }
    temporarily_ineligible_ids = set(active_cooldown_ids)
    if getattr(reward_data, "hide_pending_targets", True):
        temporarily_ineligible_ids.update(pending_target_ids)
    data["cooldown_target_count_legacy"] = len(cooldown_until_by_id)
    data["cooldown_target_count"] = len(active_cooldown_ids)
    data["pending_verification_target_count"] = len(pending_target_ids)
    data["temporarily_ineligible_target_count"] = len(temporarily_ineligible_ids)

    # Target azimuth
    if hasattr(SS1_actions_spec, "chosen_target_azimuth") and SS1_actions_spec.chosen_target_azimuth:
        data["mean_target_azimuth"] = np.mean(SS1_actions_spec.chosen_target_azimuth)
        data["std_target_azimuth"] = np.std(SS1_actions_spec.chosen_target_azimuth)


    # Target elevation (inertial)
    if hasattr(SS1_actions_spec, "chosen_target_elevation") and SS1_actions_spec.chosen_target_elevation:
        data["mean_target_elevation"] = np.mean(SS1_actions_spec.chosen_target_elevation)
        data["std_target_elevation"] = np.std(SS1_actions_spec.chosen_target_elevation)
    else:
        data["mean_target_elevation"] = -1
        data["std_target_elevation"] = -1

    # Relative position in H-frame
    if hasattr(SS1_actions_spec, "chosen_target_rel_pos_H") and SS1_actions_spec.chosen_target_rel_pos_H:
        mean_rel_pos = np.mean(SS1_actions_spec.chosen_target_rel_pos_H, axis=0)
        std_rel_pos = np.std(SS1_actions_spec.chosen_target_rel_pos_H, axis=0)
        for i, axis in enumerate(['x', 'y', 'z']):
            data[f"mean_rel_pos_H_{axis}"] = mean_rel_pos[i]
            data[f"std_rel_pos_H_{axis}"] = std_rel_pos[i]
    else:
        for axis in ['x', 'y', 'z']:
            data[f"mean_rel_pos_H_{axis}"] = -1
            data[f"std_rel_pos_H_{axis}"] = -1

    # Target elevation (local)
    if hasattr(SS1_actions_spec, "chosen_target_elevation_local") and SS1_actions_spec.chosen_target_elevation_local:
        data["mean_target_elevation_local"] = np.mean(SS1_actions_spec.chosen_target_elevation_local)
        data["std_target_elevation_local"] = np.std(SS1_actions_spec.chosen_target_elevation_local)
    else:
        data["mean_target_elevation_local"] = -1
        data["std_target_elevation_local"] = -1

    # Initial angular error
    if hasattr(SS1_actions_spec, "initial_angular_error") and SS1_actions_spec.initial_angular_error:
        data["mean_initial_ang_error"] = np.mean(SS1_actions_spec.initial_angular_error)
        data["std_initial_ang_error"] = np.std(SS1_actions_spec.initial_angular_error)
    else:
        data["mean_initial_ang_error"] = -1
        data["std_initial_ang_error"] = -1

    # Target distance
    if hasattr(SS1_actions_spec, "chosen_target_distance") and SS1_actions_spec.chosen_target_distance:
        data["mean_target_distance"] = np.mean(SS1_actions_spec.chosen_target_distance)
        data["std_target_distance"] = np.std(SS1_actions_spec.chosen_target_distance)
    else:
        data["mean_target_distance"] = -1
        data["std_target_distance"] = -1

    # Illumination status
    if hasattr(SS1_actions_spec, "chosen_target_illumination_status") and SS1_actions_spec.chosen_target_illumination_status:
        data["mean_target_illumination_status"] = np.mean(SS1_actions_spec.chosen_target_illumination_status)
        well_illuminated=0
        not_illuminated=0
        for ill in SS1_actions_spec.chosen_target_illumination_status:
            if ill > 0.5:
                well_illuminated+=1
            else:
                not_illuminated+=1
        data["num_target_above_illumination_threshold"] = well_illuminated
        data["num_target_below_illumination_threshold"] = not_illuminated
    else:
        data["mean_target_illumination_status"] = -1

    # Target priority
    if hasattr(SS1_actions_spec, "chosen_target_priority") and SS1_actions_spec.chosen_target_priority:
        chosen_priority_mean = np.mean(SS1_actions_spec.chosen_target_priority)
        chosen_priority_std = np.std(SS1_actions_spec.chosen_target_priority)
        chosen_priority_max = np.max(SS1_actions_spec.chosen_target_priority)
        data["mean_chosen_target_priority"] = chosen_priority_mean
        data["std_chosen_target_priority"] = chosen_priority_std
        data["max_chosen_target_priority"] = chosen_priority_max
    else:
        data["mean_chosen_target_priority"] = -1
        data["std_chosen_target_priority"] = -1
        data["max_chosen_target_priority"] = -1

    # Ever visible flags
    if hasattr(SS1_actions_spec, "ever_visible") and SS1_actions_spec.ever_visible:
        data["target_ever_visible_fraction"] = len(SS1_actions_spec.ever_visible) / n_targets
    else:
        data["target_ever_visible_fraction"] = -1

    return data

def sat_metrics_callback(env, satellite):
    data = {}
    if satellite.name == 'SS1':
        # Keep this callback quiet on the cluster; these values still go to RLlib metrics.
        data["RW_norm"] = np.linalg.norm(satellite.dynamics.wheel_speeds)
        data["RW1"] = satellite.dynamics.wheel_speeds[0]
        data["RW2"] = satellite.dynamics.wheel_speeds[1]
        data["RW3"] = satellite.dynamics.wheel_speeds[2]

        data["battery_charge_fraction"] = satellite.dynamics.battery_charge_fraction
        data["storage_level_fraction"] = satellite.dynamics.storage_level_fraction
        data["Total Images Downlinked"] = getattr(satellite.dynamics, "total_downlinks", 0)
        data["Useful Images Downlinked"] = getattr(satellite.dynamics, "useful_downlinks", 0)
        data["Failed Images Downlinked"] = getattr(satellite.dynamics, "failed_downlinks", 0)
        data["Bad Images Downlinked"] = getattr(satellite.dynamics, "bad_downlinks", 0)
        data["Verified Reimages"] = getattr(satellite.dynamics, "reimage_count", 0)

    else:
        data["RW_norm"] = 0
        data["RW1"] = 0
        data["RW2"] = 0
        data["RW3"] = 0

        data["battery_charge_fraction"] = 0
        data["storage_level_fraction"] = 0
        data["Total Images Downlinked"] = 0
        data["Useful Images Downlinked"] = 0
        data["Failed Images Downlinked"] = 0
        data["Bad Images Downlinked"] = 0
        data["Verified Reimages"] = 0

    return data



if __name__ == "__main__":
    import sys
    import time

    import yaml
    from bsk_rl.utils.utils import build_job_array, sanitize_np
    from bsk_rl.data.rso_targets_data import RSOTargetImageData, RSOTargetImageReward, RSOTargetImageStore
    from bsk_rl.scene.rso_targets import RandomSatellites, RSOTarget
    from bsk_rl import act, data, obs, scene, sats

    from bsk_rl import comm
    from bsk_rl.utils.orbital import walker_delta_args
    from bsk_rl.sim import dyn, fsw

    from Basilisk.utilities import (
        macros,
        orbitalMotion,
    )

    # n_targets = 100
    # n_targets_ahead = 10
    # imaging_duration = 300
    # extra_tima_factor = 1.5
    # total_time = extra_tima_factor * n_targets * 300  #I give it 10 times the minimum time to finish
    # Shared AMOS sim configuration. Priority distribution defaults live in SimConfig:
    # uniform priorities are rescaled so each episode sums to priority_sum=100.
    sim_cfg = SimConfig(
        n_targets=_env_int("BSK_RL_N_TARGETS", 100),
        n_targets_ahead=_env_int("BSK_RL_N_TARGETS_AHEAD", 10),
        imaging_duration=_env_float("BSK_RL_IMAGING_DURATION", 300.0),
        extra_time_factor=_env_float("BSK_RL_EXTRA_TIME_FACTOR", 1.5),
        obs_v=9.0,
        just_imaging=True,
        # No downlink action exists here; reward/cool down images at capture time.
        verify_image_quality_on_downlink=False,
        hide_pending_targets=False,
        dynamic_priority_event_enabled=_env_bool("BSK_RL_DYNAMIC_PRIORITY_EVENT", True),
        dynamic_priority_event_time_sec=_env_optional_float(
            "BSK_RL_DYNAMIC_PRIORITY_EVENT_TIME_SEC"
        ),
        dynamic_priority_event_fraction=_env_float(
            "BSK_RL_DYNAMIC_PRIORITY_EVENT_FRACTION", 0.5
        ),
        hio_count=_env_int("BSK_RL_HIO_COUNT", 5),
        hio_priority=_env_float("BSK_RL_HIO_PRIORITY", 5.0),
        shio_count=_env_int("BSK_RL_SHIO_COUNT", 3),
        shio_priority=_env_float("BSK_RL_SHIO_PRIORITY", 10.0),
        dynamic_priority_event_seed=_env_optional_int(
            "BSK_RL_DYNAMIC_PRIORITY_EVENT_SEED"
        ),
    )

    n_targets = sim_cfg.n_targets
    n_targets_ahead = sim_cfg.n_targets_ahead
    imaging_duration = sim_cfg.imaging_duration
    total_time = sim_cfg.total_time

    def make_rso_scenario():
        return scene.RandomSatellites(
            "SS1",
            n_targets=n_targets,
            priority_mode=sim_cfg.priority_mode,
            priority_sum=sim_cfg.priority_sum,
            rescale_priorities_to_sum=sim_cfg.rescale_priorities_to_sum,
            priority_constant=sim_cfg.priority_constant,
            priority_uniform_low=sim_cfg.priority_uniform_low,
            priority_uniform_high=sim_cfg.priority_uniform_high,
            priority_gaussian_mean=sim_cfg.priority_gaussian_mean,
            priority_gaussian_std=sim_cfg.priority_gaussian_std,
            priority_min=sim_cfg.priority_min,
            priority_max=sim_cfg.priority_max,
            dynamic_priority_event_enabled=sim_cfg.dynamic_priority_event_enabled,
            dynamic_priority_event_time_sec=sim_cfg.dynamic_priority_event_time_sec,
            dynamic_priority_event_fraction=sim_cfg.dynamic_priority_event_fraction,
            hio_count=sim_cfg.hio_count,
            hio_priority=sim_cfg.hio_priority,
            shio_count=sim_cfg.shio_count,
            shio_priority=sim_cfg.shio_priority,
            dynamic_priority_event_seed=sim_cfg.dynamic_priority_event_seed,
        )

    def make_rso_rewarder():
        return data.RSOTargetImageReward(
            reimage_cooldown_orbits=sim_cfg.reimage_cooldown_orbits,
            # The full AMOS lifecycle verifies useful images at downlink. This
            # image-only baseline has no downlink action, so match the Target-GNN
            # image-only setup and give reward/cooldown credit at capture time.
            verify_image_quality_on_downlink=False,
            hide_pending_targets=False,
            image_quality_threshold=sim_cfg.image_quality_threshold,
        )

    def make_downlink_action(duration: float):
        return act.Downlink(
            duration=duration,
            variable_duration_downlink=sim_cfg.variable_duration_downlink,
            empty_storage_threshold_bits=sim_cfg.downlink_empty_threshold_bits,
        )


    class MyScanningSatellite(sats.AccessSatellite):
        observation_spec = [
            # Imaging-only: expose only target-selection information. Battery,
            # storage, and ground-station state are intentionally removed.
            obs.PolarisScTargetProperties(
                dict(prop="priority", norm=PRIORITY_NORM),
                dict(prop="target_elevation_angle", norm=90.0),
                dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                dict(prop="rel_vel_vector_v_BR_H", norm=REL_VEL_NORM_MPS),
                dict(prop="angle_to_target", norm=90.0),
                dict(prop="target_distance", norm=15960 * 1000), # normalization calculated assuming h = 800 km and min elevation is -14 deg
                dict(prop="target_shadowFactor", norm=1.0),
                n_ahead_observe=n_targets_ahead,
            ),
        ]
        action_spec = [
            act.ImageRSO(
                n_ahead_image=n_targets_ahead,
                duration=imaging_duration,
                variable_duration_imaging=sim_cfg.variable_duration_imaging,
                min_pointing_hold_s=sim_cfg.min_pointing_hold_s,
                hold_mode=sim_cfg.hold_mode,
                require_illumination_during_hold=sim_cfg.require_illumination_during_hold,
                hold_illumination_threshold=sim_cfg.hold_illumination_threshold,
            ),  # Scan for 5 minute
            # act.Charge(duration=300.0),  # Charge for 5 minutes
            # make_downlink_action(300.0), # Downlink for 3 min
            # act.Desat(duration=150), # Desat for 2.5 min

        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    sat_args = {}
    # Set some parameters as constants
    sat_args["imageAttErrorRequirement"] = 0.0025
    sat_args["imageRateErrorRequirement"] = 0.01

    # Storage
    image_bits = 8e6 / 2
    image_storage_capacity = _env_float("BSK_RL_IMAGE_STORAGE_CAPACITY_IMAGES", 500.0)
    sat_args["dataStorageCapacity"] = image_storage_capacity * image_bits
    sat_args["storageInit"] = lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2
    sat_args["instrumentBaudRate"] = 0.5 * 8e6
    sat_args["transmitterBaudRate"] = -0.5 * 8e6

    # Power
    battery_life_multiplier = _env_float("BSK_RL_BATTERY_LIFE_MULTIPLIER", 1000.0)
    baseline_battery_ws = 500 * 3600

    sat_args["batteryStorageCapacity"] = battery_life_multiplier * baseline_battery_ws # W*s
    sat_args["storedCharge_Init"] = lambda: np.random.uniform(1.0, 1.0) * battery_life_multiplier * baseline_battery_ws # lambda: np.random.uniform(0.8, 1.0) * battery_life_multiplier * baseline_battery_ws
    sat_args["basePowerDraw"] = -10.0  # W
    sat_args["instrumentPowerDraw"] = -30.0  # W
    sat_args["transmitterPowerDraw"] = -25.0  # W
    sat_args["thrusterPowerDraw"] = -80.0  # W
    sat_args["panelArea"] = 1.0  # m^2

    # Attitude
    # sat_args["imageAttErrorRequirement"] = 0.1
    # sat_args["imageRateErrorRequirement"] = 0.1
    sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.000, size=3)  # N*m
    sat_args["maxWheelSpeed"] = 6000.0  # RPM
    sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
    sat_args["desatAttitude"] = "sun" # 'nadir' and 'sun' is the other option

    # Alpha=0.2 legacy reward split. In downlink-verification mode, reward is paid
    # after useful downlink, but these fields still recover old behavior if disabled.
    sat_args["downlink_bonus"] = 0.0
    sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
    sat_args["eclipse_threshold_for_imaging"] = 0.5 # to include both shadowed and illuminated RSOs
    sat_args["eclipse_threshold_for_reward"] = 0.5 # can be the same as sat_args["eclipse_threshold_for_imaging"] if set to a positive number between 0 and 1
    # sat_args["full_storage_penalty"] = -1
    # sat_args["low_battery_penalty"] = -1
    sat_args["empty_downlink_penalty"] = -1

    class MyTargetSatellite(sats.Satellite):
        observation_spec = [
            obs.Time(),
        ]
        action_spec = [
            act.Drift(duration=total_time),  # Scan for 1 minute
            # act.Charge(duration=600.0),  # Charge for 10 minutes
        ]
        dyn_type = dyn.BasicTargetDynamicsModel  # Passed as a type
        fsw_type = fsw.BasicTargetFSWModel

    R_E = 6371e3  # [m]
    D2R = macros.D2R

    # Default altitude bands (altitude above Earth's mean radius)
    DEFAULT_ALT_BOUNDS = {
        "LEO": (400e3, 2000e3),
        "MEO": (2000e3, 35000e3),
        "GEO": (35786e3 - 300e3, 35786e3 + 300e3),  # ~GEO ring
    }

    def _sample_for_regime(regime: str,
                           altitude_bounds: dict[str, tuple[float, float]],
                           min_perigee_alt: float) -> orbitalMotion.ClassicElements:
        oe = orbitalMotion.ClassicElements()

        h_min, h_max = altitude_bounds[regime]
        h = np.random.uniform(h_min, h_max)
        a = R_E + h

        if regime == "LEO":
            e = np.random.uniform(0.0, 0.02)
            while a * (1 - e) < (R_E + min_perigee_alt):
                e = np.random.uniform(0.0, 0.02)
            i_deg = np.random.uniform(0.0, 180.0)
        elif regime == "MEO":
            e = np.random.uniform(0.0, 0.10)
            while a * (1 - e) < (R_E + min_perigee_alt):
                e = np.random.uniform(0.0, 0.10)
            i_deg = np.random.uniform(0.0, 120.0)
        elif regime == "GEO":
            e = np.random.uniform(0.0, 0.0015)
            i_deg = np.random.uniform(0.0, 15.0)
            if a * (1 - e) < (R_E + min_perigee_alt):
                e = 0.0
        else:
            raise ValueError(f"Unknown orbit regime '{regime}'")

        oe.a = a
        oe.e = e
        oe.i = i_deg * D2R
        oe.Omega = np.random.uniform(0.0, 360.0) * D2R
        oe.omega = np.random.uniform(0.0, 360.0) * D2R
        oe.f = np.random.uniform(0.0, 360.0) * D2R
        return oe

    def custom_oe_randomizer(regime: str = "LEO",
                             mix_weights: dict[str, float] | None = None,
                             altitude_bounds: dict[str, tuple[float, float]] | None = None,
                             min_perigee_alt: float = 400e3) -> orbitalMotion.ClassicElements:
        """
        Backward-compatible zero-arg sampler of ClassicElements.
        - Called with no args -> defaults to LEO (old scripts keep working).
        - For MEO/GEO or mixed, wrap with a zero-arg lambda/partial that sets 'regime'.

        Example:
            oe = custom_oe_randomizer()                       # LEO (legacy behavior)
            oe = custom_oe_randomizer(regime="MEO")           # use via lambda/partial at call site
        """
        if altitude_bounds is None:
            altitude_bounds = DEFAULT_ALT_BOUNDS

        # Mixed regime selection if requested (called via wrapper that forwards kwargs)
        if regime.lower() == "mixed":
            if mix_weights is None:
                regimes, probs = ["LEO", "MEO", "GEO"], np.array([0.6, 0.3, 0.1])
            else:
                regimes = ["LEO", "MEO", "GEO"]
                probs = np.array([mix_weights.get(r, 0.0) for r in regimes], dtype=float)
                if probs.sum() <= 0:
                    raise ValueError("mix_weights must include positive weights.")
                probs = probs / probs.sum()
            regime = np.random.choice(regimes, p=probs)

        return _sample_for_regime(regime.upper(), altitude_bounds, min_perigee_alt)


    # For this first AMOS training pass, keep the science question LEO-to-LEO.
    # Later LEO-to-any can switch oe back to the mixed-regime partial below.
    # Keep target satellites passive/alive in this baseline entrypoint. The
    # scanner is the controlled spacecraft; targets only define opportunities.
    target_args = dict(
        oe=custom_oe_randomizer,
        batteryStorageCapacity=1.0,
        storedCharge_Init=1.0,
        basePowerDraw=0.0,
    )
    # target_args_mixed = dict(oe=partial(custom_oe_randomizer, regime="mixed", mix_weights={"LEO":0.5,"MEO":0.3,"GEO":0.2}), batteryStorageCapacity = 1, storedCharge_Init = 0.0, basePowerDraw = -10000.0 )



    # Make the satellite
    sat = MyScanningSatellite(name="SS1", sat_args=sat_args) # SO1 for satellite observer 1

    targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)]

    all_sat = [sat] + targets

    N = 0 # int(sys.argv[1])  # Passed by sweep.sh script
    default_n_envs = max(1, get_available_cores() - 4)  # leave some extra cores for other processes
    n_envs = max(1, _env_int("BSK_RL_NUM_ENVS", default_n_envs))
    batch_multiplier = _env_int("BSK_RL_BATCH_MULTIPLIER", 150)
    batch_size = max(PPO_MIN_TRAIN_BATCH_SIZE, int(batch_multiplier * n_envs))
    total_timesteps = _env_int("BSK_RL_TOTAL_TIMESTEPS", 20_000_000)
    checkpoint_frequency = _env_int("BSK_RL_CHECKPOINT_FREQUENCY", 3)
    run_tag = (
        f"amos2026_LEO_wGAE_BigNetwork_ImagingOnly_00d100i_{batch_size}batch_"
        "obs-v9_captureReward_1e-5lr_0.05cp_gradclip0.5_gamma9997"
    )
    model_name = f"{run_tag}.out_{N}"
    print(f"n_envs={n_envs}; batch_size={batch_size}; torch_threads={_TORCH_THREADS}")

    if os.environ.get("SLURM_JOB_ID"):
        scratch_root = Path(
            os.environ.get("BSK_RL_SCRATCH", f"/scratch/alpine/{os.environ.get('USER', 'dahu1128')}")
        ).expanduser()
        output_root = Path(
            os.environ.get("BSK_RL_OUTPUT_DIR", scratch_root / "rllib_results")
        ).expanduser()
        ray_tmpdir = Path(
            os.environ.get(
                "BSK_RL_RAY_TMPDIR",
                os.environ.get(
                    "TMPDIR",
                    f"/tmp/bskray_{os.environ.get('SLURM_JOB_ID')}_{N}",
                ),
            )
        ).expanduser()
    else:
        output_root = Path(
            os.environ.get(
                "BSK_RL_OUTPUT_DIR",
                "~/rllib_results/may_results/may15rllib_results",
            )
        ).expanduser()
        ray_tmpdir = Path(os.environ.get("BSK_RL_RAY_TMPDIR", f"/tmp/bskrl_{N}"))

    output_dir = output_root / f"may15_BigNetwork_ImagingOnly_00d100i_{run_tag}_{time.time()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ray_tmpdir.mkdir(parents=True, exist_ok=True)

    print(f"Tensorboard logging: tensorboard --logdir {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    jobs = build_job_array(
        training_args=dict(
            lr=[1e-5],
            gamma=[0.9997],
            train_batch_size=[batch_size],  # keep at least RLlib's default minibatch size
            num_sgd_iter=[10],
            lambda_=[0.95],
            use_kl_loss=[False],
            clip_param=[0.05],
            grad_clip=[1.0/2],
            entropy_coeff=[0.0],
        ),
        env_args=dict(
            # **{k: [v] for k, v in env_args().items()},
            satellites=[all_sat],
            scenario=[make_rso_scenario()],
            rewarder=[make_rso_rewarder()],
            world_type=[world.GroundStationWorldModel],
            time_limit=[total_time],
            failure_penalty=[-100.0],
            terminate_on_time_limit=[False],
            generate_obs_retasking_only=[False],  # For last step
            episode_data_callback=[env_metrics_callback],
            satellite_data_callback=[sat_metrics_callback],
        ),
    )

    print(f"Running job {N}: {N+1} of {len(jobs)}")
    job_args = jobs[N]

    # with open(output_dir / f"{model_name}_params_aug19.txt", "w") as file: # update this when running on cluster
    # yaml.dump(sanitize_np(job_args), file)



    # Save exactly what was used for this run (sim + training/env args)
    run_cfg = {
        "sim": asdict(sim_cfg),
        "model_family": "big_network_fc_imaging_only",
        "observation_layout": {
            "obs_v": sim_cfg.obs_v,
            "target_only": True,
            "target_features_per_target": 11,
            "target_chunk_order": [
                "priority",
                "target_elevation_angle",
                "rel_pos_vector_r_BR_H[0:3]",
                "rel_vel_vector_v_BR_H[0:3]",
                "angle_to_target",
                "target_distance",
                "target_shadowFactor",
            ],
        },
        "observation_norms": {
            "priority": PRIORITY_NORM,
            "relative_velocity_mps": REL_VEL_NORM_MPS,
        },
        "battery_life_multiplier": battery_life_multiplier,
        "image_storage_capacity_images": image_storage_capacity,
        "n_envs": n_envs,
        "batch_multiplier": batch_multiplier,
        "batch_size": batch_size,
        "total_timesteps": total_timesteps,
        "ray_tmpdir": str(ray_tmpdir),
        "torch_threads": _TORCH_THREADS,
        "wandb": {
            "enabled": _env_bool("BSK_RL_USE_WANDB", True),
            "key_path": str(_wandb_key_path()),
            "project": os.environ.get("BSK_RL_WANDB_PROJECT", "amos2026-bsk-rl"),
            "group": os.environ.get(
                "BSK_RL_WANDB_GROUP",
                "polaris-big-network-imaging-only-obs-v9-capture-reward",
            ),
        },
        "job_args": sanitize_np(job_args),
    }

    with open(output_dir / f"{model_name}_config.yaml", "w") as file:
        yaml.dump(run_cfg, file)

    wandb_logger = _maybe_init_wandb(model_name, run_cfg)

    train_model(
        model_name=model_name,
        output_directory=output_dir,
        checkpoint_frequency=checkpoint_frequency, # used to be 2
        checkpoints_to_keep=3,
        total_timesteps=total_timesteps,
        reload_frequency=500_000,
        n_envs=n_envs,
        temp_dir=str(ray_tmpdir),
        wandb_logger=wandb_logger,
        **job_args,
    )


    # train_model(
    #     model_name=model_name,
    #     output_directory=output_dir,
    #     checkpoint_frequency=3, # used to be 2
    #     checkpoints_to_keep=3,
    #     total_timesteps=20_000_000,
    #     reload_frequency=500_000,
    #     n_envs=n_envs,
    #     # temp_dir="/scratch/alpine/dahu1128/tmp", # uncomment this when running on cluster
    #     **job_args,
    # )
