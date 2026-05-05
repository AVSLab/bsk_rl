#!/usr/bin/env python3
"""AMOS 2026 LEO-to-LEO imaging-only trainer with target-wise GNN.

Local quick syntax:
    BSK_RL_NUM_ENVS=1 BSK_RL_BATCH_MULTIPLIER=32 BSK_RL_TOTAL_TIMESTEPS=10000 \
        python3 examples/amos_2026/train_leo_reimaging_target_gnn.py

Cluster syntax:
    BSK_RL_TOTAL_TIMESTEPS=20000000 BSK_RL_BATCH_MULTIPLIER=150 \
        sbatch examples/amos_2026/sbatch_train_leo_reimaging_target_gnn_debug.sh

This script intentionally removes charge, downlink, and desat actions so the
custom RLModule can score only the target choices. Because there is no downlink
action in this ablation, image quality reward is paid at capture time here only;
the normal AMOS pending/downlink-verification lifecycle remains in the baseline
`examples/updated_train_Polaris.py` path.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# This file lives one directory below the legacy examples scripts. Keep imports
# such as `from sim_config import SimConfig` working from both repo-root and Slurm.
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
import torch
import yaml
from sim_config import SimConfig

_TORCH_THREADS = int(os.environ.get("BSK_RL_TORCH_THREADS", "11"))
torch.set_num_threads(_TORCH_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))

import ray
from Basilisk.utilities import macros, orbitalMotion
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import TimeDiscountedGAEPPOTorchLearner
from bsk_rl.utils.rllib.target_gnn_module import GNNModule
from bsk_rl.utils.utils import build_job_array, get_available_cores, sanitize_np
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger

try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
except (ImportError, ModuleNotFoundError):  # Older versions of RLlib
    from ray.rllib.core.rl_module.marl_module import (
        MultiAgentRLModuleSpec as MultiRLModuleSpec,
    )
    from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec as RLModuleSpec


# Observation bookkeeping for Target-GNN's module. The observation spec below is
# ordered so the first OBS_SAT_DIM entries are spacecraft/global features and all
# remaining entries are repeated TARGET_FEATURES_PER_TARGET chunks.
OBS_SAT_DIM = 7  # storage, battery, 3 wheel speeds, next eclipse start/end
TARGET_FEATURES_PER_TARGET = 7  # elevation, rel-pos-H(3), angle, distance, shadow
NON_IMAGING_ACTIONS = 0  # image-only action space: one logit per candidate target


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    return int(default if raw_value is None else raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    return float(default if raw_value is None else raw_value)


def _cluster_scratch_root() -> Path:
    user = os.environ.get("USER", "dahu1128")
    scratch_root = Path(
        os.environ.get("BSK_RL_SCRATCH", f"/scratch/alpine/{user}")
    ).expanduser()
    if scratch_root.exists() or os.environ.get("SLURM_JOB_ID"):
        return scratch_root
    return Path("~/rllib_results").expanduser()


def _default_output_root() -> Path:
    explicit = os.environ.get("BSK_RL_OUTPUT_DIR")
    if explicit is not None:
        return Path(explicit).expanduser()
    scratch_root = _cluster_scratch_root()
    if str(scratch_root).endswith("rllib_results"):
        return scratch_root / "amos2026_target_gnn_results"
    return scratch_root / "rllib_results"


def _default_ray_tmpdir() -> Path:
    explicit = os.environ.get("BSK_RL_RAY_TMPDIR") or os.environ.get("TMPDIR")
    if explicit is not None:
        return Path(explicit).expanduser()
    scratch_root = _cluster_scratch_root()
    if str(scratch_root).endswith("rllib_results"):
        return Path("/tmp")
    return scratch_root / "tmp"


def target_gnn_model_config(n_targets_ahead: int) -> dict[str, Any]:
    """Return the target-wise GNN settings suggested for the imaging ablation."""
    return {
        "n_targets": int(n_targets_ahead),
        "obs_sat": OBS_SAT_DIM,
        "width_f": 256,
        "depth_f": 2,
        "block_f": False,
        "width_g": 128,
        "depth_g": 4,
        "tgt_encoded_dim": 128,
        "attention_depth": 1,
        "num_heads": 2,
        "attention_dim": 128,
        "dropout": 0.0,
        "critic_tgt_encoded_dim": 128,
        "critic_width_f": 256,
        "critic_depth_f": 2,
        "critic_block_f": False,
        "critic_width_g": 64,
        "critic_depth_g": 3,
        "critic_pooling_std": False,
        "non_imaging_actions": NON_IMAGING_ACTIONS,
    }


def train_model(
    model_name: str,
    output_directory: Path,
    inspector_rl_module_spec: RLModuleSpec,
    env_args: dict[str, Any],
    training_args: dict[str, Any],
    n_envs: int = 1,
    checkpoint_frequency: int = 1,
    checkpoints_to_keep: int = 2,
    reload_frequency: int = 500_000,
    total_timesteps: int = 1_000_000,
    temp_dir: str = "/tmp",
) -> None:
    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir
    output_directory = Path(output_directory)
    run_directory = output_directory / model_name
    run_directory.mkdir(exist_ok=True, parents=True)

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if "target" in agent_id:
            return "rso"
        return "inspector"

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=_env_int("BSK_RL_OBJECT_STORE_MEMORY", 2_000_000_000),
        _temp_dir=temp_dir,
    )

    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=n_envs,
            sample_timeout_s=50000.0,
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
                    "inspector": inspector_rl_module_spec,
                    # Target spacecraft only drift. Keep their policy tiny so the
                    # training signal and parameter count are dominated by SS1.
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
        learner_connector=lambda obs_space, act_space: (),
        learner_class=TimeDiscountedGAEPPOTorchLearner,
        learner_config_dict=dict(reward_time="step_start"),
    )
    config.logger_config = dict(type=UnifiedLogger, logdir=run_directory)

    ppo = PPO(config)

    iteration = 0
    step = 0
    current_best_return = -np.inf

    while True:
        prev_step = step
        results = ppo.train()
        step = results["num_env_steps_sampled_lifetime"]
        step_return = results["env_runners"].get("episode_return_mean", -np.inf)

        if step_return > current_best_return:
            checkpoint_path = run_directory / "checkpoint_best"
            try:
                shutil.rmtree(checkpoint_path)
            except FileNotFoundError:
                pass
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)
            with open(checkpoint_path / f"iteration_{str(iteration).zfill(6)}.txt", "w") as file:
                file.write(f"iter: {iteration}\n")
            current_best_return = step_return

        checkpoint_path = run_directory / f"checkpoint_{str(iteration).zfill(6)}"
        if iteration % checkpoint_frequency == 0:
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
                object_store_memory=_env_int("BSK_RL_OBJECT_STORE_MEMORY", 3_000_000_000),
                _temp_dir=temp_dir,
            )
            ppo = PPO.from_checkpoint(checkpoint_path)

        if iteration > checkpoints_to_keep * checkpoint_frequency - 1:
            for i in range(checkpoint_frequency):
                remove_dir = (
                    run_directory
                    / f"checkpoint_{str(iteration - checkpoints_to_keep * checkpoint_frequency - i).zfill(6)}"
                )
                try:
                    shutil.rmtree(remove_dir)
                except FileNotFoundError:
                    pass

        iteration += 1


def _finite_values(values):
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


def _safe_std(values, default=-1.0):
    values = _finite_values(values)
    return float(np.std(values)) if values else default


def _safe_max(values, default=-1.0):
    values = _finite_values(values)
    return float(np.max(values)) if values else default


def env_metrics_callback(env):
    """Episode-level metrics useful for the imaging-only GNN experiment."""
    data = {}
    reward_data = env.rewarder.data
    episode_duration = float(env.simulator.sim_time)
    ss1_actions = env.satellites[0].action_builder.action_spec[0]

    num_imaged = len(getattr(reward_data, "imaged", []))
    data["num_unique_targets_imaged"] = num_imaged
    data["episode_duration_sec"] = episode_duration
    data["number of alive cases"] = env.satellites[0].dynamics.is_alive()
    data["battery_valid"] = env.satellites[0].dynamics.battery_valid()
    data["rw_valid"] = env.satellites[0].dynamics.rw_speeds_valid()
    data["cumulativeRewardSS1"] = env.rewarder.cum_reward["SS1"]
    data["illuminated_images"] = len(getattr(env.rewarder, "imaged_illuminated", []))

    target_priority_by_id = {}
    try:
        target_priority_by_id = {
            int(target.id): float(target.priority)
            for target in env.scenario.target_spacecrafts
        }
    except Exception:
        target_priority_by_id = {}

    all_priorities = list(target_priority_by_id.values())
    data["target_priority_sum"] = float(np.sum(all_priorities)) if all_priorities else -1.0
    data["target_priority_mean"] = _safe_mean(all_priorities)
    data["target_priority_max"] = _safe_max(all_priorities)

    imaging_attempt_records = list(getattr(ss1_actions, "imaging_attempt_records", []))
    imaging_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if record.get("start_time") is not None and record.get("end_time") is not None
    ]
    successful_durations = [
        float(record["end_time"]) - float(record["start_time"])
        for record in imaging_attempt_records
        if record.get("success")
        and record.get("start_time") is not None
        and record.get("end_time") is not None
    ]
    attempted_priorities = [
        target_priority_by_id.get(int(record["target_id"]))
        for record in imaging_attempt_records
        if record.get("target_id") is not None
    ]
    successful_priorities = [
        target_priority_by_id.get(int(record["target_id"]))
        for record in imaging_attempt_records
        if record.get("success") and record.get("target_id") is not None
    ]

    data["num_imaging_attempts"] = len(imaging_attempt_records)
    data["actual_imaging_action_time_sec"] = float(np.sum(imaging_durations))
    data["actual_non_imaging_time_sec"] = episode_duration - data["actual_imaging_action_time_sec"]
    data["mean_imaging_action_duration_sec"] = _safe_mean(imaging_durations)
    data["mean_successful_imaging_action_duration_sec"] = _safe_mean(successful_durations)
    data["mean_imaging_slew_time_sec"] = _safe_mean(
        record.get("slew_time_s") for record in imaging_attempt_records
    )
    data["mean_attempted_target_priority"] = _safe_mean(attempted_priorities)
    data["mean_successful_capture_priority"] = _safe_mean(successful_priorities)
    data["reimage_count"] = int(getattr(env.rewarder, "reimage_count", 0))
    data["cooldown_target_count"] = len(getattr(reward_data, "cooldown_until_by_id", {}))

    if getattr(ss1_actions, "chosen_target_priority", None):
        data["mean_target_priority"] = float(np.mean(ss1_actions.chosen_target_priority))
        data["std_target_priority"] = float(np.std(ss1_actions.chosen_target_priority))
        data["max_target_priority"] = float(np.max(ss1_actions.chosen_target_priority))
        data["mean_chosen_target_priority"] = data["mean_target_priority"]
    else:
        data["mean_target_priority"] = -1.0
        data["std_target_priority"] = -1.0
        data["max_target_priority"] = -1.0
        data["mean_chosen_target_priority"] = -1.0

    if getattr(ss1_actions, "chosen_target_illumination_status", None):
        illumination = ss1_actions.chosen_target_illumination_status
        data["mean_target_illumination_status"] = float(np.mean(illumination))
        data["num_target_above_illumination_threshold"] = sum(ill > 0.5 for ill in illumination)
        data["num_target_below_illumination_threshold"] = sum(ill <= 0.5 for ill in illumination)
    else:
        data["mean_target_illumination_status"] = -1.0

    if getattr(ss1_actions, "ever_visible", None):
        data["target_ever_visible_fraction"] = len(ss1_actions.ever_visible) / n_targets
    else:
        data["target_ever_visible_fraction"] = -1.0

    # These stay at zero in this imaging-only ablation, which makes TensorBoard
    # comparisons against the downlink-verification baseline easier to read.
    data["total_downlinks"] = int(getattr(env.rewarder, "total_downlinks", 0))
    data["useful_downlinks"] = int(getattr(env.rewarder, "useful_downlinks", 0))
    data["failed_downlinks"] = int(getattr(env.rewarder, "failed_downlinks", 0))
    data["pending_images"] = int(getattr(env.rewarder, "pending_images", 0))

    return data


def sat_metrics_callback(env, satellite):
    data = {}
    if satellite.name == "SS1":
        data["RW_norm"] = np.linalg.norm(satellite.dynamics.wheel_speeds)
        data["RW1"] = satellite.dynamics.wheel_speeds[0]
        data["RW2"] = satellite.dynamics.wheel_speeds[1]
        data["RW3"] = satellite.dynamics.wheel_speeds[2]
        data["battery_charge_fraction"] = satellite.dynamics.battery_charge_fraction
        data["storage_level_fraction"] = satellite.dynamics.storage_level_fraction
    else:
        data["RW_norm"] = 0.0
        data["RW1"] = 0.0
        data["RW2"] = 0.0
        data["RW3"] = 0.0
        data["battery_charge_fraction"] = 0.0
        data["storage_level_fraction"] = 0.0
    return data


if __name__ == "__main__":
    sim_cfg = SimConfig(
        n_targets=_env_int("BSK_RL_N_TARGETS", 100),
        n_targets_ahead=_env_int("BSK_RL_N_TARGETS_AHEAD", 10),
        imaging_duration=_env_float("BSK_RL_IMAGING_DURATION", 300.0),
        extra_time_factor=_env_float("BSK_RL_EXTRA_TIME_FACTOR", 1.5),
        obs_v=7.0,
        just_imaging=True,
        verify_image_quality_on_downlink=False,
        hide_pending_targets=False,
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
        )

    def make_rso_rewarder():
        return data.RSOTargetImageReward(
            reimage_cooldown_orbits=sim_cfg.reimage_cooldown_orbits,
            verify_image_quality_on_downlink=False,
            hide_pending_targets=False,
            image_quality_threshold=sim_cfg.image_quality_threshold,
        )

    class MyScanningSatellite(sats.AccessSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
            ),
            obs.Eclipse(norm=5700),
            obs.PolarisScTargetProperties(
                dict(prop="target_elevation_angle", norm=90.0),
                dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                dict(prop="angle_to_target", norm=90.0),
                dict(prop="target_distance", norm=15960 * 1000),
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
            ),
        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    sat_args = {}
    sat_args["imageAttErrorRequirement"] = 0.0025
    sat_args["imageRateErrorRequirement"] = 0.01

    # Imaging-only ablation: no downlink exists, so make storage and battery large
    # enough that resource depletion is not the dominant learning problem.
    baseline_storage_bits = 50 * 8e6 / 2
    baseline_battery_ws = 500 * 3600
    sat_args["dataStorageCapacity"] = 10 * baseline_storage_bits
    sat_args["storageInit"] = lambda: 0.0
    sat_args["instrumentBaudRate"] = 0.5 * 8e6
    sat_args["transmitterBaudRate"] = -0.5 * 8e6

    sat_args["batteryStorageCapacity"] = 10 * baseline_battery_ws
    sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.8, 1.0) * 10 * baseline_battery_ws
    sat_args["basePowerDraw"] = -10.0
    sat_args["instrumentPowerDraw"] = -30.0
    sat_args["transmitterPowerDraw"] = -25.0
    sat_args["thrusterPowerDraw"] = -80.0
    sat_args["panelArea"] = 1.0

    sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.000, size=3)
    sat_args["maxWheelSpeed"] = 6000.0
    sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
    sat_args["desatAttitude"] = "sun"

    sat_args["downlink_bonus"] = 0.0
    sat_args["imaging_bonus"] = 1.0
    sat_args["eclipse_threshold_for_imaging"] = 0.5
    sat_args["eclipse_threshold_for_reward"] = 0.5

    class MyTargetSatellite(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=total_time)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

    R_E = 6371e3
    D2R = macros.D2R
    DEFAULT_ALT_BOUNDS = {
        "LEO": (400e3, 2000e3),
        "MEO": (2000e3, 35000e3),
        "GEO": (35786e3 - 300e3, 35786e3 + 300e3),
    }

    def _sample_for_regime(
        regime: str,
        altitude_bounds: dict[str, tuple[float, float]],
        min_perigee_alt: float,
    ) -> orbitalMotion.ClassicElements:
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

    def custom_oe_randomizer(
        regime: str = "LEO",
        mix_weights: dict[str, float] | None = None,
        altitude_bounds: dict[str, tuple[float, float]] | None = None,
        min_perigee_alt: float = 400e3,
    ) -> orbitalMotion.ClassicElements:
        if altitude_bounds is None:
            altitude_bounds = DEFAULT_ALT_BOUNDS
        if regime.lower() == "mixed":
            regimes = ["LEO", "MEO", "GEO"]
            if mix_weights is None:
                probs = np.array([0.6, 0.3, 0.1])
            else:
                probs = np.array([mix_weights.get(r, 0.0) for r in regimes], dtype=float)
                if probs.sum() <= 0:
                    raise ValueError("mix_weights must include positive weights.")
                probs = probs / probs.sum()
            regime = np.random.choice(regimes, p=probs)
        return _sample_for_regime(regime.upper(), altitude_bounds, min_perigee_alt)

    target_args = dict(
        oe=custom_oe_randomizer,
        batteryStorageCapacity=1.0,
        storedCharge_Init=1.0,
        basePowerDraw=0.0,
    )

    sat = MyScanningSatellite(name="SS1", sat_args=sat_args)
    targets = [
        MyTargetSatellite(name=f"target_{i}", sat_args=target_args)
        for i in range(n_targets)
    ]
    all_sat = [sat] + targets

    default_job_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    job_index = _env_int("SLURM_ARRAY_TASK_ID", default_job_index)
    n_envs = max(1, _env_int("BSK_RL_NUM_ENVS", get_available_cores() - 4))
    batch_multiplier = _env_int("BSK_RL_BATCH_MULTIPLIER", 150)
    batch_size = int(batch_multiplier * n_envs)
    total_timesteps = _env_int("BSK_RL_TOTAL_TIMESTEPS", 20_000_000)
    checkpoint_frequency = _env_int("BSK_RL_CHECKPOINT_FREQUENCY", 3)

    run_tag = (
        f"amos2026_LEO_targetGNN_imagingOnly_{batch_size}batch_"
        "hold10s_reimage2orb_prioritySum100"
    )
    model_name = f"{run_tag}.out_{job_index}"
    output_dir = _default_output_root() / f"{run_tag}_{time.time()}"
    ray_tmpdir = _default_ray_tmpdir()
    output_dir.mkdir(parents=True, exist_ok=True)
    ray_tmpdir.mkdir(parents=True, exist_ok=True)

    inspector_model_config = target_gnn_model_config(n_targets_ahead)
    inspector_rl_module_spec = RLModuleSpec(
        module_class=GNNModule,
        model_config_dict=inspector_model_config,
    )

    base_lr = 0.00033003435881682255
    jobs = build_job_array(
        training_args=dict(
            lr=[[[0, base_lr], [40000, base_lr / 16.749479444886223]]],
            gamma=[0.9915045428565076],
            train_batch_size=[batch_size],
            num_sgd_iter=[30],
            lambda_=[0.8713548569911232],
            use_kl_loss=[False],
            clip_param=[0.14701727973480344],
            grad_clip=[0.3104924935285628],
            entropy_coeff=[0.023694512589767867],
        ),
        env_args=dict(
            satellites=[all_sat],
            scenario=[make_rso_scenario()],
            rewarder=[make_rso_rewarder()],
            world_type=[world.BasicWorldModel],
            time_limit=[total_time],
            failure_penalty=[-100.0],
            terminate_on_time_limit=[False],
            generate_obs_retasking_only=[False],
            episode_data_callback=[env_metrics_callback],
            satellite_data_callback=[sat_metrics_callback],
        ),
    )

    print(f"n_envs={n_envs}; batch_size={batch_size}; torch_threads={_TORCH_THREADS}")
    print(f"TensorBoard: tensorboard --logdir {output_dir}")
    print(f"Ray temp dir: {ray_tmpdir}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Running job {job_index}: {job_index + 1} of {len(jobs)}")

    job_args = jobs[job_index]
    run_cfg = {
        "sim": asdict(sim_cfg),
        "observation_layout": {
            "obs_sat_dim": OBS_SAT_DIM,
            "target_features_per_target": TARGET_FEATURES_PER_TARGET,
            "n_targets_ahead": n_targets_ahead,
            "non_imaging_actions": NON_IMAGING_ACTIONS,
            "target_chunk_order": [
                "target_elevation_angle",
                "rel_pos_vector_r_BR_H[0:3]",
                "angle_to_target",
                "target_distance",
                "target_shadowFactor",
            ],
        },
        "target_gnn_model_config": inspector_model_config,
        "job_args": sanitize_np(job_args),
        "cluster": {
            "job_index": job_index,
            "n_envs": n_envs,
            "batch_multiplier": batch_multiplier,
            "batch_size": batch_size,
            "total_timesteps": total_timesteps,
            "ray_tmpdir": str(ray_tmpdir),
            "torch_threads": _TORCH_THREADS,
        },
    }
    with open(output_dir / f"{model_name}_config.yaml", "w") as file:
        yaml.dump(run_cfg, file)

    train_model(
        model_name=model_name,
        output_directory=output_dir,
        inspector_rl_module_spec=inspector_rl_module_spec,
        checkpoint_frequency=checkpoint_frequency,
        checkpoints_to_keep=3,
        total_timesteps=total_timesteps,
        reload_frequency=500_000,
        n_envs=n_envs,
        temp_dir=str(ray_tmpdir),
        **job_args,
    )
