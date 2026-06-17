#!/usr/bin/env python3
"""Local old-network GNC/Breckenridge training for LEO-to-any targets.

This script keeps the October 2025 GNC policy setup and changes only the target
catalog from LEO-only to the mixed LEO/MEO/GEO distribution used in the GNC
paper Monte Carlo study.
"""

import argparse
import os
import shutil
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import ray
import torch
import yaml
from Basilisk.utilities import macros, orbitalMotion
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from sim_config import SimConfig

from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import TimeDiscountedGAEPPOTorchLearner
from bsk_rl.utils.utils import build_job_array, get_available_cores, sanitize_np

try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
except (ImportError, ModuleNotFoundError):  # Older RLlib
    from ray.rllib.core.rl_module.marl_module import (
        MultiAgentRLModuleSpec as MultiRLModuleSpec,
    )
    from ray.rllib.core.rl_module.rl_module import (
        SingleAgentRLModuleSpec as RLModuleSpec,
    )


N_TARGETS_FOR_CALLBACK = 100


def parse_mix_weights(value: str) -> dict[str, float]:
    weights = {}
    for item in value.split(","):
        key, raw_weight = item.split("=")
        weights[key.strip().upper()] = float(raw_weight)
    total = sum(weights.get(regime, 0.0) for regime in ("LEO", "MEO", "GEO"))
    if total <= 0.0:
        raise ValueError("--mix-weights must include positive LEO/MEO/GEO weights.")
    return weights


def reward_label(downlink_bonus: float) -> str:
    downlink_percent = int(round(downlink_bonus * 100.0))
    imaging_percent = 100 - downlink_percent
    return f"{downlink_percent}d{imaging_percent:02d}i"


def mix_label(weights: dict[str, float]) -> str:
    total = sum(weights.get(regime, 0.0) for regime in ("LEO", "MEO", "GEO"))
    normalized = {
        regime: 100.0 * weights.get(regime, 0.0) / total
        for regime in ("LEO", "MEO", "GEO")
    }
    return "L{LEO:02.0f}M{MEO:02.0f}G{GEO:02.0f}".format(**normalized)


def configure_threads(torch_threads: int) -> None:
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)


def train_model(
    model_name,
    output_directory,
    env_args=None,
    n_envs=1,
    checkpoint_frequency=5,
    checkpoints_to_keep=3,
    reload_frequency=500_000,
    total_timesteps=20_000_000,
    training_args=None,
    temp_dir="/tmp/bsk_rl_ray",
):
    env_args = env_args or {}
    training_args = training_args or {}
    temp_dir = Path(temp_dir).expanduser()
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = str(temp_dir)

    output_directory = Path(output_directory).expanduser()
    output_directory.mkdir(exist_ok=True, parents=True)

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if "target" in agent_id:
            return "rso"
        return "inspector"

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=2_000_000_000,
        _temp_dir=str(temp_dir),
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
        learner_connector=lambda obs_space, act_space: (),
        learner_class=TimeDiscountedGAEPPOTorchLearner,
        learner_config_dict=dict(reward_time="step_start"),
    )
    config.logger_config = dict(
        type=UnifiedLogger, logdir=output_directory / model_name
    )

    ppo = PPO(config)
    iteration = 0
    step = 0
    current_best_return = -np.inf

    while True:
        prev_step = step
        results = ppo.train()
        step = results["num_env_steps_sampled_lifetime"]
        step_return = results["env_runners"].get("episode_return_mean", -np.inf)
        print(
            f"iter={iteration} sampled_steps={step} "
            f"episode_return_mean={step_return}"
        )

        if step_return > current_best_return:
            checkpoint_path = output_directory / model_name / "checkpoint_best"
            try:
                shutil.rmtree(checkpoint_path)
            except FileNotFoundError:
                pass
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)
            with open(checkpoint_path / f"iteration_{iteration:06d}.txt", "w") as file:
                file.write(f"iter: {iteration}\n")
            current_best_return = step_return

        checkpoint_path = output_directory / model_name / f"checkpoint_{iteration:06d}"
        if iteration % checkpoint_frequency == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)

        if step >= total_timesteps:
            break

        if step % reload_frequency < prev_step % reload_frequency:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)
            ray.shutdown()
            ray.init(
                ignore_reinit_error=True,
                num_cpus=get_available_cores(),
                object_store_memory=2_000_000_000,
                _temp_dir=str(temp_dir),
            )
            ppo = PPO.from_checkpoint(checkpoint_path)

        if iteration > checkpoints_to_keep * checkpoint_frequency - 1:
            for i in range(checkpoint_frequency):
                remove_dir = (
                    output_directory
                    / model_name
                    / f"checkpoint_{iteration - checkpoints_to_keep * checkpoint_frequency - i:06d}"
                )
                try:
                    shutil.rmtree(remove_dir)
                except FileNotFoundError:
                    pass

        iteration += 1

    ray.shutdown()


def env_metrics_callback(env):
    data = {}

    num_imaged = len(env.rewarder.data.imaged)
    data["num_unique_targets_imaged"] = num_imaged
    episode_duration = env.simulator.sim_time
    data["episode_duration_sec"] = episode_duration
    data["number of alive cases"] = env.satellites[0].dynamics.is_alive()
    data["battery_valid"] = env.satellites[0].dynamics.battery_valid()
    data["rw_valid"] = env.satellites[0].dynamics.rw_speeds_valid()
    non_imaging_time = episode_duration - num_imaged * 300
    data["non-imaging_action_count"] = int(round(non_imaging_time / 300))
    data["non-imaging_time"] = int(round(non_imaging_time))
    data["cumulativeRewardSS1"] = env.rewarder.cum_reward["SS1"]
    data["illuminated_images"] = len(env.rewarder.imaged_illuminated)

    image_action = env.satellites[0].action_builder.action_spec[0]
    if getattr(image_action, "chosen_target_elevation", None):
        data["mean_target_elevation"] = np.mean(image_action.chosen_target_elevation)
        data["std_target_elevation"] = np.std(image_action.chosen_target_elevation)
    else:
        data["mean_target_elevation"] = -1
        data["std_target_elevation"] = -1

    if getattr(image_action, "chosen_target_rel_pos_H", None):
        mean_rel_pos = np.mean(image_action.chosen_target_rel_pos_H, axis=0)
        std_rel_pos = np.std(image_action.chosen_target_rel_pos_H, axis=0)
        for i, axis in enumerate(["x", "y", "z"]):
            data[f"mean_rel_pos_H_{axis}"] = mean_rel_pos[i]
            data[f"std_rel_pos_H_{axis}"] = std_rel_pos[i]
    else:
        for axis in ["x", "y", "z"]:
            data[f"mean_rel_pos_H_{axis}"] = -1
            data[f"std_rel_pos_H_{axis}"] = -1

    if getattr(image_action, "chosen_target_distance", None):
        data["mean_target_distance"] = np.mean(image_action.chosen_target_distance)
        data["std_target_distance"] = np.std(image_action.chosen_target_distance)
    else:
        data["mean_target_distance"] = -1
        data["std_target_distance"] = -1

    if getattr(image_action, "chosen_target_illumination_status", None):
        illumination = image_action.chosen_target_illumination_status
        data["mean_target_illumination_status"] = np.mean(illumination)
        data["num_target_above_illumination_threshold"] = sum(
            value > 0.5 for value in illumination
        )
        data["num_target_below_illumination_threshold"] = sum(
            value <= 0.5 for value in illumination
        )
    else:
        data["mean_target_illumination_status"] = -1

    if getattr(image_action, "ever_visible", None):
        data["target_ever_visible_fraction"] = (
            len(image_action.ever_visible) / N_TARGETS_FOR_CALLBACK
        )
    else:
        data["target_ever_visible_fraction"] = -1

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
        data["Total Images Downlinked"] = satellite.dynamics.total_downlinks
        data["Useful Images Downlinked"] = satellite.dynamics.useful_downlinks
    else:
        data["RW_norm"] = 0
        data["RW1"] = 0
        data["RW2"] = 0
        data["RW3"] = 0
        data["battery_charge_fraction"] = 0
        data["storage_level_fraction"] = 0
        data["Total Images Downlinked"] = 0
        data["Useful Images Downlinked"] = 0

    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the old GNC/Breckenridge PPO policy locally on mixed "
            "LEO/MEO/GEO targets."
        )
    )
    parser.add_argument("--downlink-bonus", type=float, default=0.1)
    parser.add_argument("--mix-weights", default="LEO=0.5,MEO=0.3,GEO=0.2")
    parser.add_argument("--train-batch-size", type=int, default=4992)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--leave-cores", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=5)
    parser.add_argument("--checkpoints-to-keep", type=int, default=3)
    parser.add_argument("--reload-frequency", type=int, default=500_000)
    parser.add_argument("--failure-penalty", type=float, default=-10.0)
    parser.add_argument("--low-battery-penalty", type=float, default=-0.5)
    parser.add_argument(
        "--output-root",
        default="~/rllib_results/breckenridge2026_leo_any_oldnet",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Ray temp directory. Defaults to ~/ray_tmp/breckenridge2026_<pid>.",
    )
    parser.add_argument(
        "--run-prefix",
        default="breckenridge2026_LEOAny_oldnet_local",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only one train-batch worth of steps to verify the local setup.",
    )
    return parser


def main() -> None:
    global N_TARGETS_FOR_CALLBACK

    args = build_parser().parse_args()
    if not 0.0 <= args.downlink_bonus <= 1.0:
        raise ValueError("--downlink-bonus must be between 0 and 1.")
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive.")

    configure_threads(args.torch_threads)

    mixed_weights = parse_mix_weights(args.mix_weights)
    reward_mix_label = reward_label(args.downlink_bonus)
    target_mix_label = mix_label(mixed_weights)

    sim_cfg = SimConfig(
        n_targets=100,
        n_targets_ahead=10,
        imaging_duration=300.0,
        extra_time_factor=1.5,
        obs_v=7.0,
        just_imaging=False,
    )
    n_targets = sim_cfg.n_targets
    n_targets_ahead = sim_cfg.n_targets_ahead
    imaging_duration = sim_cfg.imaging_duration
    total_time = sim_cfg.total_time
    N_TARGETS_FOR_CALLBACK = n_targets

    class MyScanningSatellite(sats.AccessSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
            ),
            obs.PolarisScTargetProperties(
                dict(prop="target_elevation_angle", norm=90.0),
                dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                dict(prop="angle_to_target", norm=90.0),
                dict(prop="target_distance", norm=15960 * 1000),
                dict(prop="target_shadowFactor", norm=1.0),
                n_ahead_observe=n_targets_ahead,
            ),
            obs.Eclipse(norm=5700),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700.0),
                dict(prop="opportunity_close", norm=5700.0),
                type="ground_station",
                n_ahead_observe=2,
            ),
        ]
        action_spec = [
            act.ImageRSO(n_ahead_image=n_targets_ahead, duration=imaging_duration),
            act.Charge(duration=300.0),
            act.Downlink(duration=300.0),
            act.Desat(duration=150.0),
        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    sat_args = {}
    sat_args["imageAttErrorRequirement"] = 0.01
    sat_args["dataStorageCapacity"] = 50 * 8e6 / 2
    sat_args["storageInit"] = lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2
    sat_args["instrumentBaudRate"] = 0.5 * 8e6
    sat_args["transmitterBaudRate"] = -0.5 * 8e6
    sat_args["batteryStorageCapacity"] = 500 * 3600
    sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.15, 0.5) * 500 * 3600
    sat_args["basePowerDraw"] = -10.0
    sat_args["instrumentPowerDraw"] = -30.0
    sat_args["transmitterPowerDraw"] = -25.0
    sat_args["thrusterPowerDraw"] = -80.0
    sat_args["panelArea"] = 1.0
    sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.000, size=3)
    sat_args["maxWheelSpeed"] = 6000.0
    sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
    sat_args["desatAttitude"] = "sun"
    sat_args["downlink_bonus"] = args.downlink_bonus
    sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
    sat_args["eclipse_threshold_for_imaging"] = 0.5
    sat_args["eclipse_threshold_for_reward"] = 0.5
    sat_args["low_battery_penalty"] = args.low_battery_penalty

    class MyTargetSatellite(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=total_time)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

    r_earth = 6371e3
    default_altitude_bounds = {
        "LEO": (400e3, 2000e3),
        "MEO": (2000e3, 35000e3),
        "GEO": (35786e3 - 300e3, 35786e3 + 300e3),
    }

    def sample_for_regime(regime, altitude_bounds, min_perigee_alt):
        oe = orbitalMotion.ClassicElements()
        h_min, h_max = altitude_bounds[regime]
        oe.a = r_earth + np.random.uniform(h_min, h_max)

        if regime == "LEO":
            oe.e = np.random.uniform(0.0, 0.02)
            while oe.a * (1 - oe.e) < r_earth + min_perigee_alt:
                oe.e = np.random.uniform(0.0, 0.02)
            inclination_deg = np.random.uniform(0.0, 180.0)
        elif regime == "MEO":
            oe.e = np.random.uniform(0.0, 0.10)
            while oe.a * (1 - oe.e) < r_earth + min_perigee_alt:
                oe.e = np.random.uniform(0.0, 0.10)
            inclination_deg = np.random.uniform(0.0, 120.0)
        elif regime == "GEO":
            oe.e = np.random.uniform(0.0, 0.0015)
            inclination_deg = np.random.uniform(0.0, 15.0)
            if oe.a * (1 - oe.e) < r_earth + min_perigee_alt:
                oe.e = 0.0
        else:
            raise ValueError(f"Unknown orbit regime '{regime}'")

        oe.i = inclination_deg * macros.D2R
        oe.Omega = np.random.uniform(0.0, 360.0) * macros.D2R
        oe.omega = np.random.uniform(0.0, 360.0) * macros.D2R
        oe.f = np.random.uniform(0.0, 360.0) * macros.D2R
        return oe

    def custom_oe_randomizer(
        regime="LEO",
        mix_weights=None,
        altitude_bounds=None,
        min_perigee_alt=400e3,
    ):
        if altitude_bounds is None:
            altitude_bounds = default_altitude_bounds
        if regime.lower() == "mixed":
            regimes = ["LEO", "MEO", "GEO"]
            if mix_weights is None:
                probs = np.array([0.5, 0.3, 0.2], dtype=float)
            else:
                probs = np.array([mix_weights.get(r, 0.0) for r in regimes], dtype=float)
                if probs.sum() <= 0:
                    raise ValueError("mix_weights must include positive weights.")
                probs = probs / probs.sum()
            regime = np.random.choice(regimes, p=probs)
        return sample_for_regime(regime.upper(), altitude_bounds, min_perigee_alt)

    target_args_mixed = dict(
        oe=partial(custom_oe_randomizer, regime="mixed", mix_weights=mixed_weights),
        batteryStorageCapacity=1,
        storedCharge_Init=0.0,
        basePowerDraw=-10000.0,
    )

    sat = MyScanningSatellite(name="SS1", sat_args=sat_args)
    targets = [
        MyTargetSatellite(name=f"target_{i}", sat_args=target_args_mixed)
        for i in range(n_targets)
    ]
    all_sat = [sat] + targets

    n_envs = args.n_envs
    if n_envs is None:
        n_envs = max(1, get_available_cores() - args.leave_cores)

    total_timesteps = args.total_timesteps
    if args.smoke_test:
        total_timesteps = args.train_batch_size

    temp_dir = args.temp_dir
    if temp_dir is None:
        temp_dir = f"~/ray_tmp/breckenridge2026_{os.getpid()}"

    model_name = (
        f"{args.run_prefix}_{args.train_batch_size}batch_mix{target_mix_label}_"
        f"restrictedResources_obsv7_1e-5lr_0.1cp_gradclip1.0_gamma9997_"
        f"{reward_mix_label}.out_0"
    )
    output_dir = (
        Path(args.output_root).expanduser()
        / (
            f"{args.run_prefix}_{args.train_batch_size}batch_mix{target_mix_label}_"
            f"restrictedResources_obsv7_1e-5lr_0.1cp_gradclip1.0_gamma9997_"
            f"{reward_mix_label}_{time.time()}"
        )
    )

    print(f"n_envs={n_envs}")
    print(f"train_batch_size={args.train_batch_size}")
    print(f"total_timesteps={total_timesteps}")
    print(f"Tensorboard logging: tensorboard --logdir {output_dir}")

    jobs = build_job_array(
        training_args=dict(
            lr=[1e-5],
            gamma=[0.9997],
            train_batch_size=[args.train_batch_size],
            num_sgd_iter=[10],
            lambda_=[0.95],
            use_kl_loss=[False],
            clip_param=[0.1],
            grad_clip=[1.0],
            entropy_coeff=[0.0],
        ),
        env_args=dict(
            satellites=[all_sat],
            scenario=[scene.RandomSatellites("SS1", n_targets=n_targets)],
            rewarder=[data.RSOTargetImageReward()],
            world_type=[world.GroundStationWorldModel],
            time_limit=[total_time],
            failure_penalty=[args.failure_penalty],
            terminate_on_time_limit=[False],
            generate_obs_retasking_only=[False],
            episode_data_callback=[env_metrics_callback],
            satellite_data_callback=[sat_metrics_callback],
        ),
    )
    job_args = jobs[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = {
        "sim": asdict(sim_cfg),
        "cli_args": vars(args),
        "selected_alpha": args.downlink_bonus,
        "target_mix_weights": mixed_weights,
        "notes": (
            "Old GNC/Breckenridge setup: fixed 300 s image/downlink, "
            "uniform target priorities from January IA_Polaris_SSA branch, "
            "old 2048x2048 inspector network, no fast actions, no HIO/SHIO."
        ),
        "job_args": sanitize_np(job_args),
    }
    with open(output_dir / f"{model_name}_run_config.yaml", "w") as file:
        yaml.dump(sanitize_np(run_cfg), file)

    train_model(
        model_name=model_name,
        output_directory=output_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        checkpoints_to_keep=args.checkpoints_to_keep,
        total_timesteps=total_timesteps,
        reload_frequency=args.reload_frequency,
        n_envs=n_envs,
        temp_dir=temp_dir,
        **job_args,
    )


if __name__ == "__main__":
    main()
