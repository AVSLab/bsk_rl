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

torch.set_num_threads(3)
os.environ["MKL_NUM_THREADS"] = "3" # 11 on the cluster

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
                            "fcnet_hiddens": [2048, 2048], #[2048, 2048], also tested [1024,1024] and it was pretty much the same
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

    while True:
        prev_step = step
        results = ppo.train()
        step = results["num_env_steps_sampled_lifetime"]
        step_return = results["env_runners"].get("episode_return_mean", -np.inf)

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

def env_metrics_callback(env):
    data = {}

    # Number of unique targets successfully imaged
    num_imaged = len(env.rewarder.data.imaged)
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

    # Compute time *not* imaging a new target
    total_imaging_time = num_imaged * 300  # Each successful image takes 300s
    idle_time = episode_duration - total_imaging_time
    num_unproductive_actions = idle_time / 300
    data["non-imaging_action_count"] = int(round(num_unproductive_actions))

    # Compute time *not* imaging a new target
    total_imaging_time = num_imaged * 300  # Each successful image takes 300s
    non_imaging_time = episode_duration - total_imaging_time
    data["non-imaging_time"] = int(round(non_imaging_time))

    data["cumulativeRewardSS1"]=env.rewarder.cum_reward['SS1']
    data["illuminated_images"] = len(env.rewarder.imaged_illuminated)

    SS1_actions_spec = env.satellites[0].action_builder.action_spec[0]
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
        data["mean_target_priority"] = np.mean(SS1_actions_spec.chosen_target_priority)
        data["std_target_priority"] = np.std(SS1_actions_spec.chosen_target_priority)
    else:
        data["mean_target_priority"] = -1
        data["std_target_priority"] = -1

    # Ever visible flags
    if hasattr(SS1_actions_spec, "ever_visible") and SS1_actions_spec.ever_visible:
        data["target_ever_visible_fraction"] = len(SS1_actions_spec.ever_visible) / n_targets
    else:
        data["target_ever_visible_fraction"] = -1

    return data

def sat_metrics_callback(env, satellite):
    data = {}
    # print('if satellite.name == SS1', satellite.name == 'SS1')
    if satellite.name == 'SS1':
        print('satellite.name', satellite.name)
        print('np.linalg.norm(satellite.dynamics.wheel_speeds)', np.linalg.norm(satellite.dynamics.wheel_speeds))
        print('satellite.dynamics.wheel_speeds', satellite.dynamics.wheel_speeds)
        print('satellite.dynamics.battery_charge_fraction', satellite.dynamics.battery_charge_fraction)
        print('satellite.dynamics.storage_level_fraction', satellite.dynamics.storage_level_fraction)
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
    # Shared sim configuration (should match what you use in evaluation)
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


    class MyScanningSatellite(sats.AccessSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction")
            ),

            #observation space 1
            # obs.PolarisScTargetProperties(
            #     dict(prop="target_elevation_angle", norm=1.0),
            #     dict(prop="rel_pos_vector_r_BR_N", norm = 1596*1000),
            #     dict(prop="angle_to_target", norm=1.0),
            #     dict(prop="target_distance", norm = 1596*1000), #normalization calculated assuming h = 800 km and min elevation is -14 deg
            #     dict(prop="target_id_info", norm=1.0),
            #     dict(prop="target_imaged",  norm=1.0),
            #     n_ahead_observe=n_targets_ahead,
            #                                ),

            #observation space 2
            obs.PolarisScTargetProperties(
                dict(prop="target_elevation_angle", norm=90.0),
                dict(prop="rel_pos_vector_r_BR_H", norm = 15960*1000),
                dict(prop="angle_to_target", norm=90.0),
                dict(prop="target_distance", norm = 15960*1000), #normalization calculated assuming h = 800 km and min elevation is -14 deg
                dict(prop="target_shadowFactor", norm=1.0),
                n_ahead_observe=n_targets_ahead,
                                           ),
            obs.Eclipse(norm=5700),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=2,
            )
        ]
        action_spec = [
            act.ImageRSO(n_ahead_image=n_targets_ahead,duration=imaging_duration),  # Scan for 5 minute
            act.Charge(duration=300.0),  # Charge for 5 minutes
            act.Downlink(duration=300.0), # Downlink for 3 min
            act.Desat(duration=150), # Desat for 2.5 min

        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    sat_args = {}
    # Set some parameters as constants
    sat_args["imageAttErrorRequirement"] = 0.0025
    sat_args["imageRateErrorRequirement"] = 0.01

    # Storage
    sat_args["dataStorageCapacity"] = 50 * 8e6 / 2 #*1000000 # bits
    sat_args["storageInit"] = lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2
    sat_args["instrumentBaudRate"] = 0.5 * 8e6
    sat_args["transmitterBaudRate"] = -0.5 * 8e6

    # Power
    sat_args["batteryStorageCapacity"] = 500 * 3600 # *1000000 # W*s
    sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.10, 0.4) * 500 * 3600 #*1000000
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

    # reward bonuses and eclipse thresholds
    sat_args["downlink_bonus"] = 0.0
    sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
    sat_args["eclipse_threshold_for_imaging"] = 0.5 # to include both shadowed and illuminated RSOs
    sat_args["eclipse_threshold_for_reward"] = 0.5 # can be the same as sat_args["eclipse_threshold_for_imaging"] if set to a positive number between 0 and 1
    # sat_args["full_storage_penalty"] = -1
    # sat_args["low_battery_penalty"] = -1

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


    # target_args=dict(oe=custom_oe_randomizer, batteryStorageCapacity = 80.0 * 3600.0*1000, storedCharge_Init = 80.0 * 3600.0*900 )
    # target_args=dict(oe=custom_oe_randomizer, batteryStorageCapacity = 1, storedCharge_Init = 0.0, basePowerDraw = -10000.0 )  # testing to see if sim is faster if the other agents are killed
    target_args_mixed = dict(oe=partial(custom_oe_randomizer, regime="mixed", mix_weights={"LEO":0.5,"MEO":0.3,"GEO":0.2}), batteryStorageCapacity = 1, storedCharge_Init = 0.0, basePowerDraw = -10000.0 )



    # Make the satellite
    sat = MyScanningSatellite(name="SS1", sat_args=sat_args) # SO1 for satellite observer 1

    targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args_mixed) for i in range(n_targets)]

    all_sat = [sat] + targets

    N = 0 # int(sys.argv[1])  # Passed by sweep.sh script
    model_name = f"aug20_wGAE_4200batch_restrictedResources_obsv7_1e-5lr_0.05cp_gradclip0.5_gamma9997_0d100i.out_{N}"
    n_envs = (
        get_available_cores() - 4  # leave some extra cores for other processes
    )
    output_dir = (
        Path("~/rllib_results/july_results/july30rllib_results").expanduser() / f"aug20_wGAE_4200batch_restrictedResources_obsv7_1e-5lr_0.05cp_gradclip0.5_gamma9997_0d100i_{time.time()}" #change this when running on cluster (add /scratch/alpine/dahu1128/rllib_results as directory)
    )
    output_dir = Path(output_dir)

    print(f"Tensorboard logging: tensorboard --logdir {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    jobs = build_job_array(
        training_args=dict(
            lr=[1e-5],
            gamma=[0.9997],
            train_batch_size=[int(600 * n_envs)],  #n_envs on the Mac is 6 eventually   minimum train_batch_size = mini_batch = 128
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
            scenario=[scene.RandomSatellites("SS1",n_targets=n_targets)],
            rewarder=[data.RSOTargetImageReward()],
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
        "job_args": sanitize_np(job_args),
    }

    with open(output_dir / f"{model_name}_config.yaml", "w") as file:
        yaml.dump(run_cfg, file)

    train_model(
        model_name=model_name,
        output_directory=output_dir,
        checkpoint_frequency=3, # used to be 2
        checkpoints_to_keep=3,
        total_timesteps=20_000_000,
        reload_frequency=500_000,
        n_envs=n_envs,
        # temp_dir="/scratch/alpine/dahu1128/tmp",
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