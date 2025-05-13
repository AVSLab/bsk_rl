import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import ray
# from avs_rl_tools.relative_motion.envs import (  # InspectorSat,; RSOSat,; dt_env,; inspector_sat_args,; rso_sat_args,
#     env_args,
#     env_metrics_callback,
#     sat_metrics_callback,
# )
from bsk_rl.utils.utils import get_available_cores
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger

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
                object_store_memory=2_000_000_000,  # 2 GB
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

    # Compute time *not* imaging a new target
    total_imaging_time = num_imaged * 300  # Each successful image takes 300s
    idle_time = episode_duration - total_imaging_time
    num_unproductive_actions = idle_time / 300
    data["unproductive_action_count"] = int(round(num_unproductive_actions))

    return data

def sat_metrics_callback(env, satellite):
    data = {}
    print('if satellite.name == SS1', satellite.name == 'SS1')
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
    else:
        data["RW_norm"] = 0
        data["RW1"] = 0
        data["RW2"] = 0
        data["RW3"] = 0

        data["battery_charge_fraction"] = 0
        data["storage_level_fraction"] = 0

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

    n_targets = 100
    n_targets_ahead = 10
    extra_tima_factor = 1.5
    total_time = extra_tima_factor * n_targets * 300  #I give it 10 times the minimum time to finish

    class MyScanningSatellite(sats.Satellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),

            ),
            obs.PolarisScTargetProperties(
                dict(prop="target_elevation_angle", norm=1.0),
                dict(prop="angle_to_target", norm=1.0),
                dict(prop="rel_pos_vector_r_BR_N", norm = 1596*1000),
                dict(prop="target_distance", norm = 1596*1000), #normalization calculated assuming h = 800 km and min elevation is -14 deg
                # dict(prop="target_id_info", norm=1.0),
                dict(prop="target_imaged",  norm=1.0),
                n_ahead_observe=n_targets_ahead,
                                           ),
            obs.Eclipse(),
        ]
        action_spec = [
            act.ImageRSO(n_ahead_image=n_targets_ahead,duration=300),  # Scan for 1 minute
            act.Charge(duration=600.0),  # Charge for 10 minutes
            act.Desat(duration=150) # Desat for 2.5 minutes
        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    # MyScanningSatellite.default_sat_args() # why is this needed?

    sat_args = {}

    # Set some parameters as constants
    sat_args["imageAttErrorRequirement"] = 0.05
    sat_args["dataStorageCapacity"] = 1e10
    sat_args["instrumentBaudRate"] = 1e7
    # sat_args["storedCharge_Init"] = 50000000.0

    # Randomize the initial storage level on every reset
    sat_args["storageInit"] = lambda: np.random.uniform(0.25, 0.75) * 1e10


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

    def custom_oe_randomizer():
        rLEO = 7000. * 1000    # Minimum semi-major axis (LEO) in meters
        rUpperLEO = 1.2 * 7000. * 1000    # Minimum semi-major axis (LEO) in meters
        # rGEO = 42164. * 1000   # Maximum semi-major axis (GEO) in meters


        oe = orbitalMotion.ClassicElements()
        # oe.a = np.random.uniform(rLEO*5, rGEO)  # Random semi-major axis between LEO and GEO
        oe.a = np.random.uniform(1.01*rLEO, rUpperLEO)  # Random semi-major axis between LEO and GEO

        if oe.a < 2*rLEO:
            oe.e = np.random.uniform(0.0, 0.02)    # Random eccentricity (allowing less elliptical orbits when near LEO)
        else:
            oe.e = np.random.uniform(0.0, 0.2)    # Random eccentricity (allowing slightly elliptical orbits)
        oe.i = np.random.uniform(0, 180) * macros.D2R  # Random inclination up to 180 degrees
        oe.Omega = np.random.uniform(0, 360) * macros.D2R  # Random RAAN
        oe.omega = np.random.uniform(0, 360) * macros.D2R  # Random argument of perigee
        oe.f = np.random.uniform(0, 360) * macros.D2R  # Random true anomaly
        # print('randomized orbital elements: oe.a, oe.i, oe.e', oe.a, oe.i, oe.e, 'oe.Omega, oe.omega, oe.f', oe.Omega, oe.omega, oe.f)
        return oe

    target_args=dict(oe=custom_oe_randomizer, batteryStorageCapacity = 80.0 * 3600.0*1000, storedCharge_Init = 80.0 * 3600.0*900 )
    # Make the satellite
    sat = MyScanningSatellite(name="SS1", sat_args=sat_args) # SO1 for satellite observer 1


    targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)]

    all_sat = [sat] + targets

    N = 0 # int(sys.argv[1])  # Passed by sweep.sh script
    model_name = f"model_{N}"
    n_envs = (
        get_available_cores() - 6  # leave some extra cores for other processes
    )
    output_dir = (
        Path("~/rllib_results").expanduser() / f"battery_Polaris_simulation_{time.time()}"
    )
    output_dir = Path(output_dir)

    print(f"Tensorboard logging: tensorboard --logdir {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    jobs = build_job_array(
        training_args=dict(
            lr=[1e-5],
            gamma=[0.9995],
            train_batch_size=[int(50 * n_envs)],
            num_sgd_iter=[10],
            lambda_=[0.95],
            use_kl_loss=[False],
            clip_param=[0.1],
            grad_clip=[1.0],
            entropy_coeff=[0.0],
        ),
        env_args=dict(
            # **{k: [v] for k, v in env_args().items()},
            satellites=[all_sat],
            scenario=[scene.RandomSatellites("SS1",n_targets=n_targets)],
            rewarder=[data.RSOTargetImageReward()],
            time_limit=[total_time],
            failure_penalty=[-1.0],
            terminate_on_time_limit=[True],
            generate_obs_retasking_only=[False],  # For last step
            episode_data_callback=[env_metrics_callback],
            satellite_data_callback=[sat_metrics_callback],
        ),
    )

    print(f"Running job {N}: {N+1} of {len(jobs)}")
    job_args = jobs[N]

    with open(output_dir / f"{model_name}_params_may13th.txt", "w") as file:
        yaml.dump(sanitize_np(job_args), file)

    train_model(
        model_name=model_name,
        output_directory=output_dir,
        checkpoint_frequency=5,
        checkpoints_to_keep=3,
        total_timesteps=20_000_000,
        reload_frequency=300_000,
        n_envs=n_envs,
        # temp_dir="/scratch/alpine/dahu1128/tmp",
        **job_args,
    )