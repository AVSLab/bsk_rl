import gymnasium as gym
from requests.packages import target
import time

import os
import numpy as np

from importlib.metadata import version
version("ray")  # Parent package of RLlib

# Ensure the data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

from bsk_rl import act, data, obs, scene, sats, utils
from bsk_rl.sim import dyn, fsw

from Basilisk.utilities import (
    macros,
    orbitalMotion,
)

from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

n_targets = 5
total_time = 5*5700 # 5700.0  # approximately 1 orbit

class MyScanningSatellite(sats.Satellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction")
        ),
        obs.Eclipse(),
    ]
    action_spec = [
        act.ImageRSO(n_ahead_image=n_targets,duration=150),  # Scan for 2.5 minute
        act.Charge(duration=600.0),  # Charge for 10 minutes
    ]
    dyn_type = dyn.ImagingSCDynModel
    fsw_type = fsw.ImagingSCFSWModel

MyScanningSatellite.default_sat_args() # why is this needed?

sat_args = {}

# Set some parameters as constants
sat_args["imageAttErrorRequirement"] = 0.05
sat_args["dataStorageCapacity"] = 1e10
sat_args["instrumentBaudRate"] = 1e7
sat_args["storedCharge_Init"] = 50000.0

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
    rGEO = 42164. * 1000   # Maximum semi-major axis (GEO) in meters


    oe = orbitalMotion.ClassicElements()
    oe.a = np.random.uniform(rLEO, rGEO)  # Random semi-major axis between LEO and GEO
    if oe.a < 1.5*rLEO:
        oe.e = np.random.uniform(0.0, 0.1)    # Random eccentricity (allowing less elliptical orbits when near LEO)
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

all_sat = [sat] + targets   #oe = lambda: random_orbit(alt=np.random.uniform(1000,2000)))

env_args = dict(
    # "ConstellationTasking-v1",
    satellites=all_sat,
    scenario=scene.RandomSatellites("SS1",n_targets=n_targets),
    rewarder=data.RSOTargetImageReward(),
    time_limit=total_time,
    log_level="DEBUG",
    failure_penalty=-1.0,
    terminate_on_time_limit=True,
)

def episode_data_callback(env):
    reward = env.rewarder.cum_reward
    reward = sum(reward.values()) / len(reward)
    orbits = env.simulator.sim_time / (95 * 60)

    data = dict(
        reward=reward,
        # Are satellites dying, and how and when?
        alive=float(env.satellite.is_alive()),
        rw_status_valid=float(env.satellite.dynamics.rw_speeds_valid()),
        battery_status_valid=float(env.satellite.dynamics.battery_valid()),
        orbits_complete=orbits,
    )
    if orbits > 0:
        data["reward_per_orbit"] = reward / orbits
    if not env.satellite.is_alive():
        data["orbits_complete_partial_only"] = orbits

    return data

#%%
import bsk_rl.utils.rllib  # noqa To access "SatelliteTasking-RLlib"
from ray.rllib.algorithms.ppo import PPOConfig
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks

N_CPUS = 3

training_args = dict(
    lr=0.00003,
    gamma=0.999,
    train_batch_size=250,  # usually a larger number, like 2500
    num_sgd_iter=10,
    model=dict(fcnet_hiddens=[512, 512], vf_share_layers=False), # can change this number for example to 2048,2048
    lambda_=0.95,
    use_kl_loss=False,
    clip_param=0.1,
    grad_clip=0.5,
)

config = (
    PPOConfig()
    .training(**training_args)
    .env_runners(num_env_runners=N_CPUS-1, sample_timeout_s=1000.0)
    .environment(
        env="ConstellationTasking-RLlib",
        env_config=dict(**env_args, episode_data_callback=episode_data_callback),
    )
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
    .callbacks(WrappedEpisodeDataCallbacks)
)

import ray
from ray import tune

ray.init(
    ignore_reinit_error=True,
    num_cpus=N_CPUS,
    object_store_memory=2_000_000_000,  # 2 GB
)

# Run the training
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 10},  # Adjust the number of iterations as needed
    checkpoint_freq=10,
    checkpoint_at_end=True
)

# Shutdown Ray
ray.shutdown()

from bsk_rl import SatelliteTasking

env = SatelliteTasking(**env_args, log_level="INFO")
env.reset()
terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

print("Training Complete")
