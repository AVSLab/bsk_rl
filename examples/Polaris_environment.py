import gymnasium as gym
from requests.packages import target
import time

import os
import numpy as np

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

n_targets = 1
total_time = 5700 # 5700.0  # approximately 1 orbit

class MyScanningSatellite(sats.Satellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction")
        ),
        obs.Eclipse(),
    ]
    action_spec = [
        act.ImageRSO(n_ahead_image=n_targets,duration=2000),  # Scan for 1 minute
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
target0 = dict(
    oe=custom_oe_randomizer,
    batteryStorageCapacity=80.0 * 3600.0 * 1000,
    storedCharge_Init=80.0 * 3600.0 * 900
)
#
# target1 = dict(
#     oe=custom_oe_randomizer,
#     batteryStorageCapacity=80.0 * 3600.0 * 1000,
#     storedCharge_Init=80.0 * 3600.0 * 900
# )
# target0 = dict(
#     oe=custom_oe_randomizer(random_seed = 9),
#     batteryStorageCapacity=80.0 * 3600.0 * 1000,
#     storedCharge_Init=80.0 * 3600.0 * 900
# )
#
# target1 = dict(
#     oe=custom_oe_randomizer(random_seed = 7),
#     batteryStorageCapacity=80.0 * 3600.0 * 1000,
#     storedCharge_Init=80.0 * 3600.0 * 900
# )
# target2 = dict(
#     oe=custom_oe_randomizer(random_seed = 15),
#     batteryStorageCapacity=80.0 * 3600.0 * 1000,
#     storedCharge_Init=80.0 * 3600.0 * 900
# )
# target3 = dict(
#     oe=custom_oe_randomizer(random_seed = 18),
#     batteryStorageCapacity=80.0 * 3600.0 * 1000,
#     storedCharge_Init=80.0 * 3600.0 * 900
# )
# target4 = dict(
#     oe=custom_oe_randomizer(random_seed = 12),
#     batteryStorageCapacity=80.0 * 3600.0 * 1000,
#     storedCharge_Init=80.0 * 3600.0 * 900
# )

# targets = [MyTargetSatellite(name=f"target_0", sat_args=target0), MyTargetSatellite(name=f"target_1", sat_args=target1), MyTargetSatellite(name=f"target_2", sat_args=target2), MyTargetSatellite(name=f"target_3", sat_args=target3), MyTargetSatellite(name=f"target_4", sat_args=target4)]
# targets = [MyTargetSatellite(name=f"target_0", sat_args=target0), MyTargetSatellite(name=f"target_1", sat_args=target1)]
targets = [MyTargetSatellite(name=f"target_0", sat_args=target0)]

# targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)] # TODO: this creates the same IC of oe for all targets

all_sat = [sat] + targets   #oe = lambda: random_orbit(alt=np.random.uniform(1000,2000)))

env = gym.make(
    "ConstellationTasking-v1",
    satellites=all_sat,
    scenario=scene.RandomSatellites("SS1",n_targets=n_targets),
    rewarder=data.RSOTargetImageReward(),
    time_limit=total_time,
    log_level="DEBUG",
    disable_env_checker=True,
    # max_step_duration=700,
)

observation, info = env.reset(seed=2)

env.simulator.ShowExecutionOrder()
# Initialize storage dictionary
data_dict = {
    "sim_time": [],
    "inspector_sigmaBN": [],
    "inspector_omegaBN": [],
    "inspector_r_BN_N": [],
    "currentTarget_r_BN_N": [],
    "target_r_BN_N": {target.name: [] for target in targets}  # Store per target
}

print("Initial data level:", observation, "(randomized by sat_args)")
for target_id in range(n_targets):
    simtime = env.simulator.sim_time
    print('Simulation time: ' + str(simtime) + ' seconds')

    action_dict = {sat.name: target_id}  # Assign the main satellite to observe `target_idx`
    action_dict.update({targets[j].name: 0 for j in range(n_targets)})  # Initialize all targets to 0
    print('current action_dict to be executed', action_dict)
    observation, reward, terminated, truncated, info = env.step(action=action_dict)
    print('truncated list: ', truncated)
    data_dict["sim_time"].append(env.simulator.sim_time)

print("  Final data level:", observation)

while not truncated:
    data_dict["sim_time"].append(env.simulator.sim_time)

data_dict["inspector_sigmaBN"].append(env.satellites[0].dynamics.inspector_state_recorder.sigma_BN)
data_dict["inspector_omegaBN"].append(env.satellites[0].dynamics.inspector_state_recorder.omega_BN_B)
data_dict["inspector_r_BN_N"].append(env.satellites[0].dynamics.inspector_state_recorder.r_BN_N)
data_dict["currentTarget_r_BN_N"].append(env.satellites[0].dynamics.simpleNavObject.transOutMsg.read().r_BN_N)

for l in range (len(targets)):
    data_dict["target_r_BN_N"][targets[l].name].append(env.satellites[l+1].dynamics.target_state_recorder.r_BN_N)




data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

for key, value in data_dict.items():
    if isinstance(value, dict):  # Save per-target data separately
        for target_name, target_data in value.items():
            np.save(os.path.join(data_dir, f"{key}_{target_name}.npy"), np.array(target_data))
    else:
        np.save(os.path.join(data_dir, f"{key}.npy"), np.array(value))

print("Data saved successfully in 'data/' folder.")