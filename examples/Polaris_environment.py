import gymnasium as gym
from requests.packages import target
import time
start_time = time.time()

import os
import numpy as np

# Ensure the data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

from bsk_rl import act, data, obs, scene, sats, utils
from bsk_rl.sim import dyn, fsw, world

from Basilisk.utilities import (
    macros,
    orbitalMotion,
)

from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


save_data = True #set to False to avoid saving data

n_targets = 100
n_targets_ahead = 10
total_time = n_targets * 450  # 2100 # 5700.0  # approximately 1 orbit

class MyScanningSatellite(sats.AccessSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction"),
        ),
        obs.PolarisScTargetProperties(
            dict(prop="target_elevation_angle", norm=1.0),
            dict(prop="rel_pos_vector_r_BR_N", norm = 1596*1000),
            dict(prop="angle_to_target", norm=1.0),
            dict(prop="target_distance", norm = 1596*1000), #normalization calculated assuming h = 800 km and min elevation is -14 deg
            dict(prop="target_id_info", norm=1.0),
            dict(prop="target_imaged",  norm=1.0),
            dict(prop="target_shadowFactor", norm=1.0),
            n_ahead_observe=n_targets_ahead,
                                       ),
        obs.Eclipse(norm=5700.0),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm = 5700.0),
            dict(prop="opportunity_close", norm = 5700.0),
            type="ground_station",
            n_ahead_observe=5,
        )
    ]
    action_spec = [
        act.ImageRSO(n_ahead_image=n_targets_ahead,duration=300),  # Scan for 5 minute
        act.Charge(duration=300.0),  # Charge for 5 minutes
        act.Downlink(duration=180.0), # Downlink for 3 min
        act.Desat(duration=150), # Desat for 2.5 min

    ]
    dyn_type = dyn.ImagingSCDynModel
    fsw_type = fsw.ImagingSCFSWModel

# MyScanningSatellite.default_sat_args() # why is this needed?

sat_args = {}

# Set some parameters as constants
sat_args["imageAttErrorRequirement"] = 0.01

# Storage
sat_args["dataStorageCapacity"] = 50 * 8e6 / 2 # bits
sat_args["storageInit"] = lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2
sat_args["instrumentBaudRate"] = 0.5 * 8e6
sat_args["transmitterBaudRate"] = -0.5 * 8e6

# Power
sat_args["batteryStorageCapacity"] = 500 * 3600  # W*s
sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.4, 0.6) * 500 * 3600
sat_args["basePowerDraw"] = -10.0  # W
sat_args["instrumentPowerDraw"] = -30.0  # W
sat_args["transmitterPowerDraw"] = -25.0  # W
sat_args["thrusterPowerDraw"] = -80.0  # W
# sat_args["panelArea"] = 0.25  # m^2

# Attitude
sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.001, size=3)  # N*m
sat_args["maxWheelSpeed"] = 6000.0  # RPM
sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
sat_args["desatAttitude"] = "nadir"

# reward bonuses and eclipse thresholds
sat_args["downlink_bonus"] = 0.6
sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
sat_args["eclipse_threshold_for_imaging"] = 0.5
sat_args["eclipse_threshold_for_reward"] = sat_args["eclipse_threshold_for_imaging"]

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
    rUpperLEO = 1.1 * 7000. * 1000    # max semi-major axis  of upper LEO in meters
    # rGEO = 42164. * 1000   # Maximum semi-major axis (GEO) in meters


    oe = orbitalMotion.ClassicElements()
    # oe.a = np.random.uniform(rLEO*5, rGEO)  # Random semi-major axis between LEO and GEO
    oe.a = np.random.uniform(1.05*rLEO, rUpperLEO)  # Random semi-major axis between LEO and GEO

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
sat = MyScanningSatellite(name="SS1", sat_args=sat_args, obs_type=dict) # SO1 for satellite observer 1
# target0 = dict(
#     oe=custom_oe_randomizer,
#     batteryStorageCapacity=80.0 * 3600.0 * 1000,
#     storedCharge_Init=80.0 * 3600.0 * 900
# )
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
# targets = [MyTargetSatellite(name=f"target_0", sat_args=target0)]

targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)] # TODO: this creates the same IC of oe for all targets

all_sat = [sat] + targets   #oe = lambda: random_orbit(alt=np.random.uniform(1000,2000)))

env = gym.make(
    "ConstellationTasking-v1",
    satellites=all_sat,
    scenario=scene.RandomSatellites("SS1",n_targets=n_targets),
    rewarder=data.RSOTargetImageReward(),
    world_type=world.GroundStationWorldModel,
    time_limit=total_time,
    log_level="ERROR",
    disable_env_checker=True,
    # max_step_duration=700,
)

observation, info = env.reset(seed=0)

# env.simulator.ShowExecutionOrder() # to show execution order

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
last_downlink_time = 0
critical_storage_level = 0.95 # only task downlink if available storage_fraction is less than 0.05
critical_battery_level = 0.3
for target_id in range(n_targets*4 *100 ):
    simtime = env.simulator.sim_time
    print('\n SIMULATION TIME: ' + str(simtime) + ' seconds')

    # action_dict = {sat.name: target_id}  # Assign the main satellite to observe `target_idx` # sequentially observing each target
    action_dict = {sat.name: 0}  # Assign the closest target when the list is sorted by distance
    if env.satellites[0].dynamics.storage_level_fraction > critical_storage_level:  # downlink if storage is more than 0.95
    #if simtime - last_downlink_time > 3000:
        print('tasking DONWLINKING now: at t=',simtime)
        action_dict = {sat.name: 11} # tasking downlink
        last_downlink_time = simtime

    if env.satellites[0].dynamics.battery_charge_fraction < critical_battery_level:  # charge if battery is less than 0.05
        print('tasking CHARGING now: at t=',simtime)
        action_dict = {sat.name: 10} # tasking charging
    action_dict.update({targets[j].name: 0 for j in range(n_targets)})  # Initialize all targets to 0
    # print('current action_dict to be executed', action_dict)
    observation, reward, terminated, truncated, info = env.step(action=action_dict)
    print("storage_level", env.satellites[0].dynamics.storage_level)
    print("dynamics.storage_level_fraction", env.satellites[0].dynamics.storage_level_fraction)
    print("dynamics.battery_charge_fraction", env.satellites[0].dynamics.battery_charge_fraction)
    print("env.satellites[0].dynamics.wheel_speeds_fraction", env.satellites[0].dynamics.wheel_speeds_fraction)


    # print('truncated list: ', truncated)
    data_dict["sim_time"].append(env.simulator.sim_time)
    if any(truncated.values()) or any(terminated.values()):
        break

print("  Final data level:", observation)

while not truncated:
    data_dict["sim_time"].append(env.simulator.sim_time)

data_dict["inspector_sigmaBN"].append(env.satellites[0].dynamics.inspector_state_recorder.sigma_BN)
data_dict["inspector_omegaBN"].append(env.satellites[0].dynamics.inspector_state_recorder.omega_BN_B)
data_dict["inspector_r_BN_N"].append(env.satellites[0].dynamics.inspector_state_recorder.r_BN_N)
data_dict["currentTarget_r_BN_N"].append(env.satellites[0].dynamics.simpleNavObject.transOutMsg.read().r_BN_N)

for l in range (len(targets)):
    data_dict["target_r_BN_N"][targets[l].name].append(env.satellites[l+1].dynamics.target_state_recorder.r_BN_N)



if save_data:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for key, value in data_dict.items():
        if isinstance(value, dict):  # Save per-target data separately
            for target_name, target_data in value.items():
                np.save(os.path.join(data_dir, f"{key}_{target_name}.npy"), np.array(target_data))
        else:
            np.save(os.path.join(data_dir, f"{key}.npy"), np.array(value))

    print("Data saved successfully in 'data/' folder.")
else:
    print("Not saving data")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time:.4f} seconds")