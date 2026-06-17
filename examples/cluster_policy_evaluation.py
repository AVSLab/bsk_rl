import gymnasium as gym
from requests.packages import target
import time
start_time = time.time()
from pathlib import Path
import os
import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict

# Ensure the data directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

from bsk_rl import act, data, obs, scene, sats, utils
from bsk_rl.sim import dyn, fsw, world
# from bsk_rl.utils import utils_load_policy

from Basilisk.utilities import (
    macros,
    orbitalMotion,
)

# from examples.load_policy import load_policy
from load_policy import load_policy
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

class Policy:
    def __init__(self, policy_path, zero_element=None) -> None:
        self.zero_element = zero_element
        self.policy_function = None
        if policy_path is not None:
            # self.policy_function = load_torch_mlp_policy(policy_path, env_args)
            self.policy_function = load_policy(policy_path)

    def act(self, observation):
        if self.zero_element is not None:
            observation[self.zero_element] = 0.0
        if self.policy_function is not None:
            action = self.policy_function(observation)
        else:
            action = None

        return action


save_data = False #set to False to avoid saving data

n_targets = 100
n_targets_ahead = 10
imaging_duration = 300
total_time = n_targets * 300  # 2100 # 5700.0  # approximately 1 orbit
# obs_v = 1.5
obs_v =2
# loading the policy
# downlink_reward_policy = "/Users/dahu1128/rllib_results/june12rllib_results/lowBaudRate_5e-6lr_downlink_reward_penalties_smallest_storage_Polaris_simulation_1749771784.4956822/lowBaudRate_5e-6lr_downlink_reward_penalties_smallest_storage_0/"
downlink_reward_policy = "/Users/dahu1128/rllib_results/reward_comparison/lowBaudRate_5e-6lr_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1750741069.7033312/lowBaudRate_5e-6lr_downlink_reward_new_penalties_smallest_storage_0"
downlink_reward_policy_shorter_imaging ="/Users/dahu1128/rllib_results/reward_comparison/lowBaudRate_shorter_imaging_5e-6lr_downlink_reward_penalties_smallest_storage_0"

# june23rd
cluster_policy = "/scratch/alpine/dahu1128/june23rllib_results/lowBaudRate_5e-6lr_001torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1750740679.4746056/lowBaudRate_5e-6lr_001torque_downlink_reward_new_penalties_smallest_storage_0"

# july16th
cluster_5050policy_obs2 = "/scratch/alpine/dahu1128/july16rllib_results/july16_1e-5lr_002torque_50d50i_reward_new_penalties_smallest_storage_Polaris_simulation_1752838082.5491006/1e-5lr_002torque_50d50i_reward_new_penalties_smallest_storage_0"
cluster_7525policy_obs2 = "/scratch/alpine/dahu1128/july16rllib_results/july16_1e-5lr_002torque_75d25i_reward_new_penalties_smallest_storage_Polaris_simulation_1752742415.630985/1e-5lr_002torque_75d25i_reward_new_penalties_smallest_storage_0"
cluster_9010policy_obs2 = "/scratch/alpine/dahu1128/july16rllib_results/july16_1e-5lr_002torque_90d10i_reward_new_penalties_smallest_storage_Polaris_simulation_1752870279.3884969/1e-5lr_002torque_90d10i_reward_new_penalties_smallest_storage_0"

# july10th
cluster_6040policy_obs11 = "/scratch/alpine/dahu1128/july10rllib_results/july10_1e-5lr_002torque_60d40i_reward_new_penalties_smallest_storage_Polaris_simulation_1752241793.6701877/1e-5lr_002torque_60d40i_reward_new_penalties_smallest_storage_0"

policy_path = cluster_6040policy_obs11

# Define all known policy paths with associated obs values
policy_obs_map = {
    "cluster_5050policy_obs2": 2,
    "cluster_7525policy_obs2": 2,
    "cluster_9010policy_obs2": 2,
    "cluster_6040policy_obs11": 1.1,
    "downlink_reward_policy": 1,
    "downlink_reward_policy_shorter_imaging": 1,
    "cluster_policy": 1,
}

# Compare policy_path to known variables
for name, val in list(globals().items()):
    if isinstance(val, str) and val == policy_path and name in policy_obs_map:
        obs_v = policy_obs_map[name]
        break

# Load policy
policy = Policy(policy_path)

class MyScanningSatellite(sats.AccessSatellite):
    if obs_v==1:
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
                n_ahead_observe=n_targets_ahead,
                                           ),
            obs.Eclipse(),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=5,
            )
        ]
    elif obs_v==1.1:
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
            obs.Eclipse(),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=5,
            )
        ]
    elif obs_v==2:
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
            ),
            obs.PolarisScTargetProperties(
                dict(prop="target_elevation_angle", norm=1.0),
                dict(prop="rel_pos_vector_r_BR_H", norm = 1596*1000),
                dict(prop="angle_to_target", norm=1.0),
                dict(prop="target_distance", norm = 1596*1000), #normalization calculated assuming h = 800 km and min elevation is -14 deg
                dict(prop="target_shadowFactor", norm=1.0),
                n_ahead_observe=n_targets_ahead,
                                           ),
            obs.Eclipse(),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=5,
            )
        ]
    action_spec = [
        act.ImageRSO(n_ahead_image=n_targets_ahead,duration=imaging_duration),  # Scan for 5 minute
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

# Randomize the initial storage level on every reset
# sat_args["storageInit"] = lambda: np.random.uniform(0., 0.0) * 1e10  # 0.25, 0.75) * 1e10

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
# sat_args["imageAttErrorRequirement"] = 0.1
# sat_args["imageRateErrorRequirement"] = 0.1
sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.002, size=3)  # N*m
sat_args["maxWheelSpeed"] = 6000.0  # RPM
sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
sat_args["desatAttitude"] = "nadir"



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
# sat = MyScanningSatellite(name="SS1", sat_args=sat_args, obs_type=dict) # SO1 for satellite observer 1
sat = MyScanningSatellite(name="SS1", sat_args=sat_args) # SO1 for satellite observer 1

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

observation, info = env.reset(seed=5) # change the seed number here or make it fully random for every iteration

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

last_downlink_time = 0 # initialize downlink time

use_shield=False
critical_storage_level = 0.95 # only task downlink if available storage_fraction is less than 0.05
critical_battery_level = 0.3 # always task charging if battery is below 0.3

# storing the actions taken by the agent
action_counts = defaultdict(int)


for target_id in range(n_targets*6 *100 ):
    simtime = env.simulator.sim_time
    print('\n SIMULATION TIME: ' + str(simtime) + ' seconds')
    # Use policy to determine action
    obs_for_policy = flatten_to_single_ndarray(observation[sat.name]) # This is a dict (as your env uses `obs_type=dict`)
    # obs_flat = flatten_to_single_ndarray(env.observation_spaces[sat.name], observation[sat.name])

    policy_action = policy.act(obs_for_policy)

    if isinstance(policy_action, np.ndarray):  # Handle vector action output (if needed)
        policy_action = policy_action.item()  # or do appropriate conversion if the policy returns a torch tensor

    action_counts[policy_action] += 1

    # action_dict = {sat.name: target_id}  # Assign the main satellite to observe `target_idx` # sequentially observing each target
    action_dict = {sat.name: 0}  # Assign the closest target when the list is sorted by distance
    action_dict = {sat.name: policy_action}  # Assign the closest target when the list is sorted by distance
    if policy_action == 11:
        print('tasking DOWNLINKING now: at t=',simtime," and storage level --> "+str(env.satellites[0].dynamics.storage_level_fraction))
    elif policy_action == 10:
        print('tasking CHARGING now: at t=',simtime," and battery level --> "+str(env.satellites[0].dynamics.battery_charge_fraction))
    elif policy_action == 12:
        print('tasking DESAT now: at t=',simtime," and wheel_speeds --> "+str(env.satellites[0].dynamics.wheel_speeds_fraction))

    if use_shield == True:
        if env.satellites[0].dynamics.storage_level_fraction > critical_storage_level:  # downlink if storage is more than 0.95
        #if simtime - last_downlink_time > 3000:
            print('tasking DOWNLINKING now: at t=',simtime)
            action_dict = {sat.name: 11} # tasking downlink
            last_downlink_time = simtime

        if env.satellites[0].dynamics.battery_charge_fraction < critical_battery_level:  # charge if battery is less than 0.05
            print('tasking CHARGING now: at t=',simtime)
            action_dict = {sat.name: 10} # tasking charging
    action_dict.update({targets[j].name: 0 for j in range(n_targets)})  # Initialize all targets to 0
    print('current action_dict to be executed', action_dict['SS1'])
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

# ---- Plotting ----
total_actions = sum(action_counts.values())
actions = list(range(13))  # Actions 0–12
action_labels = [
    f"Target {i}" if i <= 9 else ["Charging", "Downlink", "Desat"][i - 10]
    for i in actions
]
counts = [action_counts[a] for a in actions]
percentages = [100 * count / total_actions for count in counts]

# Plot 1: Absolute Action Counts
plt.figure(figsize=(10, 5))
plt.bar(action_labels, counts, color="skyblue")
plt.title("Action Count Distribution")
plt.ylabel("Number of Times Action Was Taken")
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("obs1_6040_action_count_distribution.pdf")  # Save as PDF
# plt.savefig("extended_time_action_count_distribution.pdf")  # Save as PDF

plt.show()

# Plot 2: Action Percentages
plt.figure(figsize=(10, 5))
plt.bar(action_labels, percentages, color="mediumseagreen")
plt.title("Action Percentage Distribution")
plt.ylabel("Percentage of Total Actions (%)")
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("obs1_6040_action_percentage_distribution.pdf")  # Save as PDF
# plt.savefig("extended_time_action_percentage_distribution.pdf")  # Save as PDF
plt.show()

# Compute total target-imaging and non-target-imaging actions
target_imaging_count = sum(counts[i] for i in range(10))     # Actions 0–9
charge_action_count = counts[10]
downlink_action_count = counts[11]
desat_action_count = counts[12]
non_target_count = sum(counts[i] for i in range(10, 13))     # Actions 10–12

# Compute percentages
target_imaging_pct = 100 * target_imaging_count / total_actions
non_target_pct = 100 * non_target_count / total_actions

# Print summary
print("\n=== Imaging vs Non-Imaging Summary ===")
print(f"Target Imaging Actions (0–9): {target_imaging_count} ({target_imaging_pct:.2f}%)")
print(f"Other Actions (10–12):        {non_target_count} ({non_target_pct:.2f}%)")
print(f"Downlink actions: {downlink_action_count}")
print(f"Charge actions: {charge_action_count}")
print(f"desat actions: {desat_action_count}")
print("======================================\n")


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