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

from Basilisk.utilities import (
    macros,
    orbitalMotion,
)

from examples.load_policy import load_policy
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

def save_plot_unique(fig, base_filename, folder="plots", extension=".pdf"):
    """
    Saves a matplotlib figure with a unique filename if file already exists.

    Parameters:
    - fig: Matplotlib figure object
    - base_filename: Desired base filename without extension or folder
    - folder: Directory to save the plots (default is "plots")
    - extension: File extension (default is ".pdf")
    """
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, base_filename + extension)

    counter = 1
    while os.path.exists(full_path):
        full_path = os.path.join(folder, f"{base_filename}_{counter}{extension}")
        counter += 1

    fig.savefig(full_path, bbox_inches="tight")
    print(f"Plot saved to {full_path}")

class Policy:
    def __init__(self, policy_path, policy_mode,zero_element=None) -> None:
        self.zero_element = zero_element
        self.policy_function = None
        if policy_path is not None:
            # self.policy_function = load_torch_mlp_policy(policy_path, env_args)
            self.policy_function = load_policy(policy_path,policy_mode=policy_mode)

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
total_time = n_targets * 450  # 2100 # 5700.0  # approximately 1 orbit
# obs_v = 1.1
obs_v =2

# loading the policy
# downlink_reward_policy = "/Users/dahu1128/rllib_results/june12rllib_results/lowBaudRate_5e-6lr_downlink_reward_penalties_smallest_storage_Polaris_simulation_1749771784.4956822/lowBaudRate_5e-6lr_downlink_reward_penalties_smallest_storage_0/"
downlink_reward_policy = "/Users/dahu1128/rllib_results/reward_comparison/lowBaudRate_5e-6lr_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1750741069.7033312/lowBaudRate_5e-6lr_downlink_reward_new_penalties_smallest_storage_0"
downlink_reward_policy_shorter_imaging ="/Users/dahu1128/rllib_results/reward_comparison/lowBaudRate_shorter_imaging_5e-6lr_downlink_reward_penalties_smallest_storage_0"
# imaging_reward_policy = ""
# August 1st
balance75d25i_nopenalties_lowICbattery_woGAE_obs2_gamma95 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug1_nopenalties_smallerICbattery_obsv2_2e-5lr_0.1cp_gamma95_75d25i_1754054673.0471108/aug1_nopenalties_smallerICbattery_obsv2_2e-5lr_0.1cp_gamma95_75d25i.out_0"

# July 31st
balance75d25i_reward_policy_woGAE_obs2_gamma95 = "/Users/dahu1128/rllib_results/july31_obsv2_1e-6lr_0.1cp_gamma95_75d25i_1753923413.642091/july31_obsv2_1e-6lr_0.1cp_gamma95_75d25i.out_0"
balance75d25i_nopenalties_woGAE_obsv2_gamma95 = "/Users/dahu1128/rllib_results/july31_nopenalties_obsv2_1e-6lr_0.1cp_gamma95_75d25i_1753984221.842252/july31_nopenalties_obsv2_1e-6lr_0.1cp_gamma95_75d25i.out_0"

# July 22nd
balance00100_reward_policy_obs1_gamma99 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma99_0d100i_1753189664.120498/july22_obsv1_1e-5lr_0.1cp_gamma99_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs1_gamma995 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma995_0d100i_1753184464.3957765/july22_obsv1_1e-5lr_0.1cp_gamma995_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs1_gamma9995 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma9995_0d100i_1753252554.210772/july22_obsv1_1e-5lr_0.1cp_gamma9995_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs1_gamma99995 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma99995_0d100i_1753189664.1202838/july22_obsv1_1e-5lr_0.1cp_gamma99995_0d100i_job_%a-%j.out_0"

balance00100_reward_policy_obs2_gamma99 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma99_0d100i_1753349618.6042557/july22_obsv2_1e-5lr_0.1cp_gamma99_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs2_gamma995 = '/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma995_0d100i_1753350273.3467767/july22_obsv2_1e-5lr_0.1cp_gamma995_0d100i_job_%a-%j.out_0'
balance00100_reward_policy_obs2_gamma9995 = '/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma9995_0d100i_1753504923.8221972/july22_obsv2_1e-5lr_0.1cp_gamma9995_0d100i_job_%a-%j.out_0'
balance00100_reward_policy_obs2_gamma99995 = '/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma99995_0d100i_1753349618.6116135/july22_obsv2_1e-5lr_0.1cp_gamma99995_0d100i_job_%a-%j.out_0'



# July 16th
balance7525_reward_policy_obs2 = "/Users/dahu1128/rllib_results/july_results/july16rllib_results/july16_1e-5lr_002torque_75d25i_reward_new_penalties_smallest_storage_Polaris_simulation_1752742415.630985/1e-5lr_002torque_75d25i_reward_new_penalties_smallest_storage_0"
balance5050_reward_policy_obs15 = "/Users/dahu1128/rllib_results/july_results/july5rllib_results/july5_1e-5lr_0005torque_50d50i_reward_new_penalties_smallest_storage_Polaris_simulation_1751885868.6929862/1e-5lr_0005torque_50d50i_reward_new_penalties_smallest_storage_0"
balance5050_reward_policy_obs2 ="/Users/dahu1128/rllib_results/july_results/july16rllib_results/july16_1e-5lr_002torque_50d50i_reward_new_penalties_smallest_storage_Polaris_simulation_1752838082.5491006/1e-5lr_002torque_50d50i_reward_new_penalties_smallest_storage_0"
# downlink_reward_with_eclipse_policy = "/Users/dahu1128/rllib_results/june29rllib_results/june29_lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1751336442.6073096/lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_0"

# july10th
cluster_6040policy_obs11 = "/Users/dahu1128/rllib_results/july_results/july10rllib_results/july10_1e-5lr_002torque_60d40i_reward_new_penalties_smallest_storage_Polaris_simulation_1752241793.6701877/1e-5lr_002torque_60d40i_reward_new_penalties_smallest_storage_0"

# July 5th
cluster_5050policy_5e6lr_obs11 = "/Users/dahu1128/rllib_results/july_results/july5rllib_results/july5_5e-6lr_0005torque_50d50i_reward_new_penalties_smallest_storage_Polaris_simulation_1751892467.767332/5e-6lr_0005torque_50d50i_reward_new_penalties_smallest_storage_0"
cluster_5050policy_5e5lr_obs11 ="/Users/dahu1128/rllib_results/july_results/july5rllib_results/july5_5e-5lr_0005torque_50d50i_reward_new_penalties_smallest_storage_Polaris_simulation_1751887414.367687/5e-5lr_0005torque_50d50i_reward_new_penalties_smallest_storage_0"

# June 28th
cluster_1000_policy_v1_obs1 = "/Users/dahu1128/rllib_results/june_results/june29rllib_results/june29_lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1751349196.029947/lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_0"
cluster_1000_policy_v2_obs1 = "/Users/dahu1128/rllib_results/june_results/june29rllib_results/june29_lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1751336442.6073096/lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_0"

# June 23rd
cluster_1000_policy_v2_obs1_wGAE ="/Users/dahu1128/rllib_results/june_results/june23rllib_results/lowBaudRate_5e-6lr_001torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1750740679.4746056/lowBaudRate_5e-6lr_001torque_downlink_reward_new_penalties_smallest_storage_0"

policy_path = cluster_1000_policy_v2_obs1_wGAE

# Define all known policy paths with associated obs values
policy_obs_map = {
    "balance75d25i_nopenalties_lowICbattery_woGAE_obs2_gamma95": 2,
    "balance75d25i_nopenalties_woGAE_obsv2_gamma95": 2,
    "balance75d25i_reward_policy_woGAE_obs2_gamma95": 2,
    "balance00100_reward_policy_obs1_gamma99": 1,
    "balance00100_reward_policy_obs1_gamma995": 1,
    "balance00100_reward_policy_obs1_gamma9995": 1,
    "balance00100_reward_policy_obs1_gamma99995": 1,
    "balance00100_reward_policy_obs2_gamma99": 2,
    "balance00100_reward_policy_obs2_gamma995": 2,
    "balance00100_reward_policy_obs2_gamma9995": 2,
    "balance00100_reward_policy_obs2_gamma99995": 2,
    "cluster_5050policy_obs2": 2,
    "cluster_7525policy_obs2": 2,
    "cluster_9010policy_obs2": 2,
    "cluster_6040policy_obs11": 1.1,
    "cluster_5050policy_5e6lr_obs11": 1.1,
    "cluster_5050policy_5e5lr_obs11": 1.1,
    "cluster_1000_policy_v1_obs1": 1.1,
    "cluster_1000_policy_v2_obs1": 1.1,
    "cluster_1000_policy_v2_obs1_wGAE": 1,
    "downlink_reward_policy": 1,
    "downlink_reward_policy_shorter_imaging": 1,
    "cluster_policy": 1,
}

# Compare policy_path to known variables
for name, val in list(globals().items()):
    if isinstance(val, str) and val == policy_path and name in policy_obs_map:
        policy_name = name
        obs_v = policy_obs_map[name]
        break

# Load policy
# load_best=False
# load_smallest=False
# if load_best:
#     policy = Policy(policy_path,policy_mode ='best')
# elif load_smallest:
#     policy = Policy(policy_path,policy_mode ='smallest')
# else:
#     policy = Policy(policy_path,policy_mode ='latest')
policy_mode='latest'
if policy_mode =='best':
    policy = Policy(policy_path,policy_mode ='best')
elif policy_mode =='smallest':
    policy = Policy(policy_path,policy_mode ='smallest')
else:
    policy = Policy(policy_path,policy_mode ='latest')


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

# Storage
sat_args["dataStorageCapacity"] = 50 * 8e6 / 2 # bits
sat_args["storageInit"] = lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2
sat_args["instrumentBaudRate"] = 0.5 * 8e6
sat_args["transmitterBaudRate"] = -0.5 * 8e6

# Power
sat_args["batteryStorageCapacity"] = 500 * 3600  # W*s
sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.2, 0.4) * 500 * 3600
sat_args["basePowerDraw"] = -10.0  # W
sat_args["instrumentPowerDraw"] = -30.0  # W
sat_args["transmitterPowerDraw"] = -25.0  # W
sat_args["thrusterPowerDraw"] = -80.0  # W
sat_args["panelArea"] = 1 # m^2

# Attitude
sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.000, size=3)  # N*m
sat_args["maxWheelSpeed"] = 6000.0  # RPM
sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
sat_args["desatAttitude"] = "nadir"

# reward and penalty factors and eclipse thresholds
sat_args["downlink_bonus"] = 0.75
sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
sat_args["full_storage_penalty"] = -1
sat_args["low_battery_penalty"] = -1
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
    log_level="ERROR", #ERROR or DEBUG
    disable_env_checker=True,
    # max_step_duration=700,
)
seed_number=10
observation, info = env.reset(seed=seed_number) #5

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


# # Tracking structures
# charging_battery_levels = []
# downlink_storage_levels = []
# charging_eclipse_flags = []
# downlink_eclipse_flags = []
# imaging_logs = []

battery_levels = []
storage_levels = []
sim_times = []
num_imaged = []
num_downlinked = []
charging_times = []
downlink_times = []


SS1_reward=0
act_random=False
for target_id in range(n_targets*6 *100 ):
    simtime = env.simulator.sim_time
    print(f"\n SIMULATION TIME: {simtime} seconds and current reward: {SS1_reward}")
    # Use policy to determine action
    # obs_for_policy = flatten_to_single_ndarray(observation[sat.name]) # This is a dict (as your env uses `obs_type=dict`)
    # obs_flat = flatten_to_single_ndarray(env.observation_spaces[sat.name], observation[sat.name])

    # policy_action = policy.act(obs_for_policy)
    policy_action = policy.act(observation[sat.name])
    if isinstance(policy_action, np.ndarray):  # Handle vector action output (if needed)
        policy_action = policy_action.item()  # or do appropriate conversion if the policy returns a torch tensor



    # action_dict = {sat.name: target_id}  # Assign the main satellite to observe `target_idx` # sequentially observing each target
    action_dict = {sat.name: 0}  # Assign the closest target when the list is sorted by distance
    action_dict = {sat.name: policy_action}  # Assign the closest target when the list is sorted by distance
    if act_random:
        random_action = np.random.randint(0,13)
        action_dict = {sat.name: random_action}
        action_counts[random_action] += 1
    else:
        action_counts[policy_action] += 1
    if policy_action == 11:
        print('tasking DOWNLINKING now: at t=',simtime," and storage level --> "+str(env.satellites[0].dynamics.storage_level_fraction))
        downlink_times.append(env.simulator.sim_time)

    elif policy_action == 10:
        print('tasking CHARGING now: at t=',simtime," and battery level --> "+str(env.satellites[0].dynamics.battery_charge_fraction))
        charging_times.append(env.simulator.sim_time)
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

    battery_levels.append(env.satellites[0].dynamics.battery_charge_fraction)
    storage_levels.append(env.satellites[0].dynamics.storage_level_fraction)
    sim_times.append(env.simulator.sim_time)
    num_imaged.append(len(env.env.rewarder.imaged_illuminated))
    num_downlinked.append(env.env.rewarder.useful_downlinks)

    SS1_reward+=reward['SS1']
    print("storage_level", env.satellites[0].dynamics.storage_level)
    print("dynamics.storage_level_fraction", env.satellites[0].dynamics.storage_level_fraction)
    print("dynamics.battery_charge_fraction", env.satellites[0].dynamics.battery_charge_fraction)
    print("env.satellites[0].dynamics.wheel_speeds_fraction", env.satellites[0].dynamics.wheel_speeds_fraction)


    # print('truncated list: ', truncated)
    data_dict["sim_time"].append(env.simulator.sim_time)
    if any(truncated.values()) or any(terminated.values()):
        break

print("  Final data level:", observation)
print(f"final reward for SS1 {SS1_reward} should be the same as {env.env.rewarder.cum_reward['SS1']}")
print(f"and number of imaged targets {len(env.env.satellites[0].data_store.data.imaged)} out of those useful images were: {len(env.env.rewarder.imaged_illuminated)}")
print(f"Total downlinked {env.env.rewarder.total_downlinks} out of those useful downlinks were: {env.env.rewarder.useful_downlinks}")
# print(f"mean and std of chosen_target_azimuth {env.env.satellites[0].action_builder.action_spec[0].chosen_target_azimuth}")

SS1_actions_spec = env.satellites[0].action_builder.action_spec[0]
print(f"mean and std of chosen_target_azimuth: {np.mean(SS1_actions_spec.chosen_target_azimuth):.2f}, {np.std(SS1_actions_spec.chosen_target_azimuth):.2f}")
print(f"mean and std of chosen_target_elevation: {np.mean(SS1_actions_spec.chosen_target_elevation_angle):.2f}, {np.std(SS1_actions_spec.chosen_target_elevation_angle):.2f}")
print(f"mean and std of chosen_target_elevation_local: {np.mean(SS1_actions_spec.chosen_target_elevation_local):.2f}, {np.std(SS1_actions_spec.chosen_target_elevation_local):.2f}")
print(f"mean and std of chosen_target_distance: {np.mean(SS1_actions_spec.chosen_target_distance):.2f}, {np.std(SS1_actions_spec.chosen_target_distance):.2f}")
print(f"mean and std of initial angular error: {np.mean(SS1_actions_spec.initial_angular_error):.2f}, {np.std(SS1_actions_spec.initial_angular_error):.2f}")
print(f"mean and std of chosen_target_priority: {np.mean(SS1_actions_spec.chosen_target_priority):.2f}, {np.std(SS1_actions_spec.chosen_target_priority):.2f}")
print(f"fraction of targets that were illuminated: {np.mean(SS1_actions_spec.chosen_target_illumination_status):.2f}")
print(f"fraction of targets ever visible: {len(SS1_actions_spec.ever_visible)/n_targets:.2f}")
print(f"mean and std of rel pos in H-frame: {np.mean(SS1_actions_spec.chosen_target_rel_pos_H, axis=0)}, {np.std(SS1_actions_spec.chosen_target_rel_pos_H, axis=0)}")


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
print(f"Other Actions (10–12): {non_target_count} ({non_target_pct:.2f}%)")
print(f"Downlink actions: {downlink_action_count}")
print(f"Charge actions: {charge_action_count}")
print(f"Desat actions: {desat_action_count}")
print("======================================\n")

# Plot 1: Absolute Action Counts
fig1 = plt.figure(figsize=(10, 5))
plt.bar(action_labels, counts, color="skyblue")
plt.title("Action Count Distribution")
plt.ylabel("Number of Times Action Was Taken")
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
save_plot_unique(fig1, f"random{act_random}_seed{seed_number}_{policy_name}_action_count_distribution")
plt.show()

# Plot 2: Action Percentages
fig2 = plt.figure(figsize=(10, 5))
plt.bar(action_labels, percentages, color="mediumseagreen")
plt.title("Action Percentage Distribution")
plt.ylabel("Percentage of Total Actions (%)")
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
save_plot_unique(fig2, f"random{act_random}_seed{seed_number}_{policy_name}_action_percentage_distribution")
plt.show()

# Plot 3: metrics over time:
fig3, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(sim_times, battery_levels, label='Battery Level', color='tab:blue')
ax1.plot(sim_times, storage_levels, label='Storage Level', color='tab:orange')
ax1.set_xlabel("Simulation Time (s)")
ax1.set_ylabel("Battery and Storage Fraction")
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

for t in charging_times:
    ax1.axvline(t, color='deepskyblue', linestyle='--', linewidth=0.8, alpha=0.85, label='Charge' if t == charging_times[0] else "")
for t in downlink_times:
    ax1.axvline(t, color='magenta', linestyle='--', linewidth=0.8, alpha=0.6, label='Downlink' if t == downlink_times[0] else "")

ax2 = ax1.twinx()
ax2.plot(sim_times, num_imaged, label='Cumulative Imaged Targets', color='tab:green')
ax2.plot(sim_times, num_downlinked, label='Cumulative Downlinked Targets', color='tab:red')
ax2.set_ylabel("Cumulative Count")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title("Battery, Storage, Imaging and Action Events Over Time")
plt.tight_layout()
save_plot_unique(fig3, f"random{act_random}_seed{seed_number}_{policy_name}_battery_storage_imaging_over_time")
plt.show()

# Plot 4 Azimuth and Elevation angle (deg) vs time
# Convert time from seconds to minutes
imaging_times_min = np.array(SS1_actions_spec.imaging_times) / 60.0
azimuths = np.array(SS1_actions_spec.chosen_target_azimuth)
elevations = np.array(SS1_actions_spec.chosen_target_elevation_angle)
fig4, ax1 = plt.subplots(figsize=(10, 5))
color1 = 'tab:blue'
ax1.set_xlabel('Time [min]')
ax1.set_ylabel('Azimuth [deg]', color=color1)
ax1.plot(imaging_times_min, azimuths, 'o-', color=color1, label='Azimuth')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Elevation [deg]', color=color2)
ax2.plot(imaging_times_min, elevations, 'x--', color=color2, label='Elevation')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Pointing Directions During Imaging Over 60 Minutes')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
save_plot_unique(fig4, f"random{act_random}_seed{seed_number}_{policy_name}_azimuth_and_elevation_pointing_over_time")
plt.show()



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