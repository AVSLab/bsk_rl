import gymnasium as gym
from requests.packages import target
import time

start_time = time.time()

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


n_targets = 100
n_targets_ahead = 10
total_time = n_targets * 450  # 2100 # 5700.0  # approximately 1 orbit
imaging_duration = 300
seed_number = 19
eclipse_norm = 5700
save_data = False #set to False to avoid saving data
use_shield = True
use_heuristic = True
heuristic_mode = "distance"
act_random = False
policy_mode = "HEURISTIC"
obs_v = 2 # this is overwritten for all policies that have an assigned obs type
ORBIT_PERIOD_SEC = 95 * 60
ECL_SLICE  = slice(75, 77)
GS_START   = 77
N_GS       = 5          # <<< set this to your actual number of ground stations
PAIR_STRIDE = 2
GS_NORMALIZED = True    # set True if GS values are normalized offsets; False if absolute times


class MyScanningSatellite(sats.AccessSatellite):
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
        obs.Eclipse(norm=5700.0),
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
sat_args["dataStorageCapacity"] = 50 * 8e6 / 2  # bits
sat_args["storageInit"] = lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2
sat_args["instrumentBaudRate"] = 0.5 * 8e6
sat_args["transmitterBaudRate"] = -0.5 * 8e6

# Power
sat_args["batteryStorageCapacity"] = 500 * 3600 # W*s
sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.32, 0.32) * 500 * 3600
sat_args["basePowerDraw"] = -10.0  # W
sat_args["instrumentPowerDraw"] = -30.0  # W
sat_args["transmitterPowerDraw"] = -25.0  # W
sat_args["thrusterPowerDraw"] = -80.0  # W
sat_args["panelArea"] = 1.0  # m^2

# Attitude
sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.001, size=3)  # N*m
sat_args["maxWheelSpeed"] = 6000.0  # RPM
sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
sat_args["desatAttitude"] = "sun"

# reward bonuses and eclipse thresholds
sat_args["downlink_bonus"] = 0.0
sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
sat_args["eclipse_threshold_for_imaging"] = 0.5
sat_args["eclipse_threshold_for_reward"] = sat_args["eclipse_threshold_for_imaging"]
sat_args["use_heuristic"]=use_heuristic
sat_args["heuristic_mode"]=heuristic_mode

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
    rLEO = 6871. * 1000  #7000 * 1000   # Minimum semi-major axis (LEO) in meters
    rUpperLEO = 8371.0 * 1000  #8371.   # max semi-major axis  of upper LEO in meters
    # rGEO = 42164. * 1000   # Maximum semi-major axis (GEO) in meters


    oe = orbitalMotion.ClassicElements()
    # oe.a = np.random.uniform(rLEO*5, rGEO)  # Random semi-major axis between LEO and GEO
    # oe.a = np.random.uniform(1.05*rLEO, rUpperLEO)  # Random semi-major axis between LEO and GEO
    oe.a = np.random.uniform(1.00*rLEO, rUpperLEO)  # Random semi-major axis between LEO and GEO
    if oe.a < 2*rLEO:
        oe.e = np.random.uniform(0.0, 0.02)    # Random eccentricity (allowing less elliptical orbits when near LEO)
        while oe.a*(1-oe.e) < 6771. * 1000: # perigee must be at least 400 km altitude
            oe.e = np.random.uniform(0.0, 0.02)
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

targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)]

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

observation, info = env.reset(seed=seed_number)

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
critical_storage_level = 0.99 # only task downlink if available storage_fraction is less than 0.05
critical_battery_level = 0.2 # always task charging if battery is below 0.3

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
eclipse_status = []
desat_times = []
# Ground-station windows: dict[str, list[np.ndarray([open_abs, close_abs])]]
ground_station_windows = {}         # e.g., {'ground_station_0': [np.array([t_open, t_close]), ...], ...}
_last_gs_pair = {}                  # e.g., {'ground_station_0': np.array([last_open, last_close])}

# Eclipse windows (from observation 'eclipse' field)
eclipse_windows = []                # list of np.array([open_abs, close_abs]) for the next eclipse window
_last_eclipse_pair = None           # np.array([last_open, last_close])


SS1_reward=0
SS1_reward_over_time = []

for target_id in range(n_targets*4 *100 ):
    simtime = env.simulator.sim_time
    print(f"\n SIMULATION TIME: {simtime} seconds and current reward: {SS1_reward}")

    # action_dict = {sat.name: target_id}  # Assign the main satellite to observe `target_idx` # sequentially observing each target
    chosen_action_id = 0
    action_dict = {sat.name: chosen_action_id}  # Assign the closest target when the list is sorted by distance
    if use_shield == True:
        if env.satellites[0].dynamics.storage_level_fraction > critical_storage_level:  # downlink if storage is more than 0.95
            print('tasking DOWNLINKING now: at t=',simtime," and storage level --> "+str(env.satellites[0].dynamics.storage_level_fraction))
            chosen_action_id = 11
            action_dict = {sat.name: chosen_action_id} # tasking downlink
            last_downlink_time = simtime
            downlink_times.append(env.simulator.sim_time)

        if env.satellites[0].dynamics.battery_charge_fraction < critical_battery_level:  # charge if battery is less than 0.05
            print('tasking CHARGING now: at t=',simtime," and battery level --> "+str(env.satellites[0].dynamics.battery_charge_fraction))
            chosen_action_id = 10
            action_dict = {sat.name: chosen_action_id} # tasking charging
            charging_times.append(env.simulator.sim_time)
    action_counts[chosen_action_id] += 1
    action_dict.update({targets[j].name: 0 for j in range(n_targets)})  # Initialize all targets to 0
    # print('current action_dict to be executed', action_dict)
    print('current action_dict to be executed', action_dict['SS1'], "eclipse status of SS1:",env.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[env.satellites[0].dynamics.eclipse_index].read().shadowFactor)
    eclipse_status.append(env.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[env.satellites[0].dynamics.eclipse_index].read().shadowFactor)

    observation, reward, terminated, truncated, info = env.step(action=action_dict)

    battery_levels.append(env.satellites[0].dynamics.battery_charge_fraction)
    storage_levels.append(env.satellites[0].dynamics.storage_level_fraction)
    sim_times.append(env.simulator.sim_time)
    num_imaged.append(len(env.env.rewarder.imaged_illuminated))
    num_downlinked.append(env.env.rewarder.useful_downlinks)

    SS1_reward+=reward['SS1']
    SS1_reward_over_time.append(SS1_reward)
    print("storage_level", env.satellites[0].dynamics.storage_level)
    print("dynamics.storage_level_fraction", env.satellites[0].dynamics.storage_level_fraction)
    print("dynamics.battery_charge_fraction", env.satellites[0].dynamics.battery_charge_fraction)
    print("env.satellites[0].dynamics.wheel_speeds_fraction", env.satellites[0].dynamics.wheel_speeds_fraction)

    # print('truncated list: ', truncated)
    data_dict["sim_time"].append(env.simulator.sim_time)
    if all(truncated.values()) or all(terminated.values()) or len(env.env.satellites[0].data_store.data.imaged)==100:
        if len(env.env.satellites[0].data_store.data.imaged) == 100:
            print('ALL targets imaged!')
        else:
            print("Scanning Satellite is dead or episode terminated!")
        break
        # --- Ground stations from flat obs ---
    # Build/extend per-station lists in ground_station_windows dict
    else:
        if 'ground_station_windows' not in globals():
            ground_station_windows = {}
        if '_last_gs_pair' not in globals():
            _last_gs_pair = {}

        for i in range(N_GS):
            s = GS_START + i * PAIR_STRIDE
            pair = np.asarray(observation[sat.name][s:s+PAIR_STRIDE], dtype=float)  # [open_i, close_i]
            if pair.size < 2:
                continue  # robust against short obs
            pair_abs = (pair * ORBIT_PERIOD_SEC + simtime) if GS_NORMALIZED else pair

            gs_name = f"ground_station_{i}"
            if gs_name not in ground_station_windows:
                ground_station_windows[gs_name] = []
                _last_gs_pair[gs_name] = None

            # De-dup per station with 10 s tolerance
            if (_last_gs_pair[gs_name] is None) or (not np.allclose(_last_gs_pair[gs_name], pair_abs, atol=10.0)):
                ground_station_windows[gs_name].append(pair_abs.copy())
                _last_gs_pair[gs_name] = pair_abs.copy()

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

print("Target Selection comparison:", env.satellites[0].dynamics.target_selection_comparison)
# Count only the matches (non-False entries)
num_same = np.count_nonzero(np.array(env.satellites[0].dynamics.target_selection_comparison) != False)
print("Target Selection comparison numbers:", num_same)
num_diff = np.count_nonzero(np.array(env.satellites[0].dynamics.target_selection_comparison) == False)
print("Number different:", num_diff)


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
# desat_action_count = counts[12]
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
# print(f"Desat actions: {desat_action_count}")
print("======================================\n")

# Combined plot 1
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot absolute counts on the left y-axis
bars1 = ax1.bar(action_labels, counts, color="skyblue", label="Action Count")
ax1.set_ylabel("Number of Times Action Was Taken", color="skyblue")
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_xticks(range(len(action_labels)))
ax1.set_xticklabels(action_labels, rotation=45)

# Create a second y-axis for the percentages
ax2 = ax1.twinx()
bars2 = ax2.bar(action_labels, percentages, color="mediumseagreen", alpha=0.000, label="Action Percentage")
ax2.set_ylabel("Percentage of Total Actions (%)", color="mediumseagreen")
ax2.tick_params(axis='y', labelcolor="black")

# Grid on primary y-axis only
ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

# Add a combined legend
lines1, labels1 = bars1, ["Count"]
lines2, labels2 = bars2, ["Percentage"]
# ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Action Distribution: Count and Percentage")
plt.tight_layout()

# Save and show
if use_heuristic and use_shield:
    save_plot_unique(fig, f"seed{seed_number}_heuristic_policy_wShield_action_distribution_combined")
elif use_heuristic and not use_shield:
    save_plot_unique(fig, f"seed{seed_number}_heuristic_policy_NOShield_action_distribution_combined")
# else:
#     save_plot_unique(fig, f"seed{seed_number}_{policy_mode}_{policy_name}_action_distribution_combined")
plt.show()

# --- Compute eclipse & penumbra spans from shadowFactor ---
# shadowFactor: 1 = lit, 0 < sf < 1 = penumbra, 0 = umbra
umbra_spans = []
penumbra_spans = []

if eclipse_status and sim_times and len(eclipse_status) == len(sim_times):
    def _val(x):
        try: return float(getattr(x, "shadowFactor", x))
        except Exception: return 1.0

    def is_umbra(v):    return np.isclose(v, 0.0, atol=1e-6)
    def is_lit(v):      return v >= 1.0 - 1e-6
    def is_penumbra(v): return (v > 0.0 + 1e-6) and (v < 1.0 - 1e-6)

    in_umbra = False; t0_umb = None
    in_penu  = False; t0_pen = None

    for t, s in zip(sim_times, eclipse_status):
        v = _val(s)

        if is_umbra(v):
            # close penumbra if transitioning into umbra
            if in_penu:
                penumbra_spans.append((t0_pen, t))
                in_penu, t0_pen = False, None
            # open/continue umbra
            if not in_umbra:
                in_umbra, t0_umb = True, t

        elif is_penumbra(v):
            # close umbra if leaving umbra
            if in_umbra:
                umbra_spans.append((t0_umb, t))
                in_umbra, t0_umb = False, None
            # open/continue penumbra
            if not in_penu:
                in_penu, t0_pen = True, t

        else:  # lit
            if in_umbra:
                umbra_spans.append((t0_umb, t))
                in_umbra, t0_umb = False, None
            if in_penu:
                penumbra_spans.append((t0_pen, t))
                in_penu, t0_pen = False, None

    # close any open span at the end
    t_last = sim_times[-1]
    if in_umbra and t0_umb is not None:
        umbra_spans.append((t0_umb, t_last))
    if in_penu and t0_pen is not None:
        penumbra_spans.append((t0_pen, t_last))
else:
    umbra_spans, penumbra_spans = [], []

# Plot 2: metrics over time + cumulative reward
# (Uses EXTRACTED + MERGED window spans for eclipse and ground stations)

# ---- helpers to ensure we have merged spans ----
def _merge_spans(spans, gap_tol=5.0):
    """Merge overlapping or nearly-adjacent [t0, t1] spans (seconds)."""
    if not spans:
        return []
    spans = sorted((float(a), float(b)) for (a, b) in spans)
    merged = []
    s0, e0 = spans[0]
    for s, e in spans[1:]:
        if s <= e0 + gap_tol:
            e0 = max(e0, e)
        else:
            merged.append((s0, e0))
            s0, e0 = s, e
    merged.append((s0, e0))
    return merged

# Build gs_spans_merged (dict) if not already available
if 'gs_spans_merged' not in locals() and 'gs_spans_merged' not in globals():
    gs_spans_merged = {}
    if 'ground_station_windows' in locals() or 'ground_station_windows' in globals():
        for _gs, _spans in (ground_station_windows or {}).items():
            gs_spans_merged[_gs] = _merge_spans([(float(a), float(b)) for (a, b) in _spans], gap_tol=5.0)

# ---- plotting ----
fig3, ax1 = plt.subplots(figsize=(12, 6))

# Shading (all with consistent alphas; overlaps will naturally darken)
penumbra_alpha = 0.15    # eclipse window shading
gs_alpha       = 0.12    # GS window shading

# # Shade umbra (darker) and penumbra (lighter)
umbra_alpha = 0.35       # full eclipse (no charging)
penumbra_alpha = 0.15    # some light; charging may be possible
_first_u, _first_p = True, True
for (t0, t1) in umbra_spans:
    ax1.fill_between([t0, t1], 0, 1, transform=ax1.get_xaxis_transform(),
                     color='grey', alpha=umbra_alpha, zorder=0,
                     label='Umbra (full eclipse)' if _first_u else '')
    _first_u = False
# for (t0, t1) in penumbra_spans:
#     ax1.fill_between([t0, t1], 0, 1, transform=ax1.get_xaxis_transform(),
#                      color='grey', alpha=penumbra_alpha, zorder=0,
#                      label='Penumbra' if _first_p else '')
#     _first_p = False

# Ground station windows (merged, flattened; single alpha for all)
_first_gs_lbl = True
_all_gs_spans = [span for spans in (gs_spans_merged or {}).values() for span in spans]
for (t0, t1) in _all_gs_spans:
    t0, t1 = float(t0), float(t1)
    ax1.fill_between([t0, t1], 0, 1, transform=ax1.get_xaxis_transform(),
                     color='green', alpha=gs_alpha, #zorder=0.03,
                     label='GS window' if _first_gs_lbl else '')
    _first_gs_lbl = False

# Battery & storage on left y-axis
ax1.plot(sim_times, battery_levels, label='Battery Level', color='tab:blue')
ax1.plot(sim_times, storage_levels, label='Storage Level', color='tab:orange')
ax1.set_xlabel("Simulation Time (s)")
ax1.set_ylabel("Battery and Storage Fraction", color='black')
ax1.set_ylim(0, 1.05)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='-.', alpha=0.4)

# X-limit at 45,000 s
ax1.set_xlim(0, 45000)

# Mark charging and downlink events
if charging_times:
    ax1.axvline(charging_times[0], color='deepskyblue', linestyle='--', linewidth=0.8, alpha=0.85, label='Charge')
    for t in charging_times[1:]:
        ax1.axvline(t, color='deepskyblue', linestyle='--', linewidth=0.8, alpha=0.85)
if downlink_times:
    ax1.axvline(downlink_times[0], color='magenta', linestyle='--', linewidth=0.8, alpha=0.6, label='Downlink')
    for t in downlink_times[1:]:
        ax1.axvline(t, color='magenta', linestyle='--', linewidth=0.8, alpha=0.6)

# Create second y-axis for cumulative reward and target counts
ax2 = ax1.twinx()
ax2.plot(sim_times, num_imaged, label='Cumulative Imaged Targets', color='tab:green')

# Mark DESAT events
if desat_times:
    ax1.axvline(desat_times[0], color='crimson', linestyle='--', linewidth=1.2, alpha=0.85, label='Desat')
    for t in desat_times[1:]:
        ax1.axvline(t, color='crimson', linestyle='--', linewidth=1.2, alpha=0.85)

ax2.plot(sim_times, num_downlinked, label='Cumulative Downlinked Targets', color='tab:red')
# ax2.plot(sim_times, SS1_reward_over_time, label='Cumulative SS1 Reward', linestyle=':', linewidth=3.0, color='tab:purple')
ax2.set_ylabel("Cumulative Count", color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Align both y-axes at 0 and 1.0/100 respectively
ax1.set_ylim(top=1.0, bottom=0.0)
ax2.set_ylim(top=100, bottom=0.0)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Battery, Storage, Reward, Imaging and Action Events Over Time")
plt.tight_layout()
# Save and show
if use_heuristic and use_shield:
    save_plot_unique(fig3, f"seed{seed_number}_heuristic_policy_wShield_battery_storage_reward_over_time")
elif use_heuristic and not use_shield:
    save_plot_unique(fig3, f"seed{seed_number}_heuristic_policy_NOShield_battery_storage_reward_over_time")
# else:
    # save_plot_unique(fig3, f"seed{seed_number}_{policy_mode}_{policy_name}_battery_storage_reward_over_time")
plt.show()


# Plot 3 Azimuth and Elevation angle (deg) vs time
# (minutes on x-axis; same merged shading converted to minutes)
minute = 1.0
imaging_times_min = np.array(SS1_actions_spec.imaging_times) / minute
azimuths = np.array(SS1_actions_spec.chosen_target_azimuth)
elevations = np.array(SS1_actions_spec.chosen_target_elevation_angle)
fig4, ax1 = plt.subplots(figsize=(10, 5))

# Shading from merged windows (convert s→min)
penumbra_alpha = 0.15    # some light; charging may be possible
umbra_alpha = 0.35       # full eclipse (no charging)
gs_alpha       = 0.12

# Shade umbra/penumbra (convert s→min for x)
_first_u3, _first_p3 = True, True
for (t0, t1) in umbra_spans:
    ax1.fill_between([t0/minute, t1/minute], 0, 1, transform=ax1.get_xaxis_transform(),
                     color='grey', alpha=umbra_alpha, zorder=0,
                     label='Umbra (full eclipse)' if _first_u3 else '')
    _first_u3 = False
# for (t0, t1) in penumbra_spans:
#     ax1.fill_between([t0/60.0, t1/60.0], 0, 1, transform=ax1.get_xaxis_transform(),
#                      color='grey', alpha=penumbra_alpha, zorder=0,
#                      label='Penumbra' if _first_p3 else '')
#     _first_p3 = False
#
_first_gs_lbl3 = True
_all_gs_spans = [span for spans in (gs_spans_merged or {}).values() for span in spans]
for (t0, t1) in _all_gs_spans:
    t0m, t1m = float(t0)/minute, float(t1)/minute
    ax1.fill_between([t0m, t1m], 0, 1, transform=ax1.get_xaxis_transform(),
                     color='green', alpha=gs_alpha, #zorder=0.03,
                     label='GS window' if _first_gs_lbl3 else '')
    _first_gs_lbl3 = False

# Azimuth/Elevation
color1 = 'tab:blue'
if minute <60.0:
    ax1.set_xlabel('Time [sec]')
else:
    ax1.set_xlabel('Time [min]')
ax1.set_ylabel('Azimuth [deg]', color=color1)
ax1.plot(imaging_times_min, azimuths, 'o-', color=color1, label='Azimuth')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='-.', color='0.65')

# X-limit at 45,000 s equivalent
ax1.set_xlim(0, 45000/minute)

ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Elevation [deg]', color=color2)
ax2.plot(imaging_times_min, elevations, 'x--', color=color2, label='Elevation')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Pointing Directions During Imaging Over 60 Minutes')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
if use_heuristic and use_shield:
    save_plot_unique(fig4, f"seed{seed_number}_heuristic_policy_wShield_azimuth_and_elevation_pointing_over_time")
elif use_heuristic and not use_shield:
    save_plot_unique(fig4, f"seed{seed_number}_heuristic_policy_NOShield_azimuth_and_elevation_pointing_over_time")
# else:
    # save_plot_unique(fig4, f"seed{seed_number}_{policy_mode}_{policy_name}_azimuth_and_elevation_pointing_over_time")
plt.show()

SS1_actions_spec = env.satellites[0].action_builder.action_spec[0]
fields = [
    "chosen_target_azimuth",
    "chosen_target_elevation_angle",
    "chosen_target_elevation_local",
    "chosen_target_distance",
    "initial_angular_error",
    "chosen_target_priority",
    "chosen_target_illumination_status",
]

n_parts = 3

for field in fields:
    data = np.array(getattr(SS1_actions_spec, field))  # Convert to NumPy array
    total_len = len(data)
    indices = np.array_split(np.arange(total_len), n_parts)

    print(f"\n--- {field} ---")
    for i, idx in enumerate(indices):
        segment = data[idx]
        if field == "chosen_target_illumination_status":
            print(f"  Segment {i+1}/{n_parts}: illuminated fraction = {np.mean(segment):.3f}")
        else:
            mean_val = np.mean(segment)
            std_val = np.std(segment)
            print(f"  Segment {i+1}/{n_parts}: mean = {mean_val:.2f}, std = {std_val:.2f}")

if save_data:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for key, value in data_dict.items():
        if isinstance(value, dict):  # Save per-target data separately
            for target_name, target_data in value.items():
                np.save(os.path.join(data_dir, f"{key}_{target_name}.npy"), np.array(target_data))
        else:
            np.save(os.path.join(data_dir, f"{key}.npy"), np.array(value))


    # Also save high-level time series useful for analysis
    np.save(os.path.join(data_dir, "sim_times.npy"), np.array(sim_times))
    np.save(os.path.join(data_dir, "battery_levels.npy"), np.array(battery_levels))
    np.save(os.path.join(data_dir, "storage_levels.npy"), np.array(storage_levels))
    np.save(os.path.join(data_dir, "eclipse_status.npy"), np.array(eclipse_status))
    np.save(os.path.join(data_dir, "charging_times.npy"), np.array(charging_times))
    np.save(os.path.join(data_dir, "downlink_times.npy"), np.array(downlink_times))
    np.save(os.path.join(data_dir, "desat_times.npy"), np.array(desat_times))
    # Save extracted windows
    np.save(os.path.join(data_dir, "eclipse_windows.npy"), np.array(eclipse_windows, dtype=float))

    # # Ground-station windows: save each station separately
    # for gs_name, windows in ground_station_windows.items():
    #     np.save(os.path.join(data_dir, f"{gs_name}_windows.npy"), np.array(windows, dtype=float))
    print("Data saved successfully in 'data/' folder.")
else:
    print("Not saving data")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time:.4f} seconds")

data = {}
data["cumulativeRewardSS1"]=env.rewarder.cum_reward['SS1']
data["illuminated_images"] = len(env.rewarder.imaged_illuminated)
# data["Total Images Downlinked"] = env.satellites[0].dynamics.total_downlinks
# data["Useful Images Downlinked"] = env.satellites[0].dynamics.useful_downlinks

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

print("ALL DATA: ", data)


# Save metrics dictionary to JSON
try:
    import json
    json_path = os.path.join(data_dir, f"metrics_random{act_random}_seed{seed_number}_{policy_mode}_{policy_name}.json")
    def _convert(o):
        import numpy as _np
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        return o
    # Add commonly printed summary fields if present
    summary = {
        "target_imaging_count": 'target_imaging_count' in locals() and target_imaging_count or None,
        "non_target_count": 'non_target_count' in locals() and non_target_count or None,
        "charge_action_count": 'charge_action_count' in locals() and charge_action_count or None,
        "downlink_action_count": 'downlink_action_count' in locals() and downlink_action_count or None,
        # "desat_action_count": 'desat_action_count' in locals() and desat_action_count or None,
        "target_imaging_pct": 'target_imaging_pct' in locals() and target_imaging_pct or None,
        "non_target_pct": 'non_target_pct' in locals() and non_target_pct or None,
        "imaging_success_percentage": 'env' in locals() and len(env.rewarder.imaged_illuminated)/target_imaging_count*100 if ('env' in locals() and target_imaging_count) else None
    }
    payload = {"data": data, "summary": summary}
    with open(json_path, "w") as jf:
        json.dump(payload, jf, indent=2, default=_convert)
    print(f"Saved metrics JSON to {json_path}")
except Exception as e:
    print("WARNING: Failed to save metrics JSON:", e)
print(f"good images #:{len(env.rewarder.imaged_illuminated)} out of {target_imaging_count}")
print(f"imaging success percentage {len(env.rewarder.imaged_illuminated)/target_imaging_count*100}%")
