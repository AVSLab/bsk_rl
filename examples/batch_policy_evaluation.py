#!/usr/bin/env python3
"""
batch_policy_evaluation.py

Scan a directory for policies (optionally filtered by substring),
run multiple trials per policy, collect statistics, save results and plots.

Usage:
    - Edit POLICY_FOLDER (or pass via CLI if you want to extend)
    - Set FILTER_SUBSTRING to e.g. "aug1" to only test aug1 policies
    - Set N_RUNS to the number of independent trials per policy
"""

import os
from pathlib import Path
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import shutil
from datetime import datetime
import re

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")

# ---- imports from your environment (same as in your uploaded script) ----
import gymnasium as gym
from bsk_rl import act, data, obs, scene, sats, utils
from bsk_rl.sim import dyn, fsw, world
from Basilisk.utilities import macros, orbitalMotion
from examples.load_policy import load_policy
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# Make Basilisk not unload kernels on teardown (prevents CSPICE hang)
try:
    from Basilisk.utilities import simIncludeGravBody as _sigb
    def _no_unload(self):  # keep signature (self)
        return
    _sigb.gravBodyFactory.unloadSpiceKernels = _no_unload
    print("[SPICE] Unload disabled (kernels kept loaded for process lifetime).")
except Exception as _e:
    pass
# -------------------------------------------------------------------------

# ---------- USER CONFIG ----------
# POLICY_FOLDER = Path("/Users/dahu1128/rllib_results/july_results/july30rllib_results")  # Set to a local folder containing trained policies.
# POLICY_FOLDER = Path("/Users/dahu1128/rllib_results/august_results/aug1rllib_results")  # Set to a local folder containing trained policies.
POLICY_FOLDER = Path("/Users/dahu1128/rllib_results/august_results/aug18rllib_results")  # Set to a local folder containing trained policies.
# POLICY_FOLDER = Path("/Users/dahu1128/rllib_results/august_results/aug19rllib_results")  # Set to a local folder containing trained policies.

# POLICY_FOLDER = Path("/Users/dahu1128/rllib_results/august_results/locally_trained")  # Set to a local folder containing trained policies.
# POLICY_FOLDER = Path("/Users/dahu1128/rllib_results")  # Set to a local folder containing trained policies.
# FILTER_SUBSTRING = "aug13_wGAE_smallbatch"   # only load policies that contain this substring; set to "" to load all
# FILTER_SUBSTRING = "aug15_wGAE_imaging_baseline"
# FILTER_SUBSTRING = "aug3_unlimitedResources"
# FILTER_SUBSTRING = "aug5_unlimitedResources"
# FILTER_SUBSTRING = "aug1_nopenalties_woGAE_unlimitedResources"
# FILTER_SUBSTRING = "aug1_nopenalties_wGAE_unlimitedResources_obsv2"
# FILTER_SUBSTRING = "aug1_nopenalties_wGAE_restricted_Resources_obsv2"
# FILTER_SUBSTRING = "aug19_justimaging_obsv5_1e-5lr_batch3200_gamma9995_0d100i"
FILTER_SUBSTRING = "aug18_restrictedResources_obsv7_1e-5lr_batch3200_gamma9995_0d100i"



POLICY_MATCH_SUFFIX = ".out_0"  # typical suffix in your screenshot; adjust if different
N_RUNS = 10   # number of independent trials per policy (set X here)
use_shield = True
just_imaging = False
POLICY_LOAD_MODE = "best"  # 'latest'|'best'|'smallest'
SAVE_DIR = Path("batch_results")
SAVE_DIR.mkdir(exist_ok=True)
SEED_START = 100  # base seed; subsequent runs use seed+run_idx
TIME_LIMIT = None  # optional override for env time_limit (None means use default from script)
# ----------------------------------------------------------------

# Map for obs_v overrides. If your aug1 set is always obs_v==2, we override for matching policies.
DEFAULT_POLICY_OBS_MAP = {
    "aug1_nopenalties_wGAE_unlimitedResources_obsv2": 2,
    "aug5_unlimitedResources": 2,
    "aug3_unlimitedResources": 2,
    "aug13_wGAE_smallbatch": 2,  # all aug1 policies use obs_v==2
    "aug15_wGAE_imaging_baseline": 5,
    "aug19_justimaging_obsv5_1e-5lr_batch3200_gamma9995_0d100i": 5,
    "aug18_restrictedResources_obsv7_1e-5lr_batch3200_gamma9995_0d100i": 7,
    # include other mappings if desired
}

# copy of your Policy loader wrapper but minimal
class PolicyWrapper:
    def __init__(self, policy_path, policy_mode='best', zero_element=None):
        self.zero_element = zero_element
        self.policy_function = None
        if policy_path is not None:
            self.policy_function = load_policy(str(policy_path), policy_mode=policy_mode)

    def act(self, observation):
        if self.zero_element is not None:
            observation[self.zero_element] = 0.0
        if self.policy_function is not None:
            return self.policy_function(observation)
        return None

# Helper to find policy folders/files in a directory
def find_policies(folder: Path, substring="", suffix=None):
    found = []
    for p in folder.iterdir():
        name = p.name
        if substring and substring not in name:
            continue
        if suffix and suffix not in name:
            # Also accept directories that contain files matching suffix
            # e.g. folder/aug1_.../...out_0 or folder/aug1...out_0
            if p.is_dir():
                matches = list(p.rglob(f"*{suffix}*"))
                if matches:
                    # prefer the matched file or folder; store the folder path
                    found.append(p)
                continue
            else:
                continue
        found.append(p)
    # deduplicate and sort
    unique = sorted({str(x): x for x in found}.values(), key=lambda x: x.name)
    return unique
from collections import Counter, namedtuple

# ---------- Utilities ----------
def _get_rewarder(env):
    # Some versions expose rewarder at env.rewarder, others at env.env.rewarder
    return getattr(env, "rewarder", getattr(env.env, "rewarder", None))

def _get_imaged_all(env):
    # robustly get "all imaged" list (not just illuminated)
    try:
        return env.env.satellites[0].data_store.data.imaged
    except AttributeError:
        # Fallbacks in case structure differs
        try:
            return env.satellites[0].data_store.data.imaged
        except Exception:
            return []

def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return int(x) if isinstance(x, (int, float)) else 0

# ---------- Metrics structure ----------
def initialize_run_stats():
    return {
        "sim_times": [],
        "battery_levels": [],
        "storage_levels": [],
        "num_imaged": [],                   # all imaged (raw)
        "num_imaged_illuminated": [],       # useful imaging signal (illuminated)
        "num_downlinked_total": [],         # total downlinks
        "num_downlinked_useful": [],        # useful downlinks
        "charging_events": [],
        "downlink_events": [],
        "desat_events": [],
        "policy_action_counts": Counter(),   # what policy requested
        "executed_action_counts": Counter(), # what actually executed (after shield)
        "cumulative_reward_over_time": [],
        # Shield logs
        "shield_interventions": [],          # list of dict logs below
        "shield_disagreements": [],          # list[bool] per intervention
    }

# ---------- Shield ----------
ShieldDecision = namedtuple(
    "ShieldDecision",
    ["action", "intervened", "disagreed", "reason"]
)

def apply_simple_shield(
    env,
    policy_action: int,
    critical_storage_level: float = 0.99,
    critical_battery_level: float = 0.20,
):
    """
    Minimal shield:
      - If storage >= critical_storage_level → force DOWNLINK (11)
      - If battery  <= critical_battery_level → force CHARGE (10)
    CHARGE has priority over DOWNLINK if both are critical simultaneously.
    """
    # Read current state
    try:
        storage_frac = env.satellites[0].dynamics.storage_level_fraction
        battery_frac = env.satellites[0].dynamics.battery_charge_fraction
    except Exception:
        # If anything is missing, do not intervene
        return ShieldDecision(policy_action, False, False, reason=None)

    forced_action = None
    reasons = []

    # Battery safety takes precedence
    if battery_frac <= critical_battery_level:
        forced_action = 10
        reasons.append(f"battery<=thr ({battery_frac:.3f}<={critical_battery_level:.3f})")

    # Storage safety second
    if storage_frac >= critical_storage_level:
        # only override if nothing more critical already forced
        if forced_action is None:
            forced_action = 11
        reasons.append(f"storage>=thr ({storage_frac:.3f}>={critical_storage_level:.3f})")

    if forced_action is None:
        return ShieldDecision(policy_action, False, False, reason=None)

    disagreed = (forced_action != int(policy_action))
    reason = " & ".join(reasons)
    return ShieldDecision(forced_action, True, disagreed, reason)


imaging_duration= 300
# Build the observation-spec / classes similar to your original script but in functions
# We'll create a small helper to instantiate env with required obs_v and seed
def make_env(obs_v, seed_number, total_time, n_targets=100, n_targets_ahead=10, imaging_duration=imaging_duration):
    # NOTE: To keep this function lean, we reuse the classes you already use in your uploaded file.
    # We'll replicate the minimal MyScanningSatellite and MyTargetSatellite logic inline.
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
        elif obs_v == 2:
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
                    dict(prop="target_distance", norm = 1596*1000),
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
            act.ImageRSO(n_ahead_image=n_targets_ahead, duration=imaging_duration),
            act.Charge(duration=300.0),
            act.Downlink(duration=180.0),
            act.Desat(duration=150),
        ]
        if obs_v==5:
            observation_spec = [
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=90.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm = 15960*1000),
                    dict(prop="angle_to_target", norm=90.0),
                    dict(prop="target_distance", norm = 15960*1000), #normalization calculated assuming h = 800 km and min elevation is -14 deg
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=n_targets_ahead,
                                               ),
                obs.Eclipse(norm=5700),
            ]
            action_spec = [
                act.ImageRSO(n_ahead_image=n_targets_ahead,duration=imaging_duration),  # Scan for 5 minute
                ]
        if obs_v==6:
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
                obs.Eclipse(norm=5700),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm = 5700.0),
                    dict(prop="opportunity_close", norm = 5700.0),
                    type="ground_station",
                    n_ahead_observe=1,
                )
            ]
            action_spec = [
                act.ImageRSO(n_ahead_image=n_targets_ahead,duration=imaging_duration),  # Scan for 5 minute
                act.Charge(duration=300.0),  # Charge for 5 minutes
                act.Downlink(duration=300.0), # Downlink for 3 min
                act.Desat(duration=150), # Desat for 2.5 min.

            ]
        elif obs_v==7:
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),

                    ),
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

        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    class MyTargetSatellite(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=total_time)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

    # target randomizer and sat_args copied from your script
    def custom_oe_randomizer():
        rLEO = 6871. * 1000  #7000 * 1000   # Minimum semi-major axis (LEO) in meters
        rUpperLEO =  8371.0 * 1000    # max semi-major axis  of upper LEO in meters
        # rGEO = 42164. * 1000   # Maximum semi-major axis (GEO) in meters
        oe = orbitalMotion.ClassicElements()
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
        return oe

    # sat_args (same values as your script)
    sat_args = {}
    sat_args["imageAttErrorRequirement"] = 0.01
    sat_args["imageRateErrorRequirement"] = 0.05
    sat_args["dataStorageCapacity"] = 50 * 8e6 / 2
    sat_args["storageInit"] = lambda: np.random.uniform(0.0, 0.0) * 50 * 8e6 / 2
    sat_args["instrumentBaudRate"] = 0.5 * 8e6
    sat_args["transmitterBaudRate"] = -0.5 * 8e6
    sat_args["batteryStorageCapacity"] = 500 * 3600
    sat_args["storedCharge_Init"] = lambda: np.random.uniform(0.3, 0.3) * 500 * 3600
    sat_args["basePowerDraw"] = -10.0
    sat_args["instrumentPowerDraw"] = -30.0
    sat_args["transmitterPowerDraw"] = -25.0
    sat_args["thrusterPowerDraw"] = -80.0
    sat_args["panelArea"] = 1
    sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.000, size=3)
    sat_args["maxWheelSpeed"] = 6000.0
    sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
    sat_args["desatAttitude"] = "sun"
    sat_args["downlink_bonus"] = 0.0
    sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
    sat_args["full_storage_penalty"] = 0
    sat_args["low_battery_penalty"] = 0
    sat_args["eclipse_threshold_for_imaging"] = 0.5
    sat_args["eclipse_threshold_for_reward"] = sat_args["eclipse_threshold_for_imaging"]
    if just_imaging:
        sat_args["dataStorageCapacity"] = 50 * 8e6 / 2 *1000000
        sat_args["batteryStorageCapacity"] = 500 * 3600 *1000000
        sat_args["storedCharge_Init"] = lambda: np.random.uniform(1.0, 1.0) * 500 * 3600 *1000000


    target_args=dict(oe=custom_oe_randomizer, batteryStorageCapacity = 1, storedCharge_Init = 0.0, basePowerDraw = -10000.0 )  # testing to see if sim is faster if the other agents are killed


    sat = MyScanningSatellite(name="SS1", sat_args=sat_args)
    targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)]
    all_sat = [sat] + targets

    env_kwargs = dict(
        satellites=all_sat,
        scenario=scene.RandomSatellites("SS1",n_targets=n_targets),
        rewarder=data.RSOTargetImageReward(),
        world_type=world.GroundStationWorldModel,
        time_limit=total_time,
        log_level="ERROR",
        disable_env_checker=True,
    )
    if TIME_LIMIT is not None:
        env_kwargs["time_limit"] = TIME_LIMIT

    env = gym.make("ConstellationTasking-v1", **env_kwargs)
    return env


# Single-run execution: returns run-level summary and arrays/dicts for saving
def run_single_trial(
    policy_wrapper,
    obs_v,
    seed_number,
    run_idx,
    total_time,
    n_targets=100,
    use_shield: bool = True,
    critical_storage_level: float = 0.99,
    critical_battery_level: float = 0.20,
):
    env = make_env(obs_v=obs_v, seed_number=seed_number, total_time=total_time, n_targets=n_targets)
    observation, info = env.reset(seed=seed_number)

    run_stats = initialize_run_stats()
    SS1_reward = 0.0

    max_steps = int((total_time / 1.0) * 10)  # conservative cap
    rewarder = _get_rewarder(env)

    for step in range(max_steps):
        simtime = env.simulator.sim_time
        print(f"\n SIMULATION TIME: {simtime} seconds and current reward: {SS1_reward}")

        # ----- POLICY CHOICE -----
        policy_action = policy_wrapper.act(observation["SS1"])
        if isinstance(policy_action, np.ndarray):
            try:
                policy_action = policy_action.item()
            except Exception:
                policy_action = int(policy_action[0])
        policy_action = int(policy_action)
        run_stats["policy_action_counts"].update([policy_action])

        # ----- SHIELD (optional) -----
        executed_action = policy_action
        shield_reason = None
        if use_shield:
            decision = apply_simple_shield(
                env,
                policy_action,
                critical_storage_level=critical_storage_level,
                critical_battery_level=critical_battery_level,
            )
            executed_action = decision.action
            if decision.intervened:
                # Log event
                run_stats["shield_interventions"].append({
                    "sim_time": simtime,
                    "policy_action": int(policy_action),
                    "shield_action": int(executed_action),
                    "reason": decision.reason,
                })
                run_stats["shield_disagreements"].append(bool(decision.disagreed))

        # ----- ACTION DICT (SS1 + passive targets=0) -----
        action_dict = {"SS1": int(executed_action)}
        action_dict.update({f"target_{j}": 0 for j in range(n_targets)})

        # event logs by executed action
        if executed_action == 11:
            print('tasking DOWNLINKING now: at t=',simtime," and storage level --> "+str(env.satellites[0].dynamics.storage_level_fraction))
            run_stats["downlink_events"].append(simtime)
        elif executed_action == 10:
            print('tasking CHARGING now: at t=',simtime," and battery level --> "+str(env.satellites[0].dynamics.battery_charge_fraction))
            run_stats["charging_events"].append(simtime)
        elif executed_action == 12:
            run_stats["desat_events"].append(simtime)
            print('tasking DESAT now: at t=',simtime," and wheel_speeds --> "+str(env.satellites[0].dynamics.wheel_speeds_fraction))

        run_stats["executed_action_counts"].update([int(executed_action)])
        print('current action_dict to be executed', action_dict['SS1'], "eclipse status of SS1:",env.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[env.satellites[0].dynamics.eclipse_index].read().shadowFactor)

        # ----- STEP -----
        observation, reward, terminated, truncated, info = env.step(action=action_dict)
        print(f"current SS1 reward {SS1_reward}")
        # print(f"Scanning Sat inclinatiom {orbitalMotion.rv2elem(orbitalMotion.MU_EARTH * 10*9,np.array(env.satellites[0].dynamics.r_BN_N), np.array(env.satellites[0].dynamics.v_BN_N)).i} and true anomaly {orbitalMotion.rv2elem(orbitalMotion.MU_EARTH * 10*9,np.array(env.satellites[0].dynamics.r_BN_N), np.array(env.satellites[0].dynamics.v_BN_N)).f} and battery {env.satellites[0].dynamics.battery_charge_fraction}")

        # ----- TELEMETRY -----
        # basic time-series
        run_stats["sim_times"].append(simtime)
        run_stats["battery_levels"].append(env.satellites[0].dynamics.battery_charge_fraction)
        run_stats["storage_levels"].append(env.satellites[0].dynamics.storage_level_fraction)


        # imaging metrics
        imaged_all = _get_imaged_all(env)               # raw imaged list
        imaged_illum = getattr(rewarder, "imaged_illuminated", [])
        run_stats["num_imaged"].append(_safe_len(imaged_all))
        run_stats["num_imaged_illuminated"].append(_safe_len(imaged_illum))

        # downlink metrics
        total_dl = getattr(rewarder, "total_downlinks", 0)
        useful_dl = getattr(rewarder, "useful_downlinks", 0)
        run_stats["num_downlinked_total"].append(int(total_dl))
        run_stats["num_downlinked_useful"].append(int(useful_dl))

        # rewards
        SS1_reward += reward.get("SS1", 0.0)
        run_stats["cumulative_reward_over_time"].append(SS1_reward)

        # ----- TERMINATION -----
        # Prefer robust checks; fall back as needed
        all_term = all(terminated.values()) if isinstance(terminated, dict) else bool(terminated)
        all_trunc = all(truncated.values()) if isinstance(truncated, dict) else bool(truncated)

        # "100 imaged" condition: check illuminated first (if you truly want *all* imaged use imaged_all)
        done_imaging = _safe_len(imaged_illum) >= 100 if imaged_illum is not None else _safe_len(imaged_all) >= 100

        if all_term or all_trunc or done_imaging:
            break

    # ---------- SUMMARY ----------
    SS1_actions_spec = env.satellites[0].action_builder.action_spec[0]
    summary = {}
    summary["final_cum_reward"] = SS1_reward

    # Use both variants to be robust
    final_imaged_all = _safe_len(_get_imaged_all(env))
    final_imaged_illum = _safe_len(getattr(_get_rewarder(env), "imaged_illuminated", []))
    summary["final_num_imaged"] = final_imaged_all
    summary["final_num_imaged_illuminated"] = final_imaged_illum

    final_total_dl = int(getattr(_get_rewarder(env), "total_downlinks", 0))
    final_useful_dl = int(getattr(_get_rewarder(env), "useful_downlinks", 0))
    summary["final_num_downlinked_total"] = final_total_dl
    summary["final_num_downlinked_useful"] = final_useful_dl

    # overall counts
    total_executed = sum(run_stats["executed_action_counts"].values())
    summary["total_actions"] = total_executed

    imaging_count = sum(v for k, v in run_stats["executed_action_counts"].items() if 0 <= int(k) <= 9)
    non_imaging_count = sum(v for k, v in run_stats["executed_action_counts"].items() if int(k) >= 10)
    summary["imaging_action_count"] = int(imaging_count)
    summary["non_imaging_action_count"] = int(non_imaging_count)

    summary["charging_events_count"] = len(run_stats["charging_events"])
    summary["downlink_events_count"] = len(run_stats["downlink_events"])
    summary["desat_events_count"] = len(run_stats["desat_events"])

    # shield stats
    summary["shield_interventions_count"] = len(run_stats["shield_interventions"])
    summary["shield_policy_disagreements_count"] = sum(1 for d in run_stats["shield_disagreements"] if d)

    # chosen-target based metrics (keep your original logic)
    if hasattr(SS1_actions_spec, "initial_angular_error") and SS1_actions_spec.initial_angular_error:
        summary["mean_initial_ang_error"] = float(np.mean(SS1_actions_spec.initial_angular_error))
        summary["std_initial_ang_error"] = float(np.std(SS1_actions_spec.initial_angular_error))
    else:
        summary["mean_initial_ang_error"] = None
        summary["std_initial_ang_error"] = None

    if hasattr(SS1_actions_spec, "chosen_target_distance") and SS1_actions_spec.chosen_target_distance:
        summary["mean_target_distance"] = float(np.mean(SS1_actions_spec.chosen_target_distance))
        summary["std_target_distance"] = float(np.std(SS1_actions_spec.chosen_target_distance))
    else:
        summary["mean_target_distance"] = None
        summary["std_target_distance"] = None

    if hasattr(SS1_actions_spec, "chosen_target_illumination_status") and SS1_actions_spec.chosen_target_illumination_status:
        summary["mean_illumination_status"] = float(np.mean(SS1_actions_spec.chosen_target_illumination_status))
        well = sum(1 for v in SS1_actions_spec.chosen_target_illumination_status if v > 0.5)
        summary["num_target_above_illumination_threshold"] = int(well)
    else:
        summary["mean_illumination_status"] = None
        summary["num_target_above_illumination_threshold"] = None

    return summary, run_stats


# Aggregate many runs for a policy
def evaluate_policy(policy_path, policy_name, obs_v_override, n_runs, policy_mode):
    print(f"\n--- Evaluating policy {policy_name} @ {policy_path} (obs_v override: {obs_v_override}) ---")
    policy_wrapper = PolicyWrapper(str(policy_path), policy_mode=policy_mode)
    per_run_summaries = []
    per_run_arrays = []
    total_time = None  # set default; your script used total_time computed earlier; set a default fallback
    # We'll extract total_time from your script's constant if needed; for now default to 100*450 as in your file
    total_time = 150 * imaging_duration

    for run_i in range(n_runs):
        seed = SEED_START + run_i
        start = time.time()
        summary, arrays = run_single_trial(policy_wrapper, obs_v_override, seed, run_i, total_time)
        elapsed = time.time() - start
        summary["seed"] = seed
        summary["run_index"] = run_i
        summary["elapsed_seconds"] = elapsed
        per_run_summaries.append(summary)
        per_run_arrays.append(arrays)
        print(f"  run {run_i+1}/{n_runs} done (seed {seed}) final reward {summary['final_cum_reward']:.3f} elapsed {elapsed:.1f}s")

    # compute aggregated statistics
    agg = {}
    rewards = [r["final_cum_reward"] for r in per_run_summaries]
    agg["n_runs"] = len(rewards)
    agg["reward_mean"] = float(np.mean(rewards))
    agg["reward_std"] = float(np.std(rewards))
    agg["reward_min"] = float(np.min(rewards))
    agg["reward_max"] = float(np.max(rewards))

    imaged = [r["final_num_imaged"] for r in per_run_summaries]
    agg["imaged_mean"] = float(np.mean(imaged))
    agg["imaged_std"] = float(np.std(imaged))

    # NEW: keep both total and useful downlinks
    dl_total = [r["final_num_downlinked_total"] for r in per_run_summaries]
    dl_useful = [r["final_num_downlinked_useful"] for r in per_run_summaries]
    agg["downlinked_total_mean"]  = float(np.mean(dl_total))
    agg["downlinked_total_std"]   = float(np.std(dl_total))
    agg["downlinked_useful_mean"] = float(np.mean(dl_useful))
    agg["downlinked_useful_std"]  = float(np.std(dl_useful))

    # consolidate executed action-level counts across runs
    global_action_counts = Counter()
    for arr in per_run_arrays:
        # fallback if older runs only had action_counts
        c = arr.get("executed_action_counts") or arr.get("action_counts") or Counter()
        global_action_counts.update({int(k): int(v) for k, v in dict(c).items()})
    agg["action_counts_total"] = {int(k): int(v) for k, v in global_action_counts.items()}

    # Shield aggregates
    total_interventions = sum(r["shield_interventions_count"] for r in per_run_summaries)
    total_disagreements = sum(r["shield_policy_disagreements_count"] for r in per_run_summaries)
    agg["shield_interventions_total"] = int(total_interventions)
    agg["shield_policy_disagreements_total"] = int(total_disagreements)


    return agg, per_run_summaries, per_run_arrays

def save_policy_plots(policy_name, agg, per_run_arrays, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)

    # -------- Build per-run category counts from executed_action_counts --------
    imaging_counts = []
    charge_counts = []
    downlink_counts = []
    desat_counts = []

    for arr in per_run_arrays:
        c = arr.get("executed_action_counts") or arr.get("action_counts") or Counter()
        # ensure int keys
        c = {int(k): int(v) for k, v in dict(c).items()}

        img = sum(v for k, v in c.items() if 0 <= int(k) <= 9)
        chg = c.get(10, 0)
        dln = c.get(11, 0)
        dst = c.get(12, 0)

        imaging_counts.append(img)
        charge_counts.append(chg)
        downlink_counts.append(dln)
        desat_counts.append(dst)

    # Avoid div-by-zero if no actions (edge case)
    n_runs = max(1, len(per_run_arrays))

    avg_counts = np.array([
        np.mean(imaging_counts) if imaging_counts else 0.0,
        np.mean(charge_counts) if charge_counts else 0.0,
        np.mean(downlink_counts) if downlink_counts else 0.0,
        np.mean(desat_counts) if desat_counts else 0.0,
    ], dtype=float)

    total_avg = float(np.sum(avg_counts)) if np.sum(avg_counts) > 0 else 1.0
    pct = (avg_counts / total_avg) * 100.0

    labels = ["image (0–9)", "charge (10)", "downlink (11)", "desat (12)"]
    x = np.arange(len(labels))

    # -------- Plot: bars = average counts (left y), line = percentage (right y) --------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(x, avg_counts)
    ax1.set_ylabel("Average count per run")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_title(f"{policy_name} — actions per run (avg count & % share)")

    ax2 = ax1.twinx()
    ax2.plot(x, pct, marker="o", linewidth=2)
    ax2.set_ylabel("Percentage of all actions (%)")
    ax2.set_ylim(0, 100)

    # Value labels (counts on top of bars, % above markers)
    for xi, b in zip(x, bars):
        h = b.get_height()
        ax1.text(xi, h, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    for xi, p in zip(x, pct):
        ax2.text(xi, p, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)

    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    figpath = outdir / f"{slugify(policy_name)}__{SEED_START}__{tag}__action_fourbar.png"
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)

    # -------- Bonus: simple line plot for avg battery & storage across runs --------
    all_battery = [np.array(r["battery_levels"]) for r in per_run_arrays if len(r.get("battery_levels", [])) > 0]
    all_storage = [np.array(r["storage_levels"]) for r in per_run_arrays if len(r.get("storage_levels", [])) > 0]
    if all_battery and all_storage:
        min_len = min(map(len, all_battery + all_storage))
        avg_batt = np.mean([b[:min_len] for b in all_battery], axis=0)
        avg_store = np.mean([s[:min_len] for s in all_storage], axis=0)
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(avg_batt, label="battery")
        ax.plot(avg_store, label="storage")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig2path = outdir / f"{slugify(policy_name)}__{SEED_START}__{tag}__battery_storage_avg.png"
        fig2.savefig(fig2path, bbox_inches="tight")
        plt.close(fig2)


def main():
    # one tag for the whole batch
    batch_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    policies = find_policies(POLICY_FOLDER, substring=FILTER_SUBSTRING, suffix=POLICY_MATCH_SUFFIX)
    print(f"Found {len(policies)} candidate policies in {POLICY_FOLDER}")

    for p in policies:
        # resolve name and obs mapping
        policy_name = p.name
        safe_name = slugify(policy_name)

        # allow override if 'aug1' in name
        obs_v = DEFAULT_POLICY_OBS_MAP.get("aug1", 2) if "aug1" in policy_name else 1
        if "obsv2" in policy_name or "obsv2" in str(p):
            obs_v = 2
        elif "obsv5" in policy_name or "obsv5" in str(p):
            obs_v = 5
        elif "obsv1" in policy_name or "obsv1" in str(p):
            obs_v = 1
        elif "obsv7" in policy_name or "obsv7" in str(p):
            obs_v = 7


        agg, per_run_summaries, per_run_arrays = evaluate_policy(p, policy_name, obs_v, N_RUNS, POLICY_LOAD_MODE)

        # per-policy, per-batch folder (unique because of batch_tag)
        if just_imaging:
            outfolder = SAVE_DIR / f"JustImaging__seed{SEED_START}_{safe_name}__{batch_tag}"
        else:
            outfolder = SAVE_DIR / f"seed{SEED_START}_{safe_name}__{batch_tag}"
        outfolder.mkdir(parents=True, exist_ok=True)

        # include the tag in filenames too
        summary_path = outfolder / f"seed{SEED_START}__{safe_name}__{batch_tag}__results_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(
                {
                    "policy_path": str(p),
                    "policy_name": policy_name,
                    "batch_tag": batch_tag,
                    "params": {
                        "N_RUNS": N_RUNS,
                        "use_shield": use_shield,
                        "POLICY_LOAD_MODE": POLICY_LOAD_MODE,
                        "obs_v": obs_v,
                        "SEED_START": SEED_START,
                        "TIME_LIMIT": TIME_LIMIT,
                    },
                    "aggregate": agg,
                    "per_run": per_run_summaries,
                },
                fh,
                indent=2,
            )

        # save raw arrays (npz) for later detailed plotting
        shield_tag = "wShield" if use_shield else "noShield"
        if just_imaging:
            npz_path = outfolder / f"justImaging__{SEED_START}_{safe_name}__{batch_tag}__{shield_tag}__rundata.npz"
        else:
            npz_path = outfolder / f"{safe_name}__{SEED_START}__{batch_tag}__{shield_tag}__rundata.npz"

        # collate arrays into saveable format
        np_save_dict = {}
        for i, arr in enumerate(per_run_arrays):
            for k, v in arr.items():
                if k in ("executed_action_counts", "policy_action_counts", "action_counts"):
                    c = dict(v)
                    np_save_dict[f"run{i}_{k}_keys"] = np.array(list(c.keys()), dtype=object)
                    np_save_dict[f"run{i}_{k}_vals"] = np.array(list(c.values()))
                else:
                    np_save_dict[f"run{i}_{k}"] = np.array(v)
        np.savez_compressed(npz_path, **np_save_dict)

        # make and save plots (pass the same tag)
        save_policy_plots(policy_name, agg, per_run_arrays, outfolder, tag=batch_tag)

        print(f"Saved results for policy {policy_name} in {outfolder}")

if __name__ == "__main__":
    start_all = time.time()
    main()
    print(f"Batch finished in {time.time() - start_all:.1f} s")
