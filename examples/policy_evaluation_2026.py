import gymnasium as gym
from requests.packages import target
import time

start_time = time.time()
from pathlib import Path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pandas as pd
import argparse
import json

from functools import partial
from matplotlib.collections import PolyCollection

from sim_config import SimConfig


size=1
label_size = 13
tick_label_size = 13
legend_fontsize = 11
AMOS_FONTS = dict(
    title=14*size,   # figure/axes titles
    label=14*size,   # x/y label font size
    tick=12*size,    # tick label size
    legend=12*size,  # legend font size
)

matplotlib.rcParams.update({
    "axes.titlesize": AMOS_FONTS["title"],
    "axes.labelsize": AMOS_FONTS["label"],
    "xtick.labelsize": AMOS_FONTS["tick"],
    "ytick.labelsize": AMOS_FONTS["tick"],
    "legend.fontsize": AMOS_FONTS["legend"],
})
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

from load_policy import load_policy
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
def _safe(s: str) -> str:
    """Filesystem-safe string."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)

    # NEW: target environment switch
    ap.add_argument("--target_env", type=str, default="leo", choices=["leo", "mixed"])

    # NEW: optional mixed weights
    ap.add_argument("--mix_weights", type=str, default='{"LEO":0.5,"MEO":0.3,"GEO":0.2}',
                    help='JSON string, e.g. \'{"LEO":0.6,"MEO":0.2,"GEO":0.2}\'')

    # optional: quiet / save_data if you already use those in runner
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--save_data", action="store_true")
    ap.add_argument("--no_save_data", action="store_true")

    return ap.parse_args()


def make_run_dir(base_dir: str, seed: int, policy_tag: str, run_tag: str = "") -> str:
    """
    Create a unique run directory so nothing is overwritten.
    Example: data/RL_seed20_latest_obsv7_20260108_195446/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [policy_tag, f"seed{seed}"]
    if run_tag:
        name_parts.append(_safe(run_tag))
    name_parts.append(ts)
    run_dir = os.path.join(base_dir, "_".join(name_parts))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

import re
def alpha_from_tag(s: str, default=None, strict_sum_100: bool = True):
    """
    Extract alpha from a string containing patterns like '10d90i' or '75d25i'.

    Interpretation:
      - 'XdYi' means X% downlink, Y% imaging
      - alpha = X / 100

    Returns:
      float alpha, or `default` if not found.
    """
    if s is None:
        return default

    m = re.search(r"(\d+)\s*d\s*(\d+)\s*i", str(s))
    if not m:
        return default

    d = int(m.group(1))  # downlink percent
    i = int(m.group(2))  # imaging percent

    if strict_sum_100 and (d + i != 100):
        # If you want to be tolerant instead, set strict_sum_100=False
        raise ValueError(f"Found tag '{d}d{i}i' but d+i != 100 in: {s}")

    return d / 100.0

def print_alpha(policy_name: str, policy_path: str):
    a = alpha_from_tag(policy_path, default=None, strict_sum_100=False)
    print(f"\n\nPOLICY NAME: {policy_name:50s} alpha value being extracted: alpha = {a}  ({policy_path})")

import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="Override seed_number")
    p.add_argument("--save_data", action="store_true", default=None, help="Force save_data=True")
    p.add_argument("--no_save_data", action="store_true", help="Force save_data=False")
    p.add_argument("--quiet", action="store_true", help="Reduce printing")
    p.add_argument(
        "--target_env",
        choices=["leo", "mixed"],
        default="mixed",
        help="Target environment type"
    )
    p.add_argument(
        "--mix_weights",
        type=str,
        default='{"LEO":0.5,"MEO":0.3,"GEO":0.2}',
        help='JSON dict of regime weights when target_env="mixed"'
    )
    p.add_argument(
        "--policy_name",
        type=str,
        default=None,
        help="Policy variable name to evaluate, e.g. oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i.",
    )
    p.add_argument(
        "--policy_mode",
        choices=["best", "smallest", "latest"],
        default="latest",
        help="Checkpoint selection mode passed to load_policy.",
    )

    return p.parse_args()

ARGS = parse_args()

def _print(*a, **k):
    if not ARGS.quiet:
        print(*a, **k)


def save_npy(run_dir: str, name: str, arr) -> None:
    np.save(os.path.join(run_dir, f"{name}.npy"), np.asarray(arr))

def save_json(run_dir: str, name: str, obj: dict) -> None:
    path = os.path.join(run_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Saved JSON to: {path}")

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


use_shield = True
act_random = False
use_heuristic = False
heuristic_mode = "angle"  # not used unless use_heuristic is True. heuristic modes: {"angle", "distance"}
if act_random:
    policy_tag = "RANDOM"
elif use_heuristic:
    policy_tag = "HEUR"
    if heuristic_mode == "angle":
        policy_tag = "HEUR_ANGLE"
    elif heuristic_mode == "distance":
        policy_tag = "HEUR_DISTANCE"
else:
    policy_tag = "RL"


# Base simulation configuration (shared between training & evaluation)
sim_cfg = SimConfig(
    n_targets=100,
    n_targets_ahead=10,
    imaging_duration=300.0,
    variable_duration_imaging=True,  # AMOS 2026: stop after successful hold-gated image
    extra_time_factor=1.5,
    obs_v=7.0,          # default obs version; will be overwritten if policy_name known
    just_imaging=False,
)

# Local aliases to minimize refactoring elsewhere
n_targets = sim_cfg.n_targets
n_targets_ahead = sim_cfg.n_targets_ahead
imaging_duration = sim_cfg.imaging_duration
variable_duration_imaging = sim_cfg.variable_duration_imaging
variable_duration_downlink = sim_cfg.variable_duration_downlink
downlink_empty_threshold_bits = sim_cfg.downlink_empty_threshold_bits
total_time = sim_cfg.total_time
obs_v = sim_cfg.obs_v
just_imaging = sim_cfg.just_imaging

def make_rso_scenario():
    return scene.RandomSatellites(
        "SS1",
        n_targets=n_targets,
        priority_mode=sim_cfg.priority_mode,
        priority_sum=sim_cfg.priority_sum,
        rescale_priorities_to_sum=sim_cfg.rescale_priorities_to_sum,
        priority_constant=sim_cfg.priority_constant,
        priority_uniform_low=sim_cfg.priority_uniform_low,
        priority_uniform_high=sim_cfg.priority_uniform_high,
        priority_gaussian_mean=sim_cfg.priority_gaussian_mean,
        priority_gaussian_std=sim_cfg.priority_gaussian_std,
        priority_min=sim_cfg.priority_min,
        priority_max=sim_cfg.priority_max,
    )


def make_rso_rewarder():
    return data.RSOTargetImageReward(
        reimage_cooldown_orbits=sim_cfg.reimage_cooldown_orbits,
        verify_image_quality_on_downlink=sim_cfg.verify_image_quality_on_downlink,
        hide_pending_targets=sim_cfg.hide_pending_targets,
        image_quality_threshold=sim_cfg.image_quality_threshold,
    )


def make_downlink_action(duration: float):
    return act.Downlink(
        duration=duration,
        variable_duration_downlink=variable_duration_downlink,
        empty_storage_threshold_bits=downlink_empty_threshold_bits,
    )

# setting up sim parameters manually if needed
# n_targets = 100
# n_targets_ahead = 10
# imaging_duration = 300
# total_time = n_targets * imaging_duration * 1.5   # 5700.0  # approximately 1 orbit


args = ARGS
TARGET_ENV = args.target_env
MIX_WEIGHTS = json.loads(args.mix_weights) if TARGET_ENV == "mixed" else None


seed_number = 99
if args.seed is not None:
    seed_number = args.seed

save_data = True  # default
if args.no_save_data:
    save_data = False
elif args.save_data:
    save_data = True

policy_mode = args.policy_mode
eclipse_norm = 5700
save_vizard = False
viz_rate = 5.0

# Orbit / GS constants
ORBIT_PERIOD_SEC = 95 * 60
ECL_SLICE  = slice(75, 77)
GS_START   = 77     # will be refined based on obs_v below
N_GS       = 5      # updated for some obs_v
PAIR_STRIDE = 2

if use_heuristic:
    run_tag = f"HEUR_{heuristic_mode}_{TARGET_ENV}"
elif act_random:
    run_tag = "RANDOM"
else:
    run_tag = f"{policy_tag}_{policy_mode}10d90i_{TARGET_ENV}"


base_data_dir = os.path.join(os.path.dirname(__file__), "data")  # examples/data
run_dir = make_run_dir(base_data_dir, seed_number, policy_tag, run_tag)
print(f"\n=== Run outputs will be saved to: {run_dir} ===\n")



# If the local training now outputs *normalized* offsets since Aug 14, toggle these flags as appropriate. If values are already absolute times, set to False.
GS_NORMALIZED = True    # set True if GS values are normalized offsets; False if absolute times

# POLICIES
# September/October Policies
# October 14 – batch5000, gamma9997, reducedFailurePenalty
oct14_obsv7_1e_5lr_batch5000_gamma9997_0d100i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_0d100i_reducedFailurePenalty_1761159994.1781282/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_0d100i_reducedFailurePenalty.out_0"
# oct14_obsv7_1e_5lr_batch5000_gamma9997_10d90i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_1761114479.911475/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty.out_0" #does not exist...
oct14_obsv7_1e_5lr_batch5000_gamma9997_10d90i =  "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty_1761114479.911475/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_20d80i_1761099406.663396/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_20d80i.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_30d70i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_30d70i_1761079141.0804758/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_30d70i.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_40d60i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_40d60i_1761079131.258607/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_40d60i.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_50d50i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_50d50i_reducedFailurePenalty_1761078528.689754/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_50d50i_reducedFailurePenalty.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_60d40i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_60d40i_reducedFailurePenalty_lowBatPenalty_1761078528.6867332/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_60d40i_reducedFailurePenalty_lowBatPenalty.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_70d30i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_70d30i_1761251961.5538032/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_70d30i.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_80d20i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_80d20i_1761251244.9588354/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_80d20i.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_90d10i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_90d10i_reducedFailurePenalty_1761248998.453683/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_90d10i_reducedFailurePenalty.out_0"
oct14_obsv7_1e_5lr_batch5000_gamma9997_100d00i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_100d00i_reducedFailurePenalty_1761227644.112857/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_100d00i_reducedFailurePenalty.out_0"

oct14_obsv7_48hrs_1e_5lr_batch5000_gamma9997_10d90i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty_1761114479.911475/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty.out_0"

obsv7_48hrs_1e_5lr_batch5000_gamma9997_10d90i = "/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty_1761114479.911475/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty.out_0"

# downlink_reward_policy = "/Users/dahu1128/rllib_results/june12rllib_results/lowBaudRate_5e-6lr_downlink_reward_penalties_smallest_storage_Polaris_simulation_1749771784.4956822/lowBaudRate_5e-6lr_downlink_reward_penalties_smallest_storage_0/"
downlink_reward_policy = "/Users/dahu1128/rllib_results/reward_comparison/lowBaudRate_5e-6lr_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1750741069.7033312/lowBaudRate_5e-6lr_downlink_reward_new_penalties_smallest_storage_0"
downlink_reward_policy_shorter_imaging ="/Users/dahu1128/rllib_results/reward_comparison/lowBaudRate_shorter_imaging_5e-6lr_downlink_reward_penalties_smallest_storage_0"
# imaging_reward_policy = ""

# August 19th
aug19_obsv7_1e_5lr_batch5000_gamma9997_0d100i = "/Users/dahu1128/rllib_results/august_results/aug19rllib_results/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_0d100i_1755684700.6020117/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_0d100i.out_0"
aug19_justImaging_obsv5_1e5lr_batch3200_gamma9995_0d100i = "/Users/dahu1128/rllib_results/august_results/aug19rllib_results/aug19_justimaging_obsv5_1e-5lr_batch3200_gamma9995_0d100i_1755681942.7970276/aug19_justimaging_obsv5_1e-5lr_batch3200_gamma9995_0d100i.out_0"
obsv7_1e_5lr_batch5000_gamma9997_0d100i = "/Users/dahu1128/rllib_results/aug19rllib_results/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_0d100i_1755684700.6020117/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_0d100i.out_0"
obsv7_1e_5lr_batch5000_gamma9997_10d90i = "/Users/dahu1128/rllib_results/aug19rllib_results/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_1755684765.6849554/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i.out_0"
obsv7_1e_5lr_batch5000_gamma9997_20d80i = "/Users/dahu1128/rllib_results/aug19rllib_results/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_20d80i_1755685121.0937102/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_20d80i.out_0"
obsv7_1e_5lr_batch5000_gamma9997_30d70i = "/Users/dahu1128/rllib_results/aug19rllib_results/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_30d70i_1755685667.1920402/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_30d70i.out_0"
obsv7_1e_5lr_batch5000_gamma9997_40d60i = "/Users/dahu1128/rllib_results/aug19rllib_results/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_40d60i_1755688491.5682302/aug19_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_40d60i.out_0"
# obsv7_1e_5lr_batch5000_gamma9997_50d50i = ""
# obsv7_1e_5lr_batch5000_gamma9997_60d40i = ""
# obsv7_1e_5lr_batch5000_gamma9997_70d30i = ""
# obsv7_1e_5lr_batch5000_gamma9997_80d20i = ""
# obsv7_1e_5lr_batch5000_gamma9997_90d10i = ""
# obsv7_1e_5lr_batch5000_gamma9997_90d10i = ""

# August 18th
# cluster
aug18_obsv7_1e_5lr_batch320_gamma9995_0d100i_1755589893 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_1e-5lr_0.1cp_gamma9995_0d100i_1755589893.5762584/aug18_restrictedResources_obsv7_1e-5lr_batch320_gamma9995_0d100i.out_0"
# aug18_obsv7_1e_5lr_batch3200_gamma9995_0d100i_1755591535 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_1e-5lr_batch3200_gamma9995_0d100i_1755591535.5134878/aug18_restrictedResources_obsv7_1e-5lr_batch3200_gamma9995_0d100i.out_0"
aug18_obsv7_1e_5lr_batch3200_gamma9995_0d100i_1755591535 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_1e-5lr_batch3200_gamma9995_0d100i_1755591535.5134878/aug18_restrictedResources_obsv7_1e-5lr_batch3200_gamma9995_0d100i.out_0"
aug18_obsv7_1e_6lr_batch3200_gradclip0_001_cp0_005_gamma9995_0d100i_1755597947 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_1e-6lr_batch3200_gradclip0.001_cp0.005_gamma9995_0d100i_1755597947.1568983/aug18_restrictedResources_obsv7_1e-6lr_batch3200_gradclip0.001_cp0.005_gamma9995_0d100i.out_0"
aug18_obsv7_5e_6lr_batch1600_gamma999_0d100i_1755590430 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_5e-6lr_batch1600_gamma999_0d100i_1755590430.431192/aug18_restrictedResources_obsv7_5e-6lr_batch1600_gamma999_0d100i.out_0"
aug18_obsv7_5e_6lr_batch160_gamma9995_0d100i_1755599175 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_5e-6lr_batch160_gamma9995_0d100i_1755599175.0386055/aug18_restrictedResources_obsv7_5e-6lr_batch160_gamma9995_0d100i.out_0"
aug18_obsv7_5e_6lr_batch3200_gamma9995_0d100i_1755590744 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_5e-6lr_batch3200_gamma9995_0d100i_1755590744.7777116/aug18_restrictedResources_obsv7_5e-6lr_batch3200_gamma9995_0d100i.out_0"
aug18_obsv7_5e_6lr_batch320_gamma9995_0d100i_1755590009 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_5e-6lr_batch320_gamma9995_0d100i_1755590009.3689198/aug18_restrictedResources_obsv7_5e-6lr_batch320_gamma9995_0d100i.out_0"
aug18_obsv7_5e_6lr_batch320_gamma999_0d100i_1755589530 = "/Users/dahu1128/rllib_results/august_results/aug18rllib_results/aug18_restrictedResources_obsv7_5e-6lr_batch320_gamma999_0d100i_1755589530.5835643/aug18_restrictedResources_obsv7_5e-6lr_batch320_gamma999_0d100i.out_0"
# locally trained
wGAE_imaging_reward_obsv7_1e_6lr = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug18_wGAE_150batch_halfnetwork_-1penalties_smallerICbattery_restrictedResources_obsv6_1e-6lr_0.15cp_gamma9997_100d0i_1755562684.144431/aug18_wGAE_150batch_halfnetwork_-1penalties_smallerICbattery_restrictedResources_obsv6_1e-6lr_0.15cp_gamma9997_100d0i.out_0"
wGAE_imaging_reward_obsv6_1e_3lr = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug18_restrictedResources_obsv21_1e-3lr_0.1cp_gamma9995_100d0i_1755540587.565471/aug18_restrictedResources_obsv21_1e-3lr_0.1cp_gamma9995_100d0i.out_0"
wGAE_150batch_1e_6lr_15cp_gamma9997_100d0i = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug18_wGAE_150batch_halfnetwork_-1penalties_smallerICbattery_restrictedResources_obsv6_1e-6lr_0.15cp_gamma9997_100d0i_1755498452.2236679/aug18_wGAE_150batch_halfnetwork_-1penalties_smallerICbattery_restrictedResources_obsv6_1e-6lr_0.15cp_gamma9997_100d0i.out_0"

# August 15th
wGAE_justImaging_gamma9999_Episodebatch_obs5 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug15_wGAE_imaging_baseline_verysmallclip_episodebatch_halfnetwork_obsv5_1e-6lr_0.005cp_gamma9999_0d100imodel_1755324328.4317691/aug15_wGAE_imaging_baseline_verysmallclip_episodebatch_halfnetwork_obsv5_1e-6lr_0.005cp_gamma9999_0d100i.out_0"
# wGAE_justImaging_gamma9999_Episodebatch_obs5 = "/Users/dahu1128/rllib_results/aug15_wGAE_imaging_baseline_verysmallclip_episodebatch_halfnetwork_obsv5_1e-6lr_0.005cp_gamma9999_0d100imodel_1755324328.4317691/aug15_wGAE_imaging_baseline_verysmallclip_episodebatch_halfnetwork_obsv5_1e-6lr_0.005cp_gamma9999_0d100i.out_0"
wGAE_balance0d100i_gamma99993_2penalties_Episodebatch_obs4 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/model_1755241967.35119/aug15_wGAE_nodesat_verysmallclip_episodebatch_halfnetwork_2penalty_smallerICbattery_restrictedResources_obsv4_1e-6lr_0.01cp_gamma99993_0d100i.out_0"

# August 14th
wGAE_balance0d100i_gamma9999_5downlink10batterypenalties_Largebatch_obs21 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug14_wGAE_smallclip_largebatch_halfnetwork_5downlink10batterypenalty_smallerICbattery_restrictedResources_obsv2_2e-6lr_0.15cp_gamma9999_0d100i_1755210856.363682/aug14_wGAE_smallclip_largebatch_halfnetwork_5downlink10batterypenalty_smallerICbattery_restrictedResources_obsv2_2e-6lr_0.15cp_gamma9999_0d100i.out_0"
wGAE_balance0d100i_largepenalties_Largebatch_obs21 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug14_wGAE_smallclip_smallbatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i_1755148468.119241/aug14_wGAE_smallclip_largebatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i.out_0"
# August 13th
wGAE_balance0d100i_smallclip_largepenalties_smallbatch_obs2 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug13_wGAE_smallclip_smallbatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i_1755121787.721219/aug13_wGAE_smallclip_smallbatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i.out_0"
wGAE_balance0d100i_largepenalties_smallbatch_obs2 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug13_wGAE_smallbatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i_1755107128.629914/aug13_wGAE_smallbatch_halfnetwork_largepenalties_smallerICbattery_restrictedResources_obsv2_1e-6lr_0.15cp_gamma9997_0d100i.out_0"
latestbalance0d100i_largepenalties_smallbatch_obs2 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aaug13_woGAE_smallbatch_halfnetwork_smallpenalties_smallerICbattery_restrictedResources_obsv2_2e-7lr_0.15cp_gamma95_0d100i_1755059875.601228/aug13_woGAE_smallbatch_halfnetwork_smallpenalties_smallerICbattery_restrictedResources_obsv2_2e-7lr_0.15cp_gamma95_0d100i.out_0"

# August 12th
woGAE_balance100d0i_smallpenalties_smallbatch_halfnetwork_obs2 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug12_woGAE_smallbatch_halfnetwork_smallpenalties_restrictedResources_obsv2_5e-7lr_0.15cp_gamma95_100d0i_1755032040.7225652/aug12_woGAE_smallbatch_halfnetwork_smallpenalties_restrictedResources_obsv2_5e-7lr_0.15cp_gamma95_100d0i.out_0"

# August 11th
balance50d50i_smallpenalties_smallbatch_obs2 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug11_woGAE_smallbatch_smallpenalties_restrictedResources_obsv2_1e-6lr_0.15cp_gamma95_50d50i_1754970261.733622/aug11_woGAE_smallbatch_smallpenalties_restrictedResources_obsv2_1e-6lr_0.15cp_gamma95_50d50i.out_0"

# August 1st
balance75d25i_nopenalties_lowICbattery_woGAE_obs2_gamma95 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/aug1_nopenalties_smallerICbattery_obsv2_2e-5lr_0.1cp_gamma95_75d25i_1754054673.0471108/aug1_nopenalties_smallerICbattery_obsv2_2e-5lr_0.1cp_gamma95_75d25i.out_0"

# Cluster Aug1st policies:
balance0d100i_nopenalties_wGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i_1754183683.6134467/aug1_nopenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i.out_0"
balance0d100i_nopenalties_wGAE_restricted_Resources_obs2_1754513855 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i_1754513855.0651953/aug1_nopenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i.out_0"
balance0d100i_nopenalties_wGAE_restricted_Resources_obs2_5e6lr = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_wGAE_restricted_Resources_obsv2_5e-6lr_0.1cp_gamma9995_0d100i_1754224141.9058733/aug1_0d100i_nopenalties_wGAE_restricted_Resources_obsv2_5e-6lr_0.1cp_gamma9995.out_0"
balance0d100i_nopenalties_wGAE_unlimitedResources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_wGAE_unlimitedResources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i_1754165705.0502024/aug1_nopenalties_wGAE_unlimitedResources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i.out_0"
balance0d100i_nopenalties_woGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_0d100i_1754193344.0782313/aug1_nopenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_0d100i.out_0"
balance0d100i_nopenalties_woGAE_restricted_Resources_obs2_1754513855 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_0d100i_1754513855.064417/aug1_nopenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_0d100i.out_0"
balance0d100i_nopenalties_woGAE_unlimitedResources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_woGAE_unlimitedResources_obsv2_2e-5lr_0.1cp_gamma95_0d100i_1754181832.6771822/aug1_nopenalties_woGAE_unlimitedResources_obsv2_2e-5lr_0.1cp_gamma95_0d100i.out_0"
balance0d100i_nopenalties_woGAE_unlimitedResources_obs2_1754513855 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_nopenalties_woGAE_unlimitedResources_obsv2_2e-5lr_0.1cp_gamma95_0d100i_1754513855.0656035/aug1_nopenalties_woGAE_unlimitedResources_obsv2_2e-5lr_0.1cp_gamma95_0d100i.out_0"
balance25d75i_smallpenalties_wGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995_25d75i_1754217650.8114564/aug1_25d75i_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995.out_0"
balance50d50i_smallpenalties_wGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995_50d50i_1754219572.1531365/aug1_50d50i_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995.out_0"
balance75d25i_smallpenalties_wGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995_75d25i_1754220926.8928921/aug1_75d25i_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995.out_0"
balance100d0i_smallpenalties_wGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995_100d0i_1754513855.0641232/aug1_100d0i_smallpenalties_wGAE_restricted_Resources_obsv2_1e-5lr_0.1cp_gamma9995.out_0"
balance0d100i_smallpenalties_wGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i_1754186234.8776257/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i.out_0"
balance0d100i_smallpenalties_wGAE_restricted_Resources_obs2_1754513855 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i_1754513855.0647912/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma9995_0d100i.out_0"
balance75d25i_smallpenalties_wGAE_restricted_Resources_obs2_5e6lr = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_wGAE_restricted_Resources_obsv2_5e-6lr_0.1cp_gamma9995_75d25i_1754222766.3191435/aug1_75d25i_smallpenalties_wGAE_restricted_Resources_obsv2_5e-6lr_0.1cp_gamma9995.out_0"
balance0d100i_smallpenalties_woGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_0d100i_1754209168.7183337/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_0d100i.out_0"
balance25d75i_smallpenalties_woGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_25d75i_1754209052.4012935/aug1_25d75i_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95.out_0"
balance25d75i_smallpenalties_woGAE_restricted_Resources_obs2_1754513855 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_25d75i_1754513855.065375/aug1_25d75i_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95.out_0"
balance50d50i_smallpenalties_woGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_50d50i_1754211122.300214/aug1_50d50i_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95.out_0"
balance50d50i_smallpenalties_woGAE_restricted_Resources_obs2_1754513855 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_50d50i_1754513855.0651023/aug1_50d50i_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95.out_0"
balance75d25i_smallpenalties_woGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_75d25i_1754214159.7715378/aug1_75d25i_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95.out_0"
balance100d0i_smallpenalties_woGAE_restricted_Resources_obs2 = "/Users/dahu1128/rllib_results/august_results/aug1rllib_results/aug1_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95_100d0i_1754216613.0556254/aug1_100d0i_smallpenalties_woGAE_restricted_Resources_obsv2_2e-5lr_0.1cp_gamma95.out_0"

# July 31st
balance75d25i_reward_policy_woGAE_obs2_gamma95 = "/Users/dahu1128/rllib_results/july31_obsv2_1e-6lr_0.1cp_gamma95_75d25i_1753923413.642091/july31_obsv2_1e-6lr_0.1cp_gamma95_75d25i.out_0"
balance75d25i_nopenalties_woGAE_obsv2_gamma95 = "/Users/dahu1128/rllib_results/july31_nopenalties_obsv2_1e-6lr_0.1cp_gamma95_75d25i_1753984221.842252/july31_nopenalties_obsv2_1e-6lr_0.1cp_gamma95_75d25i.out_0"

# July 30th
balance0d100i_smallpenalties_wGAE_gamma9995_obs2 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/july30_wGAE_obsv2_1e-5lr_0.1cp_gamma99995_0d100i_imaging_1753832186.3342025/july30_wGAE_obsv2_1e-5lr_0.1cp_gamma99995_0d100i_imaging_job_%a-%j.out_0"
just_imaging_action_wGAE_gamma99995_obs2 = "/Users/dahu1128/rllib_results/july_results/july30rllib_results/july30_wGAE_obsv2_doublebatch_1e-6lr_0.1cp_gamma99995_0d100i_imaging1753858117.550128/july30_wGAE_obsv2_doublebatch_1e-6lr_0.1cp_gamma99995_0d100i_imaging_job_%a-%j.out.out_0"

# July 22nd
balance00100_reward_policy_obs1_gamma99 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma99_0d100i_1753189664.120498/july22_obsv1_1e-5lr_0.1cp_gamma99_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs1_gamma995 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma995_0d100i_1753184464.3957765/july22_obsv1_1e-5lr_0.1cp_gamma995_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs1_gamma9995 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma9995_0d100i_1753252554.210772/july22_obsv1_1e-5lr_0.1cp_gamma9995_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs1_gamma99995 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv1_1e-5lr_0.1cp_gamma99995_0d100i_1753189664.1202838/july22_obsv1_1e-5lr_0.1cp_gamma99995_0d100i_job_%a-%j.out_0"

balance00100_reward_policy_obs2_gamma99 = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma99_0d100i_1753349618.6042557/july22_obsv2_1e-5lr_0.1cp_gamma99_0d100i_job_%a-%j.out_0"
balance00100_reward_policy_obs2_gamma995 = '/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma995_0d100i_1753350273.3467767/july22_obsv2_1e-5lr_0.1cp_gamma995_0d100i_job_%a-%j.out_0'
balance00100_reward_policy_obs2_gamma9995 = '/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma9995_0d100i_1753504923.8221972/july22_obsv2_1e-5lr_0.1cp_gamma9995_0d100i_job_%a-%j.out_0'
balance00100_reward_policy_obs2_gamma99995 = '/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-5lr_0.1cp_gamma99995_0d100i_1753349618.6116135/july22_obsv2_1e-5lr_0.1cp_gamma99995_0d100i_job_%a-%j.out_0'

balance00d100i_obs2_gamma9995_1e6lr = "/Users/dahu1128/rllib_results/july_results/july22rllib_results/july22_obsv2_1e-6lr_0.1cp_gamma9995_0d100i_1753350277.895099/july22_obsv2_1e-6lr_0.1cp_gamma9995_0d100i.out_0"


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
cluster_5050policy_1e5lr_obs11 = "/Users/dahu1128/rllib_results/reward_comparison/july5rllib_results/july5_1e-5lr_0005torque_50d50i_reward_new_penalties_smallest_storage_Polaris_simulation_1751885868.6929862/1e-5lr_0005torque_50d50i_reward_new_penalties_smallest_storage_0"
# June 28th
cluster_1000_policy_v1_obs1 = "/Users/dahu1128/rllib_results/june_results/june29rllib_results/june29_lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1751349196.029947/lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_0"
cluster_1000_policy_v2_obs1 = "/Users/dahu1128/rllib_results/june_results/june29rllib_results/june29_lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1751336442.6073096/lowBaudRate_1e-5lr_0005torque_downlink_reward_new_penalties_smallest_storage_0"

# June 23rd
cluster_1000_policy_v2_obs1_wGAE ="/Users/dahu1128/rllib_results/june_results/june23rllib_results/lowBaudRate_5e-6lr_001torque_downlink_reward_new_penalties_smallest_storage_Polaris_simulation_1750740679.4746056/lowBaudRate_5e-6lr_001torque_downlink_reward_new_penalties_smallest_storage_0"

# June 9th
imaging_reward_smalldata_smallbat_obs1 = "/Users/dahu1128/rllib_results/reward_comparison/1e-6lr_failure_penalties_no_torque_small_battery_small_data_Polaris_sim_1749226596.4501252/model_0"
imaging_unlimitedResources_baseline = "/Users/dahu1128/rllib_results/reward_comparison/100targets_10ahead_Polaris_simulation_1746754281.876747/model_0"

#June 6th
imaging_rewarded_noeclipse_1e_6lr_failure_penalties = "/Users/dahu1128/rllib_results/june_results/june6rllib_results/1e-6lr_failure_penalties_no_torque_small_battery_small_data_Polaris_sim_1749226596.4501252/model_0"

# policy_path = obsv7_48hrs_1e_5lr_batch5000_gamma9997_10d90i #DEPRECATED... now the globals() line is used below...    #balance00d100i_obs2_gamma9995_1e6lr
# Choose which policy to evaluate by NAME
policy_name = args.policy_name or "oct14_obsv7_1e_5lr_batch5000_gamma9997_10d90i"
if policy_name not in globals():
    raise ValueError(
        f"Unknown policy_name '{policy_name}'. Add it to the policy path block or "
        "choose one of the existing policy variable names."
    )
policy_path = globals()[policy_name]

# Define all known policy paths with associated obs values
policy_obs_map = {
    "oct14_obsv7_48hrs_1e_5lr_batch5000_gamma9997_10d90i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_100d00i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_90d10i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_80d20i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_70d30i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_60d40i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_50d50i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_40d60i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_30d70i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_10d90i": 7,
    "oct14_obsv7_1e_5lr_batch5000_gamma9997_0d100i": 7,

    "obsv7_1e_5lr_batch5000_gamma9997_90d10i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_80d20i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_70d30i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_60d40i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_50d50i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_40d60i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_30d70i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_20d80i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_10d90i": 7,
    "obsv7_1e_5lr_batch5000_gamma9997_0d100i": 7,
    "aug19_justImaging_obsv5_1e5lr_batch3200_gamma9995_0d100i": 5,
    "aug19_obsv7_1e_5lr_batch5000_gamma9997_0d100i": 7,
    "imaging_rewarded_noeclipse_1e_6lr_failure_penalties": 1,
    "balance00d100i_obs2_gamma9995_1e6lr": 2,
    "aug18_obsv7_5e_6lr_batch3200_gamma9995_0d100i_1755590744": 7,
    "aug18_obsv7_1e_5lr_batch3200_gamma9995_0d100i_1755591535": 7,
    "wGAE_imaging_reward_obsv7_1e_6lr": 7,
    "wGAE_150batch_1e_6lr_15cp_gamma9997_100d0i": 4,
    "wGAE_imaging_reward_obsv6_1e_3lr": 6,
    "wGAE_justImaging_gamma9999_Episodebatch_obs5": 5,
    "wGAE_balance0d100i_gamma99993_2penalties_Episodebatch_obs4": 4,
    "wGAE_balance0d100i_gamma9999_5downlink10batterypenalties_Largebatch_obs21": 2.1,
    "just_imaging_action_wGAE_gamma99995_obs2": 2,
    "woGAE_balance100d0i_smallpenalties_smallbatch_halfnetwork_obs2": 2,
    "wGAE_balance0d100i_largepenalties_Largebatch_obs2": 2.1,
    "wGAE_balance0d100i_smallclip_largepenalties_smallbatch_obs2": 2,
    "wGAE_balance0d100i_largepenalties_smallbatch_obs2": 2,
    "latestbalance0d100i_largepenalties_smallbatch_obs2": 2,
    "balance50d50i_smallpenalties_smallbatch_obs2": 2,
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
    "balance5050_reward_policy_obs2": 2,
    "balance0d100i_smallpenalties_wGAE_gamma9995_obs2": 2,
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
    "imaging_reward_smalldata_smallbat_obs1": 1,
    "balance0d100i_nopenalties_wGAE_restricted_Resources_obs2": 2,
    "balance0d100i_nopenalties_wGAE_restricted_Resources_obs2_1754513855": 2,
    "balance0d100i_nopenalties_wGAE_restricted_Resources_obs2_5e6lr": 2,
    "balance0d100i_nopenalties_wGAE_unlimitedResources_obs2": 2,
    "balance0d100i_nopenalties_woGAE_restricted_Resources_obs2": 2,
    "balance0d100i_nopenalties_woGAE_restricted_Resources_obs2_1754513855": 2,
    "balance0d100i_nopenalties_woGAE_unlimitedResources_obs2": 2,
    "balance0d100i_nopenalties_woGAE_unlimitedResources_obs2_1754513855": 2,
    "balance25d75i_smallpenalties_wGAE_restricted_Resources_obs2": 2,
    "balance50d50i_smallpenalties_wGAE_restricted_Resources_obs2": 2,
    "balance75d25i_smallpenalties_wGAE_restricted_Resources_obs2": 2,
    "balance100d0i_smallpenalties_wGAE_restricted_Resources_obs2": 2,
    "balance0d100i_smallpenalties_wGAE_restricted_Resources_obs2": 2,
    "balance0d100i_smallpenalties_wGAE_restricted_Resources_obs2_1754513855": 2,
    "balance75d25i_smallpenalties_wGAE_restricted_Resources_obs2_5e6lr": 2,
    "balance0d100i_smallpenalties_woGAE_restricted_Resources_obs2": 2,
    "balance25d75i_smallpenalties_woGAE_restricted_Resources_obs2": 2,
    "balance25d75i_smallpenalties_woGAE_restricted_Resources_obs2_1754513855": 2,
    "balance50d50i_smallpenalties_woGAE_restricted_Resources_obs2": 2,
    "balance50d50i_smallpenalties_woGAE_restricted_Resources_obs2_1754513855": 2,
    "balance75d25i_smallpenalties_woGAE_restricted_Resources_obs2": 2,
    "balance100d0i_smallpenalties_woGAE_restricted_Resources_obs2": 2,
    "imaging_unlimitedResources_baseline": 0,
}

# Compare policy_path to known variables
for name, val in list(globals().items()):
    if isinstance(val, str) and val == policy_path and name in policy_obs_map:
        policy_name = name
        obs_v = policy_obs_map[name]
        break

alpha = alpha_from_tag(policy_name, default=0.0)
print_alpha(policy_name,policy_path)

# Update obs_v both locally and in the shared sim config
if policy_name in policy_obs_map:
    obs_v = policy_obs_map[policy_name]
    sim_cfg.obs_v = obs_v
else:
    # fall back to whatever was in SimConfig
    obs_v = sim_cfg.obs_v

VALID_POLICY_MODES = {"best", "smallest", "latest"}

if policy_mode not in VALID_POLICY_MODES:
    raise ValueError(f"Invalid policy_mode '{policy_mode}'. Expected one of {VALID_POLICY_MODES}.")

policy = Policy(policy_path, policy_mode=policy_mode)



# Ground-station settings as a function of obs_v
GS_BY_OBS = {
    1:   dict(gs_start=87, n_gs=None),
    1.1: dict(gs_start=97, n_gs=None),
    2:   dict(gs_start=77, n_gs=None),
    2.1: dict(gs_start=77, n_gs=None),
    4:   dict(gs_start=74, n_gs=1),
    0:   dict(gs_start=74, n_gs=None),
    7:   dict(gs_start=77, n_gs=2),
}

gs_cfg = GS_BY_OBS.get(obs_v, dict(gs_start=GS_START, n_gs=None))
GS_START = gs_cfg["gs_start"]
if gs_cfg["n_gs"] is not None:
    N_GS = gs_cfg["n_gs"]


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
            obs.Eclipse(norm=5700),  # 5700
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=5,
            )
        ]
        GS_START = 87
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
            obs.Eclipse(norm=5700),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=5,
            )
        ]
        GS_START = 97
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
            obs.Eclipse(norm=1.0),    #update_train_Polaris locally was not normalized for eclipse...
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=5,
            )
        ]
    elif obs_v==2.1:
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
            obs.Eclipse(norm=5700),    #update_train_Polaris locally was not normalized for eclipse...
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm = 5700.0),
                dict(prop="opportunity_close", norm = 5700.0),
                type="ground_station",
                n_ahead_observe=5,
            )
        ]
    action_spec = [
        act.ImageRSO(
            n_ahead_image=n_targets_ahead,
            duration=imaging_duration,
            variable_duration_imaging=variable_duration_imaging,
            min_pointing_hold_s=sim_cfg.min_pointing_hold_s,
            hold_mode=sim_cfg.hold_mode,
            require_illumination_during_hold=sim_cfg.require_illumination_during_hold,
            hold_illumination_threshold=sim_cfg.hold_illumination_threshold,
        ),  # Scan for 5 minute
        act.Charge(duration=300.0),  # Charge for 5 minutes
        make_downlink_action(180.0), # Downlink for 3 min
        act.Desat(duration=150), # Desat for 2.5 min

    ]
    if obs_v==4:
        observation_spec = [
            obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
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
            act.ImageRSO(
                n_ahead_image=n_targets_ahead,
                duration=imaging_duration,
                variable_duration_imaging=variable_duration_imaging,
                min_pointing_hold_s=sim_cfg.min_pointing_hold_s,
                hold_mode=sim_cfg.hold_mode,
                require_illumination_during_hold=sim_cfg.require_illumination_during_hold,
                hold_illumination_threshold=sim_cfg.hold_illumination_threshold,
            ),  # Scan for 5 minute
            act.Charge(duration=300.0),  # Charge for 5 minutes
            make_downlink_action(300.0), # Downlink for 3 min
            act.Desat(duration=150), # Desat for 2.5 min.  # FOR OBS4 this DESAT may need to be removed!
        ]
        N_GS = 1
        GS_START = 74
    if obs_v==0:
        observation_spec = [
            obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                ),
            obs.PolarisScTargetProperties(
                dict(prop="target_elevation_angle", norm=1.0),
                dict(prop="angle_to_target", norm=1.0),
                dict(prop="rel_pos_vector_r_BR_N", norm = 1596*1000),
                dict(prop="target_distance", norm = 1596*1000), #normalization calculated assuming h = 800 km and min elevation is -14 deg
                dict(prop="target_imaged",  norm=1.0),
                n_ahead_observe=n_targets_ahead,
                                           ),
            obs.Eclipse(norm=1.0),
                ]
        action_spec = [
            act.ImageRSO(
                n_ahead_image=n_targets_ahead,
                duration=imaging_duration,
                variable_duration_imaging=variable_duration_imaging,
                min_pointing_hold_s=sim_cfg.min_pointing_hold_s,
                hold_mode=sim_cfg.hold_mode,
                require_illumination_during_hold=sim_cfg.require_illumination_during_hold,
                hold_illumination_threshold=sim_cfg.hold_illumination_threshold,
            ),  # Scan for 5 minute
            act.Charge(duration=300.0),  # Charge for 5 minutes
            ]
        GS_START =74
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
            act.ImageRSO(
                n_ahead_image=n_targets_ahead,
                duration=imaging_duration,
                variable_duration_imaging=variable_duration_imaging,
                min_pointing_hold_s=sim_cfg.min_pointing_hold_s,
                hold_mode=sim_cfg.hold_mode,
                require_illumination_during_hold=sim_cfg.require_illumination_during_hold,
                hold_illumination_threshold=sim_cfg.hold_illumination_threshold,
            ),  # Scan for 5 minute
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
            act.ImageRSO(
                n_ahead_image=n_targets_ahead,
                duration=imaging_duration,
                variable_duration_imaging=variable_duration_imaging,
                min_pointing_hold_s=sim_cfg.min_pointing_hold_s,
                hold_mode=sim_cfg.hold_mode,
                require_illumination_during_hold=sim_cfg.require_illumination_during_hold,
                hold_illumination_threshold=sim_cfg.hold_illumination_threshold,
            ),  # Scan for 5 minute
            act.Charge(duration=300.0),  # Charge for 5 minutes
            make_downlink_action(300.0), # Downlink for 3 min
            act.Desat(duration=150), # Desat for 2.5 min.  # FOR OBS4 this DESAT may need to be removed!

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

# MyScanningSatellite.default_sat_args() # why is this needed?

sat_args = {}

# Set some parameters as constants
sat_args["imageAttErrorRequirement"] = 0.0025
sat_args["imageRateErrorRequirement"] = 0.01
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
sat_args["downlink_bonus"] = alpha
sat_args["imaging_bonus"] = 1.0 - sat_args["downlink_bonus"]
sat_args["full_storage_penalty"] = 0
sat_args["low_battery_penalty"] = 0
sat_args["eclipse_threshold_for_imaging"] = 0.5
sat_args["eclipse_threshold_for_reward"] = sat_args["eclipse_threshold_for_imaging"]

# if just_imaging:
if sim_cfg.just_imaging:
    sat_args["dataStorageCapacity"] = 50 * 8e6 / 2 *1000000
    sat_args["batteryStorageCapacity"] = 500 * 3600 *1000000
    sat_args["storedCharge_Init"] = lambda: np.random.uniform(1.0, 1.0) * 500 * 3600 *1000000



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

# def custom_oe_randomizer():
#     rLEO = 6871. * 1000  #7000 * 1000   # Minimum semi-major axis (LEO) in meters
#     rUpperLEO =  8371.0 * 1000    # max semi-major axis  of upper LEO in meters
#     # rGEO = 42164. * 1000   # Maximum semi-major axis (GEO) in meters
#     oe = orbitalMotion.ClassicElements()
#     oe.a = np.random.uniform(1.00*rLEO, rUpperLEO)  # Random semi-major axis between LEO and GEO
#     if oe.a < 2*rLEO:
#         oe.e = np.random.uniform(0.0, 0.02)    # Random eccentricity (allowing less elliptical orbits when near LEO)
#         while oe.a*(1-oe.e) < 6771. * 1000: # perigee must be at least 400 km altitude
#             oe.e = np.random.uniform(0.0, 0.02)
#     else:
#         oe.e = np.random.uniform(0.0, 0.2)    # Random eccentricity (allowing slightly elliptical orbits)
#     oe.i = np.random.uniform(0, 180) * macros.D2R  # Random inclination up to 180 degrees
#     oe.Omega = np.random.uniform(0, 360) * macros.D2R  # Random RAAN
#     oe.omega = np.random.uniform(0, 360) * macros.D2R  # Random argument of perigee
#     oe.f = np.random.uniform(0, 360) * macros.D2R  # Random true anomaly
#     return oe


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
            regimes, probs = ["LEO", "MEO", "GEO"], np.array([0.5, 0.3, 0.2])
        else:
            regimes = ["LEO", "MEO", "GEO"]
            probs = np.array([mix_weights.get(r, 0.0) for r in regimes], dtype=float)
            if probs.sum() <= 0:
                raise ValueError("mix_weights must include positive weights.")
            probs = probs / probs.sum()
        regime = np.random.choice(regimes, p=probs)

    return _sample_for_regime(regime.upper(), altitude_bounds, min_perigee_alt)


# # target_args=dict(oe=custom_oe_randomizer, batteryStorageCapacity = 80.0 * 3600.0*1000, storedCharge_Init = 80.0 * 3600.0*900 )
# target_args=dict(oe=custom_oe_randomizer, batteryStorageCapacity = 1, storedCharge_Init = 0.0, basePowerDraw = -10000.0 )  # testing to see if sim is faster if the other agents are killed
# target_args_mixed = dict(oe=partial(custom_oe_randomizer, regime="mixed", mix_weights={"LEO":0.5,"MEO":0.3,"GEO":0.2}), batteryStorageCapacity = 1, storedCharge_Init = 0.0, basePowerDraw = -10000.0 )
# targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)] # TODO: this creates the same IC of oe for all targets


# Make the satellite
# sat = MyScanningSatellite(name="SS1", sat_args=sat_args, obs_type=dict) # SO1 for satellite observer 1
sat = MyScanningSatellite(name="SS1", sat_args=sat_args) # SO1 for satellite observer 1

base_target_args = dict(
    batteryStorageCapacity=1,
    storedCharge_Init=0.0,
    basePowerDraw=-10000.0,
)
if TARGET_ENV == "mixed":
    target_args = dict(
        oe=partial(custom_oe_randomizer, regime="mixed", mix_weights=MIX_WEIGHTS),
        **base_target_args,
    )
else:
    # default LEO (or your default oe randomizer)
    target_args = dict(
        oe=custom_oe_randomizer,
        **base_target_args,
    )
targets = [MyTargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(n_targets)]


all_sat = [sat] + targets   #oe = lambda: random_orbit(alt=np.random.uniform(1000,2000)))
if save_vizard == True:
    env = gym.make(
        "ConstellationTasking-v1",
        satellites=all_sat,
        scenario=make_rso_scenario(),
        rewarder=make_rso_rewarder(),
        world_type=world.GroundStationWorldModel,
        time_limit=total_time,
        log_level="WARNING", #ERROR or DEBUG
        disable_env_checker=True,
        vizard_dir="/Users/dahu1128/Documents",
        vizard_settings=dict(vizard_rate=viz_rate), # in seconds
        # max_step_duration=700,
    )
else:
    env = gym.make(
        "ConstellationTasking-v1",
        satellites=all_sat,
        scenario=make_rso_scenario(),
        rewarder=make_rso_rewarder(),
        world_type=world.GroundStationWorldModel,
        time_limit=total_time,
        log_level="WARNING", #ERROR or DEBUG
        disable_env_checker=True,
        # vizard_settings=dict(vizard_rate=viz_rate), # in seconds
        # max_step_duration=700,
    )


observation, info = env.reset(seed=seed_number) #5

sat0 = env.unwrapped.satellites[0]

try:
    ins_cmd_rec = sat0.fsw.ins_cmd_recorder
    print('ins_cmd_rec set up in fsw!')
except Exception:
    # Fallback: attach recorder here (works if insControl exists after reset)
    ins_cmd_rec = sat0.fsw.insControl.deviceCmdOutMsg.recorder(macros.sec2nano(1.0))
    env.simulator.AddModelToTask("locPointTask" + sat0.name, ins_cmd_rec, ModelPriority=980)
    print('ins_cmd_rec NOT set up in fsw!')

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
step_log = []  # each element is a dict (one per env.step)

# Ground-station windows: dict[str, list[np.ndarray([open_abs, close_abs])]]
ground_station_windows = {}         # e.g., {'ground_station_0': [np.array([t_open, t_close]), ...], ...}
_last_gs_pair = {}                  # e.g., {'ground_station_0': np.array([last_open, last_close])}

# Eclipse windows (from observation 'eclipse' field)
eclipse_windows = []                # list of np.array([open_abs, close_abs]) for the next eclipse window
_last_eclipse_pair = None           # np.array([last_open, last_close])


SS1_reward=0
SS1_reward_over_time = []

for target_id in range(n_targets*6 *100 ):
    simtime = env.simulator.sim_time
    print(f"\n SIMULATION TIME: {simtime:.1f} seconds and current reward: {SS1_reward:.2f}")

    policy_action = policy.act(observation[sat.name])
    if isinstance(policy_action, np.ndarray):  # Handle vector action output
        policy_action = policy_action.item()  # conversion
    if act_random:
        random_action = np.random.randint(0,13)
        action_dict = {sat.name: random_action}
        action_counts[random_action] += 1
    else:
        action_counts[policy_action] += 1


    # action_dict = {sat.name: target_id}  # Assign the main satellite to observe `target_idx` # sequentially observing each target
    action_dict = {sat.name: 0}  # Assign the closest target when the list is sorted by distance
    chosen_action_id = policy_action # Assign policy_action to action dictionary for env.step

    if use_heuristic:
        policy_action=0 #assign action 0 to heuristic
        action_dict = {sat.name: policy_action}
    else:
        action_dict = {sat.name: policy_action}
    if policy_action == 11:
        print('tasking DOWNLINKING now: at t=',simtime," and storage level --> "+str(env.unwrapped.satellites[0].dynamics.storage_level_fraction))
        downlink_times.append(env.simulator.sim_time)

    elif policy_action == 10:
        print('tasking CHARGING now: at t=',simtime," and battery level --> "+str(env.unwrapped.satellites[0].dynamics.battery_charge_fraction))
        charging_times.append(env.simulator.sim_time)
    elif policy_action == 12:
        print('tasking DESAT now: at t=',simtime," and wheel_speeds --> "+str(env.unwrapped.satellites[0].dynamics.wheel_speeds_fraction))
        desat_times.append(env.simulator.sim_time)
    if use_shield == True:
        if env.unwrapped.satellites[0].dynamics.storage_level_fraction > critical_storage_level:  # downlink if storage is more than 0.95
            print('tasking DOWNLINKING now: at t=',simtime," and storage level --> "+str(env.unwrapped.satellites[0].dynamics.storage_level_fraction))
            chosen_action_id = 11
            action_dict = {sat.name: chosen_action_id} # tasking downlink
            last_downlink_time = simtime
            downlink_times.append(env.simulator.sim_time)

        if env.unwrapped.satellites[0].dynamics.battery_charge_fraction < critical_battery_level:  # charge if battery is less than 0.05
            print('tasking CHARGING now: at t=',simtime," and battery level --> "+str(env.unwrapped.satellites[0].dynamics.battery_charge_fraction))
            chosen_action_id = 10
            action_dict = {sat.name: chosen_action_id} # tasking charging
            charging_times.append(env.simulator.sim_time)

    action_dict.update({targets[j].name: 0 for j in range(n_targets)})  # Initialize all targets to 0
    print('current action_dict to be executed', action_dict['SS1'], "eclipse status of SS1:",env.unwrapped.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[env.unwrapped.satellites[0].dynamics.eclipse_index].read().shadowFactor)
    eclipse_status.append(env.unwrapped.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[env.unwrapped.satellites[0].dynamics.eclipse_index].read().shadowFactor)

    sat0 = env.satellites[0]
    sat_shadow_cmd = float(sat0.dynamics.world.eclipseObject.eclipseOutMsgs[sat0.dynamics.eclipse_index].read().shadowFactor)

    step_log.append({
        "t_cmd": float(simtime),
        "action_id": int(chosen_action_id if "chosen_action_id" in locals() else policy_action),
        "sat_shadow_cmd": sat_shadow_cmd,
        "battery_frac_cmd": float(sat0.dynamics.battery_charge_fraction),
        "storage_frac_cmd": float(sat0.dynamics.storage_level_fraction),
    })

    #STEPPING IN THE SIM
    observation, reward, terminated, truncated, info = env.step(action=action_dict)

    SS1_reward += float(reward["SS1"])

    sat_shadow_after = float(sat0.dynamics.world.eclipseObject.eclipseOutMsgs[sat0.dynamics.eclipse_index].read().shadowFactor)
    step_log[-1].update({
        "t_after": float(env.simulator.sim_time),
        "sat_shadow_after": sat_shadow_after,
        "battery_frac_after": float(sat0.dynamics.battery_charge_fraction),
        "storage_frac_after": float(sat0.dynamics.storage_level_fraction),
        "reward_step": float(reward["SS1"]),
        "reward_cum": float(SS1_reward),
    })


    battery_levels.append(env.unwrapped.satellites[0].dynamics.battery_charge_fraction)
    storage_levels.append(env.unwrapped.satellites[0].dynamics.storage_level_fraction)
    sim_times.append(env.simulator.sim_time)
    num_imaged.append(len(env.env.unwrapped.rewarder.imaged_illuminated))
    num_downlinked.append(env.env.unwrapped.rewarder.useful_downlinks)

    SS1_reward_over_time.append(SS1_reward)
    print("storage_level", env.unwrapped.satellites[0].dynamics.storage_level)
    print("dynamics.storage_level_fraction", env.unwrapped.satellites[0].dynamics.storage_level_fraction)
    print("dynamics.battery_charge_fraction", env.unwrapped.satellites[0].dynamics.battery_charge_fraction)
    print("dynamics.wheel_speeds_fraction", env.unwrapped.satellites[0].dynamics.wheel_speeds_fraction)

    # print('truncated list: ', truncated)
    data_dict["sim_time"].append(env.simulator.sim_time)
    if all(truncated.values()) or all(terminated.values()) or len(env.env.unwrapped.satellites[0].data_store.data.imaged)==n_targets:
        if len(env.env.unwrapped.satellites[0].data_store.data.imaged) == n_targets:
            print('ALL targets imaged!')
        else:
            simtime = env.simulator.sim_time
            print("Episode terminated at time: {simtime}")
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

rec = env.unwrapped.satellites[0].dynamics.inspector_eclipse_recorder
ecl_sf = np.asarray(rec.shadowFactor, dtype=float).ravel()   # length ~ 45000
t_ecl_sf = range(len(ecl_sf))

# Try to use recorder timestamps if present; otherwise assume 1 Hz from t=0
try:
    t_raw = np.asarray(rec.times, dtype=float).ravel()       # some recorders expose .times
    ecl_t = t_raw*macros.NANO2SEC if t_raw.max() > 1e6 else t_raw
except Exception:
    ecl_t = np.arange(ecl_sf.size, dtype=float)              # 0,1,2,... seconds



#compute "actual acquisition times" from deviceCmd rising edges
dev_cmd = np.asarray(ins_cmd_rec.deviceCmd, dtype=float).ravel()

# Recorder timestamps (try to use .times if present; otherwise assume 1 Hz)
try:
    t_raw = np.asarray(ins_cmd_rec.times, dtype=float).ravel()
    t_dev = t_raw * macros.NANO2SEC if t_raw.size and t_raw.max() > 1e6 else t_raw
except Exception:
    # fallback: assume 1 Hz samples starting at 0
    t_dev = np.arange(dev_cmd.size, dtype=float)

# Rising edges where deviceCmd goes 0 -> 1
if dev_cmd.size >= 2:
    edges = np.where((dev_cmd[1:] > 0.5) & (dev_cmd[:-1] <= 0.5))[0] + 1
else:
    edges = np.array([], dtype=int)

acq_event_times = t_dev[edges]  # "capture triggered" times

# Command times from your ImageRSO action spec (these are when you issued the imaging action)
SS1_actions_spec = env.unwrapped.satellites[0].action_builder.action_spec[0]
cmd_times = np.asarray(SS1_actions_spec.imaging_times, dtype=float).ravel()
cmd_target_ids = np.asarray(SS1_actions_spec.chosen_target_ids, dtype=int).ravel()
assert cmd_times.size == cmd_target_ids.size, "cmd_times and cmd_target_ids must align"


# Align each command with the next acquisition event within the imaging window
imaging_window = float(imaging_duration)  # 300s in your sim config

acq_times_for_cmd = np.full(cmd_times.shape, np.nan, dtype=float)
acq_dt_for_cmd = np.full(cmd_times.shape, np.nan, dtype=float)
acq_success = np.zeros(cmd_times.shape, dtype=bool)

j = 0
for i, t0 in enumerate(cmd_times):
    # advance acquisition pointer to the first event after command time
    while j < acq_event_times.size and acq_event_times[j] <= t0:
        j += 1

    # accept the next event if it occurs within the imaging window
    if j < acq_event_times.size and acq_event_times[j] <= (t0 + imaging_window + 1e-6):
        acq_times_for_cmd[i] = acq_event_times[j]
        acq_dt_for_cmd[i] = acq_event_times[j] - t0
        acq_success[i] = True
        j += 1

# Convenience aliases used by later logging
t_acq  = np.asarray(acq_times_for_cmd, dtype=float).ravel()
dt_acq = np.asarray(acq_dt_for_cmd, dtype=float).ravel()
acq_ok = np.asarray(acq_success, dtype=bool).ravel()
# --- target shadowFactor at acquisition time (per command index) ---
target_shadow_acq = np.full_like(t_acq, np.nan, dtype=float)

if np.any(acq_ok):
    _t_cache = {}
    _sf_cache = {}

    for i in np.where(acq_ok)[0]:
        tid = int(cmd_target_ids[i])  # must align with cmd_times indexing

        # mapping: sat[0] is inspector, targets are sat[tid+1]
        try:
            tgt_sat = env.unwrapped.satellites[tid + 1]
            rec = tgt_sat.dynamics.target_eclipse_recorder
        except Exception:
            continue

        if tid not in _t_cache:
            sf = np.asarray(rec.shadowFactor, dtype=float).ravel()
            try:
                t_raw = np.asarray(rec.times, dtype=float).ravel()
                tt = t_raw * macros.NANO2SEC if (t_raw.size and t_raw.max() > 1e6) else t_raw
            except Exception:
                tt = np.arange(sf.size, dtype=float)

            _t_cache[tid] = tt
            _sf_cache[tid] = sf

        tt = _t_cache[tid]
        sf = _sf_cache[tid]
        target_shadow_acq[i] = float(np.interp(float(t_acq[i]), tt, sf))


# Compute spacecraft shadowFactor at command and at acquisition (interpolate eclipse recorder)
sat_sf_cmd = np.interp(cmd_times, ecl_t, ecl_sf) if cmd_times.size else np.array([])
sat_sf_acq = np.interp(acq_times_for_cmd[acq_success], ecl_t, ecl_sf) if np.any(acq_success) else np.array([])

# Summary metrics
avg_acq_time_sec = float(np.nanmean(acq_dt_for_cmd)) if np.any(acq_success) else float("nan")
median_acq_time_sec = float(np.nanmedian(acq_dt_for_cmd)) if np.any(acq_success) else float("nan")
acq_success_rate = float(np.mean(acq_success)) if cmd_times.size else float("nan")

tau_umbra = 0.05
pct_acq_in_umbra = float(np.mean(sat_sf_acq <= tau_umbra)) if sat_sf_acq.size else float("nan")
pct_cmd_in_umbra = float(np.mean(sat_sf_cmd <= tau_umbra)) if sat_sf_cmd.size else float("nan")

# store raw series for plotting later
data_dict["image_command_times"] = cmd_times.tolist()
data_dict["image_command_target_ids"] = cmd_target_ids.tolist()
data_dict["image_acq_times"] = acq_times_for_cmd.tolist()
data_dict["image_acq_dt"] = acq_dt_for_cmd.tolist()
data_dict["image_acq_success"] = acq_success.astype(int).tolist()

# build images.csv table

SS1_actions_spec = env.unwrapped.satellites[0].action_builder.action_spec[0]

img_cmd_times = np.asarray(SS1_actions_spec.imaging_times, dtype=float).ravel()
img_target_ids = np.asarray(SS1_actions_spec.chosen_target_ids, dtype=int).ravel()

az = np.asarray(getattr(SS1_actions_spec, "chosen_target_azimuth", []), dtype=float).ravel()
# el_loc = np.asarray(getattr(SS1_actions_spec, "chosen_target_elevation_local", []), dtype=float).ravel()
el_loc = np.asarray(getattr(SS1_actions_spec, "chosen_target_elevation_angle", []), dtype=float).ravel()
rng = np.asarray(getattr(SS1_actions_spec, "chosen_target_distance", []), dtype=float).ravel()
tgt_sf_cmd = np.asarray(getattr(SS1_actions_spec, "chosen_target_illumination_status", []), dtype=float).ravel()

# Look-ahead index: +1 forward, -1 backward
# azimuth is in degrees
look_ahead = np.cos(np.deg2rad(az)) if az.size else np.array([])
UMBRA_TAU = 0.05
SUNLIT_TAU = 0.95

# shadowFactor at command time (satellite)
sat_sf_cmd_img = np.interp(img_cmd_times, ecl_t, ecl_sf) if img_cmd_times.size else np.array([])

def _shadow(t):
    return float(np.interp(t, ecl_t, ecl_sf))

delta = 60.0
eps = 0.05

phase = []          # combined, interpretable label
phase_state = []    # sunlit / umbra / penumbra
phase_slope = []    # entering / exiting / flat_slope
ds_list = []        # optional: keep the slope value for debugging/analysis

for t, s_cmd in zip(img_cmd_times, sat_sf_cmd_img):
    t0 = max(float(ecl_t[0]), float(t - delta))
    t1 = min(float(ecl_t[-1]), float(t + delta))
    ds = _shadow(t1) - _shadow(t0)
    ds_list.append(ds)

    # illumination state at command time
    if s_cmd <= UMBRA_TAU:
        st = "umbra"
    elif s_cmd >= SUNLIT_TAU:
        st = "sunlit"
    else:
        st = "penumbra"

    # transition direction (slope)
    if ds < -eps:
        sl = "entering"
    elif ds > eps:
        sl = "exiting"
    else:
        sl = "flat_slope"

    phase_state.append(st)
    phase_slope.append(sl)

    # combined label (this is the one you’ll plot most of the time)
    if st in ("umbra", "sunlit"):
        phase.append(st)   # keep it simple: umbra or sunlit
    else:
        # only in penumbra do we want entering/exiting; otherwise it's ambiguous
        if sl in ("entering", "exiting"):
            phase.append(sl)
        else:
            phase.append("penumbra_flat")


 # WINDOW METRICS (sat eclipse during imaging action window)
imaging_window = float(imaging_duration)  # 300s from config
tau_umbra = UMBRA_TAU
tau_sunlit = SUNLIT_TAU

# Helper: window min/max of shadowFactor using recorder samples
# We assume ecl_t is monotonically increasing and in seconds.
ecl_t_arr = np.asarray(ecl_t, dtype=float)
ecl_sf_arr = np.asarray(ecl_sf, dtype=float)

sat_shadow_cmd_img = np.interp(img_cmd_times, ecl_t_arr, ecl_sf_arr) if img_cmd_times.size else np.array([])

sat_shadow_min_win = np.full(img_cmd_times.shape, np.nan, dtype=float)
sat_shadow_max_win = np.full(img_cmd_times.shape, np.nan, dtype=float)
win_bucket = np.array(["UNK"] * img_cmd_times.size, dtype=object)

for i, t0 in enumerate(img_cmd_times):
    t1 = t0 + imaging_window

    # indices of recorder samples within [t0, t1]
    # using searchsorted keeps this O(log N) per command
    j0 = int(np.searchsorted(ecl_t_arr, t0, side="left"))
    j1 = int(np.searchsorted(ecl_t_arr, t1, side="right"))

    if j0 >= ecl_t_arr.size:
        continue
    if j1 <= j0:
        # no sample strictly inside window; fallback to interpolation at endpoints
        s0 = float(np.interp(t0, ecl_t_arr, ecl_sf_arr))
        s1 = float(np.interp(t1, ecl_t_arr, ecl_sf_arr))
        s_min = min(s0, s1)
        s_max = max(s0, s1)
    else:
        seg = ecl_sf_arr[j0:j1]
        if seg.size == 0:
            continue
        s_min = float(np.min(seg))
        s_max = float(np.max(seg))

    sat_shadow_min_win[i] = s_min
    sat_shadow_max_win[i] = s_max

    s_cmd = float(sat_shadow_cmd_img[i]) if i < sat_shadow_cmd_img.size else float("nan")

    # Bucket definitions
    if np.isfinite(s_min) and s_max <= tau_umbra:
        win_bucket[i] = "always_umbra"
    elif np.isfinite(s_max) and s_min >= tau_sunlit:
        win_bucket[i] = "always_sunlit"
    elif np.isfinite(s_cmd) and (s_cmd >= tau_sunlit) and (s_min <= tau_umbra):
        win_bucket[i] = "sunlit_to_umbra"
    elif np.isfinite(s_cmd) and (s_cmd <= tau_umbra) and (s_max >= tau_sunlit):
        win_bucket[i] = "umbra_to_sunlit"
    else:
        win_bucket[i] = "mixed_penumbra"


sat_sf_acq_img = np.full_like(t_acq, np.nan, dtype=float)
if np.any(acq_ok):
    sat_sf_acq_img[acq_ok] = np.interp(t_acq[acq_ok], ecl_t, ecl_sf)

# Optional: include orbit regime if you add it later in discrete_actions.py
alt_km = np.asarray(getattr(SS1_actions_spec, "chosen_target_alt_km", []), dtype=float).ravel()
regime = np.asarray(getattr(SS1_actions_spec, "chosen_target_orbit_regime", []), dtype=object).ravel()
if alt_km.size == 0:
    alt_km = np.full(img_cmd_times.shape, np.nan, dtype=float)
if regime.size == 0:
    regime = np.array(["UNK"] * img_cmd_times.size, dtype=object)

# Build dataframe (trim to the shortest common length to be safe)
L = min(img_cmd_times.size, img_target_ids.size, az.size, el_loc.size, rng.size, tgt_sf_cmd.size, t_acq.size, dt_acq.size, look_ahead.size, sat_sf_cmd_img.size, regime.size, alt_km.size, sat_shadow_min_win.size, sat_shadow_max_win.size, win_bucket.size, target_shadow_acq.size)

df_images = pd.DataFrame({
    "t_cmd": img_cmd_times[:L],
    "target_id": img_target_ids[:L],
    "azimuth_deg": az[:L],
    "elevation_local_deg": el_loc[:L],
    "range_m": rng[:L],
    "target_shadow_cmd": tgt_sf_cmd[:L],
    "sat_shadow_cmd": sat_sf_cmd_img[:L],
    "phase": np.array(phase, dtype=object)[:L],
    "phase_state": np.array(phase_state, dtype=object)[:L],
    "phase_slope": np.array(phase_slope, dtype=object)[:L],
    "ecl_ds": np.array(ds_list, dtype=float)[:L],
    "look_ahead": look_ahead[:L],
    "t_acq": t_acq[:L],
    "dt_acq": dt_acq[:L],
    "acq_success": acq_ok[:L].astype(int),
    "sat_shadow_acq": sat_sf_acq_img[:L],
    "target_alt_km": alt_km[:L],
    "target_regime": regime[:L],
    "sat_shadow_min_win": sat_shadow_min_win[:L],
    "sat_shadow_max_win": sat_shadow_max_win[:L],
    "win_bucket": win_bucket[:L],
    "target_shadow_acq": target_shadow_acq[:L],
})


print("df_images columns:", list(df_images.columns))
print("win_bucket counts:\n", df_images["win_bucket"].value_counts(dropna=False))
print("non-nan target_shadow_acq:", np.isfinite(df_images["target_shadow_acq"].to_numpy(dtype=float)).sum())

images_csv = os.path.join(run_dir, "images.csv")
df_images.to_csv(images_csv, index=False)
print(f"Saved: {images_csv}")

def _mean_safe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")

def _frac_safe(mask):
    mask = np.asarray(mask, dtype=bool)
    return float(np.mean(mask)) if mask.size else float("nan")

tau_umbra = UMBRA_TAU
tau_sunlit = SUNLIT_TAU

df = df_images.copy()

# Helpful masks
m_umbra = df["sat_shadow_cmd"].astype(float) <= tau_umbra
m_sunlit = df["sat_shadow_cmd"].astype(float) >= tau_sunlit
m_enter = (df["phase_slope"].astype(str) == "entering")
m_exit  = (df["phase_slope"].astype(str) == "exiting")

look = df["look_ahead"].astype(float).to_numpy()

# Interpretable "behavior" rates
enter_lookback_rate = _frac_safe((look < 0) & m_enter.to_numpy())
exit_lookahead_rate = _frac_safe((look > 0) & m_exit.to_numpy())

# Conditional means (these are what you’ll MC-average across seeds)
look_metrics = {
    "N_img_cmd": int(len(df)),
    "N_entering": int(m_enter.sum()),
    "N_exiting": int(m_exit.sum()),
    "N_umbra_cmd": int(m_umbra.sum()),
    "N_sunlit_cmd": int(m_sunlit.sum()),

    "E_lookAhead_all": _mean_safe(look),
    "E_lookAhead_umbra": _mean_safe(df.loc[m_umbra, "look_ahead"]),
    "E_lookAhead_sunlit": _mean_safe(df.loc[m_sunlit, "look_ahead"]),
    "E_lookAhead_entering": _mean_safe(df.loc[m_enter, "look_ahead"]),
    "E_lookAhead_exiting": _mean_safe(df.loc[m_exit, "look_ahead"]),

    "P_lookBack_given_entering": enter_lookback_rate,
    "P_lookAhead_given_exiting": exit_lookahead_rate,
}




# Regime metrics (requires target_regime column populated)
if "target_regime" in df.columns:
    # Overall selection distribution by regime
    vc = df["target_regime"].value_counts(dropna=False)
    total = float(vc.sum()) if vc.size else 0.0
    frac_by_regime = {str(k): float(v)/total for k, v in vc.items()} if total > 0 else {}

    # Regime distribution during umbra
    vc_u = df.loc[m_umbra, "target_regime"].value_counts(dropna=False)
    total_u = float(vc_u.sum()) if vc_u.size else 0.0
    frac_by_regime_umbra = {str(k): float(v)/total_u for k, v in vc_u.items()} if total_u > 0 else {}

    # Acquisition time by regime (success only)
    if "acq_success" in df.columns and "dt_acq" in df.columns:
        succ = df["acq_success"].astype(int) == 1
        g = df.loc[succ].groupby("target_regime")["dt_acq"]
        dt_mean = {str(k): _mean_safe(v) for k, v in g}
    else:
        dt_mean = {}

    # Illumination at command by regime (uses target_shadow_cmd)
    if "target_shadow_cmd" in df.columns:
        g2 = df.groupby("target_regime")["target_shadow_cmd"]
        illum_mean = {str(k): _mean_safe(v) for k, v in g2}
    else:
        illum_mean = {}

    regime_metrics = {
        "frac_target_regime_all": frac_by_regime,
        "frac_target_regime_umbra": frac_by_regime_umbra,
        "mean_dt_acq_success_by_regime": dt_mean,
        "mean_target_shadow_cmd_by_regime": illum_mean,
    }
else:
    regime_metrics = {}




# keep x-limit consistent with the longest series plotted
tmax = float(ecl_t[-1])
if sim_times:
    tmax = max(tmax, float(np.max(sim_times)))

print("  Final data level:", observation)
print(f"final reward for SS1 {SS1_reward} should be the same as {env.env.unwrapped.rewarder.cum_reward['SS1']}")
print(f"and number of imaged targets {len(env.env.unwrapped.satellites[0].data_store.data.imaged)} out of those useful images were: {len(env.env.unwrapped.rewarder.imaged_illuminated)}")
print(f"Total downlinked {env.env.unwrapped.rewarder.total_downlinks} out of those useful downlinks were: {env.env.unwrapped.rewarder.useful_downlinks}")
# print(f"mean and std of chosen_target_azimuth {env.env.unwrapped.satellites[0].action_builder.action_spec[0].chosen_target_azimuth}")

SS1_actions_spec = env.unwrapped.satellites[0].action_builder.action_spec[0]
print(f"mean and std of chosen_target_azimuth: {np.mean(SS1_actions_spec.chosen_target_azimuth):.2f}, {np.std(SS1_actions_spec.chosen_target_azimuth):.2f}")
print(f"mean and std of chosen_target_elevation: {np.mean(SS1_actions_spec.chosen_target_elevation_angle):.2f}, {np.std(SS1_actions_spec.chosen_target_elevation_angle):.2f}")
print(f"mean and std of chosen_target_elevation_local: {np.mean(SS1_actions_spec.chosen_target_elevation_local):.2f}, {np.std(SS1_actions_spec.chosen_target_elevation_local):.2f}")
print(f"mean and std of chosen_target_distance: {np.mean(SS1_actions_spec.chosen_target_distance):.2f}, {np.std(SS1_actions_spec.chosen_target_distance):.2f}")
print(f"mean and std of initial angular error: {np.mean(SS1_actions_spec.initial_angular_error):.2f}, {np.std(SS1_actions_spec.initial_angular_error):.2f}")
print(f"mean and std of chosen_target_priority: {np.mean(SS1_actions_spec.chosen_target_priority):.2f}, {np.std(SS1_actions_spec.chosen_target_priority):.2f}")
print("\n=== Imaging Acquisition Timing (actual capture trigger) ===")
print(f"Imaging commands issued: {cmd_times.size}")
print(f"Acquisition success rate: {acq_success_rate:.3f}")
print(f"Average acquisition time [s]: {avg_acq_time_sec:.2f}")
print(f"Median acquisition time [s]: {median_acq_time_sec:.2f}")
print(f"% commands in umbra (SF<=0.05): {pct_cmd_in_umbra:.3f}")
print(f"% acquisitions in umbra (SF<=0.05): {pct_acq_in_umbra:.3f}")
print("==========================================================\n")
print(f"fraction of targets that were illuminated: {np.mean(SS1_actions_spec.chosen_target_illumination_status):.2f}")
print(f"fraction of targets ever visible: {len(SS1_actions_spec.ever_visible)/n_targets:.2f}")
print(f"mean and std of rel pos in H-frame: {np.mean(SS1_actions_spec.chosen_target_rel_pos_H, axis=0)}, {np.std(SS1_actions_spec.chosen_target_rel_pos_H, axis=0)}")

print("Target Selection comparison:", env.unwrapped.satellites[0].dynamics.target_selection_comparison)
# Count only the matches (non-False entries)
num_same = np.count_nonzero(np.array(env.unwrapped.satellites[0].dynamics.target_selection_comparison) != False)
print("Target Selection comparison numbers:", num_same)
num_diff = np.count_nonzero(np.array(env.unwrapped.satellites[0].dynamics.target_selection_comparison) == False)
print("Number different:", num_diff)


data_dict["inspector_sigmaBN"].append(env.unwrapped.satellites[0].dynamics.inspector_state_recorder.sigma_BN)
data_dict["inspector_omegaBN"].append(env.unwrapped.satellites[0].dynamics.inspector_state_recorder.omega_BN_B)
data_dict["inspector_r_BN_N"].append(env.unwrapped.satellites[0].dynamics.inspector_state_recorder.r_BN_N)
data_dict["currentTarget_r_BN_N"].append(env.unwrapped.satellites[0].dynamics.simpleNavObject.transOutMsg.read().r_BN_N)

for l in range (len(targets)):
    data_dict["target_r_BN_N"][targets[l].name].append(env.unwrapped.satellites[l+1].dynamics.target_state_recorder.r_BN_N)

# ---- Plotting ----
total_actions = sum(action_counts.values())
num_actions = 13 # 13
actions = list(range(num_actions))  # Actions 0–12
action_labels = [
    f"Target {i}" if i <= 9 else ["Charging", "Downlink", "Desat"][i - 10]
    for i in actions
]
# action_labels = [
#     f"Target {i+1}" for i in actions
# ]
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

# plt.title("Action Distribution: Count and Percentage")
plt.tight_layout()
if use_shield:
    if use_heuristic:
        save_plot_unique(fig, f"seed{seed_number}_HEURISTIC+SHIELD_action_distribution_combined")
    else:
        save_plot_unique(fig, f"seed{seed_number}_{policy_mode}_{policy_name}+SHIELD_action_distribution_combined")
else:
    if use_heuristic:
        save_plot_unique(fig, f"seed{seed_number}_HEURISTIC_action_distribution_combined")
    else:
        save_plot_unique(fig, f"seed{seed_number}_{policy_mode}_{policy_name}_action_distribution_combined")
# plt.show()
plt.close(fig)

# --- Compute eclipse & penumbra spans from shadowFactor ---
# shadowFactor: 1 = lit, 0 < sf < 1 = penumbra, 0 = umbra
umbra_spans = []
penumbra_spans = []

if len(ecl_sf) == len(t_ecl_sf):
    def _val(x):
        try: return float(getattr(x, "shadowFactor", x))
        except Exception: return 1.0

    def is_umbra(v):    return np.isclose(v, 0.0, atol=1e-6)
    def is_lit(v):      return v >= 1.0 - 1e-6
    def is_penumbra(v): return (v > 0.0 + 1e-6) and (v < 1.0 - 1e-6)

    in_umbra = False; t0_umb = None
    in_penu  = False; t0_pen = None

    for t, s in zip(t_ecl_sf, ecl_sf):
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
    t_last = t_ecl_sf[-1]
    if in_umbra and t0_umb is not None:
        umbra_spans.append((t0_umb, t_last))
    if in_penu and t0_pen is not None:
        penumbra_spans.append((t0_pen, t_last))
else:
    umbra_spans, penumbra_spans = [], []

######### PLOT2: metrics over time + cumulative reward
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
                     label='Groundstation window' if _first_gs_lbl else '')
    _first_gs_lbl = False

# Battery & storage on left y-axis
ax1.plot(sim_times, battery_levels,linewidth=1.4, label='Battery Level', color='tab:blue')
ax1.plot(sim_times, storage_levels, linewidth=1.4, label='Storage Level', color='tab:orange')
ax1.set_xlabel("Time [sec]" , fontsize = label_size)
ax1.set_ylabel("Battery and Storage Fraction", color='black', fontsize = label_size)
ax1.tick_params(axis='y', labelcolor='black', labelsize = tick_label_size)
ax1.tick_params(axis='x', labelcolor='black', labelsize = tick_label_size)
ax1.grid(True, linestyle='-.', alpha=0.4)

# X-limit at 45,000 s
ax1.set_xlim(0, np.max(sim_times))

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
ax2.plot(sim_times, num_imaged, label='Illuminated Images (cumulative)', color='tab:green')

# Mark DESAT events
if desat_times:
    ax1.axvline(desat_times[0], color='crimson', linestyle='--', linewidth=0.8, alpha=0.85, label='Desat')
    for t in desat_times[1:]:
        ax1.axvline(t, color='crimson', linestyle='--', linewidth=0.8, alpha=0.85)

ax2.plot(sim_times, num_downlinked, label='Downlinked Targets (cumulative)', color='tab:red')
# ax2.plot(sim_times, SS1_reward_over_time, label='Cumulative SS1 Reward', linestyle=':', linewidth=3.0, color='tab:purple')
ax2.set_ylabel("Cumulative Count", color='black', fontsize = label_size)
ax2.tick_params(axis='y', labelcolor='black', labelsize = tick_label_size)

# Align both y-axes at 0 and 1.0/100 respectively
ax1.set_ylim(top=1.0, bottom=0.0)
ax2.set_ylim(top=100, bottom=0.0)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=legend_fontsize)

# plt.title("Battery, Storage, Reward, Imaging and Action Events Over Time")
plt.tight_layout()
if use_shield:
    if use_heuristic:
        save_plot_unique(fig3, f"seed{seed_number}_HEURISTIC+SHIELD_battery_storage_reward_over_time")
    else:
        save_plot_unique(fig3, f"seed{seed_number}_{policy_mode}_{policy_name}+SHIELD_battery_storage_reward_over_time")
else:
    if use_heuristic:
        save_plot_unique(fig3, f"seed{seed_number}_HEURISTIC_battery_storage_reward_over_time")
    else:
        save_plot_unique(fig3, f"seed{seed_number}_{policy_mode}_{policy_name}_battery_storage_reward_over_time")
# plt.show()
plt.close(fig)

# Plot 3 Azimuth and Elevation angle (deg) vs time
# (minutes on x-axis; same merged shading converted to minutes)
minute = 1.0 # 60.0
imaging_times =  np.array(SS1_actions_spec.imaging_times)
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

_first_gs_lbl3 = True
_all_gs_spans = [span for spans in (gs_spans_merged or {}).values() for span in spans]
for (t0, t1) in _all_gs_spans:
    t0m, t1m = float(t0)/minute, float(t1)/minute
    ax1.fill_between([t0m, t1m], 0, 1, transform=ax1.get_xaxis_transform(),
                     color='green', alpha=gs_alpha, #zorder=0.03,
                     label='Groundstation window' if _first_gs_lbl3 else '')
    _first_gs_lbl3 = False

# Azimuth/Elevation
color1 = 'tab:blue'
ax1.set_xlabel('Time [sec]', fontsize = label_size)
ax1.set_ylabel('Azimuth [deg]', color=color1, fontsize = label_size)
ax1.plot(imaging_times, azimuths, 'o-', color=color1, label='Azimuth')
ax1.tick_params(axis='y', labelcolor=color1, labelsize = tick_label_size)
ax1.grid(True, linestyle='-.', color='0.65')
ax1.tick_params(axis='x', labelcolor='black', labelsize = tick_label_size)
ax1.set_xlim(0, np.max(sim_times)/minute)
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Elevation [deg]', color=color2 , fontsize = label_size)
ax2.plot(imaging_times, elevations, 'x--', color=color2, label='Elevation')
ax2.tick_params(axis='y', labelcolor=color2, labelsize = tick_label_size)


# plt.title('Pointing Directions During Episode')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=legend_fontsize)

plt.tight_layout()
if use_shield:
    if use_heuristic:
        save_plot_unique(fig4, f"seed{seed_number}_HEURISTIC+SHIELD_azimuth_and_elevation_pointing_over_time")
    else:
        save_plot_unique(fig4, f"seed{seed_number}_{policy_mode}_{policy_name}+SHIELD_azimuth_and_elevation_pointing_over_time")
else:
    if use_heuristic:
        save_plot_unique(fig4, f"seed{seed_number}_HEURISTIC_azimuth_and_elevation_pointing_over_time")
    else:
        save_plot_unique(fig4, f"seed{seed_number}_{policy_mode}_{policy_name}_azimuth_and_elevation_pointing_over_time")

# plt.show()
plt.close(fig)


# # ---- AMOS plotting helpers ----
# SHOW_TITLES = True  # << toggle titles globally (True/False)
#
# AMOS_FONTS = dict(
#     title=20,   # figure/axes titles
#     label=18,   # x/y label font size
#     tick=16,    # tick label size
#     legend=16,  # legend font size
# )
#
# def amosify_axes(ax, *, xlabel=None, ylabel=None, title=None, show_title=SHOW_TITLES):
#     """Apply AMOS-like font sizes and optional title to a single axes."""
#     if xlabel is not None:
#         ax.set_xlabel(xlabel, fontsize=AMOS_FONTS["label"])
#     if ylabel is not None:
#         ax.set_ylabel(ylabel, fontsize=AMOS_FONTS["label"])
#     ax.tick_params(axis='both', labelsize=AMOS_FONTS["tick"])
#     if show_title and (title is not None):
#         ax.set_title(title, fontsize=AMOS_FONTS["title"], pad=8)
#
# def amos_legend(ax, *args, **kwargs):
#     """Legend with AMOS font size."""
#     return ax.legend(*args, fontsize=AMOS_FONTS["legend"], **kwargs)
#
# import matplotlib as mpl
# mpl.rcParams.update({
#     "axes.titlesize": AMOS_FONTS["title"],
#     "axes.labelsize": AMOS_FONTS["label"],
#     "xtick.labelsize": AMOS_FONTS["tick"],
#     "ytick.labelsize": AMOS_FONTS["tick"],
#     "legend.fontsize": AMOS_FONTS["legend"],
# })
#
# # Combined plot 1
# fig, ax1 = plt.subplots(figsize=(11, 6))
#
# # Plot absolute counts on the left y-axis
# bars1 = ax1.bar(action_labels, counts, color="skyblue", label="Action Count")
# amosify_axes(ax1, ylabel="Number of Times Action Was Taken")
# ax1.tick_params(axis='y', labelcolor="black")
# ax1.set_xticks(range(len(action_labels)))
# ax1.set_xticklabels(action_labels, rotation=45, ha="right", fontsize=AMOS_FONTS["tick"])
#
# # Percentages on right y-axis
# ax2 = ax1.twinx()
# _ = ax2.bar(action_labels, percentages, color="mediumseagreen", alpha=0.000, label="Action Percentage")
# ax2.set_ylabel("Percentage of Total Actions (%)", color="mediumseagreen", fontsize=AMOS_FONTS["label"])
# ax2.tick_params(axis='y', labelcolor="black", labelsize=AMOS_FONTS["tick"])
#
# # Grid on primary y-axis only
# ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
#
# # Title (toggleable)
# amosify_axes(ax1, title="Action Distribution: Count and Percentage", show_title=SHOW_TITLES)
#
# plt.tight_layout()
#
# # Save and show
# if use_shield:
#     save_plot_unique(fig, f"seed{seed_number}_{policy_mode}_{policy_name}+SHIELD_action_distribution_combined")
# else:
#     save_plot_unique(fig, f"seed{seed_number}_{policy_mode}_{policy_name}_action_distribution_combined")
# plt.show()
#
# fig3, ax1 = plt.subplots(figsize=(12, 6))
#
# # --- shading as you had (umbra / GS windows) ---
# # (your existing shading code unchanged)
#
# # Battery & storage (left y-axis)
# ax1.plot(sim_times, battery_levels, linewidth=2.0, label='Battery Level', color='tab:blue')
# ax1.plot(sim_times, storage_levels, linewidth=2.0, label='Storage Level', color='tab:orange')
# amosify_axes(ax1, xlabel="Time [sec]", ylabel="Battery and Storage Fraction")
#
# # X-limit across full run
# ax1.set_xlim(0, np.max(sim_times))
# ax1.tick_params(axis='y', labelcolor='black')
#
# # Event markers (charge/downlink/desat)
# if charging_times:
#     ax1.axvline(charging_times[0], color='deepskyblue', linestyle='--', linewidth=1.3, alpha=0.85, label='Charge')
#     for t in charging_times[1:]:
#         ax1.axvline(t, color='deepskyblue', linestyle='--', linewidth=1.3, alpha=0.85)
# if downlink_times:
#     ax1.axvline(downlink_times[0], color='magenta', linestyle='--', linewidth=1.3, alpha=0.6, label='Downlink')
#     for t in downlink_times[1:]:
#         ax1.axvline(t, color='magenta', linestyle='--', linewidth=1.3, alpha=0.6)
# if desat_times:
#     ax1.axvline(desat_times[0], color='crimson', linestyle='--', linewidth=1.6, alpha=0.85, label='Desat')
#     for t in desat_times[1:]:
#         ax1.axvline(t, color='crimson', linestyle='--', linewidth=1.6, alpha=0.85)
#
# # Cumulative imaged/downlinked (right y-axis)
# ax2 = ax1.twinx()
# ax2.plot(sim_times, num_imaged,      label='Cumulative Imaged Targets',      color='tab:green', linewidth=2.0)
# ax2.plot(sim_times, num_downlinked,  label='Cumulative Downlinked Targets',  color='tab:red',   linewidth=2.0)
# ax2.set_ylabel("Cumulative Count", color='black', fontsize=AMOS_FONTS["label"])
# ax2.tick_params(axis='y', labelcolor='black', labelsize=AMOS_FONTS["tick"])
#
# # Axis limits
# ax1.set_ylim(0.0, 1.0)
# ax2.set_ylim(0.0, 100.0)
#
# # Legends (combined)
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# amos_legend(ax1, lines1 + lines2, labels1 + labels2, loc='upper left')
#
# # Title (toggleable)
# amosify_axes(ax1, title="Battery, Storage, Reward, Imaging and Action Events Over Time", show_title=SHOW_TITLES)
#
# plt.tight_layout()
# if use_shield:
#     save_plot_unique(fig3, f"seed{seed_number}_{policy_mode}_{policy_name}+SHIELD_battery_storage_reward_over_time")
# else:
#     save_plot_unique(fig3, f"seed{seed_number}_{policy_mode}_{policy_name}_battery_storage_reward_over_time")
# plt.show()
#
# minute = 1.0  # seconds per "minute" on x-axis in your current code
# imaging_times = np.array(SS1_actions_spec.imaging_times)
# azimuths      = np.array(SS1_actions_spec.chosen_target_azimuth)
# elevations    = np.array(SS1_actions_spec.chosen_target_elevation_angle)
#
# fig4, ax1 = plt.subplots(figsize=(11, 6))
#
# # Shading (convert s→min on x if you change minute)
# for (t0, t1) in umbra_spans:
#     ax1.fill_between([t0/minute, t1/minute], 0, 1, transform=ax1.get_xaxis_transform(),
#                      color='grey', alpha=0.35, zorder=0, label='Umbra (full eclipse)')
#     break  # label only once
# for (t0, t1) in [span for spans in (gs_spans_merged or {}).values() for span in spans]:
#     ax1.fill_between([t0/minute, t1/minute], 0, 1, transform=ax1.get_xaxis_transform(),
#                      color='green', alpha=0.12)
# # Azimuth
# ax1.plot(imaging_times/minute, azimuths, 'o-', color='tab:blue', linewidth=2.0, markersize=5, label='Azimuth')
# amosify_axes(ax1, xlabel='Time [sec]', ylabel='Azimuth [deg]')  # change label if you switch to minutes
#
# # Elevation on twin axis
# ax2 = ax1.twinx()
# ax2.plot(imaging_times/minute, elevations, 'x--', color='tab:green', linewidth=2.0, markersize=6, label='Elevation')
# ax2.set_ylabel('Elevation [deg]', color='tab:green', fontsize=AMOS_FONTS["label"])
# ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=AMOS_FONTS["tick"])
#
# # Legend (combined)
# l1, lab1 = ax1.get_legend_handles_labels()
# l2, lab2 = ax2.get_legend_handles_labels()
# amos_legend(ax1, l1 + l2, lab1 + lab2, loc='upper right')
#
# # Title (toggleable)
# amosify_axes(ax1, title='Pointing Directions During Episode', show_title=SHOW_TITLES)
#
# plt.tight_layout()
# if use_shield:
#     save_plot_unique(fig4, f"seed{seed_number}_{policy_mode}_{policy_name}+SHIELD_azimuth_and_elevation_pointing_over_time")
# else:
#     save_plot_unique(fig4, f"seed{seed_number}_{policy_mode}_{policy_name}_azimuth_and_elevation_pointing_over_time")
# plt.show()


SS1_actions_spec = env.unwrapped.satellites[0].action_builder.action_spec[0]
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
    # steps.csv
    df_steps = pd.DataFrame(step_log)
    steps_csv = os.path.join(run_dir, "steps.csv")
    df_steps.to_csv(steps_csv, index=False)
    print(f"Saved: {steps_csv}")

    # Your existing arrays (save into run_dir, not shared data/)
    save_npy(run_dir, "sim_times", sim_times)
    save_npy(run_dir, "battery_levels", battery_levels)
    save_npy(run_dir, "storage_levels", storage_levels)
    save_npy(run_dir, "eclipse_status", eclipse_status)
    save_npy(run_dir, "downlink_times", downlink_times)
    save_npy(run_dir, "charging_times", charging_times)
    save_npy(run_dir, "desat_times", desat_times)

    # Acquisition arrays you already created
    save_npy(run_dir, "image_command_times", cmd_times)
    save_npy(run_dir, "image_command_target_ids", cmd_target_ids)
    save_npy(run_dir, "image_acq_times", acq_times_for_cmd)
    save_npy(run_dir, "image_acq_dt", acq_dt_for_cmd)
    save_npy(run_dir, "image_acq_success", acq_success.astype(int))

    # save_npy(run_dir, "sun_azimuth", np.asarray(getattr(SS1_actions_spec, "sun_azimuth", []), dtype=float))
    # save_npy(run_dir, "sun_elevation_local", np.asarray(getattr(SS1_actions_spec, "sun_elevation_local", []), dtype=float))
    # save_npy(run_dir, "sun_target_sep_deg", np.asarray(getattr(SS1_actions_spec, "sun_target_sep_deg", []), dtype=float))
    # save_npy(run_dir, "sun_target_dot", np.asarray(getattr(SS1_actions_spec, "sun_target_dot", []), dtype=float))
    # save_npy(run_dir, "sun_target_daz_deg", np.asarray(getattr(SS1_actions_spec, "sun_target_daz_deg", []), dtype=float))
    # save_npy(run_dir, "scanner_shadowFactor", np.asarray(getattr(SS1_actions_spec, "scanner_shadowFactor", []), dtype=float))


    # If you already save eclipse windows:
    try:
        save_npy(run_dir, "eclipse_windows", eclipse_windows)
    except Exception:
        pass
else:
    print("Not saving data (save_data=False).")

# if save_data:
#     data_dir = "data"
#     os.makedirs(data_dir, exist_ok=True)
#
#     for key, value in data_dict.items():
#         if isinstance(value, dict):  # Save per-target data separately
#             for target_name, target_data in value.items():
#                 np.save(os.path.join(data_dir, f"{key}_{target_name}.npy"), np.array(target_data))
#         else:
#             np.save(os.path.join(data_dir, f"{key}.npy"), np.array(value))
#
#
#     # Also save high-level time series useful for analysis
#     np.save(os.path.join(data_dir, "sim_times.npy"), np.array(sim_times))
#     np.save(os.path.join(data_dir, "battery_levels.npy"), np.array(battery_levels))
#     np.save(os.path.join(data_dir, "storage_levels.npy"), np.array(storage_levels))
#     np.save(os.path.join(data_dir, "eclipse_status.npy"), np.array(eclipse_status))
#     np.save(os.path.join(data_dir, "charging_times.npy"), np.array(charging_times))
#     np.save(os.path.join(data_dir, "downlink_times.npy"), np.array(downlink_times))
#     np.save(os.path.join(data_dir, "desat_times.npy"), np.array(desat_times))
#     # Save extracted windows
#     np.save(os.path.join(data_dir, "eclipse_windows.npy"), np.array(eclipse_windows, dtype=float))
#
#     # # Ground-station windows: save each station separately
#     # for gs_name, windows in ground_station_windows.items():
#     #     np.save(os.path.join(data_dir, f"{gs_name}_windows.npy"), np.array(windows, dtype=float))
#     print("Data saved successfully in 'data/' folder.")
# else:
#     print("Not saving data")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time:.4f} seconds")

data = {}
data["cumulativeRewardSS1"] = round(env.unwrapped.rewarder.cum_reward['SS1'], 2)
data["illuminated_images"] = len(env.unwrapped.rewarder.imaged_illuminated)
# data["Total Images Downlinked"] = env.unwrapped.satellites[0].dynamics.total_downlinks
# data["Useful Images Downlinked"] = env.unwrapped.satellites[0].dynamics.useful_downlinks

SS1_actions_spec = env.unwrapped.satellites[0].action_builder.action_spec[0]
# -----------------------------
# Umbra "smart decision" metrics (if ImageRSO collected them)
# -----------------------------
if hasattr(SS1_actions_spec, "umbra_imaging_decisions"):
    umbra_total = int(getattr(SS1_actions_spec, "umbra_imaging_decisions", 0))
    umbra_smart = int(getattr(SS1_actions_spec, "umbra_smart_decisions", 0))
    umbra_regular = int(getattr(SS1_actions_spec, "umbra_regular_decisions", max(0, umbra_total - umbra_smart)))
    data["umbra_imaging_decisions"] = umbra_total
    data["umbra_smart_decisions"] = umbra_smart
    data["umbra_regular_decisions"] = umbra_regular
    data["umbra_smart_fraction"] = (umbra_smart / umbra_total) if umbra_total > 0 else None

    if hasattr(SS1_actions_spec, "umbra_smart_reason_counts"):
        data["umbra_smart_reason_counts"] = getattr(SS1_actions_spec, "umbra_smart_reason_counts")

    # Optional: mean sun alignment during umbra
    try:
        sc_sf = np.asarray(getattr(SS1_actions_spec, "scanner_shadowFactor", []), dtype=float)
        dots = np.asarray(getattr(SS1_actions_spec, "sun_target_dot", []), dtype=float)
        seps = np.asarray(getattr(SS1_actions_spec, "sun_target_sep_deg", []), dtype=float)
        m = np.isfinite(sc_sf) & (sc_sf < 0.5)
        if m.any():
            data["umbra_mean_sun_target_dot"] = float(np.nanmean(dots[m])) if len(dots) == len(sc_sf) else None
            data["umbra_mean_sun_target_sep_deg"] = float(np.nanmean(seps[m])) if len(seps) == len(sc_sf) else None
    except Exception:
        pass

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
    json_path = os.path.join(
        run_dir,
        f"metrics_{policy_tag}_random{act_random}_seed{seed_number}_{policy_mode}_{policy_name}.json"
    )
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
        "desat_action_count": 'desat_action_count' in locals() and desat_action_count or None,
        "target_imaging_pct": 'target_imaging_pct' in locals() and target_imaging_pct or None,
        "non_target_pct": 'non_target_pct' in locals() and non_target_pct or None,
        "imaging_success_percentage": 'env' in locals() and len(env.unwrapped.rewarder.imaged_illuminated)/target_imaging_count*100 if ('env' in locals() and target_imaging_count) else None
    }
    summary.update({
    "acq_success_rate": acq_success_rate,
    "avg_acquisition_time_sec": avg_acq_time_sec,
    "median_acquisition_time_sec": median_acq_time_sec,
    "pct_cmd_in_umbra": pct_cmd_in_umbra,
    "pct_acq_in_umbra": pct_acq_in_umbra,
    })
    summary.update({"look_metrics": look_metrics})
    summary.update({"regime_metrics": regime_metrics})


    run_meta = {
    "seed": seed_number,
    "policy_tag": policy_tag,
    "policy_mode": policy_mode,
    "policy_name": policy_name,
    "act_random": bool(act_random),
    "use_heuristic": bool(use_heuristic),
    "run_dir": run_dir,
    }


    payload = {"meta": run_meta, "data": data, "summary": summary}
    with open(json_path, "w") as jf:
        json.dump(payload, jf, indent=2, default=_convert)
    print(f"Saved metrics JSON to {json_path}")
except Exception as e:
    print("WARNING: Failed to save metrics JSON:", e)
print(f"good images #:{len(env.unwrapped.rewarder.imaged_illuminated)} out of {target_imaging_count}")
print(f"imaging success percentage {len(env.unwrapped.rewarder.imaged_illuminated)/target_imaging_count*100}%")
