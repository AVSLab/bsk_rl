#!/usr/bin/env python3
"""Standalone AMOS/JAS policy evaluator.

This is the public, single-entry evaluation workflow for the AMOS-2025/JAS
policy settings retained in this branch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from Basilisk.architecture import bskLogging
from Basilisk.utilities import macros, orbitalMotion
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw, world
from load_policy import load_policy

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


POLICY_CATALOG: Dict[str, str] = {
    "amos2025_seed184": str(
        THIS_DIR
        / "policies"
        / "amos2025"
        / "wGAE_balance0d100i_largepenalties_smallbatch_obs2"
    ),
}


R_E = 6371e3
D2R = macros.D2R
DEFAULT_ALT_BOUNDS = {
    "LEO": (400e3, 2000e3),
    "MEO": (2000e3, 35000e3),
    "GEO": (35786e3 - 300e3, 35786e3 + 300e3),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AMOS-era policies with JAS settings")
    p.add_argument("--mode", choices=["policy", "heuristic", "random"], default="policy")
    p.add_argument("--heuristic_mode", choices=["distance", "angle"], default="distance")
    p.add_argument("--heuristic_top_k", type=int, default=10)

    p.add_argument("--policy_key", default="amos2025_seed184", choices=sorted(POLICY_CATALOG.keys()))
    p.add_argument("--policy_path", default=None, help="Override policy path directly")
    p.add_argument("--policy_mode", choices=["latest", "best", "smallest"], default="latest")

    p.add_argument("--seed", type=int, default=184)
    p.add_argument("--obs_v", type=float, default=2.0)
    p.add_argument("--n_targets", type=int, default=100)
    p.add_argument("--n_targets_ahead", type=int, default=10)
    p.add_argument("--imaging_duration", type=float, default=300.0)
    p.add_argument("--extra_time_factor", type=float, default=1.5)

    p.add_argument("--target_env", choices=["leo", "mixed"], default="mixed")
    p.add_argument(
        "--mix_weights",
        type=str,
        default='{"LEO": 1, "MEO": 1, "GEO": 1}',
        help="JSON dict for mixed regime weights",
    )

    p.add_argument("--failure_penalty", type=float, default=-100.0)
    p.add_argument("--full_storage_penalty", type=float, default=0.0)
    p.add_argument("--low_battery_penalty", type=float, default=0.0)

    p.add_argument(
        "--generate_obs_retasking_only",
        action="store_true",
        help="Use retasking-only observations (default: False)",
    )
    p.add_argument(
        "--disable_fast_retasking",
        dest="disable_fast_retasking",
        action="store_true",
        help="Disable early image-success termination and force full image action duration",
    )
    p.add_argument(
        "--enable_fast_retasking",
        dest="disable_fast_retasking",
        action="store_false",
        help="Allow early image-success termination",
    )
    p.set_defaults(disable_fast_retasking=True)

    p.add_argument("--save_root", default=str(THIS_DIR / "outputs"))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def parse_mix_weights(raw: str) -> dict[str, float]:
    weights = json.loads(raw)
    required = ["LEO", "MEO", "GEO"]
    for key in required:
        if key not in weights:
            raise ValueError(f"mix_weights missing key: {key}")
    vals = np.array([float(weights[k]) for k in required], dtype=float)
    if np.any(vals < 0) or vals.sum() <= 0:
        raise ValueError("mix_weights must be nonnegative and not all zero")
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(required, vals)}


def extract_reward_balance(policy_id: str) -> tuple[float, float]:
    m = re.search(r"(\d+)d(\d+)i", policy_id)
    if not m:
        return 0.0, 1.0
    d = float(m.group(1))
    i = float(m.group(2))
    total = d + i
    if total <= 0:
        return 0.0, 1.0
    return d / total, i / total


def _sample_for_regime(
    regime: str,
    altitude_bounds: dict[str, tuple[float, float]],
    min_perigee_alt: float,
) -> orbitalMotion.ClassicElements:
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


def custom_oe_randomizer(
    regime: str = "LEO",
    mix_weights: dict[str, float] | None = None,
    altitude_bounds: dict[str, tuple[float, float]] | None = None,
    min_perigee_alt: float = 400e3,
) -> orbitalMotion.ClassicElements:
    if altitude_bounds is None:
        altitude_bounds = DEFAULT_ALT_BOUNDS

    if regime.lower() == "mixed":
        regimes = ["LEO", "MEO", "GEO"]
        if mix_weights is None:
            probs = np.array([1.0, 1.0, 1.0], dtype=float)
        else:
            probs = np.array([mix_weights.get(r, 0.0) for r in regimes], dtype=float)
        if probs.sum() <= 0:
            raise ValueError("mix_weights must include positive mass")
        probs /= probs.sum()
        regime = str(np.random.choice(regimes, p=probs))

    return _sample_for_regime(regime.upper(), altitude_bounds, min_perigee_alt)


class NoFastRetaskImageRSO(act.ImageRSO):
    """Image action variant that disables early-success terminal events."""

    def _enable_image_success_event(self, target) -> None:  # noqa: D401
        # Intentionally disabled: keep action active until duration/window timeout.
        return


def build_scanning_satellite_class(
    obs_v: float,
    n_targets_ahead: int,
    imaging_duration: float,
    disable_fast_retasking: bool,
):
    if abs(obs_v - 2.0) > 1e-9:
        raise ValueError(
            "This evaluator currently supports obs_v=2 only (AMOS setting with GS lookahead=5)."
        )

    image_cls = NoFastRetaskImageRSO if disable_fast_retasking else act.ImageRSO

    class ScanningSatellite(sats.AccessSatellite):
        observation_spec = [
            obs.SatProperties(
                dict(prop="storage_level_fraction"),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
            ),
            obs.PolarisScTargetProperties(
                dict(prop="target_elevation_angle", norm=1.0),
                dict(prop="rel_pos_vector_r_BR_H", norm=1596 * 1000),
                dict(prop="angle_to_target", norm=1.0),
                dict(prop="target_distance", norm=1596 * 1000),
                dict(prop="target_shadowFactor", norm=1.0),
                n_ahead_observe=n_targets_ahead,
            ),
            obs.Eclipse(norm=1.0),
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=5700.0),
                dict(prop="opportunity_close", norm=5700.0),
                type="ground_station",
                n_ahead_observe=5,
            ),
        ]
        action_spec = [
            image_cls(n_ahead_image=n_targets_ahead, duration=imaging_duration),
            act.Charge(duration=300.0),
            act.Downlink(duration=180.0),
            act.Desat(duration=150.0),
        ]
        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    return ScanningSatellite


class TargetSatellite(sats.Satellite):
    observation_spec = [obs.Time()]
    action_spec = [act.Drift(duration=1e9)]
    dyn_type = dyn.BasicTargetDynamicsModel
    fsw_type = fsw.BasicTargetFSWModel


def select_policy_path(args: argparse.Namespace) -> str:
    if args.policy_path:
        return args.policy_path
    if args.policy_key not in POLICY_CATALOG:
        raise ValueError(f"Unknown policy key: {args.policy_key}")
    return POLICY_CATALOG[args.policy_key]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    policy_path = select_policy_path(args)
    policy_label = args.policy_key if args.policy_path is None else Path(policy_path).name
    downlink_bonus, imaging_bonus = extract_reward_balance(policy_path)
    mix_weights = parse_mix_weights(args.mix_weights)

    total_time = args.n_targets * args.imaging_duration * args.extra_time_factor
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.mode}_seed{args.seed}_{policy_label}_{run_stamp}"
    out_dir = Path(args.save_root) / run_name
    ensure_dir(out_dir)

    ScanningSatellite = build_scanning_satellite_class(
        obs_v=args.obs_v,
        n_targets_ahead=args.n_targets_ahead,
        imaging_duration=args.imaging_duration,
        disable_fast_retasking=args.disable_fast_retasking,
    )

    sat_args: Dict[str, Any] = {}
    sat_args["imageAttErrorRequirement"] = 0.0025
    sat_args["imageRateErrorRequirement"] = 0.01
    sat_args["dataStorageCapacity"] = 50 * 8e6 / 2
    sat_args["storageInit"] = lambda: 0.0
    sat_args["instrumentBaudRate"] = 0.5 * 8e6
    sat_args["transmitterBaudRate"] = -0.5 * 8e6
    sat_args["batteryStorageCapacity"] = 500 * 3600
    sat_args["storedCharge_Init"] = lambda: 0.3 * 500 * 3600
    sat_args["basePowerDraw"] = -10.0
    sat_args["instrumentPowerDraw"] = -30.0
    sat_args["transmitterPowerDraw"] = -25.0
    sat_args["thrusterPowerDraw"] = -80.0
    sat_args["panelArea"] = 1
    sat_args["disturbance_vector"] = lambda: np.random.normal(scale=0.0, size=3)
    sat_args["maxWheelSpeed"] = 6000.0
    sat_args["wheelSpeeds"] = lambda: np.random.uniform(-500, 500, 3)
    sat_args["desatAttitude"] = "sun"
    sat_args["downlink_bonus"] = downlink_bonus
    sat_args["imaging_bonus"] = imaging_bonus
    sat_args["full_storage_penalty"] = args.full_storage_penalty
    sat_args["low_battery_penalty"] = args.low_battery_penalty
    sat_args["eclipse_threshold_for_imaging"] = 0.5
    sat_args["eclipse_threshold_for_reward"] = 0.5

    scanner = ScanningSatellite(name="SS1", sat_args=sat_args)

    if args.target_env == "mixed":
        oe_func = partial(custom_oe_randomizer, regime="mixed", mix_weights=mix_weights)
    else:
        oe_func = partial(custom_oe_randomizer, regime="LEO")

    target_args = dict(
        oe=oe_func,
        batteryStorageCapacity=1,
        storedCharge_Init=0.0,
        basePowerDraw=-10000.0,
    )
    targets = [TargetSatellite(name=f"target_{i}", sat_args=target_args) for i in range(args.n_targets)]

    all_satellites = [scanner] + targets

    env = gym.make(
        "ConstellationTasking-v1",
        satellites=all_satellites,
        scenario=scene.RandomSatellites("SS1", n_targets=args.n_targets),
        rewarder=data.RSOTargetImageReward(),
        world_type=world.GroundStationWorldModel,
        time_limit=total_time,
        failure_penalty=args.failure_penalty,
        generate_obs_retasking_only=args.generate_obs_retasking_only,
        log_level="WARNING",
        disable_env_checker=True,
    )

    policy_fn = None
    if args.mode == "policy":
        policy_fn = load_policy(Path(policy_path), policy_mode=args.policy_mode)

    observation, info = env.reset(seed=args.seed)

    main_sat = env.unwrapped.satellites[0]
    main_sat.dynamics.use_heuristic = args.mode == "heuristic"
    main_sat.dynamics.heuristic_mode = args.heuristic_mode
    main_sat.dynamics.heuristic_top_k = int(args.heuristic_top_k)

    n_actions = int(getattr(main_sat.action_space, "n", args.n_targets_ahead + 3))

    step_rows: list[dict[str, Any]] = []
    t0 = datetime.now()
    done = False
    step_idx = 0

    log(
        f"Run: {run_name}\n"
        f"mode={args.mode}, policy_mode={args.policy_mode}, obs_v={args.obs_v}, "
        f"disable_fast_retasking={args.disable_fast_retasking}, target_env={args.target_env}, "
        f"mix_weights={mix_weights}, failure_penalty={args.failure_penalty}",
        args.quiet,
    )

    while not done:
        sat_obs = observation[main_sat.name]

        if args.mode == "policy":
            flat_obs = flatten_to_single_ndarray(sat_obs)
            action_id = int(policy_fn(flat_obs))
        elif args.mode == "heuristic":
            action_id = 0
        else:
            action_id = int(np.random.randint(0, n_actions))

        action_dict = {main_sat.name: action_id}
        action_dict.update({t.name: 0 for t in targets})

        observation, reward, terminated, truncated, info = env.step(action=action_dict)

        imaged_count = len(main_sat.data_store.data.imaged)
        downlinked_count = len(getattr(main_sat.data_store.data, "downlinked", []))
        sim_time = float(main_sat.simulator.sim_time)

        step_rows.append(
            {
                "step": step_idx,
                "sim_time_sec": sim_time,
                "reward": float(reward.get(main_sat.name, 0.0)),
                "num_unique_targets_imaged": int(imaged_count),
                "num_unique_targets_downlinked": int(downlinked_count),
                "action_id": int(action_id),
            }
        )

        if all(truncated.values()) or all(terminated.values()) or imaged_count >= args.n_targets:
            done = True

        step_idx += 1

    t1 = datetime.now()
    elapsed_wallclock_sec = (t1 - t0).total_seconds()

    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "mode": args.mode,
        "heuristic_mode": args.heuristic_mode,
        "policy_key": args.policy_key,
        "policy_path": policy_path,
        "policy_mode": args.policy_mode,
        "obs_v": args.obs_v,
        "n_targets": args.n_targets,
        "n_targets_ahead": args.n_targets_ahead,
        "imaging_duration_sec": args.imaging_duration,
        "extra_time_factor": args.extra_time_factor,
        "total_time_limit_sec": total_time,
        "disable_fast_retasking": args.disable_fast_retasking,
        "generate_obs_retasking_only": args.generate_obs_retasking_only,
        "target_env": args.target_env,
        "mix_weights": mix_weights,
        "failure_penalty": args.failure_penalty,
        "full_storage_penalty": args.full_storage_penalty,
        "low_battery_penalty": args.low_battery_penalty,
        "downlink_bonus": downlink_bonus,
        "imaging_bonus": imaging_bonus,
        "steps": step_idx,
        "final_sim_time_sec": float(main_sat.simulator.sim_time),
        "final_num_unique_targets_imaged": int(len(main_sat.data_store.data.imaged)),
        "final_num_unique_targets_downlinked": int(len(getattr(main_sat.data_store.data, "downlinked", []))),
        "elapsed_wallclock_sec": elapsed_wallclock_sec,
    }

    df = pd.DataFrame(step_rows)
    df.to_csv(out_dir / "step_metrics.csv", index=False)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    log(f"Saved run outputs to: {out_dir}", args.quiet)
    log(
        f"Final targets imaged: {summary['final_num_unique_targets_imaged']} | "
        f"sim time: {summary['final_sim_time_sec']:.1f} s | "
        f"wall-clock: {elapsed_wallclock_sec/3600.0:.2f} hr",
        args.quiet,
    )


if __name__ == "__main__":
    main()
