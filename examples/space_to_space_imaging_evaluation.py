#!/usr/bin/env python3
"""Public evaluation entry point for the late-summer 2025 space-to-space imaging setup.

This script is intended as the public-facing evaluator for the imaging-only
snapshot derived from the ``IA_Polaris_imaging_june10`` line of development.
It keeps the historically relevant environment choices:

- space-to-space imaging
- imaging reward only (no downlink reward bonus)
- no fast target switching additions
- optional angle-based heuristic baseline

Examples
--------
Policy evaluation:

    python examples/space_to_space_imaging_evaluation.py \
        --mode policy \
        --policy-path /path/to/policy_dir \
        --obs-version 7

Angle heuristic baseline:

    python examples/space_to_space_imaging_evaluation.py \
        --mode heuristic \
        --heuristic-mode angle
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import bskLogging
from Basilisk.utilities import macros, orbitalMotion
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw, world
from examples.load_policy import load_policy

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# Keep SPICE kernels loaded for the process lifetime. This avoids teardown hangs
# seen in some Basilisk evaluation runs.
try:
    from Basilisk.utilities import simIncludeGravBody as _sim_include_grav_body

    def _no_unload(self) -> None:
        return

    _sim_include_grav_body.gravBodyFactory.unloadSpiceKernels = _no_unload
except Exception:
    pass

ACTION_LABELS = {
    10: "charge",
    11: "downlink",
    12: "desat",
}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def default_output_dir(args: argparse.Namespace) -> Path:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_tag = "policy"
    if args.policy_path:
        policy_tag = slugify(Path(args.policy_path).name)
    return Path("examples") / "public_outputs" / f"{args.mode}_{policy_tag}_{tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a late-summer 2025 space-to-space imaging policy or heuristic baseline.",
    )
    parser.add_argument(
        "--mode",
        choices=["policy", "heuristic", "random"],
        default="policy",
        help="Evaluation mode. Policy mode loads an RLlib checkpoint; heuristic uses the built-in angle/distance selector.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Directory containing the RLlib policy checkpoints. Required in policy mode.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["latest", "smallest", "best"],
        default="best",
        help="Which checkpoint to load from the policy directory.",
    )
    parser.add_argument(
        "--obs-version",
        type=float,
        default=7,
        choices=[1, 1.1, 2, 5, 6, 7],
        help="Observation version used by the policy or baseline.",
    )
    parser.add_argument("--seed", type=int, default=184, help="Environment reset seed.")
    parser.add_argument(
        "--n-targets",
        type=int,
        default=100,
        help="Number of target satellites in the scenario.",
    )
    parser.add_argument(
        "--n-targets-ahead",
        type=int,
        default=10,
        help="Number of imaging opportunities exposed in the observation/action builder.",
    )
    parser.add_argument(
        "--imaging-duration",
        type=float,
        default=300.0,
        help="Duration, in seconds, of each imaging action.",
    )
    parser.add_argument(
        "--extra-time-factor",
        type=float,
        default=1.5,
        help="Episode time limit multiplier relative to n_targets * imaging_duration.",
    )
    parser.add_argument(
        "--heuristic-mode",
        choices=["angle", "distance"],
        default="angle",
        help="Built-in heuristic mode. The historical late-summer 2025 default is angle.",
    )
    parser.add_argument(
        "--heuristic-top-k",
        type=int,
        default=10,
        help="Distance heuristic top-k visibility filter. Kept for compatibility with the action logic.",
    )
    parser.add_argument(
        "--use-shield",
        dest="use_shield",
        action="store_true",
        default=True,
        help="Apply the simple battery/storage shield used in the original evaluation scripts.",
    )
    parser.add_argument(
        "--no-shield",
        dest="use_shield",
        action="store_false",
        help="Disable the simple battery/storage shield.",
    )
    parser.add_argument(
        "--critical-storage-level",
        type=float,
        default=0.99,
        help="Storage fraction threshold above which the shield forces downlink.",
    )
    parser.add_argument(
        "--critical-battery-level",
        type=float,
        default=0.20,
        help="Battery fraction threshold below which the shield forces charge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/CSV/plot outputs. Defaults to examples/public_outputs/<timestamp>.",
    )
    args = parser.parse_args()
    if args.mode == "policy" and args.policy_path is None:
        parser.error("--policy-path is required when --mode policy is selected.")
    if args.output_dir is None:
        args.output_dir = default_output_dir(args)
    return args


def _get_rewarder(env: gym.Env) -> Any:
    return getattr(env, "rewarder", getattr(env.env, "rewarder", None))


def _get_imaged_all(env: gym.Env) -> list[Any]:
    try:
        return env.env.satellites[0].data_store.data.imaged
    except AttributeError:
        try:
            return env.satellites[0].data_store.data.imaged
        except Exception:
            return []


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        if isinstance(value, (int, float)):
            return int(value)
        return 0


def custom_oe_randomizer() -> orbitalMotion.ClassicElements:
    r_leo = 6871.0 * 1000
    r_upper_leo = 8371.0 * 1000

    oe = orbitalMotion.ClassicElements()
    oe.a = np.random.uniform(1.00 * r_leo, r_upper_leo)
    if oe.a < 2 * r_leo:
        oe.e = np.random.uniform(0.0, 0.02)
        while oe.a * (1 - oe.e) < 6771.0 * 1000:
            oe.e = np.random.uniform(0.0, 0.02)
    else:
        oe.e = np.random.uniform(0.0, 0.2)
    oe.i = np.random.uniform(0, 180) * macros.D2R
    oe.Omega = np.random.uniform(0, 360) * macros.D2R
    oe.omega = np.random.uniform(0, 360) * macros.D2R
    oe.f = np.random.uniform(0, 360) * macros.D2R
    return oe


def make_env(
    obs_v: float,
    total_time: float,
    n_targets: int,
    n_targets_ahead: int,
    imaging_duration: float,
    use_heuristic: bool,
    heuristic_mode: str,
    heuristic_top_k: int,
) -> gym.Env:
    class MyScanningSatellite(sats.AccessSatellite):
        if obs_v == 1:
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=1.0),
                    dict(prop="rel_pos_vector_r_BR_N", norm=1596 * 1000),
                    dict(prop="angle_to_target", norm=1.0),
                    dict(prop="target_distance", norm=1596 * 1000),
                    dict(prop="target_id_info", norm=1.0),
                    dict(prop="target_imaged", norm=1.0),
                    n_ahead_observe=n_targets_ahead,
                ),
                obs.Eclipse(),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=5,
                ),
            ]
        elif obs_v == 1.1:
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=1.0),
                    dict(prop="rel_pos_vector_r_BR_N", norm=1596 * 1000),
                    dict(prop="angle_to_target", norm=1.0),
                    dict(prop="target_distance", norm=1596 * 1000),
                    dict(prop="target_id_info", norm=1.0),
                    dict(prop="target_imaged", norm=1.0),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=n_targets_ahead,
                ),
                obs.Eclipse(),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=5,
                ),
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
                    dict(prop="rel_pos_vector_r_BR_H", norm=1596 * 1000),
                    dict(prop="angle_to_target", norm=1.0),
                    dict(prop="target_distance", norm=1596 * 1000),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=n_targets_ahead,
                ),
                obs.Eclipse(),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=5,
                ),
            ]
        else:
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),
                ),
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=90.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                    dict(prop="angle_to_target", norm=90.0),
                    dict(prop="target_distance", norm=15960 * 1000),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=n_targets_ahead,
                ),
                obs.Eclipse(norm=5700),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700.0),
                    dict(prop="opportunity_close", norm=5700.0),
                    type="ground_station",
                    n_ahead_observe=1 if obs_v == 6 else 2,
                ),
            ]

        action_spec = [
            act.ImageRSO(n_ahead_image=n_targets_ahead, duration=imaging_duration),
            act.Charge(duration=300.0),
            act.Downlink(duration=180.0),
            act.Desat(duration=150.0),
        ]

        if obs_v == 5:
            observation_spec = [
                obs.PolarisScTargetProperties(
                    dict(prop="target_elevation_angle", norm=90.0),
                    dict(prop="rel_pos_vector_r_BR_H", norm=15960 * 1000),
                    dict(prop="angle_to_target", norm=90.0),
                    dict(prop="target_distance", norm=15960 * 1000),
                    dict(prop="target_shadowFactor", norm=1.0),
                    n_ahead_observe=n_targets_ahead,
                ),
                obs.Eclipse(norm=5700),
            ]
            action_spec = [
                act.ImageRSO(
                    n_ahead_image=n_targets_ahead,
                    duration=imaging_duration,
                ),
            ]
        elif obs_v == 6:
            action_spec = [
                act.ImageRSO(
                    n_ahead_image=n_targets_ahead,
                    duration=imaging_duration,
                ),
                act.Charge(duration=300.0),
                act.Downlink(duration=300.0),
                act.Desat(duration=150.0),
            ]

        dyn_type = dyn.ImagingSCDynModel
        fsw_type = fsw.ImagingSCFSWModel

    class MyTargetSatellite(sats.Satellite):
        observation_spec = [obs.Time()]
        action_spec = [act.Drift(duration=total_time)]
        dyn_type = dyn.BasicTargetDynamicsModel
        fsw_type = fsw.BasicTargetFSWModel

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
    sat_args["imaging_bonus"] = 1.0
    sat_args["full_storage_penalty"] = 0
    sat_args["low_battery_penalty"] = 0
    sat_args["eclipse_threshold_for_imaging"] = 0.5
    sat_args["eclipse_threshold_for_reward"] = sat_args["eclipse_threshold_for_imaging"]
    sat_args["use_heuristic"] = use_heuristic
    sat_args["heuristic_mode"] = heuristic_mode
    sat_args["heuristic_top_k"] = heuristic_top_k
    sat_args["print_info"] = False

    target_args = dict(
        oe=custom_oe_randomizer,
        batteryStorageCapacity=1,
        storedCharge_Init=0.0,
        basePowerDraw=-10000.0,
    )

    sat = MyScanningSatellite(name="SS1", sat_args=sat_args)
    targets = [
        MyTargetSatellite(name=f"target_{i}", sat_args=target_args)
        for i in range(n_targets)
    ]
    env = gym.make(
        "ConstellationTasking-v1",
        satellites=[sat] + targets,
        scenario=scene.RandomSatellites("SS1", n_targets=n_targets),
        rewarder=data.RSOTargetImageReward(),
        world_type=world.GroundStationWorldModel,
        time_limit=total_time,
        log_level="ERROR",
        disable_env_checker=True,
    )
    return env


def apply_simple_shield(
    env: gym.Env,
    requested_action: int,
    critical_storage_level: float,
    critical_battery_level: float,
) -> tuple[int, str | None]:
    try:
        storage_fraction = env.satellites[0].dynamics.storage_level_fraction
        battery_fraction = env.satellites[0].dynamics.battery_charge_fraction
    except Exception:
        return requested_action, None

    forced_action = None
    reason_parts = []
    if battery_fraction <= critical_battery_level:
        forced_action = 10
        reason_parts.append(
            f"battery<=thr ({battery_fraction:.3f}<={critical_battery_level:.3f})"
        )
    if storage_fraction >= critical_storage_level and forced_action is None:
        forced_action = 11
        reason_parts.append(
            f"storage>=thr ({storage_fraction:.3f}>={critical_storage_level:.3f})"
        )
    if forced_action is None:
        return requested_action, None
    return forced_action, " & ".join(reason_parts)


def action_label(action_id: int) -> str:
    if 0 <= action_id <= 9:
        return f"image_{action_id}"
    return ACTION_LABELS.get(action_id, f"action_{action_id}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_inference_timings_csv(path: Path, timings_ms: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["call_index", "inference_time_ms"])
        for index, value in enumerate(timings_ms, start=1):
            writer.writerow([index, f"{value:.9f}"])


def save_histogram(path: Path, timings_ms: list[float]) -> None:
    if not timings_ms:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(timings_ms, bins=min(20, max(5, len(timings_ms) // 3)), edgecolor="black")
    ax.set_title("Policy inference time distribution")
    ax.set_xlabel("Inference time [ms]")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_episode(args: argparse.Namespace) -> tuple[dict, dict | None, dict | None, list[float]]:
    total_time = args.n_targets * args.imaging_duration * args.extra_time_factor
    use_heuristic = args.mode == "heuristic"
    env = make_env(
        obs_v=args.obs_version,
        total_time=total_time,
        n_targets=args.n_targets,
        n_targets_ahead=args.n_targets_ahead,
        imaging_duration=args.imaging_duration,
        use_heuristic=use_heuristic,
        heuristic_mode=args.heuristic_mode,
        heuristic_top_k=args.heuristic_top_k,
    )

    policy = None
    rng = np.random.default_rng(args.seed)
    if args.mode == "policy":
        policy = load_policy(args.policy_path.expanduser(), policy_mode=args.policy_mode)
        if hasattr(policy, "reset_timing"):
            policy.reset_timing()

    observation, info = env.reset(seed=args.seed)
    rewarder = _get_rewarder(env)
    requested_action_counts: dict[str, int] = {}
    executed_action_counts: dict[str, int] = {}
    shield_interventions = []
    telemetry = {
        "sim_time": [],
        "battery_fraction": [],
        "storage_fraction": [],
        "wheel_speed_fraction": [],
        "cumulative_reward": [],
        "num_imaged_raw": [],
        "num_imaged_illuminated": [],
        "num_downlinked_total": [],
        "num_downlinked_useful": [],
        "requested_action": [],
        "executed_action": [],
    }
    cumulative_reward = 0.0
    max_steps = max(1, int(total_time * 10))

    for step_index in range(max_steps):
        sim_time = env.simulator.sim_time
        if args.mode == "policy":
            requested_action = int(policy(observation["SS1"]))
        elif args.mode == "heuristic":
            requested_action = 0
        else:
            requested_action = int(rng.integers(0, 13))

        executed_action = requested_action
        shield_reason = None
        if args.use_shield:
            executed_action, shield_reason = apply_simple_shield(
                env,
                requested_action=requested_action,
                critical_storage_level=args.critical_storage_level,
                critical_battery_level=args.critical_battery_level,
            )
            if shield_reason is not None:
                shield_interventions.append(
                    {
                        "step": step_index,
                        "sim_time": float(sim_time),
                        "requested_action": int(requested_action),
                        "executed_action": int(executed_action),
                        "reason": shield_reason,
                    }
                )

        requested_action_counts[action_label(requested_action)] = (
            requested_action_counts.get(action_label(requested_action), 0) + 1
        )
        executed_action_counts[action_label(executed_action)] = (
            executed_action_counts.get(action_label(executed_action), 0) + 1
        )

        action_dict = {"SS1": int(executed_action)}
        action_dict.update({f"target_{j}": 0 for j in range(args.n_targets)})
        observation, reward, terminated, truncated, info = env.step(action=action_dict)

        cumulative_reward += float(reward.get("SS1", 0.0))
        imaged_raw = _safe_len(_get_imaged_all(env))
        imaged_illuminated = _safe_len(getattr(rewarder, "imaged_illuminated", []))
        total_downlinks = int(getattr(rewarder, "total_downlinks", 0))
        useful_downlinks = int(getattr(rewarder, "useful_downlinks", 0))

        telemetry["sim_time"].append(float(sim_time))
        telemetry["battery_fraction"].append(
            float(env.satellites[0].dynamics.battery_charge_fraction)
        )
        telemetry["storage_fraction"].append(
            float(env.satellites[0].dynamics.storage_level_fraction)
        )
        telemetry["wheel_speed_fraction"].append(
            float(env.satellites[0].dynamics.wheel_speeds_fraction)
        )
        telemetry["cumulative_reward"].append(cumulative_reward)
        telemetry["num_imaged_raw"].append(imaged_raw)
        telemetry["num_imaged_illuminated"].append(imaged_illuminated)
        telemetry["num_downlinked_total"].append(total_downlinks)
        telemetry["num_downlinked_useful"].append(useful_downlinks)
        telemetry["requested_action"].append(int(requested_action))
        telemetry["executed_action"].append(int(executed_action))

        terminated_all = all(terminated.values()) if isinstance(terminated, dict) else bool(terminated)
        truncated_all = all(truncated.values()) if isinstance(truncated, dict) else bool(truncated)
        if terminated_all or truncated_all or imaged_raw >= args.n_targets:
            break

    action_spec = env.satellites[0].action_builder.action_spec[0]
    summary = {
        "config": {
            "mode": args.mode,
            "policy_path": str(args.policy_path) if args.policy_path else None,
            "policy_mode": args.policy_mode,
            "obs_version": args.obs_version,
            "seed": args.seed,
            "n_targets": args.n_targets,
            "n_targets_ahead": args.n_targets_ahead,
            "imaging_duration": args.imaging_duration,
            "extra_time_factor": args.extra_time_factor,
            "time_limit_seconds": total_time,
            "use_shield": args.use_shield,
            "heuristic_mode": args.heuristic_mode,
            "heuristic_top_k": args.heuristic_top_k,
        },
        "results": {
            "final_cumulative_reward": cumulative_reward,
            "final_num_imaged_raw": telemetry["num_imaged_raw"][-1] if telemetry["num_imaged_raw"] else 0,
            "final_num_imaged_illuminated": telemetry["num_imaged_illuminated"][-1] if telemetry["num_imaged_illuminated"] else 0,
            "final_num_downlinked_total": telemetry["num_downlinked_total"][-1] if telemetry["num_downlinked_total"] else 0,
            "final_num_downlinked_useful": telemetry["num_downlinked_useful"][-1] if telemetry["num_downlinked_useful"] else 0,
            "policy_calls": len(telemetry["requested_action"]) if args.mode == "policy" else 0,
            "requested_action_counts": requested_action_counts,
            "executed_action_counts": executed_action_counts,
            "shield_interventions": len(shield_interventions),
            "shield_intervention_log": shield_interventions,
        },
        "target_selection_metrics": {
            "mean_initial_ang_error": (
                float(np.mean(action_spec.initial_angular_error))
                if getattr(action_spec, "initial_angular_error", [])
                else None
            ),
            "std_initial_ang_error": (
                float(np.std(action_spec.initial_angular_error))
                if getattr(action_spec, "initial_angular_error", [])
                else None
            ),
            "mean_target_distance": (
                float(np.mean(action_spec.chosen_target_distance))
                if getattr(action_spec, "chosen_target_distance", [])
                else None
            ),
            "std_target_distance": (
                float(np.std(action_spec.chosen_target_distance))
                if getattr(action_spec, "chosen_target_distance", [])
                else None
            ),
            "mean_target_elevation": (
                float(np.mean(action_spec.chosen_target_elevation_angle))
                if getattr(action_spec, "chosen_target_elevation_angle", [])
                else None
            ),
            "std_target_elevation": (
                float(np.std(action_spec.chosen_target_elevation_angle))
                if getattr(action_spec, "chosen_target_elevation_angle", [])
                else None
            ),
        },
        "telemetry": telemetry,
    }

    inference_summary = None
    model_summary = None
    inference_timings_ms: list[float] = []
    if policy is not None and hasattr(policy, "timing_summary"):
        inference_summary = policy.timing_summary()
        model_summary = policy.model_summary()
        inference_timings_ms = list(getattr(policy, "inference_times_ms", []))

    env.close()
    return summary, inference_summary, model_summary, inference_timings_ms


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, inference_summary, model_summary, inference_timings_ms = run_episode(args)
    summary_path = args.output_dir / "summary.json"
    save_json(summary_path, summary)

    if inference_summary is not None and model_summary is not None:
        save_json(args.output_dir / "inference_summary.json", inference_summary)
        save_json(args.output_dir / "model_summary.json", model_summary)
        if inference_timings_ms:
            save_inference_timings_csv(
                args.output_dir / "inference_timings.csv",
                inference_timings_ms,
            )
            save_histogram(
                args.output_dir / "inference_time_histogram.png",
                inference_timings_ms,
            )

    print(f"Saved evaluation outputs to {args.output_dir}")
    print(json.dumps(summary["results"], indent=2))
    if inference_summary is not None:
        print(json.dumps(inference_summary, indent=2))


if __name__ == "__main__":
    main()
