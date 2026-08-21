#!/usr/bin/env python3
"""Detailed aggregate analysis for AMOS 2026 GAT reward-sweep MC outputs.

This script is intentionally post-processing only: it does not run Basilisk or
reload policies. It scans the saved Monte Carlo output tree, extracts the rich
per-episode metrics JSON plus steps/images CSVs, and writes policy-level tables
and plots.

Some downlink timing metrics are proxies because current evaluation outputs do
not store per-packet delivery IDs. See metric_definitions.json for details.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_TAGS = (
    "00d100i",
    "10d90i",
    "20d80i",
    "30d70i",
    "40d60i",
    "50d50i",
    "60d40i",
    "70d30i",
    "75d25i",
    "80d20i",
    "90d10i",
    "100d00i",
)
ACTION_ID_TO_NAME = {
    0: "charge",
    1: "downlink",
    2: "desat",
}
IMAGE_ACTION_MIN = 3
IMAGE_ACTION_MAX = 12

# High-value fields that are present in metrics_*.json payload["data"].
DATA_FIELDS = [
    "cumulativeRewardSS1",
    "illuminated_images",
    "confirmed_illuminated_images",
    "pending_illuminated_images_onboard",
    "pending_images_onboard",
    "total_illuminated_images",
    "umbra_imaging_decisions",
    "umbra_smart_decisions",
    "umbra_regular_decisions",
    "umbra_smart_fraction",
    "umbra_mean_sun_target_dot",
    "umbra_mean_sun_target_sep_deg",
    "mean_target_azimuth",
    "std_target_azimuth",
    "mean_target_elevation",
    "std_target_elevation",
    "mean_target_elevation_local",
    "std_target_elevation_local",
    "mean_initial_ang_error",
    "std_initial_ang_error",
    "mean_target_distance",
    "std_target_distance",
    "mean_target_illumination_status",
    "num_target_above_illumination_threshold",
    "num_target_below_illumination_threshold",
    "mean_target_priority",
    "std_target_priority",
    "target_ever_visible_fraction",
    "mean_rel_pos_H_x",
    "mean_rel_pos_H_y",
    "mean_rel_pos_H_z",
    "std_rel_pos_H_x",
    "std_rel_pos_H_y",
    "std_rel_pos_H_z",
    # Available in newly instrumented evaluations. Older MC outputs retain NaN.
    "dynamic_priority_event_applied",
    "dynamic_priority_event_time_sec",
    "dynamic_priority_event_applied_time_sec",
    "hio_target_count",
    "shio_target_count",
    "hio_candidate_access_count",
    "shio_candidate_access_count",
    "time_to_first_hio_candidate_access_sec",
    "time_to_first_shio_candidate_access_sec",
    "hio_command_count_after_event",
    "shio_command_count_after_event",
    "hio_successful_capture_count_after_event",
    "shio_successful_capture_count_after_event",
    "time_to_first_hio_command_sec",
    "time_to_first_shio_command_sec",
    "time_to_first_hio_success_sec",
    "time_to_first_shio_success_sec",
]
SUMMARY_FIELDS = [
    "target_imaging_count",
    "non_target_count",
    "charge_action_count",
    "downlink_action_count",
    "desat_action_count",
    "target_imaging_pct",
    "non_target_pct",
    "imaging_success_percentage",
    "acq_success_rate",
    "avg_acquisition_time_sec",
    "median_acquisition_time_sec",
    "raw_device_cmd_acq_success_rate",
    "raw_device_cmd_avg_acquisition_time_sec",
    "raw_device_cmd_median_acquisition_time_sec",
    "max_abs_accepted_acq_action_duration_delta_sec",
    "pct_cmd_in_umbra",
    "pct_acq_in_umbra",
    "num_imaging_attempts",
    "imaging_attempt_success_rate",
    "total_imaging_action_time_sec",
    "mean_imaging_action_duration_sec",
    "median_imaging_action_duration_sec",
    "mean_successful_imaging_action_duration_sec",
    "median_successful_imaging_action_duration_sec",
    "mean_unsuccessful_imaging_action_duration_sec",
    "median_unsuccessful_imaging_action_duration_sec",
    "mean_imaging_slew_time_sec",
    "median_imaging_slew_time_sec",
    "mean_successful_imaging_slew_time_sec",
    "median_successful_imaging_slew_time_sec",
    "mean_unsuccessful_imaging_slew_time_sec",
    "median_unsuccessful_imaging_slew_time_sec",
]
CORE_SUMMARY_METRICS = [
    "delivered_ground_value_100d00i",
    "net_evaluation_return_100d00i",
    "operational_penalty_total_100d00i",
    "operational_penalty_count_100d00i",
    "score_ground_value_100d00i",
    "illuminated_images",
    "confirmed_illuminated_images",
    "pending_illuminated_images_onboard",
    "pending_images_onboard",
    "mean_target_priority",
    "std_target_priority",
    "mean_target_illumination_status",
    "num_target_above_illumination_threshold",
    "num_target_below_illumination_threshold",
    "umbra_smart_fraction",
    "umbra_mean_sun_target_dot",
    "umbra_mean_sun_target_sep_deg",
    "target_imaging_count",
    "charge_action_count",
    "downlink_action_count",
    "desat_action_count",
    "frac_image_actions",
    "frac_charge_actions",
    "frac_downlink_actions",
    "frac_desat_actions",
    "downlink_success_rate_reward_proxy",
    "downlink_success_rate_storage_proxy",
    "downlink_positive_reward_count",
    "downlink_storage_reduction_count",
    "downlink_empty_or_noop_count_storage_proxy",
    "downlink_empty_at_command_count",
    "downlink_empty_at_command_fraction",
    "downlink_nonempty_no_transfer_count",
    "downlink_nonempty_no_transfer_fraction",
    "mean_downlink_duration_sec",
    "median_downlink_duration_sec",
    "mean_successful_downlink_duration_sec",
    "median_successful_downlink_duration_sec",
    "mean_empty_downlink_duration_sec",
    "median_empty_downlink_duration_sec",
    "mean_nonempty_no_transfer_downlink_duration_sec",
    "median_nonempty_no_transfer_downlink_duration_sec",
    "mean_ground_value_per_positive_downlink",
    "mean_est_images_removed_per_storage_downlink",
    "image_to_next_reward_downlink_latency_mean_sec",
    "image_to_next_reward_downlink_latency_median_sec",
    "image_to_next_reward_downlink_latency_p90_sec",
    "image_to_next_storage_downlink_latency_mean_sec",
    "image_to_next_storage_downlink_latency_median_sec",
    "image_to_next_storage_downlink_latency_p90_sec",
    "successful_images_with_next_reward_downlink_frac",
    "successful_images_with_next_storage_downlink_frac",
    "final_battery_fraction",
    "final_storage_fraction",
    "final_sim_time_sec",
    "elapsed_seconds",
    "mean_imaging_action_duration_sec",
    "mean_successful_imaging_action_duration_sec",
    "mean_unsuccessful_imaging_action_duration_sec",
    "mean_imaging_slew_time_sec",
    "mean_successful_imaging_slew_time_sec",
    "mean_unsuccessful_imaging_slew_time_sec",
    "raw_device_cmd_avg_acquisition_time_sec",
    "max_abs_accepted_acq_action_duration_delta_sec",
    "mean_initial_ang_error",
    "mean_target_elevation_local",
    "mean_target_distance",
    "time_to_first_hio_candidate_access_sec",
    "time_to_first_shio_candidate_access_sec",
    "time_to_first_hio_command_sec",
    "time_to_first_shio_command_sec",
    "time_to_first_hio_success_sec",
    "time_to_first_shio_success_sec",
    "hio_command_count_after_event",
    "shio_command_count_after_event",
    "hio_successful_capture_count_after_event",
    "shio_successful_capture_count_after_event",
]


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "unknown")
    default_root = Path(
        f"/scratch/alpine/{user}/amos2026_mc/gat_full_actions_eval_100d00i"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-root>/analysis_detailed.",
    )
    parser.add_argument(
        "--expected-seeds",
        default="0:100",
        help="Expected Python-style seed range, for example 0:100.",
    )
    parser.add_argument(
        "--policy-tags",
        default=",".join(POLICY_TAGS),
        help="Comma-separated policy tags expected in the campaign.",
    )
    parser.add_argument(
        "--storage-capacity-images",
        type=float,
        default=50.0,
        help=(
            "Image-equivalent storage capacity used to convert storage fraction "
            "drop during downlink into approximate images transmitted. Full-action "
            "restricted-resource runs currently use 50 images."
        ),
    )
    parser.add_argument(
        "--storage-eps",
        type=float,
        default=1e-6,
        help="Storage-fraction tolerance for detecting non-empty/no-op downlinks.",
    )
    parser.add_argument(
        "--reward-eps",
        type=float,
        default=1e-9,
        help="Reward tolerance for detecting value-bearing downlinks.",
    )
    return parser.parse_args()


def parse_expected_seeds(spec: str) -> list[int]:
    start_text, stop_text = spec.split(":", 1)
    start, stop = int(start_text), int(stop_text)
    if stop <= start:
        raise ValueError("--expected-seeds stop must be greater than start")
    return list(range(start, stop))


def nested_get(payload: dict[str, Any], *keys: str, default=np.nan):
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def latest_file(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def status_paths_under_root(input_root: Path) -> list[Path]:
    """Find MC status files using the known campaign layout without deep scans."""
    return sorted(input_root.glob("seeds_*/*/seed_*/mc_status.json"))


def metrics_files_for_seed(seed_dir: Path) -> list[Path]:
    """Find metrics files one level below a seed dir without walking plots/data."""
    return sorted(seed_dir.glob("metrics_*.json")) + sorted(
        seed_dir.glob("*/metrics_*.json")
    )


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def numeric_column(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def safe_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else np.nan


def action_name(action_id: int) -> str:
    if IMAGE_ACTION_MIN <= action_id <= IMAGE_ACTION_MAX:
        return "image"
    return ACTION_ID_TO_NAME.get(action_id, f"other_{action_id}")


def status_sort_key(status_path: Path) -> tuple[int, float]:
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return (0, status_path.stat().st_mtime)
    completed = 1 if status.get("state") == "completed" else 0
    has_metrics = 1 if metrics_files_for_seed(status_path.parent) else 0
    return (completed + has_metrics, status_path.stat().st_mtime)


def selected_status_paths(input_root: Path) -> list[Path]:
    """Select one best status per policy/seed, preferring completed runs."""
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    unknown: list[Path] = []
    for status_path in status_paths_under_root(input_root):
        try:
            status = json.loads(status_path.read_text())
            key = (str(status.get("policy_tag")), int(status.get("seed")))
        except Exception:
            unknown.append(status_path)
            continue
        grouped[key].append(status_path)

    selected = [max(paths, key=status_sort_key) for paths in grouped.values()]
    return sorted(selected + unknown)


def read_metrics(seed_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    metrics_path = latest_file(metrics_files_for_seed(seed_dir))
    if metrics_path is None:
        return None, None
    try:
        return metrics_path, json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return metrics_path, None


def load_steps(run_dir: Path | None) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "steps.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_images(run_dir: Path | None) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "images.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def step_metrics(
    steps: pd.DataFrame,
    *,
    storage_capacity_images: float,
    storage_eps: float,
    reward_eps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    result: dict[str, Any] = {}
    if steps.empty:
        return result, pd.DataFrame(), pd.DataFrame()

    action = pd.to_numeric(steps.get("action_id"), errors="coerce")
    action = action.fillna(-999).astype(int)
    action_duration = (
        numeric_column(steps, "t_after") - numeric_column(steps, "t_cmd")
    ).clip(lower=0.0)
    reward_step_all = numeric_column(steps, "reward_step", 0.0).fillna(0.0)
    result.update(
        {
            "delivered_ground_value_100d00i": float(
                reward_step_all.clip(lower=0.0).sum()
            ),
            "net_evaluation_return_100d00i": float(reward_step_all.sum()),
            "operational_penalty_total_100d00i": float(
                reward_step_all.clip(upper=0.0).sum()
            ),
            "operational_penalty_count_100d00i": int(
                (reward_step_all < -reward_eps).sum()
            ),
        }
    )
    total = int(len(action))
    for action_id in range(IMAGE_ACTION_MAX + 1):
        result[f"action_id_{action_id}_count"] = int((action == action_id).sum())

    result["num_steps"] = total
    for action_label in ("image", "charge", "downlink", "desat"):
        if action_label == "image":
            mask = (action >= IMAGE_ACTION_MIN) & (action <= IMAGE_ACTION_MAX)
        else:
            action_id = {"charge": 0, "downlink": 1, "desat": 2}[action_label]
            mask = action == action_id
        count = int(mask.sum())
        result[f"{action_label}_action_count_steps_csv"] = count
        result[f"frac_{action_label}_actions"] = count / total if total else np.nan

    image_steps = steps.loc[
        (action >= IMAGE_ACTION_MIN) & (action <= IMAGE_ACTION_MAX)
    ].copy()
    if not image_steps.empty:
        image_steps["action_name"] = "image"
        image_steps["candidate_slot"] = (
            pd.to_numeric(image_steps["action_id"], errors="coerce") - IMAGE_ACTION_MIN
        ).astype("Int64")
        image_steps["action_duration_sec"] = action_duration.loc[image_steps.index]
        image_count = len(image_steps)
        for slot in range(IMAGE_ACTION_MAX - IMAGE_ACTION_MIN + 1):
            slot_count = int((image_steps["candidate_slot"] == slot).sum())
            result[f"image_candidate_slot_{slot}_count"] = slot_count
            result[f"image_candidate_slot_{slot}_fraction"] = (
                slot_count / image_count if image_count else np.nan
            )

    for source, output in (
        ("battery_frac_after", "final_battery_fraction"),
        ("storage_frac_after", "final_storage_fraction"),
        ("t_after", "final_sim_time_sec"),
    ):
        values = numeric_series(steps, source)
        if not values.empty:
            result[output] = float(values.iloc[-1])

    downlink = steps.loc[action == 1].copy()
    if downlink.empty:
        result.update(
            {
                "downlink_success_rate_reward_proxy": np.nan,
                "downlink_success_rate_storage_proxy": np.nan,
                "downlink_positive_reward_count": 0,
                "downlink_storage_reduction_count": 0,
                "downlink_empty_or_noop_count_storage_proxy": 0,
                "downlink_empty_at_command_count": 0,
                "downlink_empty_at_command_fraction": np.nan,
                "downlink_nonempty_no_transfer_count": 0,
                "downlink_nonempty_no_transfer_fraction": np.nan,
                "mean_downlink_duration_sec": np.nan,
                "median_downlink_duration_sec": np.nan,
                "mean_successful_downlink_duration_sec": np.nan,
                "median_successful_downlink_duration_sec": np.nan,
                "mean_empty_downlink_duration_sec": np.nan,
                "median_empty_downlink_duration_sec": np.nan,
                "mean_nonempty_no_transfer_downlink_duration_sec": np.nan,
                "median_nonempty_no_transfer_downlink_duration_sec": np.nan,
                "mean_ground_value_per_positive_downlink": np.nan,
                "mean_est_images_removed_per_storage_downlink": np.nan,
            }
        )
        return result, downlink, image_steps

    reward_step = numeric_column(downlink, "reward_step", 0.0).fillna(0.0)
    storage_cmd = numeric_column(downlink, "storage_frac_cmd")
    storage_after = numeric_column(downlink, "storage_frac_after")
    storage_drop = (storage_cmd - storage_after).clip(lower=0.0).fillna(0.0)
    downlink_duration = action_duration.loc[downlink.index].fillna(0.0)

    positive_reward = reward_step > reward_eps
    storage_reduced = storage_drop > storage_eps
    started_nonempty = storage_cmd.fillna(0.0) > storage_eps
    empty_at_command = ~started_nonempty
    nonempty_no_transfer = started_nonempty & ~storage_reduced
    downlink = downlink.assign(
        action_name="downlink",
        action_duration_sec=downlink_duration,
        positive_reward_proxy=positive_reward.to_numpy(dtype=bool),
        storage_reduced_proxy=storage_reduced.to_numpy(dtype=bool),
        started_nonempty_proxy=started_nonempty.to_numpy(dtype=bool),
        empty_at_command=empty_at_command.to_numpy(dtype=bool),
        nonempty_no_transfer=nonempty_no_transfer.to_numpy(dtype=bool),
        outcome=np.select(
            [
                empty_at_command.to_numpy(dtype=bool),
                storage_reduced.to_numpy(dtype=bool),
                nonempty_no_transfer.to_numpy(dtype=bool),
            ],
            ["empty_at_command", "storage_reduced", "nonempty_no_transfer"],
            default="other",
        ),
        storage_drop_fraction=storage_drop,
        estimated_images_removed=storage_drop * float(storage_capacity_images),
    )

    total_downlinks = len(downlink)
    result.update(
        {
            "downlink_positive_reward_count": int(positive_reward.sum()),
            "downlink_storage_reduction_count": int(storage_reduced.sum()),
            "downlink_started_nonempty_count": int(started_nonempty.sum()),
            "downlink_empty_or_noop_count_storage_proxy": int((~storage_reduced).sum()),
            "downlink_empty_at_command_count": int(empty_at_command.sum()),
            "downlink_empty_at_command_fraction": float(empty_at_command.mean()),
            "downlink_nonempty_no_transfer_count": int(nonempty_no_transfer.sum()),
            "downlink_nonempty_no_transfer_fraction": float(nonempty_no_transfer.mean()),
            "downlink_success_rate_reward_proxy": float(positive_reward.mean()),
            "downlink_success_rate_storage_proxy": float(storage_reduced.mean()),
            "mean_downlink_duration_sec": safe_mean(downlink_duration.to_numpy(dtype=float)),
            "median_downlink_duration_sec": safe_median(downlink_duration.to_numpy(dtype=float)),
            "mean_successful_downlink_duration_sec": safe_mean(
                downlink_duration[storage_reduced].to_numpy(dtype=float)
            ),
            "median_successful_downlink_duration_sec": safe_median(
                downlink_duration[storage_reduced].to_numpy(dtype=float)
            ),
            "mean_empty_downlink_duration_sec": safe_mean(
                downlink_duration[empty_at_command].to_numpy(dtype=float)
            ),
            "median_empty_downlink_duration_sec": safe_median(
                downlink_duration[empty_at_command].to_numpy(dtype=float)
            ),
            "mean_nonempty_no_transfer_downlink_duration_sec": safe_mean(
                downlink_duration[nonempty_no_transfer].to_numpy(dtype=float)
            ),
            "median_nonempty_no_transfer_downlink_duration_sec": safe_median(
                downlink_duration[nonempty_no_transfer].to_numpy(dtype=float)
            ),
            "mean_ground_value_per_downlink_action": float(reward_step.mean()),
            "mean_ground_value_per_positive_downlink": safe_mean(
                reward_step[positive_reward].to_numpy(dtype=float)
            ),
            "sum_ground_value_from_downlink_steps": float(reward_step.sum()),
            "mean_storage_drop_fraction_per_downlink": safe_mean(storage_drop.to_numpy(dtype=float)),
            "mean_est_images_removed_per_downlink": safe_mean(
                (storage_drop * float(storage_capacity_images)).to_numpy(dtype=float)
            ),
            "mean_est_images_removed_per_storage_downlink": safe_mean(
                (storage_drop[storage_reduced] * float(storage_capacity_images)).to_numpy(dtype=float)
            ),
            "total_est_images_removed_by_downlink": float(
                (storage_drop * float(storage_capacity_images)).sum()
            ),
            "downlink_action_count_steps_csv": total_downlinks,
        }
    )
    return result, downlink, image_steps


def next_event_latency(event_times: np.ndarray, image_times: np.ndarray) -> np.ndarray:
    event_times = np.asarray(event_times, dtype=float)
    event_times = event_times[np.isfinite(event_times)]
    image_times = np.asarray(image_times, dtype=float)
    latencies = np.full(image_times.shape, np.nan, dtype=float)
    if event_times.size == 0 or image_times.size == 0:
        return latencies
    event_times.sort()
    indices = np.searchsorted(event_times, image_times, side="left")
    valid = indices < event_times.size
    latencies[valid] = event_times[indices[valid]] - image_times[valid]
    return latencies


def image_metrics(images: pd.DataFrame, downlink_events: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if images.empty:
        return result

    acq_success = pd.to_numeric(images.get("acq_success"), errors="coerce").fillna(0).astype(int)
    successful = images.loc[acq_success == 1].copy()
    result["images_csv_rows"] = int(len(images))
    result["images_csv_successful_acq_count"] = int(len(successful))
    if len(images):
        result["images_csv_acq_success_rate"] = float(len(successful) / len(images))

    for source, prefix in (
        ("target_shadow_cmd", "target_shadow_cmd"),
        ("target_shadow_acq", "target_shadow_acq"),
        ("sat_shadow_cmd", "sat_shadow_cmd"),
        ("sat_shadow_acq", "sat_shadow_acq"),
        ("target_shadow_cmd", "chosen_target_illumination"),
        ("range_m", "range_m"),
        ("elevation_local_deg", "elevation_local_deg"),
        ("azimuth_deg", "azimuth_deg"),
    ):
        values = numeric_series(images, source)
        if not values.empty:
            result[f"{prefix}_mean"] = float(values.mean())
            result[f"{prefix}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan

    if "win_bucket" in images:
        counts = images["win_bucket"].value_counts(dropna=False)
        denom = max(1, len(images))
        for bucket, count in counts.items():
            clean = str(bucket).replace(" ", "_").replace("/", "_")
            result[f"win_bucket_frac_{clean}"] = float(count / denom)

    if "phase_state" in images:
        counts = images["phase_state"].value_counts(dropna=False)
        denom = max(1, len(images))
        for phase, count in counts.items():
            clean = str(phase).replace(" ", "_").replace("/", "_")
            result[f"phase_state_frac_{clean}"] = float(count / denom)

    if successful.empty:
        return result

    image_time_source = "t_acq" if "t_acq" in successful else "t_cmd"
    image_times = pd.to_numeric(successful[image_time_source], errors="coerce").to_numpy(dtype=float)

    reward_event_times = np.array([], dtype=float)
    storage_event_times = np.array([], dtype=float)
    if not downlink_events.empty:
        t_event = pd.to_numeric(downlink_events.get("t_after"), errors="coerce")
        reward_mask = downlink_events.get("positive_reward_proxy", False)
        storage_mask = downlink_events.get("storage_reduced_proxy", False)
        reward_event_times = t_event[np.asarray(reward_mask, dtype=bool)].to_numpy(dtype=float)
        storage_event_times = t_event[np.asarray(storage_mask, dtype=bool)].to_numpy(dtype=float)

    for event_label, event_times in (
        ("reward", reward_event_times),
        ("storage", storage_event_times),
    ):
        lat = next_event_latency(event_times, image_times)
        finite = lat[np.isfinite(lat)]
        result[f"image_to_next_{event_label}_downlink_latency_mean_sec"] = safe_mean(finite)
        result[f"image_to_next_{event_label}_downlink_latency_median_sec"] = safe_median(finite)
        result[f"image_to_next_{event_label}_downlink_latency_p90_sec"] = safe_percentile(finite, 90)
        result[f"successful_images_with_next_{event_label}_downlink_frac"] = (
            float(finite.size / len(successful)) if len(successful) else np.nan
        )
    return result


def build_image_events(images: pd.DataFrame, image_steps: pd.DataFrame) -> pd.DataFrame:
    """Attach policy action-slot and duration data to each saved image command."""
    if images.empty:
        return pd.DataFrame()
    events = images.copy()
    if image_steps.empty:
        events["action_id"] = np.nan
        events["candidate_slot"] = np.nan
        events["action_duration_sec"] = np.nan
        return events

    step_columns = [
        column
        for column in ("t_cmd", "action_id", "candidate_slot", "action_duration_sec")
        if column in image_steps
    ]
    step_data = image_steps[step_columns].copy()
    # Command times originate from the same step loop, so this is an exact join.
    return events.merge(step_data, on="t_cmd", how="left", validate="one_to_one")


def flatten_metrics_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if payload is None:
        return row
    data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}

    for field in DATA_FIELDS:
        row[field] = data.get(field, np.nan)
    row["net_evaluation_return_100d00i"] = data.get(
        "cumulativeRewardSS1", np.nan
    )
    # Deprecated compatibility alias. This quantity includes operational
    # penalties and is not pure delivered ground value.
    row["score_ground_value_100d00i"] = row["net_evaluation_return_100d00i"]

    reason_counts = data.get("umbra_smart_reason_counts", {})
    if isinstance(reason_counts, dict):
        for reason, value in reason_counts.items():
            row[f"umbra_smart_reason_count_{reason}"] = value

    for field in SUMMARY_FIELDS:
        row[field] = summary.get(field, np.nan)

    def flatten_group(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                flatten_group(f"{prefix}_{key}", nested_value)
            return
        if isinstance(value, (int, float, str)) or value is None:
            row[prefix] = value

    for group_name in ("look_metrics", "regime_metrics"):
        group = summary.get(group_name, {})
        if isinstance(group, dict):
            for key, value in group.items():
                flatten_group(f"{group_name}_{key}", value)

    for field in (
        "policy_layout",
        "obs_v",
        "reward_mix_tag",
        "use_shield",
        "dynamic_priority_event_enabled",
        "dynamic_priority_event_fraction",
        "dynamic_priority_event_time_sec",
    ):
        row[field] = meta.get(field, np.nan)
    return row


def load_record(
    status_path: Path,
    *,
    storage_capacity_images: float,
    storage_eps: float,
    reward_eps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    status = json.loads(status_path.read_text())
    seed_dir = status_path.parent
    metrics_path, payload = read_metrics(seed_dir)
    run_dir = metrics_path.parent if metrics_path is not None else None
    steps = load_steps(run_dir)
    images = load_images(run_dir)
    step_row, downlink_events, image_steps = step_metrics(
        steps,
        storage_capacity_images=storage_capacity_images,
        storage_eps=storage_eps,
        reward_eps=reward_eps,
    )
    image_row = image_metrics(images, downlink_events)
    image_events = build_image_events(images, image_steps)

    row: dict[str, Any] = {
        "policy_tag": status.get("policy_tag"),
        "seed": status.get("seed"),
        "state": status.get("state"),
        "returncode": status.get("returncode"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "status_path": str(status_path),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "run_dir": str(run_dir) if run_dir else None,
        "checkpoint_iteration": nested_get(status, "policy", "checkpoint_iteration"),
        "checkpoint_dir": nested_get(status, "policy", "checkpoint_dir", default=None),
        "evaluation_reward_mix": status.get("evaluation_reward_mix"),
        "target_env": status.get("target_env"),
        "mix_weights": status.get("mix_weights"),
        "exact_mix_counts": status.get("exact_mix_counts"),
        "priority_sum": status.get("priority_sum", 100.0),
        "n_targets": status.get("n_targets", 100),
        "n_targets_ahead": status.get("n_targets_ahead", 10),
        "dynamic_priority_event": status.get("dynamic_priority_event"),
        "use_shield": status.get("use_shield"),
    }
    row.update(flatten_metrics_payload(payload))
    row.update(step_row)
    row.update(image_row)

    if not downlink_events.empty:
        event_rows = downlink_events.copy()
        event_rows.insert(0, "policy_tag", row["policy_tag"])
        event_rows.insert(1, "seed", row["seed"])
        event_rows.insert(2, "status_path", str(status_path))
    else:
        event_rows = pd.DataFrame()
    if not image_events.empty:
        image_events.insert(0, "policy_tag", row["policy_tag"])
        image_events.insert(1, "seed", row["seed"])
        image_events.insert(2, "status_path", str(status_path))
    return row, event_rows, image_events


def aggregate_summary(records: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    completed = records[(records["state"] == "completed") & records["metrics_path"].notna()].copy()
    for tag in POLICY_TAGS:
        subset = completed[completed["policy_tag"] == tag]
        row: dict[str, Any] = {"policy_tag": tag, "n_runs": int(len(subset))}
        for metric in metrics:
            values = numeric_series(subset, metric)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
            row[f"{metric}_ci95"] = (
                float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def action_distribution(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tag in POLICY_TAGS:
        subset = records[(records["policy_tag"] == tag) & (records["state"] == "completed")]
        totals = {
            "image": numeric_series(subset, "image_action_count_steps_csv").sum(),
            "charge": numeric_series(subset, "charge_action_count_steps_csv").sum(),
            "downlink": numeric_series(subset, "downlink_action_count_steps_csv").sum(),
            "desat": numeric_series(subset, "desat_action_count_steps_csv").sum(),
        }
        total_actions = float(sum(totals.values()))
        row: dict[str, Any] = {"policy_tag": tag, "total_actions": total_actions}
        for name, count in totals.items():
            row[f"{name}_actions"] = float(count)
            row[f"frac_{name}_actions"] = float(count / total_actions) if total_actions else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def action_id_distribution(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every discrete action ID, including each image candidate slot."""
    rows: list[dict[str, Any]] = []
    completed = records[records["state"] == "completed"]
    for tag in POLICY_TAGS:
        subset = completed[completed["policy_tag"] == tag]
        counts = {
            action_id: float(numeric_series(subset, f"action_id_{action_id}_count").sum())
            for action_id in range(IMAGE_ACTION_MAX + 1)
        }
        total = float(sum(counts.values()))
        for action_id, count in counts.items():
            rows.append(
                {
                    "policy_tag": tag,
                    "action_id": action_id,
                    "action_name": action_name(action_id),
                    "candidate_slot": (
                        action_id - IMAGE_ACTION_MIN
                        if IMAGE_ACTION_MIN <= action_id <= IMAGE_ACTION_MAX
                        else np.nan
                    ),
                    "count": count,
                    "fraction_all_actions": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def image_candidate_summary(image_events: pd.DataFrame) -> pd.DataFrame:
    """Summarize which elevation-ordered image candidate slot the policy selected."""
    if image_events.empty or "candidate_slot" not in image_events:
        return pd.DataFrame()
    events = image_events.copy()
    events["candidate_slot"] = pd.to_numeric(
        events["candidate_slot"], errors="coerce"
    )
    events = events[events["candidate_slot"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for tag in POLICY_TAGS:
        policy_events = events[events["policy_tag"] == tag]
        total = len(policy_events)
        for slot in range(IMAGE_ACTION_MAX - IMAGE_ACTION_MIN + 1):
            subset = policy_events[policy_events["candidate_slot"] == slot]
            success = numeric_series(subset, "acq_success")
            rows.append(
                {
                    "policy_tag": tag,
                    "candidate_slot": slot,
                    "action_id": slot + IMAGE_ACTION_MIN,
                    "count": int(len(subset)),
                    "fraction_image_actions": len(subset) / total if total else np.nan,
                    "acquisition_success_rate": float(success.mean()) if len(success) else np.nan,
                    "mean_selected_elevation_deg": safe_mean(
                        numeric_series(subset, "elevation_local_deg").to_numpy(dtype=float)
                    ),
                    "mean_action_duration_sec": safe_mean(
                        numeric_series(subset, "action_duration_sec").to_numpy(dtype=float)
                    ),
                    "mean_range_km": safe_mean(
                        numeric_series(subset, "range_m").to_numpy(dtype=float) / 1000.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def downlink_summary(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "downlink_action_count_steps_csv",
        "downlink_positive_reward_count",
        "downlink_storage_reduction_count",
        "downlink_started_nonempty_count",
        "downlink_empty_or_noop_count_storage_proxy",
        "downlink_empty_at_command_count",
        "downlink_empty_at_command_fraction",
        "downlink_nonempty_no_transfer_count",
        "downlink_nonempty_no_transfer_fraction",
        "downlink_success_rate_reward_proxy",
        "downlink_success_rate_storage_proxy",
        "mean_downlink_duration_sec",
        "median_downlink_duration_sec",
        "mean_successful_downlink_duration_sec",
        "median_successful_downlink_duration_sec",
        "mean_empty_downlink_duration_sec",
        "median_empty_downlink_duration_sec",
        "mean_nonempty_no_transfer_downlink_duration_sec",
        "median_nonempty_no_transfer_downlink_duration_sec",
        "mean_ground_value_per_positive_downlink",
        "mean_est_images_removed_per_storage_downlink",
        "total_est_images_removed_by_downlink",
        "image_to_next_reward_downlink_latency_mean_sec",
        "image_to_next_reward_downlink_latency_median_sec",
        "image_to_next_reward_downlink_latency_p90_sec",
        "successful_images_with_next_reward_downlink_frac",
    ]
    return aggregate_summary(records, metrics)


def plot_score_box(records: pd.DataFrame, output_dir: Path) -> None:
    metric = "delivered_ground_value_100d00i"
    completed = records[
        (records["state"] == "completed") & records[metric].notna()
    ]
    if completed.empty:
        return
    values = [
        numeric_series(completed[completed["policy_tag"] == tag], metric)
        for tag in POLICY_TAGS
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot(values, tick_labels=POLICY_TAGS, showmeans=True)
    ax.set_ylabel("Delivered ground value under 100d00i evaluation")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Training reward mix")
    ax.set_title("AMOS 2026 GAT full-action policies, common 100d00i scoring")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "score_ground_value_boxplot.png", dpi=180)
    plt.close(fig)


def plot_action_stack(actions: pd.DataFrame, output_dir: Path) -> None:
    if actions.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(actions))
    x = np.arange(len(actions))
    colors = {
        "image": "#2f7ed8",
        "charge": "#7cb342",
        "downlink": "#f39c12",
        "desat": "#8e44ad",
    }
    for name in ("image", "charge", "downlink", "desat"):
        vals = actions[f"frac_{name}_actions"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=name, color=colors[name])
        bottom += np.nan_to_num(vals)
    ax.set_xticks(x, actions["policy_tag"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of actions")
    ax.set_xlabel("Training reward mix")
    ax.set_title("Action distribution by policy")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "action_distribution_stacked.png", dpi=180)
    plt.close(fig)


def plot_downlink_success(records: pd.DataFrame, output_dir: Path) -> None:
    completed = records[records["state"] == "completed"].copy()
    if completed.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(POLICY_TAGS))
    reward_means = []
    storage_means = []
    for tag in POLICY_TAGS:
        subset = completed[completed["policy_tag"] == tag]
        reward_means.append(numeric_series(subset, "downlink_success_rate_reward_proxy").mean())
        storage_means.append(numeric_series(subset, "downlink_success_rate_storage_proxy").mean())
    width = 0.36
    ax.bar(x - width / 2, reward_means, width, label="positive reward proxy")
    ax.bar(x + width / 2, storage_means, width, label="storage reduction proxy")
    ax.set_xticks(x, POLICY_TAGS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Successful fraction of downlink actions")
    ax.set_xlabel("Training reward mix")
    ax.set_title("Downlink usefulness proxies")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "downlink_success_proxy_by_policy.png", dpi=180)
    plt.close(fig)


def plot_latency(records: pd.DataFrame, output_dir: Path) -> None:
    completed = records[records["state"] == "completed"].copy()
    values = [
        numeric_series(
            completed[completed["policy_tag"] == tag],
            "image_to_next_reward_downlink_latency_median_sec",
        )
        / 60.0
        for tag in POLICY_TAGS
    ]
    if not any(len(v) for v in values):
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot(values, tick_labels=POLICY_TAGS, showmeans=True)
    ax.set_ylabel("Median image-to-next-value-downlink latency proxy [min]")
    ax.set_xlabel("Training reward mix")
    ax.set_title("Data timeliness proxy by policy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "image_to_downlink_latency_proxy_boxplot.png", dpi=180)
    plt.close(fig)


def plot_priority_vs_score(records: pd.DataFrame, output_dir: Path) -> None:
    metric = "delivered_ground_value_100d00i"
    completed = records[
        (records["state"] == "completed") & records[metric].notna()
    ].copy()
    if completed.empty or "mean_target_priority" not in completed:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for tag in POLICY_TAGS:
        subset = completed[completed["policy_tag"] == tag]
        x = numeric_series(subset, "mean_target_priority")
        y = numeric_series(subset, metric)
        joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if not joined.empty:
            ax.scatter(joined["x"], joined["y"], s=18, alpha=0.55, label=tag)
    ax.set_xlabel("Mean chosen target priority")
    ax.set_ylabel("Delivered ground value")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Does selecting higher-priority targets correlate with score?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_target_priority_vs_score.png", dpi=180)
    plt.close(fig)


def write_metric_definitions(output_dir: Path) -> None:
    definitions = {
        "umbra_mean_sun_target_dot": (
            "Mean dot product between the scanner-to-Sun direction and the scanner-to-target "
            "direction during scanner umbra/low-shadowFactor decisions. Range is [-1, 1]. "
            "Larger values mean the target lies more sunward from the scanner; 0 is roughly "
            "perpendicular; negative is anti-sunward."
        ),
        "umbra_mean_sun_target_sep_deg": (
            "Mean angular separation, in degrees, between the scanner-to-Sun and scanner-to-target "
            "directions during those umbra decisions. It is approximately arccos(dot). Smaller "
            "angles mean more sunward alignment."
        ),
        "downlink_success_rate_reward_proxy": (
            "Fraction of downlink actions whose step reward is positive under the common 100d00i "
            "evaluation. This identifies value-bearing downlinks but does not count exact packets."
        ),
        "delivered_ground_value_100d00i": (
            "Sum of positive reward_step values under alpha_eval=1. This is the "
            "nonnegative priority-weighted value delivered to the ground."
        ),
        "net_evaluation_return_100d00i": (
            "Sum of all reward_step values under alpha_eval=1, including negative "
            "operational penalties such as an empty-start downlink."
        ),
        "operational_penalty_total_100d00i": (
            "Sum of negative reward_step values. This is reported separately from "
            "delivered ground value."
        ),
        "downlink_success_rate_storage_proxy": (
            "Fraction of downlink actions that reduced onboard storage fraction by more than the "
            "configured tolerance. This identifies storage-clearing downlinks, including cases where "
            "the delivered packets may have low/no 100d00i value."
        ),
        "downlink_empty_at_command_fraction": (
            "Fraction of downlink actions for which storage_frac_cmd is at or below "
            "--storage-eps. This is the direct empty-storage command metric, distinct "
            "from a downlink that starts nonempty but transfers no data."
        ),
        "downlink_nonempty_no_transfer_fraction": (
            "Fraction of downlink actions that start with nonzero storage but do not "
            "reduce storage by more than --storage-eps."
        ),
        "*_downlink_duration_sec": (
            "Wall-clock simulation duration t_after - t_cmd for downlink actions, "
            "reported for all, storage-reducing, empty-at-command, and nonempty-no-transfer cases."
        ),
        "image_candidate_slot": (
            "For GAT full-action policies, action IDs 3..12 map to candidate slots 0..9. "
            "The ImageRSO action sorts currently visible eligible targets by ascending local "
            "elevation before taking the first ten. If fewer than ten are visible, remaining "
            "slots are padded by closest eligible targets, so the slot is not always a pure "
            "elevation rank."
        ),
        "HIO_SHIO_metrics": (
            "Exact HIO/SHIO access, command, and capture latency requires event target IDs. "
            "These fields are available only for evaluations produced after explicit priority-event "
            "logging was added; the existing 800-run archive does not contain those identities."
        ),
        "mean_est_images_removed_per_storage_downlink": (
            "Mean storage fraction decrease during storage-reducing downlinks multiplied by "
            "--storage-capacity-images. This is image-equivalent throughput, not an exact packet count."
        ),
        "image_to_next_reward_downlink_latency_*": (
            "For every successful image acquisition in images.csv, time to the next later downlink "
            "step with positive reward. This is a timeliness proxy because current outputs do not "
            "store per-image packet IDs linking each capture to its exact downlink event."
        ),
        "image_to_next_storage_downlink_latency_*": (
            "For every successful image acquisition, time to the next later downlink that reduced "
            "storage. Also a proxy, not exact packet-level delivery latency."
        ),
    }
    (output_dir / "metric_definitions.json").write_text(
        json.dumps(definitions, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    global POLICY_TAGS
    args = parse_args()
    POLICY_TAGS = tuple(
        tag.strip() for tag in args.policy_tags.split(",") if tag.strip()
    )
    if not POLICY_TAGS:
        raise ValueError("--policy-tags cannot be empty")
    output_dir = args.output_dir or args.input_root / "analysis_detailed"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = parse_expected_seeds(args.expected_seeds)

    status_paths = selected_status_paths(args.input_root)
    if not status_paths:
        print(f"No mc_status.json files found below {args.input_root}")
        return 1

    records: list[dict[str, Any]] = []
    downlink_event_frames: list[pd.DataFrame] = []
    image_event_frames: list[pd.DataFrame] = []
    for status_path in status_paths:
        row, downlink_events, image_events = load_record(
            status_path,
            storage_capacity_images=args.storage_capacity_images,
            storage_eps=args.storage_eps,
            reward_eps=args.reward_eps,
        )
        records.append(row)
        if not downlink_events.empty:
            downlink_event_frames.append(downlink_events)
        if not image_events.empty:
            image_event_frames.append(image_events)

    records_df = pd.DataFrame(records)
    records_df["seed"] = pd.to_numeric(records_df["seed"], errors="coerce")
    records_df.to_csv(output_dir / "detailed_per_run.csv", index=False)

    metrics = [m for m in CORE_SUMMARY_METRICS if m in records_df.columns]
    summary = aggregate_summary(records_df, metrics)
    summary.to_csv(output_dir / "detailed_summary_by_policy.csv", index=False)

    actions = action_distribution(records_df)
    actions.to_csv(output_dir / "action_distribution_by_policy.csv", index=False)

    action_ids = action_id_distribution(records_df)
    action_ids.to_csv(output_dir / "action_id_distribution_by_policy.csv", index=False)

    downlinks = downlink_summary(records_df)
    downlinks.to_csv(output_dir / "downlink_summary_by_policy.csv", index=False)

    if downlink_event_frames:
        pd.concat(downlink_event_frames, ignore_index=True).to_csv(
            output_dir / "downlink_events.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "downlink_events.csv", index=False)

    if image_event_frames:
        image_events_df = pd.concat(image_event_frames, ignore_index=True)
        image_events_df.to_csv(output_dir / "image_events.csv", index=False)
    else:
        image_events_df = pd.DataFrame()
        image_events_df.to_csv(output_dir / "image_events.csv", index=False)

    candidate_summary = image_candidate_summary(image_events_df)
    candidate_summary.to_csv(
        output_dir / "image_candidate_slot_summary_by_policy.csv", index=False
    )

    observed = {
        (str(row.policy_tag), int(row.seed))
        for row in records_df.itertuples()
        if pd.notna(row.seed)
    }
    expected = {(tag, seed) for tag in POLICY_TAGS for seed in expected_seeds}
    missing = sorted(expected - observed)
    failed = records_df[records_df["state"] != "completed"].copy()
    pd.DataFrame(missing, columns=["policy_tag", "seed"]).to_csv(
        output_dir / "missing_runs.csv", index=False
    )
    failed.to_csv(output_dir / "failed_runs.csv", index=False)

    plot_score_box(records_df, output_dir)
    plot_action_stack(actions, output_dir)
    plot_downlink_success(records_df, output_dir)
    plot_latency(records_df, output_dir)
    plot_priority_vs_score(records_df, output_dir)
    write_metric_definitions(output_dir)

    report = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": str(args.input_root.resolve()),
        "expected_seeds": args.expected_seeds,
        "selected_status_files": len(status_paths),
        "completed_runs": int((records_df["state"] == "completed").sum()),
        "failed_or_incomplete_runs": int(len(failed)),
        "missing_runs": len(missing),
        "storage_capacity_images": args.storage_capacity_images,
        "notes": [
            "Downlink latency metrics are next-event proxies, not exact packet-level delivery latency.",
            "Exact image-to-ground latency requires logging capture packet IDs and downlink verification events during evaluation.",
        ],
    }
    (output_dir / "detailed_analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    display_cols = [
        "policy_tag",
        "n_runs",
        "delivered_ground_value_100d00i_mean",
        "net_evaluation_return_100d00i_mean",
        "illuminated_images_mean",
        "confirmed_illuminated_images_mean",
        "mean_target_priority_mean",
        "frac_downlink_actions_mean",
        "downlink_success_rate_reward_proxy_mean",
        "image_to_next_reward_downlink_latency_median_sec_mean",
    ]
    display_cols = [col for col in display_cols if col in summary]
    print(summary[display_cols].to_string(index=False))
    print()
    print(f"Completed runs: {report['completed_runs']}")
    print(f"Failed or incomplete runs: {report['failed_or_incomplete_runs']}")
    print(f"Missing expected runs: {report['missing_runs']}")
    print(f"Detailed analysis written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
