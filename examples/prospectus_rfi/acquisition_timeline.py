"""Acquisition-time recording and common-grid utilities for paired MC analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from examples.prospectus_rfi.heuristic_mc_design import CATALOG_SIZES

EPISODE_DURATION_S = 45_000.0
PLOT_GRID_INTERVAL_S = 100.0
TABLE_CHECKPOINTS_S = (15_000.0, 30_000.0, 45_000.0)
METHODS = ("heuristic_historical", "legacy_amos2025_alpha0_policy")
TASKS_PER_METHOD = len(CATALOG_SIZES) * 100
TOTAL_TIMELINE_TASKS = len(METHODS) * TASKS_PER_METHOD


@dataclass(frozen=True)
class TimelineTask:
    """One method, catalog size, and scenario seed per Slurm task."""

    task_id: int
    method: str
    catalog_size: int
    seed: int


def timeline_task_spec(task_id: int) -> TimelineTask:
    """Map task IDs 0..599 onto two paired 300-episode campaigns."""

    if not 0 <= task_id < TOTAL_TIMELINE_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TIMELINE_TASKS - 1}]")
    method_index, method_task_id = divmod(task_id, TASKS_PER_METHOD)
    catalog_index, seed = divmod(method_task_id, 100)
    return TimelineTask(
        task_id=task_id,
        method=METHODS[method_index],
        catalog_size=CATALOG_SIZES[catalog_index],
        seed=seed,
    )


def timeline_stem(method: str, catalog_size: int, seed: int) -> str:
    """Return the basename shared by a timeline CSV and its metadata."""

    return f"{method}_n{catalog_size}_seed{seed:03d}"


def trajectory_snapshot(base_env: Any) -> dict[str, float]:
    """Capture cumulative physical metrics at the current decision epoch."""

    base = getattr(base_env, "env", base_env)
    scanner = base.satellites[0]
    rewarder = base.rewarder
    reward_data = rewarder.data
    action_counts = getattr(scanner, "study_action_counts", {})
    return {
        "sim_time_s": float(base.simulator.sim_time),
        "cumulative_successful_observations": float(
            len(getattr(reward_data, "imaged", []))
        ),
        "cumulative_illuminated_observations": float(
            len(set(getattr(rewarder, "imaged_illuminated_names", set())))
        ),
        "cumulative_useful_deliveries": float(getattr(rewarder, "useful_downlinks", 0)),
        "onboard_backlog_fraction": float(scanner.dynamics.storage_level_fraction),
        "battery_fraction": float(scanner.dynamics.battery_charge_fraction),
        "image_action_count": float(action_counts.get("image", 0)),
        "charge_action_count": float(action_counts.get("charge", 0)),
        "downlink_action_count": float(action_counts.get("downlink", 0)),
        "desaturation_action_count": float(action_counts.get("desaturate", 0)),
        "resource_constraint_interventions": float(
            getattr(base, "study_constraint_interventions", 0)
        ),
    }


def append_trajectory_snapshot(rows: list[dict[str, Any]], base_env: Any) -> None:
    """Append one decision-epoch snapshot, replacing a duplicate timestamp."""

    snapshot = trajectory_snapshot(base_env)
    if rows and np.isclose(rows[-1]["sim_time_s"], snapshot["sim_time_s"]):
        rows[-1] = snapshot
    else:
        rows.append(snapshot)


def resample_step_trajectory(
    frame: pd.DataFrame,
    *,
    interval_s: float = PLOT_GRID_INTERVAL_S,
    duration_s: float = EPISODE_DURATION_S,
) -> pd.DataFrame:
    """Forward-fill decision-epoch values onto a uniform, unsmoothed time grid."""

    if interval_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("interval_s and duration_s must be positive")
    if frame.empty:
        raise ValueError("trajectory is empty")
    source = frame.sort_values("sim_time_s").drop_duplicates("sim_time_s", keep="last")
    times = source["sim_time_s"].to_numpy(dtype=float)
    if times[0] > 0.0 or times[-1] < duration_s:
        raise ValueError("trajectory must span time 0 through the episode duration")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("trajectory times must be nondecreasing")
    grid = np.arange(0.0, duration_s + interval_s * 0.5, interval_s)
    indices = np.searchsorted(times, grid, side="right") - 1
    sampled = source.iloc[indices].reset_index(drop=True).copy()
    sampled["sim_time_s"] = grid
    sampled["timeline_grid_interval_s"] = float(interval_s)
    return sampled


def verify_replay(reference: pd.Series, replay: dict[str, Any]) -> dict[str, Any]:
    """Require a timeline replay to reproduce the accepted episode outcome."""

    exact_columns = ("scenario_fingerprint", "method")
    integer_columns = ("catalog_size", "scenario_seed")
    count_columns = (
        "successful_observations",
        "illuminated_observations",
        "useful_deliveries",
        "total_downlinks",
    )
    errors: list[str] = []
    for column in exact_columns:
        if str(reference[column]) != str(replay[column]):
            errors.append(
                f"{column}: replay={replay[column]!r}, reference={reference[column]!r}"
            )
    for column in integer_columns:
        if int(reference[column]) != int(replay[column]):
            errors.append(
                f"{column}: replay={replay[column]!r}, reference={reference[column]!r}"
            )
    for column in count_columns:
        if not np.isclose(float(reference[column]), float(replay[column])):
            errors.append(
                f"{column}: replay={replay[column]!r}, reference={reference[column]!r}"
            )
    if not np.isclose(
        float(reference["episode_duration_s"]),
        float(replay["episode_duration_s"]),
    ):
        errors.append("episode_duration_s differs from the accepted episode")
    if errors:
        raise ValueError(
            "timeline replay did not reproduce raw result:\n- " + "\n- ".join(errors)
        )
    return {
        "verified_against_existing_raw": True,
        "verified_columns": [
            *exact_columns,
            *integer_columns,
            *count_columns,
            "episode_duration_s",
        ],
    }


def load_timeline_files(root: Path, method: str) -> pd.DataFrame:
    """Load all timeline sidecars for one method from an MC campaign root."""

    paths = sorted((root / "timeline" / "raw").glob("n*/*.timeline.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no timeline CSV files under {root / 'timeline' / 'raw'}"
        )
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if set(frame["method"].astype(str)) != {method}:
            raise ValueError(f"unexpected method in {path}")
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "EPISODE_DURATION_S",
    "METHODS",
    "PLOT_GRID_INTERVAL_S",
    "TABLE_CHECKPOINTS_S",
    "TOTAL_TIMELINE_TASKS",
    "TimelineTask",
    "append_trajectory_snapshot",
    "load_timeline_files",
    "resample_step_trajectory",
    "timeline_stem",
    "timeline_task_spec",
    "trajectory_snapshot",
    "verify_replay",
]
