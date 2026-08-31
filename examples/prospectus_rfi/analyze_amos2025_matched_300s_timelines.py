#!/usr/bin/env python3
"""Analyze cumulative image acquisition for the matched 300-second campaign."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from examples.prospectus_rfi.acquisition_timeline import (
    EPISODE_DURATION_S,
    resample_step_trajectory,
)
from examples.prospectus_rfi.amos2025_matched_300s_design import METHODS
from examples.prospectus_rfi.analyze_amos2025_matched_300s_results import (
    METHOD_LABELS,
    REFERENCE_METHOD,
    holm_adjust,
)

COUNT_COLUMN = "cumulative_illuminated_observations"
SEEDS = tuple(range(100))
CHECKPOINTS_S = (15_000.0, 30_000.0, 45_000.0)
COLORS = {
    "breckenridge2026_alpha0_mlp": "#365f9d",
    "target_set_attention": "#009e73",
    "smallest_angle_heuristic": "#d55e00",
    "closest_distance_heuristic": "#cc79a7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--grid-seconds", type=float, default=300.0)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    return parser.parse_args()


def load_and_validate(root: Path) -> pd.DataFrame:
    frames = []
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    for method in METHODS:
        paths = sorted((root / "timeline" / "raw" / method).glob("*.timeline.csv"))
        if len(paths) != len(SEEDS):
            raise ValueError(
                f"{method} has {len(paths)} timelines; expected {len(SEEDS)}"
            )
        for path in paths:
            frame = pd.read_csv(path)
            if set(frame["method"].astype(str)) != {method}:
                raise ValueError(f"wrong method label in {path}")
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    actual = {
        (str(method), int(seed))
        for method, seed in combined[["method", "scenario_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if actual != expected:
        raise ValueError(
            f"timeline design mismatch: missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    identity = combined[
        ["method", "scenario_seed", "scenario_fingerprint"]
    ].drop_duplicates()
    if not np.all(identity.groupby("scenario_seed")["method"].nunique() == 4):
        raise ValueError("one or more seeds do not contain all four methods")
    if not np.all(
        identity.groupby("scenario_seed")["scenario_fingerprint"].nunique() == 1
    ):
        raise ValueError("scenario fingerprints differ between methods")
    for (method, seed), episode in combined.groupby(
        ["method", "scenario_seed"], sort=False
    ):
        ordered = episode.sort_values("sim_time_s")
        times = ordered["sim_time_s"].to_numpy(dtype=float)
        counts = ordered[COUNT_COLUMN].to_numpy(dtype=float)
        if not np.isclose(times[0], 0.0) or not np.isclose(
            times[-1], EPISODE_DURATION_S
        ):
            raise ValueError(f"incomplete timeline for {method}, seed={seed}")
        if np.any(np.diff(times) <= 0.0) or np.any(np.diff(counts) < 0.0):
            raise ValueError(f"invalid trajectory for {method}, seed={seed}")
    return combined


def resample_all(frame: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    pieces = []
    for _, episode in frame.groupby(["method", "scenario_seed"], sort=True):
        pieces.append(
            resample_step_trajectory(
                episode, interval_s=interval_s, duration_s=EPISODE_DURATION_S
            )
        )
    return pd.concat(pieces, ignore_index=True)


def bootstrap_mean_curve(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    boot = np.empty((draws, values.shape[1]), dtype=np.float32)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        indices = rng.integers(0, values.shape[0], size=(stop - start, values.shape[0]))
        boot[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return low.astype(float), high.astype(float)


def curve_tables(
    frame: pd.DataFrame, draws: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(2_026_08_31)
    summaries = []
    matrices: dict[str, np.ndarray] = {}
    grid: np.ndarray | None = None
    for method in METHODS:
        pivot = (
            frame[frame["method"] == method]
            .pivot(index="scenario_seed", columns="sim_time_s", values=COUNT_COLUMN)
            .sort_index()
        )
        if tuple(pivot.index.astype(int)) != SEEDS:
            raise ValueError(f"seed order is incomplete for {method}")
        matrix = pivot.to_numpy(dtype=float)
        matrices[method] = matrix
        grid = pivot.columns.to_numpy(dtype=float)
        low, high = bootstrap_mean_curve(matrix, rng, draws)
        summaries.append(
            pd.DataFrame(
                {
                    "method": method,
                    "sim_time_s": grid,
                    "seed_count": len(SEEDS),
                    "mean": matrix.mean(axis=0),
                    "std": matrix.std(axis=0, ddof=1),
                    "median": np.median(matrix, axis=0),
                    "q25": np.quantile(matrix, 0.25, axis=0),
                    "q75": np.quantile(matrix, 0.75, axis=0),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                }
            )
        )
    assert grid is not None
    differences = []
    reference = matrices[REFERENCE_METHOD]
    for method in METHODS:
        if method == REFERENCE_METHOD:
            continue
        delta = matrices[method] - reference
        low, high = bootstrap_mean_curve(delta, rng, draws)
        differences.append(
            pd.DataFrame(
                {
                    "method": method,
                    "reference_method": REFERENCE_METHOD,
                    "sim_time_s": grid,
                    "paired_seed_count": len(SEEDS),
                    "mean_paired_difference": delta.mean(axis=0),
                    "median_paired_difference": np.median(delta, axis=0),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                }
            )
        )
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(differences, ignore_index=True),
        matrices,
    )


def first_time_at_least(
    times: np.ndarray, counts: np.ndarray, threshold: float
) -> float:
    indices = np.flatnonzero(counts >= threshold)
    return float(times[indices[0]]) if indices.size else np.nan


def episode_timing_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, seed), episode in frame.groupby(
        ["method", "scenario_seed"], sort=True
    ):
        ordered = episode.sort_values("sim_time_s")
        times = ordered["sim_time_s"].to_numpy(dtype=float)
        counts = ordered[COUNT_COLUMN].to_numpy(dtype=float)
        final = counts[-1]
        row = {
            "method": method,
            "scenario_seed": int(seed),
            "scenario_fingerprint": ordered.iloc[0]["scenario_fingerprint"],
            "final_illuminated_observations": final,
            "normalized_illuminated_auc": float(
                np.trapz(counts / 100.0, times) / EPISODE_DURATION_S
            ),
            "time_to_50_images_s": first_time_at_least(times, counts, 50.0),
            "time_to_75_images_s": first_time_at_least(times, counts, 75.0),
            "time_to_90_images_s": first_time_at_least(times, counts, 90.0),
            "time_to_90pct_final_s": first_time_at_least(times, counts, 0.90 * final),
        }
        for checkpoint in CHECKPOINTS_S:
            index = int(np.searchsorted(times, checkpoint, side="right") - 1)
            row[f"illuminated_observations_at_{int(checkpoint)}s"] = counts[index]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_timing_statistics(episodes: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "illuminated_observations_at_15000s",
        "illuminated_observations_at_30000s",
        "final_illuminated_observations",
        "normalized_illuminated_auc",
        "time_to_50_images_s",
        "time_to_75_images_s",
        "time_to_90_images_s",
        "time_to_90pct_final_s",
    ]
    indexed = episodes.set_index(["scenario_seed", "method"])
    rows = []
    for metric in metrics:
        reference = indexed.xs(REFERENCE_METHOD, level="method")[metric]
        for method in METHODS:
            if method == REFERENCE_METHOD:
                continue
            candidate = indexed.xs(method, level="method")[metric]
            paired = pd.concat(
                [candidate.rename("candidate"), reference.rename("reference")], axis=1
            ).dropna()
            difference = (paired["candidate"] - paired["reference"]).to_numpy(
                dtype=float
            )
            if len(difference) == 0:
                p_value = np.nan
            elif np.allclose(difference, 0.0):
                p_value = 1.0
            else:
                try:
                    p_value = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_value = 1.0
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "reference_method": REFERENCE_METHOD,
                    "paired_seed_count": len(difference),
                    "method_mean": float(paired["candidate"].mean()),
                    "reference_mean": float(paired["reference"].mean()),
                    "mean_paired_difference": (
                        float(np.mean(difference)) if len(difference) else np.nan
                    ),
                    "median_paired_difference": (
                        float(np.median(difference)) if len(difference) else np.nan
                    ),
                    "wilcoxon_p_raw": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = result.groupby("metric", group_keys=False)[
        "wilcoxon_p_raw"
    ].apply(holm_adjust)
    return result


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    output = root / "analysis" / "timeline" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_mean_curves(summary: pd.DataFrame, root: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for method in METHODS:
        data = summary[summary["method"] == method]
        x = data["sim_time_s"].to_numpy(dtype=float)
        axis.plot(x, data["mean"], color=COLORS[method], label=METHOD_LABELS[method])
        axis.fill_between(
            x,
            data["bootstrap_95_ci_low"],
            data["bootstrap_95_ci_high"],
            color=COLORS[method],
            alpha=0.14,
            linewidth=0.0,
        )
    axis.set(
        xlabel="Simulation time (s)",
        ylabel="Cumulative illuminated images",
        xlim=(0, 45_000),
    )
    axis.set_xticks((0, 15_000, 30_000, 45_000))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    save_figure(fig, root, "cumulative_illuminated_images_mean_95ci")


def plot_median_curves(summary: pd.DataFrame, root: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for method in METHODS:
        data = summary[summary["method"] == method]
        x = data["sim_time_s"].to_numpy(dtype=float)
        axis.plot(
            x,
            data["median"],
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        axis.fill_between(
            x,
            data["q25"],
            data["q75"],
            color=COLORS[method],
            alpha=0.14,
            linewidth=0.0,
        )
    axis.set(
        xlabel="Simulation time (s)",
        ylabel="Cumulative illuminated images",
        xlim=(0, 45_000),
    )
    axis.set_xticks((0, 15_000, 30_000, 45_000))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    save_figure(fig, root, "cumulative_illuminated_images_median_iqr")


def plot_differences(differences: pd.DataFrame, root: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.axhline(0.0, color="black", linewidth=0.8)
    for method in METHODS:
        if method == REFERENCE_METHOD:
            continue
        data = differences[differences["method"] == method]
        x = data["sim_time_s"].to_numpy(dtype=float)
        axis.plot(
            x,
            data["mean_paired_difference"],
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        axis.fill_between(
            x,
            data["bootstrap_95_ci_low"],
            data["bootstrap_95_ci_high"],
            color=COLORS[method],
            alpha=0.14,
            linewidth=0.0,
        )
    axis.set(
        xlabel="Simulation time (s)",
        ylabel="Paired difference from smallest-angle",
        xlim=(0, 45_000),
    )
    axis.set_xticks((0, 15_000, 30_000, 45_000))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    save_figure(fig, root, "paired_difference_from_smallest_angle_over_time")


def main() -> int:
    args = parse_args()
    if args.grid_seconds <= 0.0 or not np.isclose(
        EPISODE_DURATION_S % args.grid_seconds, 0.0
    ):
        raise SystemExit("--grid-seconds must divide 45,000 exactly")
    root = args.input_root.resolve()
    decision_epochs = load_and_validate(root)
    resampled = resample_all(decision_epochs, args.grid_seconds)
    summary, differences, _ = curve_tables(resampled, args.bootstrap_draws)
    episode_metrics = episode_timing_metrics(resampled)
    paired = paired_timing_statistics(episode_metrics)
    output = root / "analysis" / "timeline"
    output.mkdir(parents=True, exist_ok=True)
    decision_epochs.to_csv(output / "decision_epoch_timelines.csv", index=False)
    resampled.to_csv(output / "timelines_resampled_300s.csv", index=False)
    summary.to_csv(output / "curve_summary_300s.csv", index=False)
    differences.to_csv(output / "paired_curve_differences_300s.csv", index=False)
    episode_metrics.to_csv(output / "episode_timing_metrics.csv", index=False)
    paired.to_csv(output / "paired_timing_statistics.csv", index=False)
    plot_mean_curves(summary, root)
    plot_median_curves(summary, root)
    plot_differences(differences, root)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methods": list(METHODS),
        "seeds": {"start": 0, "stop_inclusive": 99},
        "episode_duration_s": EPISODE_DURATION_S,
        "recording": "every decision epoch",
        "resampling": f"forward-filled {args.grid_seconds:g}-second grid",
        "curve_interval": f"95% seed bootstrap, {args.bootstrap_draws} draws",
        "reference_method": REFERENCE_METHOD,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
