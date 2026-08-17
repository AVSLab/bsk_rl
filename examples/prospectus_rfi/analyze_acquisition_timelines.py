#!/usr/bin/env python3
"""Paired saturation-aware analysis of AMOS 2025 acquisition timelines."""

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
    METHODS,
    PLOT_GRID_INTERVAL_S,
    TABLE_CHECKPOINTS_S,
    load_timeline_files,
    resample_step_trajectory,
)

METHOD_LABELS = {
    "heuristic_historical": "Closest-angle heuristic",
    "legacy_amos2025_alpha0_policy": "Frozen AMOS 2025 alpha=0 policy",
}
COLORS = {
    "heuristic_historical": "#d55e00",
    "legacy_amos2025_alpha0_policy": "#365f9d",
}
CATALOG_SIZES = (100, 200, 400)
SEEDS = tuple(range(100))
COUNT_COLUMN = "cumulative_illuminated_observations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grid-seconds", type=float, default=PLOT_GRID_INTERVAL_S)
    parser.add_argument("--curve-bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--table-bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def validate_method_timelines(frame: pd.DataFrame, method: str) -> None:
    required = {
        "sim_time_s",
        "method",
        "catalog_size",
        "scenario_seed",
        "scenario_fingerprint",
        "cumulative_successful_observations",
        COUNT_COLUMN,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{method} timelines lack columns: {missing}")
    expected = {
        (catalog_size, seed) for catalog_size in CATALOG_SIZES for seed in SEEDS
    }
    actual = {
        (int(catalog_size), int(seed))
        for catalog_size, seed in frame[["catalog_size", "scenario_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if actual != expected:
        missing_pairs = sorted(expected - actual)
        extra_pairs = sorted(actual - expected)
        raise ValueError(
            f"{method} timeline design mismatch: missing={missing_pairs[:10]} "
            f"(n={len(missing_pairs)}), unexpected={extra_pairs[:10]} "
            f"(n={len(extra_pairs)})"
        )
    if set(frame["method"].astype(str)) != {method}:
        raise ValueError(f"{method} timeline files contain another method")
    for key, episode in frame.groupby(["catalog_size", "scenario_seed"], sort=False):
        ordered = episode.sort_values("sim_time_s")
        times = ordered["sim_time_s"].to_numpy(dtype=float)
        counts = ordered[COUNT_COLUMN].to_numpy(dtype=float)
        if not np.isclose(times[0], 0.0) or not np.isclose(
            times[-1], EPISODE_DURATION_S
        ):
            raise ValueError(f"timeline {method}, N/seed={key} lacks full time support")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(f"timeline {method}, N/seed={key} has duplicate times")
        if np.any(np.diff(counts) < 0.0):
            raise ValueError(f"timeline {method}, N/seed={key} decreases")
        if ordered["scenario_fingerprint"].nunique() != 1:
            raise ValueError(f"timeline {method}, N/seed={key} changes fingerprint")


def resample_all(frame: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, episode in frame.groupby(
        ["method", "catalog_size", "scenario_seed"], sort=True
    ):
        pieces.append(
            resample_step_trajectory(
                episode,
                interval_s=interval_s,
                duration_s=EPISODE_DURATION_S,
            )
        )
    return pd.concat(pieces, ignore_index=True)


def validate_pairing(frame: pd.DataFrame) -> None:
    identity = frame[
        ["method", "catalog_size", "scenario_seed", "scenario_fingerprint"]
    ].drop_duplicates()
    for key, group in identity.groupby(["catalog_size", "scenario_seed"]):
        if set(group["method"]) != set(METHODS):
            raise ValueError(f"methods are not paired at N/seed={key}")
        if group["scenario_fingerprint"].nunique() != 1:
            raise ValueError(f"scenario fingerprint mismatch at N/seed={key}")


def bootstrap_mean_curve(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile interval for a seed-resampled mean curve."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("curve bootstrap needs at least two seed traces")
    if draws < 100:
        raise ValueError("use at least 100 bootstrap draws")
    boot = np.empty((draws, values.shape[1]), dtype=np.float32)
    chunk = 100
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, values.shape[0], size=(stop - start, values.shape[0]))
        boot[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return low.astype(float), high.astype(float)


def curve_tables(
    frame: pd.DataFrame, draws: int, value_column: str = COUNT_COLUMN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(202_508_16)
    summaries: list[pd.DataFrame] = []
    differences: list[pd.DataFrame] = []
    for catalog_size in CATALOG_SIZES:
        matrices: dict[str, np.ndarray] = {}
        grid = None
        for method in METHODS:
            subset = frame[
                (frame["catalog_size"] == catalog_size) & (frame["method"] == method)
            ].sort_values(["scenario_seed", "sim_time_s"])
            pivot = subset.pivot(
                index="scenario_seed", columns="sim_time_s", values=value_column
            ).sort_index()
            if tuple(pivot.index.astype(int)) != SEEDS:
                raise ValueError(
                    f"seed order is incomplete for {method}, N={catalog_size}"
                )
            matrix = pivot.to_numpy(dtype=float)
            matrices[method] = matrix
            grid = pivot.columns.to_numpy(dtype=float)
            low, high = bootstrap_mean_curve(matrix, rng, draws)
            summaries.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "catalog_size": catalog_size,
                        "sim_time_s": grid,
                        "seed_count": matrix.shape[0],
                        "mean": np.mean(matrix, axis=0),
                        "std": np.std(matrix, axis=0, ddof=1),
                        "median": np.median(matrix, axis=0),
                        "q25": np.quantile(matrix, 0.25, axis=0),
                        "q75": np.quantile(matrix, 0.75, axis=0),
                        "bootstrap_95_ci_low": low,
                        "bootstrap_95_ci_high": high,
                    }
                )
            )
        difference = (
            matrices["legacy_amos2025_alpha0_policy"] - matrices["heuristic_historical"]
        )
        low, high = bootstrap_mean_curve(difference, rng, draws)
        differences.append(
            pd.DataFrame(
                {
                    "catalog_size": catalog_size,
                    "sim_time_s": grid,
                    "paired_seed_count": difference.shape[0],
                    "mean_policy_minus_heuristic": np.mean(difference, axis=0),
                    "median_policy_minus_heuristic": np.median(difference, axis=0),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                }
            )
        )
    return pd.concat(summaries, ignore_index=True), pd.concat(
        differences, ignore_index=True
    )


def time_to_fraction_of_final(
    times: np.ndarray, counts: np.ndarray, fraction: float
) -> float:
    final = float(counts[-1])
    if final <= 0.0:
        return float("nan")
    reached = np.flatnonzero(counts >= fraction * final)
    return float(times[reached[0]]) if reached.size else float("nan")


def episode_saturation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (method, catalog_size, seed), episode in frame.groupby(
        ["method", "catalog_size", "scenario_seed"], sort=True
    ):
        ordered = episode.sort_values("sim_time_s")
        times = ordered["sim_time_s"].to_numpy(dtype=float)
        counts = ordered[COUNT_COLUMN].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "method": str(method),
            "catalog_size": int(catalog_size),
            "scenario_seed": int(seed),
            "scenario_fingerprint": str(ordered.iloc[0]["scenario_fingerprint"]),
            "final_illuminated_observations": float(counts[-1]),
            "normalized_illuminated_auc": float(
                np.trapz(counts / float(catalog_size), times) / EPISODE_DURATION_S
            ),
            "time_to_50pct_final_s": time_to_fraction_of_final(times, counts, 0.50),
            "time_to_80pct_final_s": time_to_fraction_of_final(times, counts, 0.80),
            "time_to_90pct_final_s": time_to_fraction_of_final(times, counts, 0.90),
            "time_to_95pct_final_s": time_to_fraction_of_final(times, counts, 0.95),
        }
        for checkpoint in TABLE_CHECKPOINTS_S:
            index = int(np.searchsorted(times, checkpoint, side="right") - 1)
            value = float(counts[index])
            label = int(checkpoint)
            row[f"illuminated_observations_at_{label}s"] = value
            row[f"catalog_fraction_at_{label}s"] = value / float(catalog_size)
            row[f"own_final_plateau_fraction_at_{label}s"] = (
                value / float(counts[-1]) if counts[-1] > 0.0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_metric_table(episodes: pd.DataFrame, draws: int) -> pd.DataFrame:
    metrics = [
        "illuminated_observations_at_15000s",
        "illuminated_observations_at_30000s",
        "illuminated_observations_at_45000s",
        "own_final_plateau_fraction_at_15000s",
        "own_final_plateau_fraction_at_30000s",
        "normalized_illuminated_auc",
        "time_to_90pct_final_s",
        "final_illuminated_observations",
    ]
    rng = np.random.default_rng(202_508_17)
    rows: list[dict[str, float | int | str]] = []
    for catalog_size in CATALOG_SIZES:
        subset = episodes[episodes["catalog_size"] == catalog_size]
        heuristic = subset[subset["method"] == "heuristic_historical"]
        policy = subset[subset["method"] == "legacy_amos2025_alpha0_policy"]
        for metric in metrics:
            paired = (
                policy[["scenario_seed", metric]]
                .merge(
                    heuristic[["scenario_seed", metric]],
                    on="scenario_seed",
                    suffixes=("_policy", "_heuristic"),
                    validate="one_to_one",
                )
                .dropna()
            )
            difference = (
                paired[f"{metric}_policy"] - paired[f"{metric}_heuristic"]
            ).to_numpy(dtype=float)
            low, high = paired_bootstrap_ci(difference, rng, draws)
            try:
                p_value = float(wilcoxon(difference).pvalue)
            except ValueError:
                p_value = 1.0
            rows.append(
                {
                    "catalog_size": catalog_size,
                    "metric": metric,
                    "difference_definition": "frozen_policy_minus_closest_angle_heuristic",
                    "paired_seed_count": len(difference),
                    "policy_mean": float(paired[f"{metric}_policy"].mean()),
                    "heuristic_mean": float(paired[f"{metric}_heuristic"].mean()),
                    "mean_paired_difference": float(np.mean(difference)),
                    "median_paired_difference": float(np.median(difference)),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "wilcoxon_p_raw": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p_raw"].tolist())
    return result


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(figure_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def decorate_axes(axis: plt.Axes, catalog_size: int, ylabel: str | None = None) -> None:
    axis.set_title(f"Catalog size N = {catalog_size}")
    axis.set_xlabel("Simulation time (thousands of seconds)")
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.set_xlim(0.0, EPISODE_DURATION_S / 1000.0)
    axis.grid(alpha=0.2)
    for checkpoint in TABLE_CHECKPOINTS_S:
        axis.axvline(checkpoint / 1000.0, color="#777777", linewidth=0.7, alpha=0.35)


def plot_absolute_curves(summary: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharex=True)
    for axis, catalog_size in zip(axes, CATALOG_SIZES, strict=True):
        for method in METHODS:
            data = summary[
                (summary["catalog_size"] == catalog_size)
                & (summary["method"] == method)
            ]
            x = data["sim_time_s"].to_numpy() / 1000.0
            axis.plot(
                x,
                data["mean"],
                color=COLORS[method],
                linewidth=2.0,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                x,
                data["bootstrap_95_ci_low"],
                data["bootstrap_95_ci_high"],
                color=COLORS[method],
                alpha=0.18,
                linewidth=0.0,
            )
        decorate_axes(
            axis,
            catalog_size,
            "Cumulative illuminated targets" if axis is axes[0] else None,
        )
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Illuminated target acquisition over the 45,000-second task",
        x=0.04,
        y=1.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.95,
        "Unsmoothed 100-second forward-filled grid; lines are 100-seed means and bands are 95% seed-bootstrap intervals.",
        ha="left",
        fontsize=8,
        color="#4d4d4d",
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.22)
    save_figure(fig, output_root, "cumulative_illuminated_observations_over_time")


def plot_plateau_fraction_curves(summary: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharex=True, sharey=True)
    for axis, catalog_size in zip(axes, CATALOG_SIZES, strict=True):
        for method in METHODS:
            data = summary[
                (summary["catalog_size"] == catalog_size)
                & (summary["method"] == method)
            ]
            x = data["sim_time_s"].to_numpy() / 1000.0
            axis.plot(
                x,
                data["mean"],
                color=COLORS[method],
                linewidth=2.0,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                x,
                data["bootstrap_95_ci_low"],
                data["bootstrap_95_ci_high"],
                color=COLORS[method],
                alpha=0.18,
                linewidth=0.0,
            )
        decorate_axes(
            axis,
            catalog_size,
            "Fraction of own final illuminated count" if axis is axes[0] else None,
        )
        axis.set_ylim(0.0, 1.04)
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Time to each method's empirical acquisition plateau",
        x=0.04,
        y=1.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.95,
        "Normalization exposes acquisition speed when final counts are similar; it does not label unobserved targets as unreachable.",
        ha="left",
        fontsize=8,
        color="#4d4d4d",
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.22)
    save_figure(fig, output_root, "fraction_of_final_plateau_over_time")


def plot_paired_difference(difference: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharex=True)
    for axis, catalog_size in zip(axes, CATALOG_SIZES, strict=True):
        data = difference[difference["catalog_size"] == catalog_size]
        x = data["sim_time_s"].to_numpy() / 1000.0
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(
            x, data["mean_policy_minus_heuristic"], color="#365f9d", linewidth=2.0
        )
        axis.fill_between(
            x,
            data["bootstrap_95_ci_low"],
            data["bootstrap_95_ci_high"],
            color="#365f9d",
            alpha=0.18,
            linewidth=0.0,
        )
        decorate_axes(
            axis,
            catalog_size,
            "Policy − heuristic illuminated targets" if axis is axes[0] else None,
        )
    fig.suptitle(
        "Paired acquisition advantage over time",
        x=0.04,
        y=1.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.95,
        "Each trace differences identical scenario seeds before aggregation; positive values favor the frozen policy.",
        ha="left",
        fontsize=8,
        color="#4d4d4d",
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.22)
    save_figure(fig, output_root, "paired_policy_minus_heuristic_over_time")


def main() -> int:
    args = parse_args()
    if args.grid_seconds <= 0.0 or not np.isclose(
        EPISODE_DURATION_S % args.grid_seconds, 0.0
    ):
        raise SystemExit("--grid-seconds must divide 45,000 exactly")
    roots = {
        "heuristic_historical": args.heuristic_root.resolve(),
        "legacy_amos2025_alpha0_policy": args.policy_root.resolve(),
    }
    raw = []
    for method in METHODS:
        frame = load_timeline_files(roots[method], method)
        validate_method_timelines(frame, method)
        raw.append(frame)
    decision_epochs = pd.concat(raw, ignore_index=True)
    validate_pairing(decision_epochs)
    resampled = resample_all(decision_epochs, args.grid_seconds)
    validate_pairing(resampled)
    final_counts = resampled.groupby(["method", "catalog_size", "scenario_seed"])[
        COUNT_COLUMN
    ].transform("max")
    resampled["fraction_of_own_final_illuminated_count"] = np.where(
        final_counts > 0.0,
        resampled[COUNT_COLUMN] / final_counts,
        np.nan,
    )

    output_root = args.output_root.resolve()
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    grid_label = f"{args.grid_seconds:g}s".replace(".", "p")
    resampled.to_csv(
        analysis_dir / f"timelines_resampled_{grid_label}.csv", index=False
    )
    try:
        resampled.to_parquet(
            analysis_dir / f"timelines_resampled_{grid_label}.parquet", index=False
        )
    except (ImportError, ModuleNotFoundError):
        pass

    curve_summary, paired_curve = curve_tables(resampled, args.curve_bootstrap_draws)
    plateau_summary, paired_plateau_curve = curve_tables(
        resampled,
        args.curve_bootstrap_draws,
        value_column="fraction_of_own_final_illuminated_count",
    )
    episode_metrics = episode_saturation_metrics(resampled)
    paired_metrics = paired_metric_table(episode_metrics, args.table_bootstrap_draws)
    curve_summary.to_csv(analysis_dir / f"curve_summary_{grid_label}.csv", index=False)
    paired_curve.to_csv(
        analysis_dir / f"paired_curve_difference_{grid_label}.csv", index=False
    )
    plateau_summary.to_csv(
        analysis_dir / f"plateau_fraction_curve_summary_{grid_label}.csv",
        index=False,
    )
    paired_plateau_curve.to_csv(
        analysis_dir / f"paired_plateau_fraction_difference_{grid_label}.csv",
        index=False,
    )
    episode_metrics.to_csv(analysis_dir / "episode_saturation_metrics.csv", index=False)
    paired_metrics.to_csv(
        analysis_dir / "paired_saturation_statistics.csv", index=False
    )
    plot_absolute_curves(curve_summary, output_root)
    plot_plateau_fraction_curves(plateau_summary, output_root)
    plot_paired_difference(paired_curve, output_root)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "heuristic_root": str(roots["heuristic_historical"]),
        "legacy_policy_root": str(roots["legacy_amos2025_alpha0_policy"]),
        "methods": list(METHODS),
        "catalog_sizes": list(CATALOG_SIZES),
        "scenario_seeds": {"start": 0, "stop_inclusive": 99},
        "episode_duration_s": EPISODE_DURATION_S,
        "recording": "every decision epoch",
        "resampling": (
            f"forward-fill onto {args.grid_seconds:g}-second grid; no interpolation "
            "and no smoothing"
        ),
        "table_checkpoints_s": list(TABLE_CHECKPOINTS_S),
        "curve_uncertainty": (
            f"pointwise 95% percentile bootstrap of the mean across paired seeds; "
            f"{args.curve_bootstrap_draws} draws"
        ),
        "table_uncertainty": (
            f"paired 95% percentile bootstrap of mean method differences; "
            f"{args.table_bootstrap_draws} draws"
        ),
        "hypothesis_policy_reaches_plateau_earlier": (
            "tested, not assumed; use absolute curves, paired difference curves, "
            "normalized acquisition AUC, and time to 90% of each run's final plateau"
        ),
        "limitations": (
            "Final-plateau normalization is an empirical saturation diagnostic, not "
            "a claim that every unobserved catalog target was geometrically impossible."
        ),
    }
    (analysis_dir / "timeline_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
