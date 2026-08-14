#!/usr/bin/env python3
"""Paired statistics, publication figures, and prospectus-ready results text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from examples.prospectus_rfi.config import load_study_config


METHOD_LABELS = {
    "mlp": "Fixed-input monolithic MLP",
    "attention": "Target-set attention",
    "heuristic_historical": "Historical heuristic (full catalog)",
    "heuristic_matched": "Heuristic (information matched)",
}
COLORS = {
    "mlp": "#365f9d",
    "attention": "#d55e00",
    "heuristic_historical": "#6a6a6a",
    "heuristic_matched": "#009e73",
}
PRIMARY_METRICS = [
    "successful_observation_fraction",
    "illuminated_observation_fraction",
    "useful_deliveries",
    "onboard_backlog_fraction",
    "resource_constraint_interventions",
    "mean_inference_ms",
]


def save_figure(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def read_evaluation(input_dir: Path) -> pd.DataFrame:
    files = sorted((input_dir / "evaluation" / "raw").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no evaluation CSV files under {input_dir}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    duplicate_keys = ["method", "candidate_count", "catalog_size", "scenario_seed"]
    frame = frame.drop_duplicates(duplicate_keys, keep="last")
    return frame


def validate_pairing(frame: pd.DataFrame) -> None:
    expected = set(frame["method"].unique())
    grouped = frame.groupby(["candidate_count", "catalog_size", "scenario_seed"])
    for key, group in grouped:
        if set(group["method"]) != expected:
            raise ValueError(f"unpaired methods for K,N,seed={key}")
        if group["scenario_fingerprint"].nunique() != 1:
            raise ValueError(f"scenario fingerprint mismatch for K,N,seed={key}")


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for metric in PRIMARY_METRICS:
        if metric not in frame:
            continue
        grouped = frame.groupby(["method", "candidate_count", "catalog_size"])[metric]
        summary = grouped.agg(["count", "mean", "std", "median"]).reset_index()
        quantiles = grouped.quantile([0.25, 0.75]).unstack().reset_index()
        quantiles.columns = ["method", "candidate_count", "catalog_size", "q25", "q75"]
        summary = summary.merge(quantiles)
        summary["metric"] = metric
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def paired_bootstrap_ci(values: np.ndarray, rng, draws: int = 10_000):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_table(frame: pd.DataFrame, margin: float) -> pd.DataFrame:
    reference = "heuristic_matched"
    rng = np.random.default_rng(202_508)
    rows = []
    for metric in PRIMARY_METRICS:
        if metric not in frame:
            continue
        for (candidate_count, catalog_size), group in frame.groupby(
            ["candidate_count", "catalog_size"]
        ):
            ref = group[group["method"] == reference][["scenario_seed", metric]].rename(
                columns={metric: "reference"}
            )
            for method in ("mlp", "attention", "heuristic_historical"):
                policy = group[group["method"] == method][
                    ["scenario_seed", metric]
                ].rename(columns={metric: "method_value"})
                paired = policy.merge(ref, on="scenario_seed", validate="one_to_one")
                difference = (paired["method_value"] - paired["reference"]).to_numpy()
                ci_low, ci_high = paired_bootstrap_ci(difference, rng)
                try:
                    p_value = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_value = 1.0
                equivalence = "not_applicable"
                if metric == "successful_observation_fraction":
                    if ci_low >= -margin and ci_high <= margin:
                        equivalence = "practically_equivalent"
                    elif ci_low > margin:
                        equivalence = "meaningfully_higher"
                    elif ci_high < -margin:
                        equivalence = "meaningfully_lower"
                    else:
                        equivalence = "inconclusive"
                rows.append(
                    {
                        "method": method,
                        "reference": reference,
                        "candidate_count": candidate_count,
                        "catalog_size": catalog_size,
                        "metric": metric,
                        "paired_count": len(difference),
                        "mean_paired_difference": float(np.mean(difference)),
                        "median_paired_difference": float(np.median(difference)),
                        "bootstrap_95_ci_low": float(ci_low),
                        "bootstrap_95_ci_high": float(ci_high),
                        "wilcoxon_p_raw": p_value,
                        "practical_equivalence_margin": (
                            margin
                            if metric == "successful_observation_fraction"
                            else np.nan
                        ),
                        "equivalence_classification": equivalence,
                    }
                )
    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p_raw"].tolist())
    return result


def training_validation_data(input_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((input_dir / "training").glob("*_k*_seed*")):
        validation_path = run_dir / "validation_metrics.csv"
        training_path = run_dir / "training_metrics.csv"
        metadata_path = run_dir / "metadata.json"
        if not validation_path.exists() or not training_path.exists():
            continue
        validation = pd.read_csv(validation_path)
        validation = validation.groupby("checkpoint", as_index=False).agg(
            validation_score=("physical_validation_score", "mean"),
            validation_q25=("physical_validation_score", lambda x: x.quantile(0.25)),
            validation_q75=("physical_validation_score", lambda x: x.quantile(0.75)),
        )
        training = pd.read_csv(training_path)
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        for _, item in validation.iterrows():
            name = str(item["checkpoint"])
            if name.startswith("iteration_"):
                iteration = int(name.split("_")[-1])
                match = training[training["training_iteration"] == iteration]
            else:
                match = training.tail(1)
            if match.empty:
                continue
            row = match.iloc[-1]
            rows.append(
                {
                    **item.to_dict(),
                    "architecture": metadata["study_config"]["architecture"]["name"],
                    "candidate_count": metadata["candidate_count"],
                    "seed": metadata["seed"],
                    "environment_steps": row["environment_steps"],
                    "wall_clock_h": row["wall_clock_h"],
                }
            )
    return pd.DataFrame(rows)


def training_speed_summary(training: pd.DataFrame, thresholds) -> pd.DataFrame:
    if training.empty:
        return pd.DataFrame()
    rows = []
    keys = ["architecture", "candidate_count", "seed"]
    for key, group in training.groupby(keys):
        group = group.sort_values("environment_steps")
        x = group["environment_steps"].to_numpy(dtype=float)
        y = group["validation_score"].to_numpy(dtype=float)
        record = dict(zip(keys, key))
        record.update(
            {
                "final_validation_score": float(y[-1]),
                "best_validation_score": float(np.max(y)),
                "final_environment_steps": float(x[-1]),
                "final_wall_clock_h": float(group["wall_clock_h"].iloc[-1]),
                "step_normalized_validation_auc": (
                    float(np.trapz(y, x) / (x[-1] - x[0]))
                    if len(x) > 1 and x[-1] > x[0]
                    else float(y[-1])
                ),
            }
        )
        for threshold in thresholds:
            reached = group[group["validation_score"] >= threshold]
            tag = str(threshold).replace(".", "p")
            record[f"steps_to_score_{tag}"] = (
                float(reached["environment_steps"].iloc[0])
                if not reached.empty
                else np.nan
            )
            record[f"hours_to_score_{tag}"] = (
                float(reached["wall_clock_h"].iloc[0]) if not reached.empty else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def plot_training_curves(training: pd.DataFrame, figure_dir: Path) -> None:
    if training.empty:
        return
    for x, stem, xlabel in (
        ("environment_steps", "training_performance_vs_steps", "Environment steps"),
        ("wall_clock_h", "training_performance_vs_wall_clock", "Wall-clock hours"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for axis, candidate_count in zip(axes, (5, 10, 20)):
            subset = training[training["candidate_count"] == candidate_count]
            for architecture, group in subset.groupby("architecture"):
                label_key = (
                    "mlp"
                    if architecture == "fixed_input_monolithic_mlp"
                    else "attention"
                )
                trajectories = []
                for _, seed_group in group.groupby("seed"):
                    seed_group = seed_group.sort_values(x)
                    axis.plot(
                        seed_group[x],
                        seed_group["validation_score"],
                        color=COLORS[label_key],
                        alpha=0.35,
                        linewidth=0.8,
                    )
                    trajectories.append(seed_group)
                # Interpolate replications onto their shared support, then show
                # the unsmoothed pointwise median and IQR.
                lower_support = max(item[x].min() for item in trajectories)
                upper_support = min(item[x].max() for item in trajectories)
                grid = np.array(
                    sorted(
                        set(
                            value
                            for item in trajectories
                            for value in item[x]
                            if lower_support <= value <= upper_support
                        )
                    )
                )
                values = np.vstack(
                    [
                        np.interp(grid, item[x], item["validation_score"])
                        for item in trajectories
                    ]
                )
                median = np.median(values, axis=0)
                q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
                axis.plot(
                    grid,
                    median,
                    color=COLORS[label_key],
                    linewidth=2.0,
                    label=METHOD_LABELS[label_key],
                )
                if len(trajectories) > 1:
                    axis.fill_between(
                        grid,
                        q25,
                        q75,
                        color=COLORS[label_key],
                        alpha=0.18,
                    )
            axis.set_title(f"K = {candidate_count}")
            axis.set_xlabel(xlabel)
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Held-out physical validation score")
        axes[-1].legend(frameon=False, fontsize=8)
        save_figure(fig, figure_dir, stem)


def plot_performance(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for axis, candidate_count in zip(axes, (5, 10, 20)):
        subset = frame[frame["candidate_count"] == candidate_count]
        for method, group in subset.groupby("method"):
            statistics = group.groupby("catalog_size")[
                "successful_observation_fraction"
            ].agg(["mean", "sem"])
            axis.errorbar(
                statistics.index,
                statistics["mean"],
                yerr=1.96 * statistics["sem"],
                marker="o",
                capsize=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"K = {candidate_count}")
        axis.set_xlabel("Catalog size N")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Successfully observed fraction")
    axes[-1].legend(frameon=False, fontsize=7)
    save_figure(fig, figure_dir, "performance_vs_catalog_size")


def plot_differences(paired: pd.DataFrame, figure_dir: Path) -> None:
    data = paired[paired["metric"] == "successful_observation_fraction"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for axis, candidate_count in zip(axes, (5, 10, 20)):
        subset = data[data["candidate_count"] == candidate_count]
        offsets = {"mlp": -7, "attention": 0, "heuristic_historical": 7}
        for method, group in subset.groupby("method"):
            x = group["catalog_size"] + offsets[method]
            y = group["mean_paired_difference"]
            yerr = np.vstack(
                [y - group["bootstrap_95_ci_low"], group["bootstrap_95_ci_high"] - y]
            )
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                capsize=3,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhspan(-0.02, 0.02, color="#999999", alpha=0.15)
        axis.set_title(f"K = {candidate_count}")
        axis.set_xlabel("Catalog size N")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Method − matched heuristic\n(successful fraction)")
    axes[-1].legend(frameon=False, fontsize=7)
    save_figure(fig, figure_dir, "paired_policy_minus_heuristic")


def plot_resources(frame: pd.DataFrame, figure_dir: Path) -> None:
    columns = [
        "image_action_count",
        "charge_action_count",
        "downlink_action_count",
        "desaturation_action_count",
        "resource_constraint_interventions",
    ]
    means = frame.groupby(["method", "candidate_count"])[columns].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, candidate_count in zip(axes, (5, 10, 20)):
        subset = means[means["candidate_count"] == candidate_count].set_index("method")
        bottom = np.zeros(len(subset))
        for column in columns:
            axis.bar(
                [METHOD_LABELS[item] for item in subset.index],
                subset[column],
                bottom=bottom,
                label=column.replace("_", " "),
            )
            bottom += subset[column].to_numpy()
        axis.set_title(f"K = {candidate_count}")
        axis.tick_params(axis="x", labelrotation=55, labelsize=7)
    axes[0].set_ylabel("Mean episode count")
    axes[-1].legend(frameon=False, fontsize=7)
    save_figure(fig, figure_dir, "resource_action_allocation_and_interventions")


def plot_computation(frame: pd.DataFrame, input_dir: Path, figure_dir: Path) -> None:
    records = []
    for metadata_path in sorted((input_dir / "training").glob("*/metadata.json")):
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        status_path = metadata_path.parent / "training_metrics.csv"
        throughput = np.nan
        if status_path.exists():
            throughput = pd.read_csv(status_path)["samples_per_second"].median()
        records.append(
            {
                "method": (
                    "mlp"
                    if metadata["study_config"]["architecture"]["name"]
                    == "fixed_input_monolithic_mlp"
                    else "attention"
                ),
                "candidate_count": metadata["candidate_count"],
                "parameters": metadata["model"]["trainable_parameters"],
                "samples_per_second": throughput,
            }
        )
    computation = pd.DataFrame(records)
    inference = (
        frame[frame["method"].isin(["mlp", "attention"])]
        .groupby(["method", "candidate_count"], as_index=False)["mean_inference_ms"]
        .mean()
    )
    computation = computation.merge(
        inference, on=["method", "candidate_count"], how="left"
    )
    computation.to_csv(input_dir / "analysis" / "computation_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for axis, metric, ylabel in zip(
        axes,
        ("parameters", "mean_inference_ms", "samples_per_second"),
        ("Trainable parameters", "Inference time (ms)", "Samples per second"),
    ):
        for method, group in computation.groupby("method"):
            axis.plot(
                group["candidate_count"],
                group[metric],
                marker="o",
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xlabel("Presented candidates K")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[-1].legend(frameon=False, fontsize=8)
    save_figure(fig, figure_dir, "parameters_inference_and_throughput")


def write_prospectus_results(
    output: Path, summary: pd.DataFrame, paired: pd.DataFrame, figure_dir: Path
) -> None:
    primary = paired[paired["metric"] == "successful_observation_fraction"]
    lines = [
        "# Research Focus I: architecture and baseline comparison",
        "",
        "> Status: exploratory candidate-list sweep. Each learned configuration currently has one training seed. "
        "The paired Monte Carlo intervals quantify scenario variability, not policy-training variability; no "
        "architecture-superiority claim is warranted until the planned three-seed confirmatory campaign.",
        "",
        "The practical-equivalence margin was predeclared as ±0.02 in successfully observed catalog fraction.",
        "",
        "## Paired differences from the information-matched heuristic",
        "",
        "| Method | K | N | Mean difference | Paired 95% bootstrap CI | Holm-adjusted p | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {int(row['candidate_count'])} | "
            f"{int(row['catalog_size'])} | {row['mean_paired_difference']:.4f} | "
            f"[{row['bootstrap_95_ci_low']:.4f}, {row['bootstrap_95_ci_high']:.4f}] | "
            f"{row['wilcoxon_p_holm']:.4g} | {row['equivalence_classification']} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            *[
                f"- `{path.relative_to(output.parent)}`"
                for path in sorted(figure_dir.glob("*.pdf"))
            ],
            "",
            "Full means, standard deviations, medians, and interquartile ranges are in "
            "`analysis/summary_statistics.csv`; paired tests are in `analysis/paired_differences.csv`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.resolve()
    analysis_dir = root / "analysis"
    figure_dir = root / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame = read_evaluation(root)
    validate_pairing(frame)
    frame.to_csv(analysis_dir / "evaluation_combined.csv", index=False)
    try:
        frame.to_parquet(analysis_dir / "evaluation_combined.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        pass

    config_root = Path(__file__).resolve().parent / "configs"
    study = load_study_config(
        config_root / "mlp_selected.yaml", config_root / "base.yaml"
    )
    summary = summary_table(frame)
    paired = paired_table(frame, study.validation.practical_equivalence_margin)
    summary.to_csv(analysis_dir / "summary_statistics.csv", index=False)
    paired.to_csv(analysis_dir / "paired_differences.csv", index=False)
    training = training_validation_data(root)
    training.to_csv(analysis_dir / "training_validation_curves.csv", index=False)
    training_speed_summary(training, study.validation.score_thresholds).to_csv(
        analysis_dir / "training_speed_summary.csv", index=False
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    plot_training_curves(training, figure_dir)
    plot_performance(frame, figure_dir)
    plot_differences(paired, figure_dir)
    plot_resources(frame, figure_dir)
    plot_computation(frame, root, figure_dir)
    write_prospectus_results(
        root / "prospectus_results.md", summary, paired, figure_dir
    )


if __name__ == "__main__":
    main()
