#!/usr/bin/env python3
"""Statistical analysis for the matched 300-second Research Focus I campaign."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel, wilcoxon

from examples.prospectus_rfi.amos2025_matched_300s_design import METHODS
from examples.prospectus_rfi.collect_amos2025_matched_300s import validate_campaign

REFERENCE_METHOD = "smallest_angle_heuristic"
METHOD_LABELS = {
    "breckenridge2026_alpha0_mlp": "Breckenridge alpha=0 MLP",
    "target_set_attention": "Target-set attention",
    "smallest_angle_heuristic": "Smallest-angle heuristic",
    "closest_distance_heuristic": "Closest-distance heuristic",
}
TABLE_METHODS = (
    "smallest_angle_heuristic",
    "closest_distance_heuristic",
    "breckenridge2026_alpha0_mlp",
    "target_set_attention",
)
COLORS = {
    "breckenridge2026_alpha0_mlp": "#365f9d",
    "target_set_attention": "#009e73",
    "smallest_angle_heuristic": "#d55e00",
    "closest_distance_heuristic": "#cc79a7",
}
SHORT_LABELS = {
    "smallest_angle_heuristic": "Angle",
    "closest_distance_heuristic": "Distance",
    "breckenridge2026_alpha0_mlp": "Breck. MLP",
    "target_set_attention": "Attention",
}
METRICS = (
    "successful_observations",
    "illuminated_observations",
    "illumination_quality_fraction",
    "illuminated_catalog_fraction",
    "useful_deliveries",
    "delivery_fraction",
    "useful_images_left_onboard",
    "resource_constraint_interventions",
    "constraint_intervention_rate",
    "final_battery_fraction",
    "survival_fraction",
    "total_action_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def holm_adjust(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(array))
    adjusted = np.full(len(array), np.nan, dtype=float)
    order = finite_indices[np.argsort(array[finite_indices])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * array[index])
        adjusted[index] = min(1.0, running)
    return pd.Series(adjusted, index=values.index)


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    successful = frame["successful_observations"].to_numpy(dtype=float)
    illuminated = frame["illuminated_observations"].to_numpy(dtype=float)
    delivered = frame["useful_deliveries"].to_numpy(dtype=float)
    catalog = frame["catalog_size"].to_numpy(dtype=float)
    frame["illumination_quality_fraction"] = np.divide(
        illuminated,
        successful,
        out=np.full_like(illuminated, np.nan),
        where=successful > 0.0,
    )
    frame["illuminated_catalog_fraction"] = illuminated / catalog
    frame["delivery_fraction"] = np.divide(
        delivered,
        successful,
        out=np.full_like(delivered, np.nan),
        where=successful > 0.0,
    )
    frame["useful_images_left_onboard"] = successful - delivered
    return frame


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        subset = frame[frame["method"] == method]
        for metric in METRICS:
            values = subset[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            sem = std / np.sqrt(n)
            half_width = float(t.ppf(0.975, n - 1) * sem)
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "mean_95_ci_low": mean - half_width,
                    "mean_95_ci_high": mean + half_width,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def paired_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["scenario_seed", "method"])
    rows: list[dict[str, float | int | str]] = []
    for metric in METRICS:
        reference = indexed.xs(REFERENCE_METHOD, level="method")[metric]
        for method in METHODS:
            if method == REFERENCE_METHOD:
                continue
            candidate = indexed.xs(method, level="method")[metric]
            paired = pd.concat(
                [candidate.rename("candidate"), reference.rename("reference")],
                axis=1,
                join="inner",
            ).dropna()
            difference = (paired["candidate"] - paired["reference"]).to_numpy(
                dtype=float
            )
            n = len(difference)
            mean = float(np.mean(difference))
            std = float(np.std(difference, ddof=1))
            half_width = float(t.ppf(0.975, n - 1) * std / np.sqrt(n))
            if np.allclose(difference, 0.0):
                paired_t_p = 1.0
                wilcoxon_p = 1.0
            else:
                paired_t_p = float(
                    ttest_rel(paired["candidate"], paired["reference"]).pvalue
                )
                try:
                    wilcoxon_p = float(wilcoxon(difference).pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "reference_method": REFERENCE_METHOD,
                    "reference_label": METHOD_LABELS[REFERENCE_METHOD],
                    "metric": metric,
                    "paired_seed_count": n,
                    "method_mean": float(paired["candidate"].mean()),
                    "reference_mean": float(paired["reference"].mean()),
                    "mean_paired_difference": mean,
                    "paired_difference_std": std,
                    "paired_difference_95_ci_low": mean - half_width,
                    "paired_difference_95_ci_high": mean + half_width,
                    "paired_effect_size_dz": mean / std if std > 0.0 else np.nan,
                    "paired_t_p_raw": paired_t_p,
                    "wilcoxon_p_raw": wilcoxon_p,
                }
            )
    result = pd.DataFrame(rows)
    result["paired_t_p_holm"] = result.groupby("metric", group_keys=False)[
        "paired_t_p_raw"
    ].apply(holm_adjust)
    result["wilcoxon_p_holm"] = result.groupby("metric", group_keys=False)[
        "wilcoxon_p_raw"
    ].apply(holm_adjust)
    return result


def all_pairwise_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["scenario_seed", "method"])
    rows: list[dict[str, float | int | str]] = []
    for metric in METRICS:
        for method_a, method_b in combinations(TABLE_METHODS, 2):
            values_a = indexed.xs(method_a, level="method")[metric]
            values_b = indexed.xs(method_b, level="method")[metric]
            paired = pd.concat(
                [values_a.rename("a"), values_b.rename("b")], axis=1, join="inner"
            ).dropna()
            difference = (paired["a"] - paired["b"]).to_numpy(dtype=float)
            n = len(difference)
            mean = float(np.mean(difference))
            std = float(np.std(difference, ddof=1))
            half_width = float(t.ppf(0.975, n - 1) * std / np.sqrt(n))
            if np.allclose(difference, 0.0):
                paired_t_p = 1.0
                wilcoxon_p = 1.0
            else:
                paired_t_p = float(ttest_rel(paired["a"], paired["b"]).pvalue)
                try:
                    wilcoxon_p = float(wilcoxon(difference).pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
            rows.append(
                {
                    "metric": metric,
                    "method_a": method_a,
                    "method_a_label": METHOD_LABELS[method_a],
                    "method_b": method_b,
                    "method_b_label": METHOD_LABELS[method_b],
                    "difference_definition": "method_a_minus_method_b",
                    "paired_seed_count": n,
                    "method_a_mean": float(paired["a"].mean()),
                    "method_b_mean": float(paired["b"].mean()),
                    "mean_paired_difference": mean,
                    "paired_difference_std": std,
                    "paired_difference_95_ci_low": mean - half_width,
                    "paired_difference_95_ci_high": mean + half_width,
                    "paired_effect_size_dz": mean / std if std > 0.0 else np.nan,
                    "paired_t_p_raw": paired_t_p,
                    "wilcoxon_p_raw": wilcoxon_p,
                }
            )
    result = pd.DataFrame(rows)
    result["paired_t_p_holm"] = result.groupby("metric", group_keys=False)[
        "paired_t_p_raw"
    ].apply(holm_adjust)
    result["wilcoxon_p_holm"] = result.groupby("metric", group_keys=False)[
        "wilcoxon_p_raw"
    ].apply(holm_adjust)
    return result


def prospectus_table(descriptive: pd.DataFrame) -> str:
    rows = (
        ("Illuminated images", "illuminated_observations", 1.0),
        ("Illumination quality [\\%]", "illumination_quality_fraction", 100.0),
        ("Illuminated catalog coverage [\\%]", "illuminated_catalog_fraction", 100.0),
        ("Useful downlinks", "useful_deliveries", 1.0),
        ("Delivery fraction [\\%]", "delivery_fraction", 100.0),
        ("Useful images left onboard", "useful_images_left_onboard", 1.0),
        ("Shield interventions per run", "resource_constraint_interventions", 1.0),
    )
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Angle} & \\textbf{Distance} & "
        "\\textbf{Breckenridge MLP} & \\textbf{Target attention} \\\\",
        "\\midrule",
    ]
    for label, metric, scale in rows:
        values = []
        for method in TABLE_METHODS:
            record = descriptive[
                (descriptive["method"] == method) & (descriptive["metric"] == metric)
            ].iloc[0]
            values.append(
                f"${record['mean'] * scale:.2f}\\pm{record['std'] * scale:.2f}$"
            )
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_endpoint_summary(descriptive: pd.DataFrame, output: Path) -> None:
    panels = (
        ("illuminated_observations", "Illuminated images", 1.0),
        ("useful_deliveries", "Useful deliveries", 1.0),
        ("delivery_fraction", "Delivery fraction (%)", 100.0),
        (
            "resource_constraint_interventions",
            "Shield interventions per run",
            1.0,
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    x = np.arange(len(TABLE_METHODS))
    for axis, (metric, ylabel, scale) in zip(axes.flat, panels):
        subset = descriptive[descriptive["metric"] == metric].set_index("method")
        means = np.array([subset.loc[method, "mean"] for method in TABLE_METHODS])
        lows = np.array(
            [subset.loc[method, "mean_95_ci_low"] for method in TABLE_METHODS]
        )
        highs = np.array(
            [subset.loc[method, "mean_95_ci_high"] for method in TABLE_METHODS]
        )
        means, lows, highs = means * scale, lows * scale, highs * scale
        for position, method, mean, low, high in zip(
            x, TABLE_METHODS, means, lows, highs
        ):
            axis.errorbar(
                position,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="o",
                markersize=6,
                capsize=3,
                color=COLORS[method],
            )
        axis.set_xticks(x, [SHORT_LABELS[method] for method in TABLE_METHODS])
        axis.tick_params(axis="x", labelrotation=20)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Matched AMOS 2025 evaluation (100 paired seeds; mean and 95% CI)")
    save_figure(fig, output, "matched_300s_endpoint_summary")


def plot_operational_tradeoff(descriptive: pd.DataFrame, output: Path) -> None:
    indexed = descriptive.set_index(["method", "metric"])
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    annotation_offsets = {
        "smallest_angle_heuristic": (5, -14),
        "closest_distance_heuristic": (5, -14),
        "breckenridge2026_alpha0_mlp": (5, -14),
        "target_set_attention": (5, 7),
    }
    for method in TABLE_METHODS:
        images = indexed.loc[(method, "illuminated_observations")]
        delivery = indexed.loc[(method, "delivery_fraction")]
        x = float(images["mean"])
        y = float(delivery["mean"]) * 100.0
        axis.errorbar(
            x,
            y,
            xerr=np.array(
                [
                    [x - float(images["mean_95_ci_low"])],
                    [float(images["mean_95_ci_high"]) - x],
                ]
            ),
            yerr=np.array(
                [
                    [
                        (float(delivery["mean"]) - float(delivery["mean_95_ci_low"]))
                        * 100
                    ],
                    [
                        (float(delivery["mean_95_ci_high"]) - float(delivery["mean"]))
                        * 100
                    ],
                ]
            ),
            fmt="o",
            markersize=8,
            capsize=3,
            color=COLORS[method],
        )
        axis.annotate(
            SHORT_LABELS[method],
            (x, y),
            xytext=annotation_offsets[method],
            textcoords="offset points",
            fontsize=9,
        )
    axis.set(
        xlabel="Illuminated images",
        ylabel="Useful-delivery fraction (%)",
        title="Acquisition–delivery tradeoff (mean and 95% CI)",
    )
    axis.grid(alpha=0.2)
    save_figure(fig, output, "matched_300s_acquisition_delivery_tradeoff")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    analysis_root = input_root / "analysis"
    if not (analysis_root / "episodes_combined.csv").is_file():
        analysis_root = input_root
    combined = analysis_root / "episodes_combined.csv"
    if not combined.is_file():
        raise FileNotFoundError(f"collector output is missing: {combined}")
    frame = add_derived_metrics(pd.read_csv(combined))
    validate_campaign(frame)
    output = analysis_root / "statistical_analysis"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "episodes_with_derived_metrics.csv", index=False)
    descriptive = descriptive_statistics(frame)
    paired = paired_statistics(frame)
    pairwise = all_pairwise_statistics(frame)
    descriptive.to_csv(output / "descriptive_statistics.csv", index=False)
    paired.to_csv(output / "paired_statistics_vs_smallest_angle.csv", index=False)
    pairwise.to_csv(output / "all_pairwise_statistics.csv", index=False)
    (output / "prospectus_table_rows.tex").write_text(prospectus_table(descriptive))
    figure_output = output / "figures"
    plot_endpoint_summary(descriptive, figure_output)
    plot_operational_tradeoff(descriptive, figure_output)
    print(
        descriptive[
            descriptive["metric"].isin(
                {"illuminated_observations", "illumination_quality_fraction"}
            )
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
