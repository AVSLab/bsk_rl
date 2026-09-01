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
    "breckenridge2026_alpha0_mlp": "Monolithic MLP",
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
    # Shared dark-purple Plasma color for learned policies.
    "breckenridge2026_alpha0_mlp": "#41049d",
    "target_set_attention": "#41049d",
    # Separated Viridis colors for heuristic baselines.
    "smallest_angle_heuristic": "#2c738e",
    "closest_distance_heuristic": "#5ec962",
}
LINE_STYLES = {
    "breckenridge2026_alpha0_mlp": "-",
    "target_set_attention": "--",
    "smallest_angle_heuristic": "-",
    "closest_distance_heuristic": "-",
}
MARKERS = {
    "breckenridge2026_alpha0_mlp": "o",
    "target_set_attention": "s",
    "smallest_angle_heuristic": "^",
    "closest_distance_heuristic": "v",
}
SHORT_LABELS = {
    "smallest_angle_heuristic": "Angle",
    "closest_distance_heuristic": "Distance",
    "breckenridge2026_alpha0_mlp": "Monolithic MLP",
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
        "\\textbf{Monolithic MLP} & \\textbf{Target attention} \\\\",
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


ENDPOINT_PANELS = (
    (
        "illuminated_observations",
        "Illuminated images",
        1.0,
        "matched_300s_illuminated_images_box_scatter",
    ),
    (
        "useful_deliveries",
        "Useful downlinked images",
        1.0,
        "matched_300s_useful_deliveries_box_scatter",
    ),
    (
        "delivery_fraction",
        "Useful-delivery fraction (%)",
        100.0,
        "matched_300s_delivery_fraction_box_scatter",
    ),
    (
        "resource_constraint_interventions",
        "Shield interventions per run",
        1.0,
        "matched_300s_shield_interventions_box_scatter",
    ),
)


def _draw_box_scatter_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    descriptive: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    scale: float,
) -> None:
    """Draw all 100 seeds, a box distribution, and the mean +/- one SD."""

    positions = np.arange(len(TABLE_METHODS), dtype=float)
    values_by_method = [
        frame.loc[frame["method"] == method, metric].to_numpy(dtype=float) * scale
        for method in TABLE_METHODS
    ]
    boxes = axis.boxplot(
        values_by_method,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
    )
    for patch, method in zip(boxes["boxes"], TABLE_METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_edgecolor(COLORS[method])
        patch.set_alpha(0.18)

    indexed = descriptive.set_index(["method", "metric"])
    for method_index, (position, method, values) in enumerate(
        zip(positions, TABLE_METHODS, values_by_method)
    ):
        # Fixed random seeds make the display reproducible; jitter does not alter values.
        generator = np.random.default_rng(20260831 + method_index)
        jitter = generator.uniform(-0.19, 0.19, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=13,
            marker=MARKERS[method],
            color=COLORS[method],
            alpha=0.30,
            linewidths=0.0,
            zorder=2,
        )
        record = indexed.loc[(method, metric)]
        mean = float(record["mean"]) * scale
        std = float(record["std"]) * scale
        axis.errorbar(
            position,
            mean,
            yerr=std,
            fmt=MARKERS[method],
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.4,
            capsize=4,
            elinewidth=1.6,
            color=COLORS[method],
            zorder=4,
        )

    axis.set_xticks(positions, [SHORT_LABELS[method] for method in TABLE_METHODS])
    axis.tick_params(axis="x", labelrotation=20)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)


def plot_endpoint_summary(
    frame: pd.DataFrame, descriptive: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), constrained_layout=True)
    for axis, (metric, ylabel, scale, _) in zip(axes.flat, ENDPOINT_PANELS):
        _draw_box_scatter_panel(
            axis,
            frame,
            descriptive,
            metric=metric,
            ylabel=ylabel,
            scale=scale,
        )
    fig.suptitle("Matched AMOS 2025 evaluation (100 paired seeds)", fontsize=13)
    fig.text(
        0.5,
        -0.015,
        "Dots: individual seeds; boxes: median and IQR; open markers: mean ± one SD",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, output, "matched_300s_endpoint_summary")


def plot_individual_endpoint_panels(
    frame: pd.DataFrame, descriptive: pd.DataFrame, output: Path
) -> None:
    for metric, ylabel, scale, stem in ENDPOINT_PANELS:
        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        _draw_box_scatter_panel(
            axis,
            frame,
            descriptive,
            metric=metric,
            ylabel=ylabel,
            scale=scale,
        )
        axis.set_title(f"{ylabel} across 100 paired seeds")
        axis.text(
            0.5,
            -0.23,
            "Dots: individual seeds; box: median and IQR; open marker: mean ± one SD",
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
        )
        save_figure(fig, output, stem)


def plot_operational_tradeoff(
    frame: pd.DataFrame,
    descriptive: pd.DataFrame,
    output: Path,
    *,
    y_metric: str,
    ylabel: str,
    y_scale: float,
    stem: str,
) -> None:
    indexed = descriptive.set_index(["method", "metric"])
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    annotation_offsets = {
        "smallest_angle_heuristic": (6, -15),
        "closest_distance_heuristic": (6, -15),
        "breckenridge2026_alpha0_mlp": (6, -15),
        "target_set_attention": (6, 7),
    }
    for method in TABLE_METHODS:
        seed_rows = frame[frame["method"] == method]
        axis.scatter(
            seed_rows["illuminated_observations"],
            seed_rows[y_metric] * y_scale,
            s=16,
            marker=MARKERS[method],
            color=COLORS[method],
            alpha=0.22,
            linewidths=0.0,
            zorder=2,
        )
        images = indexed.loc[(method, "illuminated_observations")]
        delivery = indexed.loc[(method, y_metric)]
        x = float(images["mean"])
        y = float(delivery["mean"]) * y_scale
        axis.errorbar(
            x,
            y,
            xerr=float(images["std"]),
            yerr=float(delivery["std"]) * y_scale,
            fmt=MARKERS[method],
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=1.5,
            capsize=3,
            elinewidth=1.5,
            color=COLORS[method],
            zorder=4,
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
        ylabel=ylabel,
        title="Acquisition–delivery tradeoff (100 paired seeds; mean ± SD)",
    )
    axis.grid(alpha=0.2)
    save_figure(fig, output, stem)


def plot_shield_components(frame: pd.DataFrame, output: Path) -> None:
    """Show whether intervention differences arise from battery or storage."""

    components = (
        ("battery_shield_interventions", "Battery", "#f6c85f"),
        ("storage_shield_interventions", "Storage", "#6f4e7c"),
        ("wheel_shield_interventions", "Reaction wheel", "#9dd866"),
    )
    positions = np.arange(len(TABLE_METHODS))
    bottom = np.zeros(len(TABLE_METHODS), dtype=float)
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for metric, label, color in components:
        means = np.array(
            [
                frame.loc[frame["method"] == method, metric].mean()
                for method in TABLE_METHODS
            ]
        )
        axis.bar(
            positions,
            means,
            bottom=bottom,
            width=0.62,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += means
    axis.set_xticks(positions, [SHORT_LABELS[method] for method in TABLE_METHODS])
    axis.tick_params(axis="x", labelrotation=20)
    axis.set(
        ylabel="Mean shield interventions per run",
        title="Shield-intervention components across 100 paired seeds",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    save_figure(fig, output, "matched_300s_shield_intervention_components")


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
    plot_endpoint_summary(frame, descriptive, figure_output)
    plot_individual_endpoint_panels(frame, descriptive, figure_output)
    plot_operational_tradeoff(
        frame,
        descriptive,
        figure_output,
        y_metric="delivery_fraction",
        ylabel="Useful-delivery fraction (%)",
        y_scale=100.0,
        stem="matched_300s_acquisition_delivery_tradeoff",
    )
    plot_operational_tradeoff(
        frame,
        descriptive,
        figure_output,
        y_metric="useful_deliveries",
        ylabel="Useful downlinked images",
        y_scale=1.0,
        stem="matched_300s_illuminated_vs_useful_deliveries",
    )
    plot_shield_components(frame, figure_output)
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
