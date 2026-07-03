#!/usr/bin/env python3
"""Create publication-ready plots for the Breckenridge four-cell MC study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd


CELL_ORDER = [
    "leo_trained__leo_eval",
    "mixed_trained__leo_eval",
    "leo_trained__mixed_eval",
    "mixed_trained__mixed_eval",
]
CELL_LABELS = {
    "leo_trained__leo_eval": "LEO train\nLEO eval",
    "mixed_trained__leo_eval": "Mixed train\nLEO eval",
    "leo_trained__mixed_eval": "LEO train\nMixed eval",
    "mixed_trained__mixed_eval": "Mixed train\nMixed eval",
}
CELL_LEGEND_LABELS = {
    "leo_trained__leo_eval": "LEO-trained, LEO evaluation",
    "mixed_trained__leo_eval": "Mixed-trained, LEO evaluation",
    "leo_trained__mixed_eval": "LEO-trained, mixed evaluation",
    "mixed_trained__mixed_eval": "Mixed-trained, mixed evaluation",
}

ALPHA_COLORMAP = "plasma"
COMPARISON_COLORMAP = "viridis"
COMPARISON_PLASMA_COLORMAP = "plasma"
DPI = 320
GRID_ALPHA = 0.22
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 10
TITLE_FONTSIZE = 14
PUBLICATION_DIR_NAME = "plots_publication_20260623"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.home() / "rllib_results" / "breckenridge2026_mc",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Defaults to <input-root>/analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to <analysis-dir>/plots, or "
            f"<analysis-dir>/{PUBLICATION_DIR_NAME} with --publication."
        ),
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help=(
            "Write title-free comparison figures into a clean publication "
            "folder. Alpha-sweep-only figures are skipped."
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing generated files from the selected output folder first.",
    )
    parser.add_argument(
        "--force-analysis",
        action="store_true",
        help="Regenerate the portable analysis tables before plotting.",
    )
    parser.add_argument(
        "--alpha-per-seed",
        type=Path,
        default=(
            repo_root
            / "examples"
            / "results"
            / "per_seed_metrics_allPolicies_20260116_150922.csv"
        ),
        help="Latest complete alpha-sweep per-seed CSV.",
    )
    return parser.parse_args()


def resolve_alpha_per_seed(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root
        / "examples"
        / "data"
        / "archived_alpha_sweep"
        / "results"
        / "per_seed_metrics_allPolicies_20260116_150922.csv",
        repo_root
        / "examples"
        / "results"
        / "per_seed_metrics_allPolicies_20260116_150922.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the January 16 alpha-sweep per-seed CSV. "
        "Pass it explicitly with --alpha-per-seed."
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.dpi": 140,
            "savefig.dpi": DPI,
            "axes.linewidth": 1.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def cell_colors(
    colormap: str = COMPARISON_COLORMAP,
) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap(colormap)
    return {
        cell: cmap(value)
        for cell, value in zip(CELL_ORDER, np.linspace(0.08, 0.92, len(CELL_ORDER)))
    }


def comparison_encoding(
    palette: str,
) -> tuple[
    dict[str, tuple[float, float, float, float]],
    dict[str, str],
]:
    if palette == "viridis":
        colors = cell_colors(COMPARISON_COLORMAP)
        markers = {
            cell: "s" if "mixed_eval" in cell else "o" for cell in CELL_ORDER
        }
        return colors, markers
    if palette == "plasma_shapes":
        cmap = plt.get_cmap(COMPARISON_PLASMA_COLORMAP)
        evaluation_colors = {"leo": cmap(0.24), "mixed": cmap(0.78)}
        colors = {
            cell: evaluation_colors[
                "mixed" if cell.endswith("__mixed_eval") else "leo"
            ]
            for cell in CELL_ORDER
        }
        markers = {
            cell: "^" if cell.startswith("mixed_trained__") else "o"
            for cell in CELL_ORDER
        }
        return colors, markers
    raise ValueError(f"Unknown comparison palette: {palette}")


def finish_numeric_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=GRID_ALPHA)
    ax.grid(True, which="minor", alpha=GRID_ALPHA * 0.45)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))


def finish_categorical_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=GRID_ALPHA)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=DPI,
            pad_inches=0.06,
        )
    plt.close(fig)


def set_optional_title(ax: plt.Axes, title: str, show_titles: bool) -> None:
    if show_titles:
        ax.set_title(title)
    else:
        ax.set_title("")


def set_optional_suptitle(
    fig: plt.Figure, title: str, show_titles: bool
) -> None:
    if show_titles:
        fig.suptitle(title)


def clean_output_folder(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix in {".png", ".pdf", ".csv", ".json", ".md"}:
            path.unlink()


def clear_previous_figures(output_dir: Path) -> None:
    manifest_path = output_dir / "plot_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    for stem in manifest.get("figure_stems", []):
        for suffix in (".png", ".pdf", "_population_stats.csv"):
            path = output_dir / f"{stem}{suffix}"
            if path.is_file():
                path.unlink()


def ensure_analysis(input_root: Path, analysis_dir: Path, force: bool) -> None:
    per_seed = analysis_dir / "new_mc_per_seed.csv"
    if per_seed.is_file() and not force:
        return
    analyzer = Path(__file__).with_name("analyze_mc_comparison.py")
    subprocess.run(
        [
            sys.executable,
            str(analyzer),
            "--input-root",
            str(input_root),
            "--output-dir",
            str(analysis_dir),
        ],
        check=True,
    )


def ordered_data(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["cell"] = pd.Categorical(result["cell"], CELL_ORDER, ordered=True)
    sort_columns = ["cell"] + (["seed"] if "seed" in result.columns else [])
    return result.sort_values(sort_columns)


def add_derived_comparison_metrics(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["acq_success_percent"] = result["acq_success_rate"] * 100.0
    result["umbra_smart_percent"] = result["umbra_smart_fraction"] * 100.0
    result["images_per_imaging_action"] = safe_divide(
        result["illuminated_images"], result["target_imaging_count"]
    )
    result["useful_per_downlink_action"] = safe_divide(
        result["useful_downlinks_paper_estimate"], result["downlink_action_count"]
    )
    result["delivery_fraction_percent"] = (
        safe_divide(
            result["useful_downlinks_paper_estimate"], result["illuminated_images"]
        )
        * 100.0
    )
    result["downlink_action_fraction_percent"] = (
        safe_divide(
            result["downlink_action_count"],
            result["downlink_action_count"] + result["target_imaging_count"],
        )
        * 100.0
    )
    return result


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator.astype(float) / denominator
    return pd.Series(values, index=numerator.index).replace([np.inf, -np.inf], np.nan)


def add_covariance_ellipses(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color,
) -> dict[str, float]:
    points = np.column_stack([x, y])
    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    for sigma, alpha, linewidth in ((2.0, 0.07, 1.7), (1.0, 0.14, 1.25)):
        width, height = 2.0 * sigma * np.sqrt(eigvals)
        ax.add_patch(
            Ellipse(
                mean,
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=1,
            )
        )
    corr = np.corrcoef(x, y)[0, 1]
    return {
        "x_mean": float(mean[0]),
        "y_mean": float(mean[1]),
        "x_std": float(np.std(x, ddof=1)),
        "y_std": float(np.std(y, ddof=1)),
        "correlation": float(corr),
    }


def plot_gaussian_population(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    stem: str,
    palette: str,
    identity_line: bool = False,
    show_titles: bool = True,
) -> None:
    colors, markers = comparison_encoding(palette)
    fig, ax = plt.subplots(figsize=(8.5, 5.7), constrained_layout=True)
    stats_rows = []

    for cell in CELL_ORDER:
        group = data[data["cell"] == cell]
        x = pd.to_numeric(group[x_metric], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[y_metric], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        color = colors[cell]
        marker = markers[cell]
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=25,
            alpha=0.42,
            edgecolor="none",
            zorder=2,
        )
        stats = add_covariance_ellipses(ax, x, y, color)
        stats.update({"cell": cell, "N": len(x)})
        stats_rows.append(stats)
        if cell.startswith("mixed_trained__"):
            mean_marker = "o"
            mean_size = 190
        else:
            mean_marker = "*"
            mean_size = 165
        ax.scatter(
            stats["x_mean"],
            stats["y_mean"],
            color=color,
            marker=mean_marker,
            s=mean_size,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

    if identity_line:
        low = min(ax.get_xlim()[0], ax.get_ylim()[0])
        high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=1.1,
            color="0.32",
            alpha=0.65,
            label="1:1",
            zorder=0,
        )

    cell_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[cell],
            color="none",
            markerfacecolor=colors[cell],
            markeredgecolor="none",
            markersize=8,
            label=CELL_LEGEND_LABELS[cell],
        )
        for cell in CELL_ORDER
    ]
    contour_handle = Line2D(
        [0],
        [0],
        color="0.25",
        linewidth=1.6,
        label=r"1$\sigma$/2$\sigma$ Gaussian contours",
    )
    leo_mean_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="0.55",
        markeredgecolor="black",
        markersize=12,
        label="LEO-trained mean",
    )
    mixed_mean_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="0.55",
        markeredgecolor="black",
        markersize=10,
        label="Mixed-trained mean",
    )
    ax.legend(
        handles=cell_handles + [leo_mean_handle, mixed_mean_handle, contour_handle],
        loc="best",
        framealpha=0.94,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    set_optional_title(ax, title, show_titles)
    finish_numeric_axis(ax)
    save_figure(fig, output_dir, stem)
    pd.DataFrame(stats_rows).to_csv(
        output_dir / f"{stem}_population_stats.csv", index=False
    )


def metric_mean_std(summary: pd.DataFrame, cell: str, metric: str) -> tuple[float, float]:
    row = summary[summary["cell"] == cell].iloc[0]
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def plot_overview(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    palette: str,
    stem: str,
    show_titles: bool = True,
) -> None:
    colors, _ = comparison_encoding(palette)
    hatches = [
        "//"
        if cell.startswith("mixed_trained__") and palette == "plasma_shapes"
        else ""
        for cell in CELL_ORDER
    ]
    x = np.arange(len(CELL_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    reward_mean = [metric_mean_std(summary, cell, "total_reward")[0] for cell in CELL_ORDER]
    reward_std = [metric_mean_std(summary, cell, "total_reward")[1] for cell in CELL_ORDER]
    bars = axes[0, 0].bar(
        x,
        reward_mean,
        color=[colors[cell] for cell in CELL_ORDER],
        alpha=0.88,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[0, 0].errorbar(x, reward_mean, yerr=reward_std, fmt="none", color="0.15", capsize=4)
    axes[0, 0].set_ylabel("Total reward")
    set_optional_title(axes[0, 0], "Episode return", show_titles)

    for offset, metric, label, marker in (
        (-0.09, "illuminated_images", "Illuminated images", "o"),
        (0.09, "useful_downlinks_paper_estimate", "Useful downlinks", "s"),
    ):
        means = [metric_mean_std(summary, cell, metric)[0] for cell in CELL_ORDER]
        stds = [metric_mean_std(summary, cell, metric)[1] for cell in CELL_ORDER]
        for index, cell in enumerate(CELL_ORDER):
            axes[0, 1].errorbar(
                index + offset,
                means[index],
                yerr=stds[index],
                marker=marker,
                markersize=7,
                color=colors[cell],
                markeredgecolor="black",
                markeredgewidth=0.55,
                capsize=3,
                linestyle="none",
            )
    axes[0, 1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.3", linestyle="none", label="Illuminated images"),
            Line2D([0], [0], marker="s", color="0.3", linestyle="none", label="Useful downlinks"),
        ],
        framealpha=0.94,
    )
    axes[0, 1].set_ylabel("Count per episode")
    set_optional_title(axes[0, 1], "Collection and delivery", show_titles)

    success_mean = [metric_mean_std(summary, cell, "acq_success_rate")[0] for cell in CELL_ORDER]
    success_std = [metric_mean_std(summary, cell, "acq_success_rate")[1] for cell in CELL_ORDER]
    bars = axes[1, 0].bar(
        x,
        np.array(success_mean) * 100.0,
        color=[colors[cell] for cell in CELL_ORDER],
        alpha=0.88,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[1, 0].errorbar(
        x,
        np.array(success_mean) * 100.0,
        yerr=np.array(success_std) * 100.0,
        fmt="none",
        color="0.15",
        capsize=4,
    )
    axes[1, 0].set_ylabel("Acquisition success [%]")
    set_optional_title(axes[1, 0], "Imaging-command effectiveness", show_titles)

    umbra_mean = [metric_mean_std(summary, cell, "umbra_smart_fraction")[0] for cell in CELL_ORDER]
    umbra_std = [metric_mean_std(summary, cell, "umbra_smart_fraction")[1] for cell in CELL_ORDER]
    bars = axes[1, 1].bar(
        x,
        np.array(umbra_mean) * 100.0,
        color=[colors[cell] for cell in CELL_ORDER],
        alpha=0.88,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[1, 1].errorbar(
        x,
        np.array(umbra_mean) * 100.0,
        yerr=np.array(umbra_std) * 100.0,
        fmt="none",
        color="0.15",
        capsize=4,
    )
    axes[1, 1].set_ylabel("Smart umbra decisions [%]")
    set_optional_title(axes[1, 1], "Eclipse-time target selection", show_titles)

    for ax in axes.flat:
        ax.set_xticks(x, [CELL_LABELS[cell] for cell in CELL_ORDER])
        finish_categorical_axis(ax)
    save_figure(fig, output_dir, stem)


def paired_frame(data: pd.DataFrame, environment: str, metric: str) -> pd.DataFrame:
    selected = data[data["evaluation_environment"] == environment]
    pivot = selected.pivot(index="seed", columns="training_environment", values=metric)
    return pivot.dropna(subset=["leo", "mixed"])


def paired_ci(delta: pd.Series) -> tuple[float, float, float]:
    mean = float(delta.mean())
    half_width = 1.984 * float(delta.std(ddof=1)) / math.sqrt(len(delta))
    return mean, mean - half_width, mean + half_width


def evaluation_paired_frame(data: pd.DataFrame, training: str, metric: str) -> pd.DataFrame:
    selected = data[data["training_environment"] == training]
    pivot = selected.pivot(index="seed", columns="evaluation_environment", values=metric)
    return pivot.dropna(subset=["leo", "mixed"])


def paired_delta_values(
    data: pd.DataFrame,
    *,
    mode: str,
    group: str,
    metric: str,
) -> np.ndarray:
    if mode == "training":
        pivot = paired_frame(data, group, metric)
        delta = pivot["mixed"] - pivot["leo"]
    elif mode == "evaluation":
        pivot = evaluation_paired_frame(data, group, metric)
        delta = pivot["mixed"] - pivot["leo"]
    else:
        raise ValueError(f"Unknown paired-delta mode: {mode}")
    return delta.dropna().to_numpy(dtype=float)


def plot_delta_distribution_grid(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: tuple[tuple[str, str], ...],
    mode: str,
    stem: str,
    show_titles: bool = True,
) -> None:
    if mode == "training":
        groups = ("leo", "mixed")
        group_labels = ("LEO eval", "Mixed eval")
        delta_label_prefix = "Mixed-trained minus LEO-trained"
    elif mode == "evaluation":
        groups = ("leo", "mixed")
        group_labels = ("LEO-trained", "Mixed-trained")
        delta_label_prefix = "Mixed eval minus LEO eval"
    else:
        raise ValueError(f"Unknown paired-delta mode: {mode}")

    viridis = plt.get_cmap(COMPARISON_COLORMAP)
    group_colors = {groups[0]: viridis(0.24), groups[1]: viridis(0.76)}
    columns = min(4, len(metrics))
    rows = math.ceil(len(metrics) / columns)
    fig_width = 3.65 * columns
    fig_height = 4.35 * rows
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        squeeze=False,
    )

    rng = np.random.default_rng(20260623)
    for ax, (metric, label) in zip(axes.flat, metrics):
        values = [
            paired_delta_values(data, mode=mode, group=group, metric=metric)
            for group in groups
        ]
        violins = ax.violinplot(
            values,
            positions=np.arange(len(groups)),
            widths=0.72,
            showextrema=False,
        )
        for body, group in zip(violins["bodies"], groups):
            body.set_facecolor(group_colors[group])
            body.set_edgecolor(group_colors[group])
            body.set_alpha(0.60)
        ax.boxplot(
            values,
            positions=np.arange(len(groups)),
            widths=0.22,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "0.18"},
            medianprops={"color": "0.1", "linewidth": 1.5},
            whiskerprops={"color": "0.25"},
            capprops={"color": "0.25"},
        )
        for index, (group, array) in enumerate(zip(groups, values)):
            if len(array) == 0:
                continue
            jitter = rng.normal(0.0, 0.035, size=len(array))
            ax.scatter(
                np.full(len(array), index) + jitter,
                array,
                s=9,
                color=group_colors[group],
                edgecolor="none",
                alpha=0.22,
                zorder=1,
            )
            mean, ci_low, ci_high = paired_ci(pd.Series(array))
            ax.errorbar(
                index,
                mean,
                yerr=[[mean - ci_low], [ci_high - mean]],
                marker="o",
                markersize=5.2,
                color="black",
                markerfacecolor="white",
                capsize=3,
                linewidth=1.0,
                zorder=5,
            )
        ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
        ax.set_xticks(np.arange(len(groups)), group_labels)
        ax.set_ylabel(f"{delta_label_prefix}\n{label}")
        set_optional_title(ax, label, show_titles)
        finish_categorical_axis(ax)

    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")
    save_figure(fig, output_dir, stem)


def paired_effect_ranking(
    data: pd.DataFrame,
    metrics: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    rows = []
    for environment in ("leo", "mixed"):
        for metric, label in metrics:
            values = paired_delta_values(
                data, mode="training", group=environment, metric=metric
            )
            if len(values) < 2:
                continue
            mean, ci_low, ci_high = paired_ci(pd.Series(values))
            std = float(np.std(values, ddof=1))
            standardized = mean / std if std > 0 else np.nan
            rows.append(
                {
                    "evaluation_environment": environment,
                    "metric": metric,
                    "label": label,
                    "N": len(values),
                    "mean_delta_mixed_minus_leo": mean,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "std_delta": std,
                    "standardized_paired_effect": standardized,
                    "absolute_standardized_paired_effect": abs(standardized),
                    "mixed_trained_win_fraction": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "absolute_standardized_paired_effect", ascending=False
    )


def plot_paired_effect_ranking(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    show_titles: bool = True,
) -> None:
    metrics = (
        ("total_reward", "Reward"),
        ("illuminated_images", "Illuminated images"),
        ("useful_downlinks_paper_estimate", "Useful downlinks"),
        ("target_imaging_count", "Imaging actions"),
        ("downlink_action_count", "Downlink actions"),
        ("charge_action_count", "Charge actions"),
        ("desat_action_count", "Desat actions"),
        ("acq_success_percent", "Acquisition success [pp]"),
        ("avg_acquisition_time_sec", "Acquisition time [s]"),
        ("umbra_smart_percent", "Smart umbra decisions [pp]"),
        ("images_per_imaging_action", "Images per imaging action"),
        ("useful_per_downlink_action", "Useful per downlink action"),
        ("delivery_fraction_percent", "Delivery fraction [pp]"),
        ("downlink_action_fraction_percent", "Downlink action fraction [pp]"),
    )
    ranking = paired_effect_ranking(data, metrics)
    ranking.to_csv(output_dir / f"{stem}.csv", index=False)
    top = ranking.head(10).copy()
    top["display_label"] = (
        top["label"] + " (" + top["evaluation_environment"].str.upper() + " eval)"
    )
    top = top.iloc[::-1]
    viridis = plt.get_cmap(COMPARISON_COLORMAP)
    colors = {"leo": viridis(0.24), "mixed": viridis(0.76)}

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    y = np.arange(len(top))
    for index, (_, row) in enumerate(top.iterrows()):
        environment = row["evaluation_environment"]
        estimate = row["standardized_paired_effect"]
        std = row["std_delta"]
        low = row["ci95_low"] / std if std > 0 else np.nan
        high = row["ci95_high"] / std if std > 0 else np.nan
        ax.plot(
            [low, high],
            [index, index],
            color=colors[environment],
            linewidth=2.4,
            alpha=0.8,
        )
        ax.scatter(
            estimate,
            index,
            s=68,
            color=colors[environment],
            edgecolor="black",
            linewidth=0.65,
            zorder=4,
        )
    ax.axvline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, top["display_label"])
    ax.set_xlabel("Standardized paired effect")
    ax.set_ylabel("")
    set_optional_title(
        ax,
        "Largest same-seed training-distribution effects",
        show_titles,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=colors["leo"],
                markerfacecolor=colors["leo"],
                markeredgecolor="black",
                label="LEO evaluation",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=colors["mixed"],
                markerfacecolor=colors["mixed"],
                markeredgecolor="black",
                label="Mixed evaluation",
            ),
        ],
        framealpha=0.94,
        loc="lower right",
    )
    finish_numeric_axis(ax)
    save_figure(fig, output_dir, stem)


def plot_paired_reward(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str = "mc_paired_reward_by_evaluation_environment",
    show_titles: bool = True,
) -> None:
    colors = cell_colors(COMPARISON_COLORMAP)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax, environment in zip(axes, ("leo", "mixed")):
        pivot = paired_frame(data, environment, "total_reward")
        color = colors[f"mixed_trained__{environment}_eval"]
        ax.scatter(
            pivot["leo"],
            pivot["mixed"],
            color=color,
            alpha=0.52,
            s=28,
            edgecolor="none",
        )
        low = min(float(pivot.min().min()), float(pivot.min().min())) - 0.6
        high = max(float(pivot.max().max()), float(pivot.max().max())) + 0.6
        ax.plot([low, high], [low, high], "--", color="0.3", linewidth=1.1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        mean, ci_low, ci_high = paired_ci(pivot["mixed"] - pivot["leo"])
        ax.text(
            0.04,
            0.95,
            rf"Paired $\Delta$ = {mean:+.2f}"
            f"\n95% CI [{ci_low:+.2f}, {ci_high:+.2f}]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
        ax.set_xlabel("LEO-trained policy reward")
        ax.set_ylabel("Mixed-trained policy reward")
        set_optional_title(ax, f"{environment.upper()} evaluation", show_titles)
        finish_numeric_axis(ax)
    save_figure(fig, output_dir, stem)


def plot_paired_delta_distributions(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str = "mc_paired_training_effect_distributions",
    show_titles: bool = True,
) -> None:
    metrics = [
        ("total_reward", "Reward difference"),
        ("illuminated_images", "Illuminated-image difference"),
        ("useful_downlinks_paper_estimate", "Useful-downlink difference"),
    ]
    environment_colors = {
        "leo": cell_colors(COMPARISON_COLORMAP)["mixed_trained__leo_eval"],
        "mixed": cell_colors(COMPARISON_COLORMAP)["mixed_trained__mixed_eval"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.45), constrained_layout=True)
    for ax, (metric, label) in zip(axes, metrics):
        values = []
        for environment in ("leo", "mixed"):
            pivot = paired_frame(data, environment, metric)
            values.append((pivot["mixed"] - pivot["leo"]).to_numpy(dtype=float))
        violins = ax.violinplot(values, positions=[0, 1], widths=0.72, showextrema=False)
        for body, environment in zip(violins["bodies"], ("leo", "mixed")):
            body.set_facecolor(environment_colors[environment])
            body.set_edgecolor(environment_colors[environment])
            body.set_alpha(0.56)
        ax.boxplot(
            values,
            positions=[0, 1],
            widths=0.22,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "0.18"},
            medianprops={"color": "0.1", "linewidth": 1.5},
            whiskerprops={"color": "0.25"},
            capprops={"color": "0.25"},
        )
        ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
        ax.set_xticks([0, 1], ["LEO eval", "Mixed eval"])
        ax.set_ylabel(f"Mixed-trained minus LEO-trained\n{label.lower()}")
        set_optional_title(ax, label, show_titles)
        finish_categorical_axis(ax)
    save_figure(fig, output_dir, stem)


def plot_regime_mix(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str = "mc_mixed_evaluation_regime_selection",
    show_titles: bool = True,
) -> None:
    selected = data[data["evaluation_environment"] == "mixed"]
    colors = plt.get_cmap(COMPARISON_COLORMAP)([0.18, 0.52, 0.86])
    labels = ["LEO-trained policy", "Mixed-trained policy", "Catalog draw weights"]
    rows = []
    for training in ("leo", "mixed"):
        group = selected[selected["training_environment"] == training]
        rows.append(
            [group[f"frac_all_{regime}"].mean() for regime in ("LEO", "MEO", "GEO")]
        )
    rows.append([0.5, 0.3, 0.2])
    values = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    left = np.zeros(len(values))
    for index, regime in enumerate(("LEO", "MEO", "GEO")):
        ax.barh(labels, values[:, index], left=left, color=colors[index], label=regime)
        for row_index, value in enumerate(values[:, index]):
            if value > 0.08:
                ax.text(
                    left[row_index] + value / 2.0,
                    row_index,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if index == 0 else "black",
                    fontsize=10,
                )
        left += values[:, index]
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of imaging actions by target regime")
    set_optional_title(
        ax,
        "Mixed-catalog selection compared with catalog composition",
        show_titles,
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    finish_numeric_axis(ax)
    save_figure(fig, output_dir, stem)


def add_alpha_colorbar(fig: plt.Figure, ax, norm: Normalize) -> None:
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(ALPHA_COLORMAP))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.025, fraction=0.045)
    colorbar.set_label(r"Downlink reward weight $\alpha$")


def latest_alpha_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
    data = data.dropna(subset=["alpha", "seed"]).copy()
    data = data.drop_duplicates(["alpha", "env", "seed"], keep="first")
    counts = data.groupby("alpha")["seed"].nunique()
    if len(counts) != 11 or not counts.eq(100).all():
        raise ValueError(
            "Latest alpha sweep must contain 11 policies with 100 unique seeds each"
        )
    return data.sort_values(["alpha", "seed"])


def plot_latest_alpha_overview(data: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        "total_reward",
        "illuminated_images",
        "useful_downlinks_est",
        "target_imaging_count",
        "downlink_action_count",
        "acq_success_rate",
        "avg_acquisition_time_sec",
    ]
    summary = data.groupby("alpha")[metrics].agg(["mean", "std"])
    alphas = summary.index.to_numpy(dtype=float)
    cmap = plt.get_cmap(ALPHA_COLORMAP)
    norm = Normalize(0.0, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.5), constrained_layout=True)

    def series(
        ax: plt.Axes,
        metric: str,
        label: str,
        marker: str,
        divisor: float = 1.0,
    ) -> None:
        means = summary[(metric, "mean")].to_numpy(dtype=float) / divisor
        stds = summary[(metric, "std")].to_numpy(dtype=float) / divisor
        ax.plot(alphas, means, color="0.38", linewidth=1.0, alpha=0.52)
        ax.errorbar(
            alphas,
            means,
            yerr=stds,
            fmt="none",
            ecolor="0.4",
            alpha=0.45,
            capsize=3,
        )
        ax.scatter(
            alphas,
            means,
            c=alphas,
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=68,
            edgecolor="black",
            linewidth=0.55,
            label=label,
            zorder=3,
        )

    for metric, label, marker in (
        ("total_reward", "Total reward", "o"),
        ("illuminated_images", "Illuminated images", "s"),
        ("useful_downlinks_est", "Useful downlinks", "D"),
    ):
        series(axes[0, 0], metric, label, marker)
    axes[0, 0].set_ylabel("Episode count / reward")
    axes[0, 0].set_title("Collection and delivery outcomes")
    axes[0, 0].legend(framealpha=0.94)

    for metric, label, marker in (
        ("target_imaging_count", "Imaging actions", "o"),
        ("downlink_action_count", "Downlink actions", "s"),
    ):
        series(axes[0, 1], metric, label, marker)
    axes[0, 1].set_ylabel("Actions per episode")
    axes[0, 1].set_title("Action allocation")
    axes[0, 1].legend(framealpha=0.94)

    series(axes[1, 0], "acq_success_rate", "Acquisition success", "o")
    axes[1, 0].set_ylabel("Acquisition success fraction")
    axes[1, 0].set_title("Imaging-command effectiveness")

    series(
        axes[1, 1],
        "avg_acquisition_time_sec",
        "Mean acquisition time",
        "o",
        divisor=60.0,
    )
    axes[1, 1].set_ylabel("Mean acquisition time [min]")
    axes[1, 1].set_title("Acquisition cadence")

    for ax in axes.flat:
        ax.set_xlabel(r"Downlink reward weight $\alpha$")
        ax.set_xlim(-0.03, 1.03)
        finish_numeric_axis(ax)
    add_alpha_colorbar(fig, axes, norm)
    save_figure(
        fig,
        output_dir,
        "alpha_sweep_100seed_overview_plasma_20260622",
    )


def plot_alpha_gaussian_population(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    stem: str,
    identity_line: bool = False,
) -> None:
    cmap = plt.get_cmap(ALPHA_COLORMAP)
    norm = Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8.7, 5.8), constrained_layout=True)
    stats_rows = []

    for alpha in sorted(data["alpha"].unique()):
        group = data[data["alpha"] == alpha]
        x = pd.to_numeric(group[x_metric], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[y_metric], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        color = cmap(norm(float(alpha)))
        ax.scatter(
            x,
            y,
            color=color,
            s=15,
            alpha=0.18,
            edgecolor="none",
            zorder=2,
        )
        stats = add_covariance_ellipses(ax, x, y, color)
        stats.update({"alpha": float(alpha), "N": len(x)})
        stats_rows.append(stats)
        ax.scatter(
            stats["x_mean"],
            stats["y_mean"],
            color=color,
            marker="*",
            s=125,
            edgecolor="black",
            linewidth=0.65,
            zorder=4,
        )

    if identity_line:
        low = min(ax.get_xlim()[0], ax.get_ylim()[0])
        high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=1.05,
            color="0.3",
            alpha=0.65,
            zorder=0,
        )

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markersize=11,
                label="Policy-population mean",
            ),
            Line2D(
                [0],
                [0],
                color="0.3",
                linewidth=1.5,
                label=r"1$\sigma$/2$\sigma$ Gaussian contours",
            ),
        ],
        framealpha=0.94,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    finish_numeric_axis(ax)
    add_alpha_colorbar(fig, ax, norm)
    save_figure(fig, output_dir, stem)
    pd.DataFrame(stats_rows).sort_values("alpha").to_csv(
        output_dir / f"{stem}_population_stats.csv", index=False
    )


def plot_comparison_ecdfs(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    palette: str,
    stem: str,
    show_titles: bool = True,
) -> None:
    colors, markers = comparison_encoding(palette)
    metrics = (
        ("total_reward", "Total reward"),
        ("useful_downlinks_paper_estimate", "Useful downlinks"),
        ("acq_success_percent", "Acquisition success [%]"),
        ("umbra_smart_percent", "Smart umbra decisions [%]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        for cell in CELL_ORDER:
            values = np.sort(
                pd.to_numeric(
                    data.loc[data["cell"] == cell, metric], errors="coerce"
                ).dropna()
            )
            probabilities = np.arange(1, len(values) + 1) / len(values)
            ax.plot(
                values,
                probabilities,
                color=colors[cell],
                marker=markers[cell],
                markevery=max(1, len(values) // 9),
                markersize=4.2,
                linewidth=1.7,
                alpha=0.92,
                label=CELL_LEGEND_LABELS[cell],
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Empirical cumulative probability")
        finish_numeric_axis(ax)
    axes[0, 0].legend(framealpha=0.94)
    set_optional_suptitle(fig, "Monte Carlo outcome distributions", show_titles)
    save_figure(fig, output_dir, stem)


def plot_policy_win_rates(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str = "mc_mixed_trained_win_rates_plasma",
    colormap: str = COMPARISON_PLASMA_COLORMAP,
    show_titles: bool = True,
) -> None:
    metrics = (
        ("total_reward", "Reward"),
        ("illuminated_images", "Images"),
        ("useful_downlinks_paper_estimate", "Useful\ndownlinks"),
        ("acq_success_rate", "Acquisition\nsuccess"),
        ("umbra_smart_fraction", "Smart umbra\ndecisions"),
    )
    environments = ("leo", "mixed")
    cmap = plt.get_cmap(colormap)
    colors = {"leo": cmap(0.24), "mixed": cmap(0.78)}
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
    rows = []
    for offset, environment in zip((-width / 2, width / 2), environments):
        wins = []
        for metric, _ in metrics:
            pivot = paired_frame(data, environment, metric)
            delta = pivot["mixed"] - pivot["leo"]
            wins.append(float((delta > 0).mean()))
            rows.append(
                {
                    "evaluation_environment": environment,
                    "metric": metric,
                    "mixed_trained_win_fraction": float((delta > 0).mean()),
                    "tie_fraction": float((delta == 0).mean()),
                    "leo_trained_win_fraction": float((delta < 0).mean()),
                }
            )
        bars = ax.bar(
            x + offset,
            wins,
            width=width,
            color=colors[environment],
            label=f"{environment.upper()} evaluation",
            alpha=0.9,
        )
        ax.bar_label(bars, labels=[f"{100 * value:.0f}%" for value in wins], padding=3)
    ax.axhline(0.5, color="0.3", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Seeds where mixed-trained policy is better")
    set_optional_title(ax, "Same-seed policy win rates", show_titles)
    ax.legend(framealpha=0.94)
    finish_categorical_axis(ax)
    save_figure(fig, output_dir, stem)
    pd.DataFrame(rows).to_csv(
        output_dir / f"{stem}.csv", index=False
    )


def plot_alpha_efficiency(data: pd.DataFrame, output_dir: Path) -> None:
    derived = data.copy()
    derived["images_per_imaging_action"] = (
        derived["illuminated_images"] / derived["target_imaging_count"]
    )
    derived["useful_per_downlink_action"] = (
        derived["useful_downlinks_est"] / derived["downlink_action_count"]
    )
    derived["delivery_fraction"] = (
        derived["useful_downlinks_est"] / derived["illuminated_images"]
    )
    derived["downlink_action_fraction"] = derived["downlink_action_count"] / (
        derived["downlink_action_count"] + derived["target_imaging_count"]
    )
    metrics = (
        ("images_per_imaging_action", "Images per imaging action"),
        ("useful_per_downlink_action", "Useful images per downlink action"),
        ("delivery_fraction", "Useful / illuminated images"),
        ("downlink_action_fraction", "Downlink fraction of image/downlink actions"),
    )
    cmap = plt.get_cmap(ALPHA_COLORMAP)
    norm = Normalize(0.0, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        summary = derived.groupby("alpha")[metric].agg(["mean", "std"])
        alpha = summary.index.to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].to_numpy(dtype=float)
        ax.plot(alpha, mean, color="0.35", linewidth=1.0, alpha=0.55)
        ax.errorbar(
            alpha,
            mean,
            yerr=std,
            fmt="none",
            ecolor="0.4",
            alpha=0.45,
            capsize=3,
        )
        ax.scatter(
            alpha,
            mean,
            c=alpha,
            cmap=cmap,
            norm=norm,
            s=68,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
        ax.set_xlabel(r"Downlink reward weight $\alpha$")
        ax.set_ylabel(label)
        ax.set_xlim(-0.03, 1.03)
        finish_numeric_axis(ax)
    add_alpha_colorbar(fig, axes, norm)
    fig.suptitle("LEO-trained policy efficiency across reward weights")
    save_figure(fig, output_dir, "alpha_sweep_efficiency_plasma_20260622")


def plot_alpha_metric_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("total_reward", "Reward"),
        ("illuminated_images", "Illuminated images"),
        ("useful_downlinks_est", "Useful downlinks"),
        ("target_imaging_count", "Imaging actions"),
        ("downlink_action_count", "Downlink actions"),
        ("acq_success_rate", "Acquisition success"),
        ("avg_acquisition_time_sec", "Acquisition time"),
    )
    summary = data.groupby("alpha")[[metric for metric, _ in metrics]].mean()
    values = summary.to_numpy(dtype=float).T
    spread = np.ptp(values, axis=1, keepdims=True)
    normalized = (values - values.min(axis=1, keepdims=True)) / np.where(
        spread == 0.0, 1.0, spread
    )
    fig, ax = plt.subplots(figsize=(11.0, 5.3), constrained_layout=True)
    image = ax.imshow(normalized, cmap=ALPHA_COLORMAP, aspect="auto", vmin=0, vmax=1)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            normalized_value = normalized[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized_value < 0.48 else "black",
                fontsize=8,
            )
    ax.set_xticks(
        np.arange(len(summary.index)),
        [f"{alpha:.1f}" for alpha in summary.index],
    )
    ax.set_yticks(
        np.arange(len(metrics)),
        [label for _, label in metrics],
    )
    ax.set_xlabel(r"Downlink reward weight $\alpha$")
    ax.set_title("LEO-trained policy metric landscape")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Within-metric normalized mean")
    save_figure(fig, output_dir, "alpha_sweep_metric_heatmap_plasma_20260622")


def write_readme(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    publication: bool = False,
) -> None:
    lines = [
        "# Breckenridge 2026 Four-Cell Monte Carlo Figures",
        "",
        "All panels use 100 paired seeds per cell. Error bars in the overview are",
        "sample standard deviations; paired annotations use 95% confidence intervals",
        "for same-seed differences.",
        "",
        "## Main paired effects",
        "",
    ]
    for environment in ("leo", "mixed"):
        pivot = paired_frame(data, environment, "total_reward")
        mean, low, high = paired_ci(pivot["mixed"] - pivot["leo"])
        lines.append(
            f"- {environment.upper()} evaluation reward, mixed-trained minus "
            f"LEO-trained: {mean:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}]."
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `mc_four_cell_overview`: outcome, acquisition, and umbra summary.",
            "- `mc_gaussian_performance_landscape`: per-seed outcome populations.",
            "- `mc_gaussian_action_landscape`: imaging/downlink action populations.",
            "- `mc_paired_reward_by_evaluation_environment`: same-seed reward comparison.",
            "- `mc_paired_training_effect_distributions`: paired effect distributions.",
            "- `mc_mixed_evaluation_regime_selection`: selected target regimes.",
            "- `mc_gaussian_acquisition_landscape`: acquisition cadence/effectiveness.",
            "- `mc_gaussian_umbra_reward_landscape`: umbra behavior versus reward.",
            "- `mc_*_viridis`: four-cell comparisons with one viridis color per cell.",
            "- `mc_paired_training_effect_*_viridis`: same-seed training-effect",
            "  distributions for outcome, action, behavior, and efficiency metrics.",
            "- `mc_evaluation_environment_effect_*_viridis`: same-seed LEO-vs-mixed",
            "  evaluation-environment effect distributions.",
            "- `mc_paired_training_effect_ranked_viridis`: largest standardized",
            "  paired training effects.",
            "- `mc_comparison_ecdfs_*`: complete empirical outcome distributions.",
        ]
    )
    if publication:
        lines.extend(
            [
                "",
                "Publication mode uses title-free figures and the viridis colormap.",
                "LEO-trained Gaussian means are stars; mixed-trained Gaussian means",
                "are larger filled circles.",
            ]
        )
    else:
        lines.extend(
            [
                "- `mc_*_plasma_shapes`: evaluation environment in plasma colors, with",
                "  circles for LEO-trained and triangles for mixed-trained policies.",
                "- `mc_mixed_trained_win_rates_plasma`: same-seed win fractions.",
                "- `alpha_sweep_100seed_overview_plasma_20260622`: complete alpha sweep.",
                "- `alpha_sweep_gaussian_*_plasma_*`: alpha-sweep population views.",
                "- `alpha_sweep_efficiency_plasma_20260622`: derived policy efficiencies.",
                "- `alpha_sweep_metric_heatmap_plasma_20260622`: normalized metric landscape.",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def plot_comparison_suite(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    palettes: tuple[str, ...],
    show_titles: bool,
    publication: bool,
) -> None:
    gaussian_specs = (
        (
            "illuminated_images",
            "useful_downlinks_paper_estimate",
            "Illuminated images acquired",
            "Useful images downlinked",
            "Four-cell collection-delivery populations",
            "mc_gaussian_performance_landscape",
            True,
        ),
        (
            "target_imaging_count",
            "downlink_action_count",
            "Imaging actions",
            "Downlink actions",
            "Four-cell action-allocation populations",
            "mc_gaussian_action_landscape",
            False,
        ),
        (
            "avg_acquisition_time_sec",
            "acq_success_percent",
            "Mean successful acquisition time [s]",
            "Acquisition success [%]",
            "Four-cell acquisition-behavior populations",
            "mc_gaussian_acquisition_landscape",
            False,
        ),
        (
            "umbra_smart_percent",
            "total_reward",
            "Smart umbra decisions [%]",
            "Total reward",
            "Umbra effectiveness and episode return",
            "mc_gaussian_umbra_reward_landscape",
            False,
        ),
    )
    for palette in palettes:
        plot_overview(
            summary,
            output_dir,
            palette=palette,
            stem=f"mc_four_cell_overview_{palette}",
            show_titles=show_titles,
        )
        for (
            x_metric,
            y_metric,
            x_label,
            y_label,
            title,
            stem,
            identity_line,
        ) in gaussian_specs:
            plot_gaussian_population(
                data,
                output_dir,
                x_metric=x_metric,
                y_metric=y_metric,
                x_label=x_label,
                y_label=y_label,
                title=title,
                stem=f"{stem}_{palette}",
                palette=palette,
                identity_line=identity_line,
                show_titles=show_titles,
            )
        plot_comparison_ecdfs(
            data,
            output_dir,
            palette=palette,
            stem=f"mc_comparison_ecdfs_{palette}",
            show_titles=show_titles,
        )

    suffix = "_viridis" if publication else ""
    plot_paired_reward(
        data,
        output_dir,
        stem=f"mc_paired_reward_by_evaluation_environment{suffix}",
        show_titles=show_titles,
    )
    plot_paired_delta_distributions(
        data,
        output_dir,
        stem=f"mc_paired_training_effect_distributions{suffix}",
        show_titles=show_titles,
    )
    training_distribution_specs = (
        (
            "mc_paired_training_effect_outcome_distributions_viridis",
            (
                ("total_reward", "Reward difference"),
                ("illuminated_images", "Illuminated-image difference"),
                ("useful_downlinks_paper_estimate", "Useful-downlink difference"),
                ("acq_success_percent", "Acquisition-success difference [pp]"),
            ),
        ),
        (
            "mc_paired_training_effect_action_distributions_viridis",
            (
                ("target_imaging_count", "Imaging-action difference"),
                ("downlink_action_count", "Downlink-action difference"),
                ("charge_action_count", "Charge-action difference"),
                ("desat_action_count", "Desat-action difference"),
            ),
        ),
        (
            "mc_paired_training_effect_behavior_distributions_viridis",
            (
                ("umbra_smart_percent", "Smart-umbra difference [pp]"),
                ("avg_acquisition_time_sec", "Acquisition-time difference [s]"),
                ("charge_action_count", "Charge-action difference"),
                ("desat_action_count", "Desat-action difference"),
            ),
        ),
        (
            "mc_paired_training_effect_efficiency_distributions_viridis",
            (
                ("images_per_imaging_action", "Images per imaging-action difference"),
                ("useful_per_downlink_action", "Useful per downlink-action difference"),
                ("delivery_fraction_percent", "Delivery-fraction difference [pp]"),
                ("downlink_action_fraction_percent", "Downlink-action fraction [pp]"),
            ),
        ),
    )
    for stem, metrics in training_distribution_specs:
        plot_delta_distribution_grid(
            data,
            output_dir,
            metrics=metrics,
            mode="training",
            stem=stem,
            show_titles=show_titles,
        )

    evaluation_distribution_specs = (
        (
            "mc_evaluation_environment_effect_outcome_distributions_viridis",
            (
                ("total_reward", "Reward difference"),
                ("illuminated_images", "Illuminated-image difference"),
                ("useful_downlinks_paper_estimate", "Useful-downlink difference"),
                ("acq_success_percent", "Acquisition-success difference [pp]"),
            ),
        ),
        (
            "mc_evaluation_environment_effect_behavior_distributions_viridis",
            (
                ("umbra_smart_percent", "Smart-umbra difference [pp]"),
                ("avg_acquisition_time_sec", "Acquisition-time difference [s]"),
                ("target_imaging_count", "Imaging-action difference"),
                ("downlink_action_count", "Downlink-action difference"),
            ),
        ),
    )
    for stem, metrics in evaluation_distribution_specs:
        plot_delta_distribution_grid(
            data,
            output_dir,
            metrics=metrics,
            mode="evaluation",
            stem=stem,
            show_titles=show_titles,
        )

    plot_paired_effect_ranking(
        data,
        output_dir,
        stem="mc_paired_training_effect_ranked_viridis",
        show_titles=show_titles,
    )
    plot_regime_mix(
        data,
        output_dir,
        stem=f"mc_mixed_evaluation_regime_selection{suffix}",
        show_titles=show_titles,
    )
    plot_policy_win_rates(
        data,
        output_dir,
        stem=f"mc_mixed_trained_win_rates{suffix or '_plasma'}",
        colormap=COMPARISON_COLORMAP if publication else COMPARISON_PLASMA_COLORMAP,
        show_titles=show_titles,
    )
    write_readme(data, output_dir, publication=publication)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    analysis_dir = (
        args.analysis_dir.expanduser().resolve()
        if args.analysis_dir
        else input_root / "analysis"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else analysis_dir / (PUBLICATION_DIR_NAME if args.publication else "plots")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_analysis(input_root, analysis_dir, args.force_analysis)
    setup_style()
    if args.clean_output or args.publication:
        clean_output_folder(output_dir)
    else:
        clear_previous_figures(output_dir)

    data = add_derived_comparison_metrics(
        ordered_data(pd.read_csv(analysis_dir / "new_mc_per_seed.csv"))
    )
    summary = ordered_data(pd.read_csv(analysis_dir / "new_mc_summary.csv"))
    alpha_path = resolve_alpha_per_seed(args.alpha_per_seed)

    if set(data["cell"].dropna().astype(str)) != set(CELL_ORDER):
        raise ValueError("Analysis does not contain the expected four Monte Carlo cells")

    plot_comparison_suite(
        data,
        summary,
        output_dir,
        palettes=("viridis",) if args.publication else ("viridis", "plasma_shapes"),
        show_titles=not args.publication,
        publication=args.publication,
    )
    if args.publication:
        alpha_data = pd.DataFrame()
    else:
        alpha_data = latest_alpha_data(alpha_path)
        plot_latest_alpha_overview(alpha_data, output_dir)
        plot_alpha_gaussian_population(
            alpha_data,
            output_dir,
            x_metric="illuminated_images",
            y_metric="useful_downlinks_est",
            x_label="Illuminated images acquired",
            y_label="Useful images downlinked",
            title="Latest 100-seed alpha-sweep outcome populations",
            stem="alpha_sweep_gaussian_outcome_landscape_plasma_20260622",
            identity_line=True,
        )
        plot_alpha_gaussian_population(
            alpha_data,
            output_dir,
            x_metric="target_imaging_count",
            y_metric="downlink_action_count",
            x_label="Imaging actions",
            y_label="Downlink actions",
            title="Latest 100-seed alpha-sweep action populations",
            stem="alpha_sweep_gaussian_action_landscape_plasma_20260622",
        )
        plot_alpha_gaussian_population(
            alpha_data,
            output_dir,
            x_metric="avg_acquisition_time_sec",
            y_metric="acq_success_rate",
            x_label="Mean successful acquisition time [s]",
            y_label="Acquisition success fraction",
            title="Latest 100-seed alpha-sweep acquisition populations",
            stem="alpha_sweep_gaussian_acquisition_landscape_plasma_20260622",
        )
        plot_alpha_efficiency(alpha_data, output_dir)
        plot_alpha_metric_heatmap(alpha_data, output_dir)

    manifest = {
        "input_root": str(input_root),
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "alpha_colormap": ALPHA_COLORMAP,
        "comparison_colormaps": [
            COMPARISON_COLORMAP,
            COMPARISON_PLASMA_COLORMAP,
        ],
        "cells": CELL_ORDER,
        "seed_rows": int(len(data)),
        "latest_alpha_per_seed": str(alpha_path),
        "latest_alpha_seed_rows": int(len(alpha_data)),
        "publication": bool(args.publication),
        "show_titles": not args.publication,
        "figure_stems": sorted(path.stem for path in output_dir.glob("*.pdf")),
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Loaded seed rows: {len(data)}")
    print(f"Wrote figures: {output_dir}")


if __name__ == "__main__":
    main()
