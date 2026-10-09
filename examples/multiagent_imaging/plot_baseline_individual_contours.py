#!/usr/bin/env python3
"""Six separate, matched-seed sensor comparisons: three metrics by two scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from plot_baseline_agent_contours import PLOT_SPECS
from plot_baseline_overlay_contours import (
    CASES,
    CASE_LABELS,
    ELLIPSE_COLORS,
    ENVIRONMENTS,
    MARKERS,
    SENSORS,
    fit_panel,
    fitted_contours,
    palette,
    read_rows,
    setup_style,
    three_sigma_ellipse,
)


def panel_limits(fit: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    xlow, xhigh = fit["xlim"]
    ylow, yhigh = fit["ylim"]
    # Reserve the upper part of each data axis for the three-row legend.
    return (xlow, xhigh), (ylow, yhigh + 0.16 * (yhigh - ylow))


def plot_one(env: str, spec: tuple, fit: dict,
             limits: tuple, output_dir: Path) -> dict:
    stem, title, x_key, x_label, y_key, y_label = spec
    colors = palette()
    xlim, ylim = limits
    xx, yy = np.meshgrid(np.linspace(*xlim, 360), np.linspace(*ylim, 360))
    fig, ax = plt.subplots(figsize=(11.8, 7.8))
    agent_stats = {}
    for case in CASES:
        for sensor in SENSORS:
            xy, mean, covariance = fit["agents"][case, sensor]
            color = colors[case, sensor]
            ax.scatter(
                xy[:, 0], xy[:, 1], marker=MARKERS[case],
                s=25 if case == "independent" else 28,
                color=color, alpha=0.43, edgecolors="none", zorder=2,
            )
            fitted_contours(ax, xx, yy, mean, covariance, color)
            ax.scatter(
                *mean, marker=MARKERS[case],
                s=75 if case == "independent" else 82,
                color=color, edgecolor="#242424", linewidth=0.7, zorder=5,
            )
            agent_stats[f"{case}/{sensor}"] = {
                "n": 50, "mean": mean.tolist(),
                "covariance": covariance.tolist(), "color": color,
            }
    overall_stats = {}
    for case in CASES:
        mean, covariance = fit["overall"][case]
        ax.add_patch(three_sigma_ellipse(mean, covariance, ELLIPSE_COLORS[case]))
        ax.scatter(
            *mean, marker="D", s=65, color=ELLIPSE_COLORS[case],
            edgecolor="#202020", linewidth=0.75, zorder=7,
        )
        overall_stats[case] = {
            "n_seed_means": 50, "mean": mean.tolist(),
            "covariance": covariance.tolist(), "sigma": 3,
            "color": ELLIPSE_COLORS[case],
        }
    handles = [
        Line2D(
            [0], [0], color=colors[case, sensor], marker=MARKERS[case],
            markersize=7, linewidth=1.5,
            label=f"{'Independent' if case == 'independent' else 'Centralized'} · sensor {index}",
        )
        for case in CASES for index, sensor in enumerate(SENSORS)
    ]
    handles += [
        Line2D(
            [0], [0], color=ELLIPSE_COLORS[case], marker="D",
            markersize=6, linewidth=2.5,
            label=f"{'Independent' if case == 'independent' else 'Centralized'} · 3σ summary",
        )
        for case in CASES
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, fontsize=10.0, columnspacing=1.6,
        handletextpad=0.55,
    )
    scene = "LEO" if env == "leo" else "Mixed LEO/MEO/GEO"
    ax.set(
        xlim=xlim, ylim=ylim, xlabel=x_label, ylabel=y_label,
        title=f"{scene}: {title}",
    )
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, alpha=0.23)
    ax.grid(True, which="minor", alpha=0.10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    fig.text(
        0.5, 0.045,
        "Points: all 50 matched seeds per sensor and case. Thin Gaussian contours: 50% dashed, 80% solid.",
        ha="center", fontsize=9.5, color="#484848",
    )
    fig.text(
        0.5, 0.023,
        "Thick outlines: 3σ covariance ellipses of seed-wise three-sensor means; descriptive, not confidence intervals.",
        ha="center", fontsize=9.5, color="#484848",
    )
    fig.subplots_adjust(left=0.11, right=0.975, top=0.92, bottom=0.16)
    name = f"{stem}_{env}"
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=320,
                    bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {
        "environment": env, "x_metric": x_key, "y_metric": y_key,
        "xlim": xlim, "ylim": ylim,
        "agent_fits": agent_stats, "constellation_mean_fits": overall_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = read_rows(args.agent_csv)
    results = {}
    for spec in PLOT_SPECS:
        fits = {env: fit_panel(rows, env, spec) for env in ENVIRONMENTS}
        for env in ENVIRONMENTS:
            results[f"{spec[0]}_{env}"] = plot_one(
                env, spec, fits[env], panel_limits(fits[env]), args.output_dir
            )
    (args.output_dir / "fit_statistics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    provenance = {
        "source_agent_csv": str(args.agent_csv.resolve()),
        "source_agent_csv_sha256": hashlib.sha256(args.agent_csv.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_sensor_seed_rows": len(rows), "figures": list(results),
        "independent_colormap": "viridis", "centralized_colormap": "plasma",
        "sensor_marker": MARKERS, "summary_sigma": 3,
        "axis_limits": "fitted separately per environment to keep seed and contour structure legible",
        "controller_labels": CASE_LABELS,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps({"status": "PASS", **provenance}, indent=2))


if __name__ == "__main__":
    main()
