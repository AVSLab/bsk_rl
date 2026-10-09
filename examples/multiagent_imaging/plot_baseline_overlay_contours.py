#!/usr/bin/env python3
"""Overlay independent and centralized agent contours, one figure per environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator

from plot_baseline_agent_contours import PLOT_SPECS, fit_gaussian

CASES = ("independent", "centralized_full_state")
SENSORS = ("sensor_0", "sensor_1", "sensor_2")
ENVIRONMENTS = ("leo", "mixed")
CASE_LABELS = {
    "independent": "Independent local greedy",
    "centralized_full_state": "Centralized full-state joint greedy",
}
CASE_CMAPS = {"independent": "viridis", "centralized_full_state": "plasma"}
CASE_SAMPLES = {
    "independent": (0.18, 0.45, 0.70),
    "centralized_full_state": (0.46, 0.65, 0.85),
}
ELLIPSE_COLORS = {
    "independent": "#175F78",
    "centralized_full_state": "#A52F66",
}
MARKERS = {"independent": "o", "centralized_full_state": "^"}


def palette() -> dict[tuple[str, str], str]:
    return {
        (case, sensor): to_hex(
            plt.get_cmap(CASE_CMAPS[case])(CASE_SAMPLES[case][index])
        )
        for case in CASES
        for index, sensor in enumerate(SENSORS)
    }


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 600:
        raise ValueError("Expected 600 sensor/seed observations")
    if (
        len({(r["environment"], r["case"], r["seed"], r["sensor"]) for r in rows})
        != 600
    ):
        raise ValueError("Duplicate sensor/seed/case row")
    for env in ENVIRONMENTS:
        for case in CASES:
            for sensor in SENSORS:
                subset = [
                    r
                    for r in rows
                    if r["environment"] == env
                    and r["case"] == case
                    and r["sensor"] == sensor
                ]
                if {int(r["seed"]) for r in subset} != set(range(50)):
                    raise ValueError(f"Incomplete seeds: {env}, {case}, {sensor}")
    return rows


def fitted_contours(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    color: str,
) -> None:
    inv = np.linalg.inv(covariance)
    residual = np.stack((xx - mean[0], yy - mean[1]), axis=-1)
    radius_sq = np.einsum("...i,ij,...j->...", residual, inv, residual)
    peak = 1 / (2 * math.pi * math.sqrt(np.linalg.det(covariance)))
    density = peak * np.exp(-radius_sq / 2)
    ax.contour(
        xx,
        yy,
        density,
        levels=[peak * 0.2, peak * 0.5],
        colors=[color, color],
        linewidths=[1.55, 1.1],
        linestyles=["solid", "dashed"],
        alpha=0.88,
        zorder=3,
    )


def three_sigma_ellipse(
    mean: np.ndarray, covariance: np.ndarray, color: str
) -> Ellipse:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    vector = eigenvectors[:, order[0]]
    return Ellipse(
        xy=mean,
        width=6 * math.sqrt(eigenvalues[order[0]]),
        height=6 * math.sqrt(eigenvalues[order[1]]),
        angle=math.degrees(math.atan2(vector[1], vector[0])),
        fill=False,
        edgecolor=color,
        linewidth=2.55,
        alpha=0.99,
        zorder=6,
    )


def fit_panel(rows: list[dict], env: str, spec: tuple) -> dict:
    _, _, x_key, _, y_key, _ = spec
    grouped = {
        (case, sensor): [
            r
            for r in rows
            if r["environment"] == env and r["case"] == case and r["sensor"] == sensor
        ]
        for case in CASES
        for sensor in SENSORS
    }
    agents = {}
    extents = []
    for key, subset in grouped.items():
        xy = np.array([[float(r[x_key]), float(r[y_key])] for r in subset], dtype=float)
        mean, covariance = fit_gaussian(xy)
        agents[key] = (xy, mean, covariance)
        radius = math.sqrt(-2 * math.log(0.2))
        extents.extend(
            (
                xy.min(axis=0),
                xy.max(axis=0),
                mean - radius * np.sqrt(np.diag(covariance)),
                mean + radius * np.sqrt(np.diag(covariance)),
            )
        )
    overall = {}
    for case in CASES:
        by_seed = defaultdict(list)
        for sensor in SENSORS:
            for row in grouped[case, sensor]:
                by_seed[int(row["seed"])].append((float(row[x_key]), float(row[y_key])))
        if set(by_seed) != set(range(50)) or any(
            len(values) != 3 for values in by_seed.values()
        ):
            raise ValueError("Incomplete three-sensor seed means")
        seed_means = np.array([np.mean(by_seed[seed], axis=0) for seed in range(50)])
        mean, covariance = fit_gaussian(seed_means)
        overall[case] = (mean, covariance)
        extents.extend(
            (
                mean - 3 * np.sqrt(np.diag(covariance)),
                mean + 3 * np.sqrt(np.diag(covariance)),
            )
        )
    extremes = np.array(extents)
    lo, hi = extremes.min(axis=0), extremes.max(axis=0)
    pad = np.maximum(0.07 * (hi - lo), 0.8)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])
    return {"agents": agents, "overall": overall, "xlim": xlim, "ylim": ylim}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 12.0,
            "axes.titlesize": 14,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.linewidth": 1.1,
            "savefig.dpi": 320,
        }
    )


def plot_environment(rows: list[dict], env: str, output: Path) -> dict:
    colors = palette()
    title = "LEO" if env == "leo" else "Mixed LEO/MEO/GEO"
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 14.7))
    all_stats = {}
    for index, (ax, spec) in enumerate(zip(axes, PLOT_SPECS)):
        stem, panel_title, x_key, x_label, y_key, y_label = spec
        fit = fit_panel(rows, env, spec)
        xx, yy = np.meshgrid(
            np.linspace(*fit["xlim"], 320),
            np.linspace(*fit["ylim"], 320),
        )
        agent_stats = {}
        for case in CASES:
            for sensor in SENSORS:
                xy, mean, covariance = fit["agents"][case, sensor]
                color = colors[case, sensor]
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    marker=MARKERS[case],
                    s=23 if case == "independent" else 26,
                    color=color,
                    alpha=0.43,
                    edgecolors="none",
                    zorder=2,
                )
                fitted_contours(ax, xx, yy, mean, covariance, color)
                ax.scatter(
                    *mean,
                    marker=MARKERS[case],
                    s=73 if case == "independent" else 82,
                    color=color,
                    edgecolor="#262626",
                    linewidth=0.7,
                    zorder=5,
                )
                agent_stats[f"{case}/{sensor}"] = {
                    "n": 50,
                    "mean": mean.tolist(),
                    "covariance": covariance.tolist(),
                    "color": color,
                }
        overall_stats = {}
        for case in CASES:
            mean, covariance = fit["overall"][case]
            ax.add_patch(three_sigma_ellipse(mean, covariance, ELLIPSE_COLORS[case]))
            ax.scatter(
                *mean,
                marker="D",
                s=64 if case == "centralized_full_state" else 70,
                color=ELLIPSE_COLORS[case],
                edgecolor="#202020",
                linewidth=0.75,
                zorder=7,
            )
            overall_stats[case] = {
                "n_seed_means": 50,
                "mean": mean.tolist(),
                "covariance": covariance.tolist(),
                "sigma": 3,
                "color": ELLIPSE_COLORS[case],
            }
        ax.set(
            xlim=fit["xlim"],
            ylim=fit["ylim"],
            xlabel=x_label,
            ylabel=y_label,
        )
        ax.set_title(
            f"{chr(ord('A') + index)}.  {panel_title}",
            loc="left",
            pad=9,
        )
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(True, alpha=0.23)
        ax.grid(True, which="minor", alpha=0.10)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        all_stats[stem] = {
            "x_key": x_key,
            "y_key": y_key,
            "agent_fits": agent_stats,
            "constellation_mean_fits": overall_stats,
            "xlim": fit["xlim"],
            "ylim": fit["ylim"],
        }

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[case, sensor],
            marker=MARKERS[case],
            markersize=7,
            linewidth=1.55,
            label=f"{'Independent' if case == 'independent' else 'Centralized'} · sensor {index}",
        )
        for case in CASES
        for index, sensor in enumerate(SENSORS)
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=ELLIPSE_COLORS[case],
            marker="D",
            markersize=6,
            linewidth=2.6,
            label=f"{'Independent' if case == 'independent' else 'Centralized'} · 3σ summary",
        )
        for case in CASES
    )
    fig.suptitle(
        f"{title}: independent and centralized information",
        fontsize=20,
        y=0.988,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.958),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        columnspacing=1.75,
        handletextpad=0.6,
    )
    fig.text(
        0.5,
        0.075,
        "Dots: all 50 observed seeds per sensor and case. Thin contours: Gaussian 50% dashed, 80% solid.",
        ha="center",
        fontsize=10.2,
        color="#484848",
    )
    fig.text(
        0.5,
        0.058,
        "Thick outlines: 3σ covariance ellipses of seed-wise three-sensor means (per-sensor scale); not confidence intervals.",
        ha="center",
        fontsize=10.2,
        color="#484848",
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.975,
        top=0.86,
        bottom=0.145,
        hspace=0.52,
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            output / f"overlay_{env}.{ext}",
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)
    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = read_rows(args.agent_csv)
    fits = {env: plot_environment(rows, env, args.output_dir) for env in ENVIRONMENTS}
    (args.output_dir / "overlay_fit_statistics.json").write_text(
        json.dumps(fits, indent=2) + "\n"
    )
    provenance = {
        "source_agent_csv": str(args.agent_csv.resolve()),
        "source_agent_csv_sha256": hashlib.sha256(
            args.agent_csv.read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_colormap": "viridis",
        "centralized_colormap": "plasma",
        "sensor_colormap_positions": CASE_SAMPLES,
        "seed_rows": len(rows),
        "figures": ["overlay_leo", "overlay_mixed"],
        "sigma": 3,
        "summary_population": "50 seed-wise means of three sensors per case and environment",
        "gaussian_contours_probability_mass": [0.5, 0.8],
        "ellipse_interpretation": "Three principal-axis standard deviations, not a confidence interval.",
    }
    (args.output_dir / "overlay_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
