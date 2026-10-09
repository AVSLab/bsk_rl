#!/usr/bin/env python3
"""Per-agent Gaussian contour views of the preserved three-sensor baselines."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator

SENSORS = ("sensor_0", "sensor_1", "sensor_2")
ENVIRONMENTS = ("leo", "mixed")
CASES = ("independent", "centralized_full_state")
SENSOR_COLORS = {
    "sensor_0": "#F5A623",
    "sensor_1": "#31BB70",
    "sensor_2": "#B975F2",
}
CASE_COLORS = {
    "independent": "#3573B9",
    "centralized_full_state": "#B77E18",
}
CASE_LABELS = {
    "independent": "Independent local greedy",
    "centralized_full_state": "Centralized full-state joint greedy",
}
PLOT_SPECS = (
    (
        "images_vs_credited_visits",
        "Image effort versus useful delivered target visits",
        "qualifying_images",
        "Qualifying images taken per sensor",
        "credited_visits",
        "Credited target visits delivered per sensor",
    ),
    (
        "latency_vs_useful_yield",
        "Delivery delay versus useful image yield",
        "median_latency_min",
        "Median acquisition-to-ground delay per sensor [min]",
        "useful_yield_pct",
        "Credited visits per 100 qualifying images [%]",
    ),
    (
        "target_reach_vs_reimaging",
        "Distinct target reach versus credited reimaging",
        "credited_distinct_targets",
        "Distinct targets with credited delivery per sensor",
        "credited_reimages",
        "Credited repeat visits after cooldown per sensor",
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replay_one(episode: dict) -> tuple[list[dict], dict, int]:
    """Reconstruct physical delivery credit using the evaluator's ordering."""
    captures = episode["capture_records"]
    deliveries = episode["ground_delivery_records"]
    capture_by_id = {item["record_id"]: item for item in captures}
    if len(capture_by_id) != len(captures):
        raise ValueError("Duplicate capture record IDs")
    delivery_ids = [item["record_id"] for item in deliveries]
    if len(set(delivery_ids)) != len(delivery_ids):
        raise ValueError("Duplicate delivery record IDs")
    if not set(delivery_ids) <= set(capture_by_id):
        raise ValueError("Delivery without a capture record")

    qualified = defaultdict(list)
    delivered = defaultdict(list)
    for item in captures:
        if item["quality"] >= 0.5:
            qualified[item["source_sensor"]].append(item)
    for item in deliveries:
        prior = capture_by_id[item["record_id"]]
        if any(
            item[key] != prior[key]
            for key in ("target_id", "source_sensor", "capture_time", "completion_time")
        ):
            raise ValueError("Delivered product does not match capture")
        if item["quality"] >= 0.5:
            delivered[item["source_sensor"]].append(item)

    if (
        sum(map(len, qualified.values()))
        != episode["coverage"]["qualified_exposure_count"]
    ):
        raise ValueError("Qualified exposure count mismatch")

    by_boundary = defaultdict(list)
    for item in deliveries:
        by_boundary[item["delivery_time"]].append(item)
    last_capture = {}
    credited = defaultdict(list)
    reimages = defaultdict(int)
    simultaneous_credited = 0
    for _, boundary in sorted(by_boundary.items()):
        boundary.sort(
            key=lambda item: (
                item["target_id"],
                item["request_epoch"],
                item["capture_time"],
                item["source_sensor"],
                item["record_id"],
            )
        )
        index = 0
        while index < len(boundary):
            first = boundary[index]
            key = (first["target_id"], first["request_epoch"])
            group = [first]
            index += 1
            while index < len(boundary):
                other = boundary[index]
                if (other["target_id"], other["request_epoch"]) != key or abs(
                    other["capture_time"] - first["capture_time"]
                ) > 1e-6:
                    break
                group.append(other)
                index += 1
            qualifying_group = [item for item in group if item["quality"] >= 0.5]
            if not qualifying_group or (
                key in last_capture
                and first["capture_time"]
                < last_capture[key] + episode["reimage_cooldown_s"]
            ):
                continue
            simultaneous_credited += len(qualifying_group) > 1
            if len(qualifying_group) != 1:
                raise ValueError(
                    "A simultaneous credited group needs an explicit attribution rule"
                )
            product = qualifying_group[0]
            sensor = product["source_sensor"]
            if key in last_capture:
                reimages[sensor] += 1
            last_capture[key] = first["capture_time"]
            credited[sensor].append(product)

    credited_total = sum(map(len, credited.values()))
    if credited_total != episode["team_summary"]["unique_service_count"]:
        raise ValueError(f"Team delivery credit mismatch: {episode['task_id']}")
    rows = []
    for sensor in SENSORS:
        captured = qualified[sensor]
        ground = delivered[sensor]
        accepted = credited[sensor]
        if not captured or not ground:
            raise ValueError(f"Missing sensor observations: {sensor}")
        saved = episode["coverage"]["per_sensor"][sensor]
        if {p["target_id"] for p in captured} != set(saved["capture_target_ids"]):
            raise ValueError(f"Sensor capture coverage mismatch: {sensor}")
        if {p["target_id"] for p in ground} != set(saved["ground_delivery_target_ids"]):
            raise ValueError(f"Sensor ground coverage mismatch: {sensor}")
        latencies = [(p["delivery_time"] - p["completion_time"]) / 60 for p in ground]
        if min(latencies) < 0:
            raise ValueError("Negative physical delivery delay")
        row = {
            "task_id": episode["task_id"],
            "environment": episode["target_environment"],
            "case": episode["case"],
            "seed": episode["seed"],
            "sensor": sensor,
            "qualifying_images": len(captured),
            "physically_delivered_images": len(ground),
            "credited_visits": len(accepted),
            "credited_distinct_targets": len({p["target_id"] for p in accepted}),
            "credited_reimages": reimages[sensor],
            "median_latency_min": float(np.median(latencies)),
            "useful_yield_pct": 100 * len(accepted) / len(captured),
        }
        rows.append(row)
    if sum(row["credited_visits"] for row in rows) != credited_total:
        raise ValueError("Per-sensor credit sum mismatch")
    if sum(row["credited_reimages"] for row in rows) != (
        credited_total - len(last_capture)
    ):
        raise ValueError("Reimaging sum mismatch")
    team = {
        "task_id": episode["task_id"],
        "environment": episode["target_environment"],
        "case": episode["case"],
        "seed": episode["seed"],
        "qualifying_images": sum(r["qualifying_images"] for r in rows),
        "physically_delivered_images": sum(
            r["physically_delivered_images"] for r in rows
        ),
        "credited_visits": credited_total,
        "credited_reimages": sum(r["credited_reimages"] for r in rows),
        "duplicate_attempts": episode["team_summary"]["duplicate_attempt_count"],
        "ground_coverage_pct": 100
        * episode["coverage"]["ground_delivery_coverage_fraction"],
        "physical_delivery_pct": 100
        * sum(r["physically_delivered_images"] for r in rows)
        / sum(r["qualifying_images"] for r in rows),
        "useful_yield_pct": 100
        * credited_total
        / sum(r["qualifying_images"] for r in rows),
    }
    return rows, team, simultaneous_credited


def load_campaign(directory: Path) -> tuple[list[dict], list[dict], dict]:
    paths = sorted(directory.glob("episode_*.json"))
    if len(paths) != 200:
        raise ValueError(f"Expected 200 episodes; found {len(paths)}")
    agents = []
    teams = []
    input_hashes = {}
    pair_hashes = defaultdict(dict)
    source_fingerprints = set()
    simultaneous = 0
    for path in paths:
        content = path.read_bytes()
        episode = json.loads(content)
        task_id = episode["task_id"]
        if path.name != f"episode_{task_id:03}.json":
            raise ValueError(f"Task and filename mismatch: {path}")
        input_hashes[path.name] = sha256(content)
        if episode["pettingzoo_agents"] != list(SENSORS):
            raise ValueError("Unexpected sensor set")
        if episode["passive_target_count"] != 100:
            raise ValueError("Unexpected catalog size")
        if episode["baseline_config"]["n_candidates"] != 10:
            raise ValueError("Unexpected candidate count")
        if episode["baseline_config"]["reimage_cooldown_orbits"] != 2:
            raise ValueError("Unexpected cooldown policy")
        if episode["sim_time_s"] != 45000 or not episode["horizon_reached"]:
            raise ValueError("Incomplete episode")
        if episode["communication"]["radio_action_count"] != 0:
            raise ValueError("Radio action found")
        source_fingerprints.add(episode["source"]["source_fingerprint"])
        key = (episode["target_environment"], episode["seed"])
        if episode["case"] in pair_hashes[key]:
            raise ValueError(f"Duplicate information case for {key}")
        pair_hashes[key][episode["case"]] = episode["initial_conditions_sha256"]
        arows, team, n_simultaneous = replay_one(episode)
        agents.extend(arows)
        teams.append(team)
        simultaneous += n_simultaneous
    if {t["task_id"] for t in teams} != set(range(200)):
        raise ValueError("Task IDs are incomplete")
    for env in ENVIRONMENTS:
        for case in CASES:
            seeds = {
                t["seed"]
                for t in teams
                if t["environment"] == env and t["case"] == case
            }
            if seeds != set(range(50)):
                raise ValueError(f"Seed set mismatch: {env}, {case}")
    if len(pair_hashes) != 100 or any(
        set(hashes) != set(CASES) or len(set(hashes.values())) != 1
        for hashes in pair_hashes.values()
    ):
        raise ValueError("Initial-condition pairs do not match")
    if len(source_fingerprints) != 1:
        raise ValueError("Multiple historical source fingerprints")
    return (
        agents,
        teams,
        {
            "episode_sha256": input_hashes,
            "combined_input_sha256": sha256(
                json.dumps(input_hashes, sort_keys=True).encode()
            ),
            "historical_source_fingerprint": next(iter(source_fingerprints)),
            "simultaneous_credited_groups": simultaneous,
            "matched_pairs": len(pair_hashes),
        },
    )


def fit_gaussian(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = xy.mean(axis=0)
    cov = np.cov(xy, rowvar=False)
    if np.min(np.linalg.eigvalsh(cov)) <= 1e-8:
        raise ValueError("A Gaussian covariance is singular")
    return mean, cov


def probability_contour(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    color: str,
) -> None:
    """Plot actual bivariate-normal density contours at 50% and 80% mass."""
    inv = np.linalg.inv(cov)
    delta = np.stack((xx - mean[0], yy - mean[1]), axis=-1)
    sq = np.einsum("...i,ij,...j->...", delta, inv, delta)
    peak = 1 / (2 * math.pi * math.sqrt(np.linalg.det(cov)))
    density = peak * np.exp(-0.5 * sq)
    levels = [peak * 0.2, peak * 0.5]  # 80% and 50% enclosed mass in 2D
    ax.contour(
        xx,
        yy,
        density,
        levels=levels,
        colors=[color, color],
        linewidths=[1.55, 1.1],
        linestyles=["solid", "dashed"],
        alpha=0.84,
        zorder=3,
    )


def covariance_ellipse(mean: np.ndarray, cov: np.ndarray, color: str) -> Ellipse:
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vector = eigenvectors[:, order[0]]
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    return Ellipse(
        mean,
        width=4 * math.sqrt(eigenvalues[0]),
        height=4 * math.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=2.55,
        alpha=0.99,
        zorder=5,
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.1,
            "savefig.dpi": 320,
        }
    )


def plot_one(
    agents: list[dict],
    output: Path,
    spec: tuple[str, str, str, str, str, str],
) -> dict:
    stem, title, x_key, x_label, y_key, y_label = spec
    groups = {
        (env, case, sensor): [
            row
            for row in agents
            if row["environment"] == env
            and row["case"] == case
            and row["sensor"] == sensor
        ]
        for env in ENVIRONMENTS
        for case in CASES
        for sensor in SENSORS
    }
    fits = {}
    all_extents = []
    for (env, case, sensor), rows in groups.items():
        xy = np.array([[r[x_key], r[y_key]] for r in rows], dtype=float)
        if len(xy) != 50:
            raise ValueError("Expected exactly 50 seeds per sensor")
        mean, cov = fit_gaussian(xy)
        fits[env, case, sensor] = (xy, mean, cov)
        radius = np.sqrt(-2 * np.log(0.2))  # 80% Gaussian contour
        all_extents.extend(
            (
                mean - radius * np.sqrt(np.diag(cov)),
                mean + radius * np.sqrt(np.diag(cov)),
            )
        )
        all_extents.extend((xy.min(axis=0), xy.max(axis=0)))
    overall = {}
    for env in ENVIRONMENTS:
        for case in CASES:
            by_seed = defaultdict(list)
            for sensor in SENSORS:
                for row in groups[env, case, sensor]:
                    by_seed[row["seed"]].append((row[x_key], row[y_key]))
            if set(by_seed) != set(range(50)) or any(
                len(values) != 3 for values in by_seed.values()
            ):
                raise ValueError("Incomplete three-agent seed mean")
            means = np.array([np.mean(by_seed[seed], axis=0) for seed in range(50)])
            mean, cov = fit_gaussian(means)
            overall[env, case] = (mean, cov)
            all_extents.extend(
                (mean - 2 * np.sqrt(np.diag(cov)), mean + 2 * np.sqrt(np.diag(cov)))
            )
    limits = np.array(all_extents)
    low, high = limits.min(axis=0), limits.max(axis=0)
    pad = np.maximum((high - low) * 0.075, 0.75)
    xlim, ylim = (
        (low[0] - pad[0], high[0] + pad[0]),
        (
            low[1] - pad[1],
            high[1] + pad[1],
        ),
    )
    xx, yy = np.meshgrid(np.linspace(*xlim, 260), np.linspace(*ylim, 260))

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.0), sharex=True, sharey=True)
    for i, env in enumerate(ENVIRONMENTS):
        for j, case in enumerate(CASES):
            ax = axes[i, j]
            for sensor in SENSORS:
                xy, mean, cov = fits[env, case, sensor]
                color = SENSOR_COLORS[sensor]
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=21,
                    color=color,
                    alpha=0.43,
                    edgecolors="none",
                    zorder=2,
                )
                probability_contour(ax, xx, yy, mean, cov, color)
                ax.scatter(
                    *mean,
                    s=67,
                    marker="o",
                    facecolor=color,
                    edgecolor="#252525",
                    linewidth=0.8,
                    zorder=4,
                )
            mean, cov = overall[env, case]
            ax.add_patch(covariance_ellipse(mean, cov, CASE_COLORS[case]))
            ax.scatter(
                *mean,
                s=115,
                marker="D",
                facecolor=CASE_COLORS[case],
                edgecolor="#252525",
                linewidth=0.85,
                zorder=6,
            )
            ax.set(
                xlim=xlim,
                ylim=ylim,
                title=f"{env.upper() if env == 'leo' else 'Mixed LEO/MEO/GEO'}  ·  {CASE_LABELS[case]}",
            )
            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.grid(True, alpha=0.23)
            ax.grid(True, which="minor", alpha=0.10)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            if i == 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(y_label)
    fig.suptitle(title, fontsize=20, y=0.984)
    fig.text(
        0.5,
        0.946,
        "Three deterministic sensors · 50 matched seeds per case/environment · 45,000-second mission",
        ha="center",
        color="#4d4d4d",
        fontsize=11.5,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=SENSOR_COLORS[sensor],
            marker="o",
            linestyle="-",
            linewidth=1.4,
            markersize=6,
            label=f"Sensor {index}: seeds + Gaussian contours",
        )
        for index, sensor in enumerate(SENSORS)
    ]
    handles += [
        Line2D(
            [0],
            [0],
            color=CASE_COLORS[case],
            marker="D",
            linestyle="-",
            linewidth=2.5,
            markersize=7,
            label=f"{'Independent' if case == 'independent' else 'Centralized'}: 2σ mean ellipse",
        )
        for case in CASES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    fig.text(
        0.5,
        0.033,
        "Dots = observed sensor/seed outcomes. Thin contours = fitted 50% (dashed) and 80% (solid) Gaussian regions."
        "  Thick ellipse = 2σ spread of 50 seed-wise constellation means per sensor.",
        ha="center",
        fontsize=9.4,
        color="#4b4b4b",
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, top=0.88, bottom=0.16, hspace=0.31, wspace=0.16
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            output / f"{stem}.{ext}", dpi=320, bbox_inches="tight", pad_inches=0.08
        )
    plt.close(fig)
    return {
        "stem": stem,
        "x_metric": x_key,
        "y_metric": y_key,
        "sensor_gaussians": {
            f"{env}/{case}/{sensor}": {
                "n": 50,
                "mean": fits[env, case, sensor][1].tolist(),
                "covariance": fits[env, case, sensor][2].tolist(),
            }
            for env in ENVIRONMENTS
            for case in CASES
            for sensor in SENSORS
        },
        "overall_three_sensor_mean_ellipses": {
            f"{env}/{case}": {
                "n": 50,
                "mean": overall[env, case][0].tolist(),
                "covariance": overall[env, case][1].tolist(),
                "principal_axis_sigma": np.sqrt(
                    np.linalg.eigvalsh(overall[env, case][1])
                ).tolist(),
            }
            for env in ENVIRONMENTS
            for case in CASES
        },
        "xlim": xlim,
        "ylim": ylim,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    agents, teams, evidence = load_campaign(args.episodes_dir)
    if len(agents) != 600 or len(teams) != 200:
        raise ValueError("Unexpected observation count")
    write_csv(args.output_dir / "agent_seed_metrics.csv", agents)
    write_csv(args.output_dir / "constellation_seed_metrics.csv", teams)
    fits = [plot_one(agents, args.output_dir, spec) for spec in PLOT_SPECS]
    summary = {}
    for env in ENVIRONMENTS:
        for case in CASES:
            rows = [
                row
                for row in teams
                if row["environment"] == env and row["case"] == case
            ]
            summary[f"{env}/{case}"] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in (
                    "qualifying_images",
                    "physically_delivered_images",
                    "credited_visits",
                    "credited_reimages",
                    "duplicate_attempts",
                    "ground_coverage_pct",
                    "physical_delivery_pct",
                    "useful_yield_pct",
                )
            }
    (args.output_dir / "fit_statistics.json").write_text(
        json.dumps({"plots": fits, "team_means": summary}, indent=2) + "\n"
    )
    source = Path(__file__).read_bytes()
    style_reference = (
        Path(__file__).resolve().parents[1]
        / "amos_2026"
        / "plot_amos2026_gat_research_figures.py"
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                **evidence,
                "plot_script_sha256": sha256(source),
                "style_reference": str(style_reference),
                "style_reference_sha256": sha256(style_reference.read_bytes()),
                "python_numpy_matplotlib": {
                    "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__,
                },
                "definitions": {
                    "credited_visit": "One quality-qualified physical product earning ground credit under target/request-epoch/capture-anchored cooldown accounting.",
                    "credited_reimage": "A credited visit to a target/request epoch with an earlier credited visit in the same episode.",
                    "useful_yield_pct": "100 times sensor credited visits divided by sensor qualified physical capture records.",
                    "median_latency_min": "Median among delivered qualified sensor products of delivery time minus exposure completion time, in minutes.",
                    "gaussian_contours": "Bivariate normal fit to 50 observed sensor seed outcomes; 50% and 80% enclosed probability mass.",
                    "overall_2sigma": "Covariance ellipse with principal radii 2 standard deviations, fitted to 50 seed-wise means of the three agent x/y observations. Not a confidence interval or a 95% region.",
                },
            },
            indent=2,
        )
        + "\n"
    )
    verification = {
        "status": "PASS",
        "episodes": len(teams),
        "matched_pairs": evidence["matched_pairs"],
        "agent_seed_rows": len(agents),
        "simultaneous_credited_groups": evidence["simultaneous_credited_groups"],
        "all_team_credits_recomputed": True,
        "all_sensor_coverage_recomputed": True,
        "horizon_s": 45000,
        "figures": [spec[0] for spec in PLOT_SPECS],
        "formats": ["png", "pdf"],
    }
    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps({"verification": verification, "team_means": summary}, indent=2))


if __name__ == "__main__":
    main()
