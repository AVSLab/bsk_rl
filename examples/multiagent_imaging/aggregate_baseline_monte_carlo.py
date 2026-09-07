"""Validate and summarize the four-cell paired baseline Monte Carlo campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from examples.multiagent_imaging.baseline_monte_carlo import (
    CAMPAIGN_VERSION,
    CELLS,
    digest,
    task_spec,
    write_json,
)


def flatten(result):
    """Keep all episodes, including early sensor failures, in coverage denominators."""
    row = {
        key: result[key]
        for key in (
            "task_id",
            "case",
            "target_environment",
            "seed",
            "sim_time_s",
            "horizon_reached",
            "wall_time_s",
            "peak_process_rss_bytes",
            "simulated_seconds_per_wall_second",
            "total_constellation_reward",
        )
    }
    row.update(
        {
            key: result["coverage"][key]
            for key in (
                "capture_coverage_fraction",
                "ground_delivery_coverage_fraction",
                "capture_target_count",
                "ground_delivery_target_count",
                "qualified_exposure_count",
                "unqualified_exposure_count",
                "qualified_ground_delivery_count",
                "cross_sensor_capture_overlap_count",
            )
        }
    )
    row.update(
        {
            key: result["coordination"][key]
            for key in (
                "duplicate_sensor_time_s",
                "interrupted_nonduplicate_sensor_time_s",
                "wasted_time_fraction",
            )
        }
    )
    row["policy_decisions"] = sum(result["coordination"]["policy_decisions"].values())
    row["duplicate_attempt_count"] = result["team_summary"].get(
        "duplicate_attempt_count", 0
    )
    row["successful_duplicate_count"] = result["team_summary"].get(
        "successful_duplicate_count", 0
    )
    row["unique_acquisition_service_count"] = result["team_summary"].get(
        "unique_acquisition_count", 0
    )
    row["unique_ground_service_count"] = result["team_summary"].get(
        "unique_service_count", 0
    )
    row["radio_occupied_time_s"] = sum(
        result["coordination"]["communication_time_s"].values()
    )
    states = [
        state for series in result["resource_history"].values() for state in series
    ]
    row["minimum_battery_fraction"] = min(state["battery_fraction"] for state in states)
    row["maximum_storage_fraction"] = max(state["storage_fraction"] for state in states)
    row["maximum_wheel_fraction"] = max(state["max_wheel_fraction"] for state in states)
    row["sensors_alive_at_end"] = sum(
        series[-1]["alive"] for series in result["resource_history"].values()
    )
    return row


METRICS = (
    "capture_coverage_fraction",
    "ground_delivery_coverage_fraction",
    "total_constellation_reward",
    "qualified_exposure_count",
    "cross_sensor_capture_overlap_count",
    "duplicate_attempt_count",
    "duplicate_sensor_time_s",
    "interrupted_nonduplicate_sensor_time_s",
    "wasted_time_fraction",
    "policy_decisions",
    "wall_time_s",
    "peak_process_rss_bytes",
    "simulated_seconds_per_wall_second",
    "minimum_battery_fraction",
    "maximum_storage_fraction",
    "maximum_wheel_fraction",
)


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or len(values) == 0:
        raise ValueError("Metrics must contain finite observations.")
    if len(values) > 1:
        # Bootstrap independent initial-condition seeds, never individual actions.
        draws = (
            np.random.default_rng(20260906)
            .choice(values, (10000, len(values)), replace=True)
            .mean(axis=1)
        )
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
    else:
        interval = None
    return dict(
        n=len(values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        std=float(values.std(ddof=1)) if len(values) > 1 else None,
        min=float(values.min()),
        max=float(values.max()),
        bootstrap_mean_ci95=interval,
    )


def validate_results(manifest, results, allow_partial=False):
    if manifest["campaign_version"] != CAMPAIGN_VERSION:
        raise ValueError("Wrong campaign version.")
    seen = set()
    by_pair = defaultdict(dict)
    for result in results:
        task = result["task_id"]
        if task in seen:
            raise ValueError(f"Duplicate task {task}.")
        seen.add(task)
        if any(result[k] != v for k, v in task_spec(task).items()):
            raise ValueError(f"Task mapping mismatch: {task}.")
        if result["campaign_version"] != CAMPAIGN_VERSION or result[
            "manifest_sha256"
        ] != digest(manifest):
            raise ValueError("Episode uses another manifest/schema.")
        if result["baseline_config"] != manifest["baseline_config"]:
            raise ValueError("Episode configuration differs from the manifest.")
        if (
            result["source"]["source_fingerprint"]
            != manifest["source"]["source_fingerprint"]
        ):
            raise ValueError("Mixed source versions in campaign.")
        if result["initial_conditions_sha256"] != digest(result["initial_conditions"]):
            raise ValueError("Initial-condition fingerprint is invalid.")
        if (
            result["pettingzoo_agents"] != ["sensor_0", "sensor_1"]
            or result["passive_target_count"]
            != manifest["baseline_config"]["n_targets"]
        ):
            raise ValueError("Sensor/passive-target population changed.")
        if result["communication"]["radio_action_count"] != 0 or any(
            result["coordination"]["communication_time_s"].values()
        ):
            raise ValueError("No-radio baseline executed a radio task.")
        by_pair[(result["target_environment"], result["seed"])][result["case"]] = (
            result["initial_conditions_sha256"]
        )
    missing = sorted(set(range(200)) - seen)
    if missing and not allow_partial:
        raise ValueError(
            f"Missing {len(missing)} of 200 episodes. Use --allow-partial only for diagnostics."
        )
    pairs = 0
    for key, cases in by_pair.items():
        if len(cases) == 2:
            if len(set(cases.values())) != 1:
                raise ValueError(
                    f"Independent and centralized initial conditions differ for {key}."
                )
            pairs += 1
    return {
        "complete_campaign": not missing,
        "completed_episodes": len(seen),
        "missing_task_ids": missing,
        "validated_information_pairs": pairs,
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(manifest, results, output_dir, allow_partial=False):
    validation = validate_results(manifest, results, allow_partial)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted((flatten(result) for result in results), key=lambda r: r["task_id"])
    groups = {}
    for case, environment in CELLS:
        group = [
            row
            for row in rows
            if row["case"] == case and row["target_environment"] == environment
        ]
        if group:
            groups[f"{case}_{environment}"] = {
                metric: summarize([row[metric] for row in group]) for metric in METRICS
            }
            groups[f"{case}_{environment}"]["episodes_at_100_percent_capture"] = sum(
                row["capture_coverage_fraction"] == 1.0 for row in group
            )
            groups[f"{case}_{environment}"][
                "episodes_at_100_percent_ground_delivery"
            ] = sum(row["ground_delivery_coverage_fraction"] == 1.0 for row in group)
            groups[f"{case}_{environment}"]["horizon_reached_count"] = sum(
                row["horizon_reached"] for row in group
            )
    indexed = {
        (row["case"], row["target_environment"], row["seed"]): row for row in rows
    }
    differences = []
    for environment in ("leo", "mixed"):
        for seed in range(50):
            independent = indexed.get(("independent", environment, seed))
            central = indexed.get(("centralized_full_state", environment, seed))
            if independent is not None and central is not None:
                differences.append(
                    {
                        "target_environment": environment,
                        "seed": seed,
                        **{
                            metric: central[metric] - independent[metric]
                            for metric in METRICS
                        },
                    }
                )
    paired = {
        environment: {
            metric: summarize(
                [
                    row[metric]
                    for row in differences
                    if row["target_environment"] == environment
                ]
            )
            for metric in METRICS
        }
        for environment in ("leo", "mixed")
        if any(row["target_environment"] == environment for row in differences)
    }
    summary = {
        "campaign_version": CAMPAIGN_VERSION,
        "validation": validation,
        "groups": groups,
        "paired_central_minus_independent": paired,
        "uncertainty": "95% percentile bootstrap intervals over initial-condition seeds, 10000 resamples; one-seed intervals unavailable. Paired differences use identical seeds and initial conditions.",
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "episodes.csv", rows)
    write_csv(output_dir / "paired_differences.csv", differences)
    lines = [
        "# Two-sensor deterministic Monte Carlo baselines",
        "",
        f"Completed {validation['completed_episodes']}/200 episodes; verified {validation['validated_information_pairs']} matched information pairs.",
        "",
        "Coverage counts the union of distinct qualified targets over the 100-target mission catalog. Ground coverage additionally requires full physical downlink. Repeat services do not increase first-coverage percentages.",
        "",
        "| Cell | N | Mean capture coverage | Mean ground coverage | Capture 100% episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        c, g = (
            group["capture_coverage_fraction"],
            group["ground_delivery_coverage_fraction"],
        )
        lines.append(
            f"| {name} | {c['n']} | {100*c['mean']:.2f}% | {100*g['mean']:.2f}% | {group['episodes_at_100_percent_capture']} |"
        )
    lines.extend(
        [
            "",
            "See summary.json for seed uncertainty and paired central-minus-independent differences. Early terminations remain in these statistics. Geometric visibility is sampled at event boundaries and does not prove feasibility or infeasibility.",
            "",
            "These are deterministic heuristic information/control baselines. Centralized full-state access is an information advantage; this greedy controller is not a global optimality bound, and no 100% coverage claim is assumed.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def plot_diagnostics(results, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("capture_coverage_fraction", "ground_delivery_coverage_fraction"),
        ("Qualified capture", "Ground-confirmed delivery"),
    ):
        for index, (case, environment) in enumerate(CELLS):
            values = [
                100 * r["coverage"][metric]
                for r in results
                if r["case"] == case and r["target_environment"] == environment
            ]
            if values:
                ax.scatter(np.full(len(values), index), values, alpha=0.4, s=12)
                ax.scatter([index], [np.mean(values)], marker="_", s=160, color="black")
        ax.set_xticks(
            range(4), ["Indep.\nLEO", "Indep.\nmixed", "Central\nLEO", "Central\nmixed"]
        )
        ax.set_ylim(0, 102)
        ax.axhline(100, color="gray", linewidth=0.6)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Distinct targets / mission catalog (%)")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "coverage.png", dpi=160)
    fig.savefig(Path(output_dir) / "coverage.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    results = [
        json.loads(path.read_text())
        for path in sorted(args.episodes_dir.glob("episode_*.json"))
    ]
    summary = aggregate(manifest, results, args.output_dir, args.allow_partial)
    if args.plots:
        plot_diagnostics(results, args.output_dir)
    print(json.dumps(summary["validation"]))


if __name__ == "__main__":
    main()
