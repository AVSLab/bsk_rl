"""Validate and summarize the four-cell, three-sensor baseline campaign."""

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
            key: result["product_duplicates"][key]
            for key in (
                "stale_cross_sensor_ground_delivery_count",
                "stale_cross_sensor_ground_delivery_target_count",
                "causally_avoidable_stale_ground_delivery_count",
                "cross_sensor_onboard_overlap_target_count",
                "cross_sensor_onboard_overlap_product_count",
                "cross_sensor_onboard_redundant_acquisition_count",
                "cross_sensor_onboard_redundant_sensor_time_s",
                "cross_sensor_onboard_overlap_sensor_time_s",
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
    # The prospectus waste numerator combines time spent finishing an already
    # duplicated target with useful tasks that were interrupted by retasking.
    # Preserve both components above and expose their sum for plots and tables.
    row["wasted_sensor_time_s"] = (
        row["duplicate_sensor_time_s"] + row["interrupted_nonduplicate_sensor_time_s"]
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
    row["successful_duplicate_count"] = result["team_summary"].get(
        "successful_duplicate_count", 0
    )
    # The baseline uses the identity reward function. Keep its two positive
    # mission-value terms explicit so plots do not hide them inside one total.
    # Any operational or duplicate adjustment remains a separate diagnostic.
    alpha = float(result["baseline_config"]["alpha"])
    row["acquisition_reward_component"] = (1.0 - alpha) * float(
        result["team_summary"].get("team_acquisition_value", 0)
    )
    row["ground_delivery_reward_component"] = alpha * float(
        result["team_summary"].get("team_value", 0)
    )
    adjustment = float(result["total_constellation_reward"]) - (
        row["acquisition_reward_component"] + row["ground_delivery_reward_component"]
    )
    row["reward_adjustment_component"] = 0.0 if abs(adjustment) < 1e-10 else adjustment
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
    "acquisition_reward_component",
    "ground_delivery_reward_component",
    "reward_adjustment_component",
    "qualified_exposure_count",
    "unique_acquisition_service_count",
    "unique_ground_service_count",
    "successful_duplicate_count",
    "cross_sensor_capture_overlap_count",
    "stale_cross_sensor_ground_delivery_count",
    "stale_cross_sensor_ground_delivery_target_count",
    "causally_avoidable_stale_ground_delivery_count",
    "cross_sensor_onboard_overlap_target_count",
    "cross_sensor_onboard_overlap_product_count",
    "cross_sensor_onboard_redundant_acquisition_count",
    "cross_sensor_onboard_redundant_sensor_time_s",
    "cross_sensor_onboard_overlap_sensor_time_s",
    "duplicate_attempt_count",
    "duplicate_sensor_time_s",
    "interrupted_nonduplicate_sensor_time_s",
    "wasted_sensor_time_s",
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


def summarize_seed_blocked(rows, metric):
    """Pool environments while resampling the shared seed as one block.

    LEO and mixed reuse seed IDs. Their effects therefore give 100 paired
    observations, but only 50 independently sampled sensing-team realizations.
    Keeping both environments together whenever a seed is redrawn avoids a
    falsely narrow interval from treating all 100 effects as independent.
    """
    statistics = summarize([row[metric] for row in rows])
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(float(row[metric]))
    seed_blocks = np.asarray(
        [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=float
    )
    if len(seed_blocks) > 1:
        draws = (
            np.random.default_rng(20260906)
            .choice(seed_blocks, (10000, len(seed_blocks)), replace=True)
            .mean(axis=1)
        )
        statistics["bootstrap_mean_ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    statistics["n_seed_blocks"] = len(seed_blocks)
    statistics["bootstrap_unit"] = "seed block containing LEO and mixed effects"
    return statistics


def build_metric_audit(results, rows):
    """Explain reward composition and finite-horizon coverage shortfalls.

    Geometry counters are sampled only when the asynchronous environment returns
    control. They diagnose the saved episodes but do not claim continuous-time
    reachability. Reward adjustments expose every term outside the two positive
    priority-value components used by this campaign.
    """
    reward_adjustments = [row["reward_adjustment_component"] for row in rows]
    agent_rewards = [
        float(value)
        for result in results
        for value in result["cumulative_reward"].values()
    ]
    opportunity = {}
    for case, environment in CELLS:
        episodes = [
            result
            for result in results
            if result["case"] == case and result["target_environment"] == environment
        ]
        diagnostics = []
        for result in episodes:
            missing = result["coverage"]["never_captured_target_ids"]
            geometry = result["sampled_target_geometry"]
            diagnostics.append(
                {
                    "missing_target_count": len(missing),
                    "missing_with_zero_event_boundary_illuminated_los_samples": sum(
                        geometry[str(target_id)]["illuminated_los_samples"] == 0
                        for target_id in missing
                    ),
                    "missing_with_positive_event_boundary_illuminated_los_samples": sum(
                        geometry[str(target_id)]["illuminated_los_samples"] > 0
                        for target_id in missing
                    ),
                    "missing_with_zero_candidate_samples": sum(
                        geometry[str(target_id)]["candidate_samples"] == 0
                        for target_id in missing
                    ),
                    "missing_with_positive_candidate_samples": sum(
                        geometry[str(target_id)]["candidate_samples"] > 0
                        for target_id in missing
                    ),
                }
            )
        if diagnostics:
            opportunity[f"{case}_{environment}"] = {
                metric: summarize([row[metric] for row in diagnostics])
                for metric in diagnostics[0]
            }
    return {
        "reward": {
            "formula": (
                "total = 0.9 * team acquisition value + 0.1 * team "
                "ground-delivery value + reward adjustments"
            ),
            "maximum_absolute_reward_adjustment": max(
                (abs(value) for value in reward_adjustments), default=0.0
            ),
            "negative_agent_cumulative_reward_count": sum(
                value < 0 for value in agent_rewards
            ),
            "minimum_agent_cumulative_reward": min(agent_rewards, default=0.0),
            "interpretation": (
                "The two value components are nonnegative. Reward adjustments "
                "contain configured operational, duplicate, or communication "
                "penalties; they were exactly zero throughout this campaign."
            ),
        },
        "missing_target_opportunity": opportunity,
        "opportunity_limit": (
            "Illuminated-LOS and candidate counters are sampled at asynchronous "
            "decision boundaries, not continuously in simulation time."
        ),
    }


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
        expected_agents = [
            f"sensor_{index}"
            for index in range(manifest["baseline_config"]["n_sensors"])
        ]
        if (
            result["pettingzoo_agents"] != expected_agents
            or result["passive_target_count"]
            != manifest["baseline_config"]["n_targets"]
        ):
            raise ValueError("Sensor/passive-target population changed.")
        if result["communication"]["radio_action_count"] != 0 or any(
            result["coordination"]["communication_time_s"].values()
        ):
            raise ValueError("No-radio baseline executed a radio task.")
        audit = result["centralized_information_audit"]
        centralized = result["case"] == "centralized_full_state"
        if bool(audit["enabled"]) != centralized:
            raise ValueError("Centralized information-audit mode is inconsistent.")
        if centralized and (
            audit["decision_boundaries"] != result["event_steps"]
            or audit["sensor_state_reads"] < audit["decision_boundaries"]
            or audit["last_snapshot_sha256"] is None
        ):
            raise ValueError(
                "Central controller did not audit every decision boundary."
            )
        if not centralized and (
            audit["decision_boundaries"] or audit["sensor_state_reads"]
        ):
            raise ValueError("Independent controller read centralized state.")
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
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
    pooled_paired = (
        {metric: summarize_seed_blocked(differences, metric) for metric in METRICS}
        if differences
        else {}
    )
    summary = {
        "campaign_version": CAMPAIGN_VERSION,
        "validation": validation,
        "groups": groups,
        "paired_central_minus_independent": paired,
        "paired_central_minus_independent_pooled": pooled_paired,
        "metric_audit": build_metric_audit(results, rows),
        "uncertainty": (
            "95% percentile bootstrap intervals, 10000 resamples. Cell and "
            "environment-specific paired intervals resample initial-condition seeds. "
            "The pooled result contains 100 paired environment observations but "
            "resamples 50 seed blocks so each draw keeps a seed's LEO and mixed "
            "effects together. One-seed intervals are unavailable."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "episodes.csv", rows)
    write_csv(output_dir / "paired_differences.csv", differences)
    sensor_count = manifest["baseline_config"]["n_sensors"]
    target_count = manifest["baseline_config"]["n_targets"]
    lines = [
        f"# {sensor_count}-sensor deterministic Monte Carlo baselines",
        "",
        f"Completed {validation['completed_episodes']}/200 episodes; verified {validation['validated_information_pairs']} matched information pairs.",
        "",
        f"Coverage counts the union of distinct qualified targets over the {target_count}-target mission catalog. Ground coverage additionally requires full physical downlink. Repeat services do not increase first-coverage percentages. Stale ground deliveries and simultaneous cross-sensor onboard ownership are reported separately.",
        "",
        "| Cell | N | Mean capture coverage | Mean ground coverage | Unique acquisition services | Unique ground services | Successful duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        c, g = (
            group["capture_coverage_fraction"],
            group["ground_delivery_coverage_fraction"],
        )
        lines.append(
            f"| {name} | {c['n']} | {100*c['mean']:.2f}% | {100*g['mean']:.2f}% | "
            f"{group['unique_acquisition_service_count']['mean']:.2f} | "
            f"{group['unique_ground_service_count']['mean']:.2f} | "
            f"{group['successful_duplicate_count']['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Matched information effects",
            "",
            "Each effect is centralized minus independent for an identical environment/seed initial condition. The pooled row contains 100 paired observations in 50 seed blocks; its bootstrap keeps each seed's LEO and mixed effects together.",
            "",
            "| Environment | Pairs | Capture coverage | Ground coverage | Unique acquisitions | Unique ground services | Reward | Duplicate attempts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for environment in ("leo", "mixed"):
        if environment not in paired:
            continue
        effect = paired[environment]
        lines.append(
            f"| {environment.upper()} | {effect['capture_coverage_fraction']['n']} | "
            f"{100 * effect['capture_coverage_fraction']['mean']:+.2f} pp | "
            f"{100 * effect['ground_delivery_coverage_fraction']['mean']:+.2f} pp | "
            f"{effect['unique_acquisition_service_count']['mean']:+.2f} | "
            f"{effect['unique_ground_service_count']['mean']:+.2f} | "
            f"{effect['total_constellation_reward']['mean']:+.2f} | "
            f"{effect['duplicate_attempt_count']['mean']:+.2f} |"
        )
    if pooled_paired:
        lines.append(
            f"| Pooled, seed-blocked | {pooled_paired['capture_coverage_fraction']['n']} | "
            f"{100 * pooled_paired['capture_coverage_fraction']['mean']:+.2f} pp | "
            f"{100 * pooled_paired['ground_delivery_coverage_fraction']['mean']:+.2f} pp | "
            f"{pooled_paired['unique_acquisition_service_count']['mean']:+.2f} | "
            f"{pooled_paired['unique_ground_service_count']['mean']:+.2f} | "
            f"{pooled_paired['total_constellation_reward']['mean']:+.2f} | "
            f"{pooled_paired['duplicate_attempt_count']['mean']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "See summary.json for all intervals and paired metrics. Early terminations remain in these statistics. Geometric visibility is sampled at event boundaries and does not prove feasibility or infeasibility.",
            "",
            "These are deterministic heuristic information/control baselines. Centralized full-state access is an information advantage; this greedy controller is not a global optimality bound, and no 100% coverage claim is assumed.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def coverage_timeline(result, *, ground=False):
    """Count each qualified target once, when its physical product is available.

    Exposure completion (after the hold) makes a qualified capture available;
    full delivery makes it available on the ground. Catalog receipt times and
    repeated services must not move either physical coverage curve.
    """
    records = result["ground_delivery_records" if ground else "capture_records"]
    timestamp = "delivery_time" if ground else "completion_time"
    first = {}
    for record in records:
        time = record[timestamp]
        if record["quality"] >= 0.5 and time is not None:
            target = record["target_id"]
            first[target] = min(first.get(target, float("inf")), float(time))
    expected = result["coverage"][
        "ground_delivery_target_count" if ground else "capture_target_count"
    ]
    if len(first) != expected:
        raise ValueError("Physical event records do not reproduce endpoint coverage.")
    times = np.asarray([0.0, *sorted(first.values()), result["sim_time_s"]])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("Coverage timestamps must lie within the simulated episode.")
    values = np.asarray([0, *range(1, len(first) + 1), len(first)], dtype=float)
    return times, 100 * values / result["coverage"]["catalog_target_count"]


def plot_diagnostics(results, output_dir):
    """Export paired coverage points and individual-seed physical step curves.

    Chart contract: compare information cases at matched initial-condition seeds,
    on a fixed 0–100% mission-catalog denominator. Endpoint panels show every
    observed seed, paired connectors, and means; the subtitle discloses missing
    campaign cells. Small preflights additionally show step curves by seed using
    recorded completion/delivery times. Blue/gold plus solid/dashed lines identify
    the cases without relying on color alone. PNG/PDF are the research artifacts.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No baseline episodes are available to plot.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = ("independent", "centralized_full_state")
    labels = {"independent": "Independent", "centralized_full_state": "Centralized"}
    present = [
        (case, environment)
        for environment in ("leo", "mixed")
        for case in cases
        if any(
            r["case"] == case and r["target_environment"] == environment
            for r in results
        )
    ]
    first = results[0]
    config = first["baseline_config"]
    context = (
        f"{config['n_sensors']} sensors · {config['n_targets']} passive targets · "
        f"{config['episode_duration_s']:,.0f} s episodes · {len(results)}/200 episodes"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharey=True)
    fig.suptitle("Baseline coverage comparison", x=0.08, ha="left", fontsize=16)
    fig.text(0.08, 0.91, context, fontsize=10, color="#444444")
    for ax, metric, title in zip(
        axes,
        ("capture_coverage_fraction", "ground_delivery_coverage_fraction"),
        ("Qualified capture", "Ground-confirmed delivery"),
    ):
        for index, (case, environment) in enumerate(present):
            values = [
                100 * r["coverage"][metric]
                for r in results
                if r["case"] == case and r["target_environment"] == environment
            ]
            if values:
                ax.scatter(
                    np.full(len(values), index),
                    values,
                    alpha=0.6,
                    s=32,
                    facecolors="none",
                    edgecolors="#3573B9",
                    zorder=3,
                )
                ax.scatter(
                    [index],
                    [np.mean(values)],
                    marker="_",
                    s=200,
                    color="#222222",
                    zorder=4,
                )
                ax.annotate(
                    f"{np.mean(values):.1f}%",
                    (index, np.mean(values)),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        # Thin connectors represent paired seeds, not trends or confidence bounds.
        for environment in ("leo", "mixed"):
            if all((case, environment) in present for case in cases):
                x = [present.index((case, environment)) for case in cases]
                indexed = {
                    (r["case"], r["seed"]): r
                    for r in results
                    if r["target_environment"] == environment
                }
                for seed in sorted({r["seed"] for r in indexed.values()}):
                    if all((case, seed) in indexed for case in cases):
                        ax.plot(
                            x,
                            [
                                100 * indexed[case, seed]["coverage"][metric]
                                for case in cases
                            ],
                            color="#AAAAAA",
                            lw=0.7,
                            zorder=1,
                        )
        ax.set_xticks(
            range(len(present)),
            [
                f"{labels[case]}\n{environment.upper()} · n="
                f"{sum(r['case'] == case and r['target_environment'] == environment for r in results)}"
                for case, environment in present
            ],
        )
        ax.set_xlim(-0.5, len(present) - 0.5)
        ax.set_ylim(0, 109)
        ax.set_yticks(range(0, 101, 20))
        ax.axhline(100, color="#666666", linewidth=0.7, linestyle=":")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Distinct targets / mission catalog (%)")
    note = "Circles: individual seeds; bars: means; connectors: matched seeds."
    if len(results) < 200:
        note += " Partial campaign; missing episodes are not zeros."
    fig.text(0.08, 0.035, note, fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.07, 1, 0.87))
    fig.savefig(output_dir / "coverage.png", dpi=180)
    fig.savefig(output_dir / "coverage.pdf")
    plt.close(fig)

    # Keep the two user-defined duplicate concepts separate.  The left panel is
    # retrospective data freshness at ground; the right panel is coverage waste
    # caused by different spacecraft physically holding the same target product.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    fig.suptitle("Baseline duplicate-work comparison", x=0.08, ha="left", fontsize=16)
    fig.text(0.08, 0.91, context, fontsize=10, color="#444444")
    duplicate_panels = (
        (
            "stale_cross_sensor_ground_delivery_count",
            "Older products delivered to ground",
            "Qualified product count",
        ),
        (
            "cross_sensor_onboard_redundant_acquisition_count",
            "Redundant cross-sensor onboard acquisitions",
            "Qualified acquisition count",
        ),
    )
    case_colors = {"independent": "#3573B9", "centralized_full_state": "#A87819"}
    for ax, (metric, title, ylabel) in zip(axes, duplicate_panels):
        for index, (case, environment) in enumerate(present):
            values = [
                result["product_duplicates"][metric]
                for result in results
                if result["case"] == case
                and result["target_environment"] == environment
            ]
            ax.scatter(
                np.full(len(values), index),
                values,
                alpha=0.65,
                s=32,
                facecolors="none",
                edgecolors=case_colors[case],
                zorder=3,
            )
            if values:
                ax.scatter(
                    [index],
                    [np.mean(values)],
                    marker="_",
                    s=220,
                    color="#222222",
                    zorder=4,
                )
        for environment in ("leo", "mixed"):
            if all((case, environment) in present for case in cases):
                x = [present.index((case, environment)) for case in cases]
                indexed = {
                    (result["case"], result["seed"]): result
                    for result in results
                    if result["target_environment"] == environment
                }
                for seed in sorted({key[1] for key in indexed}):
                    if all((case, seed) in indexed for case in cases):
                        ax.plot(
                            x,
                            [
                                indexed[case, seed]["product_duplicates"][metric]
                                for case in cases
                            ],
                            color="#AAAAAA",
                            lw=0.7,
                            zorder=1,
                        )
        ax.set_xticks(
            range(len(present)),
            [f"{labels[case]}\n{environment.upper()}" for case, environment in present],
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.08,
        0.035,
        "Circles: individual seeds; bars: means; connectors: matched seeds. Onboard count uses overlapping physical-storage intervals.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.87))
    fig.savefig(output_dir / "duplicates.png", dpi=180)
    fig.savefig(output_dir / "duplicates.pdf")
    plt.close(fig)

    def values_for(case, environment, source, metric, scale=1.0):
        """Read one metric at the episode grain for a case/environment cell."""
        values = []
        for result in results:
            if result["case"] != case or result["target_environment"] != environment:
                continue
            if source == "flat":
                value = flatten(result)[metric]
            else:
                value = result[source][metric]
            values.append(float(value) * scale)
        return values

    def plot_metric_grid(filename, title, panels, note):
        """Plot individual seeds and cell means without hiding the distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
        fig.suptitle(title, x=0.07, ha="left", fontsize=16)
        fig.text(0.07, 0.935, context, fontsize=10, color="#444444")
        for ax, panel in zip(axes.flat, panels):
            source, metric, panel_title, ylabel, scale = panel
            for index, (case, environment) in enumerate(present):
                values = values_for(case, environment, source, metric, scale)
                # Deterministic horizontal offsets reveal coincident integer-valued
                # seeds while keeping the x position tied to the information cell.
                offsets = (
                    np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else [0]
                )
                ax.scatter(
                    index + np.asarray(offsets),
                    values,
                    alpha=0.55,
                    s=23,
                    facecolors="none",
                    edgecolors=case_colors[case],
                    zorder=3,
                )
                if values:
                    ax.scatter(
                        [index],
                        [np.mean(values)],
                        marker="_",
                        s=240,
                        color="#111111",
                        zorder=4,
                    )
            ax.set_xticks(
                range(len(present)),
                [
                    f"{'Indep.' if case == 'independent' else 'Central'}\n"
                    f"{environment.upper()}"
                    for case, environment in present
                ],
            )
            ax.set_title(panel_title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.text(0.07, 0.025, note, fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, 0.055, 1, 0.91))
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{filename}.{extension}", dpi=180)
        plt.close(fig)

    # These are the existing action-level waste measures.  They answer whether
    # the scheduler spent sensor time on work that another sensor made redundant,
    # independently of whether an obsolete product was eventually downlinked.
    plot_metric_grid(
        "coordination_waste",
        "Action-level coordination waste",
        (
            (
                "flat",
                "duplicate_attempt_count",
                "Duplicate task attempts",
                "Attempt count",
                1.0,
            ),
            (
                "flat",
                "duplicate_sensor_time_s",
                "Duplicate-task sensor time",
                "Sensor-hours",
                1 / 3600,
            ),
            (
                "flat",
                "wasted_sensor_time_s",
                "Total prospectus waste numerator",
                "Sensor-hours",
                1 / 3600,
            ),
            (
                "flat",
                "wasted_time_fraction",
                "Wasted mission-time fraction",
                "Percent of team sensor-time",
                100.0,
            ),
        ),
        "Circles: individual seeds; black bars: means. Total waste includes duplicate-task time and interrupted nonduplicate task time.",
    )

    # Keep data freshness and physical onboard overlap distinct.  A stale ground
    # delivery can occur after products no longer overlap onboard; conversely,
    # overlapping holders expose duplicated acquisition effort before downlink.
    plot_metric_grid(
        "catalog_and_product_duplicates",
        "Catalog freshness and physical product duplication",
        (
            (
                "product_duplicates",
                "stale_cross_sensor_ground_delivery_count",
                "Stale cross-sensor ground deliveries",
                "Qualified product count",
                1.0,
            ),
            (
                "product_duplicates",
                "causally_avoidable_stale_ground_delivery_count",
                "Causally avoidable stale deliveries",
                "Qualified product count",
                1.0,
            ),
            (
                "product_duplicates",
                "cross_sensor_onboard_overlap_target_count",
                "Targets held by multiple sensors",
                "Distinct target count",
                1.0,
            ),
            (
                "product_duplicates",
                "cross_sensor_onboard_redundant_acquisition_count",
                "Redundant onboard acquisitions",
                "Qualified acquisition count",
                1.0,
            ),
        ),
        "Stale delivery is timestamp-based. Onboard duplication is based on overlapping physical-storage intervals and product ownership.",
    )

    plot_metric_grid(
        "onboard_overlap_time",
        "Duration of cross-sensor onboard overlap",
        (
            (
                "product_duplicates",
                "cross_sensor_onboard_overlap_product_count",
                "Products involved in onboard overlap",
                "Qualified product count",
                1.0,
            ),
            (
                "product_duplicates",
                "cross_sensor_onboard_overlap_sensor_time_s",
                "All-holder overlap time",
                "Sensor-hours",
                1 / 3600,
            ),
            (
                "product_duplicates",
                "cross_sensor_onboard_redundant_sensor_time_s",
                "Integrated excess-holder time",
                "Excess-holder sensor-hours",
                1 / 3600,
            ),
            (
                "flat",
                "total_constellation_reward",
                "Constellation reward",
                "Cumulative reward",
                1.0,
            ),
        ),
        "Excess-holder time integrates holders beyond the first sensor; all-holder time includes every sensor during an overlap interval.",
    )

    # Unique services count useful revisits after the capture-anchored cooldown,
    # while first coverage above counts each catalog target only once. This plot
    # tests the expected mechanism directly: centralized coordination should
    # redirect duplicate work into additional credited services.
    plot_metric_grid(
        "productive_services_and_reward",
        "Productive services and positive reward components",
        (
            (
                "flat",
                "unique_acquisition_service_count",
                "Cooldown-qualified acquisition services",
                "Service count",
                1.0,
            ),
            (
                "flat",
                "unique_ground_service_count",
                "Ground-confirmed unique services",
                "Service count",
                1.0,
            ),
            (
                "flat",
                "acquisition_reward_component",
                "Positive acquisition reward (90%)",
                "Reward",
                1.0,
            ),
            (
                "flat",
                "ground_delivery_reward_component",
                "Positive ground-delivery reward (10%)",
                "Reward",
                1.0,
            ),
        ),
        "Unique services include useful revisits after cooldown; first target coverage remains capped at 100. The saved campaign has zero reward adjustments or penalties.",
    )

    # A paired effect is the centralized result minus the independent result for
    # the exact same seed and spacecraft initial conditions.  Plotting these
    # differences makes the information effect visible without conflating it
    # with the often much larger seed-to-seed geometry variation.
    paired_panels = (
        (
            "flat",
            "capture_coverage_fraction",
            "Qualified capture coverage",
            "Percentage points",
            100.0,
        ),
        (
            "flat",
            "ground_delivery_coverage_fraction",
            "Ground-confirmed coverage",
            "Percentage points",
            100.0,
        ),
        (
            "flat",
            "unique_acquisition_service_count",
            "Cooldown-qualified acquisitions",
            "Service difference",
            1.0,
        ),
        (
            "flat",
            "unique_ground_service_count",
            "Ground-confirmed unique services",
            "Service difference",
            1.0,
        ),
        (
            "flat",
            "total_constellation_reward",
            "Constellation reward",
            "Reward difference",
            1.0,
        ),
        (
            "flat",
            "duplicate_attempt_count",
            "Duplicate task attempts",
            "Attempt difference",
            1.0,
        ),
        (
            "product_duplicates",
            "cross_sensor_onboard_redundant_acquisition_count",
            "Redundant onboard acquisitions",
            "Acquisition difference",
            1.0,
        ),
        (
            "product_duplicates",
            "cross_sensor_onboard_redundant_sensor_time_s",
            "Integrated excess-holder time",
            "Sensor-hour difference",
            1 / 3600,
        ),
    )
    indexed_results = {
        (result["case"], result["target_environment"], result["seed"]): result
        for result in results
    }

    def result_value(result, source, metric):
        return flatten(result)[metric] if source == "flat" else result[source][metric]

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.2))
    fig.suptitle(
        "Matched-seed effect of centralized information", x=0.07, ha="left", fontsize=16
    )
    fig.text(0.07, 0.935, context, fontsize=10, color="#444444")
    for ax, (source, metric, panel_title, ylabel, scale) in zip(
        axes.flat, paired_panels
    ):
        paired_rows = []
        for environment in ("leo", "mixed"):
            for seed in range(50):
                independent = indexed_results.get(("independent", environment, seed))
                centralized = indexed_results.get(
                    ("centralized_full_state", environment, seed)
                )
                if independent is None or centralized is None:
                    continue
                paired_rows.append(
                    {
                        "target_environment": environment,
                        "seed": seed,
                        "effect": (
                            result_value(centralized, source, metric)
                            - result_value(independent, source, metric)
                        )
                        * scale,
                    }
                )
        comparison_groups = (
            (
                "LEO",
                [r for r in paired_rows if r["target_environment"] == "leo"],
                "#3573B9",
                "o",
                False,
            ),
            (
                "Mixed",
                [r for r in paired_rows if r["target_environment"] == "mixed"],
                "#A87819",
                "s",
                False,
            ),
            ("Pooled", paired_rows, "#5A4A78", "^", True),
        )
        for x_index, (_, group_rows, color, marker, seed_blocked) in enumerate(
            comparison_groups
        ):
            paired_values = [row["effect"] for row in group_rows]
            if not paired_values:
                continue
            offsets = (
                np.linspace(-0.10, 0.10, len(paired_values))
                if len(paired_values) > 1
                else [0]
            )
            ax.scatter(
                x_index + np.asarray(offsets),
                paired_values,
                facecolors="none",
                edgecolors=color,
                marker=marker,
                alpha=0.55,
                s=24,
                zorder=2,
            )
            statistics = (
                summarize_seed_blocked(group_rows, "effect")
                if seed_blocked
                else summarize(paired_values)
            )
            mean = statistics["mean"]
            interval = statistics["bootstrap_mean_ci95"]
            if interval is not None:
                ax.errorbar(
                    [x_index],
                    [mean],
                    yerr=[[mean - interval[0]], [interval[1] - mean]],
                    color="#111111",
                    marker="D",
                    markersize=4,
                    capsize=4,
                    linewidth=1.4,
                    zorder=4,
                )
            else:
                ax.scatter([x_index], [mean], marker="D", color="#111111", zorder=4)
            ax.annotate(
                f"n={len(paired_values)}",
                (x_index, mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#444444",
            )
        ax.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
        ax.set_xticks((0, 1, 2), ("LEO", "Mixed", "Pooled"))
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.07,
        0.018,
        "Centralized minus independent. Open points: matched environment/seed pairs; diamonds and bars: mean and 95% bootstrap CI.\n"
        "Pooled CIs resample 50 seed blocks, each containing both environments.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.91))
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"paired_information_effects.{extension}", dpi=180)
    plt.close(fig)

    # Keep individual time histories readable in the bounded preflight. The full
    # 50-seed campaign uses the endpoint distribution and paired summary tables.
    for environment in ("leo", "mixed"):
        seeds = sorted(
            {r["seed"] for r in results if r["target_environment"] == environment}
        )
        if len(seeds) > 5:
            continue
        for seed in seeds:
            pair = [
                r
                for r in results
                if r["target_environment"] == environment and r["seed"] == seed
            ]
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), sharey=True)
            fig.suptitle(
                f"Baseline coverage over time — {environment.upper()}, seed {seed}",
                x=0.08,
                ha="left",
                fontsize=15,
            )
            fig.text(
                0.08,
                0.90,
                context + " · single-seed diagnostic",
                fontsize=10,
                color="#444444",
            )
            for ax, ground, title in zip(
                axes, (False, True), ("Qualified capture", "Ground-confirmed delivery")
            ):
                for result in sorted(pair, key=lambda r: cases.index(r["case"])):
                    independent = result["case"] == "independent"
                    x, y = coverage_timeline(result, ground=ground)
                    ax.step(
                        x / 3600,
                        y,
                        where="post",
                        color="#3573B9" if independent else "#A87819",
                        linestyle="-" if independent else "--",
                        lw=1.8,
                        label=f"{labels[result['case']]} ({y[-1]:.0f}%)",
                    )
                ax.set(
                    title=title,
                    xlabel="Elapsed mission time (hours)",
                    xlim=(0, config["episode_duration_s"] / 3600),
                    ylim=(0, 105),
                )
                ax.axhline(100, color="#666666", linestyle=":", lw=0.7)
                ax.grid(axis="y", alpha=0.2)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(loc="lower right", frameon=False, fontsize=9)
            axes[0].set_ylabel("Distinct targets / mission catalog (%)")
            fig.text(
                0.08,
                0.03,
                "First qualified exposure completion / full physical ground delivery. Repeats do not add coverage.",
                fontsize=9,
                color="#444444",
            )
            fig.tight_layout(rect=(0, 0.065, 1, 0.87))
            for extension in ("png", "pdf"):
                fig.savefig(
                    output_dir / f"coverage_time_{environment}_seed{seed}.{extension}",
                    dpi=180,
                )
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
