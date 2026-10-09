"""Standalone baseline bars, physical delivery latency and mission performance.

Uses preserved per-episode evidence only; does not execute a simulator. Latency
starts at qualified exposure completion and ends at complete physical delivery.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples.multiagent_imaging.visualize_baseline_comparison import (
    CASE_COLORS,
    CASE_LABELS,
    digest,
)

ENVIRONMENTS = {"leo": "LEO", "mixed": "Mixed LEO/MEO/GEO"}
HORIZON_S = 45000.0
COHORT_END_S = HORIZON_S / 2


def ground_services(episode):
    """Replay ground credit, including simultaneous captures and valid revisits.

    Mirrors the documented evaluator semantics independently from saved totals:
    group deliveries at an event boundary, then group equal target/epoch/capture
    times, credit one qualified service if its capture clears the cooldown.
    """
    boundaries = defaultdict(list)
    for product in episode["ground_delivery_records"]:
        boundaries[product["delivery_time"]].append(product)
    latest = {}
    accepted = []
    cooldown = episode["reimage_cooldown_s"]
    for time_s, products in sorted(boundaries.items()):
        products.sort(
            key=lambda p: (
                p["target_id"],
                p["request_epoch"],
                p["capture_time"],
                p["source_sensor"],
                p["record_id"],
            )
        )
        index = 0
        while index < len(products):
            first = products[index]
            group = [first]
            index += 1
            key = (first["target_id"], first["request_epoch"])
            while index < len(products):
                other = products[index]
                if (other["target_id"], other["request_epoch"]) != key or abs(
                    other["capture_time"] - first["capture_time"]
                ) > 1e-6:
                    break
                group.append(other)
                index += 1
            if any(p["quality"] >= 0.5 for p in group) and (
                key not in latest or first["capture_time"] >= latest[key] + cooldown
            ):
                latest[key] = first["capture_time"]
                accepted.append(time_s)
    if len(accepted) != episode["team_summary"]["unique_service_count"]:
        raise ValueError(
            f"Ground-service replay disagrees for task {episode['task_id']}"
        )
    return np.array(accepted)


def latency_records(episode):
    qualified = {
        p["record_id"]: p for p in episode["capture_records"] if p["quality"] >= 0.5
    }
    assert len(qualified) == sum(
        p["quality"] >= 0.5 for p in episode["capture_records"]
    )
    delivery = {p["record_id"]: p for p in episode["ground_delivery_records"]}
    assert len(delivery) == len(episode["ground_delivery_records"])
    rows = []
    for record_id, product in qualified.items():
        start = product["completion_time"]
        end = delivery[record_id]["delivery_time"] if record_id in delivery else None
        assert 0 <= product["capture_time"] <= start <= HORIZON_S
        if end is not None:
            ground = delivery[record_id]
            assert (
                start <= end <= HORIZON_S
                and ground["target_id"] == product["target_id"]
            )
            assert ground["source_sensor"] == product["source_sensor"]
        rows.append(
            {
                "task_id": episode["task_id"],
                "environment": episode["target_environment"],
                "case": episode["case"],
                "seed": episode["seed"],
                "record_id": record_id,
                "target_id": product["target_id"],
                "sensor": product["source_sensor"],
                "request_epoch": product["request_epoch"],
                "quality": product["quality"],
                "acquisition_complete_s": start,
                "ground_delivery_s": end,
                "delivered": end is not None,
                "latency_min": (end - start) / 60 if end is not None else None,
                "observed_duration_min": (
                    (end if end is not None else HORIZON_S) - start
                )
                / 60,
                "early_cohort": start <= COHORT_END_S,
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(episodes_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["individual", "latency", "performance"]:
        (output_dir / sub).mkdir(exist_ok=True)
    episodes = []
    input_hashes = {}
    for path in sorted(episodes_dir.glob("episode_*.json")):
        raw = path.read_bytes()
        input_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        episodes.append(json.loads(raw))
    assert {d["task_id"] for d in episodes} == set(range(200)) and len(episodes) == 200
    groups = {
        (case, env): sorted(
            [
                d
                for d in episodes
                if d["case"] == case and d["target_environment"] == env
            ],
            key=lambda d: d["seed"],
        )
        for case in CASE_LABELS
        for env in ENVIRONMENTS
    }
    assert all(
        [d["seed"] for d in group] == list(range(50)) for group in groups.values()
    )
    for env in ENVIRONMENTS:
        for left, right in zip(
            groups["independent", env], groups["centralized_full_state", env]
        ):
            assert (
                left["initial_conditions_sha256"] == right["initial_conditions_sha256"]
            )
    episode_rows = []
    products = []
    physical = {}
    services = {}
    for d in episodes:
        assert d["horizon_reached"] and d["sim_time_s"] == HORIZON_S
        assert d["coverage"]["catalog_target_count"] == 100
        rows = latency_records(d)
        products.extend(rows)
        physical[d["task_id"]] = rows
        services[d["task_id"]] = ground_services(d)
        delivered = np.array([r["latency_min"] for r in rows if r["delivered"]])
        assert len(delivered) and len(rows)
        assert len(rows) == d["coverage"]["qualified_exposure_count"]
        assert len(delivered) == d["coverage"]["qualified_ground_delivery_count"]
        episode_rows.append(
            {
                "task_id": d["task_id"],
                "environment": d["target_environment"],
                "case": d["case"],
                "seed": d["seed"],
                "qualified_products": len(rows),
                "delivered_products": len(delivered),
                "pending_products": len(rows) - len(delivered),
                "physical_delivery_fraction": len(delivered) / len(rows),
                "latency_median_min": float(np.median(delivered)),
                "latency_p90_min": float(np.quantile(delivered, 0.9)),
                "unique_ground_services": len(services[d["task_id"]]),
                "unique_ground_services_per_100_exposures": len(services[d["task_id"]])
                / len(rows)
                * 100,
                "unique_acquisitions": d["team_summary"]["unique_acquisition_count"],
                "duplicate_attempts": d["team_summary"]["duplicate_attempt_count"],
                "capture_coverage_pct": 100
                * d["coverage"]["capture_coverage_fraction"],
                "ground_coverage_pct": 100
                * d["coverage"]["ground_delivery_coverage_fraction"],
                "reward": d["total_constellation_reward"],
            }
        )
    indexed = {r["task_id"]: r for r in episode_rows}
    cells = {
        (case, env): [indexed[d["task_id"]] for d in ds]
        for (case, env), ds in groups.items()
    }
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "physical_product_latencies.csv", products)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#242424",
            "axes.labelcolor": "#242424",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    style = {
        "independent": {"linestyle": "--", "marker": "o"},
        "centralized_full_state": {"linestyle": "-", "marker": "D"},
    }
    created = []

    def figure(title, subtitle, two=True):
        fig, axes = plt.subplots(
            1, 2 if two else 1, figsize=(12.2, 6.6) if two else (8.5, 6.5)
        )
        fig.suptitle(title, x=0.07, y=0.98, ha="left", fontsize=18)
        fig.text(0.07, 0.90, subtitle, fontsize=10.5, color="#505050", va="top")
        for ax in np.atleast_1d(axes):
            ax.grid(alpha=0.16)
            ax.set_axisbelow(True)
        if two:
            for ax, label in zip(axes, ENVIRONMENTS.values()):
                ax.set_title(label, fontsize=13)
        return fig, axes

    def save(fig, stem, note, legend=True):
        if legend:
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=CASE_COLORS[c],
                    label=CASE_LABELS[c],
                    linewidth=2,
                    **style[c],
                )
                for c in CASE_LABELS
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.079),
                ncol=2,
                frameon=False,
                fontsize=10,
            )
        fig.text(0.07, 0.022, note, fontsize=9.5, color="#505050", va="bottom")
        fig.tight_layout(rect=(0.01, 0.19, 0.99, 0.81))
        for ext in ["png", "pdf"]:
            fig.savefig(output_dir / f"{stem}.{ext}", dpi=190, bbox_inches="tight")
        created.append(stem)
        plt.close(fig)

    # Individual exports of exactly the six metrics in the original grid.
    specs = [
        (
            "capture_coverage",
            "Capture coverage",
            "Targets captured at least once (%)",
            "capture_coverage_pct",
        ),
        (
            "ground_coverage",
            "Ground-delivered coverage",
            "Targets delivered at least once (%)",
            "ground_coverage_pct",
        ),
        (
            "qualified_acquisitions",
            "Cooldown-qualified acquisitions",
            "Qualified acquisition services per episode",
            "unique_acquisitions",
        ),
        (
            "unique_ground_services",
            "Credited target visits delivered",
            "Credited delivered target visits per episode",
            "unique_ground_services",
        ),
        (
            "constellation_reward",
            "Constellation reward",
            "Total constellation reward per episode",
            "reward",
        ),
        (
            "duplicate_attempts",
            "Duplicate image attempts",
            "Duplicate attempts per episode",
            "duplicate_attempts",
        ),
    ]
    for name, title, unit, key in specs:
        fig, ax = figure(
            title,
            "Deterministic coverage-first greedy · 50 matched seeds per environment\n3 sensors · 100 targets · 10 candidates · 45,000 s",
            two=False,
        )
        maximum = max(r[key] for r in episode_rows)
        for j, case in enumerate(CASE_LABELS):
            for i, env in enumerate(ENVIRONMENTS):
                values = np.array([r[key] for r in cells[case, env]])
                x = i + (j - 0.5) * 0.34
                ax.bar(
                    x,
                    values.mean(),
                    width=0.31,
                    color=CASE_COLORS[case],
                    edgecolor="#333333",
                    linewidth=0.7,
                    hatch="//" if j == 0 else None,
                )
                ax.scatter(
                    x + np.linspace(-0.10, 0.10, 50),
                    values,
                    s=12,
                    color="#242424",
                    alpha=0.22,
                    zorder=3,
                )
                ax.text(
                    x,
                    values.mean() + maximum * 0.025,
                    f"{values.mean():.2f}",
                    ha="center",
                    fontsize=11,
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=list(ENVIRONMENTS.values()),
            ylabel=unit,
            ylim=(0, 108 if "coverage" in name else maximum * 1.12),
        )
        save(
            fig,
            f"individual/{name}",
            "Bars: episode means. Points: individual seeds.\nCoverage uses 100 targets; credited target visits include valid revisits after cooldown.",
        )

    fig, axes = figure(
        "Episode median acquisition-to-ground latency",
        "Exposure completion → full physical delivery · minutes · 50 episodes per case/environment\nEach point is one episode's median over its delivered qualified products; pending products are excluded.",
    )
    for ax, env in zip(axes, ENVIRONMENTS):
        for x, case in enumerate(CASE_LABELS, 1):
            values = np.array([r["latency_median_min"] for r in cells[case, env]])
            box = ax.boxplot(
                [values],
                positions=[x],
                widths=0.42,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                whis=(5, 95),
            )
            for patch in box["boxes"]:
                patch.set(
                    facecolor="white",
                    edgecolor=CASE_COLORS[case],
                    linewidth=1.8,
                    hatch="//" if x == 1 else None,
                )
            for component in ["whiskers", "caps", "medians"]:
                for line in box[component]:
                    line.set(color=CASE_COLORS[case], linewidth=1.6)
            ax.scatter(
                x + np.linspace(-0.16, 0.16, 50),
                values,
                s=24,
                facecolors="none" if x == 1 else CASE_COLORS[case],
                edgecolors=CASE_COLORS[case],
                marker=style[case]["marker"],
                alpha=0.6,
                zorder=3,
            )
            ax.text(
                x,
                values.max() + 2,
                f"Median {np.median(values):.2f} min",
                ha="center",
                fontsize=10,
            )
        ax.set(
            xticks=[1, 2],
            xticklabels=["Independent", "Centralized"],
            ylabel="Episode median latency (minutes)",
            ylim=(0, 60),
        )
    save(
        fig,
        "latency/episode_median_latency",
        "Boxes: middle 50% of episode medians; whiskers: 5th–95th percentiles.\nPhysical products include duplicate copies. Undelivered counts are retained in the data tables.",
    )

    fig, axes = figure(
        "Matched median acquisition-to-ground latency",
        "Same environment and seed · delivered qualified products · 50 matched points per panel",
    )
    for ax, env in zip(axes, ENVIRONMENTS):
        x = np.array([r["latency_median_min"] for r in cells["independent", env]])
        y = np.array(
            [r["latency_median_min"] for r in cells["centralized_full_state", env]]
        )
        ax.plot([0, 60], [0, 60], color="#777777", linestyle=":", linewidth=1.1)
        ax.scatter(
            x,
            y,
            s=40,
            facecolors="none",
            edgecolors=CASE_COLORS["centralized_full_state"],
            linewidths=1.2,
        )
        ax.set(
            xlabel="Independent episode median (minutes)",
            ylabel="Centralized episode median (minutes)",
            xlim=(10, 55),
            ylim=(10, 55),
            aspect="equal",
        )
        ax.text(
            0.035,
            0.97,
            f"Centralized faster in {int((y<x).sum())}/50 seeds\nMean paired change: {(y-x).mean():+.2f} min",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
    save(
        fig,
        "latency/matched_median_latency",
        "Diagonal: equal latency. Below the line: centralized faster; above: slower.\nDelivered-only medians describe different product sets under the two controllers.",
        legend=False,
    )

    delays = np.linspace(0, COHORT_END_S / 60, 376)
    delay_stats = {}
    fig, axes = figure(
        "Physical delivery fraction versus time since acquisition",
        "Acquisitions completed in the first 6.25 mission hours · every product has ≥6.25 hours follow-up\nEpisode-weighted curves retain products not delivered within each delay; 50 episodes per case/environment.",
    )
    for ax, env in zip(axes, ENVIRONMENTS):
        for case in CASE_LABELS:
            curves = []
            n = 0
            for d in groups[case, env]:
                cohort = [r for r in physical[d["task_id"]] if r["early_cohort"]]
                assert cohort
                n += len(cohort)
                values = np.sort(
                    [r["latency_min"] if r["delivered"] else np.inf for r in cohort]
                )
                curves.append(
                    np.searchsorted(values, delays, side="right") / len(values) * 100
                )
            curves = np.array(curves)
            avg = curves.mean(axis=0)
            q25, q75 = np.quantile(curves, [0.25, 0.75], axis=0)
            ax.fill_between(
                delays, q25, q75, color=CASE_COLORS[case], alpha=0.13, linewidth=0
            )
            ax.plot(
                delays,
                avg,
                color=CASE_COLORS[case],
                linestyle=style[case]["linestyle"],
                linewidth=2,
            )
            delay_stats[f"{case}_{env}"] = {
                "products": n,
                "episode_mean_delivered_within_30min_pct": float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    r["delivered"] and r["latency_min"] <= 30
                                    for r in physical[d["task_id"]]
                                    if r["early_cohort"]
                                ]
                            )
                            * 100
                            for d in groups[case, env]
                        ]
                    )
                ),
                "episode_mean_delivered_within_375min_pct": float(avg[-1]),
            }
        ax.set(
            xlabel="Time since acquisition completion (minutes)",
            ylabel="Products fully delivered (%)",
            xlim=(0, 375),
            ylim=(0, 104),
        )
    save(
        fig,
        "latency/delivery_delay_curves",
        "Solid/dashed lines: mean episode delivery fraction. Bands: middle 50% of seeds, not confidence intervals.\nThe fixed early cohort avoids dropping end-of-mission products with insufficient follow-up.",
    )

    times = np.linspace(0, HORIZON_S, 151)
    fig, axes = figure(
        "Ground-confirmed unique services over mission time",
        "Physical full deliveries credited with the existing capture-anchored cooldown\nMean over 50 matched seeds per case/environment · shaded bands show the middle 50% of episodes",
    )
    for ax, env in zip(axes, ENVIRONMENTS):
        for case in CASE_LABELS:
            values = np.array(
                [
                    np.searchsorted(services[d["task_id"]], times, side="right")
                    for d in groups[case, env]
                ]
            )
            avg = values.mean(axis=0)
            q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
            ax.fill_between(
                times / 3600, q25, q75, color=CASE_COLORS[case], alpha=0.13, linewidth=0
            )
            ax.plot(
                times / 3600,
                avg,
                color=CASE_COLORS[case],
                linestyle=style[case]["linestyle"],
                linewidth=2,
            )
            ax.annotate(
                f"{avg[-1]:.2f}",
                (12.5, avg[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=10,
                color=CASE_COLORS[case],
                va="center",
                clip_on=False,
            )
        ax.set(
            xlabel="Simulated mission time (hours)",
            ylabel="Cumulative credited target visits delivered",
            xlim=(0, 13.3),
            ylim=(0, 370),
        )
    save(
        fig,
        "performance/ground_services_over_time",
        "Valid revisits after cooldown can make the count exceed 100 targets; duplicate copies add no credit.\nAll 200 reconstructed final counts agree exactly with saved episode totals.",
    )

    fig, axes = figure(
        "Matched credited target visits delivered",
        "Independent x-axis · centralized full-state y-axis · 50 matched environment/seed pairs per panel",
    )
    for ax, env in zip(axes, ENVIRONMENTS):
        x = np.array([r["unique_ground_services"] for r in cells["independent", env]])
        y = np.array(
            [r["unique_ground_services"] for r in cells["centralized_full_state", env]]
        )
        lo, hi = min(x.min(), y.min()) - 12, max(x.max(), y.max()) + 12
        ax.plot([lo, hi], [lo, hi], color="#777777", linestyle=":", linewidth=1.1)
        ax.scatter(
            x,
            y,
            s=40,
            facecolors="none",
            edgecolors=CASE_COLORS["centralized_full_state"],
            linewidths=1.2,
        )
        ax.set(
            xlabel="Independent credited target visits",
            ylabel="Centralized credited target visits",
            xlim=(lo, hi),
            ylim=(lo, hi),
            aspect="equal",
        )
        ax.text(
            0.035,
            0.97,
            f"Centralized higher in {int((y>x).sum())}/50 seeds\nMean paired change: {(y-x).mean():+.2f} visits",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
    save(
        fig,
        "performance/matched_ground_services",
        "Diagonal: equal credited visits. Above the line: more visits with centralized information.\nEach point is one matched seed; valid revisits after cooldown are included.",
        legend=False,
    )

    fig, axes = figure(
        "Useful delivered target visits versus image delivery delay",
        "Each point is one episode · 50 seeds × 2 controllers per environment\nHorizontal: median delay of delivered images; vertical: credited visits per 100 qualifying images taken",
    )
    for ax, env in zip(axes, ENVIRONMENTS):
        for case in CASE_LABELS:
            cell = cells[case, env]
            ax.scatter(
                [r["latency_median_min"] for r in cell],
                [r["unique_ground_services_per_100_exposures"] for r in cell],
                s=35,
                facecolors="none" if case == "independent" else CASE_COLORS[case],
                edgecolors=CASE_COLORS[case],
                marker=style[case]["marker"],
                alpha=0.65,
            )
        ax.set(
            xlabel="Episode median acquisition-to-ground delay (minutes)",
            ylabel="Credited delivered visits per 100 qualifying images",
            xlim=(10, 55),
            ylim=(0, 105),
        )
    save(
        fig,
        "performance/useful_services_vs_latency",
        "One credit = a qualifying image downlinked for a target visit outside cooldown; duplicate copies add no credit.\nThe cases image different targets; delivery speed alone does not measure mission usefulness.",
    )

    summary = {}
    for (case, env), cell in cells.items():
        qualified_rows = [
            p for p in products if p["case"] == case and p["environment"] == env
        ]
        latency = [p["latency_min"] for p in qualified_rows if p["delivered"]]
        summary[f"{case}_{env}"] = {
            "episodes": len(cell),
            "qualified_products": len(qualified_rows),
            "delivered_products": len(latency),
            "pending_products": len(qualified_rows) - len(latency),
            "median_of_episode_median_latency_min": float(
                np.median([r["latency_median_min"] for r in cell])
            ),
            "pooled_delivered_product_median_latency_min": float(np.median(latency)),
            "mean_episode_physical_delivery_pct": float(
                np.mean([r["physical_delivery_fraction"] for r in cell]) * 100
            ),
            "mean_unique_ground_services_per_100_exposures": float(
                np.mean([r["unique_ground_services_per_100_exposures"] for r in cell])
            ),
            **delay_stats[f"{case}_{env}"],
        }
    provenance = {
        "episodes_dir": str(episodes_dir.resolve()),
        "episodes": 200,
        "matched_pairs": 100,
        "input_sha256": input_hashes,
        "input_fingerprint": digest(input_hashes),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_plot_module_sha256": hashlib.sha256(
            Path(__file__).with_name("visualize_baseline_comparison.py").read_bytes()
        ).hexdigest(),
        "python": sys.version,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "campaign_source_fingerprints": sorted(
            {d["source"]["source_fingerprint"] for d in episodes}
        ),
        "campaign_manifest_hashes": sorted({d["manifest_sha256"] for d in episodes}),
        "heuristics": CASE_LABELS,
        "latency_start": "qualified physical exposure completion",
        "latency_end": "complete physical ground delivery",
        "quality_threshold": 0.5,
        "delivered_only_medians": True,
        "pending_products": "right-censored at 45000 s; retained in product table",
        "fixed_delay_cohort": "acquisition completion <=22500 s, delay <=22500 s",
        "curves": "equal episode weights; bands are seed IQR, not confidence intervals",
        "ground_service_replay_all_200": "PASS",
        "artifacts": created,
    }
    for name, value in [
        ("latency_summary.json", summary),
        ("plot_provenance.json", provenance),
    ]:
        (output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "figures": len(created),
                "episodes": len(episodes),
                "qualified_product_rows": len(products),
                "output": str(output_dir.resolve()),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.episodes_dir, args.output_dir)
