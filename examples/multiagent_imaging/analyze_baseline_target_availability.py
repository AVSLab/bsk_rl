#!/usr/bin/env python3
"""Reconstruct a conservative team target pool from preserved episode records.

This counts catalog targets not held by an active image task or any recent
qualifying capture. It deliberately does not claim instantaneous line of sight,
illumination, candidate-slot membership, or per-sensor imageability.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_baseline_overlay_contours import CASES, ENVIRONMENTS, SENSORS, palette, setup_style


def pools_at(episode: dict, times: list[float]) -> tuple[list[int], list[int]]:
    """Over-reserve by treating every recent qualified capture as a reservation."""
    cooldown = float(episode["reimage_cooldown_s"])
    captures = defaultdict(list)
    for capture in episode["capture_records"]:
        if capture["quality"] >= 0.5:
            captures[int(capture["target_id"])].append(float(capture["capture_time"]))
    for values in captures.values():
        values.sort()
    image_tasks = [
        task for task in episode["coordination"]["task_history"]
        if task["mode"] == "image" and task["target_id"] is not None
    ]
    free, never_captured = [], []
    for now in times:
        recent = set()
        ever = set()
        for target_id, values in captures.items():
            index = bisect.bisect_right(values, now)
            if index:
                ever.add(target_id)
                if values[index - 1] + cooldown > now:
                    recent.add(target_id)
        active = {
            int(task["target_id"]) for task in image_tasks
            if task["start"] <= now < task["end"]
        }
        free.append(episode["passive_target_count"] - len(recent | active))
        never_captured.append(
            episode["passive_target_count"] - len(ever | active)
        )
    return free, never_captured


def check_and_load(episodes_dir: Path, provenance: dict) -> list[dict]:
    episodes = []
    for task_id in range(200):
        name = f"episode_{task_id:03}.json"
        content = (episodes_dir / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != provenance["episode_sha256"][name]:
            raise ValueError(f"Saved raw record differs from verified input: {name}")
        episode = json.loads(content)
        if episode["task_id"] != task_id or episode["sim_time_s"] != 45000:
            raise ValueError(f"Invalid task or horizon: {name}")
        episodes.append(episode)
    return episodes


def summarize(episodes: list[dict]) -> dict:
    summary = {}
    for env in ENVIRONMENTS:
        for case in CASES:
            free_at_charge = []
            never_at_charge = []
            alone = 0
            charge_count = 0
            for episode in episodes:
                if (episode["target_environment"], episode["case"]) != (env, case):
                    continue
                tasks = episode["coordination"]["task_history"]
                charges = [task for task in tasks if task["mode"] == "charge"]
                starts = sorted({float(task["start"]) for task in charges})
                free_pool, never_pool = pools_at(episode, starts)
                pool_by_time = dict(zip(starts, free_pool))
                never_by_time = dict(zip(starts, never_pool))
                task_starts = defaultdict(set)
                for task in tasks:
                    task_starts[float(task["start"])].add(task["sensor"])
                for task in charges:
                    now = float(task["start"])
                    free_at_charge.append(pool_by_time[now])
                    never_at_charge.append(never_by_time[now])
                    alone += task_starts[now] == {task["sensor"]}
                    charge_count += 1
            if not free_at_charge:
                raise ValueError(f"No charge decisions in {env}/{case}")
            summary[f"{env}/{case}"] = {
                "episode_count": 50,
                "charge_starts": charge_count,
                "conservative_free_at_charge_min": min(free_at_charge),
                "conservative_free_at_charge_median": statistics.median(free_at_charge),
                "conservative_free_at_charge_p10": float(np.percentile(free_at_charge, 10)),
                "conservative_free_at_charge_p90": float(np.percentile(free_at_charge, 90)),
                "never_qualified_at_charge_min": min(never_at_charge),
                "never_qualified_at_charge_median": statistics.median(never_at_charge),
                "charge_starts_with_zero_conservative_free": sum(
                    count == 0 for count in free_at_charge
                ),
                "charge_starts_without_other_sensor_start": alone,
                "charge_starts_without_other_sensor_start_fraction": alone / charge_count,
            }
    return summary


def matched_fallback_proxy(episodes: list[dict]) -> dict:
    """Compare same-sensor independent image starts inside central charge spans.

    This is a matched-scene opportunity proxy, not proof that the centralized
    sensor could complete that image from its different attitude/resource state.
    """
    by_cell = {
        (episode["target_environment"], episode["case"], episode["seed"]): episode
        for episode in episodes
    }
    report = {}
    for env in ENVIRONMENTS:
        counts = defaultdict(int)
        clear_lags = []
        for seed in range(50):
            independent = by_cell[env, "independent", seed]
            centralized = by_cell[env, "centralized_full_state", seed]
            if (
                independent["initial_conditions_sha256"]
                != centralized["initial_conditions_sha256"]
            ):
                raise ValueError(f"Unmatched initial state: {env}, {seed}")
            cooldown = centralized["reimage_cooldown_s"]
            qualified_central = defaultdict(list)
            for capture in centralized["capture_records"]:
                if capture["quality"] >= 0.5:
                    qualified_central[int(capture["target_id"])].append(
                        float(capture["capture_time"])
                    )
            for times in qualified_central.values():
                times.sort()
            delivered_at = {
                product["record_id"]: float(product["delivery_time"])
                for product in centralized["ground_delivery_records"]
            }
            own_captures = defaultdict(list)
            for capture in centralized["capture_records"]:
                own_captures[capture["source_sensor"], int(capture["target_id"])].append(
                    capture
                )
            central_images = [
                task for task in centralized["coordination"]["task_history"]
                if task["mode"] == "image"
            ]
            independent_qualified = defaultdict(list)
            for capture in independent["capture_records"]:
                if capture["quality"] >= 0.5:
                    independent_qualified[
                        capture["source_sensor"], int(capture["target_id"])
                    ].append(float(capture["capture_time"]))

            def clear_at_charge(sensor: str, target_id: int, now: float) -> bool:
                times = qualified_central[target_id]
                index = bisect.bisect_right(times, now)
                if index and times[index - 1] + cooldown > now:
                    return False
                if any(
                    task["target_id"] == target_id
                    and task["start"] <= now < task["end"]
                    for task in central_images
                ):
                    return False
                if any(
                    capture["capture_time"] <= now
                    and delivered_at.get(capture["record_id"], float("inf")) > now
                    for capture in own_captures[sensor, target_id]
                ):
                    return False
                return True

            for sensor in SENSORS:
                independent_images = sorted(
                    (
                        task for task in independent["coordination"]["task_history"]
                        if task["sensor"] == sensor and task["mode"] == "image"
                    ),
                    key=lambda task: task["start"],
                )
                starts = [task["start"] for task in independent_images]
                central_charges = [
                    task for task in centralized["coordination"]["task_history"]
                    if task["sensor"] == sensor and task["mode"] == "charge"
                ]
                for charge in central_charges:
                    counts["central_charge_starts"] += 1
                    left = bisect.bisect_left(starts, charge["start"])
                    right = bisect.bisect_left(starts, charge["end"])
                    selected = independent_images[left:right]
                    counts["independent_image_starts_inside_charge"] += len(selected)
                    counts["charges_with_independent_image_start"] += bool(selected)
                    clear_hit = False
                    for image_task in selected:
                        target_id = int(image_task["target_id"])
                        qualified = any(
                            image_task["start"] <= time <= image_task["end"]
                            for time in independent_qualified[sensor, target_id]
                        )
                        if not qualified:
                            continue
                        counts["independent_qualifying_images_inside_charge"] += 1
                        if clear_at_charge(sensor, target_id, charge["start"]):
                            counts["clear_target_qualifying_image_starts_inside_charge"] += 1
                            clear_hit = True
                            clear_lags.append(image_task["start"] - charge["start"])
                    counts["charges_with_clear_target_qualifying_image"] += clear_hit
        report[env] = {
            **dict(counts),
            "charge_fraction_with_independent_image_start": (
                counts["charges_with_independent_image_start"]
                / counts["central_charge_starts"]
            ),
            "charge_fraction_with_clear_target_qualifying_image": (
                counts["charges_with_clear_target_qualifying_image"]
                / counts["central_charge_starts"]
            ),
            "median_clear_target_image_start_delay_s": statistics.median(clear_lags),
        }
    return report


def plot_seed_zero(episodes: list[dict], env: str, output_dir: Path) -> None:
    colors = palette()
    fig, axes = plt.subplots(
        2, 2, figsize=(15.2, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.1], "hspace": 0.07, "wspace": 0.10},
    )
    for column, case in enumerate(CASES):
        episode = next(
            e for e in episodes if e["target_environment"] == env
            and e["case"] == case and e["seed"] == 0
        )
        times = [float(state["time_s"]) for state in episode["resource_history"]["sensor_0"]]
        free, never = pools_at(episode, times)
        top, bottom = axes[:, column]
        top.step(times, free, where="post", color="#246C9A", linewidth=1.5,
                 label="Team cooldown-free catalog (conservative)")
        top.step(times, never, where="post", color="#C98518", linestyle="--",
                 linewidth=1.3, label="Never qualified, outside active tasks")
        top.set_ylim(0, 105)
        top.set_title(
            "Independent local greedy" if case == "independent"
            else "Centralized full-state joint greedy", fontsize=13,
        )
        top.grid(True, alpha=0.2)
        if column == 0:
            top.set_ylabel("Target count of 100")
        else:
            top.tick_params(labelleft=False)
        for index, sensor in enumerate(SENSORS):
            sensor_tasks = [
                task for task in episode["coordination"]["task_history"]
                if task["sensor"] == sensor
            ]
            for mode, color in (
                ("charge", "#C9CDD1"),
                ("downlink", "#2E7DB5"),
                ("image", colors[case, sensor]),
            ):
                spans = [
                    (float(task["start"]), float(task["end"] - task["start"]))
                    for task in sensor_tasks if task["mode"] == mode
                    and task["end"] > task["start"]
                ]
                if spans:
                    bottom.broken_barh(spans, (index - 0.34, 0.68), facecolors=color,
                                       edgecolors="none")
        bottom.set_yticks(range(3), ["Sensor 0", "Sensor 1", "Sensor 2"])
        bottom.set_ylim(-0.6, 2.6)
        bottom.set_xlim(0, 45000)
        bottom.set_xlabel("Simulation time [s]")
        bottom.grid(True, axis="x", alpha=0.2)
        if column == 1:
            bottom.tick_params(labelleft=False)
    handles = [
        Line2D([0], [0], color="#246C9A", label="Outside recent team cooldown"),
        Line2D([0], [0], color="#C98518", linestyle="--", label="Never qualified"),
        Patch(facecolor="#C9CDD1", label="300-s charge fallback"),
        Patch(facecolor="#2E7DB5", label="Physical downlink"),
        Patch(facecolor="#A52F66", label="Image task (sensor shade)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=5, frameon=False, fontsize=9.5)
    fig.suptitle(
        f"{env.upper() if env == 'leo' else 'Mixed LEO/MEO/GEO'} seed 0: catalog pool and sensor task time",
        y=1.015, fontsize=17,
    )
    fig.text(
        0.5, 0.015,
        "Catalog pool is a lower bound on targets not team-reserved; it does not test LOS, illumination or the 10-slot shortlist.",
        ha="center", fontsize=9.7, color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.12)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"availability_seed0_{env}.{extension}",
                    dpi=320, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(args.source_provenance.read_text())
    episodes = check_and_load(args.episodes_dir, provenance)
    setup_style()
    summary = summarize(episodes)
    for env in ENVIRONMENTS:
        plot_seed_zero(episodes, env, args.output_dir)
    result = {
        "status": "PASS", "episode_count": len(episodes),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_combined_input_sha256": provenance["combined_input_sha256"],
        "source_fingerprint": provenance["historical_source_fingerprint"],
        "method": (
            "Each qualified capture within the recorded cooldown and each active "
            "image task is counted as reserved. This may overstate actual central "
            "reservations, so the remainder is a conservative lower bound on "
            "catalog targets outside those reservations. Geometry, illumination, "
            "shortlist membership, and per-sensor eligibility were not saved."
        ),
        "cells": summary,
        "matched_fallback_proxy": matched_fallback_proxy(episodes),
    }
    (args.output_dir / "availability_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
