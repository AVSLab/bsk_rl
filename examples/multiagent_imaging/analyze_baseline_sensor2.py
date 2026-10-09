#!/usr/bin/env python3
"""Summarize sensor-2 workload using saved baseline episode and agent records."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

SENSORS = ("sensor_0", "sensor_1", "sensor_2")
ENVIRONMENTS = ("leo", "mixed")
CASES = ("independent", "centralized_full_state")
IMAGE_ACTIONS = range(5, 15)  # Five non-imaging actions, ten candidate slots.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--agent-csv", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.agent_csv.open(newline="") as handle:
        agents = list(csv.DictReader(handle))
    if len(agents) != 600:
        raise ValueError("Expected 600 agent/seed rows")
    by_task = defaultdict(dict)
    for row in agents:
        task = int(row["task_id"])
        sensor = row["sensor"]
        if sensor in by_task[task]:
            raise ValueError("Duplicate sensor row")
        by_task[task][sensor] = row
    provenance = json.loads(args.source_provenance.read_text())
    observations = defaultdict(list)
    top_counts = defaultdict(int)
    initial_state_hashes = defaultdict(set)
    minimum_charge_start_battery = 1.0
    forced_low_battery_charge_starts = 0
    for task in range(200):
        path = args.episodes_dir / f"episode_{task:03}.json"
        content = path.read_bytes()
        if (
            hashlib.sha256(content).hexdigest()
            != provenance["episode_sha256"][path.name]
        ):
            raise ValueError(f"Raw episode hash mismatch: {task}")
        episode = json.loads(content)
        case = episode["case"]
        env = episode["target_environment"]
        seed = episode["seed"]
        per_sensor = by_task[task]
        if set(per_sensor) != set(SENSORS):
            raise ValueError(f"Incomplete sensor rows: {task}")
        visits = {
            sensor: int(per_sensor[sensor]["credited_visits"]) for sensor in SENSORS
        }
        if visits["sensor_2"] > max(visits["sensor_0"], visits["sensor_1"]):
            top_counts[env, case] += 1
        for sensor in SENSORS:
            row = per_sensor[sensor]
            if (row["environment"], row["case"], int(row["seed"])) != (
                env,
                case,
                seed,
            ):
                raise ValueError("Agent row does not match episode")
            action_counts = episode["action_counts"][sensor]
            imaging_starts = sum(
                action_counts.get(str(slot), 0) for slot in IMAGE_ACTIONS
            )
            tasks = [
                task
                for task in episode["coordination"]["task_history"]
                if task["sensor"] == sensor
            ]
            images = [task for task in tasks if task["mode"] == "image"]
            charges = [task for task in tasks if task["mode"] == "charge"]
            if len(images) != imaging_starts or len(charges) != action_counts.get("0", 0):
                raise ValueError(f"Action/task count mismatch: {task}, {sensor}")
            resources = episode["resource_history"][sensor]
            resource_times = [state["time_s"] for state in resources]
            charge_start_battery = []
            for charge in charges:
                index = bisect.bisect_left(resource_times, charge["start"])
                if (
                    index >= len(resources)
                    or abs(resource_times[index] - charge["start"]) > 1e-4
                ):
                    raise ValueError(f"Missing charge-start resource sample: {task}, {sensor}")
                charge_start_battery.append(resources[index]["battery_fraction"])
            threshold = episode["baseline_config"]["battery_charge_threshold"]
            minimum_charge_start_battery = min(
                minimum_charge_start_battery, *charge_start_battery
            )
            forced_low_battery_charge_starts += sum(
                battery < threshold for battery in charge_start_battery
            )
            observations[env, case, sensor].append(
                {
                    "imaging_starts": imaging_starts,
                    "charge_starts": action_counts.get("0", 0),
                    "downlink_starts": action_counts.get("1", 0),
                    "imaging_time_s": sum(t["end"] - t["start"] for t in images),
                    "charge_time_s": sum(t["end"] - t["start"] for t in charges),
                    "mean_image_duration_s": statistics.mean(
                        t["end"] - t["start"] for t in images
                    ),
                    "charge_fallback_starts": sum(
                        battery >= threshold for battery in charge_start_battery
                    ),
                    "median_charge_start_battery": statistics.median(
                        charge_start_battery
                    ),
                    "qualifying_images": int(row["qualifying_images"]),
                    "physically_delivered_images": int(
                        row["physically_delivered_images"]
                    ),
                    "credited_visits": visits[sensor],
                    "useful_yield_pct": float(row["useful_yield_pct"]),
                    "credited_reimages": int(row["credited_reimages"]),
                }
            )
            state = episode["initial_conditions"][sensor]
            initial_state_hashes[sensor].add(
                hashlib.sha256(
                    json.dumps(
                        {
                            "position_N_m": state["position_N_m"],
                            "velocity_N_m_s": state["velocity_N_m_s"],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
            )
    summary = {}
    for env in ENVIRONMENTS:
        for case in CASES:
            for sensor in SENSORS:
                rows = observations[env, case, sensor]
                if len(rows) != 50:
                    raise ValueError("Expected 50 episodes per sensor/cell")
                summary[f"{env}/{case}/{sensor}"] = {
                    "n": len(rows),
                    **{
                        f"mean_{key}": statistics.mean(row[key] for row in rows)
                        for key in rows[0]
                    },
                }
    report = {
        "status": "PASS",
        "source_agent_csv_sha256": hashlib.sha256(
            args.agent_csv.read_bytes()
        ).hexdigest(),
        "episode_count": 200,
        "sensor_seed_rows": 600,
        "minimum_charge_start_battery": minimum_charge_start_battery,
        "forced_low_battery_charge_starts": forced_low_battery_charge_starts,
        "sensor2_top_credited_seeds_out_of_50": {
            f"{env}/{case}": top_counts[env, case]
            for env in ENVIRONMENTS
            for case in CASES
        },
        "unique_initial_orbit_states_by_sensor": {
            sensor: len(hashes) for sensor, hashes in initial_state_hashes.items()
        },
        "mean_metrics": summary,
        "interpretation_limit": (
            "Saved episodes prove sensor-2 task volume and credit, but do not save "
            "the complete rejected candidate geometry at every boundary; orbital "
            "geometry versus greedy selection cannot be causally separated here."
        ),
    }
    (args.output_dir / "sensor2_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
