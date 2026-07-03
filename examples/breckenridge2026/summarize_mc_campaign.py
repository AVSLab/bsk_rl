#!/usr/bin/env python3
"""Summarize completed Breckenridge Monte Carlo campaign metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = (
    ("cumulative_reward", "data", "cumulativeRewardSS1"),
    ("illuminated_images", "data", "illuminated_images"),
    ("useful_images_downlinked", "data", "useful_images_downlinked"),
    ("total_images_downlinked", "data", "total_images_downlinked"),
    ("target_imaging_count", "summary", "target_imaging_count"),
    ("downlink_action_count", "summary", "downlink_action_count"),
    ("charge_action_count", "summary", "charge_action_count"),
    ("desat_action_count", "summary", "desat_action_count"),
    ("umbra_smart_fraction", "data", "umbra_smart_fraction"),
    ("episode_end_time_sec", "data", "episode_end_time_sec"),
)

ACTION_COUNT_METRICS = {
    "target_imaging_count",
    "downlink_action_count",
    "charge_action_count",
    "desat_action_count",
}


def finite_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def metric_number(payload, output_name, section, source_name):
    raw_value = payload.get(section, {}).get(source_name)
    # Older evaluator output encoded a legitimate action count of zero as null.
    if raw_value is None and output_name in ACTION_COUNT_METRICS:
        return 0.0
    return finite_number(raw_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    manifest_paths = sorted(input_root.glob("*manifest.json"))
    expected_rows = None
    if len(manifest_paths) == 1:
        manifest = json.loads(manifest_paths[0].read_text())
        expected_rows = len(manifest.get("cells", {})) * len(
            manifest.get("seeds", [])
        )

    rows = []
    for status_path in sorted(input_root.glob("*/seed_*/mc_status.json")):
        status = json.loads(status_path.read_text())
        if status.get("state") != "completed":
            continue
        metrics_path = Path(status["metrics_path"])
        payload = json.loads(metrics_path.read_text())
        row = {"cell": status["cell"], "seed": int(status["seed"])}
        for output_name, section, source_name in METRICS:
            row[output_name] = metric_number(
                payload, output_name, section, source_name
            )
        rows.append(row)

    per_seed_path = input_root / "breckenridge2026_mc_per_seed.csv"
    fieldnames = ["cell", "seed"] + [metric[0] for metric in METRICS]
    with per_seed_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for cell in sorted({row["cell"] for row in rows}):
        cell_rows = [row for row in rows if row["cell"] == cell]
        summary = {"cell": cell, "completed_seeds": len(cell_rows)}
        for output_name, _, _ in METRICS:
            values = [
                row[output_name]
                for row in cell_rows
                if row[output_name] is not None
            ]
            summary[f"{output_name}_mean"] = (
                statistics.mean(values) if values else None
            )
            summary[f"{output_name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        summary_rows.append(summary)

    summary_path = input_root / "breckenridge2026_mc_summary.csv"
    summary_fields = ["cell", "completed_seeds"]
    for output_name, _, _ in METRICS:
        summary_fields.extend([f"{output_name}_mean", f"{output_name}_std"])
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    expected_text = str(expected_rows) if expected_rows is not None else "unknown"
    print(f"Completed seed rows: {len(rows)} / {expected_text}")
    print(f"Per-seed CSV: {per_seed_path}")
    print(f"Campaign summary:  {summary_path}")


if __name__ == "__main__":
    main()
