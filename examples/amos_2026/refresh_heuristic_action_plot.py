#!/usr/bin/env python3
"""Recreate a truthful four-category action plot from heuristic metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEURISTIC_TITLES = {
    "angle": "Greedy minimum-angle heuristic",
    "priority_angle": "Greedy angle-to-priority heuristic",
    "candidate_priority": "Greedy maximum-priority candidate heuristic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.metrics.read_text())
    meta = payload["meta"]
    summary = payload["summary"]

    labels = ["Charge", "Downlink", "Desat", "Imaging"]
    counts = [
        int(summary["charge_action_count"]),
        int(summary["downlink_action_count"]),
        int(summary["desat_action_count"]),
        int(summary["target_imaging_count"]),
    ]
    total = sum(counts)
    percentages = [100.0 * count / total for count in counts]

    cooldown_orbits = float(meta["reimage_cooldown_orbits"])
    cooldown = (
        "ground-confirmation re-imaging"
        if cooldown_orbits == 0.0
        else f"{cooldown_orbits:g}-orbit cooldown"
    )
    mode = str(meta["heuristic_mode"])
    controller = (
        "safety shield"
        if meta.get("heuristic_shield_only")
        else "resource-rule controller and safety shield"
    )
    title = (
        f"{HEURISTIC_TITLES.get(mode, f'Greedy {mode} heuristic')} with "
        f"{controller} - seed {meta['seed']}, {meta['n_targets']} targets, {cooldown}"
    )

    fig, left = plt.subplots(figsize=(10, 5))
    left.bar(labels, counts, color="skyblue")
    left.set_ylabel("Number of Times Action Was Taken", color="skyblue")
    left.tick_params(axis="y", labelcolor="black")
    left.tick_params(axis="x", rotation=45)
    left.grid(True, axis="y", linestyle="--", alpha=0.6)

    right = left.twinx()
    right.bar(labels, percentages, color="mediumseagreen", alpha=0.0)
    right.set_ylabel("Percentage of Total Actions (%)", color="mediumseagreen")
    right.tick_params(axis="y", labelcolor="black")
    left.set_title(title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
