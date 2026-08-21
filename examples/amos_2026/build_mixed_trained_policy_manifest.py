#!/usr/bin/env python3
"""Build a frozen custom-policy specification for the mixed-fixed GAT sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
WANDB_GROUP = (
    "polaris-gat-full-actions-obs-v9-mixed-fixed-"
    "50leo30meo20geo-100targets-reward-sweep"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tags", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = {}
    for row in rows:
        if row.get("wandb_group", "").lower() != WANDB_GROUP:
            continue
        if row.get("fixed_target_count_100", "").lower() != "true":
            continue
        if not row.get("latest_checkpoint"):
            continue
        try:
            alpha = round(float(row["alpha"]), 6)
            rank = (
                int(float(row.get("progress_iteration") or -1)),
                int(float(row.get("latest_checkpoint_iteration") or -1)),
                row.get("run_dir", ""),
            )
        except (TypeError, ValueError):
            continue
        if alpha not in EXPECTED_ALPHAS:
            continue
        if alpha not in selected or rank > selected[alpha][0]:
            selected[alpha] = (rank, row)

    policies = {}
    tags = []
    for alpha in EXPECTED_ALPHAS:
        if alpha not in selected:
            continue
        row = selected[alpha][1]
        alpha_text = f"{alpha:g}".replace(".", "p")
        tag = f"mixed_a{alpha_text}"
        tags.append(tag)
        policies[tag] = {
            "checkpoint_dir": row["latest_checkpoint"],
            "alpha": alpha,
            "label": f"Mixed-trained alpha={alpha:g}",
            "training_run_dir": row["run_dir"],
            "training_iteration": row.get("progress_iteration"),
        }

    payload = {
        "schema_version": 1,
        "training_environment": "mixed_exact_50LEO_30MEO_20GEO_100targets",
        "required_wandb_group": WANDB_GROUP,
        "expected_alphas": list(EXPECTED_ALPHAS),
        "found_alphas": sorted(selected),
        "missing_alphas": [
            alpha for alpha in EXPECTED_ALPHAS if alpha not in selected
        ],
        "policies": policies,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tags.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_tags.write_text(",".join(tags) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if len(policies) == len(EXPECTED_ALPHAS) else 3


if __name__ == "__main__":
    raise SystemExit(main())
