#!/usr/bin/env python3
"""Validate, combine, and summarize the matched 300-second comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from examples.prospectus_rfi.amos2025_matched_300s_design import (
    CANDIDATE_COUNT,
    CATALOG_SIZE,
    CHARGE_DURATION_S,
    DESATURATION_DURATION_S,
    DOWNLINK_DURATION_S,
    EPISODE_DURATION_S,
    IMAGING_DURATION_S,
    METHODS,
    REWARD_ALPHA,
    SEED_START,
    SEED_STOP_INCLUSIVE,
    TOTAL_TASKS,
)
from examples.prospectus_rfi.collect_heuristic_mc import (
    SUMMARY_METRICS,
    load_episode_rows,
    summarize,
)


EXPECTED_PAIRS = {
    (method, seed)
    for method in METHODS
    for seed in range(SEED_START, SEED_STOP_INCLUSIVE + 1)
}


def validate_campaign(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "method",
        "scenario_seed",
        "scenario_fingerprint",
        "initial_battery_fraction",
        "catalog_size",
        "candidate_count",
        "shield_enabled",
        "wheel_guard_enabled",
        "episode_duration_s",
        "imaging_duration_s",
        "charge_duration_s",
        "downlink_duration_s",
        "desaturation_duration_s",
        "reward_alpha",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    pairs = list(
        zip(frame["method"], frame["scenario_seed"].astype(int), strict=True)
    )
    actual_pairs = set(pairs)
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    missing = sorted(EXPECTED_PAIRS - actual_pairs)
    unexpected = sorted(actual_pairs - EXPECTED_PAIRS)
    errors = []
    if duplicates:
        errors.append(f"duplicate method/seed pairs: {duplicates[:10]}")
    if missing:
        errors.append(f"missing method/seed pairs: {missing[:10]} (total={len(missing)})")
    if unexpected:
        errors.append(
            f"unexpected method/seed pairs: {unexpected[:10]} (total={len(unexpected)})"
        )
    exact_numeric = {
        "catalog_size": CATALOG_SIZE,
        "candidate_count": CANDIDATE_COUNT,
        "episode_duration_s": EPISODE_DURATION_S,
        "imaging_duration_s": IMAGING_DURATION_S,
        "charge_duration_s": CHARGE_DURATION_S,
        "downlink_duration_s": DOWNLINK_DURATION_S,
        "desaturation_duration_s": DESATURATION_DURATION_S,
        "reward_alpha": REWARD_ALPHA,
    }
    for column, expected in exact_numeric.items():
        if not np.allclose(frame[column].astype(float), expected):
            errors.append(f"{column} is not consistently {expected}")
    if set(frame["shield_enabled"].astype(str).str.lower()) != {"true"}:
        errors.append("shield_enabled must be true for every episode")
    if set(frame["wheel_guard_enabled"].astype(str).str.lower()) != {"false"}:
        errors.append("wheel_guard_enabled must be false for the historical comparison")
    fingerprints_per_seed = frame.groupby("scenario_seed")[
        "scenario_fingerprint"
    ].nunique()
    mismatched_seeds = fingerprints_per_seed[fingerprints_per_seed != 1].index.tolist()
    if mismatched_seeds:
        errors.append(
            "scenario fingerprints differ between methods for seeds "
            f"{mismatched_seeds[:10]}"
        )
    battery_spread = frame.groupby("scenario_seed")[
        "initial_battery_fraction"
    ].agg(lambda values: float(np.max(values) - np.min(values)))
    mismatched_battery_seeds = battery_spread[battery_spread > 1.0e-12].index.tolist()
    if mismatched_battery_seeds:
        errors.append(
            "initial battery differs between methods for seeds "
            f"{mismatched_battery_seeds[:10]}"
        )
    if errors:
        raise ValueError("campaign validation failed:\n- " + "\n- ".join(errors))
    return {
        "complete": True,
        "episode_count": len(frame.index),
        "expected_episode_count": TOTAL_TASKS,
        "methods": list(METHODS),
        "catalog_size": CATALOG_SIZE,
        "seed_start": SEED_START,
        "seed_stop_inclusive": SEED_STOP_INCLUSIVE,
        "paired_scenario_fingerprints": True,
        "paired_initial_battery": True,
    }


def paired_differences(frame: pd.DataFrame) -> pd.DataFrame:
    reference = "smallest_angle_heuristic"
    indexed = frame.set_index(["scenario_seed", "method"])
    rows = []
    for seed in range(SEED_START, SEED_STOP_INCLUSIVE + 1):
        reference_row = indexed.loc[(seed, reference)]
        for method in METHODS:
            if method == reference:
                continue
            method_row = indexed.loc[(seed, method)]
            row: dict[str, float | int | str] = {
                "scenario_seed": seed,
                "method": method,
                "reference_method": reference,
            }
            for metric in SUMMARY_METRICS:
                if metric in frame.columns:
                    row[f"delta_{metric}"] = float(method_row[metric]) - float(
                        reference_row[metric]
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    frame = load_episode_rows(input_root)
    completion = validate_campaign(frame)
    analysis_dir = input_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["method", "scenario_seed"]).reset_index(drop=True)
    frame.to_csv(analysis_dir / "episodes_combined.csv", index=False)
    summary_parts = [
        summarize(group, method=method)
        for method, group in frame.groupby("method", sort=False)
    ]
    pd.concat(summary_parts, ignore_index=True).to_csv(
        analysis_dir / "summary_statistics.csv", index=False
    )
    paired_differences(frame).to_csv(
        analysis_dir / "paired_differences_vs_smallest_angle.csv", index=False
    )
    completion.update(
        {
            "validated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "combined_csv": str(analysis_dir / "episodes_combined.csv"),
            "summary_csv": str(analysis_dir / "summary_statistics.csv"),
            "paired_differences_csv": str(
                analysis_dir / "paired_differences_vs_smallest_angle.csv"
            ),
        }
    )
    (analysis_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
