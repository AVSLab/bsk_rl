#!/usr/bin/env python3
"""Validate and combine the AMOS 2025 closest-angle Monte Carlo campaign."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from examples.prospectus_rfi.heuristic_mc import CATALOG_SIZES, METHOD

EXPECTED_SEEDS = range(100)
EXPECTED_PAIRS = {
    (catalog_size, seed) for catalog_size in CATALOG_SIZES for seed in EXPECTED_SEEDS
}
SUMMARY_METRICS = (
    "episode_reward",
    "successful_observations",
    "illuminated_observations",
    "successful_observation_fraction",
    "illuminated_observation_fraction",
    "useful_deliveries",
    "onboard_backlog_bits",
    "onboard_backlog_fraction",
    "initial_battery_fraction",
    "final_battery_fraction",
    "survival_fraction",
    "image_action_count",
    "charge_action_count",
    "downlink_action_count",
    "desaturation_action_count",
    "resource_constraint_interventions",
    "constraint_intervention_rate",
    "mean_inference_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def load_episode_rows(input_root: Path) -> pd.DataFrame:
    paths = sorted((input_root / "raw").glob("n*/*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no per-seed CSV files found under {input_root / 'raw'}"
        )
    rows: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if len(frame.index) != 1:
            raise ValueError(
                f"{path} contains {len(frame.index)} rows; expected exactly one"
            )
        frame["source_file"] = str(path.resolve())
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def validate_campaign(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "catalog_size",
        "scenario_seed",
        "method",
        "heuristic_mode",
        "information_scope",
        "candidate_count",
        "shield_enabled",
        "episode_duration_s",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    pairs = [
        (int(catalog_size), int(seed))
        for catalog_size, seed in zip(
            frame["catalog_size"], frame["scenario_seed"], strict=True
        )
    ]
    duplicate_pairs = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    actual_pairs = set(pairs)
    missing_pairs = sorted(EXPECTED_PAIRS - actual_pairs)
    unexpected_pairs = sorted(actual_pairs - EXPECTED_PAIRS)
    errors: list[str] = []
    if duplicate_pairs:
        errors.append(f"duplicate pairs: {duplicate_pairs[:10]}")
    if missing_pairs:
        errors.append(
            f"missing pairs: {missing_pairs[:10]} (total={len(missing_pairs)})"
        )
    if unexpected_pairs:
        errors.append(
            f"unexpected pairs: {unexpected_pairs[:10]} (total={len(unexpected_pairs)})"
        )
    expected_values = {
        "method": {METHOD},
        "heuristic_mode": {"angle"},
        "information_scope": {"full_visible_eligible_catalog"},
        "candidate_count": {10},
    }
    for column, expected in expected_values.items():
        actual = set(frame[column].dropna().tolist())
        if actual != expected:
            errors.append(
                f"{column} values are {sorted(actual)!r}, expected {sorted(expected)!r}"
            )
    shield_values = set(frame["shield_enabled"].astype(str).str.lower())
    if shield_values != {"true"}:
        errors.append(
            f"shield_enabled values are {sorted(shield_values)!r}, expected ['true']"
        )
    if not np.allclose(frame["episode_duration_s"].astype(float), 45_000.0):
        errors.append("one or more episodes did not reach exactly 45,000 seconds")
    if errors:
        raise ValueError("campaign validation failed:\n- " + "\n- ".join(errors))
    return {
        "complete": True,
        "episode_count": len(frame.index),
        "catalog_sizes": list(CATALOG_SIZES),
        "seeds_per_catalog_size": len(EXPECTED_SEEDS),
        "seed_start": min(EXPECTED_SEEDS),
        "seed_stop_inclusive": max(EXPECTED_SEEDS),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for catalog_size, group in frame.groupby("catalog_size", sort=True):
        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "method": METHOD,
                    "catalog_size": int(catalog_size),
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    frame = load_episode_rows(input_root)
    completion = validate_campaign(frame)
    analysis_dir = input_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["catalog_size", "scenario_seed"]).reset_index(drop=True)
    frame.to_csv(analysis_dir / "episodes_combined.csv", index=False)
    try:
        frame.to_parquet(analysis_dir / "episodes_combined.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        pass
    summary = summarize(frame)
    summary.to_csv(analysis_dir / "summary_statistics.csv", index=False)
    completion.update(
        {
            "validated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "combined_csv": str((analysis_dir / "episodes_combined.csv").resolve()),
            "summary_csv": str((analysis_dir / "summary_statistics.csv").resolve()),
        }
    )
    (analysis_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
