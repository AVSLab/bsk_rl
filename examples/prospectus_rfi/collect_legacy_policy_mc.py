#!/usr/bin/env python3
"""Validate and combine the frozen-policy transfer Monte Carlo campaign."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from examples.prospectus_rfi.collect_heuristic_mc import (
    load_episode_rows,
    summarize,
)
from examples.prospectus_rfi.legacy_policy_mc_design import (
    CANDIDATE_COUNT,
    CATALOG_SIZES,
    EVALUATION_IMAGING_DURATION_S,
    EXPECTED_MODULE_STATE_SHA256,
    METHOD,
    POLICY_BEST_ITERATION,
    SEED_START,
    SEED_STOP_INCLUSIVE,
    TRAINED_IMAGING_DURATION_S,
)

EXPECTED_PAIRS = {
    (catalog_size, seed)
    for catalog_size in CATALOG_SIZES
    for seed in range(SEED_START, SEED_STOP_INCLUSIVE + 1)
}


def validate_campaign(frame) -> dict[str, object]:
    required = {
        "catalog_size",
        "scenario_seed",
        "scenario_fingerprint",
        "method",
        "candidate_count",
        "shield_enabled",
        "episode_duration_s",
        "policy_best_iteration",
        "policy_module_state_sha256",
        "policy_training_imaging_duration_s",
        "evaluation_imaging_duration_s",
        "observation_contract",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    pairs = list(
        zip(
            frame["catalog_size"].astype(int),
            frame["scenario_seed"].astype(int),
            strict=True,
        )
    )
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    missing = sorted(EXPECTED_PAIRS - set(pairs))
    unexpected = sorted(set(pairs) - EXPECTED_PAIRS)
    errors: list[str] = []
    if duplicates:
        errors.append(f"duplicate pairs: {duplicates[:10]}")
    if missing:
        errors.append(f"missing pairs: {missing[:10]} (total={len(missing)})")
    if unexpected:
        errors.append(f"unexpected pairs: {unexpected[:10]} (total={len(unexpected)})")
    exact_values = {
        "method": {METHOD},
        "candidate_count": {CANDIDATE_COUNT},
        "policy_best_iteration": {POLICY_BEST_ITERATION},
        "policy_module_state_sha256": {EXPECTED_MODULE_STATE_SHA256},
    }
    for column, expected in exact_values.items():
        actual = set(frame[column].dropna().tolist())
        if actual != expected:
            errors.append(f"{column} values are {actual!r}, expected {expected!r}")
    if set(frame["shield_enabled"].astype(str).str.lower()) != {"true"}:
        errors.append("shield_enabled must be true for every episode")
    if not np.allclose(frame["episode_duration_s"].astype(float), 45_000.0):
        errors.append("one or more episodes did not reach 45,000 seconds")
    if not np.allclose(
        frame["policy_training_imaging_duration_s"].astype(float),
        TRAINED_IMAGING_DURATION_S,
    ):
        errors.append("policy training imaging duration is not consistently 300 s")
    if not np.allclose(
        frame["evaluation_imaging_duration_s"].astype(float),
        EVALUATION_IMAGING_DURATION_S,
    ):
        errors.append("evaluation imaging duration is not consistently 100 s")
    if errors:
        raise ValueError("campaign validation failed:\n- " + "\n- ".join(errors))
    return {
        "complete": True,
        "episode_count": len(frame.index),
        "catalog_sizes": list(CATALOG_SIZES),
        "seeds_per_catalog_size": SEED_STOP_INCLUSIVE - SEED_START + 1,
        "seed_start": SEED_START,
        "seed_stop_inclusive": SEED_STOP_INCLUSIVE,
    }


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
    frame = frame.sort_values(["catalog_size", "scenario_seed"]).reset_index(drop=True)
    frame.to_csv(analysis_dir / "episodes_combined.csv", index=False)
    try:
        frame.to_parquet(analysis_dir / "episodes_combined.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        pass
    summarize(frame, method=METHOD).to_csv(
        analysis_dir / "summary_statistics.csv", index=False
    )
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
