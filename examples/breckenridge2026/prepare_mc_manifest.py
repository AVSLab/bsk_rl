#!/usr/bin/env python3
"""Freeze the two Breckenridge policies used by the 2x2 Monte Carlo study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile


REQUIRED_MODULE_FILES = (
    "class_and_ctor_args.pkl",
    "metadata.json",
    "module_state.pt",
)


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def valid_checkpoint(path: Path) -> bool:
    module_dir = path / "learner_group" / "learner" / "rl_module" / "inspector"
    return path.is_dir() and all(
        (module_dir / filename).is_file() for filename in REQUIRED_MODULE_FILES
    )


def resolve_checkpoint(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if valid_checkpoint(path):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {path}")

    candidates = [
        candidate
        for candidate in path.rglob("checkpoint_[0-9]*")
        if valid_checkpoint(candidate) and checkpoint_iteration(candidate) >= 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No numeric RLlib inspector checkpoint found below {path}"
        )
    return max(candidates, key=checkpoint_iteration)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leo-policy", required=True)
    parser.add_argument("--mixed-policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    leo_checkpoint = resolve_checkpoint(args.leo_policy)
    mixed_checkpoint = resolve_checkpoint(args.mixed_policy)
    policies = {
        "leo_trained": {
            "name": "breckenridge2026_leo_trained_10d90i",
            "training_environment": "leo",
            "checkpoint": str(leo_checkpoint),
            "checkpoint_iteration": checkpoint_iteration(leo_checkpoint),
        },
        "mixed_trained": {
            "name": "breckenridge2026_mixed_trained_10d90i",
            "training_environment": "mixed",
            "checkpoint": str(mixed_checkpoint),
            "checkpoint_iteration": checkpoint_iteration(mixed_checkpoint),
        },
    }
    cells = {
        f"{policy_key}__{evaluation_environment}_eval": {
            "policy": policy_key,
            "evaluation_environment": evaluation_environment,
        }
        for policy_key in policies
        for evaluation_environment in ("leo", "mixed")
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "study": "breckenridge2026_leo_vs_mixed_training_2x2",
        "seeds": list(range(100)),
        "policies": policies,
        "cells": cells,
        "evaluation": {
            "source": "examples/policy_evaluation_2026.py",
            "reward_mix": "10d90i",
            "obs_v": 7,
            "n_targets": 100,
            "n_targets_ahead": 10,
            "total_time_sec": 45000.0,
            "mixed_weights": {"LEO": 0.5, "MEO": 0.3, "GEO": 0.2},
            "shield_enabled": True,
            "uniform_target_priority": 1.0,
            "action_durations_sec": {
                "image": 300.0,
                "charge": 300.0,
                "downlink": 180.0,
                "desat": 150.0,
            },
            "fast_imaging": False,
            "fast_downlink": False,
            "dynamic_priority_event": False,
            "hio_shio": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    atomic_write_json(output, manifest)
    print(f"Wrote frozen manifest: {output}")
    print(f"LEO-trained checkpoint:   {leo_checkpoint}")
    print(f"Mixed-trained checkpoint: {mixed_checkpoint}")


if __name__ == "__main__":
    main()
