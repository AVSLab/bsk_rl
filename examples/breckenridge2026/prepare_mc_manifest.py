#!/usr/bin/env python3
"""Freeze the mixed-trained policy used to complete the Monte Carlo comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


REQUIRED_MODULE_FILES = (
    "class_and_ctor_args.pkl",
    "metadata.json",
    "module_state.pt",
)
DEFAULT_MIXED_POLICY = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "breckenridge2026_mixed_10d90i"
    / "checkpoint_000160"
)
DEFAULT_LEO_POLICY = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "breckenridge2026_alpha_sweep"
    / "10d90i"
    / "checkpoint_000145"
)
ALPHA_SWEEP_LABELS = (
    "0d100i",
    "10d90i",
    "20d80i",
    "30d70i",
    "40d60i",
    "50d50i",
    "60d40i",
    "70d30i",
    "80d20i",
    "90d10i",
    "100d00i",
)


def default_alpha_policy(label: str) -> Path:
    policy_root = Path(__file__).resolve().parents[2] / "policies"
    alpha_root = policy_root / "breckenridge2026_alpha_sweep" / label
    candidates = [
        candidate
        for candidate in alpha_root.glob("checkpoint_[0-9]*")
        if valid_checkpoint(candidate) and checkpoint_iteration(candidate) >= 0
    ]
    if not candidates:
        raise FileNotFoundError(f"No bundled alpha-sweep checkpoint for {label}")
    return max(candidates, key=checkpoint_iteration)


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


def module_state_sha256(checkpoint: Path) -> str:
    state_path = (
        checkpoint
        / "learner_group"
        / "learner"
        / "rl_module"
        / "inspector"
        / "module_state.pt"
    )
    digest = hashlib.sha256()
    with state_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


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
    parser.add_argument(
        "--policy-set",
        choices=("mixed", "leo", "both", "alpha_sweep"),
        default="mixed",
    )
    parser.add_argument("--leo-policy", default=str(DEFAULT_LEO_POLICY))
    parser.add_argument("--mixed-policy", default=str(DEFAULT_MIXED_POLICY))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    policies = {}
    if args.policy_set in ("mixed", "both"):
        mixed_checkpoint = resolve_checkpoint(args.mixed_policy)
        policies["mixed_trained"] = {
            "name": "breckenridge2026_mixed_trained_10d90i",
            "training_environment": "mixed",
            "checkpoint": str(mixed_checkpoint),
            "checkpoint_iteration": checkpoint_iteration(mixed_checkpoint),
            "module_state_sha256": module_state_sha256(mixed_checkpoint),
        }
    if args.policy_set in ("leo", "both"):
        leo_checkpoint = resolve_checkpoint(args.leo_policy)
        policies["leo_trained"] = {
            "name": "breckenridge2026_leo_trained_10d90i",
            "training_environment": "leo",
            "checkpoint": str(leo_checkpoint),
            "checkpoint_iteration": checkpoint_iteration(leo_checkpoint),
            "module_state_sha256": module_state_sha256(leo_checkpoint),
        }
    if args.policy_set == "alpha_sweep":
        for label in ALPHA_SWEEP_LABELS:
            checkpoint = default_alpha_policy(label)
            downlink_weight = int(label.split("d", 1)[0]) / 100.0
            policies[f"leo_trained_{label}"] = {
                "name": f"breckenridge2026_leo_trained_{label}",
                "training_environment": "leo",
                "reward_mix": label,
                "downlink_reward_weight": downlink_weight,
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": checkpoint_iteration(checkpoint),
                "module_state_sha256": module_state_sha256(checkpoint),
            }

    if args.policy_set == "alpha_sweep":
        cells = {
            f"{policy_key}__mixed_eval": {
                "policy": policy_key,
                "evaluation_environment": "mixed",
            }
            for policy_key in policies
        }
    else:
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
        "study": f"breckenridge2026_{args.policy_set}_trained_evaluations",
        "repository": {
            "commit": git_value(repo_root, "rev-parse", "HEAD"),
            "branch": git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        },
        "seeds": list(range(100)),
        "policies": policies,
        "cells": cells,
        "evaluation": {
            "source": "examples/policy_evaluation_2026.py",
            "reward_mix": (
                "varies_by_policy" if args.policy_set == "alpha_sweep" else "10d90i"
            ),
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
    if args.policy_set in ("mixed", "both"):
        print(f"Mixed-trained checkpoint: {mixed_checkpoint}")
    if args.policy_set in ("leo", "both"):
        print(f"LEO-trained checkpoint:   {leo_checkpoint}")


if __name__ == "__main__":
    main()
