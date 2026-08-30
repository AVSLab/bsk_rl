#!/usr/bin/env python3
"""One paired seed per task for the matched AMOS 2025 300-second comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from examples.prospectus_rfi.amos2025_matched_300s_design import (
    BRECKENRIDGE_ALPHA0_CHECKPOINT_ITERATION,
    BRECKENRIDGE_ALPHA0_MODULE_STATE_SHA256,
    BRECKENRIDGE_ALPHA0_TRAINABLE_PARAMETERS,
    CANDIDATE_COUNT,
    CATALOG_SIZE,
    METHODS,
    SEEDS_PER_METHOD,
    SEED_START,
    TOTAL_TASKS,
)
from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.environment import (
    AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT,
    BRECKENRIDGE2026_OBSERVATION_CONTRACT,
    BRECKENRIDGE2026_OBSERVATION_SIZE,
)
from examples.prospectus_rfi.evaluate import load_policy, run_episode
from examples.prospectus_rfi.heuristic_mc import atomic_write_csv, atomic_write_json
from examples.prospectus_rfi.legacy_policy_mc import module_state_sha256


@dataclass(frozen=True)
class Matched300sTask:
    task_id: int
    method: str
    seed: int


def task_spec(task_id: int) -> Matched300sTask:
    """Map 400 array tasks to four methods and the same 100 seeds."""

    if not 0 <= task_id < TOTAL_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS - 1}]")
    method_index, seed_offset = divmod(task_id, SEEDS_PER_METHOD)
    return Matched300sTask(
        task_id=task_id,
        method=METHODS[method_index],
        seed=SEED_START + seed_offset,
    )


def validate_attention_policy(metadata: dict[str, Any]) -> dict[str, Any]:
    """Reject a checkpoint that cannot implement the K=10 control policy."""

    errors = []
    if metadata.get("observation_shape") != [97]:
        errors.append(
            f"observation shape is {metadata.get('observation_shape')}, expected [97]"
        )
    if metadata.get("action_count") != 13:
        errors.append(f"action count is {metadata.get('action_count')}, expected 13")
    if errors:
        raise ValueError("wrong 300-second attention artifact:\n- " + "\n- ".join(errors))
    return metadata


def validate_breckenridge_alpha0_policy(
    checkpoint: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Require the exact 0d100i policy bundled with the GNC paper snapshot."""

    module_path, digest = module_state_sha256(checkpoint)
    expected = {
        "observation_shape": [BRECKENRIDGE2026_OBSERVATION_SIZE],
        "action_count": 13,
        "trainable_parameters": BRECKENRIDGE_ALPHA0_TRAINABLE_PARAMETERS,
    }
    errors = [
        f"{key} is {metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if digest != BRECKENRIDGE_ALPHA0_MODULE_STATE_SHA256:
        errors.append(
            f"module_state.pt SHA-256 is {digest}, expected "
            f"{BRECKENRIDGE_ALPHA0_MODULE_STATE_SHA256}"
        )
    if errors:
        raise ValueError(
            "wrong Breckenridge 2026 alpha=0 policy artifact:\n- "
            + "\n- ".join(errors)
        )
    return {
        **metadata,
        "module_path": str(module_path),
        "module_state_sha256": digest,
        "paper_policy_label": "0d100i",
        "selected_checkpoint_iteration": BRECKENRIDGE_ALPHA0_CHECKPOINT_ITERATION,
        "training_date": "2025-10-14",
        "training_distribution": "LEO-only",
        "observation_version": 7,
    }


def _load_task_policy(
    task: Matched300sTask,
    legacy_checkpoint: Path | None,
    attention_checkpoint: Path | None,
):
    if task.method == "breckenridge2026_alpha0_mlp":
        if legacy_checkpoint is None:
            raise ValueError("the Breckenridge alpha=0 MLP checkpoint is required")
        policy, metadata = load_policy(legacy_checkpoint)
        return policy, validate_breckenridge_alpha0_policy(
            legacy_checkpoint, metadata
        )
    if task.method == "target_set_attention":
        if attention_checkpoint is None:
            raise ValueError("the 300-second attention checkpoint is required")
        policy, metadata = load_policy(attention_checkpoint)
        return policy, validate_attention_policy(metadata)
    return None, None


def _execution_contract(method: str) -> tuple[str, str]:
    if method == "breckenridge2026_alpha0_mlp":
        return "mlp", BRECKENRIDGE2026_OBSERVATION_CONTRACT
    if method == "target_set_attention":
        return "attention", AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT
    if method == "smallest_angle_heuristic":
        return "heuristic_historical", AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT
    if method == "closest_distance_heuristic":
        return (
            "heuristic_distance_historical",
            AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT,
        )
    raise ValueError(f"unsupported method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        default=(
            Path(os.environ["BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT"])
            if "BSK_RL_BRECKENRIDGE_ALPHA0_CHECKPOINT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--attention-checkpoint",
        type=Path,
        default=(
            Path(os.environ["BSK_RL_AMOS2025_ATTENTION_CHECKPOINT"])
            if "BSK_RL_AMOS2025_ATTENTION_CHECKPOINT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "BSK_RL_AMOS2025_MATCHED_300S_OUTPUT_ROOT",
                f"/scratch/alpine/{os.environ.get('USER', 'unknown')}/"
                "prospectus_rfi/amos2025_matched_300s",
            )
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = task_spec(args.task_id)
    root = Path(__file__).resolve().parent
    study = load_study_config(
        root / "configs" / "attention_amos2025_control.yaml",
        root / "configs" / "base_amos2025_attention_control.yaml",
    )
    study.validate()
    execution_method, observation_contract = _execution_contract(task.method)
    metadata = {
        **asdict(task),
        "campaign": "amos2025_matched_300s_rfi",
        "catalog_size": CATALOG_SIZE,
        "candidate_count": CANDIDATE_COUNT,
        "information_scope": (
            "full_visible_eligible_catalog"
            if task.method.endswith("heuristic")
            else "10_candidate_policy_observation"
        ),
        "heuristic_mode": (
            "angle"
            if task.method == "smallest_angle_heuristic"
            else "distance"
            if task.method == "closest_distance_heuristic"
            else None
        ),
        "observation_contract": observation_contract,
        "shield_enabled": True,
        "wheel_guard_enabled": False,
        "study_config": study.to_dict(),
        "git": git_metadata(Path.cwd()),
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    policy, policy_metadata = _load_task_policy(
        task, args.legacy_checkpoint, args.attention_checkpoint
    )
    metadata["policy"] = policy_metadata
    output_root = args.output_root.resolve()
    stem = f"{task.method}_seed{task.seed:03d}"
    csv_path = output_root / "raw" / task.method / f"{stem}.csv"
    metadata_path = output_root / "raw" / task.method / f"{stem}.metadata.json"
    status_path = output_root / "status" / task.method / f"seed_{task.seed:03d}.json"
    if csv_path.is_file() and metadata_path.is_file() and status_path.is_file():
        try:
            prior = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            prior = {}
        if prior.get("state") == "completed":
            print(f"SKIP completed method={task.method} seed={task.seed}", flush=True)
            return 0

    started_clock = time.monotonic()
    status = {
        **metadata,
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_json(status_path, status)
    try:
        metrics = run_episode(
            study,
            method=execution_method,
            seed=task.seed,
            catalog_size=CATALOG_SIZE,
            learned_policy=policy,
            shield=True,
            wheel_guard=False,
            observation_contract=observation_contract,
        )
        if not np.isclose(metrics["episode_duration_s"], 45_000.0):
            raise RuntimeError(
                f"episode ended at {metrics['episode_duration_s']} s, expected 45000 s"
            )
        metrics.update(
            {
                "method": task.method,
                "campaign": "amos2025_matched_300s_rfi",
                "heuristic_mode": metadata["heuristic_mode"],
                "information_scope": metadata["information_scope"],
                "imaging_duration_s": study.environment.imaging_duration_s,
                "charge_duration_s": study.environment.charge_duration_s,
                "downlink_duration_s": study.environment.downlink_duration_s,
                "desaturation_duration_s": study.environment.desaturation_duration_s,
                "reward_alpha": study.environment.alpha,
                "slurm_array_task_id": task.task_id,
            }
        )
        if policy_metadata is not None:
            metrics["policy_checkpoint"] = policy_metadata["checkpoint"]
        atomic_write_csv(csv_path, metrics)
        try:
            pd.DataFrame([metrics]).to_parquet(csv_path.with_suffix(".parquet"), index=False)
        except (ImportError, ModuleNotFoundError):
            pass
        atomic_write_json(
            metadata_path,
            {
                **metadata,
                "output_csv": str(csv_path),
                "scenario_fingerprint": metrics["scenario_fingerprint"],
            },
        )
        status.update(
            {
                "state": "completed",
                "finished_at_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "elapsed_seconds": time.monotonic() - started_clock,
                "output_csv": str(csv_path),
            }
        )
        atomic_write_json(status_path, status)
        print(
            f"PASS method={task.method} seed={task.seed} "
            f"observed={metrics['successful_observations']}",
            flush=True,
        )
        return 0
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "finished_at_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "elapsed_seconds": time.monotonic() - started_clock,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        print(f"FAIL method={task.method} seed={task.seed}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
