#!/usr/bin/env python3
"""One-seed-per-task evaluation of the frozen AMOS 2025 alpha=0 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.environment import (
    LEGACY_AMOS2025_OBSERVATION_CONTRACT,
    legacy_amos2025_policy_contract,
)
from examples.prospectus_rfi.evaluate import (
    find_inspector_module,
    load_policy,
    run_episode,
)
from examples.prospectus_rfi.heuristic_mc import atomic_write_csv, atomic_write_json
from examples.prospectus_rfi.legacy_policy_mc_design import (
    CANDIDATE_COUNT,
    CATALOG_SIZES,
    EVALUATION_IMAGING_DURATION_S,
    EXPECTED_ACTION_COUNT,
    EXPECTED_MODULE_STATE_SHA256,
    EXPECTED_OBSERVATION_SIZE,
    EXPECTED_TRAINABLE_PARAMETERS,
    METHOD,
    POLICY_BEST_ITERATION,
    POLICY_FAMILY,
    SEED_START,
    SEEDS_PER_CATALOG,
    TOTAL_TASKS,
    TRAINED_IMAGING_DURATION_S,
)


@dataclass(frozen=True)
class LegacyPolicyMCTask:
    task_id: int
    catalog_size: int
    seed: int


def task_spec(task_id: int) -> LegacyPolicyMCTask:
    """Map 300 Slurm tasks bijectively to N={100,200,400} and seeds 0..99."""

    if not 0 <= task_id < TOTAL_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS - 1}]")
    catalog_index, seed_offset = divmod(task_id, SEEDS_PER_CATALOG)
    return LegacyPolicyMCTask(
        task_id=task_id,
        catalog_size=CATALOG_SIZES[catalog_index],
        seed=SEED_START + seed_offset,
    )


def module_state_sha256(checkpoint: Path) -> tuple[Path, str]:
    module_path = find_inspector_module(checkpoint)
    state_path = module_path / "module_state.pt"
    digest = hashlib.sha256()
    with state_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return module_path, digest.hexdigest()


def validate_frozen_policy(
    checkpoint: Path, checkpoint_metadata: dict[str, Any]
) -> dict[str, Any]:
    module_path, state_sha256 = module_state_sha256(checkpoint)
    errors: list[str] = []
    if state_sha256 != EXPECTED_MODULE_STATE_SHA256:
        errors.append(
            f"module_state.pt SHA-256 is {state_sha256}, expected "
            f"{EXPECTED_MODULE_STATE_SHA256}"
        )
    if checkpoint_metadata.get("observation_shape") != [EXPECTED_OBSERVATION_SIZE]:
        errors.append(
            f"observation shape is {checkpoint_metadata.get('observation_shape')}, "
            f"expected [{EXPECTED_OBSERVATION_SIZE}]"
        )
    if checkpoint_metadata.get("action_count") != EXPECTED_ACTION_COUNT:
        errors.append(
            f"action count is {checkpoint_metadata.get('action_count')}, "
            f"expected {EXPECTED_ACTION_COUNT}"
        )
    if checkpoint_metadata.get("trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS:
        errors.append(
            "trainable parameter count is "
            f"{checkpoint_metadata.get('trainable_parameters')}, expected "
            f"{EXPECTED_TRAINABLE_PARAMETERS}"
        )
    if errors:
        raise ValueError("wrong AMOS 2025 policy artifact:\n- " + "\n- ".join(errors))
    return {
        **checkpoint_metadata,
        "module_path": str(module_path),
        "module_state_sha256": state_sha256,
        "policy_family": POLICY_FAMILY,
        "best_training_iteration": POLICY_BEST_ITERATION,
    }


def seed_stem(catalog_size: int, seed: int) -> str:
    return f"{METHOD}_n{catalog_size}_seed{seed:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ["BSK_RL_AMOS2025_POLICY_CHECKPOINT"])
        if "BSK_RL_AMOS2025_POLICY_CHECKPOINT" in os.environ
        else None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT",
                f"/scratch/alpine/{os.environ.get('USER', 'unknown')}/"
                "prospectus_rfi/legacy_policy_mc/amos2025_alpha0_300s_to_100s",
            )
        ),
    )
    parser.add_argument("--validate-checkpoint-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.checkpoint is None:
        raise SystemExit(
            "provide --checkpoint or set BSK_RL_AMOS2025_POLICY_CHECKPOINT"
        )
    learned_policy, raw_checkpoint_metadata = load_policy(args.checkpoint)
    checkpoint_metadata = validate_frozen_policy(
        args.checkpoint, raw_checkpoint_metadata
    )
    if args.validate_checkpoint_only:
        print(json.dumps(checkpoint_metadata, indent=2, sort_keys=True))
        return 0

    task = task_spec(args.task_id)
    output_root = args.output_root.resolve()
    root = Path(__file__).resolve().parent
    study = load_study_config(
        root / "configs" / "mlp_selected.yaml",
        root / "configs" / "base.yaml",
    )
    study = replace(
        study,
        environment=replace(study.environment, candidate_count=CANDIDATE_COUNT),
    )
    study.validate()
    repository = git_metadata(Path.cwd())
    task_metadata = {
        **asdict(task),
        "method": METHOD,
        "policy": checkpoint_metadata,
        "policy_training": {
            "catalog_size": 100,
            "candidate_count": 10,
            "imaging_duration_s": TRAINED_IMAGING_DURATION_S,
            "reward_alpha": 0.0,
            "observation_contract": LEGACY_AMOS2025_OBSERVATION_CONTRACT,
        },
        "evaluation": {
            "imaging_duration_s": EVALUATION_IMAGING_DURATION_S,
            "episode_duration_s": study.environment.episode_duration_s,
            "catalog_size": task.catalog_size,
            "candidate_count": CANDIDATE_COUNT,
            "shield_enabled": True,
        },
        "deliberate_transfer_mismatch": (
            "The frozen policy was trained at N=100 with 300-second imaging; "
            "it is evaluated without retraining at N in {100,200,400} with "
            "100-second imaging in the current matched heuristic environment."
        ),
        "observation_contract": legacy_amos2025_policy_contract(),
        "study_config": study.to_dict(),
        "git": repository,
    }
    if args.dry_run:
        print(json.dumps(task_metadata, indent=2, sort_keys=True))
        return 0

    stem = seed_stem(task.catalog_size, task.seed)
    raw_dir = output_root / "raw" / f"n{task.catalog_size}"
    csv_path = raw_dir / f"{stem}.csv"
    parquet_path = raw_dir / f"{stem}.parquet"
    metadata_path = raw_dir / f"{stem}.metadata.json"
    status_path = (
        output_root / "status" / f"n{task.catalog_size}" / f"seed_{task.seed:03d}.json"
    )
    if csv_path.is_file() and metadata_path.is_file() and status_path.is_file():
        try:
            prior_status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            prior_status = {}
        if prior_status.get("state") == "completed":
            print(f"SKIP completed N={task.catalog_size} seed={task.seed}", flush=True)
            return 0

    started = datetime.now(timezone.utc)
    status = {
        **task_metadata,
        "state": "running",
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_json(status_path, status)
    started_clock = time.monotonic()
    try:
        print(
            f"RUN frozen AMOS2025 alpha0 policy N={task.catalog_size} "
            f"seed={task.seed} trained_image_s={TRAINED_IMAGING_DURATION_S:g} "
            f"evaluation_image_s={EVALUATION_IMAGING_DURATION_S:g}",
            flush=True,
        )
        metrics = run_episode(
            study,
            method=METHOD,
            seed=task.seed,
            catalog_size=task.catalog_size,
            learned_policy=learned_policy,
            shield=True,
            observation_contract=LEGACY_AMOS2025_OBSERVATION_CONTRACT,
        )
        if not np.isclose(
            metrics["episode_duration_s"], study.environment.episode_duration_s
        ):
            raise RuntimeError(
                f"episode ended at {metrics['episode_duration_s']} s, expected "
                f"{study.environment.episode_duration_s} s"
            )
        metrics.update(
            {
                "campaign": "amos2025_alpha0_policy_transfer_300s_to_100s",
                "policy_family": POLICY_FAMILY,
                "policy_best_iteration": POLICY_BEST_ITERATION,
                "policy_module_state_sha256": EXPECTED_MODULE_STATE_SHA256,
                "policy_training_imaging_duration_s": TRAINED_IMAGING_DURATION_S,
                "evaluation_imaging_duration_s": EVALUATION_IMAGING_DURATION_S,
                "policy_training_catalog_size": 100,
                "information_scope": "historical_10_candidate_policy_observation",
                "slurm_array_task_id": task.task_id,
            }
        )
        atomic_write_csv(csv_path, metrics)
        try:
            pd.DataFrame([metrics]).to_parquet(parquet_path, index=False)
        except (ImportError, ModuleNotFoundError):
            pass
        atomic_write_json(
            metadata_path,
            {
                **task_metadata,
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
            f"PASS N={task.catalog_size} seed={task.seed} "
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
        print(f"FAIL N={task.catalog_size} seed={task.seed}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
