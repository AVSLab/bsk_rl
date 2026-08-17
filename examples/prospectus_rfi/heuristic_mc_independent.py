#!/usr/bin/env python3
"""One-seed-per-task runner used to complete an interrupted heuristic campaign."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.evaluate import run_episode
from examples.prospectus_rfi.heuristic_mc import (
    atomic_write_csv,
    atomic_write_json,
    seed_stem,
)
from examples.prospectus_rfi.heuristic_mc_design import (
    CANDIDATE_COUNT,
    CATALOG_SIZES,
    HEURISTIC_MODE,
    METHOD,
    SEED_START,
    SEED_STOP_INCLUSIVE,
)

SEEDS_PER_CATALOG = SEED_STOP_INCLUSIVE - SEED_START + 1
TOTAL_INDEPENDENT_TASKS = len(CATALOG_SIZES) * SEEDS_PER_CATALOG


@dataclass(frozen=True)
class IndependentHeuristicTask:
    task_id: int
    catalog_size: int
    seed: int


def independent_task_spec(task_id: int) -> IndependentHeuristicTask:
    if not 0 <= task_id < TOTAL_INDEPENDENT_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_INDEPENDENT_TASKS - 1}]")
    catalog_index, seed_offset = divmod(task_id, SEEDS_PER_CATALOG)
    return IndependentHeuristicTask(
        task_id=task_id,
        catalog_size=CATALOG_SIZES[catalog_index],
        seed=SEED_START + seed_offset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ["BSK_RL_HEURISTIC_MC_OUTPUT_ROOT"])
        if "BSK_RL_HEURISTIC_MC_OUTPUT_ROOT" in os.environ
        else None,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root is None:
        raise SystemExit("provide --output-root or set BSK_RL_HEURISTIC_MC_OUTPUT_ROOT")
    task = independent_task_spec(args.task_id)
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
    task_metadata = {
        **asdict(task),
        "method": METHOD,
        "heuristic_mode": HEURISTIC_MODE,
        "information_scope": "full_visible_eligible_catalog",
        "candidate_count": CANDIDATE_COUNT,
        "shield_enabled": True,
        "recovery_mode": "one_seed_per_slurm_task",
        "study_config": study.to_dict(),
        "git": git_metadata(Path.cwd()),
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

    status = {
        **task_metadata,
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_json(status_path, status)
    started_clock = time.monotonic()
    try:
        print(
            f"RUN recovery closest-angle AMOS2025 N={task.catalog_size} "
            f"seed={task.seed}",
            flush=True,
        )
        metrics = run_episode(
            study,
            method=METHOD,
            seed=task.seed,
            catalog_size=task.catalog_size,
            learned_policy=None,
            shield=True,
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
                "campaign": "amos2025_closest_angle_mc_100s",
                "heuristic_mode": HEURISTIC_MODE,
                "information_scope": "full_visible_eligible_catalog",
                "seed_block": task.seed // 10,
                "slurm_array_task_id": task.task_id,
                "recovery_mode": "one_seed_per_slurm_task",
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
