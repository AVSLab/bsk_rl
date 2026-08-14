#!/usr/bin/env python3
"""Restartable AMOS 2025 closest-angle heuristic Monte Carlo blocks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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
from examples.prospectus_rfi.evaluate import run_episode
from examples.prospectus_rfi.heuristic_mc_design import (
    BLOCKS_PER_CATALOG,
    CANDIDATE_COUNT,
    CATALOG_SIZES,
    HEURISTIC_MODE,
    METHOD,
    SEEDS_PER_BLOCK,
    TOTAL_TASKS,
)


@dataclass(frozen=True)
class HeuristicMCTask:
    task_id: int
    catalog_size: int
    block_index: int
    seed_start: int
    seed_stop_inclusive: int

    @property
    def seeds(self) -> range:
        return range(self.seed_start, self.seed_stop_inclusive + 1)


def task_spec(task_id: int) -> HeuristicMCTask:
    """Map one 0-based Slurm task to one N and ten exact seeds."""

    if not 0 <= task_id < TOTAL_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS - 1}]")
    catalog_index, block_index = divmod(task_id, BLOCKS_PER_CATALOG)
    seed_start = block_index * SEEDS_PER_BLOCK
    return HeuristicMCTask(
        task_id=task_id,
        catalog_size=CATALOG_SIZES[catalog_index],
        block_index=block_index,
        seed_start=seed_start,
        seed_stop_inclusive=seed_start + SEEDS_PER_BLOCK - 1,
    )


def seed_stem(catalog_size: int, seed: int) -> str:
    return f"{METHOD}_n{catalog_size}_seed{seed:03d}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".csv", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        pd.DataFrame([row]).to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
        default=Path(
            os.environ.get(
                "BSK_RL_HEURISTIC_MC_OUTPUT_ROOT",
                f"/scratch/alpine/{os.environ.get('USER', 'unknown')}/"
                "prospectus_rfi/heuristic_mc/amos2025_closest_angle_100s",
            )
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        "heuristic_mode": HEURISTIC_MODE,
        "information_scope": "full_visible_eligible_catalog",
        "candidate_count": CANDIDATE_COUNT,
        "shield_enabled": True,
        "study_config": study.to_dict(),
        "git": repository,
    }
    if args.dry_run:
        print(json.dumps(task_metadata, indent=2, sort_keys=True))
        return 0

    block_status_path = output_root / "status" / f"task_{task.task_id:02d}.json"
    failures = 0
    for seed in task.seeds:
        stem = seed_stem(task.catalog_size, seed)
        raw_dir = output_root / "raw" / f"n{task.catalog_size}"
        csv_path = raw_dir / f"{stem}.csv"
        parquet_path = raw_dir / f"{stem}.parquet"
        metadata_path = raw_dir / f"{stem}.metadata.json"
        status_path = (
            output_root / "status" / f"n{task.catalog_size}" / f"seed_{seed:03d}.json"
        )
        if csv_path.is_file() and metadata_path.is_file() and status_path.is_file():
            try:
                status = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                status = {}
            if status.get("state") == "completed":
                print(f"SKIP completed N={task.catalog_size} seed={seed}", flush=True)
                continue

        started = datetime.now(timezone.utc)
        status = {
            **task_metadata,
            "seed": seed,
            "state": "running",
            "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        atomic_write_json(status_path, status)
        started_clock = time.monotonic()
        try:
            print(
                f"RUN closest-angle AMOS2025 N={task.catalog_size} seed={seed}",
                flush=True,
            )
            metrics = run_episode(
                study,
                method=METHOD,
                seed=seed,
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
                    "seed_block": task.block_index,
                    "slurm_array_task_id": task.task_id,
                }
            )
            atomic_write_csv(csv_path, metrics)
            try:
                pd.DataFrame([metrics]).to_parquet(parquet_path, index=False)
            except (ImportError, ModuleNotFoundError):
                pass
            episode_metadata = {
                **task_metadata,
                "seed": seed,
                "output_csv": str(csv_path),
                "scenario_fingerprint": metrics["scenario_fingerprint"],
            }
            atomic_write_json(metadata_path, episode_metadata)
            finished = datetime.now(timezone.utc)
            status.update(
                {
                    "state": "completed",
                    "finished_at_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "elapsed_seconds": time.monotonic() - started_clock,
                    "output_csv": str(csv_path),
                }
            )
            atomic_write_json(status_path, status)
            print(
                f"PASS N={task.catalog_size} seed={seed} "
                f"observed={metrics['successful_observations']}",
                flush=True,
            )
        except Exception as error:
            failures += 1
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
            print(f"FAIL N={task.catalog_size} seed={seed}: {error}", flush=True)

    atomic_write_json(
        block_status_path,
        {
            **task_metadata,
            "state": "completed" if failures == 0 else "failed",
            "failure_count": failures,
            "finished_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
