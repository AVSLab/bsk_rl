#!/usr/bin/env python3
"""Replay completed paired episodes and record acquisition histories only."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from examples.prospectus_rfi.acquisition_timeline import (
    EPISODE_DURATION_S,
    timeline_stem,
    timeline_task_spec,
    verify_replay,
)
from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.environment import LEGACY_AMOS2025_OBSERVATION_CONTRACT
from examples.prospectus_rfi.evaluate import load_policy, run_episode
from examples.prospectus_rfi.heuristic_mc import atomic_write_json
from examples.prospectus_rfi.legacy_policy_mc import validate_frozen_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--heuristic-root",
        type=Path,
        default=(
            Path(os.environ["BSK_RL_HEURISTIC_MC_OUTPUT_ROOT"])
            if "BSK_RL_HEURISTIC_MC_OUTPUT_ROOT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=(
            Path(os.environ["BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT"])
            if "BSK_RL_LEGACY_POLICY_MC_OUTPUT_ROOT" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path(os.environ["BSK_RL_AMOS2025_POLICY_CHECKPOINT"])
            if "BSK_RL_AMOS2025_POLICY_CHECKPOINT" in os.environ
            else None
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".csv", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.heuristic_root is None or args.policy_root is None:
        raise SystemExit("provide both completed heuristic and policy MC roots")
    task = timeline_task_spec(args.task_id)
    output_root = (
        args.heuristic_root.resolve()
        if task.method == "heuristic_historical"
        else args.policy_root.resolve()
    )
    if task.method == "legacy_amos2025_alpha0_policy" and args.checkpoint is None:
        raise SystemExit("the legacy policy timeline replay requires --checkpoint")

    root = Path(__file__).resolve().parent
    study = load_study_config(
        root / "configs" / "mlp_selected.yaml",
        root / "configs" / "base.yaml",
    )
    study = replace(study, environment=replace(study.environment, candidate_count=10))
    study.validate()
    stem = timeline_stem(task.method, task.catalog_size, task.seed)
    reference_path = output_root / "raw" / f"n{task.catalog_size}" / f"{stem}.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"accepted raw episode is missing: {reference_path}")
    reference_frame = pd.read_csv(reference_path)
    if len(reference_frame.index) != 1:
        raise ValueError(f"expected one accepted row in {reference_path}")
    reference = reference_frame.iloc[0]

    timeline_dir = output_root / "timeline" / "raw" / f"n{task.catalog_size}"
    timeline_path = timeline_dir / f"{stem}.timeline.csv"
    metadata_path = timeline_dir / f"{stem}.timeline.metadata.json"
    status_path = (
        output_root / "timeline" / "status" / f"n{task.catalog_size}" / f"{stem}.json"
    )
    task_metadata = {
        **asdict(task),
        "existing_campaign_root": str(output_root),
        "existing_raw_episode": str(reference_path),
        "timeline_csv": str(timeline_path),
        "recording": "every decision epoch; analysis forward-fills to a 100-second grid",
        "git": git_metadata(Path.cwd()),
    }
    if args.dry_run:
        print(json.dumps(task_metadata, indent=2, sort_keys=True))
        return 0
    if timeline_path.is_file() and metadata_path.is_file() and status_path.is_file():
        try:
            prior_status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            prior_status = {}
        if prior_status.get("state") == "completed":
            print(
                f"SKIP timeline N={task.catalog_size} seed={task.seed} "
                f"method={task.method}",
                flush=True,
            )
            return 0

    learned_policy = None
    checkpoint_metadata = None
    observation_contract = None
    if task.method == "legacy_amos2025_alpha0_policy":
        learned_policy, raw_checkpoint_metadata = load_policy(args.checkpoint)
        checkpoint_metadata = validate_frozen_policy(
            args.checkpoint, raw_checkpoint_metadata
        )
        observation_contract = LEGACY_AMOS2025_OBSERVATION_CONTRACT

    status = {
        **task_metadata,
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_json(status_path, status)
    started_clock = time.monotonic()
    try:
        print(
            f"RUN timeline method={task.method} N={task.catalog_size} seed={task.seed}",
            flush=True,
        )
        rows: list[dict] = []
        run_kwargs = {}
        if observation_contract is not None:
            run_kwargs["observation_contract"] = observation_contract
        replay = run_episode(
            study,
            method=task.method,
            seed=task.seed,
            catalog_size=task.catalog_size,
            learned_policy=learned_policy,
            shield=True,
            trajectory_rows=rows,
            **run_kwargs,
        )
        verification = verify_replay(reference, replay)
        trajectory = pd.DataFrame(rows)
        if trajectory.empty or not np.isclose(trajectory.iloc[0]["sim_time_s"], 0.0):
            raise ValueError("timeline does not begin at time zero")
        if not np.isclose(trajectory.iloc[-1]["sim_time_s"], EPISODE_DURATION_S):
            raise ValueError("timeline does not reach 45,000 seconds")
        if not trajectory[
            "cumulative_illuminated_observations"
        ].is_monotonic_increasing:
            raise ValueError("cumulative illuminated count decreased")
        trajectory.insert(0, "decision_epoch_index", np.arange(len(trajectory)))
        atomic_write_frame(timeline_path, trajectory)
        atomic_write_json(
            metadata_path,
            {
                **task_metadata,
                "checkpoint": checkpoint_metadata,
                "trajectory_row_count": len(trajectory.index),
                "replay_verification": verification,
                "final_replay_metrics": replay,
            },
        )
        status.update(
            {
                "state": "completed",
                "finished_at_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "elapsed_seconds": time.monotonic() - started_clock,
                "trajectory_row_count": len(trajectory.index),
                "replay_verification": verification,
            }
        )
        atomic_write_json(status_path, status)
        print(
            f"PASS timeline method={task.method} N={task.catalog_size} "
            f"seed={task.seed} rows={len(trajectory.index)}",
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
        print(
            f"FAIL timeline method={task.method} N={task.catalog_size} "
            f"seed={task.seed}: {error}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
