#!/usr/bin/env python3
"""Replay a matched 300-second episode and record its acquisition history."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from examples.prospectus_rfi.acquisition_timeline import (
    EPISODE_DURATION_S,
    verify_replay,
)
from examples.prospectus_rfi.amos2025_matched_300s_design import CATALOG_SIZE
from examples.prospectus_rfi.amos2025_matched_300s_mc import (
    _execution_contract,
    _load_task_policy,
    task_spec,
)
from examples.prospectus_rfi.config import git_metadata, load_study_config
from examples.prospectus_rfi.evaluate import run_episode
from examples.prospectus_rfi.heuristic_mc import atomic_write_json


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--attention-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = task_spec(args.task_id)
    campaign_root = args.campaign_root.resolve()
    stem = f"{task.method}_seed{task.seed:03d}"
    reference_path = campaign_root / "raw" / task.method / f"{stem}.csv"
    timeline_path = (
        campaign_root / "timeline" / "raw" / task.method / f"{stem}.timeline.csv"
    )
    metadata_path = timeline_path.with_suffix(".metadata.json")
    status_path = (
        campaign_root
        / "timeline"
        / "status"
        / task.method
        / f"seed_{task.seed:03d}.json"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"accepted raw episode is missing: {reference_path}")
    reference_frame = pd.read_csv(reference_path)
    if len(reference_frame.index) != 1:
        raise ValueError(f"expected one accepted row in {reference_path}")
    reference = reference_frame.iloc[0]

    root = Path(__file__).resolve().parent
    study = load_study_config(
        root / "configs" / "attention_amos2025_control.yaml",
        root / "configs" / "base_amos2025_attention_control.yaml",
    )
    study.validate()
    execution_method, observation_contract = _execution_contract(task.method)
    task_metadata = {
        **asdict(task),
        "campaign_root": str(campaign_root),
        "accepted_raw_episode": str(reference_path),
        "timeline_csv": str(timeline_path),
        "recording": "every decision epoch",
        "observation_contract": observation_contract,
        "wheel_guard_enabled": False,
        "git": git_metadata(Path.cwd()),
    }
    if args.dry_run:
        print(json.dumps(task_metadata, indent=2, sort_keys=True))
        return 0
    if timeline_path.is_file() and metadata_path.is_file() and status_path.is_file():
        try:
            prior = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            prior = {}
        if prior.get("state") == "completed":
            print(f"SKIP timeline method={task.method} seed={task.seed}", flush=True)
            return 0

    policy, policy_metadata = _load_task_policy(
        task, args.legacy_checkpoint, args.attention_checkpoint
    )
    status = {
        **task_metadata,
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_json(status_path, status)
    started = time.monotonic()
    try:
        rows: list[dict] = []
        replay = run_episode(
            study,
            method=execution_method,
            seed=task.seed,
            catalog_size=CATALOG_SIZE,
            learned_policy=policy,
            shield=True,
            wheel_guard=False,
            observation_contract=observation_contract,
            trajectory_rows=rows,
        )
        replay["method"] = task.method
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
        trajectory["method"] = task.method
        trajectory.insert(0, "decision_epoch_index", np.arange(len(trajectory)))
        atomic_write_frame(timeline_path, trajectory)
        atomic_write_json(
            metadata_path,
            {
                **task_metadata,
                "policy": policy_metadata,
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
                "elapsed_seconds": time.monotonic() - started,
                "trajectory_row_count": len(trajectory.index),
                "replay_verification": verification,
            }
        )
        atomic_write_json(status_path, status)
        print(
            f"PASS timeline method={task.method} seed={task.seed} "
            f"rows={len(trajectory.index)}",
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
                "elapsed_seconds": time.monotonic() - started,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        print(
            f"FAIL timeline method={task.method} seed={task.seed}: {error}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
