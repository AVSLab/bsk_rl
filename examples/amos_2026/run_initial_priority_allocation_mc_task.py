#!/usr/bin/env python3
"""Run one paired initial-priority AMOS 2026 Monte Carlo episode.

Even array tasks evaluate ground-confirmation-only re-imaging and odd tasks
evaluate a one-observer-orbit cooldown.  Adjacent task pairs use the same seed
from 0--49, the same exact 100/60/40 mixed catalog, and the same
target-promotion seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


POLICY_TAG = "mixed_a0p1"
N_TARGETS = 200
HIO_COUNT = 20
SHIO_COUNT = 20
NORMAL_TARGET_COUNT = N_TARGETS - HIO_COUNT - SHIO_COUNT
CASE_BY_INDEX = (
    ("ground_confirmation", 0.0),
    ("one_orbit", 1.0),
)
REQUIRED_OUTPUTS = (
    "steps.csv",
    "images.csv",
    "target_catalog.csv",
    "verified_deliveries.csv",
    "priority_response_targets.csv",
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "unknown")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--seeds-per-case",
        type=int,
        default=int(os.environ.get("AMOS_INITIAL_PRIORITY_SEEDS", "50")),
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "BSK_RL_REPO_DIR", f"/projects/{user}/bsk_rl"
            )
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "AMOS_INITIAL_PRIORITY_OUTPUT_ROOT",
                f"/scratch/alpine/{user}/amos2026_mc/initial_priority_allocation",
            )
        ),
    )
    parser.add_argument(
        "--policy-spec",
        type=Path,
        default=(
            Path(os.environ["AMOS_INITIAL_PRIORITY_POLICY_SPEC"])
            if os.environ.get("AMOS_INITIAL_PRIORITY_POLICY_SPEC")
            else None
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def task_assignment(task_id: int, seeds_per_case: int) -> tuple[str, float, int]:
    if seeds_per_case <= 0:
        raise ValueError("--seeds-per-case must be positive")
    task_count = len(CASE_BY_INDEX) * seeds_per_case
    if not 0 <= task_id < task_count:
        raise ValueError(f"task id must be in [0, {task_count - 1}]")
    seed, case_index = divmod(task_id, len(CASE_BY_INDEX))
    case_name, cooldown_orbits = CASE_BY_INDEX[case_index]
    return case_name, cooldown_orbits, seed


def resolve_checkpoint(policy_spec: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(policy_spec.expanduser().read_text())
    policy = payload.get("policies", {}).get(POLICY_TAG)
    if policy is None:
        raise KeyError(f"{POLICY_TAG!r} is missing from {policy_spec}")
    if float(policy.get("alpha", -1.0)) != 0.1:
        raise ValueError(
            f"{POLICY_TAG} must have alpha=0.1, got {policy.get('alpha')}"
        )
    checkpoint = Path(policy["checkpoint_dir"]).expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Policy checkpoint is missing: {checkpoint}")
    return checkpoint.resolve(), policy


def find_evaluation_dir(seed_dir: Path) -> Path | None:
    candidates = []
    for path in seed_dir.iterdir() if seed_dir.is_dir() else ():
        if not path.is_dir():
            continue
        if all((path / filename).is_file() for filename in REQUIRED_OUTPUTS):
            metrics = list(path.glob("metrics_*.json"))
            if metrics:
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()
    if args.policy_spec is None:
        raise ValueError(
            "Set AMOS_INITIAL_PRIORITY_POLICY_SPEC or pass --policy-spec"
        )

    repo_dir = args.repo_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    policy_spec = args.policy_spec.expanduser().resolve()
    case_name, cooldown_orbits, seed = task_assignment(
        args.task_id, args.seeds_per_case
    )
    checkpoint, policy = resolve_checkpoint(policy_spec)

    seed_dir = output_root / case_name / f"seed_{seed:03d}"
    status_path = seed_dir / "mc_status.json"
    existing_evaluation = find_evaluation_dir(seed_dir)
    if existing_evaluation is not None and status_path.is_file():
        try:
            prior_status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            prior_status = {}
        if (
            prior_status.get("state") == "completed"
            and prior_status.get("case") == case_name
            and float(prior_status.get("cooldown_orbits", -1.0))
            == cooldown_orbits
            and int(prior_status.get("seed", -1)) == seed
        ):
            print(f"Already complete: {existing_evaluation}")
            return 0

    seed_dir.mkdir(parents=True, exist_ok=True)
    evaluator = repo_dir / "examples" / "updated_policy_evaluation.py"
    policy_name = (
        "amos2026_INITIAL_PRIORITY_10pctHIO_10pctSHIO_"
        f"{case_name}_MIXED_GAT_alpha0p1"
    )
    command = [
        args.python,
        "-u",
        str(evaluator),
        "--policy_path",
        str(checkpoint),
        "--policy_name",
        policy_name,
        "--policy_layout",
        "gat_full",
        "--obs_v",
        "9",
        "--policy_mode",
        "latest",
        "--reward_alpha",
        "0.1",
        "--target_env",
        "mixed",
        "--mix_weights",
        '{"LEO":0.5,"MEO":0.3,"GEO":0.2}',
        "--exact_mix_counts",
        "--n_targets",
        str(N_TARGETS),
        "--n_targets_ahead",
        "10",
        "--priority_sum",
        str(N_TARGETS),
        "--priority_uniform_low",
        "0",
        "--priority_uniform_high",
        "2",
        "--total_time_sec",
        "45000",
        "--reimage_cooldown_orbits",
        str(cooldown_orbits),
        "--dynamic_priority_event",
        "on",
        "--dynamic_priority_event_time_sec",
        "0",
        "--hio_count",
        str(HIO_COUNT),
        "--hio_priority_max_multiplier",
        "5",
        "--shio_count",
        str(SHIO_COUNT),
        "--shio_priority_max_multiplier",
        "10",
        "--priority_control_count",
        str(NORMAL_TARGET_COUNT),
        "--priority_control_seed",
        str(20260729 + seed),
        "--dynamic_priority_event_seed",
        str(seed),
        "--seed",
        str(seed),
        "--save_data",
        "--output_dir",
        str(seed_dir),
        "--skip_plots",
        "--quiet",
    ]

    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "planned" if args.dry_run else "running",
        "created_at_utc": timestamp(),
        "task_id": args.task_id,
        "case": case_name,
        "seed": seed,
        "cooldown_orbits": cooldown_orbits,
        "promotion_timing": "initial_observation_t0",
        "policy_tag": POLICY_TAG,
        "policy": policy,
        "policy_spec": str(policy_spec),
        "checkpoint": str(checkpoint),
        "n_targets": N_TARGETS,
        "n_targets_ahead": 10,
        "target_catalog": "exact_100LEO_60MEO_40GEO",
        "total_time_sec": 45000,
        "hio_count": HIO_COUNT,
        "hio_fraction": HIO_COUNT / N_TARGETS,
        "hio_priority": "5x_realized_initial_maximum",
        "shio_count": SHIO_COUNT,
        "shio_fraction": SHIO_COUNT / N_TARGETS,
        "shio_priority": "10x_realized_initial_maximum",
        "normal_target_count": NORMAL_TARGET_COUNT,
        "normal_allocation_groups": "within-seed_initial-priority_tertiles",
        "vizard": False,
        "per_seed_plots": False,
        "command": command,
    }
    atomic_write_json(status_path, status)
    print(
        f"Task {args.task_id}: case={case_name}, seed={seed}, "
        f"cooldown={cooldown_orbits:g} orbit(s)"
    )
    print("Evaluator command:")
    print(" ".join(command))
    if args.dry_run:
        return 0

    before = {path.resolve() for path in seed_dir.iterdir() if path.is_dir()}
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=evaluator.parent,
        check=False,
    )
    finished = datetime.now(timezone.utc)
    after = {path.resolve() for path in seed_dir.iterdir() if path.is_dir()}
    new_directories = sorted(after - before)
    evaluation_dir = (
        new_directories[-1] if len(new_directories) == 1 else find_evaluation_dir(seed_dir)
    )

    missing_outputs: list[str] = []
    if evaluation_dir is None:
        missing_outputs = list(REQUIRED_OUTPUTS) + ["metrics_*.json"]
    else:
        missing_outputs = [
            name for name in REQUIRED_OUTPUTS if not (evaluation_dir / name).is_file()
        ]
        if not list(evaluation_dir.glob("metrics_*.json")):
            missing_outputs.append("metrics_*.json")

    success = completed.returncode == 0 and not missing_outputs
    status.update(
        {
            "state": "completed" if success else "failed",
            "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": (finished - started).total_seconds(),
            "returncode": completed.returncode,
            "evaluation_dir": str(evaluation_dir) if evaluation_dir else None,
            "missing_outputs": missing_outputs,
        }
    )
    atomic_write_json(status_path, status)
    if not success:
        print(f"Task failed validation; missing outputs: {missing_outputs}")
        return completed.returncode or 3
    print(f"Validated outputs: {evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
