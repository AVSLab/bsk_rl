#!/usr/bin/env python3
"""Run one frozen-checkpoint AMOS 2026 GAT Monte Carlo evaluation task.

Each Slurm array task evaluates exactly one policy and one seed in a fresh
interpreter. This avoids the Basilisk/CSPICE state accumulation and memory
pressure seen when many episodes are run sequentially in one Python process.
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


POLICY_TAGS = (
    "00d100i",
    "10d90i",
    "20d80i",
    "30d70i",
    "40d60i",
    "50d50i",
    "75d25i",
    "100d00i",
)
RUN_PREFIX_TEMPLATE = (
    "amos2026_LEO_GAT_fullActions_{tag}_4200batch_restrictedResources_"
    "obs-v9_hold10s_reimage2orb_prioritySum100"
)
DEFAULT_EVALUATION_REWARD_MIX = "100d00i"
DEFAULT_SEEDS_PER_BLOCK = 10


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def valid_numeric_checkpoints(model_dir: Path) -> list[Path]:
    checkpoints = []
    for path in model_dir.glob("checkpoint_[0-9]*"):
        module_dir = path / "learner_group" / "learner" / "rl_module" / "inspector"
        required_module_files = (
            module_dir / "module_state.pt",
            module_dir / "class_and_ctor_args.pkl",
            module_dir / "metadata.json",
        )
        if (
            path.is_dir()
            and module_dir.is_dir()
            and all(file.is_file() for file in required_module_files)
            and checkpoint_iteration(path) >= 0
        ):
            checkpoints.append(path)
    return sorted(checkpoints, key=checkpoint_iteration)


def model_dirs_for_tag(policy_root: Path, tag: str) -> list[Path]:
    prefix = RUN_PREFIX_TEMPLATE.format(tag=tag)
    model_dirs = []
    for run_dir in policy_root.glob(f"{prefix}_*"):
        if not run_dir.is_dir():
            continue
        model_dirs.extend(path for path in run_dir.glob("*.out_0") if path.is_dir())
    return model_dirs


def latest_checkpoint_for_tag(policy_root: Path, tag: str) -> dict[str, Any]:
    candidates = []
    for model_dir in model_dirs_for_tag(policy_root, tag):
        checkpoints = valid_numeric_checkpoints(model_dir)
        if not checkpoints:
            continue
        checkpoint = checkpoints[-1]
        run_dir = model_dir.parent
        candidates.append(
            (
                run_dir.stat().st_mtime,
                checkpoint.stat().st_mtime,
                checkpoint_iteration(checkpoint),
                model_dir,
                checkpoint,
            )
        )
    if not candidates:
        prefix = RUN_PREFIX_TEMPLATE.format(tag=tag)
        raise FileNotFoundError(
            f"No checkpoint-bearing model directory found for {tag!r} below "
            f"{policy_root}. Expected a run starting with {prefix!r}."
        )

    _, _, iteration, model_dir, checkpoint = max(candidates)
    return {
        "tag": tag,
        "run_dir": str(model_dir.parent.resolve()),
        "model_dir": str(model_dir.resolve()),
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_iteration": iteration,
    }


def build_manifest(policy_root: Path) -> dict[str, Any]:
    policies = {
        tag: latest_checkpoint_for_tag(policy_root, tag) for tag in POLICY_TAGS
    }
    return {
        "schema_version": 1,
        "created_at_utc": timestamp(),
        "policy_root": str(policy_root.resolve()),
        "evaluation_reward_mix": DEFAULT_EVALUATION_REWARD_MIX,
        "obs_v": 9,
        "policy_layout": "gat_full",
        "policy_tags": list(POLICY_TAGS),
        "policies": policies,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    missing = [tag for tag in POLICY_TAGS if tag not in manifest.get("policies", {})]
    if missing:
        raise ValueError(f"Manifest {path} is missing policies: {missing}")
    return manifest


def task_assignment(task_id: int, seed_start: int, seeds_per_block: int) -> tuple[str, int]:
    task_count = len(POLICY_TAGS) * seeds_per_block
    if not 0 <= task_id < task_count:
        raise ValueError(f"task_id must be in [0, {task_count - 1}], got {task_id}")
    policy_index, seed_offset = divmod(task_id, seeds_per_block)
    return POLICY_TAGS[policy_index], seed_start + seed_offset


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "unknown")
    default_policy_root = Path(f"/scratch/alpine/{user}/rllib_results")
    default_output_root = Path(
        f"/scratch/alpine/{user}/amos2026_mc/gat_full_actions_eval_100d00i"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="Array-task index. Maps to one policy and one seed.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=int(os.environ.get("BSK_RL_MC_SEED_START", "0")),
        help="First seed in this ten-seed campaign block.",
    )
    parser.add_argument(
        "--seeds-per-block",
        type=int,
        default=DEFAULT_SEEDS_PER_BLOCK,
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=Path(os.environ.get("BSK_RL_MC_POLICY_ROOT", default_policy_root)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("BSK_RL_MC_OUTPUT_ROOT", default_output_root)),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            Path(os.environ["BSK_RL_MC_MANIFEST"])
            if os.environ.get("BSK_RL_MC_MANIFEST")
            else None
        ),
        help="Frozen checkpoint manifest created before array submission.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Resolve all policies once, write a frozen manifest, then exit.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "updated_policy_evaluation.py",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--policy-mode",
        choices=["best", "smallest", "latest"],
        default="latest",
        help="Recorded for provenance. Exact checkpoint manifests make selection stable.",
    )
    parser.add_argument(
        "--target-env",
        choices=["leo", "mixed"],
        default="leo",
    )
    parser.add_argument(
        "--evaluation-reward-mix",
        default=DEFAULT_EVALUATION_REWARD_MIX,
        help="Common reward used to score every trained policy.",
    )
    parser.add_argument(
        "--dynamic-priority-event",
        choices=["on", "off"],
        default="on",
    )
    parser.add_argument(
        "--use-shield",
        action="store_true",
        help="Enable evaluation overrides for critical battery/storage. Off by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assignment and evaluator command without running Basilisk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seeds_per_block <= 0:
        raise ValueError("--seeds-per-block must be positive")

    if args.write_manifest:
        manifest = build_manifest(args.policy_root)
        atomic_write_json(args.write_manifest, manifest)
        print(f"Wrote frozen checkpoint manifest: {args.write_manifest}")
        for tag in POLICY_TAGS:
            policy = manifest["policies"][tag]
            print(
                f"  {tag}: checkpoint_{policy['checkpoint_iteration']:06d} "
                f"{policy['checkpoint_dir']}"
            )
        return 0

    if args.manifest is None:
        raise ValueError(
            "Pass --manifest or set BSK_RL_MC_MANIFEST. Create it once with "
            "--write-manifest before submitting the Slurm array."
        )

    manifest = load_manifest(args.manifest)
    policy_tag, seed = task_assignment(args.task_id, args.seed_start, args.seeds_per_block)
    policy = manifest["policies"][policy_tag]
    block_name = f"seeds_{args.seed_start:03d}_{args.seed_start + args.seeds_per_block - 1:03d}"
    seed_dir = args.output_root / block_name / policy_tag / f"seed_{seed:03d}"
    status_path = seed_dir / "mc_status.json"
    seed_dir.mkdir(parents=True, exist_ok=True)

    policy_name = f"amos2026_mc_GAT_fullActions_{policy_tag}_obs_v9"
    command = [
        args.python,
        "-u",
        str(args.eval_script),
        "--policy_name",
        policy_name,
        "--policy_path",
        policy["checkpoint_dir"],
        "--policy_layout",
        "gat_full",
        "--obs_v",
        "9",
        "--policy_mode",
        args.policy_mode,
        "--seed",
        str(seed),
        "--reward_mix_tag",
        args.evaluation_reward_mix,
        "--target_env",
        args.target_env,
        "--dynamic_priority_event",
        args.dynamic_priority_event,
        "--output_dir",
        str(seed_dir),
        "--save_data",
        "--quiet",
        "--skip_plots",
    ]
    if not args.use_shield:
        command.append("--no_shield")

    status = {
        "schema_version": 1,
        "state": "planned" if args.dry_run else "running",
        "created_at_utc": timestamp(),
        "task_id": args.task_id,
        "seed_start": args.seed_start,
        "seeds_per_block": args.seeds_per_block,
        "seed": seed,
        "policy_tag": policy_tag,
        "policy_name": policy_name,
        "policy": policy,
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(seed_dir.resolve()),
        "evaluation_reward_mix": args.evaluation_reward_mix,
        "target_env": args.target_env,
        "dynamic_priority_event": args.dynamic_priority_event,
        "use_shield": args.use_shield,
        "command": command,
    }
    atomic_write_json(status_path, status)

    print(
        f"MC task {args.task_id}: policy={policy_tag}, seed={seed}, "
        f"checkpoint={policy['checkpoint_iteration']}, score={args.evaluation_reward_mix}"
    )
    print("Evaluator command:")
    print(" ".join(command))
    if args.dry_run:
        return 0

    started_at = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=args.eval_script.parent, check=False)
    finished_at = datetime.now(timezone.utc)
    status.update(
        {
            "state": "completed" if completed.returncode == 0 else "failed",
            "started_at_utc": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at_utc": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
            "returncode": completed.returncode,
        }
    )
    atomic_write_json(status_path, status)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
