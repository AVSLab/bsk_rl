#!/usr/bin/env python3
"""Run one AMOS 2026 heuristic Monte Carlo evaluation."""

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


HEURISTIC_MODES = ("angle", "priority_angle")


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


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "unknown")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=int(os.environ.get("BSK_RL_HEUR_SEED_START", "0")),
    )
    parser.add_argument(
        "--seeds-per-block",
        type=int,
        default=int(os.environ.get("BSK_RL_HEUR_SEEDS_PER_BLOCK", "10")),
    )
    parser.add_argument(
        "--modes",
        default=os.environ.get("BSK_RL_HEUR_MODES", ",".join(HEURISTIC_MODES)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "BSK_RL_HEUR_OUTPUT_ROOT",
                f"/scratch/alpine/{user}/amos2026_mc/heuristics_mixed100",
            )
        ),
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "updated_policy_evaluation.py",
    )
    parser.add_argument(
        "--target-env",
        choices=["leo", "mixed"],
        default=os.environ.get("BSK_RL_HEUR_TARGET_ENV", "mixed"),
    )
    parser.add_argument(
        "--mix-weights",
        default=os.environ.get(
            "BSK_RL_HEUR_MIX_WEIGHTS", '{"LEO":0.5,"MEO":0.3,"GEO":0.2}'
        ),
    )
    parser.add_argument(
        "--exact-mix-counts",
        action="store_true",
        default=os.environ.get("BSK_RL_HEUR_EXACT_MIX_COUNTS", "1").lower()
        not in {"0", "false", "no", "off"},
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=int(os.environ.get("BSK_RL_HEUR_N_TARGETS", "100")),
    )
    parser.add_argument(
        "--n-targets-ahead",
        type=int,
        default=int(os.environ.get("BSK_RL_HEUR_N_TARGETS_AHEAD", "10")),
    )
    parser.add_argument(
        "--total-time-sec",
        type=float,
        default=float(os.environ.get("BSK_RL_HEUR_TOTAL_TIME_SEC", "45000")),
    )
    parser.add_argument(
        "--evaluation-reward-mix",
        default=os.environ.get("BSK_RL_HEUR_REWARD_MIX", "100d00i"),
    )
    parser.add_argument(
        "--dynamic-priority-event",
        choices=["on", "off"],
        default=os.environ.get("BSK_RL_HEUR_DYNAMIC_PRIORITY_EVENT", "on"),
    )
    parser.add_argument(
        "--hio-count", type=int, default=int(os.environ.get("BSK_RL_HEUR_HIO_COUNT", "5"))
    )
    parser.add_argument(
        "--hio-priority",
        type=float,
        default=float(os.environ.get("BSK_RL_HEUR_HIO_PRIORITY", "5.0")),
    )
    parser.add_argument(
        "--shio-count", type=int, default=int(os.environ.get("BSK_RL_HEUR_SHIO_COUNT", "3"))
    )
    parser.add_argument(
        "--shio-priority",
        type=float,
        default=float(os.environ.get("BSK_RL_HEUR_SHIO_PRIORITY", "10.0")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def metrics_files(seed_dir: Path) -> list[Path]:
    return sorted(seed_dir.glob("*/metrics_*.json")) + sorted(
        seed_dir.glob("metrics_*.json")
    )


def main() -> int:
    args = parse_args()
    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    unknown = sorted(set(modes) - set(HEURISTIC_MODES))
    if unknown:
        raise ValueError(f"Unknown heuristic modes: {unknown}")
    if args.seeds_per_block <= 0:
        raise ValueError("--seeds-per-block must be positive")

    task_count = len(modes) * args.seeds_per_block
    if not 0 <= args.task_id < task_count:
        raise ValueError(f"--task-id must be in [0, {task_count - 1}]")
    mode_index, seed_offset = divmod(args.task_id, args.seeds_per_block)
    mode = modes[mode_index]
    seed = args.seed_start + seed_offset
    mode_tag = f"heur_{mode}"
    block_name = (
        f"seeds_{args.seed_start:03d}_"
        f"{args.seed_start + args.seeds_per_block - 1:03d}"
    )
    seed_dir = args.output_root / block_name / mode_tag / f"seed_{seed:03d}"
    status_path = seed_dir / "mc_status.json"

    if status_path.is_file() and metrics_files(seed_dir):
        try:
            existing = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            existing = {}
        if existing.get("state") == "completed" and existing.get("returncode") == 0:
            print(f"Skipping completed mode={mode}, seed={seed}: {status_path}")
            return 0

    command = [
        sys.executable,
        "-u",
        str(args.eval_script),
        "--heuristic_mode",
        mode,
        "--policy_layout",
        "gat_full",
        "--obs_v",
        "9",
        "--seed",
        str(seed),
        "--reward_mix_tag",
        args.evaluation_reward_mix,
        "--target_env",
        args.target_env,
        "--mix_weights",
        args.mix_weights,
        "--n_targets",
        str(args.n_targets),
        "--n_targets_ahead",
        str(args.n_targets_ahead),
        "--total_time_sec",
        str(args.total_time_sec),
        "--dynamic_priority_event",
        args.dynamic_priority_event,
        "--hio_count",
        str(args.hio_count),
        "--hio_priority",
        str(args.hio_priority),
        "--shio_count",
        str(args.shio_count),
        "--shio_priority",
        str(args.shio_priority),
        "--output_dir",
        str(seed_dir),
        "--save_data",
        "--quiet",
        "--skip_plots",
        "--no_show_plots",
        "--plots_in_run_dir",
        "--no_shield",
    ]
    if args.exact_mix_counts:
        command.append("--exact_mix_counts")

    status = {
        "schema_version": 1,
        "state": "planned" if args.dry_run else "running",
        "created_at_utc": timestamp(),
        "task_id": args.task_id,
        "seed_start": args.seed_start,
        "seeds_per_block": args.seeds_per_block,
        "seed": seed,
        "policy_tag": mode_tag,
        "heuristic_mode": mode,
        "evaluation_reward_mix": args.evaluation_reward_mix,
        "target_env": args.target_env,
        "mix_weights": args.mix_weights,
        "exact_mix_counts": args.exact_mix_counts,
        "n_targets": args.n_targets,
        "n_targets_ahead": args.n_targets_ahead,
        "total_time_sec": args.total_time_sec,
        "dynamic_priority_event": args.dynamic_priority_event,
        "hio_count": args.hio_count,
        "hio_priority": args.hio_priority,
        "shio_count": args.shio_count,
        "shio_priority": args.shio_priority,
        "use_shield": False,
        "command": command,
    }
    atomic_write_json(status_path, status)
    print(f"Heuristic MC task {args.task_id}: mode={mode}, seed={seed}")
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
