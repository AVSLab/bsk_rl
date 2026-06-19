#!/usr/bin/env python3
"""Run one isolated policy/environment/seed cell from the frozen MC manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def completed_status(status_path: Path, manifest_path: Path, cell: str, seed: int) -> bool:
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    metrics_path = Path(status.get("metrics_path", ""))
    return (
        status.get("state") == "completed"
        and status.get("manifest") == str(manifest_path)
        and status.get("cell") == cell
        and status.get("seed") == seed
        and metrics_path.is_file()
        and metrics_path.stat().st_size > 0
    )


def print_log_tail(log_path: Path, line_count: int = 120) -> None:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError as error:
        print(f"Could not read evaluator log {log_path}: {error}", file=sys.stderr)
        return
    print(f"\n===== Last {line_count} lines of {log_path} =====", file=sys.stderr)
    for line in lines[-line_count:]:
        print(line, file=sys.stderr)
    print("===== End evaluator log =====\n", file=sys.stderr)


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    source_commit = git_commit(repo_root)

    if args.cell not in manifest["cells"]:
        raise ValueError(
            f"Unknown cell {args.cell!r}; choose from {sorted(manifest['cells'])}"
        )
    if args.seed not in manifest["seeds"]:
        raise ValueError(f"Seed {args.seed} is not included in the frozen manifest")

    cell_config = manifest["cells"][args.cell]
    policy = manifest["policies"][cell_config["policy"]]
    evaluation = manifest["evaluation"]
    seed_dir = output_root / args.cell / f"seed_{args.seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    status_path = seed_dir / "mc_status.json"
    if not args.force and completed_status(
        status_path, manifest_path, args.cell, args.seed
    ):
        print(f"Already completed: {args.cell} seed {args.seed}")
        return

    target_env = cell_config["evaluation_environment"]
    command = [
        sys.executable,
        "-u",
        str(repo_root / "examples" / "policy_evaluation_2026.py"),
        "--seed",
        str(args.seed),
        "--target_env",
        target_env,
        "--mix_weights",
        json.dumps(evaluation["mixed_weights"], separators=(",", ":")),
        "--policy_name",
        policy["name"],
        "--policy_path",
        policy["checkpoint"],
        "--policy_mode",
        "latest",
        "--obs_v",
        str(evaluation["obs_v"]),
        "--output_dir",
        str(seed_dir),
        "--total_time_sec",
        str(evaluation["total_time_sec"]),
        "--no_save_data",
        "--quiet",
        "--skip_plots",
    ]
    command_path = seed_dir / "command.json"
    atomic_write_json(
        command_path,
        {
            "command": command,
            "manifest": str(manifest_path),
            "cell": args.cell,
            "seed": args.seed,
            "git_commit": source_commit,
            "started_at_utc": timestamp(),
        },
    )
    atomic_write_json(
        status_path,
        {
            "state": "running",
            "manifest": str(manifest_path),
            "cell": args.cell,
            "seed": args.seed,
            "git_commit": source_commit,
            "policy": policy,
            "evaluation_environment": target_env,
            "started_at_utc": timestamp(),
        },
    )

    existing_metrics = set(seed_dir.rglob("metrics_*.json"))
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    log_path = seed_dir / "evaluator.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    metrics_files = sorted(
        path
        for path in seed_dir.rglob("metrics_*.json")
        if path not in existing_metrics and path.is_file() and path.stat().st_size > 0
    )
    state = "completed" if result.returncode == 0 and len(metrics_files) == 1 else "failed"
    status = {
        "state": state,
        "manifest": str(manifest_path),
        "cell": args.cell,
        "seed": args.seed,
        "git_commit": source_commit,
        "policy": policy,
        "evaluation_environment": target_env,
        "returncode": result.returncode,
        "metrics_path": str(metrics_files[0]) if len(metrics_files) == 1 else None,
        "metrics_file_count": len(metrics_files),
        "log_path": str(log_path),
        "finished_at_utc": timestamp(),
    }
    atomic_write_json(status_path, status)
    if state != "completed":
        print_log_tail(log_path)
        raise SystemExit(
            f"{args.cell} seed {args.seed} failed; inspect {log_path}"
        )
    print(f"Completed {args.cell} seed {args.seed}: {metrics_files[0]}")


if __name__ == "__main__":
    main()
