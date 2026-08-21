#!/usr/bin/env python3
"""Inventory AMOS 2026 mixed-regime obs-v9 GAT training runs.

The script uses only the Python standard library so it can run on an Alpine
login node before the project virtual environment is activated. It records
configuration provenance, validates RLlib checkpoints, and can stage the latest
checkpoint from every fixed-100-target mixed run for transfer to a workstation.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any


TOP_LEVEL_SECTIONS = {
    "cluster",
    "observation_layout",
    "reward_split",
    "sim",
    "target_count",
    "target_regime",
    "wandb",
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return float(value) if any(char in value for char in ".eE") else int(value)
        except ValueError:
            return value


def parse_config_sections(path: Path) -> dict[str, dict[str, Any]]:
    """Read selected shallow mappings from the trainer's YAML without PyYAML."""
    sections: dict[str, dict[str, Any]] = {
        section: {} for section in TOP_LEVEL_SECTIONS
    }
    current_section: str | None = None
    nested_key: str | None = None

    for raw_line in path.read_text(errors="replace").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0:
            key = stripped[:-1] if stripped.endswith(":") else ""
            current_section = key if key in TOP_LEVEL_SECTIONS else None
            nested_key = None
            continue
        if current_section is None or ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        if indent == 2:
            nested_key = key if raw_value.strip() == "" else None
            sections[current_section][key] = (
                {} if nested_key else parse_scalar(raw_value)
            )
        elif indent == 4 and nested_key:
            nested = sections[current_section].setdefault(nested_key, {})
            if isinstance(nested, dict):
                nested[key] = parse_scalar(raw_value)
    return sections


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"checkpoint_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def has_valid_checkpoint_payload(path: Path) -> bool:
    module_dir = path / "learner_group" / "learner" / "rl_module" / "inspector"
    return (
        path.is_dir()
        and all(
            (module_dir / filename).is_file()
            for filename in (
                "module_state.pt",
                "class_and_ctor_args.pkl",
                "metadata.json",
            )
        )
    )


def is_valid_checkpoint(path: Path) -> bool:
    return checkpoint_iteration(path) >= 0 and has_valid_checkpoint_payload(path)


def best_checkpoint_iteration(path: Path) -> int:
    if not has_valid_checkpoint_payload(path):
        return -1
    markers = sorted(path.glob("iteration_*.txt"))
    if markers:
        match = re.fullmatch(r"iteration_(\d+)\.txt", markers[-1].name)
        if match:
            return int(match.group(1))
    return -1


def valid_checkpoints(model_dir: Path) -> list[Path]:
    return sorted(
        (
            checkpoint
            for checkpoint in model_dir.glob("checkpoint_[0-9]*")
            if is_valid_checkpoint(checkpoint)
        ),
        key=checkpoint_iteration,
    )


def last_progress_values(model_dir: Path) -> tuple[str, str]:
    progress_path = model_dir / "progress.csv"
    if not progress_path.is_file():
        return "", ""
    try:
        with progress_path.open(newline="", errors="replace") as handle:
            rows = csv.DictReader(handle)
            last_row = None
            for last_row in rows:
                pass
        if not last_row:
            return "", ""
        iteration = (
            last_row.get("training_iteration")
            or last_row.get("iterations_since_restore")
            or ""
        )
        reward = (
            last_row.get("env_runners/episode_return_mean")
            or last_row.get("episode_reward_mean")
            or last_row.get("sampler_results/episode_reward_mean")
            or ""
        )
        return str(iteration), str(reward)
    except (OSError, csv.Error):
        return "", ""


def model_dirs_for_run(run_dir: Path, config_paths: list[Path]) -> list[Path]:
    model_dirs = {
        path for path in run_dir.glob("*.out*") if path.is_dir()
    }
    for config_path in config_paths:
        model_name = config_path.name.removesuffix("_config.yaml")
        candidate = run_dir / model_name
        if candidate.is_dir():
            model_dirs.add(candidate)
    return sorted(model_dirs)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def copy_checkpoint_bundle(
    destination_root: Path,
    row: dict[str, Any],
    config_path: Path | None,
    model_dir: Path,
    checkpoint: Path,
    best_checkpoint: Path | None,
) -> Path:
    label = safe_name(
        f"{row.get('alpha_tag') or row.get('reward_tag') or 'mixed'}"
        f"__{model_dir.parent.name}__{checkpoint.name}"
    )
    destination = destination_root / label
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(checkpoint, destination / checkpoint.name, symlinks=True)
    if best_checkpoint is not None:
        shutil.copytree(
            best_checkpoint,
            destination / best_checkpoint.name,
            symlinks=True,
        )

    for source in (
        config_path,
        model_dir / "progress.csv",
        model_dir / "params.json",
        model_dir / "result.json",
    ):
        if source is not None and source.is_file():
            shutil.copy2(source, destination / source.name)

    provenance = {
        "source_run_dir": str(model_dir.parent.resolve()),
        "source_model_dir": str(model_dir.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
        "checkpoint_iteration": checkpoint_iteration(checkpoint),
        "source_best_checkpoint": (
            str(best_checkpoint.resolve()) if best_checkpoint else None
        ),
        "best_checkpoint_iteration": (
            best_checkpoint_iteration(best_checkpoint) if best_checkpoint else None
        ),
        "inventory_row": row,
    }
    (destination / "checkpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return destination


def inventory_run(run_dir: Path) -> list[tuple[dict[str, Any], Path | None, Path | None]]:
    config_paths = sorted(run_dir.glob("*_config.yaml"))
    model_dirs = model_dirs_for_run(run_dir, config_paths)
    config_by_model = {
        path.name.removesuffix("_config.yaml"): path for path in config_paths
    }
    model_names = sorted(
        set(config_by_model) | {path.name for path in model_dirs}
    )
    rows = []

    for model_name in model_names or [""]:
        model_dir = run_dir / model_name if model_name else None
        if model_dir is not None and not model_dir.is_dir():
            model_dir = None
        config_path = config_by_model.get(model_name)
        sections = parse_config_sections(config_path) if config_path else {}
        sim = sections.get("sim", {})
        observation = sections.get("observation_layout", {})
        reward = sections.get("reward_split", {})
        target_count = sections.get("target_count", {})
        target_regime = sections.get("target_regime", {})
        wandb = sections.get("wandb", {})

        checkpoints = valid_checkpoints(model_dir) if model_dir else []
        latest = checkpoints[-1] if checkpoints else None
        possible_best = model_dir / "checkpoint_best" if model_dir else None
        best = (
            possible_best
            if possible_best is not None
            and has_valid_checkpoint_payload(possible_best)
            else None
        )
        progress_iteration, progress_reward = (
            last_progress_values(model_dir) if model_dir else ("", "")
        )
        n_targets = (
            target_count.get("capacity_n_targets")
            or sim.get("n_targets")
            or ""
        )
        randomize_n_targets = bool(
            target_count.get("randomize_n_targets", False)
        )
        row = {
            "run_dir": str(run_dir.resolve()),
            "model_dir": str(model_dir.resolve()) if model_dir else "",
            "config_path": str(config_path.resolve()) if config_path else "",
            "target_env": target_regime.get("target_env", ""),
            "obs_version": "9" if "obs-v9" in run_dir.name.lower() else "",
            "n_targets": n_targets,
            "n_targets_ahead": observation.get(
                "n_targets_ahead", sim.get("n_targets_ahead", "")
            ),
            "randomize_n_targets": randomize_n_targets,
            "n_targets_min": target_count.get("n_targets_min", ""),
            "n_targets_max": target_count.get("n_targets_max", ""),
            "randomize_mix_weights": bool(
                target_regime.get("randomize_mix_weights", False)
            ),
            "fixed_mix_weights": json.dumps(
                target_regime.get("fixed_mix_weights"), sort_keys=True
            ),
            "exact_mix_counts": bool(
                target_regime.get("exact_mix_counts", False)
            ),
            "exact_regime_counts": json.dumps(
                target_regime.get("exact_regime_counts"), sort_keys=True
            ),
            "alpha": reward.get("downlink_bonus", ""),
            "alpha_tag": reward.get("alpha_tag", ""),
            "reward_tag": reward.get("tag", ""),
            "wandb_enabled": wandb.get("enabled", ""),
            "wandb_project": wandb.get("project", ""),
            "wandb_group": wandb.get("group", ""),
            "checkpoint_count": len(checkpoints),
            "latest_checkpoint": str(latest.resolve()) if latest else "",
            "latest_checkpoint_iteration": (
                checkpoint_iteration(latest) if latest else ""
            ),
            "best_checkpoint": str(best.resolve()) if best else "",
            "best_checkpoint_iteration": (
                best_checkpoint_iteration(best) if best else ""
            ),
            "progress_iteration": progress_iteration,
            "progress_reward_mean": progress_reward,
            "fixed_target_count_100": (
                str(target_regime.get("target_env", "")).lower() == "mixed"
                and str(n_targets) == "100"
                and not randomize_n_targets
            ),
        }
        rows.append((row, config_path, model_dir))
    return rows


def parse_args() -> argparse.Namespace:
    user = Path.home().name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=Path(f"/scratch/alpine/{user}/rllib_results"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-fixed100-checkpoints",
        action="store_true",
        help="Stage the latest valid checkpoint from every fixed-100 mixed obs-v9 run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_run_dirs = sorted(
        path
        for path in args.policy_root.iterdir()
        if path.is_dir()
        and "mixed" in path.name.lower()
        and "gat" in path.name.lower()
        and "obs-v9" in path.name.lower()
    )

    records: list[tuple[dict[str, Any], Path | None, Path | None]] = []
    for run_dir in candidate_run_dirs:
        records.extend(inventory_run(run_dir))
    rows = [record[0] for record in records]

    fieldnames = list(rows[0]) if rows else [
        "run_dir",
        "model_dir",
        "config_path",
        "target_env",
        "obs_version",
        "n_targets",
        "n_targets_ahead",
        "randomize_n_targets",
        "n_targets_min",
        "n_targets_max",
        "randomize_mix_weights",
        "fixed_mix_weights",
        "exact_mix_counts",
        "exact_regime_counts",
        "alpha",
        "alpha_tag",
        "reward_tag",
        "wandb_enabled",
        "wandb_project",
        "wandb_group",
        "checkpoint_count",
        "latest_checkpoint",
        "latest_checkpoint_iteration",
        "best_checkpoint",
        "best_checkpoint_iteration",
        "progress_iteration",
        "progress_reward_mean",
        "fixed_target_count_100",
    ]
    inventory_path = args.output_dir / "mixed_v9_training_inventory.csv"
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    copy_candidates = [
        row["latest_checkpoint"]
        for row in rows
        if row["fixed_target_count_100"] and row["latest_checkpoint"]
    ]
    (args.output_dir / "copy_candidates.txt").write_text(
        "".join(f"{path}\n" for path in copy_candidates)
    )

    copied = []
    if args.copy_fixed100_checkpoints:
        checkpoint_root = args.output_dir / "fixed100_checkpoints"
        checkpoint_root.mkdir(exist_ok=True)
        for row, config_path, model_dir in records:
            if (
                row["fixed_target_count_100"]
                and row["latest_checkpoint"]
                and model_dir is not None
            ):
                copied.append(
                    str(
                        copy_checkpoint_bundle(
                            checkpoint_root,
                            row,
                            config_path,
                            model_dir,
                            Path(row["latest_checkpoint"]),
                            (
                                Path(row["best_checkpoint"])
                                if row["best_checkpoint"]
                                else None
                            ),
                        ).resolve()
                    )
                )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "policy_root": str(args.policy_root.resolve()),
        "mixed_obs_v9_run_directory_count": len(candidate_run_dirs),
        "inventory_row_count": len(rows),
        "checkpoint_bearing_row_count": sum(
            bool(row["latest_checkpoint"]) for row in rows
        ),
        "fixed_target_count_100_row_count": sum(
            bool(row["fixed_target_count_100"]) for row in rows
        ),
        "fixed_target_count_100_checkpoint_count": len(copy_candidates),
        "copied_checkpoint_bundles": copied,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Inventory: {inventory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
