#!/usr/bin/env python3
"""Run one frozen-checkpoint AMOS 2026 GAT Monte Carlo evaluation task.

Each invocation evaluates exactly one policy and one seed in a fresh
interpreter. The Slurm wrapper may invoke this script repeatedly inside one
policy-level allocation, but every Basilisk episode still runs in its own
subprocess. This avoids the CSPICE state accumulation and memory pressure seen
when many episodes are run sequentially in one Python process.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ALL_POLICY_TAGS = (
    "00d100i",
    "10d90i",
    "20d80i",
    "30d70i",
    "40d60i",
    "50d50i",
    "60d40i",
    "70d30i",
    "75d25i",
    "80d20i",
    "90d10i",
    "100d00i",
)
DEFAULT_POLICY_TAGS = ALL_POLICY_TAGS
RUN_PREFIX_TEMPLATE = (
    "amos2026_LEO_GAT_fullActions_{tag}_4200batch_restrictedResources_"
    "obs-v9_hold10s_reimage2orb_prioritySum100"
)
DEFAULT_EVALUATION_REWARD_MIX = "100d00i"
DEFAULT_SEEDS_PER_BLOCK = 10


def standard_policy_alpha(tag: str) -> float:
    match = re.fullmatch(r"(\d+)d(\d+)i", tag)
    if not match:
        raise ValueError(f"Cannot parse standard reward-mix tag as alpha: {tag}")
    downlink_weight = float(match.group(1))
    imaging_weight = float(match.group(2))
    total = downlink_weight + imaging_weight
    if total <= 0:
        raise ValueError(f"Invalid reward-mix tag: {tag}")
    return downlink_weight / total


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


def is_valid_numeric_checkpoint(path: Path) -> bool:
    module_dir = path / "learner_group" / "learner" / "rl_module" / "inspector"
    required_module_files = (
        module_dir / "module_state.pt",
        module_dir / "class_and_ctor_args.pkl",
        module_dir / "metadata.json",
    )
    return (
        path.is_dir()
        and module_dir.is_dir()
        and all(file.is_file() for file in required_module_files)
        and checkpoint_iteration(path) >= 0
    )


def valid_numeric_checkpoints(model_dir: Path) -> list[Path]:
    checkpoints = []
    for path in model_dir.glob("checkpoint_[0-9]*"):
        if is_valid_numeric_checkpoint(path):
            checkpoints.append(path)
    return sorted(checkpoints, key=checkpoint_iteration)


def model_dirs_for_tag(policy_root: Path, tag: str) -> list[Path]:
    prefix = RUN_PREFIX_TEMPLATE.format(tag=tag)
    model_dirs = []
    for run_dir in policy_root.glob(f"{prefix}_*"):
        if not run_dir.is_dir():
            continue
        # The later 48-hour sweep uses a timestamp directly after the prefix.
        # Exclude earlier 24-hour pilot folders such as *_alpha0p2_<timestamp>.
        run_suffix = run_dir.name[len(prefix) :]
        if re.fullmatch(r"_\d+(?:\.\d+)?", run_suffix) is None:
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
            f"{policy_root}. Expected a non-alpha 48-hour run named like "
            f"{prefix!r}_<timestamp>."
        )

    _, _, iteration, model_dir, checkpoint = max(candidates)
    return {
        "tag": tag,
        "run_dir": str(model_dir.parent.resolve()),
        "model_dir": str(model_dir.resolve()),
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_iteration": iteration,
        "alpha": standard_policy_alpha(tag),
        "label": tag,
        "custom": False,
    }


def read_json_or_path(raw_value: str | None) -> Any:
    if raw_value is None or raw_value.strip() == "":
        return {}
    text = raw_value.strip()
    if text.startswith("@"):
        return json.loads(Path(text[1:]).expanduser().read_text())
    if text[0] not in "[{":
        possible_path = Path(text).expanduser()
        if possible_path.exists():
            return json.loads(possible_path.read_text())
    return json.loads(text)


def parse_custom_policies(raw_value: str | None) -> dict[str, dict[str, Any]]:
    payload = read_json_or_path(raw_value)
    if not payload:
        return {}
    if isinstance(payload, dict) and "policies" in payload:
        payload = payload["policies"]
    if isinstance(payload, list):
        payload = {str(item["tag"]): item for item in payload}
    if not isinstance(payload, dict):
        raise ValueError("--custom-policies-json must be a JSON object or list")

    custom_policies: dict[str, dict[str, Any]] = {}
    for tag, spec in payload.items():
        tag = str(tag)
        if tag in ALL_POLICY_TAGS:
            raise ValueError(
                f"Custom policy tag {tag!r} collides with a standard alpha tag."
            )
        if isinstance(spec, str):
            spec = {"checkpoint_dir": spec}
        if not isinstance(spec, dict):
            raise ValueError(f"Custom policy {tag!r} must be an object or path string")
        custom_policies[tag] = dict(spec)
    return custom_policies


def latest_checkpoint_from_path(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if is_valid_numeric_checkpoint(path):
        return path.parent, path

    if not path.is_dir():
        raise FileNotFoundError(f"Custom policy path does not exist: {path}")

    checkpoints = valid_numeric_checkpoints(path)
    if checkpoints:
        return path, checkpoints[-1]

    candidates = []
    for model_dir in path.glob("*.out*"):
        if not model_dir.is_dir():
            continue
        checkpoints = valid_numeric_checkpoints(model_dir)
        if checkpoints:
            checkpoint = checkpoints[-1]
            candidates.append(
                (
                    checkpoint.stat().st_mtime,
                    checkpoint_iteration(checkpoint),
                    model_dir,
                    checkpoint,
                )
            )
    if candidates:
        _, _, model_dir, checkpoint = max(candidates)
        return model_dir, checkpoint

    raise FileNotFoundError(
        f"No valid RLlib checkpoint found at or one level below {path}"
    )


def custom_policy_from_spec(tag: str, spec: dict[str, Any]) -> dict[str, Any]:
    raw_path = spec.get("checkpoint_dir") or spec.get("model_dir") or spec.get("run_dir")
    if raw_path is None:
        raise ValueError(
            f"Custom policy {tag!r} needs checkpoint_dir, model_dir, or run_dir."
        )
    model_dir, checkpoint = latest_checkpoint_from_path(Path(str(raw_path)))
    run_dir = model_dir.parent
    policy: dict[str, Any] = {
        "tag": tag,
        "run_dir": str(run_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_iteration": checkpoint_iteration(checkpoint),
        "custom": True,
    }
    for field in ("alpha", "color", "label", "policy_name"):
        if field in spec and spec[field] is not None:
            policy[field] = spec[field]
    if "label" not in policy:
        policy["label"] = tag
    return policy


def parse_policy_tags(
    raw_value: str | None,
    custom_policy_tags: tuple[str, ...] = (),
    *,
    allow_custom: bool = False,
) -> tuple[str, ...]:
    if raw_value is None or raw_value.strip() == "":
        return DEFAULT_POLICY_TAGS
    tags = tuple(
        tag.strip()
        for tag in re.split(r"[,;:]", raw_value)
        if tag.strip()
    )
    if not tags:
        raise ValueError("Policy tag list cannot be empty.")
    allowed_custom = set(custom_policy_tags)
    unknown = [
        tag
        for tag in tags
        if tag not in ALL_POLICY_TAGS and tag not in allowed_custom
    ]
    if unknown:
        if not allow_custom:
            raise ValueError(
                f"Unknown policy tags: {unknown}. Known standard tags: "
                f"{ALL_POLICY_TAGS}. Pass --custom-policies-json for non-sweep "
                "policies such as curriculum checkpoints."
            )
    duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
    if duplicates:
        raise ValueError(f"Duplicate policy tags are not allowed: {duplicates}")
    return tags


def build_manifest(
    policy_root: Path,
    policy_tags: tuple[str, ...],
    custom_policies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policies = {}
    for tag in policy_tags:
        if tag in custom_policies:
            policies[tag] = custom_policy_from_spec(tag, custom_policies[tag])
        elif tag in ALL_POLICY_TAGS:
            policies[tag] = latest_checkpoint_for_tag(policy_root, tag)
        else:
            raise ValueError(
                f"Policy tag {tag!r} is not a standard tag and has no custom spec."
            )
    return {
        "schema_version": 1,
        "created_at_utc": timestamp(),
        "policy_root": str(policy_root.resolve()),
        "evaluation_reward_mix": DEFAULT_EVALUATION_REWARD_MIX,
        "training_run_selection": "non_alpha_48h",
        "obs_v": 9,
        "policy_layout": "gat_full",
        "policy_tags": list(policy_tags),
        "custom_policy_tags": sorted(custom_policies),
        "policies": policies,
    }


def load_manifest(path: Path, policy_tags: tuple[str, ...]) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    manifest_tags = tuple(manifest.get("policy_tags", []))
    if manifest_tags and manifest_tags != policy_tags:
        raise ValueError(
            f"Manifest {path} has policy tags {manifest_tags}, but this run requested "
            f"{policy_tags}. Use a fresh manifest/output root for a different subset."
        )
    missing = [tag for tag in policy_tags if tag not in manifest.get("policies", {})]
    if missing:
        raise ValueError(f"Manifest {path} is missing policies: {missing}")
    return manifest


def metrics_files_for_seed(seed_dir: Path) -> list[Path]:
    """Find metrics files one level below a seed dir without walking plots/data."""
    return sorted(seed_dir.glob("metrics_*.json")) + sorted(
        seed_dir.glob("*/metrics_*.json")
    )


def completed_status_for(
    output_root: Path,
    policy_tag: str,
    seed: int,
    evaluation_reward_mix: str,
    target_env: str,
    mix_weights: str,
    exact_mix_counts: bool,
    dynamic_priority_event: str,
    hio_count: int,
    hio_priority: float,
    hio_priority_max_multiplier: float | None,
    shio_count: int,
    shio_priority: float,
    shio_priority_max_multiplier: float | None,
    priority_control_count: int,
    use_shield: bool,
    priority_sum: float,
    priority_uniform_low: float,
    priority_uniform_high: float | None,
    n_targets: int,
    n_targets_ahead: int,
    total_time_sec: float | None,
) -> Path | None:
    """Return an existing completed policy/seed status, if one is safe to reuse."""
    if not output_root.exists():
        return None

    status_paths = sorted(
        output_root.glob(f"seeds_*/{policy_tag}/seed_{seed:03d}/mc_status.json")
    )
    for status_path in status_paths:
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("state") != "completed":
            continue
        if status.get("returncode", 0) not in (0, None):
            continue
        if status.get("policy_tag") != policy_tag or int(status.get("seed", -1)) != seed:
            continue
        if status.get("evaluation_reward_mix") != evaluation_reward_mix:
            continue
        if status.get("target_env") != target_env:
            continue
        if target_env == "mixed" and status.get("mix_weights") != mix_weights:
            continue
        if bool(status.get("exact_mix_counts", False)) != bool(exact_mix_counts):
            continue
        if status.get("dynamic_priority_event") != dynamic_priority_event:
            continue
        if int(status.get("hio_count", hio_count)) != int(hio_count):
            continue
        if float(status.get("hio_priority", hio_priority)) != float(hio_priority):
            continue
        if status.get("hio_priority_max_multiplier") != hio_priority_max_multiplier:
            continue
        if int(status.get("shio_count", shio_count)) != int(shio_count):
            continue
        if float(status.get("shio_priority", shio_priority)) != float(shio_priority):
            continue
        if status.get("shio_priority_max_multiplier") != shio_priority_max_multiplier:
            continue
        if int(status.get("priority_control_count", 0)) != int(
            priority_control_count
        ):
            continue
        if priority_control_count and int(
            status.get("priority_control_seed", -1)
        ) != 20260729 + int(seed):
            continue
        if bool(status.get("use_shield", False)) != bool(use_shield):
            continue
        if abs(float(status.get("priority_sum", 100.0)) - float(priority_sum)) > 1e-9:
            continue
        if abs(
            float(status.get("priority_uniform_low", priority_uniform_low))
            - float(priority_uniform_low)
        ) > 1e-9:
            continue
        if status.get("priority_uniform_high") != priority_uniform_high:
            continue
        if int(status.get("n_targets", 100)) != int(n_targets):
            continue
        if int(status.get("n_targets_ahead", 10)) != int(n_targets_ahead):
            continue
        stored_total_time = status.get("total_time_sec")
        if total_time_sec is not None:
            try:
                if abs(float(stored_total_time) - float(total_time_sec)) > 1e-6:
                    continue
            except (TypeError, ValueError):
                continue
        if not metrics_files_for_seed(status_path.parent):
            continue
        return status_path
    return None


def task_assignment(
    task_id: int,
    seed_start: int,
    seeds_per_block: int,
    policy_tags: tuple[str, ...],
) -> tuple[str, int]:
    task_count = len(policy_tags) * seeds_per_block
    if not 0 <= task_id < task_count:
        raise ValueError(f"task_id must be in [0, {task_count - 1}], got {task_id}")
    policy_index, seed_offset = divmod(task_id, seeds_per_block)
    return policy_tags[policy_index], seed_start + seed_offset


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
        "--policy-tags",
        default=os.environ.get("BSK_RL_MC_POLICY_TAGS", ",".join(DEFAULT_POLICY_TAGS)),
        help=(
            "Comma-separated trained policy tags to evaluate. Defaults to the full "
            "12-policy alpha set."
        ),
    )
    parser.add_argument(
        "--custom-policies-json",
        default=os.environ.get("BSK_RL_MC_CUSTOM_POLICIES_JSON", ""),
        help=(
            "JSON string, JSON path, or @JSON_PATH mapping custom policy tags to "
            "checkpoint/model/run paths plus optional alpha/color/label metadata."
        ),
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
        default=os.environ.get("BSK_RL_MC_TARGET_ENV", "leo"),
    )
    parser.add_argument(
        "--mix-weights",
        default=os.environ.get("BSK_RL_MC_MIX_WEIGHTS", '{"LEO":0.5,"MEO":0.3,"GEO":0.2}'),
        help='JSON regime weights used when --target-env mixed, e.g. \'{"LEO":0.5,"MEO":0.3,"GEO":0.2}\'.',
    )
    parser.add_argument(
        "--exact-mix-counts",
        action="store_true",
        default=os.environ.get("BSK_RL_MC_EXACT_MIX_COUNTS", "0").strip().lower()
        not in {"0", "false", "no", "off"},
        help="Use exact per-catalog regime counts instead of independent draws.",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=int(
            os.environ.get(
                "BSK_RL_MC_N_TARGETS", os.environ.get("BSK_RL_N_TARGETS", "100")
            )
        ),
        help="Number of RSO targets for the evaluation episode.",
    )
    parser.add_argument(
        "--priority-sum",
        type=float,
        default=float(os.environ.get("BSK_RL_MC_PRIORITY_SUM", "100.0")),
        help=(
            "Total baseline catalog priority. Set equal to --n-targets to keep "
            "mean initial target priority equal to one."
        ),
    )
    parser.add_argument(
        "--priority-uniform-low",
        type=float,
        default=float(os.environ.get("BSK_RL_MC_PRIORITY_UNIFORM_LOW", "0.0")),
        help="Raw lower bound for uniform baseline priorities.",
    )
    parser.add_argument(
        "--priority-uniform-high",
        type=float,
        default=(
            float(os.environ["BSK_RL_MC_PRIORITY_UNIFORM_HIGH"])
            if os.environ.get("BSK_RL_MC_PRIORITY_UNIFORM_HIGH")
            else None
        ),
        help="Raw upper bound for uniform baseline priorities.",
    )
    parser.add_argument(
        "--n-targets-ahead",
        type=int,
        default=int(
            os.environ.get(
                "BSK_RL_MC_N_TARGETS_AHEAD",
                os.environ.get("BSK_RL_N_TARGETS_AHEAD", "10"),
            )
        ),
        help="Number of candidate targets exposed to the GAT policy.",
    )
    parser.add_argument(
        "--extra-time-factor",
        type=float,
        default=float(
            os.environ.get(
                "BSK_RL_MC_EXTRA_TIME_FACTOR",
                os.environ.get("BSK_RL_EXTRA_TIME_FACTOR", "1.5"),
            )
        ),
        help="Episode length multiplier used by updated_policy_evaluation.py.",
    )
    parser.add_argument(
        "--total-time-sec",
        type=float,
        default=(
            float(os.environ["BSK_RL_MC_TOTAL_TIME_SEC"])
            if os.environ.get("BSK_RL_MC_TOTAL_TIME_SEC")
            else (
                float(os.environ["BSK_RL_TOTAL_TIME_SEC"])
                if os.environ.get("BSK_RL_TOTAL_TIME_SEC")
                else None
            )
        ),
        help="Absolute episode length in seconds. Overrides --extra-time-factor.",
    )
    parser.add_argument(
        "--evaluation-reward-mix",
        default=DEFAULT_EVALUATION_REWARD_MIX,
        help="Common reward used to score every trained policy.",
    )
    parser.add_argument(
        "--dynamic-priority-event",
        choices=["on", "off"],
        default=os.environ.get("BSK_RL_MC_DYNAMIC_PRIORITY_EVENT", "on"),
    )
    parser.add_argument(
        "--hio-count",
        type=int,
        default=int(os.environ.get("BSK_RL_MC_HIO_COUNT", "5")),
    )
    parser.add_argument(
        "--hio-priority",
        type=float,
        default=float(os.environ.get("BSK_RL_MC_HIO_PRIORITY", "5.0")),
    )
    parser.add_argument(
        "--hio-priority-max-multiplier",
        type=float,
        default=(
            float(os.environ["BSK_RL_MC_HIO_PRIORITY_MAX_MULTIPLIER"])
            if os.environ.get("BSK_RL_MC_HIO_PRIORITY_MAX_MULTIPLIER")
            else None
        ),
        help="Scale HIO priority by the realized maximum initial catalog priority.",
    )
    parser.add_argument(
        "--shio-count",
        type=int,
        default=int(os.environ.get("BSK_RL_MC_SHIO_COUNT", "3")),
    )
    parser.add_argument(
        "--shio-priority",
        type=float,
        default=float(os.environ.get("BSK_RL_MC_SHIO_PRIORITY", "10.0")),
    )
    parser.add_argument(
        "--shio-priority-max-multiplier",
        type=float,
        default=(
            float(os.environ["BSK_RL_MC_SHIO_PRIORITY_MAX_MULTIPLIER"])
            if os.environ.get("BSK_RL_MC_SHIO_PRIORITY_MAX_MULTIPLIER")
            else None
        ),
        help="Scale SHIO priority by the realized maximum initial catalog priority.",
    )
    parser.add_argument(
        "--priority-control-count",
        type=int,
        default=int(os.environ.get("BSK_RL_MC_PRIORITY_CONTROL_COUNT", "0")),
        help="Unboosted targets tracked from the priority-event time.",
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
    custom_policies = parse_custom_policies(args.custom_policies_json)
    policy_tags = parse_policy_tags(
        args.policy_tags,
        tuple(custom_policies),
        allow_custom=args.manifest is not None,
    )
    if args.seeds_per_block <= 0:
        raise ValueError("--seeds-per-block must be positive")
    if args.priority_sum <= 0.0:
        raise ValueError("--priority-sum must be positive")
    if (
        args.priority_uniform_high is not None
        and args.priority_uniform_high < args.priority_uniform_low
    ):
        raise ValueError("--priority-uniform-high must be >= --priority-uniform-low")
    for name, multiplier in (
        ("--hio-priority-max-multiplier", args.hio_priority_max_multiplier),
        ("--shio-priority-max-multiplier", args.shio_priority_max_multiplier),
    ):
        if multiplier is not None and multiplier <= 0.0:
            raise ValueError(f"{name} must be positive")

    if args.write_manifest:
        manifest = build_manifest(args.policy_root, policy_tags, custom_policies)
        atomic_write_json(args.write_manifest, manifest)
        print(f"Wrote frozen checkpoint manifest: {args.write_manifest}")
        for tag in policy_tags:
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

    manifest = load_manifest(args.manifest, policy_tags)
    policy_tag, seed = task_assignment(
        args.task_id,
        args.seed_start,
        args.seeds_per_block,
        policy_tags,
    )
    policy = manifest["policies"][policy_tag]
    existing_status = completed_status_for(
        args.output_root,
        policy_tag,
        seed,
        args.evaluation_reward_mix,
        args.target_env,
        args.mix_weights,
        args.exact_mix_counts,
        args.dynamic_priority_event,
        args.hio_count,
        args.hio_priority,
        args.hio_priority_max_multiplier,
        args.shio_count,
        args.shio_priority,
        args.shio_priority_max_multiplier,
        args.priority_control_count,
        args.use_shield,
        args.priority_sum,
        args.priority_uniform_low,
        args.priority_uniform_high,
        args.n_targets,
        args.n_targets_ahead,
        args.total_time_sec,
    )
    if existing_status is not None:
        print(
            f"Skipping policy={policy_tag}, seed={seed}: completed run already exists at "
            f"{existing_status}"
        )
        return 0

    block_name = f"seeds_{args.seed_start:03d}_{args.seed_start + args.seeds_per_block - 1:03d}"
    seed_dir = args.output_root / block_name / policy_tag / f"seed_{seed:03d}"
    status_path = seed_dir / "mc_status.json"
    seed_dir.mkdir(parents=True, exist_ok=True)

    policy_name = policy.get(
        "policy_name", f"amos2026_mc_GAT_fullActions_{policy_tag}_obs_v9"
    )
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
        "--mix_weights",
        args.mix_weights,
        "--n_targets",
        str(args.n_targets),
        "--priority_sum",
        str(args.priority_sum),
        "--priority_uniform_low",
        str(args.priority_uniform_low),
        "--n_targets_ahead",
        str(args.n_targets_ahead),
        "--extra_time_factor",
        str(args.extra_time_factor),
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
        "--priority_control_count",
        str(args.priority_control_count),
        "--priority_control_seed",
        str(20260729 + seed),
        "--output_dir",
        str(seed_dir),
        "--save_data",
        "--quiet",
        "--no_show_plots",
        "--plots_in_run_dir",
    ]
    if args.hio_priority_max_multiplier is not None:
        command.extend(
            [
                "--hio_priority_max_multiplier",
                str(args.hio_priority_max_multiplier),
            ]
        )
    if args.priority_uniform_high is not None:
        command.extend(
            ["--priority_uniform_high", str(args.priority_uniform_high)]
        )
    if args.shio_priority_max_multiplier is not None:
        command.extend(
            [
                "--shio_priority_max_multiplier",
                str(args.shio_priority_max_multiplier),
            ]
        )
    if args.exact_mix_counts:
        command.append("--exact_mix_counts")
    if args.total_time_sec is not None:
        command.extend(["--total_time_sec", str(args.total_time_sec)])
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
        "mix_weights": args.mix_weights,
        "exact_mix_counts": args.exact_mix_counts,
        "priority_sum": args.priority_sum,
        "priority_uniform_low": args.priority_uniform_low,
        "priority_uniform_high": args.priority_uniform_high,
        "n_targets": args.n_targets,
        "n_targets_ahead": args.n_targets_ahead,
        "extra_time_factor": args.extra_time_factor,
        "total_time_sec": args.total_time_sec,
        "dynamic_priority_event": args.dynamic_priority_event,
        "hio_count": args.hio_count,
        "hio_priority": args.hio_priority,
        "hio_priority_max_multiplier": args.hio_priority_max_multiplier,
        "shio_count": args.shio_count,
        "shio_priority": args.shio_priority,
        "shio_priority_max_multiplier": args.shio_priority_max_multiplier,
        "priority_control_count": args.priority_control_count,
        "priority_control_seed": 20260729 + seed,
        "use_shield": args.use_shield,
        "command": command,
    }
    atomic_write_json(status_path, status)

    print(
        f"MC task {args.task_id}: policy={policy_tag}, seed={seed}, "
        f"checkpoint={policy['checkpoint_iteration']}, score={args.evaluation_reward_mix}, "
        f"priority_sum={args.priority_sum:g}"
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
