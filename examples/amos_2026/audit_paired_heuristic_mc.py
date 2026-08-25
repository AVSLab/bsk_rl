#!/usr/bin/env python3
"""Validate completeness and seed pairing of the AMOS 2026 heuristic campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_TAGS = ("heur_angle", "heur_candidate_priority")


def parse_seed_range(spec: str) -> range:
    start_text, stop_text = spec.split(":", maxsplit=1)
    start, stop = int(start_text), int(stop_text)
    if stop <= start:
        raise ValueError("expected seed range must satisfy START < STOP")
    return range(start, stop)


def status_paths(root: Path) -> list[Path]:
    return sorted(root.glob("seeds_*/*/seed_*/mc_status.json"))


def metrics_files(seed_dir: Path) -> list[Path]:
    return sorted(seed_dir.glob("metrics_*.json")) + sorted(
        seed_dir.glob("*/metrics_*.json")
    )


def target_catalog(seed_dir: Path) -> Path | None:
    paths = sorted(seed_dir.glob("target_catalog.csv")) + sorted(
        seed_dir.glob("*/target_catalog.csv")
    )
    return paths[-1] if paths else None


def catalog_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_catalog(path: Path) -> bytes:
    rows = catalog_rows(path)
    fields = (
        "target_id",
        "initial_priority",
        "priority_event_kind",
        "priority_after_event",
        "realized_initial_priority_max",
    )
    normalized = [
        {field: row.get(field, "") for field in fields}
        for row in sorted(rows, key=lambda row: int(row["target_id"]))
    ]
    return json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()


def catalog_hash(path: Path) -> str:
    return hashlib.sha256(canonical_catalog(path)).hexdigest()


def config_errors(status: dict[str, Any]) -> list[str]:
    expected = {
        "evaluation_reward_mix": "100d00i",
        "target_env": "mixed",
        "exact_mix_counts": True,
        "n_targets": 100,
        "n_targets_ahead": 10,
        "priority_sum": 100.0,
        "priority_uniform_low": 0.0,
        "priority_uniform_high": None,
        "total_time_sec": 45000.0,
        "reimage_cooldown_orbits": 2.0,
        "dynamic_priority_event": "on",
        "hio_count": 5,
        "hio_priority": 5.0,
        "hio_priority_max_multiplier": None,
        "shio_count": 3,
        "shio_priority": 10.0,
        "shio_priority_max_multiplier": None,
        "shield_only": False,
        "use_shield": False,
    }
    errors = []
    for key, value in expected.items():
        if status.get(key) != value:
            errors.append(f"{key}={status.get(key)!r}, expected {value!r}")
    seed = int(status.get("seed", -1))
    expected_priority_seed = 20260729 + seed
    if status.get("priority_control_seed_base") != 20260729:
        errors.append(
            "priority_control_seed_base="
            f"{status.get('priority_control_seed_base')!r}, expected 20260729"
        )
    if status.get("priority_control_seed") != expected_priority_seed:
        errors.append(
            f"priority_control_seed={status.get('priority_control_seed')!r}, "
            f"expected {expected_priority_seed}"
        )
    command = [str(item) for item in status.get("command", [])]
    for required in (
        "--reimage_cooldown_orbits",
        "2.0",
        "--priority_control_seed",
        str(expected_priority_seed),
        "--no_shield",
    ):
        if required not in command:
            errors.append(f"command missing {required}")
    return errors


def reference_catalogs(root: Path) -> dict[int, set[str]]:
    hashes: dict[int, set[str]] = {}
    for path in status_paths(root):
        try:
            status = json.loads(path.read_text())
            seed = int(status["seed"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if status.get("state") != "completed" or int(status.get("returncode", -1)) != 0:
            continue
        catalog = target_catalog(path.parent)
        if catalog is not None:
            hashes.setdefault(seed, set()).add(catalog_hash(catalog))
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="0:100")
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Optional primary policy-MC root for target-catalog pairing checks.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    expected_seeds = set(parse_seed_range(args.expected_seeds))
    expected_pairs = {(tag, seed) for tag in EXPECTED_TAGS for seed in expected_seeds}
    records: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    issues: list[str] = []

    for path in status_paths(args.input_root):
        try:
            status = json.loads(path.read_text())
            key = (str(status["policy_tag"]), int(status["seed"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"unreadable status {path}: {exc}")
            continue
        if key not in expected_pairs:
            continue
        if key in records:
            issues.append(f"duplicate status for {key}: {records[key][1]} and {path}")
            continue
        records[key] = (status, path)

    missing = sorted(expected_pairs - set(records))
    issues.extend(f"missing status for {key}" for key in missing)
    catalog_hashes: dict[tuple[str, int], str] = {}

    for key, (status, path) in sorted(records.items()):
        if status.get("state") != "completed" or int(status.get("returncode", -1)) != 0:
            issues.append(
                f"{key} not completed cleanly: state={status.get('state')!r}, "
                f"returncode={status.get('returncode')!r}"
            )
        for error in config_errors(status):
            issues.append(f"{key}: {error}")
        if not metrics_files(path.parent):
            issues.append(f"{key}: missing metrics JSON")
        catalog = target_catalog(path.parent)
        if catalog is None:
            issues.append(f"{key}: missing target_catalog.csv")
        else:
            row_count = len(catalog_rows(catalog))
            if row_count != 100:
                issues.append(f"{key}: target catalog has {row_count} rows, expected 100")
            catalog_hashes[key] = catalog_hash(catalog)

    for seed in sorted(expected_seeds):
        angle = catalog_hashes.get(("heur_angle", seed))
        priority = catalog_hashes.get(("heur_candidate_priority", seed))
        if angle is not None and priority is not None and angle != priority:
            issues.append(f"seed {seed}: heuristic target catalogs are not paired")

    reference_summary: dict[str, Any] | None = None
    if args.reference_root is not None:
        reference = reference_catalogs(args.reference_root)
        reference_summary = {
            "root": str(args.reference_root.resolve()),
            "seeds_with_catalogs": len(reference),
        }
        for seed in sorted(expected_seeds):
            hashes = reference.get(seed, set())
            if not hashes:
                issues.append(f"seed {seed}: no completed reference-policy catalog")
                continue
            if len(hashes) != 1:
                issues.append(
                    f"seed {seed}: reference policy runs contain {len(hashes)} catalogs"
                )
                continue
            heuristic_hash = catalog_hashes.get(("heur_angle", seed))
            if heuristic_hash is not None and heuristic_hash not in hashes:
                issues.append(f"seed {seed}: heuristic/reference target catalogs differ")

    report = {
        "input_root": str(args.input_root.resolve()),
        "expected_controllers": list(EXPECTED_TAGS),
        "expected_seed_range": args.expected_seeds,
        "expected_episode_count": len(expected_pairs),
        "found_episode_count": len(records),
        "clean_episode_count": sum(
            status.get("state") == "completed"
            and int(status.get("returncode", -1)) == 0
            and bool(metrics_files(path.parent))
            for status, path in records.values()
        ),
        "paired_catalog_seed_count": sum(
            catalog_hashes.get(("heur_angle", seed))
            == catalog_hashes.get(("heur_candidate_priority", seed))
            and ("heur_angle", seed) in catalog_hashes
            for seed in expected_seeds
        ),
        "reference": reference_summary,
        "passed": not issues,
        "issues": issues,
    }
    output = args.output or args.input_root / "manifests" / "paired_completion_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"audit={output}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
