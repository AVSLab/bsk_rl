import csv
import json
from pathlib import Path
import sys

from examples.amos_2026.audit_paired_heuristic_mc import main


def write_episode(root: Path, tag: str, mode: str, seed: int) -> None:
    seed_dir = root / f"seeds_000_001/{tag}/seed_{seed:03d}"
    run_dir = seed_dir / "run"
    run_dir.mkdir(parents=True)
    priority_control_seed = 20260729 + seed
    command = [
        "evaluator",
        "--reimage_cooldown_orbits",
        "2.0",
        "--priority_control_seed",
        str(priority_control_seed),
        "--no_shield",
    ]
    status = {
        "state": "completed",
        "returncode": 0,
        "seed": seed,
        "policy_tag": tag,
        "controller_mode": mode,
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
        "priority_control_seed_base": 20260729,
        "priority_control_seed": priority_control_seed,
        "hio_count": 5,
        "hio_priority": 5.0,
        "hio_priority_max_multiplier": None,
        "shio_count": 3,
        "shio_priority": 10.0,
        "shio_priority_max_multiplier": None,
        "shield_only": False,
        "use_shield": False,
        "command": command,
    }
    (seed_dir / "mc_status.json").write_text(json.dumps(status))
    (run_dir / "metrics_test.json").write_text("{}")
    with (run_dir / "target_catalog.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "target_id",
                "initial_priority",
                "priority_event_kind",
                "priority_after_event",
                "realized_initial_priority_max",
            ],
        )
        writer.writeheader()
        for target_id in range(100):
            writer.writerow(
                {
                    "target_id": target_id,
                    "initial_priority": 1.0 + seed + target_id / 1000,
                    "priority_event_kind": "HIO" if target_id < 5 else "",
                    "priority_after_event": 5.0 if target_id < 5 else "",
                    "realized_initial_priority_max": 1.5 + seed,
                }
            )


def test_paired_heuristic_audit_accepts_complete_matched_grid(tmp_path, monkeypatch):
    for seed in range(2):
        write_episode(tmp_path, "heur_angle", "angle", seed)
        write_episode(
            tmp_path,
            "heur_candidate_priority",
            "candidate_priority",
            seed,
        )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_paired_heuristic_mc.py",
            "--input-root",
            str(tmp_path),
            "--expected-seeds",
            "0:2",
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["clean_episode_count"] == 4
    assert report["paired_catalog_seed_count"] == 2
