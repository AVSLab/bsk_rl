#!/usr/bin/env python3
"""Record the six-sensor, 100-target closest-angle Vizard episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.evaluate import run_rollout


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "multiagent_imaging"
    / "vizard_6sensors_100targets_3000s_seed0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a compact Vizard playback of coordinated multi-agent RSO "
            "imaging using a greedy minimum-angle-error controller."
        )
    )
    parser.add_argument("--n-sensors", type=int, default=6)
    parser.add_argument("--n-targets", type=int, default=100)
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=3000.0)
    parser.add_argument("--vizard-rate-s", type=float, default=2.0)
    parser.add_argument("--cooldown-orbits", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--information-case",
        choices=("independent", "ideal_completion", "completion"),
        default="ideal_completion",
    )
    parser.add_argument(
        "--retasking-mode", choices=("conflict", "continuous"), default="conflict"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing canonical playback and rollout summary.",
    )
    return parser.parse_args()


def canonical_playback_name(args: argparse.Namespace) -> str:
    rate = f"{args.vizard_rate_s:g}s".replace(".", "p")
    return (
        f"multiagent_closestAngle_{args.n_sensors}sensors_"
        f"{args.n_targets}targets_{args.duration_s:g}s_seed{args.seed}_"
        f"sample{rate}_UnityViz.bin"
    )


def main() -> int:
    args = parse_args()
    if args.vizard_rate_s <= 0.0:
        raise ValueError("--vizard-rate-s must be positive")
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = output_dir / canonical_playback_name(args)
    rollout_path = output_dir / "rollout.json"
    if not args.overwrite and (canonical.exists() or rollout_path.exists()):
        raise SystemExit(
            f"Output already exists in {output_dir}; pass --overwrite to replace it."
        )

    existing_bins = {path.resolve() for path in output_dir.rglob("*.bin")}
    config = MultiAgentImagingConfig(
        n_sensors=args.n_sensors,
        n_targets=args.n_targets,
        n_candidates=args.n_candidates,
        episode_duration_s=args.duration_s,
        reimage_cooldown_orbits=args.cooldown_orbits,
        information_case=args.information_case,
        retasking_mode=args.retasking_mode,
        seed=args.seed,
    )
    print(
        f"Running {args.n_sensors} sensors and {args.n_targets} targets for "
        f"{args.duration_s:g} simulated seconds"
    )
    print("Sensor constellation: original pre-Walker staggered orbit pattern")
    print("Controller: closest eligible candidate by pointing-angle error")
    print(f"Vizard sampling: every {args.vizard_rate_s:g} simulated seconds")

    result = run_rollout(
        config,
        controller="closest_angle",
        vizard_dir=str(output_dir),
        vizard_settings={
            "vizard_rate": args.vizard_rate_s,
            "multiagent_vizard": True,
            "orbitLinesOn": -1,
            "trueTrajectoryLinesOn": -1,
            "showOsculatingGroundTrackLines": -1,
            "showTruePathGroundTrackLines": -1,
            "showSpacecraftAsSprites": 0,
            "useSimpleLocationMarkers": -1,
            "showLocationCones": 1,
            "showLocationCommLines": -1,
            "showLocationLabels": 1,
            "spacecraftSizeMultiplier": 2.5,
            "linesAndFramesLineWidth": 3.0,
            "useLineRenderersForTargetLinesAndFrames": 1,
        },
    )

    rollout_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    created_bins = sorted(
        (
            path
            for path in output_dir.rglob("*.bin")
            if path.resolve() not in existing_bins
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not created_bins:
        raise RuntimeError("Simulation completed but Basilisk produced no Vizard file")
    newest = created_bins[-1]
    if canonical.exists() and canonical.resolve() != newest.resolve():
        canonical.unlink()
    if newest.resolve() != canonical.resolve():
        newest.replace(canonical)

    print(f"Simulation time: {result['sim_time_s']:.1f} s")
    print(f"Team summary: {result['team_summary']}")
    print(f"Vizard playback: {canonical}")
    print(f"Rollout summary: {rollout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
