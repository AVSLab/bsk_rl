"""Run and plot a short deterministic shared-controller multi-sensor episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.evaluate import run_rollout
from examples.multiagent_imaging.plot_evaluation import plot_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sensors", type=int, default=3)
    parser.add_argument("--n-targets", type=int, default=12)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--cooldown-orbits", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--information-case",
        choices=("independent", "ideal_completion", "completion"),
        default="completion",
    )
    parser.add_argument(
        "--los-broadcast",
        action="store_true",
        help="Require Earth-unoccluded geometry for finite completion broadcasts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/multiagent_imaging/quick_demo"),
    )
    args = parser.parse_args()

    config = MultiAgentImagingConfig(
        n_sensors=args.n_sensors,
        n_targets=args.n_targets,
        n_candidates=args.n_candidates,
        episode_duration_s=args.duration_s,
        reimage_cooldown_orbits=args.cooldown_orbits,
        information_case=args.information_case,
        link_mode="los" if args.los_broadcast else "ideal",
        seed=args.seed,
    )
    result = run_rollout(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "rollout.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(result_path.resolve())
    for path in plot_evaluation(result, args.output_dir / "plots"):
        print(path.resolve())


if __name__ == "__main__":
    main()
