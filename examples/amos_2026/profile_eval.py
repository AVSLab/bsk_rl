#!/usr/bin/env python3
"""Profile an evaluation script with cProfile and print the hottest functions.

Example:
    python3 examples/amos_2026/profile_eval.py --stats 50 -- \
        --seed 20 --target_env mixed --policy_name oct14_obsv7_1e_5lr_batch5000_gamma9997_20d80i --no_save_data
"""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import pstats
import runpy
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run policy_evaluation_2026.py under cProfile."
    )
    default_script = Path(__file__).resolve().parents[1] / "policy_evaluation_2026.py"
    default_profile = Path(__file__).resolve().parent / "policy_evaluation_2026.prof"
    parser.add_argument(
        "--script",
        type=Path,
        default=default_script,
        help="Evaluation script to run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_profile,
        help="Profile output file.",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=["cumtime", "tottime", "calls"],
        help="pstats sort key to print after the run.",
    )
    parser.add_argument(
        "--stats",
        type=int,
        default=40,
        help="Number of pstats rows to print.",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def main() -> None:
    args, script_args = parse_args()
    script = args.script.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    old_argv = sys.argv[:]
    sys.argv = [str(script), *script_args]
    try:
        profiler.enable()
        runpy.run_path(str(script), run_name="__main__")
    finally:
        profiler.disable()
        sys.argv = old_argv
        profiler.dump_stats(str(out))

    print(f"\nProfile written to: {out}\n")
    pstats.Stats(profiler).strip_dirs().sort_stats(args.sort).print_stats(args.stats)


if __name__ == "__main__":
    main()
