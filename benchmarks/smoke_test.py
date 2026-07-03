"""Smoke-test the benchmark environments.

This constructs each benchmark environment from its ``BenchmarkEnv.env_args``
and steps it a few times to verify that it builds and runs without error. It
does not train a policy; it only exercises environment construction and the
Gymnasium/PettingZoo step loop so that CI can catch breakages in the benchmark
definitions quickly. Full training is driven by ``benchmark.py``.

Run from within the ``benchmarks`` directory so the benchmark modules (which
use top-level imports like ``from benchmark import BenchmarkEnv``) are
importable::

    cd benchmarks
    python smoke_test.py
"""

import argparse
import sys
from copy import deepcopy

from aeos import aeos_single
from nadir_science import nadir_science
from rso_inspection import rso_inspection

from bsk_rl import ConstellationTasking

# One representative environment per benchmark module. Add entries here to
# smoke-test additional benchmark environments.
BENCHMARK_ENVS = {
    "nadir_science": nadir_science,
    "aeos_single": aeos_single,
    "rso_inspection": rso_inspection,
}


def smoke_test_env(name, benchmark_env, n_steps):
    """Build a benchmark environment and step it ``n_steps`` times."""
    print(f"Smoke-testing benchmark env '{name}'...")

    # env_args carries RLlib-only callbacks that ConstellationTasking does not
    # accept. Strip them the same way the RLlib env creator does.
    env_args = deepcopy(benchmark_env.env_args)
    env_args.pop("episode_data_callback", None)
    env_args.pop("satellite_data_callback", None)

    env = ConstellationTasking(**env_args)
    env.reset(seed=0)
    for step in range(n_steps):
        action = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terminated, truncated, _ = env.step(action)
        if all(terminated.values()) or all(truncated.values()):
            break
    env.close()
    print(f"Benchmark env '{name}' completed {step + 1} step(s).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--n_steps",
        type=int,
        default=3,
        help="Number of environment steps to run per benchmark env.",
    )
    args = parser.parse_args()

    for name, benchmark_env in BENCHMARK_ENVS.items():
        smoke_test_env(name, benchmark_env, args.n_steps)

    print(f"All {len(BENCHMARK_ENVS)} benchmark env(s) smoke-tested successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
