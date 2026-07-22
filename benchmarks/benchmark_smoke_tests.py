"""Smoke-test the benchmark environments.

Builds each benchmark environment and steps it a few times to check that it
constructs and runs; it does not train. Full training is in ``benchmark.py``.

Named ``*_tests.py`` so pytest's default ``*_test.py`` discovery skips it in
the regular test job, which lacks RLlib. Run from the ``benchmarks`` directory::

    pytest -v benchmark_smoke_tests.py
"""

from copy import deepcopy

import pytest
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

N_STEPS = 3


@pytest.mark.parametrize("name", list(BENCHMARK_ENVS))
def test_benchmark_env_smoke(name):
    """Build a benchmark environment and step it ``N_STEPS`` times."""
    benchmark_env = BENCHMARK_ENVS[name]

    # env_args carries RLlib-only callbacks that ConstellationTasking does not
    # accept. Strip them the same way the RLlib env creator does.
    env_args = deepcopy(benchmark_env.env_args)
    env_args.pop("episode_data_callback", None)
    env_args.pop("satellite_data_callback", None)

    env = ConstellationTasking(**env_args)
    try:
        env.reset(seed=0)
        for _ in range(N_STEPS):
            action = {agent: env.action_space(agent).sample() for agent in env.agents}
            _, _, terminated, truncated, _ = env.step(action)
            if all(terminated.values()) or all(truncated.values()):
                break
    finally:
        env.close()
