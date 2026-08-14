from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import bsk_rl  # noqa: F401
from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.environment import make_environment_args
from examples.prospectus_rfi.evaluate import scenario_fingerprint


CONFIG_DIR = Path(__file__).parents[3] / "examples" / "prospectus_rfi" / "configs"


def make_env(catalog_size, candidate_count=5):
    study = load_study_config(
        CONFIG_DIR / "mlp_selected.yaml", CONFIG_DIR / "base.yaml"
    )
    environment = replace(study.environment, candidate_count=candidate_count)
    args = make_environment_args(environment, fixed_catalog_size=catalog_size)
    args["log_level"] = "ERROR"
    return gym.make("ConstellationTasking-v1", disable_env_checker=True, **args)


@pytest.mark.parametrize("catalog_size", [100, 400])
def test_catalog_extremes_and_fixed_100_second_image_action(catalog_size):
    env = make_env(catalog_size)
    try:
        observations, _ = env.reset(seed=1234)
        base = env.unwrapped
        assert base.scenario.sampled_catalog_size == catalog_size
        assert observations["SS1"].shape == (11 + 5 * 8,)
        assert base.satellites[0].action_space.n == 5 + 3
        start = base.simulator.sim_time
        env.step({"SS1": 0})
        assert base.simulator.sim_time - start == pytest.approx(100.0)
    finally:
        env.close()


def test_identical_seed_produces_matched_scenario_and_initial_battery():
    fingerprints = []
    batteries = []
    for _ in range(2):
        env = make_env(100)
        try:
            env.reset(seed=99123)
            base = env.unwrapped
            fingerprints.append(scenario_fingerprint(base))
            batteries.append(base.satellites[0].dynamics.battery_charge_fraction)
        finally:
            env.close()

    assert fingerprints[0] == fingerprints[1]
    np.testing.assert_allclose(batteries[0], batteries[1], rtol=0.0, atol=0.0)
    assert 0.20 <= batteries[0] <= 0.60
