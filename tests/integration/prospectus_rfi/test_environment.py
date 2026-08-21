from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import bsk_rl  # noqa: F401
from examples.prospectus_rfi.acquisition_timeline import append_trajectory_snapshot
from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.environment import (
    AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT,
    LEGACY_AMOS2025_OBSERVATION_CONTRACT,
    make_environment_args,
)
from examples.prospectus_rfi.evaluate import scenario_fingerprint


CONFIG_DIR = Path(__file__).parents[3] / "examples" / "prospectus_rfi" / "configs"


def make_env(
    catalog_size,
    candidate_count=5,
    observation_contract=None,
    historical_heuristic=False,
):
    study = load_study_config(
        CONFIG_DIR / "mlp_selected.yaml", CONFIG_DIR / "base.yaml"
    )
    environment = replace(study.environment, candidate_count=candidate_count)
    kwargs = {}
    if observation_contract is not None:
        kwargs["observation_contract"] = observation_contract
    args = make_environment_args(
        environment,
        fixed_catalog_size=catalog_size,
        historical_heuristic=historical_heuristic,
        **kwargs,
    )
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
        timeline = []
        append_trajectory_snapshot(timeline, base)
        start = base.simulator.sim_time
        env.step({"SS1": 0})
        append_trajectory_snapshot(timeline, base)
        assert base.simulator.sim_time - start == pytest.approx(100.0)
        assert [row["sim_time_s"] for row in timeline] == [0.0, 100.0]
        assert timeline[-1]["cumulative_illuminated_observations"] >= 0.0
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


def test_frozen_policy_contract_preserves_scenario_seed_and_100_second_action():
    study_env = make_env(100, candidate_count=10, historical_heuristic=True)
    legacy_env = make_env(
        100,
        candidate_count=10,
        observation_contract=LEGACY_AMOS2025_OBSERVATION_CONTRACT,
    )
    try:
        study_observations, _ = study_env.reset(seed=8132025)
        legacy_observations, _ = legacy_env.reset(seed=8132025)
        study_base = study_env.unwrapped
        legacy_base = legacy_env.unwrapped

        assert study_observations["SS1"].shape == (91,)
        assert legacy_observations["SS1"].shape == (87,)
        assert legacy_base.satellites[0].action_space.n == 13
        assert scenario_fingerprint(study_base) == scenario_fingerprint(legacy_base)
        assert (
            study_base.satellites[0].dynamics.battery_charge_fraction
            == legacy_base.satellites[0].dynamics.battery_charge_fraction
        )

        start = legacy_base.simulator.sim_time
        legacy_env.step({"SS1": 0})
        assert legacy_base.simulator.sim_time - start == pytest.approx(100.0)
    finally:
        study_env.close()
        legacy_env.close()


def test_attention_control_has_checkpoint_fields_mask_and_300_second_action():
    study = load_study_config(
        CONFIG_DIR / "attention_amos2025_control.yaml",
        CONFIG_DIR / "base_amos2025_attention_control.yaml",
    )
    assert (
        study.environment.observation_layout
        == AMOS2025_ATTENTION_CONTROL_OBSERVATION_CONTRACT
    )
    args = make_environment_args(study.environment, fixed_catalog_size=100)
    args["log_level"] = "ERROR"
    env = gym.make("ConstellationTasking-v1", disable_env_checker=True, **args)
    try:
        observations, _ = env.reset(seed=8132025)
        base = env.unwrapped
        assert observations["SS1"].shape == (97,)
        assert base.satellites[0].action_space.n == 13
        assert 0.10 <= base.satellites[0].dynamics.battery_charge_fraction <= 0.40

        start = base.simulator.sim_time
        env.step({"SS1": 0})
        assert base.simulator.sim_time - start == pytest.approx(300.0)
    finally:
        env.close()
