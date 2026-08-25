import pytest

from examples.multiagent_imaging.config import (
    GLOBAL_FEATURES,
    MultiAgentImagingConfig,
    TARGET_FEATURES,
)
from examples.multiagent_imaging.environment import build_environment
from examples.multiagent_imaging.evaluate import run_rollout


def short_config():
    return MultiAgentImagingConfig(
        n_sensors=2,
        n_targets=4,
        n_candidates=2,
        episode_duration_s=360.0,
        max_step_duration_s=120.0,
        imaging_duration_s=120.0,
        seed=21,
        information_case="intent_status",
    )


def test_real_environment_exposes_sensors_and_not_passive_targets():
    config = short_config()
    env = build_environment(config)
    observations, _ = env.reset(seed=config.seed)
    assert env.possible_agents == ["sensor_0", "sensor_1"]
    assert len(env.passive_satellites) == 4
    assert all(
        value.shape == (GLOBAL_FEATURES + config.n_candidates * TARGET_FEATURES,)
        for value in observations.values()
    )
    assert env.action_space("sensor_0").n == 4 + config.n_candidates
    env.close()


@pytest.mark.parametrize(
    "information_case",
    ["independent", "centralized_information", "intent_status"],
)
def test_all_first_study_information_cases_reset(information_case):
    config = MultiAgentImagingConfig(
        n_sensors=2,
        n_targets=4,
        n_candidates=2,
        episode_duration_s=120.0,
        max_step_duration_s=60.0,
        imaging_duration_s=60.0,
        information_case=information_case,
        seed=31,
    )
    env = build_environment(config)
    observations, _ = env.reset(seed=config.seed)
    assert set(observations) == {"sensor_0", "sensor_1"}
    assert all(
        observation.shape == (GLOBAL_FEATURES + config.n_candidates * TARGET_FEATURES,)
        for observation in observations.values()
    )
    env.close()


def test_two_agent_rollout_is_deterministic():
    first = run_rollout(short_config())
    second = run_rollout(short_config())
    for key in (
        "sim_time_s",
        "pettingzoo_agents",
        "passive_target_count",
        "cumulative_reward",
        "action_counts",
        "completed_action_d_ts",
        "resource_history",
        "per_sensor_metrics",
        "team_summary",
        "team_service_ledger",
        "local_catalogs",
        "onboard_products",
    ):
        assert first[key] == second[key]
