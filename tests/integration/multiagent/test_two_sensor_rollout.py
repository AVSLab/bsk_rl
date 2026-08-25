import pytest

from examples.multiagent_imaging.config import (
    BASE_GLOBAL_FEATURES,
    GLOBAL_FEATURES,
    MultiAgentImagingConfig,
    TARGET_FEATURES,
    TEAMMATE_FEATURES,
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


def test_information_cases_strictly_separate_teammate_state():
    summaries = {}
    cases = (
        ("independent", True, "independent"),
        ("centralized_information", True, "centralized"),
        ("intent_status", True, "intent_perfect"),
        ("intent_status", False, "intent_los_without_broadcast"),
    )
    for information_case, perfect_delivery, label in cases:
        config = MultiAgentImagingConfig(
            n_sensors=2,
            n_targets=4,
            n_candidates=2,
            episode_duration_s=120.0,
            max_step_duration_s=60.0,
            imaging_duration_s=60.0,
            information_case=information_case,
            perfect_metadata_delivery=perfect_delivery,
            seed=33,
        )
        env = build_environment(config)
        observations, _ = env.reset(seed=config.seed)
        initial = observations["sensor_0"][
            BASE_GLOBAL_FEATURES : BASE_GLOBAL_FEATURES + TEAMMATE_FEATURES
        ]
        observations, *_ = env.step({"sensor_0": 4, "sensor_1": 4})
        after_step = observations["sensor_0"][
            BASE_GLOBAL_FEATURES : BASE_GLOBAL_FEATURES + TEAMMATE_FEATURES
        ]
        summaries[label] = (initial, after_step)
        env.close()
    assert (summaries["independent"][0] == 0.0).all()
    assert (summaries["independent"][1] == 0.0).all()
    assert not (summaries["centralized"][0] == 0.0).all()
    assert not (summaries["centralized"][1] == 0.0).all()
    assert (summaries["intent_perfect"][0] == 0.0).all()
    assert not (summaries["intent_perfect"][1] == 0.0).all()
    assert (summaries["intent_los_without_broadcast"][0] == 0.0).all()
    assert (summaries["intent_los_without_broadcast"][1] == 0.0).all()


@pytest.mark.parametrize("n_sensors", [1, 2, 3])
def test_observation_size_is_constant_across_sensor_count(n_sensors):
    config = MultiAgentImagingConfig(
        n_sensors=n_sensors,
        n_targets=4,
        n_candidates=2,
        episode_duration_s=60.0,
        max_step_duration_s=60.0,
        imaging_duration_s=60.0,
        information_case="centralized_information",
        seed=35,
    )
    env = build_environment(config)
    observations, _ = env.reset(seed=config.seed)
    assert len(observations) == n_sensors
    assert all(
        observation.shape == (GLOBAL_FEATURES + 2 * TARGET_FEATURES,)
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
