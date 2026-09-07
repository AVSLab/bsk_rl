"""Small real-Basilisk regressions for the prepared 200-episode campaign."""

import json

import pytest

from examples.multiagent_imaging.baseline_monte_carlo import (
    BaselineConfig,
    build_baseline,
    initial_conditions,
    run_episode,
    task_spec,
)


def tiny():
    return BaselineConfig(
        n_targets=6,
        n_candidates=3,
        episode_duration_s=120,
        max_step_duration_s=60,
        imaging_duration_s=60,
        downlink_duration_s=60,
        charge_duration_s=60,
    )


@pytest.mark.parametrize("environment", ["leo", "mixed"])
def test_information_cases_have_identical_initial_spacecraft_states(environment):
    states = []
    for case in ("independent", "centralized_full_state"):
        env = build_baseline(tiny(), case, environment, 7)
        try:
            env.reset(seed=7)
            states.append(initial_conditions(env))
            assert env.possible_agents == ["sensor_0", "sensor_1"]
            assert len(env.passive_satellites) == 6
            assert len(env.simulator.satellites) == 8
        finally:
            env.close()
    assert states[0] == states[1]


@pytest.mark.parametrize("task", [0, 50, 100, 150])
def test_small_episode_completes_without_any_radio_action(task):
    result = run_episode(tiny(), task_spec(task))
    assert result["horizon_reached"]
    assert result["sim_time_s"] == pytest.approx(120)
    assert result["communication"]["radio_action_count"] == 0
    assert sum(result["coordination"]["communication_time_s"].values()) == 0
    assert result["coverage"]["catalog_target_count"] == 6
    assert all("3" not in counts for counts in result["action_counts"].values())
    json.dumps(result, allow_nan=False)
