import numpy as np
import pandas as pd
import pytest

from examples.prospectus_rfi.acquisition_timeline import (
    METHODS,
    TOTAL_TIMELINE_TASKS,
    resample_step_trajectory,
    timeline_task_spec,
    verify_replay,
)


def test_timeline_array_mapping_pairs_both_methods_over_all_scenarios() -> None:
    specs = [timeline_task_spec(task_id) for task_id in range(TOTAL_TIMELINE_TASKS)]
    assert len(specs) == 600
    assert {spec.method for spec in specs} == set(METHODS)
    assert {(spec.method, spec.catalog_size, spec.seed) for spec in specs} == {
        (method, catalog_size, seed)
        for method in METHODS
        for catalog_size in (100, 200, 400)
        for seed in range(100)
    }


@pytest.mark.parametrize("task_id", [-1, TOTAL_TIMELINE_TASKS])
def test_timeline_array_mapping_rejects_invalid_ids(task_id: int) -> None:
    with pytest.raises(ValueError, match="task_id"):
        timeline_task_spec(task_id)


def test_decision_epoch_history_is_forward_filled_without_interpolation() -> None:
    frame = pd.DataFrame(
        {
            "sim_time_s": [0.0, 150.0, 300.0, 450.0],
            "cumulative_illuminated_observations": [0.0, 1.0, 3.0, 4.0],
            "method": ["test"] * 4,
        }
    )
    result = resample_step_trajectory(frame, interval_s=100.0, duration_s=400.0)
    assert result["sim_time_s"].tolist() == [0.0, 100.0, 200.0, 300.0, 400.0]
    assert result["cumulative_illuminated_observations"].tolist() == [
        0.0,
        0.0,
        1.0,
        3.0,
        3.0,
    ]


def test_resampling_snaps_submicrosecond_grid_drift() -> None:
    frame = pd.DataFrame(
        {
            "sim_time_s": [0.0, 300.0, 600.000000000001, 900.0],
            "cumulative_illuminated_observations": [0.0, 1.0, 2.0, 3.0],
        }
    )
    result = resample_step_trajectory(frame, interval_s=300.0, duration_s=900.0)
    assert result["cumulative_illuminated_observations"].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def accepted_episode() -> pd.Series:
    return pd.Series(
        {
            "scenario_fingerprint": "abc",
            "method": "heuristic_historical",
            "catalog_size": 100.0,
            "scenario_seed": 7.0,
            "successful_observations": 96.0,
            "illuminated_observations": 95.0,
            "useful_deliveries": 90.0,
            "total_downlinks": 92.0,
            "episode_duration_s": 45_000.0,
        }
    )


def replay_episode() -> dict:
    return {
        "scenario_fingerprint": "abc",
        "method": "heuristic_historical",
        "catalog_size": 100,
        "scenario_seed": 7,
        "successful_observations": 96,
        "illuminated_observations": 95,
        "useful_deliveries": 90,
        "total_downlinks": 92,
        "episode_duration_s": 45_000.0,
    }


def test_timeline_replay_accepts_matching_existing_episode() -> None:
    result = verify_replay(accepted_episode(), replay_episode())
    assert result["verified_against_existing_raw"] is True


def test_timeline_replay_rejects_changed_outcome() -> None:
    replay = replay_episode()
    replay["illuminated_observations"] = 94
    with pytest.raises(ValueError, match="did not reproduce"):
        verify_replay(accepted_episode(), replay)


def test_timeline_replay_rejects_changed_scenario() -> None:
    replay = replay_episode()
    replay["scenario_fingerprint"] = "different"
    with pytest.raises(ValueError, match="scenario_fingerprint"):
        verify_replay(accepted_episode(), replay)


def test_forward_fill_requires_complete_time_support() -> None:
    frame = pd.DataFrame(
        {
            "sim_time_s": [100.0, 45_000.0],
            "cumulative_illuminated_observations": [0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="span time 0"):
        resample_step_trajectory(frame)


def test_resampled_cumulative_count_remains_monotonic() -> None:
    frame = pd.DataFrame(
        {
            "sim_time_s": [0.0, 100.0, 250.0, 45_000.0],
            "cumulative_illuminated_observations": [0.0, 1.0, 2.0, 10.0],
        }
    )
    result = resample_step_trajectory(frame)
    assert np.all(np.diff(result["cumulative_illuminated_observations"]) >= 0.0)
