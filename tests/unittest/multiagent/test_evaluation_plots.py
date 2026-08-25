import pytest

from examples.multiagent_imaging.plot_evaluation import plot_evaluation


def synthetic_result(n_sensors):
    sensors = [f"sensor_{index}" for index in range(n_sensors)]
    return {
        "pettingzoo_agents": sensors,
        "reward_history": {
            sensor: [{"time_s": 60.0, "reward": 1.0, "cumulative_reward": 1.0}]
            for sensor in sensors
        },
        "resource_history": {
            sensor: [
                {
                    "time_s": 60.0,
                    "battery_fraction": 0.9,
                    "storage_fraction": 0.1,
                    "wheel_speed_fraction": [0.1, 0.2, 0.3],
                }
            ]
            for sensor in sensors
        },
        "action_time_s": {sensor: {"Image": 60.0} for sensor in sensors},
        "onboard_products": {
            sensor: [
                {
                    "record_id": f"{sensor}-0",
                    "source_sensor": sensor,
                    "target_id": 0,
                    "capture_time": 60.0,
                    "delivery_time": None,
                    "quality": 1.0,
                    "storage_owner": sensor,
                }
            ]
            for sensor in sensors
        },
        "team_service_history": [],
        "per_sensor_metrics": {
            sensor: {"captures": 1, "deliveries": 0, "duplicate_attempts": 0}
            for sensor in sensors
        },
        "team_summary": {
            "unique_acquisition_count": n_sensors,
            "unique_service_count": 0,
            "team_value": 0.0,
        },
        "intent_conflicts": {"time_s": 0.0},
    }


@pytest.mark.parametrize("n_sensors", [1, 3])
def test_team_overview_is_generated_only_for_multi_sensor_runs(tmp_path, n_sensors):
    paths = plot_evaluation(synthetic_result(n_sensors), tmp_path)
    names = {path.name for path in paths}
    assert all(path.exists() for path in paths)
    assert ("multiagent_overview.pdf" in names) is (n_sensors > 1)
    assert ("multiagent_overview.png" in names) is (n_sensors > 1)
    for index in range(n_sensors):
        assert f"sensor_{index}_diagnostics.pdf" in names
        assert f"sensor_{index}_diagnostics.png" in names
