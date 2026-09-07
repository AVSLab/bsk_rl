"""The bounded launcher fails closed before a broad or stale pilot can run."""

from copy import deepcopy
import pytest

from examples.multiagent_imaging.cluster.learning_pilot import (
    check_gate,
    mission_config,
    validate_updates,
)


def test_mission_contract_is_preserved():
    for mode in ("conflict", "continuous"):
        config = mission_config(mode)
        assert (config.n_sensors, config.n_targets, config.n_candidates) == (2, 100, 10)
        assert config.episode_duration_s == 45000
        assert config.discount_half_life_s == pytest.approx(45000)
        assert config.gae_trace_half_life_s == 6000
        assert config.communication_mode == "directed"
        assert config.information_case == "completion"
        assert config.transmit_duration_s == 300
        assert config.transmit_hold_s == 10
        assert config.metadata_bitrate_bps == 64000


def test_stale_source_or_runtime_cannot_skip_one_worker_validation():
    runtime = dict(
        source_sha256={"source.py": "abc"},
        packages={"ray": "2.35.0"},
        loaded={"basilisk_commit": "same"},
    )
    gate = dict(passed=True, validated_workers=1, runtime=deepcopy(runtime))
    check_gate(gate, runtime)
    for key, value in (("passed", False), ("validated_workers", 4)):
        bad = {**gate, key: value}
        with pytest.raises(ValueError):
            check_gate(bad, runtime)
    runtime["source_sha256"]["source.py"] = "changed"
    with pytest.raises(ValueError, match="stale"):
        check_gate(gate, runtime)


def test_completed_updates_must_cover_each_sensor_physical_time():
    config = mission_config("conflict")
    record = dict(
        losses={"total_loss": 1.0},
        finite_gradients=1,
        gradient_l2=0.1,
        parameter_change={"l2": 0.01},
        restore_validation={"matched_actions": True},
        episodes=[
            dict(
                simulated_seconds=45000,
                coordination={
                    "task_history": [
                        dict(sensor=f"sensor_{i}", start=0, end=45000) for i in range(2)
                    ]
                },
            )
        ],
    )
    validate_updates([record], config)
    short = deepcopy(record)
    short["episodes"][0]["coordination"]["task_history"][1]["end"] = 44000
    with pytest.raises(AssertionError, match="durations"):
        validate_updates([short], config)
    early = deepcopy(record)
    early["episodes"][0]["simulated_seconds"] = 44000
    with pytest.raises(AssertionError, match="horizon"):
        validate_updates([early], config)
