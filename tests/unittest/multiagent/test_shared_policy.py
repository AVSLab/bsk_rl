import pytest

from examples.multiagent_imaging.train import (
    SHARED_POLICY_ID,
    make_shared_policy_mapping,
)


def test_shared_policy_mapping_accepts_only_explicit_sensors():
    mapping = make_shared_policy_mapping({"sensor_0", "sensor_1"})
    assert mapping("sensor_0") == SHARED_POLICY_ID
    assert mapping("sensor_1") == SHARED_POLICY_ID
    with pytest.raises(KeyError, match="Non-sensing"):
        mapping("target_0")


def test_all_four_sensors_contribute_to_one_policy():
    sensors = {f"sensor_{i}" for i in range(4)}
    mapping = make_shared_policy_mapping(sensors)
    # Trajectories remain distinct; policy IDs pool all real decisions.
    assert {mapping(sensor) for sensor in sensors} == {SHARED_POLICY_ID}
