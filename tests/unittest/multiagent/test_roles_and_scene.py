from types import SimpleNamespace
from unittest.mock import MagicMock

from bsk_rl import SensingAgentConstellationTasking
from bsk_rl.sats.roles import SpacecraftRole
from bsk_rl.scene.rso_targets import RandomSatellites


def _satellite(name, role):
    return SimpleNamespace(
        name=name,
        role=role,
        sat_args_generator={
            "bufferNames": [],
            "transmitterNumBuffers": 0,
        },
        dynamics=SimpleNamespace(
            targetLocation=MagicMock(),
            scObject=SimpleNamespace(scStateOutMsg=object()),
        ),
    )


def test_sensing_agents_exclude_passive_targets():
    env = object.__new__(SensingAgentConstellationTasking)
    env._sensing_satellites = [
        _satellite("sensor_0", SpacecraftRole.SENSING_AGENT),
        _satellite("sensor_1", SpacecraftRole.SENSING_AGENT),
    ]
    env._passive_satellites = [_satellite("target_0", SpacecraftRole.PASSIVE_TARGET)]
    assert env.possible_agents == ["sensor_0", "sensor_1"]
    assert [sat.name for sat in env.passive_satellites] == ["target_0"]


def test_rso_scene_registers_every_target_with_every_sensor():
    sensors = [
        _satellite("sensor_0", SpacecraftRole.SENSING_AGENT),
        _satellite("sensor_1", SpacecraftRole.SENSING_AGENT),
        _satellite("sensor_2", SpacecraftRole.SENSING_AGENT),
    ]
    targets = [
        _satellite("target_0", SpacecraftRole.PASSIVE_TARGET),
        _satellite("target_1", SpacecraftRole.PASSIVE_TARGET),
    ]
    scenario = RandomSatellites(None, n_targets=2, hio_count=0, shio_count=0)
    scenario.link_satellites([*sensors, *targets])
    scenario.reset_during_sim_init()

    for sensor in sensors:
        assert sensor.sat_args_generator["bufferNames"] == ["target_0", "target_1"]
        assert sensor.sat_args_generator["transmitterNumBuffers"] == 2
        assert sensor.dynamics.targetLocation.addSpacecraftToModel.call_count == 2


def test_legacy_single_sensor_buffer_configuration_is_unchanged():
    sensor = _satellite("SS1", SpacecraftRole.SENSING_AGENT)
    # Existing AMOS target definitions did not carry an explicit passive role.
    targets = [
        _satellite("target_0", SpacecraftRole.SENSING_AGENT),
        _satellite("target_1", SpacecraftRole.SENSING_AGENT),
    ]
    scenario = RandomSatellites("SS1", n_targets=2, hio_count=0, shio_count=0)
    scenario.link_satellites([sensor, *targets])
    assert sensor.sat_args_generator["bufferNames"] == [
        "SS1",
        "target_0",
        "target_1",
    ]
