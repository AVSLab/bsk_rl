"""Tests for the multi-agent Vizard scene profile and live line monitor."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from Basilisk.utilities import vizSupport

from bsk_rl.sats.roles import SpacecraftRole
from bsk_rl.utils.multiagent_vizard import (
    IMAGING_LINE_HOLD_COLOR,
    IMAGING_LINE_SLEW_COLOR,
    MultiAgentVizardMonitor,
    configure_multiagent_vizard_models,
    prepare_multiagent_vizard_assets,
)
from examples.multiagent_imaging.environment import (
    _sample_leo_target_orbit,
    _sensor_orbit,
)


def _satellite(name, role, target=None):
    return SimpleNamespace(name=name, role=role, rso_target=target)


def test_assets_keep_six_sensors_and_split_100_targets_by_priority():
    sensors = [
        _satellite(f"sensor_{index}", SpacecraftRole.SENSING_AGENT)
        for index in range(6)
    ]
    targets = []
    for target_id in range(100):
        target = SimpleNamespace(id=target_id, priority=float(target_id))
        targets.append(
            _satellite(f"target_{target_id}", SpacecraftRole.PASSIVE_TARGET, target)
        )

    assets = prepare_multiagent_vizard_assets([*sensors, *targets], vizSupport)

    assert assets.sensor_names == [f"sensor_{index}" for index in range(6)]
    assert assets.sprite_list[:6] == ["bskSat"] * 6
    assert len(assets.sprite_list) == 106
    assert all(sprite.startswith("CIRCLE") for sprite in assets.sprite_list[6:])
    assert [target.name for target in assets.target_satellites] == [
        f"target_{index}" for index in range(100)
    ]
    assert {
        tier: len(names) for tier, names in assets.target_names_by_tier.items()
    } == {"lower": 34, "middle": 33, "upper": 33}


def test_models_override_only_the_six_sensor_cad_models():
    sensors = [
        _satellite(f"sensor_{index}", SpacecraftRole.SENSING_AGENT)
        for index in range(6)
    ]
    targets = [
        _satellite(
            f"target_{target_id}",
            SpacecraftRole.PASSIVE_TARGET,
            SimpleNamespace(id=target_id, priority=float(target_id)),
        )
        for target_id in range(100)
    ]
    assets = prepare_multiagent_vizard_assets([*sensors, *targets], vizSupport)
    support = SimpleNamespace(createCustomModel=Mock())

    configure_multiagent_vizard_models(Mock(), support, assets)

    assert support.createCustomModel.call_count == 1
    calls = support.createCustomModel.call_args_list
    assert calls[0].kwargs["simBodiesToModify"] == [
        f"sensor_{index}" for index in range(6)
    ]
    assert calls[0].kwargs["modelPath"] == "bskSat"
    assert calls[0].kwargs["scale"] == [1.5] * 3


def test_current_target_sampler_is_leo_only():
    for _ in range(200):
        orbit = _sample_leo_target_orbit()
        altitude_m = orbit.a - 6371e3
        assert 400e3 <= altitude_m <= 2000e3
        assert 0.0 <= orbit.e <= 0.02
        assert 0.0 <= orbit.i <= np.pi


def test_six_sensor_orbits_restore_pre_walker_pattern():
    orbits = [_sensor_orbit(index) for index in range(6)]

    np.testing.assert_allclose(
        [np.degrees(orbit.Omega) for orbit in orbits],
        [0.0, 30.0, 60.0, 90.0, 120.0, 150.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [np.degrees(orbit.f) for orbit in orbits],
        [0.0, 15.0, 30.0, 45.0, 60.0, 75.0],
        atol=1e-12,
    )
    assert [np.degrees(orbit.i) for orbit in orbits] == [
        97.0,
        70.0,
        97.0,
        70.0,
        97.0,
        70.0,
    ]
    assert [orbit.a - 6371e3 for orbit in orbits] == [
        700e3,
        800e3,
        700e3,
        800e3,
        700e3,
        800e3,
    ]
    assert {orbit.e for orbit in orbits} == {0.001}


def test_monitor_changes_line_from_yellow_to_green_during_valid_hold():
    line = SimpleNamespace(
        fromBodyName="sensor_0",
        toBodyName="target_0",
        lineColor=list(IMAGING_LINE_SLEW_COLOR),
    )
    target = SimpleNamespace(target_spacecraft=SimpleNamespace(name="target_0"))
    action = SimpleNamespace(
        _hold_target=target,
        _pointing_constraints_ok=Mock(return_value=(False, 1.0)),
    )
    sensor = SimpleNamespace(
        name="sensor_0",
        fsw=SimpleNamespace(_rso_line=line),
        _active_image_rso_action=action,
    )
    support = SimpleNamespace(targetLineList=[line], updateTargetLineList=Mock())
    monitor = MultiAgentVizardMonitor([sensor], [], Mock(), support)

    monitor.update_imaging_lines()
    assert list(line.lineColor) == IMAGING_LINE_SLEW_COLOR
    assert line.toBodyName == "target_0"

    action._pointing_constraints_ok.return_value = (True, 1.0)
    monitor.update_imaging_lines()
    assert list(line.lineColor) == IMAGING_LINE_HOLD_COLOR
    assert support.updateTargetLineList.call_count == 2
