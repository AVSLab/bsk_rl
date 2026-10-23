"""Hybrid imaging times must control physical capture and storage in an environment."""

import logging
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.utilities import macros

from bsk_rl import SatelliteTasking, act, data, obs, sats
from bsk_rl.scene import UniformTargets
from bsk_rl.sim import dyn, fsw


class TimedCamera(sats.ImagingSatellite):
    dyn_type = dyn.GroundStationDynModel
    fsw_type = fsw.ImagingFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [
        act.Image(1, hybrid=True),
        act.Drift(hybrid=True),
    ]


@pytest.fixture
def make_timed_imaging():
    environments = []

    def make(lower=2.0, upper=3.0, sim_rate=1.0, ready_at=0.0, max_duration=None):
        env = SatelliteTasking(
            satellite=TimedCamera(
                "timed_camera",
                sat_args=dict(
                    timeToleranceLower=lower,
                    timeToleranceUpper=upper,
                    instrumentBaudRate=1.0,
                    dataStorageCapacity=10.0,
                ),
            ),
            scenario=UniformTargets(1),
            rewarder=data.UniqueImageReward(),
            sim_rate=sim_rate,
            max_step_duration=sim_rate,
            time_limit=40.0,
            log_level="ERROR",
        )
        environments.append(env)
        env.satellite.action_builder.action_spec[0].max_duration = max_duration
        env.reset(seed=0)
        satellite = env.satellite
        target = env.scenario.targets[0]
        # Control visibility and guidance to isolate timing from orbital access
        # and attitude settling. The instrument and storage remain real models.
        satellite.opportunities = [
            dict(object=target, window=(0.0, 30.0), type="target")
        ]
        guidance = messaging.AttGuidMsg()
        access_payload = messaging.AccessMsgPayload()
        access_payload.hasAccess = 1
        access = messaging.AccessMsg().write(access_payload)
        controller = satellite.fsw.insControl
        if not all(
            hasattr(type(controller), name)
            for name in ("imagingTime", "timeToleranceLower", "timeToleranceUpper")
        ):
            pytest.skip("This Basilisk build does not support imaging-time tolerances.")
        controller.attGuidInMsg.subscribeTo(guidance)
        controller.locationAccessInMsg.subscribeTo(access)
        commands = controller.deviceCmdOutMsg.recorder()
        env.simulator.AddModelToTask(
            satellite.fsw.LocPointTask.name + satellite.name, commands, ModelPriority=0
        )

        def step(action):
            payload = messaging.AttGuidMsgPayload()
            if env.simulator.sim_time + sim_rate < ready_at:
                payload.sigma_BR = [1.0, 0, 0]
            guidance.write(payload)
            env.step(action)

        return SimpleNamespace(
            env=env,
            satellite=satellite,
            target=target,
            controller=controller,
            commands=commands,
            step=step,
        )

    yield make

    for env in environments:
        env.close()


@pytest.mark.parametrize(
    "lower,upper,ready_at,expected_capture,sim_rate",
    [
        (0.0, 0.0, 6.0, 6.0, 1.0),
        (0.0, 0.0, 20.0, 20.0, 1.0),
        (2.0, 0.0, 6.0, 10.0, 1.0),
        (2.0, 0.0, 20.0, 20.0, 1.0),
        (0.0, 3.0, 6.0, 6.0, 1.0),
        (0.0, 3.0, 20.0, None, 1.0),
        (2.0, 3.0, 6.0, 10.0, 1.0),
        (2.0, 3.0, 20.0, None, 1.0),
        (2.0, 3.0, 6.0, 10.0, 0.5),
        (2.0, 3.0, 20.0, None, 0.5),
    ],
    ids=[
        "disabled_early",
        "disabled_late",
        "lower_waits",
        "lower_allows_late",
        "upper_allows_early",
        "upper_blocks_late",
        "both_wait",
        "both_block_late",
        "half_step_wait",
        "half_step_block_late",
    ],
)
def test_hybrid_time_windows_control_physical_image_storage(
    make_timed_imaging,
    lower,
    upper,
    ready_at,
    expected_capture,
    sim_rate,
):
    imaging = make_timed_imaging(lower, upper, sim_rate, ready_at)
    imaging.step((1, np.array([4.0], dtype=np.float32)))
    while imaging.env.simulator.sim_time < 4.0:
        imaging.step(None)
    assert imaging.satellite.dynamics.storage_level == 0.0

    # Task at t=4 with an eight-second offset: desired capture t=12,
    # earliest capture t=10 when enabled, latest t=15 when enabled.
    imaging.step((0, np.array([8.0], dtype=np.float32)))
    assert imaging.controller.imagingTime == macros.sec2nano(12.0)
    assert imaging.controller.timeToleranceLower == macros.sec2nano(lower)
    assert imaging.controller.timeToleranceUpper == macros.sec2nano(upper)
    while imaging.env.simulator.sim_time < 23.0:
        imaging.step(None)

    commands = imaging.commands
    capture_times = commands.times()[commands.deviceCmd == 1] * macros.NANO2SEC
    if expected_capture is None:
        assert capture_times.size == 0
        assert imaging.satellite.dynamics.storage_level == 0.0
        assert not imaging.satellite.data_store.data.imaged
    else:
        np.testing.assert_allclose(capture_times, [expected_capture], rtol=0, atol=1e-9)
        assert imaging.satellite.dynamics.storage_level == pytest.approx(1.0)
        assert imaging.satellite.data_store.data.imaged == {imaging.target}


def test_retask_same_target_changes_physical_capture_time(make_timed_imaging):
    imaging = make_timed_imaging()
    imaging.step((imaging.target.id, np.array([20.0])))
    assert imaging.env.simulator.sim_time == 1.0

    imaging.step((imaging.target.id, np.array([8.0])))

    assert imaging.controller.imagingTime == macros.sec2nano(9)
    assert imaging.controller.timeToleranceLower == macros.sec2nano(2)
    assert imaging.controller.timeToleranceUpper == macros.sec2nano(3)
    while imaging.env.simulator.sim_time < 10.0:
        imaging.step(None)
    commands = imaging.commands
    np.testing.assert_array_equal(
        commands.times()[commands.deviceCmd == 1], [macros.sec2nano(7)]
    )
    assert imaging.satellite.dynamics.storage_level == pytest.approx(1.0)


def test_untimed_retask_warns_and_keeps_physical_schedule(make_timed_imaging, caplog):
    imaging = make_timed_imaging()
    imaging.step((imaging.target.id, np.array([20.0])))

    with caplog.at_level(logging.WARNING, logger=imaging.satellite.logger.name):
        imaging.step(imaging.target.id)

    assert "existing imagingTime and tolerances are retained" in caplog.text
    assert imaging.controller.imagingTime == macros.sec2nano(20)
    assert imaging.controller.timeToleranceLower == macros.sec2nano(2)
    assert imaging.controller.timeToleranceUpper == macros.sec2nano(3)
    while imaging.env.simulator.sim_time < 21.0:
        imaging.step(None)
    commands = imaging.commands
    np.testing.assert_array_equal(
        commands.times()[commands.deviceCmd == 1], [macros.sec2nano(18)]
    )
    assert imaging.satellite.dynamics.storage_level == pytest.approx(1.0)


@pytest.mark.parametrize(
    "upper,imaging_time,cap,expected_end",
    [
        (3.0, 8.0, None, 15.0),
        (3.0, 8.0, 5.0, 9.0),
        (0.0, 8.0, 5.0, 9.0),
        (0.0, 8.0, None, 30.0),
        (0.25, 0.0, None, 5.0),
        (3.0, 8.0, 0.25, 5.0),
    ],
    ids=[
        "upper_deadline",
        "duration_cap",
        "cap_without_upper",
        "access_end",
        "minimum_step",
        "cap_below_step",
    ],
)
def test_imaging_deadline_limits_real_step(
    make_timed_imaging, upper, imaging_time, cap, expected_end
):
    # An invalid pointing input prevents an early successful-image event.
    imaging = make_timed_imaging(upper=upper, ready_at=100, max_duration=cap)
    imaging.step((1, 4.0))
    while imaging.env.simulator.sim_time < 4.0:
        imaging.step(None)
    imaging.env.simulator.max_step_duration = 100.0

    imaging.step((0, np.array([imaging_time])))

    assert imaging.env.simulator.sim_time == pytest.approx(expected_end)
    assert imaging.satellite.requires_retasking
    assert imaging.satellite.dynamics.storage_level == 0.0
