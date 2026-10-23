"""Exercise BSK-RL timing configuration against the real Basilisk controller.

Guidance and access messages are controlled so that each test isolates the time
window. No pointing, orbit, or visibility randomness can hide a timing failure.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import simpleInstrumentController
from Basilisk.utilities import macros

from bsk_rl.sats import ImagingSatellite
from bsk_rl.sim.fsw import ImagingFSWModel


@pytest.fixture
def imaging():
    controller = simpleInstrumentController.simpleInstrumentController()
    if not all(
        hasattr(type(controller), name)
        for name in ("imagingTime", "timeToleranceLower", "timeToleranceUpper")
    ):
        pytest.skip("This Basilisk build does not support imaging-time tolerances.")
    guidance = messaging.AttGuidMsg().write(messaging.AttGuidMsgPayload())
    access_payload = messaging.AccessMsgPayload()
    access_payload.hasAccess = 1
    access = messaging.AccessMsg().write(access_payload)
    controller.attGuidInMsg.subscribeTo(guidance)
    controller.locationAccessInMsg.subscribeTo(access)
    controller.attErrTolerance = 0.01
    controller.useRateTolerance = 1
    controller.rateErrTolerance = 0.01
    # Initialize the native output message before calling the model directly.
    controller.SelfInit()
    controller.Reset(0)

    fsw = object.__new__(ImagingFSWModel)
    fsw.logger = Mock()
    fsw.satellite = SimpleNamespace(
        name="timed_camera",
        simulator=SimpleNamespace(sim_time=100.0, enableTask=Mock()),
        dynamics=SimpleNamespace(
            reset_for_action=Mock(),
            instrumentPowerSink=SimpleNamespace(powerStatus=0),
            imagingTarget=SimpleNamespace(r_LP_P_Init=None),
            instrument=SimpleNamespace(nodeDataName=None),
        ),
    )
    fsw.insControl = controller
    fsw.tasks = []
    fsw.fsw_proc = Mock()
    fsw._zero_gateway_msgs = Mock()
    return SimpleNamespace(
        fsw=fsw,
        controller=controller,
        guidance=guidance,
        access=access,
    )


def configure_window(imaging, lower, upper):
    """Set tolerances through the same BSK-RL setup method used at reset."""
    ImagingFSWModel.LocPointTask.setup_instrument_time_tolerance(
        SimpleNamespace(insControl=imaging.controller),
        timeToleranceLower=lower,
        timeToleranceUpper=upper,
    )


@pytest.mark.parametrize(
    "lower,upper",
    [(0.0, 0.0), (2.0, 0.0), (0.0, 3.0), (2.0, 3.0)],
    ids=["disabled", "lower_only", "upper_only", "both"],
)
@pytest.mark.parametrize(
    "time_ns",
    [
        macros.sec2nano(108) - 1,
        macros.sec2nano(108),
        macros.sec2nano(113),
        macros.sec2nano(113) + 1,
    ],
    ids=[
        "before_lower",
        "at_lower",
        "at_upper",
        "after_upper",
    ],
)
def test_capture_command_respects_enabled_bounds(imaging, lower, upper, time_ns):
    configure_window(imaging, lower, upper)
    imaging.fsw.action_image([6378137, 0, 0], "images", imaging_time=10.0)
    assert imaging.controller.imagingTime == macros.sec2nano(110)

    imaging.controller.UpdateState(time_ns)

    expected = (lower == 0 or time_ns >= macros.sec2nano(108)) and (
        upper == 0 or time_ns <= macros.sec2nano(113)
    )
    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == int(expected)
    assert imaging.controller.imaged == int(expected)


@pytest.mark.parametrize("imaging_time", [0.0, 2.5])
def test_capture_time_is_relative_to_nonzero_simulation_time(imaging, imaging_time):
    configure_window(imaging, 0.5, 0.75)

    imaging.fsw.action_image([6378137, 0, 0], "images", imaging_time=imaging_time)

    assert imaging.controller.imagingTime == macros.sec2nano(100 + imaging_time)
    assert imaging.controller.timeToleranceLower == macros.sec2nano(0.5)
    assert imaging.controller.timeToleranceUpper == macros.sec2nano(0.75)
    assert imaging.fsw.dynamics.instrumentPowerSink.powerStatus == 1
    assert imaging.fsw.dynamics.instrument.nodeDataName == "images"


def test_omitted_capture_time_preserves_existing_schedule(imaging):
    configure_window(imaging, 2.0, 3.0)
    imaging.controller.imagingTime = macros.sec2nano(110)

    imaging.fsw.action_image([6378137, 0, 0], "images")

    assert imaging.controller.imagingTime == macros.sec2nano(110)
    assert imaging.controller.timeToleranceLower == macros.sec2nano(2)
    assert imaging.controller.timeToleranceUpper == macros.sec2nano(3)
    imaging.controller.UpdateState(macros.sec2nano(107))
    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 0
    imaging.controller.UpdateState(macros.sec2nano(108))
    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 1


@pytest.mark.parametrize("imaging_time", [0.0, 2.5])
def test_retask_updates_capture_time_without_changing_tolerances(imaging, imaging_time):
    configure_window(imaging, 0.5, 0.75)
    imaging.fsw.action_image([6378137, 0, 0], "images", imaging_time=10.0)
    imaging.fsw.simulator.sim_time = 120.0
    satellite = SimpleNamespace(fsw=imaging.fsw, simulator=imaging.fsw.simulator)

    ImagingSatellite.update_imaging_time(satellite, imaging_time)

    desired_ns = macros.sec2nano(120 + imaging_time)
    assert imaging.controller.imagingTime == desired_ns
    assert imaging.controller.timeToleranceLower == macros.sec2nano(0.5)
    assert imaging.controller.timeToleranceUpper == macros.sec2nano(0.75)
    imaging.controller.UpdateState(desired_ns - macros.sec2nano(0.5) - 1)
    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 0
    imaging.controller.UpdateState(desired_ns)
    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 1


def test_disabling_bounds_removes_previous_window(imaging):
    configure_window(imaging, 2.0, 3.0)
    imaging.fsw.action_image([6378137, 0, 0], "images", imaging_time=10.0)
    imaging.controller.UpdateState(macros.sec2nano(114))
    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 0

    configure_window(imaging, 0.0, 0.0)
    imaging.controller.UpdateState(macros.sec2nano(115))

    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 1


@pytest.mark.parametrize("constraint", ["pointing", "rate", "access"])
def test_valid_time_window_does_not_bypass_other_constraints(imaging, constraint):
    configure_window(imaging, 2.0, 3.0)
    imaging.fsw.action_image([6378137, 0, 0], "images", imaging_time=10.0)
    guidance = messaging.AttGuidMsgPayload()
    access = messaging.AccessMsgPayload()
    access.hasAccess = 1
    if constraint == "pointing":
        guidance.sigma_BR = [0.1, 0, 0]
    elif constraint == "rate":
        guidance.omega_BR_B = [0.1, 0, 0]
    else:
        access.hasAccess = 0
    imaging.guidance.write(guidance)
    imaging.access.write(access)

    imaging.controller.UpdateState(macros.sec2nano(110))

    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 0


def test_time_window_keeps_one_capture_per_task(imaging):
    configure_window(imaging, 2.0, 3.0)
    imaging.fsw.action_image([6378137, 0, 0], "images", imaging_time=10.0)
    commands = []
    for time_s in (107, 108, 109, 110, 113, 114):
        imaging.controller.UpdateState(macros.sec2nano(time_s))
        commands.append(imaging.controller.deviceCmdOutMsg.read().deviceCmd)
    assert commands == [0, 1, 0, 0, 0, 0]

    imaging.fsw.simulator.sim_time = 120.0
    imaging.fsw.action_image([6378137, 0, 0], "images", imaging_time=10.0)
    imaging.controller.UpdateState(macros.sec2nano(128))
    assert imaging.controller.deviceCmdOutMsg.read().deviceCmd == 1
