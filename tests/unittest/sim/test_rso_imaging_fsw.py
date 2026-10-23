"""Sampled acquisition contracts using real Basilisk messages."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import simpleInstrumentController
from Basilisk.utilities import macros

from bsk_rl.sim.fsw.ground_imaging import ImagingFSWModel
from bsk_rl.sim.fsw.rso_imaging import (
    SpaceToSpaceImagingFSWModel,
    _AcquisitionGate,
)


class AcquisitionHarness:
    """Supply message inputs while controlling their independent timestamps."""

    def __init__(self):
        self.satellite = SimpleNamespace(name="camera_a", vizard_data={})
        self.simulator = SimpleNamespace(sim_time_ns=0)
        self.fsw_rate = 1.0  # [s]
        self.attGuidMsg = messaging.AttGuidMsg()
        self.access = messaging.AccessMsg()
        self.illumination = messaging.EclipseMsg()
        self.storage = messaging.DataStorageStatusMsg()
        self.insControl = simpleInstrumentController.simpleInstrumentController()
        self.insControl.attErrTolerance = 0.01  # [-, MRP norm]
        self.insControl.useRateTolerance = 1
        self.insControl.rateErrTolerance = 0.1  # [rad/s]
        self.dynamics = SimpleNamespace(
            dyn_rate=1.0,  # [s]
            instrument=SimpleNamespace(nodeBaudRate=1000.0),  # [bit/s]
            storageUnit=SimpleNamespace(
                storageCapacity=1000.0,  # [bit]
                storageUnitDataOutMsg=self.storage,
            ),
            storage_level=0.0,  # [bit]
            rso_partition_names={"83": "83"},
            rso_access_messages={"83": self.access},
        )
        self.target = SimpleNamespace(
            id="83",
            target_spacecraft=SimpleNamespace(
                dynamics=SimpleNamespace(
                    eclipse_index=0,
                    world=SimpleNamespace(
                        eclipseObject=SimpleNamespace(
                            eclipseOutMsgs=[self.illumination]
                        )
                    ),
                )
            ),
        )
        self.store(0.0)  # [bit]
        self.gate = _AcquisitionGate(self)
        self.start()

    def start(self, hold_s=3.0, mode="cumulative", duration=30.0, require_lit=True):
        """Start a fresh hold; durations are in seconds."""
        self.gate.start(self.target, duration, hold_s, mode, require_lit, 0.5)

    def sample(
        self,
        time_s,
        *,
        access=True,
        illumination=0.8,
        sigma=(0.0, 0.0, 0.0),
        omega=(0.0, 0.0, 0.0),
        timestamps=None,
        unwritten=None,
    ):
        """Write independent guidance/access/illumination inputs and run one tick."""
        now = macros.sec2nano(time_s)
        self.simulator.sim_time_ns = now
        timestamps = timestamps or {}
        guidance = messaging.AttGuidMsgPayload()
        guidance.sigma_BR = sigma
        guidance.omega_BR_B = omega
        access_payload = messaging.AccessMsgPayload()
        access_payload.hasAccess = int(access)
        illumination_payload = messaging.EclipseMsgPayload()
        illumination_payload.illuminationFactor = illumination
        for name, message, payload in (
            ("guidance", self.attGuidMsg, guidance),
            ("access", self.access, access_payload),
            ("illumination", self.illumination, illumination_payload),
        ):
            if name != unwritten:
                message.write(payload, macros.sec2nano(timestamps.get(name, time_s)))
        self.gate.UpdateState(now)
        return self.gate.command.read().deviceCmd

    def store(self, amount):
        """Publish a partition's physical storage amount [bit]."""
        payload = messaging.DataStorageStatusMsgPayload()
        payload.storedDataName.push_back("83")
        payload.storedData.push_back(amount)
        self.storage.write(payload)
        self.dynamics.storage_level = amount


@pytest.fixture
def acquisition():
    return AcquisitionHarness()


@pytest.mark.parametrize("mode", ["continuous", "cumulative"])
@pytest.mark.parametrize(
    "constraint_break",
    [
        {"access": False},
        {"sigma": (0.02, 0.0, 0.0)},  # [-, MRP norm]
        {"omega": (0.2, 0.0, 0.0)},  # [rad/s]
        {"illumination": 0.4},  # [-]
    ],
)
def test_each_constraint_break_resets_or_pauses_hold(
    acquisition, mode, constraint_break
):
    acquisition.start(mode=mode)
    assert acquisition.sample(1) == 0
    assert acquisition.sample(2) == 0
    assert acquisition.gate.held_s == 1  # [s]
    assert acquisition.sample(3, **constraint_break) == 0
    assert acquisition.gate.held_s == (0 if mode == "continuous" else 1)
    assert acquisition.sample(4) == 0  # The interrupted interval never counts.
    assert acquisition.sample(5) == 0
    assert acquisition.sample(6) == int(mode == "cumulative")
    if mode == "continuous":
        assert acquisition.sample(7) == 1


def test_irregular_intervals_weight_illumination_and_exclude_invalid_gaps(acquisition):
    acquisition.start(require_lit=False)
    assert acquisition.sample(1, illumination=0.2) == 0
    assert acquisition.sample(3, illumination=0.8) == 0
    assert acquisition.sample(4, access=False, illumination=1.0) == 0
    assert acquisition.sample(10, illumination=0.6) == 0
    assert acquisition.sample(11, illumination=0.4) == 1
    assert not acquisition.gate.check_capture()
    acquisition.store(1000)  # [bit]
    assert acquisition.gate.check_capture()
    record = acquisition.gate.records[0]
    assert record.hold_valid_time_s == 3  # [s]: two seconds plus one second.
    assert record.quality == pytest.approx(0.5)  # (1.0 + 0.5) / 3 seconds.
    assert record.capture_time == 11  # [s]
    assert acquisition.sample(12) == 0
    assert len(acquisition.gate.records) == 1


@pytest.mark.parametrize("message", ["guidance", "access", "illumination"])
@pytest.mark.parametrize("timestamp", [None, 0.0, 4.0])
def test_zero_hold_rejects_unwritten_stale_or_future_inputs(
    acquisition, message, timestamp
):
    acquisition.start(hold_s=0.0)  # [s]
    kwargs = (
        {"unwritten": message}
        if timestamp is None
        else {"timestamps": {message: timestamp}}
    )
    assert acquisition.sample(2, **kwargs) == 0
    assert acquisition.sample(3) == 1


@pytest.mark.parametrize("illumination", [float("nan"), -0.1, 1.1])
def test_invalid_illumination_cannot_create_a_product(acquisition, illumination):
    acquisition.start(hold_s=0.0, require_lit=False)  # [s]
    assert acquisition.sample(1, illumination=illumination) == 0
    assert acquisition.sample(2) == 1


def test_storage_readiness_and_full_image_confirmation(acquisition):
    acquisition.start(hold_s=0.0)  # [s]
    acquisition.dynamics.storageUnit.storageCapacity = 999  # [bit]
    assert acquisition.sample(1) == 0
    acquisition.dynamics.storageUnit.storageCapacity = 1000  # [bit]
    assert acquisition.sample(2) == 1
    acquisition.store(999)  # [bit]
    assert not acquisition.gate.check_capture()
    acquisition.store(1000)  # [bit]
    assert acquisition.gate.check_capture()
    assert acquisition.gate.check_capture()
    acquisition.gate.cancel()
    assert len(acquisition.gate.records) == 1
    assert acquisition.gate.command.read().deviceCmd == 0


def test_deadline_is_exclusive_and_cancellation_starts_a_new_hold(acquisition):
    acquisition.start(duration=4.0)  # [s]
    for time_s in range(1, 5):
        assert acquisition.sample(time_s) == 0
    assert acquisition.gate.target is None
    acquisition.start()
    for time_s in (5, 6):
        assert acquisition.sample(time_s) == 0
    acquisition.gate.cancel()
    assert acquisition.gate.held_s == 0
    acquisition.start(hold_s=0.0)  # [s]
    assert acquisition.sample(6) == 0  # Guidance from the previous action.
    assert acquisition.sample(7) == 1
    assert acquisition.sample(8) == 0


def test_gateway_uses_existing_controller_api_and_visualizes_the_actual_command():
    fsw = object.__new__(SpaceToSpaceImagingFSWModel)
    harness = AcquisitionHarness()
    sensor = SimpleNamespace()
    fsw.logger = Mock()
    fsw.satellite = harness.satellite
    fsw.satellite.vizard_data["genericSensorList"] = [sensor]
    fsw.attGuidMsg = harness.attGuidMsg
    fsw.insControl = harness.insControl
    fsw.satellite.simulator = SimpleNamespace(AddModelToTask=Mock())
    fsw.satellite.dynamics = SimpleNamespace(
        task_name="cameraDynamics",
        instrument=SimpleNamespace(nodeStatusInMsg=messaging.DeviceCmdMsgReader()),
    )
    with patch.object(ImagingFSWModel, "_set_gateway_msgs"):
        fsw._set_gateway_msgs()
    command = messaging.DeviceCmdMsgPayload()
    command.deviceCmd = 1
    fsw.rso_capture.command.write(command)
    assert fsw.dynamics.instrument.nodeStatusInMsg().deviceCmd == 1
    assert sensor.genericSensorCmdInMsg().deviceCmd == 1
    fsw.simulator.AddModelToTask.assert_called_once_with(
        "cameraDynamics", fsw.rso_capture, ModelPriority=900
    )
