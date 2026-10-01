"""Moving-target pointing with a physical hold-before-storage acquisition gate."""

from weakref import proxy

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros

from bsk_rl.sim.dyn.rso_imaging import SpaceToSpaceImagingDynModel
from bsk_rl.sim.fsw.base import Task, action
from bsk_rl.sim.fsw.ground_imaging import ImagingFSWModel
from bsk_rl.utils.functional import default_args
from bsk_rl.utils.rso_imaging import RSOImageRecord, illumination_factor


class _AcquisitionGate(sysModel.SysModel):
    """Write a one-tick instrument command only after a valid sampled hold.

    Runs in the dynamics task before instrument and storage. Hold time counts an
    interval only when both endpoints are valid. Guidance must have been written
    after the task started and be no older than one FSW/dynamics sampling interval.
    """

    def __init__(self, fsw) -> None:
        super().__init__()
        self.fsw = proxy(fsw)
        self.ModelTag = "rsoAcquisitionGate" + fsw.satellite.name
        self.command = messaging.DeviceCmdMsg()
        self.guidance = messaging.AttGuidMsgReader()
        self.guidance.subscribeTo(fsw.attGuidMsg)
        self.access = messaging.AccessMsgReader()
        self.illumination = messaging.EclipseMsgReader()
        self.records = []
        self.sequence = 0
        self.cancel()

    def Reset(self, current_time_ns) -> None:
        self.cancel()

    def cancel(self) -> None:
        """Disarm the physical instrument on cancellation, switching, or deadline."""
        if getattr(self, "pulse", None) is not None:
            self.check_capture()
        self.target = None
        self.pulse = None
        self.completed = False
        self.held_s = 0.0
        self.illumination_integral = 0.0
        self.previous_valid = False
        self.last_time_ns = None
        self.previous_illumination = 0.0
        self.command.write(messaging.DeviceCmdMsgPayload())

    def start(
        self,
        target,
        duration,
        hold_s,
        hold_mode,
        require_illumination,
        illumination_threshold,
    ) -> None:
        self.cancel()
        self.target = target
        # Read the gateway header: addAuthor redirects writes to this message.
        self.guidance.subscribeTo(self.fsw.attGuidMsg)
        self.access.subscribeTo(self.fsw.dynamics.rso_access_messages[target.id])
        target_dyn = target.target_spacecraft.dynamics
        self.illumination.subscribeTo(
            target_dyn.world.eclipseObject.eclipseOutMsgs[target_dyn.eclipse_index]
        )
        self.start_time_ns = self.fsw.simulator.sim_time_ns
        self.deadline_ns = self.start_time_ns + macros.sec2nano(duration)
        self.hold_s = hold_s
        self.hold_mode = hold_mode
        self.require_illumination = require_illumination
        self.illumination_threshold = illumination_threshold
        self.completed = False

    def _partition_level(self) -> float:
        message = self.fsw.dynamics.storageUnit.storageUnitDataOutMsg.read()
        name = self.fsw.dynamics.rso_partition_names[self.target.id]
        return dict(zip(message.storedDataName, message.storedData)).get(name, 0.0)

    def check_capture(self) -> bool:
        """Confirm storage contains the full physical image before publishing it."""
        if self.pulse is not None and not self.completed:
            record, initial_level = self.pulse
            if self._partition_level() - initial_level >= record.size_bits - 1e-6:
                self.records.append(record)
                self.completed = True
        return bool(getattr(self, "completed", False))

    def _valid(self, now_ns) -> tuple[bool, float]:
        guidance = self.guidance
        guidance_age = now_ns - guidance.timeWritten()
        fresh_guidance = (
            guidance.isWritten()
            and guidance.timeWritten() > self.start_time_ns
            and 0
            <= guidance_age
            <= macros.sec2nano(max(self.fsw.fsw_rate, self.fsw.dynamics.dyn_rate))
        )
        payload = guidance()
        controller = self.fsw.insControl
        pointing = np.linalg.norm(payload.sigma_BR) <= controller.attErrTolerance
        if controller.useRateTolerance:
            pointing = (
                pointing
                and np.linalg.norm(payload.omega_BR_B) <= controller.rateErrTolerance
            )
        maximum_age = macros.sec2nano(self.fsw.dynamics.dyn_rate)
        access_age = now_ns - self.access.timeWritten()
        illumination_age = now_ns - self.illumination.timeWritten()
        illumination = illumination_factor(self.illumination())
        illuminated = (
            not self.require_illumination or illumination >= self.illumination_threshold
        )
        return bool(
            fresh_guidance
            and pointing
            and self.access.isWritten()
            and 0 <= access_age <= maximum_age
            and self.access().hasAccess
            and self.illumination.isWritten()
            and 0 <= illumination_age <= maximum_age
            and np.isfinite(illumination)
            and 0 <= illumination <= 1
            and illuminated
        ), illumination

    def UpdateState(self, current_time_ns) -> None:
        now = current_time_ns * macros.NANO2SEC
        self.command.write(
            messaging.DeviceCmdMsgPayload(), current_time_ns, self.moduleID
        )
        if self.target is None:
            return
        if self.pulse is not None:
            self.check_capture()
            return
        if current_time_ns >= self.deadline_ns:
            self.cancel()
            return
        valid, illumination = self._valid(current_time_ns)
        if self.last_time_ns is not None and valid and self.previous_valid:
            dt = (current_time_ns - self.last_time_ns) * macros.NANO2SEC
            self.held_s += dt
            self.illumination_integral += (
                0.5 * (illumination + self.previous_illumination) * dt
            )
        elif not valid and self.hold_mode == "continuous":
            self.held_s = 0.0
            self.illumination_integral = 0.0
        self.last_time_ns = current_time_ns
        self.previous_valid = valid
        self.previous_illumination = illumination
        if not valid or self.held_s + 1e-9 < self.hold_s:
            return
        dynamics = self.fsw.dynamics
        size = dynamics.instrument.nodeBaudRate * dynamics.dyn_rate
        if dynamics.storageUnit.storageCapacity - dynamics.storage_level < size - 1e-6:
            return
        quality = (
            self.illumination_integral / self.held_s
            if self.held_s > 0
            else illumination
        )
        self.sequence += 1
        record = RSOImageRecord(
            f"{self.fsw.satellite.name}:{self.sequence}",
            self.target.id,
            self.fsw.satellite.name,
            now,
            size,
            quality,
            self.held_s,
        )
        self.pulse = (record, self._partition_level())
        payload = messaging.DeviceCmdMsgPayload()
        payload.deviceCmd = 1
        self.command.write(payload, current_time_ns, self.moduleID)


class SpaceToSpaceImagingFSWModel(ImagingFSWModel):
    """Point at an RSO spacecraft and gate acquisition before physical storage.

    ``rso_capture`` temporarily evaluates sampled pointing, access, illumination,
    and hold constraints in the dynamics task, before instrument and storage.
    Continuous holds reset on a constraint break; cumulative holds pause. A zero
    hold still requires fresh valid guidance, and the action deadline is exclusive.

    The gate uses the existing Basilisk message API and confirms each full stored
    image once. It will be replaced by ``simpleInstrumentController`` when the
    corresponding upstream extension is available. This migration is tracked in
    `BSK-RL #358 <https://github.com/AVSLab/bsk_rl/issues/358>`_ and
    `Basilisk #1602 <https://github.com/AVSLab/basilisk/issues/1602>`_.
    """

    @classmethod
    def _requires_dyn(cls):
        return super()._requires_dyn() + [SpaceToSpaceImagingDynModel]

    def _set_gateway_msgs(self) -> None:
        super()._set_gateway_msgs()
        self.rso_capture = _AcquisitionGate(self)
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(self.rso_capture.command)
        self.simulator.AddModelToTask(
            self.dynamics.task_name, self.rso_capture, ModelPriority=900
        )
        for sensor in getattr(self.satellite, "vizard_data", {}).get(
            "genericSensorList", []
        ):
            command = messaging.DeviceCmdMsgReader()
            command.subscribeTo(self.rso_capture.command)
            sensor.genericSensorCmdInMsg = command

    class LocPointTask(ImagingFSWModel.LocPointTask):
        """Moving-target guidance with tolerances and an independent acquisition gate."""

        @default_args(inst_pHat_B=[0, 0, 1])
        def setup_location_pointing(self, inst_pHat_B, **kwargs) -> None:
            """Subscribe to observer navigation; target is bound when tasked."""
            self.locPoint.pHat_B = inst_pHat_B
            self.locPoint.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.locPoint.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            # A harmless initial binding permits Basilisk initialization. Imaging
            # replaces it with the explicitly selected target before acquisition.
            self.locPoint.scTargetInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.locPoint.useBoresightRateDamping = 1
            messaging.AttGuidMsg_C_addAuthor(
                self.locPoint.attGuidOutMsg, self.fsw.attGuidMsg
            )
            messaging.AttRefMsg_C_addAuthor(
                self.locPoint.attRefOutMsg, self.fsw.attRefMsg
            )
            self._add_model_to_task(self.locPoint, priority=1198)

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def setup_instrument_controller(
            self, imageAttErrorRequirement, imageRateErrorRequirement, **kwargs
        ) -> None:
            """Configure tolerances; acquisition gate owns the instrument command."""
            if (
                not np.isfinite(imageAttErrorRequirement)
                or imageAttErrorRequirement < 0
            ):
                raise ValueError("Pointing tolerance must be finite and nonnegative.")
            if imageRateErrorRequirement is not None and (
                not np.isfinite(imageRateErrorRequirement)
                or imageRateErrorRequirement < 0
            ):
                raise ValueError("Rate tolerance must be finite and nonnegative.")
            self.insControl.attErrTolerance = imageAttErrorRequirement
            self.insControl.useRateTolerance = int(
                imageRateErrorRequirement is not None
            )
            if imageRateErrorRequirement is not None:
                self.insControl.rateErrTolerance = imageRateErrorRequirement
            # The inherited controller supplies tolerance configuration;
            # the physical instrument subscribes exclusively to the hold gate.
            self.insControl.controllerStatus = 0

        def reset_for_action(self) -> None:
            """Clear an old acquisition before any FSW mode change."""
            self.fsw.rso_capture.check_capture()
            self.fsw.rso_capture.cancel()
            self.locPoint.Reset(self.fsw.simulator.sim_time_ns)
            self.insControl.controllerStatus = 0
            return Task.reset_for_action(self)

    @action
    def action_image_rso(
        self,
        target,
        duration=300.0,  # [s]
        hold_s=10.0,  # [s]
        hold_mode="cumulative",
        require_illumination=True,
        illumination_threshold=0.5,  # [-]
    ) -> None:
        """Track a bound target; keep the instrument gated until acquisition."""
        if target.id not in self.dynamics.rso_access_messages:
            raise ValueError("Target is not bound to this observer.")
        self.locPoint.scTargetInMsg.subscribeTo(
            target.target_spacecraft.dynamics.simpleNavObject.transOutMsg
        )
        self.dynamics.instrument.nodeDataName = self.dynamics.rso_partition_names[
            target.id
        ]
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.rso_capture.start(
            target,
            duration,
            hold_s,
            hold_mode,
            require_illumination,
            illumination_threshold,
        )
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)

    @action
    def action_image(self, *args, **kwargs) -> None:
        """Reject ground-target actions on this spacecraft-specific FSW model."""
        raise NotImplementedError("Use ImageRSO with SpaceToSpaceImagingFSWModel.")


__doc_title__ = "Space-to-Space RSO Imaging"
__all__ = ["SpaceToSpaceImagingFSWModel"]
