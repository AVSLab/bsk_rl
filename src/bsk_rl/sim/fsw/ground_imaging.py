"""FSW models for ground imaging scenarios."""

from typing import TYPE_CHECKING, Iterable

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (
    locationPointing,
    scanningInstrumentController,
    simpleInstrumentController,
)

from bsk_rl.sim import dyn
from bsk_rl.sim.fsw import BasicFSWModel, SteeringFSWModel, Task, action
from bsk_rl.utils import vizard
from bsk_rl.utils.functional import default_args
from bsk_rl.utils.orbital import rv2HN

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sim.dyn import DynamicsModel


class ImagingFSWModel(BasicFSWModel):
    """Extend FSW with instrument pointing and triggering control."""

    @classmethod
    def _requires_dyn(cls) -> list[type["DynamicsModel"]]:
        return super()._requires_dyn() + [dyn.ImagingDynModel]

    def __init__(self, *args, **kwargs) -> None:
        """Adds instrument pointing and triggering control to FSW."""
        super().__init__(*args, **kwargs)

    @property
    def c_hat_P(self):
        """Instrument pointing direction in the planet frame."""
        c_hat_B = self.locPoint.pHat_B
        return np.matmul(self.dynamics.BP.T, c_hat_B)

    @property
    def c_hat_H(self):
        """Instrument pointing direction in the hill frame."""
        c_hat_B = self.locPoint.pHat_B
        HN = rv2HN(self.satellite.dynamics.r_BN_N, self.satellite.dynamics.v_BN_N)
        return HN @ self.satellite.dynamics.BN.T @ c_hat_B

    def _make_task_list(self) -> list[Task]:
        return super()._make_task_list() + [self.LocPointTask(self)]

    def _set_gateway_msgs(self) -> None:
        super()._set_gateway_msgs()
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )

    class LocPointTask(Task):
        """Task to point at targets and trigger the instrument."""

        name = "locPointTask"

        def __init__(self, fsw, priority=96) -> None:  # noqa: D107
            """Task to point the instrument at ground targets."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            # Location pointing configuration
            self.locPoint = self.fsw.locPoint = locationPointing.locationPointing()
            self.locPoint.ModelTag = "locPoint"

            # SimpleInstrumentController configuration
            self.insControl = self.fsw.insControl = (
                simpleInstrumentController.simpleInstrumentController()
            )
            self.insControl.ModelTag = "instrumentController"

        def _setup_fsw_objects(self, **kwargs) -> None:
            self.setup_location_pointing(**kwargs)
            self.setup_instrument_controller(**kwargs)
            self.show_sensor()

        @default_args(inst_pHat_B=[0, 0, 1])
        def setup_location_pointing(
            self, inst_pHat_B: Iterable[float], **kwargs
        ) -> None:
            """Set the Earth location pointing guidance module.

            Args:
                inst_pHat_B: Instrument pointing direction.
                kwargs: Passed to other setup functions.
            """
            self.locPoint.pHat_B = inst_pHat_B
            self.locPoint.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.locPoint.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.locPoint.locationInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.currentGroundStateOutMsg
            )
            self.locPoint.useBoresightRateDamping = 1
            messaging.AttGuidMsg_C_addAuthor(
                self.locPoint.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.locPoint, priority=1198)

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def setup_instrument_controller(
            self,
            imageAttErrorRequirement: float,
            imageRateErrorRequirement: float,
            **kwargs,
        ) -> None:
            """Set the instrument controller parameters.

            The instrument controller is used to take an image when certain relative
            attitude requirements are met, along with the access requirements of the
            target (i.e. ``imageTargetMinimumElevation`` and ``imageTargetMaximumRange``
            as set in :class:`~bsk_rl.sim.dyn.ImagingDynModel.setup_imaging_target`).

            Args:
                imageAttErrorRequirement: [MRP norm] Pointing attitude error tolerance
                    for imaging.
                imageRateErrorRequirement: [rad/s] Rate tolerance for imaging. Disable
                    with ``None``.
                kwargs: Passed to other setup functions.
            """
            self.insControl.attErrTolerance = imageAttErrorRequirement
            if imageRateErrorRequirement is not None:
                self.insControl.useRateTolerance = 1
                self.insControl.rateErrTolerance = imageRateErrorRequirement
            self.insControl.attGuidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.insControl.locationAccessInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.accessOutMsgs[-1]
            )

            self._add_model_to_task(self.insControl, priority=987)

        @vizard.visualize
        def show_sensor(self, vizInterface=None, vizSupport=None):
            """Visualize the sensor in Vizard."""
            genericSensor = vizInterface.GenericSensor()
            genericSensor.normalVector = self.locPoint.pHat_B
            genericSensor.r_SB_B = [0.0, 0.0, 0.0]
            genericSensor.fieldOfView.push_back(4 * self.insControl.attErrTolerance)
            genericSensor.color = vizInterface.IntVector(
                vizSupport.toRGBA255(self.fsw.satellite.vizard_color, alpha=0.5)
            )
            cmdInMsg = messaging.DeviceCmdMsgReader()
            cmdInMsg.subscribeTo(self.insControl.deviceCmdOutMsg)
            genericSensor.genericSensorCmdInMsg = cmdInMsg
            self.fsw.satellite.vizard_data["genericSensorList"] = [genericSensor]

        def reset_for_action(self) -> None:
            """Reset pointing controller."""
            self.fsw.dynamics.imagingTarget.Reset(self.fsw.simulator.sim_time_ns)
            self.locPoint.Reset(self.fsw.simulator.sim_time_ns)
            self.insControl.controllerStatus = 0
            return super().reset_for_action()

    @action
    def action_image(self, r_LP_P: Iterable[float], data_name: str) -> None:
        """Attempt to image a target at a location.

        This action sets the target attitude to one tracking a ground location. If the
        target is within the imaging constraints, an image will be taken and stored in
        the data buffer. The instrument power sink will be active as long as the task is
        enabled.

        Args:
            r_LP_P: [m] Planet-fixed planet relative target location.
            data_name: Data buffer to store image data to.
        """
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = r_LP_P
        self.dynamics.instrument.nodeDataName = data_name
        self.insControl.imaged = 0
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)

    @action
    def action_downlink(self) -> None:
        """Attempt to downlink data.

        This action points the satellite nadir and attempts to downlink data. If the
        satellite is in range of a ground station, data will be downlinked at the specified
        baud rate. The transmitter power sink will be active as long as the task is enabled.
        """
        self.hillPoint.Reset(self.simulator.sim_time_ns)
        self.trackingError.Reset(self.simulator.sim_time_ns)
        self.dynamics.transmitter.dataStatus = 1
        self.dynamics.transmitterPowerSink.powerStatus = 1
        self.simulator.enableTask(
            BasicFSWModel.NadirPointTask.name + self.satellite.name
        )
        self.simulator.enableTask(
            BasicFSWModel.TrackingErrorTask.name + self.satellite.name
        )


class ContinuousImagingFSWModel(ImagingFSWModel):
    """FSW model for continuous nadir scanning."""

    def __init__(self, *args, **kwargs) -> None:
        """FSW model for continuous nadir scanning.

        Instead of imaging point targets, this model is used to continuously scan the
        ground while pointing nadir.
        """
        super().__init__(*args, **kwargs)

    class LocPointTask(ImagingFSWModel.LocPointTask):
        """Task to point nadir and trigger the instrument."""

        def __init__(self, *args, **kwargs) -> None:
            """Task to point nadir and trigger the instrument."""
            super().__init__(*args, **kwargs)

        def _create_module_data(self) -> None:
            # Location pointing configuration
            self.locPoint = self.fsw.locPoint = locationPointing.locationPointing()
            self.locPoint.ModelTag = "locPoint"

            # scanningInstrumentController configuration
            self.insControl = self.fsw.insControl = (
                scanningInstrumentController.scanningInstrumentController()
            )
            self.insControl.ModelTag = "instrumentController"

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def setup_instrument_controller(
            self,
            imageAttErrorRequirement: float,
            imageRateErrorRequirement: float,
            **kwargs,
        ) -> None:
            """Set the instrument controller parameters for scanning.

            As long as these two conditions are met, scanning will occur continuously.

            Args:
                imageAttErrorRequirement: [MRP norm] Pointing attitude error tolerance
                    for imaging.
                imageRateErrorRequirement: [rad/s] Rate tolerance for imaging. Disable
                    with None.
                kwargs: Passed to other setup functions.
            """
            self.insControl.attErrTolerance = imageAttErrorRequirement
            if imageRateErrorRequirement is not None:
                self.insControl.useRateTolerance = 1
                self.insControl.rateErrTolerance = imageRateErrorRequirement
            self.insControl.attGuidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.insControl.accessInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.accessOutMsgs[-1]
            )

            self._add_model_to_task(self.insControl, priority=987)

        def reset_for_action(self) -> None:
            """Reset scanning controller."""
            self.instMsg = messaging.DeviceCmdMsg_C()
            self.instMsg.write(messaging.DeviceCmdMsgPayload())
            self.fsw.dynamics.instrument.nodeStatusInMsg.subscribeTo(self.instMsg)
            return super().reset_for_action()

    @action
    def action_nadir_scan(self) -> None:
        """Scan nadir.

        This action points the instrument nadir and continuously adds data to the buffer
        as long as attitude requirements are met. The instrument power sink is active
        as long as the action is set.
        """
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = np.array(
            [0, 0, 0.1]
        )  # All zero causes an error
        self.dynamics.instrument.nodeDataName = "nadir"
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)

    @action
    def action_image(self, *args, **kwargs) -> None:
        """Disable ``action_image`` from parent class.

        :meta private:
        """
        raise NotImplementedError("Use action_nadir_scan instead")


class SteeringImagerFSWModel(SteeringFSWModel, ImagingFSWModel):
    """Convenience type for ImagingFSWModel with MRP steering."""

    def __init__(self, *args, **kwargs) -> None:
        """Convenience type that combines the imaging FSW model with MRP steering."""
        super().__init__(*args, **kwargs)


__doc_title__ = "Ground Imaging"
__all__ = [
    "ImagingFSWModel",
    "ContinuousImagingFSWModel",
    "SteeringImagerFSWModel",
]
