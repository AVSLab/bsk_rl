"""FSW models for RSO inspection."""

from typing import TYPE_CHECKING, Iterable

from Basilisk.architecture import messaging

from bsk_rl.sim.fsw import ContinuousImagingFSWModel, action
from bsk_rl.utils.functional import default_args

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite


class RSOInspectorFSWModel(ContinuousImagingFSWModel):
    def set_target_rso(self, rso: "Satellite") -> None:
        """Set the RSO to point at."""
        self.locPoint.scTargetInMsg.subscribeTo(
            rso.dynamics.simpleNavObject.transOutMsg
        )

    class LocPointTask(ContinuousImagingFSWModel.LocPointTask):
        """Task to point at the RSO and trigger the instrument."""

        def __init__(self, *args, **kwargs) -> None:
            """Task to point at the RSO and trigger the instrument."""
            super().__init__(*args, **kwargs)

        @default_args(inst_pHat_B=[0, 0, 1])
        def setup_location_pointing(
            self, inst_pHat_B: Iterable[float], **kwargs
        ) -> None:
            """Set the location pointing guidance module to point at the RSO.

            The function ``set_target_rso`` must be called externally to connect the RSO
            to the pointing module.

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
            # Only use this module to check for pointing requirements
            self.access_msg = messaging.AccessMsg()
            payload = messaging.AccessMsgPayload()
            payload.hasAccess = 1
            self.access_msg.write(payload)
            self.insControl.accessInMsg.subscribeTo(self.access_msg)

            self._add_model_to_task(self.insControl, priority=987)

    @action
    def action_inspect_rso(self) -> None:
        """Action to inspect the RSO."""
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.instrument.nodeDataName = "inspect_rso"
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)


__doc_title__ = "RSO Inspection"
__all__ = ["RSOInspectorFSWModel"]
