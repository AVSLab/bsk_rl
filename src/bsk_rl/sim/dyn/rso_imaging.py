"""Dynamics for imaging independent spacecraft, rather than RSO surface points."""

import numpy as np
from Basilisk.simulation import spacecraftLocation

from bsk_rl.sim.dyn.base import BasicDynamicsModel
from bsk_rl.sim.dyn.ground_imaging import ImagingDynModel
from bsk_rl.utils.functional import default_args


class RSOTargetDynModel(BasicDynamicsModel):
    """Passive target retaining AMOS gravity, drag, attitude, and resource dynamics.

    Targets run before imagers so access does not depend on satellite list order.
    Use ordinary ``Satellite`` and ``FSWModel`` with ``Drift``. The catalog fixes
    wheel speeds and passive power draw; no artificial failure makes targets passive.
    """

    def __init__(self, satellite, dyn_rate, priority=250, **kwargs) -> None:
        """Propagate after world ephemerides (300), before observer dynamics (200)."""
        super().__init__(satellite, dyn_rate, priority=priority, **kwargs)


class SpaceToSpaceImagingDynModel(ImagingDynModel):
    """Imager with existing instrument, storage, transmitter, and RSO access.

    The Earth ellipsoid matches the AMOS and existing RSO inspection approximation:
    polar radius is 0.98 times equatorial radius. Range defaults to unlimited.
    Target access uses spacecraft states. Ground stations are optional: compose
    this model with :class:`~bsk_rl.sim.dyn.GroundStationDynModel` to enable
    downlink through :class:`~bsk_rl.sim.world.GroundStationWorldModel`.
    That component connects the spacecraft's transmitter to station access
    messages; the ground stations themselves belong to the world model.
    """

    @default_args(instrumentBaudRate=8e6)
    def setup_instrument(self, instrumentBaudRate, **kwargs) -> None:
        """Use the existing one-tick instrument; its argument denotes image bits."""
        if not np.isfinite(instrumentBaudRate) or instrumentBaudRate <= 0:
            raise ValueError("RSO image size must be finite and positive.")
        super().setup_instrument(instrumentBaudRate=instrumentBaudRate, **kwargs)

    @default_args(
        transmitterBaudRate=-8e6,
        transmitterNumBuffers=100,
        transmitterPacketSize=-1,
    )
    def setup_transmitter(self, transmitterPacketSize, **kwargs) -> None:
        """Allow a partially transmitted image to resume across ground contacts.

        The existing transmitter otherwise requires a full packet to restart after
        contact loss. One-bit packets allow partial buffers to drain; the image ledger,
        rather than transmitter packet size, defines complete image delivery.
        """
        super().setup_transmitter(transmitterPacketSize=transmitterPacketSize, **kwargs)

    @default_args(imageTargetMaximumRange=-1)
    def setup_imaging_target(
        self, imageTargetMaximumRange: float, priority=1900, **kwargs
    ) -> None:
        """Create per-imager access; explicitly honor an optional range [m]."""
        if imageTargetMaximumRange != -1 and (
            not np.isfinite(imageTargetMaximumRange) or imageTargetMaximumRange <= 0
        ):
            raise ValueError("RSO imaging range must be -1 (unlimited) or positive.")
        self.targetLocation = spacecraftLocation.SpacecraftLocation()
        self.targetLocation.ModelTag = "rsoAccess" + self.satellite.name
        self.targetLocation.primaryScStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.targetLocation.planetInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.body_index]
        )
        self.targetLocation.rEquator = self.world.planet.radEquator
        self.targetLocation.rPolar = self.world.planet.radEquator * 0.98
        self.targetLocation.maximumRange = imageTargetMaximumRange
        self.rso_access_messages = {}
        self.rso_partition_names = {}
        self.simulator.AddModelToTask(
            self.task_name, self.targetLocation, ModelPriority=priority
        )

    def bind_rso_target(self, target) -> None:
        """Register a target and retain its actual access output, independent of ID."""
        if target.id in self.rso_access_messages:
            raise ValueError(f"Duplicate access binding for {target.id!r}.")
        self.targetLocation.addSpacecraftToModel(
            target.target_spacecraft.dynamics.scObject.scStateOutMsg
        )
        self.rso_access_messages[target.id] = self.targetLocation.accessOutMsgs[-1]
        self.rso_partition_names[target.id] = (
            self.satellite.rso_scenario.partition_name(target.id)
        )


__doc_title__ = "Space-to-Space RSO Imaging"
__all__ = ["RSOTargetDynModel", "SpaceToSpaceImagingDynModel"]
