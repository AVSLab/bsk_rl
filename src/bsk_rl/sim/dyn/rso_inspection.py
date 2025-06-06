"""RSO inspection dynamics models."""

import numpy as np
from Basilisk.simulation import spacecraftLocation
from Basilisk.utilities import macros

from bsk_rl.sim.dyn import BasicDynamicsModel, ContinuousImagingDynModel


class RSODynModel(BasicDynamicsModel):
    """For an RSO with points targets for observation."""

    def __init__(self, *args, **kwargs) -> None:
        """Allow for body fixed inspection points to be added to a spacecraft."""
        super().__init__(*args, **kwargs)

    def _setup_dynamics_objects(self, **kwargs) -> None:
        super()._setup_dynamics_objects(**kwargs)

        rso_dyn_proc_name = "RSODynProcess"
        self.rso_dyn_proc = self.simulator.CreateNewProcess(rso_dyn_proc_name, 1)
        self.rso_task_name = "RSODynTask"
        self.rso_dyn_proc.addTask(
            self.simulator.CreateNewTask(
                self.rso_task_name, macros.sec2nano(self.dyn_rate)
            )
        )

        self.rso_points = []

    def add_rso_point(
        self,
        r_LB_B: np.ndarray,
        aHat_B: np.ndarray,
        theta: float,
        range: float,
        theta_solar: float,
        min_shadow_factor: float,
    ):
        """Add a point on the RSO for observation.

        Args:
            r_LB_B: [m] Position of point in RSO body frame.
            aHat_B: [-] Point normal in RSO body frame.
            theta: [rad] Max angle from instrument to normal.
            range: [m] Max range to point for imaging.
            theta_solar: [rad] Maximum solar incidence angle for illumination.
            min_shadow_factor: [-] Minimum shadow factor for imaging.
        """
        rso_point_model = spacecraftLocation.SpacecraftLocation()
        rso_point_model.primaryScStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

        rso_point_model.planetInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.body_index]
        )
        rso_point_model.rEquator = self.simulator.world.planet.radEquator
        rso_point_model.rPolar = self.simulator.world.planet.radEquator * 0.98

        rso_point_model.sunInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.sun_index]
        )
        rso_point_model.eclipseInMsg.subscribeTo(
            self.world.eclipseObject.eclipseOutMsgs[self.eclipse_index]
        )

        rso_point_model.r_LB_B = r_LB_B
        rso_point_model.aHat_B = aHat_B
        rso_point_model.theta = theta
        rso_point_model.theta_solar = theta_solar
        rso_point_model.min_shadow_factor = min_shadow_factor
        rso_point_model.maximumRange = range
        self.simulator.AddModelToTask(
            self.rso_task_name, rso_point_model, ModelPriority=1
        )

        self.rso_points.append(rso_point_model)
        return rso_point_model


class RSOInspectorDynModel(ContinuousImagingDynModel):
    """For a satellite observing points on an RSO."""

    def __init__(self, *args, **kwargs) -> None:
        """Allow a satellite to observe points in a RSO."""
        super().__init__(*args, **kwargs)

    def add_rso_point(self, rso_point_model):
        """Add a point on the RSO for observation."""
        rso_point_model.addSpacecraftToModel(self.scObject.scStateOutMsg)


__doc_title__ = "RSO Inspection"
__all__ = [
    "RSODynModel",
    "RSOInspectorDynModel",
]
