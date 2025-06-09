"""RSO scenarios define the geometry of a RSO.

The geometry is defined by a set of :class:`RSOPoint` objects, which give a location and
a normal vector for the point, as well as conditions for inspection (range, illumination,
etc.). Implemented geometries include:

* :class:`SphericalRSO`: Points are generated on a sphere using the Fibonacci sphere method.

This module does not consider self-shadowing effects or inspector to RSO shadowing effects.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from bsk_rl.scene.scenario import Scenario
from bsk_rl.sim.dyn import RSODynModel, RSOInspectorDynModel
from bsk_rl.sim.fsw import RSOInspectorFSWModel
from bsk_rl.utils import vizard
from bsk_rl.utils.orbital import fibonacci_sphere

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite


logger = logging.getLogger(__name__)


@dataclass
class RSOPoint:
    r_PB_B: np.ndarray
    n_B: np.ndarray
    theta_max: float
    range_max: float
    theta_solar_max: float
    min_shadow_factor: float

    def __hash__(self) -> int:
        """Hash target by unique id."""
        return hash(
            (
                self.r_PB_B[0],
                self.r_PB_B[1],
                self.r_PB_B[2],
                self.n_B[0],
                self.n_B[1],
                self.n_B[2],
                self.theta_max,
                self.range_max,
                self.theta_solar_max,
                self.min_shadow_factor,
            )
        )

    @cache
    def __str__(self) -> str:
        """String representation of the RSO point."""
        return f"RSOPoint_{self.r_PB_B}"


class RSOPoints(Scenario):
    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.rso_points = []

    def reset_pre_sim_init(self) -> None:
        """Identify RSO and Inspector satellites."""
        self.rso_points = self.generate_points()

        # Check for RSOs and inspectors
        rsos = [sat for sat in self.satellites if issubclass(sat.dyn_type, RSODynModel)]
        if len(rsos) == 0:
            logger.warning("No RSODynModel satellites found in scenario.")
            return
        assert len(rsos) == 1, "Only one RSODynModel satellite is supported."
        self.rso = rsos[0]

        self.inspectors = [
            sat
            for sat in self.satellites
            if issubclass(sat.dyn_type, RSOInspectorDynModel)
        ]
        if len(self.inspectors) == 0:
            logger.warning("No RSOInspectorDynModel satellites found in scenario.")
            return

        return super().reset_pre_sim_init()

    def reset_during_sim_init(self) -> None:
        """Add points to dynamics and fsw of RSO."""
        assert isinstance(self.rso.dynamics, RSODynModel)

        for inspector in self.inspectors:
            self.setup_inspector_camera(inspector)

        logger.debug("Adding inspection points to RSO and inspectors")
        for point in self.rso_points:
            rso_point_model = self.rso.dynamics.add_rso_point(
                point.r_PB_B,
                point.n_B,
                point.theta_max,
                point.range_max,
                point.theta_solar_max,
                point.min_shadow_factor,
            )
            # Add point to each inspector
            for inspector in self.inspectors:
                assert isinstance(inspector.dynamics, RSOInspectorDynModel)
                assert isinstance(inspector.fsw, RSOInspectorFSWModel)
                inspector.dynamics.add_rso_point(rso_point_model)

            self.visualize_rso_point(point)

        logger.debug("Targeting RSO with inspectors")
        for inspector in self.inspectors:
            inspector.fsw.set_target_rso(self.rso)

    @vizard.visualize
    def visualize_rso_point(
        self, rso_point: RSOPoint, vizSupport=None, vizInstance=None
    ):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=str(rso_point),
            parentBodyName=self.rso.name,
            r_GP_P=list(rso_point.r_PB_B),
            gHat_P=list(rso_point.n_B),
            fieldOfView=rso_point.theta_max,
            color=vizSupport.toRGBA255("gray", alpha=0.5),
            range=float(rso_point.range_max),
        )
        vizInstance.settings.showLocationCones = -1
        vizInstance.settings.showLocationCommLines = -1
        vizInstance.settings.showLocationLabels = -1

    @vizard.visualize
    def setup_inspector_camera(
        self, inspector: "Satellite", vizSupport=None, vizInstance=None
    ) -> None:
        """Visualize camera view in Vizard panel."""
        vizSupport.createStandardCamera(
            vizInstance,
            spacecraftName=inspector.name,
            setMode=0,
            bodyTarget=self.rso.name,
            setView=0,
            fieldOfView=np.radians(10),
            displayName=inspector.name,
        )

    @abstractmethod
    def generate_points(self) -> list[RSOPoint]:
        """Generate a list of RSOPoint objects based on some spacecraft geometry."""
        pass


class SphericalRSO(RSOPoints):
    def __init__(
        self,
        n_points: int = 100,
        radius: float = 1.0,
        theta_max: float = np.radians(45),
        range_max: float = -1,
        theta_solar_max: float = np.radians(60),
        min_shadow_factor: float = 0.1,
    ):
        """Generate points on a sphere using the Fibonacci sphere method.

        Args:
            n_points: Number of points to generate on the sphere.
            radius: [m] Radius of the sphere.
            theta_max: [rad] Maximum angle from the normal for inspection.
            range_max: [m] Maximum range for inspection.
            theta_solar_max: [rad] Minimum solar incidence angle for illumination.
            min_shadow_factor: Minimum shadow factor for imaging.
        """
        self.n_points = n_points
        self.radius = radius
        self.theta_max = theta_max
        self.range_max = range_max
        self.theta_solar_max = theta_solar_max
        self.min_shadow_factor = min_shadow_factor

    def generate_points(self) -> list[RSOPoint]:
        """Generate a list of RSOPoint objects on a sphere."""
        points = []

        for point in fibonacci_sphere(self.n_points):
            r_PB_B = point * self.radius
            n_B = point
            points.append(
                RSOPoint(
                    r_PB_B,
                    n_B,
                    self.theta_max,
                    self.range_max,
                    self.theta_solar_max,
                    self.min_shadow_factor,
                )
            )

        return points


__doc_title__ = "RSO Scenarios"
__all__ = ["RSOPoint", "RSOPoints", "SphericalRSO"]
