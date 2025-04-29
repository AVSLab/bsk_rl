"""TODO: Add docstring."""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from bsk_rl.scene import Scenario
from bsk_rl.sim.dyn import RSODynModel, RSOImagingDynModel
from bsk_rl.sim.fsw import RSOImagingFSWModel
from bsk_rl.utils import vizard

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


@dataclass
class RSOPoint:
    r_PB_B: np.ndarray
    n_B: np.ndarray
    theta_min: float
    range: float
    solar_incidence_min: float
    illumination_eclipse_min: float

    def __hash__(self) -> int:
        """Hash target by unique id."""
        return hash(id(self))  # THIS IS ALMOST CERTAINLY A BAD IDEA

    @cache
    def __str__(self) -> str:
        return f"RSOPoint_{self.r_PB_B}"


class RSOPoints(Scenario):
    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.rso_points = []

    def reset_pre_sim_init(self) -> None:
        self.rso_points = self.generate_points()

        # Check for RSOs and observers
        rsos = [sat for sat in self.satellites if issubclass(sat.dyn_type, RSODynModel)]
        if len(rsos) == 0:
            logger.warning("No RSODynModel satellites found in scenario.")
            return
        assert len(rsos) == 1, "Only one RSODynModel satellite is supported."
        self.rso = rsos[0]

        self.observers = [
            sat
            for sat in self.satellites
            if issubclass(sat.dyn_type, RSOImagingDynModel)
        ]
        if len(self.observers) == 0:
            logger.warning("No RSOImagingDynModel satellites found in scenario.")
            return

        return super().reset_pre_sim_init()

    def reset_during_sim_init(self) -> None:
        # Add points to dynamics and fsw of RSO
        assert isinstance(self.rso.dynamics, RSODynModel)

        for observer in self.observers:
            self.setup_inspector_camera(observer)

        logger.debug("Adding inspection points to RSO and observers")
        for point in self.rso_points:
            rso_point_model = self.rso.dynamics.add_rso_point(
                point.r_PB_B,
                point.n_B,
                point.theta_min,
                point.range,
                point.solar_incidence_min,
                point.illumination_eclipse_min,
            )
            # Add point to each observer
            for observer in self.observers:
                assert isinstance(observer.dynamics, RSOImagingDynModel)
                assert isinstance(observer.fsw, RSOImagingFSWModel)
                observer.dynamics.add_rso_point(rso_point_model)

            self.visualize_rso_point(point)

        logger.debug("Targeting RSO with observers")
        for observer in self.observers:
            observer.fsw.set_target_rso(self.rso)

    @vizard.visualize
    def visualize_rso_point(self, rso_point, vizSupport=None, vizInstance=None):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=str(rso_point),
            parentBodyName=self.rso.name,
            r_GP_P=list(rso_point.r_PB_B),
            gHat_P=list(rso_point.n_B),
            fieldOfView=rso_point.theta_min,
            color=vizSupport.toRGBA255("gray", alpha=0.5),
            range=float(rso_point.range),
        )
        vizInstance.settings.showLocationCones = -1
        vizInstance.settings.showLocationCommLines = -1
        vizInstance.settings.showLocationLabels = -1

    @vizard.visualize
    def setup_inspector_camera(
        self, observer, vizSupport=None, vizInstance=None
    ) -> None:
        """Visualize camera in Vizard."""
        vizSupport.createStandardCamera(
            vizInstance,
            spacecraftName=observer.name,
            setMode=0,
            bodyTarget=self.rso.name,
            setView=0,
            fieldOfView=np.radians(10),
            displayName=observer.name,
        )

    @abstractmethod
    def generate_points(self) -> list[RSOPoint]:
        pass


class FibonacciSphereRSOPoints(RSOPoints):
    def __init__(
        self,
        n_points: int = 100,
        radius: float = 1.0,
        theta_min: float = np.radians(45),  # inspection_angle_max
        range: float = -1,  # inspection_range_min
        solar_incidence_min: float = np.radians(60),  # solar_incidence_max
        illumination_eclipse_min: float = 0.1,
    ):
        self.n_points = n_points
        self.radius = radius
        self.theta_min = theta_min
        self.range = range
        self.solar_incidence_min = solar_incidence_min
        self.illumination_eclipse_min = illumination_eclipse_min

    def generate_points(self) -> list[RSOPoint]:
        points = []

        # https://gist.github.com/Seanmatthews/a51ac697db1a4f58a6bca7996d75f68c
        ga = (3 - np.sqrt(5)) * np.pi  # golden angle
        theta = ga * np.arange(self.n_points)
        z = np.linspace(1 / self.n_points - 1, 1 - 1 / self.n_points, self.n_points)
        radius = np.sqrt(1 - z * z)
        y = radius * np.sin(theta)
        x = radius * np.cos(theta)

        for i in range(self.n_points):
            r_PB_B = np.array([x[i], y[i], z[i]]) * self.radius
            n_B = np.array([x[i], y[i], z[i]])
            points.append(
                RSOPoint(
                    r_PB_B,
                    n_B,
                    self.theta_min,
                    self.range,
                    self.solar_incidence_min,
                    self.illumination_eclipse_min,
                )
            )

        return points


__doc_title__ = "RSO Scenarios"
__all__ = ["RSOPoints"]
