"""Target scenarios distribute ground targets with some distribution.

Currently, targets are all known to the satellites a priori and are available based on
the imaging requirements given by the dynamics and flight software models.
"""

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
from Basilisk.utilities import orbitalMotion

from bsk_rl.scene import Scenario
from bsk_rl.utils import vizard
from bsk_rl.utils.orbital import lla2ecef

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


import random


class Target:
    """Ground target with associated value."""

    def __init__(self, name: str, r_LP_P: Iterable[float], priority: float) -> None:
        """Ground target with associated priority and location.

        Args:
            name: Identifier; does not need to be unique
            r_LP_P: Planet-fixed, planet relative location [m]
            priority: Value metric.
        """
        self.name = name
        self.r_LP_P = np.array(r_LP_P)
        self.priority = priority

    @property
    def id(self) -> str:
        """Get unique, human-readable identifier."""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __hash__(self) -> int:
        """Hash target by unique id."""
        return hash((self.id))

    def __repr__(self) -> str:
        """Get string representation of target.

        Use ``target.id`` for a unique string identifier.

        Returns:
            Target string
        """
        return f"Target({self.name})"

class Strip:
    """Ground strip target with associated value."""
    def __init__(self, name: str, r_LP_P_start: Iterable[float], r_LP_P_end: Iterable[float], priority: float,aquisition_speed: float, pre_imaging_time: list[float]) -> None:
        """Strip target with name, associated priority, location of the starting point, location of the end point

        Args:
            name: Identifier; does not need to be unique
            r_LP_P_start: Starting point of the strip in the planet-fixed frame [m]
            r_LP_P_end: Ending point of the strip in the planet-fixed frame [m]
            priority: Value metric.
        """
        self.name = name
        self.r_LP_P_start = np.array(r_LP_P_start)
        self.r_LP_P_end = np.array(r_LP_P_end)
        self.priority = priority
        self.aquisition_speed = aquisition_speed
        self.pre_imaging_time = pre_imaging_time
    
    @property 
    def id(self) -> str:
        """Get unique, human-readable identifier."""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id
    
    def __hash__(self) -> int:
        """Hash strip by unique id."""
        return hash((self.id))
    
    def __repr__(self) -> str:
        """Get string representation of target.

        Use ``target.id`` for a unique string identifier.

        Returns:
        Target string
        """
        return f"Target({self.name})"

class UniformTargets(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def reset_during_sim_init(self) -> None:
        """Visualize targets in Vizard on reset."""
        for target in self.targets:
            self.visualize_target(target)

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        """Visualize target in Vizard."""
        vizSupport.addLocation(
            vizInstance,
            stationName=target.name,
            parentBodyName="earth",
            r_GP_P=list(target.r_LP_P),
            fieldOfView=np.arctan(500 / 800),
            color=vizSupport.toRGBA255("white"),
            range=1000.0 * 1000,  # meters
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1
        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = -1
        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1

    def regenerate_targets(self) -> None:
        """Regenerate targets uniformly.

        Override this method (as demonstrated in :class:`CityTargets`) to generate
        other distributions.
        """
        self.targets = []
        for i in range(self.n_targets):
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            self.targets.append(
                Target(name=f"tgt-{i}", r_LP_P=x, priority=self.priority_distribution())
            )


class CityTargets(UniformTargets):
    """Environment with targets distributed around population centers."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        n_select_from: Optional[int] = None,
        location_offset: float = 0,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """Construct environment with static targets around population centers.

        Uses the `simplemaps Word Cities Database <https://simplemaps.com/data/world-cities>`_
        for population center locations. This data is installed by ``finish_install``.

        Args:
            n_targets: Number of targets to generate, as a fixed number or a range.
            n_select_from: Generate targets from the top `n_select_from` most populous
                cities. Will use all cities in the database if not specified.
            location_offset: [m] Offset targets randomly from the city center by up to
                this amount.
            priority_distribution: Function for generating target priority.
            radius: Radius to place targets from body center.
        """
        super().__init__(n_targets, priority_distribution, radius)
        if n_select_from == "all" or n_select_from is None:
            n_select_from = sys.maxsize
        self.n_select_from = n_select_from
        self.location_offset = location_offset

    def regenerate_targets(self) -> None:
        """Regenerate targets based on cities.

        :meta private:
        """
        self.targets = []
        cities = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "simplemaps_worldcities"
            / "worldcities.csv",
        )

        if self.n_select_from > len(cities):
            self.n_select_from = len(cities)

        for i in np.random.choice(self.n_select_from, self.n_targets, replace=False):
            city = cities.iloc[i]
            location = lla2ecef(city["lat"], city["lng"], self.radius)
            offset = np.random.normal(size=3)
            offset /= np.linalg.norm(offset)
            offset *= self.location_offset
            location += offset
            location /= np.linalg.norm(location)
            location *= self.radius
            self.targets.append(
                Target(
                    name=f"{city['city']}, {city['iso2']}".replace("'", ""),
                    r_LP_P=location,
                    priority=self.priority_distribution(),
                )
            )

class UniformStripTargets(Scenario):
    """Environment with strip targets distributed uniformly."""

    def __init__(
            self,
            n_targets: Union[int, tuple[int, int]],
            length_strip_bounds: tuple[float, float],
            priority_distribution: Optional[Callable] = None,
            radius: float = orbitalMotion.REQ_EARTH * 1e3,
            aquisition_speed: float = 3*1e3, #in m/s
            pre_imaging_time: list[float] = [0,1,2] #in seconds
        ) -> None:
            """An environment with evenly-distributed static targets.

            Args:
                n_targets: Number of targets to generate. Can also be specified as a range
                    ``(low, high)`` where the number of targets generated is uniformly selected
                    ``low ≤ n_targets ≤ high``.
                length_strip_bounds: Tuple containing the lower and upper bounds for the length of the strips. The lenght of the strips will be uniformly distributed between these bounds.
                priority_distribution: Function for generating target priority. Defaults
                    to ``lambda: uniform(0, 1)`` if not specified.
                radius: [m] Radius to place targets from body center. Defaults to Earth's
                    equatorial radius.
            """
            self._n_targets = n_targets
            if priority_distribution is None:
                priority_distribution = lambda: np.random.rand() 
            self.priority_distribution = priority_distribution
            self.radius = radius
            self.length_strip_bounds = length_strip_bounds
            self.aquisition_speed = aquisition_speed
            self.pre_imaging_time = pre_imaging_time

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []
    
    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_strip_for_access_checking"):
                for target in self.targets:
                    satellite.add_strip_for_access_checking(
                        object=target,
                        r_LP_P_start=target.r_LP_P_start,
                        r_LP_P_end=target.r_LP_P_end,
                        acquisition_speed=target.aquisition_speed,
                        pre_imaging_time=target.pre_imaging_time,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def regenerate_targets(self) -> None:
        """Regenerate target strips uniformly.

        Override this method (as demonstrated in :class:`CityTargets`) to generate
        other distributions.
        """
        self.targets = []
        for i in range(self.n_targets):
            #Generate random starting point
            x_start = np.random.normal(size=3)
            x_start *= self.radius / np.linalg.norm(x_start)
            #Generate random lengh of the strip between the bounds
            length_strip = np.random.uniform(self.length_strip_bounds[0], self.length_strip_bounds[1])
            #Generate a random angle for the direction of the strip
            random_azimuth = np.random.uniform(0, 2 * np.pi)
            # Calculate the end point using spherical trigonometry
            # Convert the start point to spherical coordinates
            lat_start = np.arcsin(x_start[2] / self.radius)
            lon_start = np.arctan2(x_start[1], x_start[0])
            # Calculate the end point's latitude and longitude
            delta_sigma = length_strip / self.radius  # Angular distance
            lat_end = np.arcsin(np.sin(lat_start) * np.cos(delta_sigma) +
                                np.cos(lat_start) * np.sin(delta_sigma) * np.cos(random_azimuth))
            lon_end = lon_start + np.arctan2(np.sin(random_azimuth) * np.sin(delta_sigma) * np.cos(lat_start),
                                            np.cos(delta_sigma) - np.sin(lat_start) * np.sin(lat_end))
            # Convert the end point back to Cartesian coordinates
            x_end = self.radius * np.cos(lat_end) * np.cos(lon_end)
            y_end = self.radius * np.cos(lat_end) * np.sin(lon_end)
            z_end = self.radius * np.sin(lat_end)
            end_point = np.array([x_end, y_end, z_end])
            # Create the strip target
            self.targets.append(
                Strip(name=f"strip-{i}", r_LP_P_start=x_start, r_LP_P_end=end_point, priority=random.choice([0.1, 0.5, 1]), aquisition_speed=self.aquisition_speed, pre_imaging_time=self.pre_imaging_time)
            )



__doc_title__ = "Target Scenarios"
__all__ = ["Target", "Strip", "UniformTargets", "CityTargets","UniformStripTargets"]
