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
from avs_rl_tools.ecef2lla import ecef2lla

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class Target:
    """Ground target with associated value."""

    def __init__(self, name: str, r_LP_P: Union[np.ndarray, Callable[[float], np.ndarray]], priority: float):
        """Ground target with associated priority and location.

        Args:
            name: Identifier; does not need to be unique
            r_LP_P: Planet-fixed, planet relative location [m]
            priority: Value metric.
        """
        self.name = name
        self.priority = priority

        if callable(r_LP_P):
            self._r_LP_P_func = r_LP_P
        else:
            # Convert static vector to constant function of time
            const_pos = np.array(r_LP_P)
            self._r_LP_P_func = lambda t: const_pos

    def r_LP_P(self, t: float) -> np.ndarray:
        return self._r_LP_P_func(t)

    @property
    def id(self) -> str:
        """Get unique, human-readable identifier."""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __repr__(self) -> str:
        """Get string representation of target.

        Use ``target.id`` for a unique string identifier.

        Returns:
            Target string
        """
        return f"Target({self.name})"

class MovingTarget(Target):
    def __init__(
        self,
        name: str,
        r_LP_P: Union[Iterable[float], Callable[[float], np.ndarray]],
        priority: float,
        radius: float,
    ):
        self._initial_pos = np.array(r_LP_P).squeeze()
        assert self._initial_pos.shape == (3,), f"Initial position shape must be (3,), got {self._initial_pos.shape}"

        self.radius = radius
        self.speed = np.random.uniform(880, 1120) * 1e3 /3600  # [m/s]
        self.bearing_angle = np.random.uniform(0, 2 * np.pi)  # [rad]

        # Convert to initial geodetic coordinates once
        self.lat0, self.lon0, self.alt0 = ecef2lla(*self._initial_pos)

        def position_func(t: float) -> np.ndarray:
            d = self.speed * t  # Distance traveled [m]
            delta_lat = np.degrees(np.cos(self.bearing_angle) * d / self.radius)
            delta_lon = np.degrees(np.sin(self.bearing_angle) * d / (self.radius * np.cos(np.radians(self.lat0))))

            lat = self.lat0 + delta_lat
            lon = self.lon0 + delta_lon

            position = (
                np.array([
                    np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                    np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                    np.sin(np.radians(lat)),
                ]) * self.radius
            )
            return position.flatten()

        super().__init__(name=name, r_LP_P=position_func, priority=priority)
class UniformTargets(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        use_moving_targets: bool = False,
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
        self.use_moving_targets = use_moving_targets

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
            r_GP_P=list(target.r_LP_P(0.0)),
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
            priority = self.priority_distribution()
            if self.use_moving_targets:
                self.targets.append(
                    MovingTarget(name=f"tgt-{i}", r_LP_P=x, priority=priority, radius=self.radius)
                )
            else:
                self.targets.append(
                    Target(name=f"tgt-{i}", r_LP_P=x, priority=priority)
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
        for population center locations.

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


__doc_title__ = "Target Scenarios"
__all__ = ["Target", "UniformTargets", "CityTargets"]
