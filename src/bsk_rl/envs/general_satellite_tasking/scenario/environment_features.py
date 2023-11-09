import os
import sys
from abc import ABC
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
from Basilisk.utilities import orbitalMotion


class EnvironmentFeatures(ABC):
    """Base environment feature class"""

    def reset(self) -> None:  # pragma: no cover
        """Reset environment features"""
        pass


class Target:
    def __init__(self, name: str, location: Iterable[float], priority: float) -> None:
        """Representation of a ground target
        Args:
            name: Identifier; does not need to be unique
            location: PCPF location [m]
            priority: Value metric
        """
        self.name = name
        self.location = np.array(location)
        self.priority = priority

    @property
    def id(self) -> str:
        """str: Unique human-readable identifier"""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id

    def __hash__(self) -> int:
        return hash((self.id))

    def __repr__(self) -> str:
        return f"Target({self.name})"


class StaticTargets(EnvironmentFeatures):
    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """Environment with a set number of evenly-distributed static targets.
        Args:
            n_targets: Number (or range) of targets to generate
            priority_distribution: Function for generating target priority.
            radius: Radius to place targets from body center.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius
        self.targets = []

    def reset(self) -> None:
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        self.regenerate_targets()

    def regenerate_targets(self) -> None:
        """Regenerate targets uniformly"""
        self.targets = []
        for i in range(self.n_targets):
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            self.targets.append(
                Target(
                    name=f"tgt-{i}", location=x, priority=self.priority_distribution()
                )
            )


def lla2ecef(lat: float, long: float, radius: float):
    """
    Args:
        lat: [deg]
        long: [deg]
        radius: [any]
    """
    lat = np.radians(lat)
    long = np.radians(long)
    return radius * np.array(
        [np.cos(lat) * np.cos(long), np.cos(lat) * np.sin(long), np.sin(lat)]
    )


class CityTargets(StaticTargets):
    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        n_select_from: int = sys.maxsize,
        location_offset: float = 0,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """Environment with a set number of static targets around population centers.
        Args:
            n_targets: Number of targets to generate
            n_select_from: Generate targets from the top n most populous.
            location_offset: Offset targets randomly from the city center [m].
            priority_distribution: Function for generating target priority.
            radius: Radius to place targets from body center.
        """
        super().__init__(n_targets, priority_distribution, radius)
        if n_select_from == "all":
            n_select_from = sys.maxsize
        self.n_select_from = n_select_from
        self.location_offset = location_offset

    def regenerate_targets(self) -> None:
        """Regenerate targets based on cities"""
        self.targets = []
        cities = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "simplemaps_worldcities",
                "worldcities.csv",
            )
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
                    name=city["city"].replace("'", ""),
                    location=location,
                    priority=self.priority_distribution(),
                )
            )


class UniformNadirFeature(EnvironmentFeatures):
    """
    Defines a nadir target center at the center of the planet.
    """

    def __init__(self, value_per_second: float = 1.0) -> None:
        """ "
        Args:
            value_per_second: Amount of reward per second imaging nadir.
        """
        self.name = "NadirFeature"
        self.value_per_second = value_per_second
