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

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.cm import get_cmap
from matplotlib.colors import LightSource
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
    def __init__(self, name: str, r_LP_P_start: Iterable[float], r_LP_P_end: Iterable[float], priority: float, aquisition_speed: float, pre_imaging_time: float) -> None:
        """Strip target with name, associated priority, location of the starting point, location of the end point

        Args:
            name: Identifier does not need to be unique
            r_LP_P_start: Starting point of the strip in the planet-fixed frame [m]
            r_LP_P_end: Ending point of the strip in the planet-fixed frame [m]
            priority: Value metric.
            acquisition_speed: Speed at which the satellite can acquire the strip [m/s]
            pre_imaging_time: Time required before imaging can start [s]
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
            self.pre_imaging_time = 0
           
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
    def reset_during_sim_init(self) -> None:
        """Visualize targets in Vizard on reset."""
        for target in self.targets:
            self.visualize_target(target)

    @vizard.visualize
    def visualize_target(self, target, vizSupport=None, vizInstance=None):
        
        def compute_strip_vertices(r_start, r_end, width=100e3):
            r_start = np.array(r_start).flatten()
            r_end = np.array(r_end).flatten()

            # Direction along the centerline
            d_vec = r_end - r_start
            d_vec /= np.linalg.norm(d_vec)

            # Approximate surface normal at the midpoint
            mid_vec = (r_start + r_end) / 2
            mid_vec /= np.linalg.norm(mid_vec)

            # Perpendicular vector in the tangent plane
            p_vec = np.cross(mid_vec, d_vec)
            p_vec /= np.linalg.norm(p_vec)

            # Compute the four corner points
            half_width = width / 2.0
            v1 = r_start + half_width * p_vec  # top-left
            v2 = r_start - half_width * p_vec  # bottom-left
            v3 = r_end + half_width * p_vec    # top-right
            v4 = r_end - half_width * p_vec    # bottom-right

            vertices = list(v1) + list(v3) + list(v4) + list(v2)
            return vertices
        
        def get_priority_color(priority):
            priority = np.clip(priority, 0.0, 1.0)
            r = 255
            g = int(255 * (1 - priority))  # fades from white to red
            b = int(255 * (1 - priority))
            a = 100
            return [255, 255, 255, a]  
        
        vizSupport.addQuadMap(vizInstance,
                              ID=int(target.name[6:]),
                              parentBodyName="earth",
                              vertices=compute_strip_vertices(target.r_LP_P_start, target.r_LP_P_end, width=100e3),
                              color=get_priority_color(target.priority)
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
                Strip(name=f"strip-{i}", r_LP_P_start=x_start, r_LP_P_end=end_point, priority=self.priority_distribution(), aquisition_speed=self.aquisition_speed,pre_imaging_time=self.pre_imaging_time)
            )


def _draw(spec) -> float:
    """A scalar spec is a fixed value, a ``(low, high)`` spec a uniform draw."""
    if np.isscalar(spec):
        return float(spec)
    return float(np.random.uniform(spec[0], spec[1]))


def _draw_int(spec) -> int:
    """Integer version of :func:`_draw`; ``(low, high)`` is inclusive."""
    if np.isscalar(spec):
        return int(spec)
    return int(np.random.randint(spec[0], spec[1] + 1))


class MixedStripTargets(UniformStripTargets):
    """Strip targets drawn from a randomized mixture of isolated and clustered strips.

    Domain randomization over the request geography. Each episode first draws
    a *mode*:

    * ``isolated``: every strip stands alone, as in :class:`UniformStripTargets`.
    * ``clusters``: every strip belongs to a cluster.
    * ``mixed``: a fraction ``f_clust`` of the strips belongs to clusters, the
      rest stand alone.

    A cluster is a group of strips with a geometric relation that a planner
    can exploit, of one of two kinds:

    * ``area``: a large area cut into parallel strips side by side, one swath
      apart (the decomposition of the paper's concept figure).
    * ``chain``: a long curved feature (river, coastline, border) cut into
      strips laid end to end. The heading turns by a von Mises angle at each
      joint, so the concentration ``kappa`` sets how straight it runs.

    A dense group of strips with random positions and directions is *not* a
    cluster here: it carries no structure beyond a higher local density, and
    the number of strips and ``f_land`` already vary the density.

    Each episode also chooses how strips are placed: uniformly over the
    globe with probability ``uniform_prob`` (the paper's scenario, kept in
    the mix), otherwise with a drawn ``f_land``, the probability that an
    isolated strip midpoint or a cluster anchor falls on land. By default it is the only
    continuous draw: ``f_clust`` (0.5 in ``mixed``), ``f_chain`` (0.5, one
    coin flip per cluster) and ``kappa`` (8, turns of about 20 deg) are
    fixed, because the policy sees only the next few dozen strips, and the
    local mix of kinds already varies with the cluster positions whatever
    the episode-wide shares are. They still accept ranges for ablations. Land is read from an Earth texture with the same colour test as
    :class:`~bsk_rl.scene.risk_field.SweepRiskField`. Antarctica (land south
    of 60 S) is neither land nor sea, so no midpoint or anchor falls there:
    as sea it would put "ocean" strips on the ice sheet. Only the anchor is
    tested: a land strip can run past the coast, and
    a land chain retries its turns to stay on land but gives up after
    ``chain_land_retries`` draws rather than bias the length distribution.

    Isolated strips are placed by their midpoint rather than by their start
    point. Under a uniform law the two are the same distribution, and the
    midpoint is what the land test should see. With ``mode_probs=(1, 0, 0)``
    and no Earth map the scenario is therefore the uniform one of
    :class:`UniformStripTargets` (with a different random stream).

    The episode draw is kept in ``scene_params`` and the kind (0 isolated,
    1 area, 2 chain) and cluster index (-1 if isolated) of every strip in
    ``strip_kind`` / ``strip_cluster``, for metrics and figures. Every range
    argument also accepts a scalar, which fixes that value: this is how the
    fixed evaluation archetypes are built from the same class.
    """

    MODES = ("isolated", "clusters", "mixed")
    KINDS = ("isolated", "area", "chain")

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        length_strip_bounds: tuple[float, float],
        earth_map: Optional[str] = None,
        mode_probs: Iterable[float] = (1 / 3, 1 / 3, 1 / 3),
        uniform_prob: float = 0.5,
        land_fraction=(0.0, 1.0),
        mixed_cluster_fraction=0.5,
        chain_fraction=0.5,
        area_strips=(3, 10),
        chain_strips=(3, 10),
        swath_width: float = 20e3,
        chain_kappa=8.0,
        chain_gap: float = 0.0,
        chain_land_retries: int = 8,
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        aquisition_speed: float = 3 * 1e3,
    ) -> None:
        """Randomized mixture of isolated, area and chain strips.

        Args:
            n_targets: Number of strips, fixed or ``(low, high)``.
            length_strip_bounds: [m] Strip length range; every strip, of any
                kind, draws its length from it, so the kinds differ by
                geometry only.
            earth_map: Path to an equirectangular Earth texture for the land
                mask. Without it the land draw has no effect.
            mode_probs: Probabilities of the ``isolated``, ``clusters`` and
                ``mixed`` modes.
            uniform_prob: Probability that an episode places midpoints and
                anchors uniformly over the globe, ignoring the land mask (the
                paper's scenario); otherwise it draws ``land_fraction``.
            land_fraction: Probability that a midpoint or anchor is on land.
            mixed_cluster_fraction: Share of clustered strips in ``mixed``.
            chain_fraction: Probability that a cluster is a chain.
            area_strips: Number of strips of an area cluster.
            chain_strips: Number of strips of a chain cluster.
            swath_width: [m] Cross-track spacing of area strips.
            chain_kappa: von Mises concentration of the chain turn angle.
            chain_gap: [m] Gap between consecutive chain strips.
            chain_land_retries: Turn redraws that try to keep a land chain on
                land.
            priority_distribution: Strip priority law, ``U(0, 1)`` by default.
            radius: [m] Planet radius.
            aquisition_speed: [m/s] Acquisition speed of every strip.
        """
        super().__init__(
            n_targets,
            length_strip_bounds,
            priority_distribution=priority_distribution,
            radius=radius,
            aquisition_speed=aquisition_speed,
        )
        mode_probs = np.asarray(mode_probs, dtype=float)
        self.mode_probs = mode_probs / mode_probs.sum()
        self.uniform_prob = uniform_prob
        self.land_fraction = land_fraction
        # Whether the current episode uses the land mask; set per episode.
        self._masked = earth_map is not None
        self.mixed_cluster_fraction = mixed_cluster_fraction
        self.chain_fraction = chain_fraction
        self.area_strips = area_strips
        self.chain_strips = chain_strips
        self.swath_width = swath_width
        self.chain_kappa = chain_kappa
        self.chain_gap = chain_gap
        self.chain_land_retries = chain_land_retries
        self.earth_map = earth_map
        self.surface = self._load_surface_mask(earth_map)
        self.land = None if self.surface is None else self.surface == 1
        self.scene_params: dict = {}
        self.strip_kind = np.zeros(0, dtype=int)
        self.strip_cluster = np.zeros(0, dtype=int)

    @staticmethod
    def _load_surface_mask(earth_map: Optional[str]) -> Optional[np.ndarray]:
        """(lat, lon) grid, north row first: 1 land, 0 sea, -1 Antarctica;
        None without a map.

        Loaded once per scenario rather than per reset: the texture is
        several megabytes and every episode reuses the same mask.
        """
        if earth_map is None:
            return None
        from matplotlib.image import imread

        img = imread(earth_map)[::4, ::4, :3].astype(float)
        if img.max() > 1.5:
            img /= 255.0
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        land = ~((b > r) & (b > g))
        lat_rows = 90.0 - (np.arange(land.shape[0]) + 0.5) * 180.0 / land.shape[0]
        surface = land.astype(int)
        surface[land & (lat_rows[:, None] <= -60.0)] = -1
        return surface

    def on_land(self, p: np.ndarray) -> np.ndarray:
        """Land test for an ``(n, 3)`` array of points (any norm)."""
        if self.surface is None:
            return np.zeros(len(np.atleast_2d(p)), dtype=bool)
        return self.surface_code(p) == 1

    def surface_code(self, p: np.ndarray) -> np.ndarray:
        """Mask code (1 land, 0 sea, -1 Antarctica) of ``(n, 3)`` points."""
        p = np.atleast_2d(p)
        lat = np.degrees(np.arcsin(p[:, 2] / np.linalg.norm(p, axis=1)))
        lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        n_rows, n_cols = self.surface.shape
        rows = np.clip(((90.0 - lat) / 180.0 * n_rows).astype(int), 0, n_rows - 1)
        cols = np.clip(((lon + 180.0) / 360.0 * n_cols).astype(int), 0, n_cols - 1)
        return self.surface[rows, cols]

    def _sample_surface(self, land: np.ndarray) -> np.ndarray:
        """One uniform unit vector per flag, on land where ``land`` is True and
        at sea elsewhere (rejection on the mask; plain uniform without one)."""
        land = np.asarray(land, dtype=bool)
        out = np.empty((land.size, 3))
        groups = [np.arange(land.size)] if not self._masked else [
            np.flatnonzero(land), np.flatnonzero(~land)]
        for g, idx in enumerate(groups):
            filled = 0
            while filled < idx.size:
                # Land is ~26% of the sphere once Antarctica is cut, so a
                # batch four times the shortfall usually finishes in one pass.
                x = np.random.normal(size=(4 * (idx.size - filled) + 16, 3))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                if self._masked:
                    x = x[self.surface_code(x) == (1 if g == 0 else 0)]
                take = min(len(x), idx.size - filled)
                out[idx[filled:filled + take]] = x[:take]
                filled += take
        return out

    @staticmethod
    def _tangent(p: np.ndarray, bearing: float) -> np.ndarray:
        """Unit tangent at unit point ``p`` with ``bearing`` from north."""
        e = np.cross([0.0, 0.0, 1.0], p)
        n_e = np.linalg.norm(e)
        e = np.array([0.0, 1.0, 0.0]) if n_e < 1e-9 else e / n_e
        n = np.cross(p, e)
        return np.cos(bearing) * n + np.sin(bearing) * e

    @staticmethod
    def _move(p: np.ndarray, d: np.ndarray, delta: float):
        """Great-circle step of angle ``delta`` from ``p`` along tangent ``d``;
        returns the end point and the transported tangent there."""
        return (p * np.cos(delta) + d * np.sin(delta),
                -p * np.sin(delta) + d * np.cos(delta))

    def _strip_about(self, m: np.ndarray, u: np.ndarray, length: float):
        """Strip of ``length`` centred on ``m`` along tangent ``u``."""
        h = 0.5 * length / self.radius
        return m * np.cos(h) - u * np.sin(h), m * np.cos(h) + u * np.sin(h)

    def _area(self, c: np.ndarray, n: int) -> list:
        """``n`` parallel strips centred on ``c``, one swath apart.

        Every strip shares one length and one direction. ``u`` is
        perpendicular to both ``c`` and the cross-track axis ``v``, so it
        stays tangent at every offset midpoint: the strips are parallel
        without any per-strip heading correction.
        """
        u = self._tangent(c, np.random.uniform(0, 2 * np.pi))
        v = np.cross(c, u)
        length = np.random.uniform(*self.length_strip_bounds)
        segs = []
        for j in range(n):
            off = (j - 0.5 * (n - 1)) * self.swath_width / self.radius
            m = c * np.cos(off) + v * np.sin(off)
            segs.append(self._strip_about(m, u, length))
        return segs

    def _chain(self, c: np.ndarray, n: int, keep_on_land: bool) -> list:
        """``n`` strips end to end from ``c``, turning at each joint."""
        kappa = _draw(self.chain_kappa)
        p = c
        d_in = self._tangent(c, np.random.uniform(0, 2 * np.pi))
        segs = []
        for i in range(n):
            delta = np.random.uniform(*self.length_strip_bounds) / self.radius
            for attempt in range(self.chain_land_retries + 1):
                if i == 0:
                    d = d_in if attempt == 0 else self._tangent(
                        p, np.random.uniform(0, 2 * np.pi))
                else:
                    turn = np.random.vonmises(0.0, kappa)
                    d = d_in * np.cos(turn) + np.cross(p, d_in) * np.sin(turn)
                end, d_end = self._move(p, d, delta)
                if not keep_on_land or not self._masked or self.on_land(end)[0]:
                    break
            segs.append((p, end))
            if self.chain_gap > 0:
                p, d_in = self._move(end, d_end, self.chain_gap / self.radius)
            else:
                p, d_in = end, d_end
        return segs

    def regenerate_targets(self) -> None:
        """Draw the episode's mode and parameters, then its strips."""
        n_total = self.n_targets
        mode = int(np.random.choice(len(self.MODES), p=self.mode_probs))
        # Uniform episodes keep the paper's scenario in the training mix; the
        # others place strips by a drawn land fraction.
        uniform = self.surface is None or np.random.rand() < self.uniform_prob
        self._masked = not uniform
        f_land = float("nan") if uniform else _draw(self.land_fraction)
        f_clust = (0.0, 1.0, _draw(self.mixed_cluster_fraction))[mode]
        f_chain = _draw(self.chain_fraction) if mode > 0 else float("nan")

        segs, kinds, clusters = [], [], []
        remaining = int(round(f_clust * n_total))
        n_area = n_chain = 0
        while remaining > 0:
            chain = np.random.rand() < f_chain
            n = min(_draw_int(self.chain_strips if chain else self.area_strips),
                    remaining)
            land = np.random.rand() < f_land
            c = self._sample_surface(np.array([land]))[0]
            new = self._chain(c, n, land) if chain else self._area(c, n)
            segs += new
            kinds += [2 if chain else 1] * len(new)
            clusters += [n_area + n_chain] * len(new)
            n_area += not chain
            n_chain += chain
            remaining -= n

        n_iso = n_total - len(segs)
        mids = self._sample_surface(np.random.rand(n_iso) < f_land)
        for m in mids:
            u = self._tangent(m, np.random.uniform(0, 2 * np.pi))
            segs.append(self._strip_about(
                m, u, np.random.uniform(*self.length_strip_bounds)))
        kinds += [0] * n_iso
        clusters += [-1] * n_iso

        self.strip_kind = np.array(kinds, dtype=int)
        self.strip_cluster = np.array(clusters, dtype=int)
        self.targets = [
            Strip(name=f"strip-{i}", r_LP_P_start=a * self.radius,
                  r_LP_P_end=b * self.radius, priority=self.priority_distribution(),
                  aquisition_speed=self.aquisition_speed,
                  pre_imaging_time=self.pre_imaging_time)
            for i, (a, b) in enumerate(segs)
        ]
        mid_land = self.on_land(np.array([a + b for a, b in segs])) if segs else []
        self.scene_params = dict(
            mode=self.MODES[mode],
            uniform=bool(uniform),
            f_land=f_land,
            f_clust=f_clust,
            f_chain=f_chain,
            n_isolated=n_iso,
            n_area_clusters=n_area,
            n_chain_clusters=n_chain,
            land_share=float(np.mean(mid_land)) if self.land is not None else float("nan"),
        )
        logger.info(f"Strip scene: {self.scene_params}")


__doc_title__ = "Target Scenarios"
__all__ = ["Target", "Strip", "UniformTargets", "CityTargets", "UniformStripTargets",
           "MixedStripTargets"]
