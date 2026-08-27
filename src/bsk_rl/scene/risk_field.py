"""Random continuous risk field for the sweep-imaging environment.

The field gives every point on Earth a risk (importance) value in [0, 1],
built with the same recipe as ``avs_rl_tools/sweep_imaging/risk_map.py``:
randomly placed seed points with a max-importance core and a raised-cosine
falloff, combined as a probabilistic union over a smooth random ambient
background.

The field is defined in an analytic Earth-fixed frame that rotates by
``OMEGA_EARTH * t`` about +z from the inertial frame at simulation time
zero. Observations and rewards that consume the field must use
:func:`eci_to_latlon` so that both live in the same frame.
"""

import logging
from typing import Optional

import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros
from scipy.ndimage import uniform_filter, zoom

from bsk_rl.scene.scenario import Scenario

logger = logging.getLogger(__name__)

R_EARTH_KM = 6371.0            # spherical Earth, as in risk_map.py
R_EARTH_M = R_EARTH_KM * 1e3
OMEGA_EARTH = 7.2921159e-5     # rad/s


def eci_to_latlon(p_N: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lon (deg) of inertial unit vectors in the analytic Earth frame.

    Args:
        p_N: Inertial unit vectors, shape (..., 3).
        t: [s] Simulation time of each vector (broadcastable).

    Returns:
        Latitude and longitude in degrees, longitude wrapped to [-180, 180).
    """
    lat = np.degrees(np.arcsin(np.clip(p_N[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(p_N[..., 1], p_N[..., 0])) \
        - np.degrees(OMEGA_EARTH * np.asarray(t))
    return lat, (lon + 180.0) % 360.0 - 180.0


def central_angle_from_roll(roll, r_orbit: float):
    """Signed Earth central angle of the boresight ground intercept.

    Args:
        roll: [rad] Off-nadir roll angle(s).
        r_orbit: [m] Orbit radius.

    Returns:
        [rad] Central angle between the subsatellite point and the intercept,
        with the same sign as ``roll``.
    """
    return (
        np.arcsin(np.clip(r_orbit / R_EARTH_M * np.sin(roll), -1.0, 1.0)) - roll
    )


def _great_circle_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between (lat1, lon1) grids and a point."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon1 - lon2)
    cosc = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return R_EARTH_KM * np.arccos(np.clip(cosc, -1.0, 1.0))


class SweepRiskField(Scenario):
    """Scenario with a random global risk field for sweep imaging.

    Satellites linked to the scenario get a ``sweep_scene`` attribute
    pointing back at it, for use by :class:`~bsk_rl.obs.RiskTokens` and
    :class:`~bsk_rl.data.SweptRiskReward`.
    """

    def __init__(
        self,
        n_seeds: int = 1000,
        resolution_deg: float = 0.5,
        core_radius_range: tuple[float, float] = (10.0, 10.0),
        falloff_range: tuple[float, float] = (10.0, 500.0),
        peak_range: tuple[float, float] = (0.1, 1.0),
        background_max: float = 0,
        swath_km: float = 100.0,
        earth_map: Optional[str] = None,
        regenerate_each_reset: bool = True,
        attitude_swath: bool = False,
    ) -> None:
        """Random continuous risk field, following the risk_map.py recipe.

        Args:
            n_seeds: Number of risk hot spots. Keep ``n_seeds * pi *
                falloff^2`` well under the Earth's surface area, or the spots
                tile the globe and the probabilistic union saturates: every
                point then sits inside several spots and the field goes flat,
                with no visible decay around any individual centre.
            resolution_deg: Grid step of the risk field.
            core_radius_range: [km] Radius of the max-importance core.
            falloff_range: [km] Outer radius of a hot spot, measured from its
                centre: the risk holds at ``peak`` out to ``core_radius`` and
                has decayed to zero by this distance.
            peak_range: Risk value inside each core.
            background_max: Upper bound of the smooth ambient background.
            swath_km: [km] Ground width of the image swath, centered on the
                boresight intercept (simplified swath model, as in
                sweep_imaging_map.py).
            earth_map: Optional path to an Earth texture; if given, seeds are
                biased toward land and coastlines as in risk_map.py. If None,
                seeds are uniform over the sphere with the same polar taper.
            regenerate_each_reset: Draw a new random field on every reset.
            attitude_swath: Derive the swath line direction from the flown
                body attitude instead of the orbit normal. Required for the
                yaw to affect the reward when the satellite commands it
                (``Sweep(control_yaw=True)``); at zero yaw both derivations
                give the same line, so the flag is safe either way.
        """
        super().__init__()
        self.n_seeds = n_seeds
        self.resolution_deg = resolution_deg
        self.core_radius_range = core_radius_range
        self.falloff_range = falloff_range
        self.peak_range = peak_range
        self.background_max = background_max
        self.swath_km = swath_km
        self.earth_map = earth_map
        self.regenerate_each_reset = regenerate_each_reset
        self.attitude_swath = attitude_swath
        self.risk: Optional[np.ndarray] = None
        self._risk_orig: Optional[np.ndarray] = None
        self._coverage: dict = {}

    def link_satellites(self, satellites) -> None:
        """Give satellites access to the risk field."""
        super().link_satellites(satellites)
        for satellite in satellites:
            satellite.sweep_scene = self

    def _seed_weight(self) -> np.ndarray:
        """Seed sampling density on a (lat, lon) grid, north to south.

        With an Earth map, seeds are spread uniformly over land and kept off
        the ocean; the land / ocean / coastline weights below are the knobs
        for any other split. Without one the density is uniform over the
        sphere. There is no latitude taper -- the Arctic is seeded like any
        other land -- but Antarctica is weighted as ocean.
        """
        if self.earth_map is not None:
            from matplotlib.image import imread

            img = imread(self.earth_map)[::4, ::4, :3].astype(float)
            if img.max() > 1.5:
                img /= 255.0
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            ocean = (b > r) & (b > g)
            # Everything that is not ocean counts as land, deserts and ice
            # sheets alike. Testing for ice by brightness cannot work on this
            # texture: pale desert sand is as bright as snow, so it deleted
            # the Sahara, Arabia and inland Australia along with the poles.
            land = ~ocean
            # Antarctica counts as ocean: it is land by colour but carries
            # nothing worth imaging, and a hot spot that far south smears
            # across tens of degrees of longitude on this grid.
            lat_rows = 90.0 - (np.arange(land.shape[0]) + 0.5) * 180.0 / land.shape[0]
            land &= lat_rows[:, None] > -60.0
            coast = land & (uniform_filter(ocean.astype(float), 9) > 0.05)
            weight = np.where(land, 1.0, 0.05)
            weight[coast] = 3.0     # raise for a coastline preference
        else:
            weight = np.ones((90, 180))

        return weight

    def _sample_seeds(self, rng: np.random.Generator):
        """Random seed points by rejection sampling from the seed density,
        uniform on the sphere before weighting."""
        weight = self._seed_weight()
        w_max = weight.max()
        n_rows, n_cols = weight.shape

        lats = np.empty(self.n_seeds)
        lons = np.empty(self.n_seeds)
        n_kept = 0
        while n_kept < self.n_seeds:
            cand_lat = np.degrees(
                np.arcsin(rng.uniform(-1.0, 1.0, 4 * self.n_seeds))
            )
            cand_lon = rng.uniform(-180.0, 180.0, 4 * self.n_seeds)
            rows = np.clip(
                ((90.0 - cand_lat) / 180.0 * n_rows).astype(int), 0, n_rows - 1
            )
            cols = np.clip(
                ((cand_lon + 180.0) / 360.0 * n_cols).astype(int), 0, n_cols - 1
            )
            keep = rng.uniform(0.0, w_max, cand_lat.size) < weight[rows, cols]
            take = min(int(keep.sum()), self.n_seeds - n_kept)
            lats[n_kept:n_kept + take] = cand_lat[keep][:take]
            lons[n_kept:n_kept + take] = cand_lon[keep][:take]
            n_kept += take

        cores = rng.uniform(*self.core_radius_range, self.n_seeds)
        falloffs = rng.uniform(*self.falloff_range, self.n_seeds)
        peaks = rng.uniform(*self.peak_range, self.n_seeds)
        return lats, lons, cores, falloffs, peaks

    def build_risk_field(self, rng: np.random.Generator) -> None:
        """Build the continuous risk grid over the whole Earth."""
        res = self.resolution_deg
        self.lats = np.arange(-90.0 + res / 2, 90.0, res)
        self.lons = np.arange(-180.0 + res / 2, 180.0, res)
        lon_g, lat_g = np.meshgrid(self.lons, self.lats)

        # Smooth ambient background risk: coarse random field interpolated up
        # to the grid resolution. Per-axis zoom factors, so any resolution
        # works (risk_map.py hard-codes 16, requiring divisible grid sizes).
        coarse = rng.uniform(
            0.0,
            self.background_max,
            (max(self.lats.size // 16, 1), max(self.lons.size // 16, 1)),
        )
        risk = np.clip(
            zoom(
                coarse,
                (self.lats.size / coarse.shape[0], self.lons.size / coarse.shape[1]),
                order=3,
                mode="reflect",
            ),
            0.0,
            self.background_max,
        )

        # Combine background and seeds as a probabilistic union
        # (1 - prod(1 - r_i)): smooth, overlaps reinforce, bounded by 1.
        miss = 1.0 - risk
        for s, (lat0, lon0, core, falloff, peak) in enumerate(
            zip(*self._sample_seeds(rng))
        ):
            d = _great_circle_km(lat_g, lon_g, lat0, lon0)
            # `falloff` is the outer radius of the spot, not the width of the
            # ramp: the risk is at `peak` out to `core` and has decayed to
            # zero by `falloff` from the centre. A spot whose sampled falloff
            # does not clear its core is simply a hard-edged disc.
            ramp = 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * np.clip((d - core) / max(falloff - core, 1e-6), 0.0, 1.0)
                )
            )
            contribution = peak * ramp
            miss *= 1.0 - contribution
            # Hook for subclasses that attribute cells to spots (the cloud
            # field): it sees every spot's contribution in this same pass,
            # so no second sweep over the raster is needed. No-op here, and
            # it touches nothing the union depends on.
            self._note_seed_contribution(s, 0, contribution)
        self.risk = 1.0 - miss

    def _note_seed_contribution(
        self, seed_idx: int, category: int, contribution: np.ndarray
    ) -> None:
        """Per-seed hook of :meth:`build_risk_field` (no-op here).

        Called once per spot with its ``peak * ramp`` contribution over the
        whole grid, before it is folded into the union. ``category`` is
        always 0 for this single-channel field; the signature is
        :class:`CategorizedSweepRiskField`'s so that a subclass tracking
        which spot dominates each cell reads the same call in either
        hierarchy (:class:`CloudSweepRiskField` here, the quota field there).
        """

    def reset_overwrite_previous(self) -> None:
        """Regenerate the risk field and clear the coverage state."""
        super().reset_overwrite_previous()
        self._coverage = {}
        if self._risk_orig is not None and not self.regenerate_each_reset:
            # Coverage zeroes the field in place; restore the pristine copy
            self.risk = self._risk_orig.copy()
            self.covered = np.zeros(self.risk.shape, dtype=bool)
            return
        rng = np.random.default_rng(np.random.randint(2**31))
        self.build_risk_field(rng)
        self._risk_orig = self.risk.copy()
        self.covered = np.zeros(self.risk.shape, dtype=bool)
        logger.info(
            f"Risk field generated: {self.risk.shape} grid, "
            f"{self.n_seeds} seeds, mean risk {self.risk.mean():.3f}"
        )

    def _new_swath_cells(self, satellite, state):
        """Grid cells newly covered by the swath since the last call.

        The logged navigation states are used to intersect the instrument
        boresight (-b1) with the Earth; at each sample, the swath spans
        ``swath_km`` on the ground across-track, centered on the intercept
        (simplified swath model: the native swath translated to the
        boresight intercept). Consumes the log watermark in
        ``state["idx"]``, so each nav sample is processed exactly once per
        episode; how the cells are scored is the caller's business, which
        lets :class:`CategorizedSweepRiskField` reuse the geometry with a
        per-category tally.

        Args:
            satellite: Satellite whose logs to sweep.
            state: The satellite's ``_coverage`` entry (``idx`` watermark).

        Returns:
            ``(ri, ci, cell_km2)`` row/column indices of the newly swept
            grid cells and their areas, or None when there is no new
            coverage.
        """
        dynamics = satellite.dynamics
        t_s = dynamics.navLogtrans.times() * macros.NANO2SEC
        r_N = np.array(dynamics.navLogtrans.r_BN_N)
        v_N = np.array(dynamics.navLogtrans.v_BN_N)
        sigma = np.array(dynamics.navLogatt.sigma_BN)
        n = min(len(t_s), len(sigma))
        if state["idx"] >= n:
            return None
        # Start one sample early so the along-track densification below also
        # bridges the gap across the step boundary. Cells are zeroed once
        # collected, so re-covering one contributes nothing.
        idx = np.arange(max(state["idx"] - 1, 0), n)
        state["idx"] = n

        # Boresight (-b1: nadir when tracking the Hill frame) intersected
        # with the Earth sphere
        b_N = np.array(
            [-rbk.MRP2C(s).T @ np.array([1.0, 0.0, 0.0]) for s in sigma[idx]]
        )
        r = r_N[idx]
        rb = np.sum(r * b_N, axis=1)
        disc = rb**2 - (np.sum(r * r, axis=1) - R_EARTH_M**2)
        hits = disc > 0.0
        if not np.any(hits):
            return None
        s_int = -rb[hits] - np.sqrt(disc[hits])
        p_hat = r[hits] + s_int[:, None] * b_N[hits]
        p_hat /= np.linalg.norm(p_hat, axis=1, keepdims=True)

        if self.attitude_swath:
            # Detector-line axis b3 in inertial, from the flown attitude. At
            # zero yaw its tangent-plane projection equals the orbit-normal
            # projection below (b3, the orbit normal and p_hat are coplanar),
            # so this branch only changes anything once yaw is commanded. The
            # projection norm is bounded below by cos(incidence), so the
            # normalization is safe.
            b3_N = np.array(
                [rbk.MRP2C(s).T @ np.array([0.0, 0.0, 1.0]) for s in sigma[idx]]
            )[hits]
            c_hat = b3_N - np.sum(b3_N * p_hat, axis=1, keepdims=True) * p_hat
            c_hat /= np.linalg.norm(c_hat, axis=1, keepdims=True)
        else:
            # Cross-track direction on the ground: the orbit normal projected
            # into the tangent plane at the intercept
            h_N = np.cross(r[hits], v_N[idx][hits])
            h_hat = h_N / np.linalg.norm(h_N, axis=1, keepdims=True)
            c_hat = h_hat - np.sum(h_hat * p_hat, axis=1, keepdims=True) * p_hat
            c_hat /= np.linalg.norm(c_hat, axis=1, keepdims=True)
        t_hit = t_s[idx][hits]
        idx_hit = idx[hits]

        # Sample spacing has to stay below one grid cell, or the stamped
        # swath comes out as isolated dots instead of a continuous band. A
        # cell is `resolution_deg` on a side, but a longitude cell is only
        # cos(lat) as wide on the ground, so the bound tightens toward the
        # poles: both the across-track and along-track steps are sized to
        # half the narrowest cell the swath actually touches (floored so the
        # sample count stays bounded as cos(lat) -> 0).
        cell_km = R_EARTH_KM * np.radians(self.resolution_deg)
        half_swath = 0.5 * self.swath_km / R_EARTH_KM
        lat_max = min(
            np.degrees(np.arcsin(np.abs(p_hat[:, 2]).max()) + half_swath), 89.5
        )
        step_km = 0.5 * max(cell_km * np.cos(np.radians(lat_max)), 0.02 * cell_km)

        # Along-track: the recorder samples are tens of km apart, so fill in
        # between temporally adjacent ones (a boresight that misses the Earth
        # breaks adjacency, and those gaps must not be bridged).
        adj = np.diff(idx_hit) == 1 if idx_hit.size > 1 else np.zeros(0, bool)
        if adj.any():
            cos_seg = np.sum(p_hat[:-1][adj] * p_hat[1:][adj], axis=1)
            seg_km = R_EARTH_KM * np.arccos(np.clip(cos_seg, -1.0, 1.0)).max()
            w = np.linspace(0.0, 1.0, max(1, int(np.ceil(seg_km / step_km))),
                            endpoint=False)[1:]
            if w.size:
                wa = w[None, :, None]
                p_mid = p_hat[:-1][adj][:, None, :] * (1.0 - wa) \
                    + p_hat[1:][adj][:, None, :] * wa
                c_mid = c_hat[:-1][adj][:, None, :] * (1.0 - wa) \
                    + c_hat[1:][adj][:, None, :] * wa
                t_mid = t_hit[:-1][adj][:, None] * (1.0 - w[None, :]) \
                    + t_hit[1:][adj][:, None] * w[None, :]
                p_hat = np.concatenate([p_hat, p_mid.reshape(-1, 3)])
                c_hat = np.concatenate([c_hat, c_mid.reshape(-1, 3)])
                t_hit = np.concatenate([t_hit, t_mid.reshape(-1)])
                # Re-normalize: the interpolants are chords, not arcs.
                p_hat /= np.linalg.norm(p_hat, axis=1, keepdims=True)
                c_hat -= np.sum(c_hat * p_hat, axis=1, keepdims=True) * p_hat
                c_hat /= np.linalg.norm(c_hat, axis=1, keepdims=True)

        # Across-track sub-points spanning the swath width
        n_sub = max(3, 2 * int(np.ceil(0.5 * self.swath_km / step_km)) + 1)
        beta = np.linspace(-1.0, 1.0, n_sub) * half_swath
        q = (
            np.cos(beta)[None, :, None] * p_hat[:, None, :]
            + np.sin(beta)[None, :, None] * c_hat[:, None, :]
        )
        lat, lon = eci_to_latlon(q, t_hit[:, None])

        # Nearest cells, counted once, then zeroed in place
        res = self.resolution_deg
        ri = np.clip(
            np.rint((lat - self.lats[0]) / res).astype(int), 0, self.lats.size - 1
        )
        ci = np.rint((lon - self.lons[0]) / res).astype(int) % self.lons.size
        cells = np.unique(ri * self.lons.size + ci)
        ri, ci = cells // self.lons.size, cells % self.lons.size
        cell_km2 = (
            R_EARTH_KM**2
            * np.radians(res) ** 2
            * np.cos(np.radians(self.lats[ri]))
        )
        return ri, ci, cell_km2

    def update_coverage(self, satellite) -> float:
        """Consume the risk under the satellite's swath since the last call.

        Grid cells touched by the swath (:meth:`_new_swath_cells`)
        contribute ``risk * cell_area`` to the satellite's collected total
        exactly once, and their risk is zeroed in place -- so subsequent
        rewards and :class:`~bsk_rl.obs.RiskTokens` observations both see
        the reduced importance.

        Idempotent within a step: safe to call from both the observation
        and the reward path, whichever comes first does the work.

        Args:
            satellite: Satellite whose coverage to update.

        Returns:
            [risk * km^2] Cumulative risk collected this episode.
        """
        state = self._coverage.setdefault(
            satellite.name, {"idx": 1, "collected": 0.0}
        )
        cells = self._new_swath_cells(satellite, state)
        if cells is None:
            return state["collected"]
        ri, ci, cell_km2 = cells
        state["collected"] += float(np.sum(self.risk[ri, ci] * cell_km2))
        self.risk[ri, ci] = 0.0
        self.covered[ri, ci] = True
        return state["collected"]

    def sample_risk(self, lat, lon):
        """Bilinear sample of the risk field, periodic in longitude.

        Args:
            lat: [deg] Latitudes to sample at.
            lon: [deg] Longitudes to sample at.

        Returns:
            Risk values in [0, 1] with the broadcast shape of lat/lon.
        """
        lats, lons, risk = self.lats, self.lons, self.risk
        fi = (lat - lats[0]) / (lats[1] - lats[0])
        fj = (lon - lons[0]) / (lons[1] - lons[0])
        i0 = np.clip(np.floor(fi).astype(int), 0, lats.size - 2)
        wi = np.clip(fi - i0, 0.0, 1.0)
        j0 = np.floor(fj).astype(int)
        wj = fj - j0
        j0 %= lons.size
        j1 = (j0 + 1) % lons.size
        lo = risk[i0, j0] * (1.0 - wj) + risk[i0, j1] * wj
        hi = risk[i0 + 1, j0] * (1.0 - wj) + risk[i0 + 1, j1] * wj
        return lo * (1.0 - wi) + hi * wi


__doc_title__ = "Sweep Risk Field"
__all__ = ["SweepRiskField", "eci_to_latlon", "central_angle_from_roll"]
