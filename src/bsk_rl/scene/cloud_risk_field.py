"""Risk field under a synthetic global cloud cover, observed through a forecast.

Every reset draws a **cloud-fraction map** over the whole grid that looks
like an instantaneous satellite cloud product — a filamentary texture of
clear ground, partly cloudy patches and overcast decks: isotropic fractal
noise on the sphere, domain-warped into swirls, then pushed through a soft
threshold — with a
three-region climatology: the non-polar ocean, the non-polar land and the
polar caps (``|lat| >= polar_lat``) each get their own mean cover
(defaults 0.70 / 0.55 / 0.72, about 0.66 over the Earth); optionally the
globe as a whole is pinned to ``cloud_mean_global`` by re-solving one
region (``pin_region``), all blended smoothly across coastlines and the
polar boundary. Every hot spot
("target") of the base :class:`~bsk_rl.scene.SweepRiskField` then draws
its true cover and a forecast of it around the map's value :math:`\\mu_t`
at its centre:

.. math::

    c_p = \\max(0, \\min(U(0, 2\\mu_t), 1)), \\qquad
    c_f = \\max(0, \\min(N(c_p, \\sigma), 1)), \\quad \\sigma \\sim U(\\sigma_0, \\sigma_1)

so a target in a clear region stays clear and one under the overcast
draws anywhere up to fully overcast. On the grid the draw acts as a
smooth multiplicative modulation of the map — full at the spot's core,
fading to nothing at its edge — so the per-cell truth keeps the texture
and equals :math:`c_p` at the target's centre; the forecast adds the
target's :math:`N(0, \\sigma)` error over its footprint and a smooth
background error elsewhere.

A swept cell is **paid** its priority only if its *true* cover is below
``cloud_tau``; it is zeroed either way (the ground was imaged, the image is
just useless under cloud), so the priority picture and the coverage
bookkeeping behave exactly as in the base field. Observations see the
**forecast**, never the truth.

For a given generator state the priority field is the base field bit for
bit — the clouds are drawn from the same generator *after* the field is
built — and the global ``np.random`` stream is not touched, so an episode
seed draws the same field and the same orbit as the cloud-free env: an
episode here is the cloud-free episode with clouds added.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from bsk_rl.scene.risk_field import R_EARTH_KM, SweepRiskField

# --- Cloud-map recipe (calibrated 2026-08-26 on the 0.5 deg grid) --------
# Texture: isotropic fractal noise ON THE SPHERE — octaves of Gaussian 3-D
# lattices spanning the unit cube, cubic-interpolated at every grid cell's
# unit vector (coarsest 8 nodes ~ 1600 km, finest 128 ~ 100 km, persistence
# 0.55) — warped three times along the surface by random vector fields of
# the given rms [km] (large swirls, then filaments). Synthesizing in 3-D
# rather than in lat/lon cells keeps every feature its physical size up to
# the poles (a lat/lon texture squeezes into streaks converging on them)
# and makes the map seamless at the antimeridian.
_FBM_BASE = 8
_FBM_OCTAVES = 5
_FBM_PERSISTENCE = 0.55
_WARP_KM = (1400.0, 660.0, 280.0)
# Cover = clip(0.5 + (noise + offset) / (2 * ramp)) with the unit-variance
# noise: the ramp half-width sets how much of the ground saturates at
# clear / overcast (0.45 leaves roughly half of it saturated, the look of
# a cloud-fraction product), the offset — one per region, solved on every
# draw for that region's mean — sets its level.
_RAMP = 0.45
_OFFSET_BOUNDS = (-3.0, 3.0)
# Smoothing of the region masks, in grid cells: coastlines and the polar
# boundary hand over between offsets over about a degree, not a cell.
_REGION_SMOOTH = 2.0
# Nodes of the smooth background forecast error, in grid cells.
_FORECAST_NOISE_CELLS = 16


def _bilinear_weights(lats, lons, lat, lon):
    """Corner indices and weights of a bilinear sample, periodic in
    longitude — the base class's ``sample_risk`` arithmetic, factored out so
    several grids can be sampled at the same points for one index
    computation, with the same operation order (so the priority channel is
    bit-identical to the base class's single-channel observation)."""
    fi = (lat - lats[0]) / (lats[1] - lats[0])
    fj = (lon - lons[0]) / (lons[1] - lons[0])
    i0 = np.clip(np.floor(fi).astype(int), 0, lats.size - 2)
    wi = np.clip(fi - i0, 0.0, 1.0)
    j0 = np.floor(fj).astype(int)
    wj = fj - j0
    j0 %= lons.size
    j1 = (j0 + 1) % lons.size
    return i0, j0, j1, wi, wj


def _bilinear_apply(grid, i0, j0, j1, wi, wj):
    lo = grid[i0, j0] * (1.0 - wj) + grid[i0, j1] * wj
    hi = grid[i0 + 1, j0] * (1.0 - wj) + grid[i0 + 1, j1] * wj
    return lo * (1.0 - wi) + hi * wi


def _sphere_points(lats, lons):
    """Unit vectors of the grid cells, shape (n_lat, n_lon, 3)."""
    lat = np.radians(lats)[:, None]
    lon = np.radians(lons)[None, :]
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
         np.sin(lat) * np.ones_like(lon)],
        axis=-1,
    )


def _fbm_sphere(rng, points, base, octaves, persistence):
    """Isotropic fractal noise at unit vectors ``points``: octaves of
    Gaussian lattices over [-1, 1]^3, each cubic-interpolated at the points
    and summed with geometrically decaying weights; unit std."""
    field = np.zeros(points.shape[:-1])
    flat = points.reshape(-1, 3)
    amp = 1.0
    for k in range(octaves):
        n = base * 2 ** k
        lattice = rng.standard_normal((n, n, n))
        coords = (flat + 1.0) * 0.5 * (n - 1)          # [-1, 1] -> [0, n-1]
        field += amp * map_coordinates(
            lattice, coords.T, order=3, mode="reflect"
        ).reshape(field.shape)
        amp *= persistence
    return field / field.std()


def _warp_sphere(points, rng, km):
    """Displace the sample points along the surface by an isotropic random
    vector field (``km`` rms per component, tangent part kept) — the swirls
    and filaments of an advected cloud field, the same size everywhere."""
    v = np.stack(
        [_fbm_sphere(rng, points, 4, 3, 0.5) for _ in range(3)], axis=-1
    ) * (km / R_EARTH_KM)
    v -= np.sum(v * points, axis=-1, keepdims=True) * points
    p = points + v
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


class CloudSweepRiskField(SweepRiskField):
    """Sweep risk field with a synthetic cloud cover and its forecast.

    ``sample_risk`` returns the priority and the forecast channel-LAST, so
    :class:`~bsk_rl.obs.RiskTokens`'s ``rectified_map`` works unchanged and
    yields an ``(n_time, n_roll, 2)`` picture; ``update_coverage`` returns
    the cumulative *paid* priority, so the base
    :class:`~bsk_rl.data.SweptRiskReward` rewards it in its own units.
    """

    def __init__(
        self,
        cloud_mean_ocean: float = 0.70,
        cloud_mean_land: float = 0.55,
        cloud_mean_polar: float = 0.72,
        cloud_mean_global: Optional[float] = None,
        pin_region: str = "polar",
        polar_lat: float = 66.5,
        cloud_tau: float = 0.65,
        forecast_sigma_range: tuple[float, float] = (0.10, 0.10),
        **kwargs,
    ) -> None:
        """Cloud-covered risk field.

        Args:
            cloud_mean_ocean: Mean cover of the non-polar water (~65% of
                the Earth). The default region means give an area-weighted
                global mean of about 0.66 (no global pin by default).
            cloud_mean_land: Mean cover of the non-polar land (~27%).
            cloud_mean_polar: Mean cover of the polar caps (~8%), water and
                land alike.
            cloud_mean_global: Area-weighted global mean to pin exactly, or
                None to let it follow from the three region means. When
                set, two regions keep their means and ``pin_region``'s
                offset is re-solved so the globe averages to this.
            pin_region: ``"ocean"``, ``"land"`` or ``"polar"`` — the
                region that absorbs the global pin; its own mean argument
                is then only the starting point. Pinning 0.674 on the
                default means would push the polar caps (~8% of the Earth)
                to ~0.97; the ocean, two thirds of the Earth, moves least.
            polar_lat: [deg] Latitude beyond which a cell is polar.
            cloud_tau: A swept cell is paid its priority only if its true
                cover is strictly below this.
            forecast_sigma_range: Bounds of the per-target forecast standard
                deviation, drawn uniformly (equal bounds fix it); the
                forecast is the true cover plus Gaussian noise of that
                width, clipped to [0, 1].
            kwargs: Forwarded to :class:`SweepRiskField`. The land / water
                split comes from its ``earth_map``; without one the whole
                non-polar globe takes the ocean mean.
        """
        super().__init__(**kwargs)
        sigma_lo, sigma_hi = forecast_sigma_range
        for mean in (cloud_mean_ocean, cloud_mean_land, cloud_mean_polar):
            assert 0.0 < mean < 1.0, mean
        assert cloud_mean_global is None or 0.0 < cloud_mean_global < 1.0, cloud_mean_global
        assert pin_region in ("ocean", "land", "polar"), pin_region
        assert 0.0 <= cloud_tau <= 1.0 and 0.0 < polar_lat < 90.0, (cloud_tau, polar_lat)
        assert 0.0 <= sigma_lo <= sigma_hi, forecast_sigma_range
        self.cloud_mean_ocean = cloud_mean_ocean
        self.cloud_mean_land = cloud_mean_land
        self.cloud_mean_polar = cloud_mean_polar
        self.cloud_mean_global = cloud_mean_global
        self.pin_region = pin_region
        self.polar_lat = polar_lat
        self.cloud_tau = cloud_tau
        self.forecast_sigma_range = (sigma_lo, sigma_hi)
        # Grids of the last build: the cloud-fraction map, the true cover
        # (map modulated by the targets' draws), the forecast, the payable
        # mask, the dominant spot of every cell (-1 outside every footprint)
        # and the cell areas [km^2].
        self.cloud_map: Optional[np.ndarray] = None
        self.cloud_true: Optional[np.ndarray] = None
        self.cloud_forecast: Optional[np.ndarray] = None
        self.clear: Optional[np.ndarray] = None
        self.cell_seed: Optional[np.ndarray] = None
        self.cell_area: Optional[np.ndarray] = None
        # Per-target values of the last build, for gates and figures: the
        # map at the target, its true cover, forecast sigma and forecast.
        self.spot_map: Optional[np.ndarray] = None
        self.spot_cloud: Optional[np.ndarray] = None
        self.spot_sigma: Optional[np.ndarray] = None
        self.spot_forecast: Optional[np.ndarray] = None
        self._seeds = None
        self._best_val = None
        self._best_seed = None
        self._regions = None

    # --- build: spot attribution, then the cloud map and the draws --------

    def _sample_seeds(self, rng):
        """The base sampler, keeping the draw (positions, sizes, peaks):
        the per-target cover needs the centres and the attribution needs
        the peaks. Same RNG consumption."""
        self._seeds = super()._sample_seeds(rng)
        return self._seeds

    def _note_seed_contribution(self, seed_idx, category, contribution):
        """Track the spot contributing most to each cell (the base loop's
        per-seed hook)."""
        if self._best_val is None:
            self._best_val = np.zeros(contribution.shape)
            self._best_seed = np.full(contribution.shape, -1, dtype=int)
        better = contribution > self._best_val
        self._best_val[better] = contribution[better]
        self._best_seed[better] = seed_idx

    def _area_weights(self) -> np.ndarray:
        area = np.cos(np.radians(self.lats))[:, None] * np.ones(self.lons.size)[None, :]
        return area / area.sum()

    def _region_weights(self) -> np.ndarray:
        """(3, n_lat, n_lon) membership of every cell in the non-polar
        ocean, the non-polar land and the polar caps, summing to one:
        hard masks from the Earth texture (the seed density's colour test)
        and the polar latitude, smoothed over ``_REGION_SMOOTH`` cells so
        the offsets hand over gradually. Cached: the grid does not change
        between resets."""
        if self._regions is not None:
            return self._regions
        n_lat, n_lon = self.lats.size, self.lons.size
        if self.earth_map is not None:
            from matplotlib.image import imread

            img = imread(self.earth_map)[::4, ::4, :3].astype(float)
            if img.max() > 1.5:
                img /= 255.0
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            land = ~((b > r) & (b > g))
            n_rows, n_cols = land.shape
            rows = np.clip(((90.0 - self.lats) / 180.0 * n_rows).astype(int), 0, n_rows - 1)
            cols = np.clip(((self.lons + 180.0) / 360.0 * n_cols).astype(int), 0, n_cols - 1)
            land = land[rows][:, cols].astype(float)
        else:
            land = np.zeros((n_lat, n_lon))
        polar = (np.abs(self.lats) >= self.polar_lat).astype(float)[:, None] * np.ones(n_lon)[None, :]
        land = gaussian_filter(land, sigma=_REGION_SMOOTH)
        polar = gaussian_filter(polar, sigma=_REGION_SMOOTH)
        self._regions = np.stack([
            (1.0 - polar) * (1.0 - land),   # non-polar ocean
            (1.0 - polar) * land,           # non-polar land
            polar,                          # polar caps
        ])
        return self._regions

    def build_cloud_map(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a global cloud-fraction map (module docstring recipe).

        The sample points are warped first and the noise evaluated at the
        warped points, which is the same thing as warping the noise field
        and is exact on the sphere. Requires ``self.lats`` / ``self.lons``
        (set by the field build).
        """
        shape = (self.lats.size, self.lons.size)
        points = _sphere_points(self.lats, self.lons)
        for km in _WARP_KM:
            points = _warp_sphere(points, rng, km)
        z = _fbm_sphere(rng, points, _FBM_BASE, _FBM_OCTAVES, _FBM_PERSISTENCE)
        z = (z - z.mean()) / z.std()
        area = self._area_weights()
        regions = self._region_weights()
        targets = (self.cloud_mean_ocean, self.cloud_mean_land, self.cloud_mean_polar)

        def cover(offsets):
            offset = np.tensordot(np.asarray(offsets), regions, axes=1)
            return np.clip(0.5 + (z + offset) / (2.0 * _RAMP), 0.0, 1.0)

        # One offset per region, each solved by bisection for its region's
        # area-weighted mean; the regions only interact across the smoothed
        # boundaries, so two sweeps settle them.
        offsets = [0.0, 0.0, 0.0]
        for _ in range(2):
            for k, target in enumerate(targets):
                weight = regions[k] * area
                weight /= weight.sum()
                lo, hi = _OFFSET_BOUNDS
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    trial = list(offsets)
                    trial[k] = mid
                    if float(np.sum(cover(trial) * weight)) < target:
                        lo = mid
                    else:
                        hi = mid
                offsets[k] = 0.5 * (lo + hi)
        if self.cloud_mean_global is not None:
            # The global pin: re-solve the pinned region's offset alone for
            # the area-weighted mean of the whole globe, the others kept.
            k = ("ocean", "land", "polar").index(self.pin_region)
            lo, hi = _OFFSET_BOUNDS
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                trial = list(offsets)
                trial[k] = mid
                if float(np.sum(cover(trial) * area)) < self.cloud_mean_global:
                    lo = mid
                else:
                    hi = mid
            offsets[k] = 0.5 * (lo + hi)
        return cover(offsets)

    def cover_distribution(self, bins: int = 50, grid=None) -> dict:
        """Area-weighted distribution of the cloud cover: percentage of area
        per cover bin, for the whole globe (``global``) and within each
        region (``ocean`` / ``land`` / ``polar``, each as a percentage of
        its own area), plus the bin ``edges``. ``grid`` defaults to the true
        cover of the last build; pass ``cloud_map`` for the climatology
        before the targets' draws."""
        grid = self.cloud_true if grid is None else grid
        edges = np.linspace(0.0, 1.0, bins + 1)
        area = self._area_weights()
        out = {"edges": edges}
        weights = [("global", np.ones_like(area))]
        weights += list(zip(("ocean", "land", "polar"), self._region_weights()))
        for name, weight in weights:
            w = weight * area
            out[name], _ = np.histogram(grid, bins=edges, weights=w / w.sum() * 100.0)
        return out

    def region_cloud_means(self) -> dict:
        """Area-weighted mean of the last cloud-fraction map per region
        (``ocean`` / ``land`` / ``polar``) and its ``global`` mean."""
        area = self._area_weights()
        regions = self._region_weights()
        means = {}
        for name, weight in zip(("ocean", "land", "polar"), regions):
            w = weight * area
            means[name] = float(np.sum(self.cloud_map * w) / w.sum())
        means["global"] = float(np.sum(self.cloud_map * area))
        return means

    def _map_at_targets(self, cloud_map, lat_t, lon_t):
        """Map value of the cell holding each target's centre — the same
        nearest-cell rule as the coverage stamping, so a target's scalar
        cover is exactly its centre cell's."""
        res = self.resolution_deg
        ri = np.clip(np.rint((lat_t - self.lats[0]) / res).astype(int), 0, self.lats.size - 1)
        ci = np.rint((lon_t - self.lons[0]) / res).astype(int) % self.lons.size
        return cloud_map[ri, ci]

    def global_cloud_mean(self) -> float:
        """Area-weighted mean of the last cloud-fraction map."""
        return float(np.sum(self.cloud_map * self._area_weights()))

    def build_risk_field(self, rng: np.random.Generator) -> None:
        """The base field, then the cloud map and the targets' draws from
        the same generator.

        The base build is untouched (its RNG consumption included), so the
        priority field is the cloud-free one for the same generator state;
        the cloud draws come after it in a fixed order — map, then per
        target cover, sigma and forecast error, then the background error —
        so a field is reproducible from its generator state alone.
        """
        self._best_val = None
        self._best_seed = None
        super().build_risk_field(rng)

        n_lat, n_lon = self.risk.shape
        self.cloud_map = self.build_cloud_map(rng)

        # Per-target draws around the map value at the target's centre.
        n = self.n_seeds
        if n > 0 and self._seeds is not None:
            lat_t, lon_t, _, _, peak_t = self._seeds
            mu_t = self._map_at_targets(self.cloud_map, lat_t, lon_t)
        else:
            mu_t, peak_t = np.zeros(0), np.ones(0)
        u_t = rng.uniform(0.0, 2.0, n)
        cp_t = np.clip(u_t * mu_t, 0.0, 1.0)
        sigma_t = rng.uniform(*self.forecast_sigma_range, n)
        e_t = rng.normal(0.0, 1.0, n) * sigma_t
        cf_t = np.clip(cp_t + e_t, 0.0, 1.0)

        # Background forecast error: smooth, with the per-node sigma drawn
        # from the same range as the targets'.
        cells = _FORECAST_NOISE_CELLS
        shape = (max(n_lat // cells, 1), max(n_lon // cells, 1))
        sigma_bg = rng.uniform(*self.forecast_sigma_range, shape)
        e_bg = zoom(
            rng.normal(0.0, 1.0, shape) * sigma_bg,
            (n_lat / shape[0], n_lon / shape[1]), order=3, mode="reflect",
        )

        if self._best_seed is None or n == 0:
            seed = np.full(self.risk.shape, -1, dtype=int)
            weight = np.zeros(self.risk.shape)
        else:
            seed = np.where(self._best_val > 0.0, self._best_seed, -1)
            # The dominant spot's own profile, 1 at its core and 0 at its
            # edge: how much of the target's draw a cell takes on.
            weight = np.where(
                seed >= 0,
                self._best_val / np.maximum(peak_t[np.maximum(seed, 0)], 1e-12),
                0.0,
            )
        idx = np.maximum(seed, 0)
        self.cell_seed = seed
        factor = 1.0 + (u_t[idx] - 1.0) * weight if n > 0 else np.ones(self.risk.shape)
        self.cloud_true = np.clip(self.cloud_map * factor, 0.0, 1.0)
        error = (weight * e_t[idx] if n > 0 else 0.0) + (1.0 - weight) * e_bg
        self.cloud_forecast = np.clip(self.cloud_true + error, 0.0, 1.0)
        self.clear = self.cloud_true < self.cloud_tau
        res = self.resolution_deg
        self.cell_area = (
            R_EARTH_KM ** 2 * np.radians(res) ** 2
            * np.cos(np.radians(self.lats))[:, None]
            * np.ones(n_lon)[None, :]
        )
        self.spot_map, self.spot_cloud = mu_t, cp_t
        self.spot_sigma, self.spot_forecast = sigma_t, cf_t
        self._best_val = None
        self._best_seed = None

    def sample_target_cloud(self, rng: np.random.Generator, n_maps: int = 10) -> np.ndarray:
        """True cover of the targets of ``n_maps`` fresh draws — seeds, map
        and per-target draw only, no field build — for checking the
        calibrated distribution (module docstring) cheaply."""
        if getattr(self, "lats", None) is None:
            res = self.resolution_deg
            self.lats = np.arange(-90.0 + res / 2, 90.0, res)
            self.lons = np.arange(-180.0 + res / 2, 180.0, res)
        out = []
        for _ in range(n_maps):
            lat_t, lon_t, _, _, _ = super()._sample_seeds(rng)
            cloud_map = self.build_cloud_map(rng)
            mu_t = self._map_at_targets(cloud_map, lat_t, lon_t)
            out.append(np.clip(rng.uniform(0.0, 2.0, mu_t.size) * mu_t, 0.0, 1.0))
        return np.concatenate(out)

    # --- episode ------------------------------------------------------

    def payable_priority(self) -> float:
        """[risk * km^2] Priority of the pristine field under clear sky —
        the most any policy could be paid this episode."""
        return float(np.sum(self._risk_orig * self.clear * self.cell_area))

    def update_coverage(self, satellite) -> float:
        """Consume the priority under the swath, paying only clear cells.

        Same swath geometry and idempotency as the base class. Every newly
        swept cell is zeroed and counted as swept; its ``risk * area`` is
        paid if its true cover is below ``cloud_tau`` and lost to cloud
        otherwise.

        Returns:
            [risk * km^2] Cumulative PAID priority this episode — what the
            :class:`~bsk_rl.data.SweptRiskReward` store reads.
        """
        state = self._coverage.setdefault(
            satellite.name,
            {"idx": 1, "collected": 0.0, "swept": 0.0, "clouded": 0.0},
        )
        cells = self._new_swath_cells(satellite, state)
        if cells is None:
            return state["collected"]
        ri, ci, cell_km2 = cells
        value = self.risk[ri, ci] * cell_km2
        clear = self.clear[ri, ci]
        state["swept"] += float(np.sum(value))
        state["collected"] += float(np.sum(value[clear]))
        state["clouded"] += float(np.sum(value[~clear]))
        self.risk[ri, ci] = 0.0
        self.covered[ri, ci] = True
        return state["collected"]

    def coverage_state(self, satellite_name: str) -> dict:
        """The episode's bookkeeping for ``satellite_name``: ``collected``
        (paid), ``swept`` (all, i.e. what the cloud-free reward would have
        paid on this flight) and ``clouded`` (lost), in risk * km^2."""
        state = self._coverage.get(satellite_name)
        if state is None:
            return dict(collected=0.0, swept=0.0, clouded=0.0)
        return {k: float(v) for k, v in state.items() if k != "idx"}

    def sample_true_cover(self, lat, lon) -> np.ndarray:
        """Bilinear sample of the TRUE cover — what the reward is gated on
        and the observation never shows — periodic in longitude, for
        rendering a flight against the ground that could actually pay."""
        return _bilinear_apply(
            self.cloud_true, *_bilinear_weights(self.lats, self.lons, lat, lon)
        )

    def sample_risk(self, lat, lon) -> np.ndarray:
        """Bilinear sample of the priority and the forecast, periodic in
        longitude.

        Returns:
            Channel-LAST array, shape ``lat.shape + (2,)``: priority, then
            forecast. The trailing axis keeps ``RiskTokens.rectified_map``
            working untouched.
        """
        weights = _bilinear_weights(self.lats, self.lons, lat, lon)
        return np.stack(
            [
                _bilinear_apply(self.risk, *weights),
                _bilinear_apply(self.cloud_forecast, *weights),
            ],
            axis=-1,
        )


__doc_title__ = "Cloud Sweep Risk Field"
__all__ = ["CloudSweepRiskField"]
