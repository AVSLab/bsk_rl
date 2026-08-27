"""Risk field with typed hot spots (terrestrial / hydrological / military).

Same construction recipe as :class:`~bsk_rl.scene.SweepRiskField`, but the
seeds carry a category and the field keeps one channel per category instead
of collapsing everything into a single grid: ``risk`` has shape
``(n_categories, n_lat, n_lon)``. Two ways of assigning the categories:

* ``seeding="v1_partition"`` (default) — the hot-spot distribution is
  EXACTLY the base class's: same sampler, same density and coast band,
  same RNG consumption, same sizes and peaks, so seed ``k`` of this field
  is seed ``k`` of :class:`SweepRiskField` with labels painted on
  afterwards. Labels follow the base geography with equal counts per
  category: a third of the seeds on the base coast band are hydrological;
  every seed at sea, the remaining coastal seeds and as many interior
  seeds as needed are military; the remaining interior seeds are
  terrestrial. The channels partition the base raster cell by cell, so the
  channel sum is the base field bit for bit — a category-blind agent sees
  exactly the picture it was trained on, and a categorized policy can be
  scored against it on the very same field.
* ``seeding="per_category"`` — the earlier design: each category draws its
  own seeds from its own geographic density (terrestrial inland only,
  hydrological on the coast band only, military 25/25/50 interior / coast /
  sea), with winner-take-all resolving overlaps. This changes the overall
  spatial distribution of hot spots relative to the base field.

The scenario also owns the per-episode *preference* vector over categories,
drawn on reset (Dirichlet) or pinned with ``fixed_preference`` for
evaluation. It lives here rather than on the satellite because both the
reward (:class:`~bsk_rl.data.ProportionalSweptRiskReward`, which reads it
as the operator's required composition, or
:class:`~bsk_rl.data.DiminishingSweptRiskReward`, which reads it as a
budget share) and the observation need the same draw, and both already
reach the scenario through ``satellite.sweep_scene``.
"""

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter, zoom

from bsk_rl.scene.risk_field import SweepRiskField, _great_circle_km

logger = logging.getLogger(__name__)

# EM-DAT-style naming: terrestrial natural hazards (e.g. wildfire),
# hydrological hazards (e.g. riverine/coastal flooding), military activity.
CATEGORY_NAMES = ("terrestrial", "hydrological", "military")


class CategorizedSweepRiskField(SweepRiskField):
    """Sweep risk field with one channel per risk category.

    ``update_coverage`` tallies collected risk per category (a vector) and
    zeroes all channels of a swept cell — the ground is imaged once,
    whichever categories happened to be there. ``sample_risk`` returns the
    channel values channel-LAST, so :class:`~bsk_rl.obs.RiskTokens`'s
    ``rectified_map`` works unchanged and simply yields a
    ``(n_time, n_roll, n_categories)`` picture.
    """

    def __init__(
        self,
        n_seeds: int = 1500,
        seed_split: tuple = (1 / 3, 1 / 3, 1 / 3),
        category_land_weights: tuple = (
            dict(land=1.0, ocean=0.0, coast=0.0),    # terrestrial: interior only
            dict(land=0.0, ocean=0.0, coast=1.0),    # hydrological: coast band only
            dict(land=1.0, ocean=0.484, coast=2.364),  # military: 25/25/50 interior/coast/sea
        ),
        preference_alpha: tuple = (0.2, 0.2, 0.2),
        preference_min: float = 0.05,
        preference_onehot_prob: float = 0.0,
        fixed_preference: Optional[tuple] = None,
        exclusive_categories: bool = True,
        seeding: str = "v1_partition",
        **kwargs,
    ) -> None:
        """Categorized risk field over the base recipe.

        Args:
            n_seeds: Total number of hot spots across all categories.
            seeding: ``"v1_partition"`` keeps the base class's seed set
                and labels it on the base class's own land / coast / sea
                masks (module docstring); ``"per_category"`` draws each
                category from its own density below. With the base
                weights (1.0 / 0.05 / 3.0) the base seeds fall ~40%
                interior / 51% coast band / 8% sea, so under
                ``v1_partition`` hydrological is purely coastal, terrestrial
                purely interior, and military is ~25% sea / ~54% coastal
                surplus / ~21% interior.
            seed_split: (``per_category`` only) Fraction of ``n_seeds``
                given to each category (rounded per category, so the total
                can be off by one).
            category_land_weights: (``per_category`` only) One
                land/ocean/coast density triple per category, filling the
                same three slots the base class fixes at 1.0 / 0.05 / 3.0.
                The defaults make the geographies mutually distinct at the
                source: terrestrial seeds ONLY on the land interior;
                hydrological ONLY on the coastline band; military 25%
                interior / 25% coast / 50% at sea. The weights are
                DENSITIES, not seed shares — the mask's areas are ~74%
                ocean (Antarctica included), ~8% coast band, ~18% interior,
                so military's target shares solve to ocean=0.484 /
                coast=2.364 against interior=1. Its length also fixes the
                number of categories in both modes, and the order defines
                the category order everywhere: channels, preference,
                collected vectors.
            preference_alpha: Dirichlet concentration for the per-episode
                preference draw. Below 1 the mass sits toward the simplex
                corners. The floor below compresses the draw toward the
                interior (a dominant weight of 0.8 needs a raw draw of
                0.88), so the concentration is chosen AFTER the floor:
                0.2 with a 0.05 floor gives a dominant category above 0.8
                in ~half the draws, a near-uniform draw (all below 0.5)
                in ~7%, and two categories with real weight in ~60% —
                so the composition term of the reward carries a strong
                signal in most training episodes instead of vanishing on
                episodes where a non-steering collector is already
                on-blend, while balancing several categories still gets
                exercised.
            preference_min: Floor on every component. The Dirichlet draw is
                mapped affinely onto the sub-simplex ``w_c >= preference_min``
                (``w = min + (1 - n*min) * draw``): the draw's shape and the
                RNG consumption are unchanged, the components still sum to
                one, and no category can ever be don't-care — a zero (or
                near-zero) weight makes that category free to ignore, and
                the intent is a policy that always has to account for all
                of them. The most extreme draw is ``1 - (n-1)*min`` on one
                category. Does not apply to ``fixed_preference``.
            preference_onehot_prob: Probability that an episode's preference
                is an exact one-hot vector (uniform over categories) instead
                of the floored Dirichlet draw — the corners a Dirichlet
                never reaches. For the linear architecture benchmark, where
                a one-hot ``w`` makes the reward exactly the single-channel
                one on that category; pair with ``preference_min=0`` there.
            fixed_preference: Pin the preference instead of drawing it
                (normalized to sum to 1) — for evaluation and deployment.
            exclusive_categories: (``per_category`` only; inherent under
                ``v1_partition``) Keep only the dominant category in each
                cell, so risks of different kinds never sit on top of each
                other: a swath cell pays out exactly one category, which
                keeps the portfolio trade-off sharp (imaging terrestrial
                ground is a choice *against* the hydrological risk nearby,
                never both at once).
            kwargs: Forwarded to :class:`SweepRiskField`.
        """
        super().__init__(n_seeds=n_seeds, **kwargs)
        assert len(seed_split) == len(category_land_weights) == len(preference_alpha)
        assert seeding in ("v1_partition", "per_category"), seeding
        assert 0.0 <= preference_min * len(preference_alpha) < 1.0, preference_min
        self.seeding = seeding
        self.seed_split = seed_split
        self.category_land_weights = category_land_weights
        self.preference_alpha = preference_alpha
        self.preference_min = preference_min
        assert 0.0 <= preference_onehot_prob <= 1.0, preference_onehot_prob
        self.preference_onehot_prob = preference_onehot_prob
        self.fixed_preference = fixed_preference
        self.exclusive_categories = exclusive_categories
        self.n_categories = len(category_land_weights)
        self.preference: Optional[np.ndarray] = None
        self._masks = None
        # Seed bookkeeping of the last build (``v1_partition`` only):
        # positions, region code (0 interior, 1 coast band, 2 sea) and
        # category of every seed, for figures and gates.
        self.seed_lats: Optional[np.ndarray] = None
        self.seed_lons: Optional[np.ndarray] = None
        self.seed_region: Optional[np.ndarray] = None
        self.seed_category: Optional[np.ndarray] = None

    def _land_masks(self):
        """(land, coast) bool grids from the Earth texture, or None.

        Same colour test and Antarctica cut as the base ``_seed_weight``,
        but computed once and cached: every category re-weights the same
        masks, and reloading the texture per category per reset would be
        pure waste.
        """
        if self._masks is not None or self.earth_map is None:
            return self._masks
        from matplotlib.image import imread

        img = imread(self.earth_map)[::4, ::4, :3].astype(float)
        if img.max() > 1.5:
            img /= 255.0
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        ocean = (b > r) & (b > g)
        land = ~ocean
        lat_rows = 90.0 - (np.arange(land.shape[0]) + 0.5) * 180.0 / land.shape[0]
        land &= lat_rows[:, None] > -60.0
        coast = land & (uniform_filter(ocean.astype(float), 9) > 0.05)
        self._masks = (land, coast)
        return self._masks

    def _seed_regions(self, lats, lons) -> Optional[np.ndarray]:
        """Region code per seed for labelling: 0 interior, 1 coast band,
        2 sea — the base class's own masks on the grid its sampler uses,
        so "border" means exactly what it meant when the seed was drawn."""
        masks = self._land_masks()
        if masks is None:
            return None
        land, coast = masks
        n_rows, n_cols = land.shape
        rows = np.clip(((90.0 - lats) / 180.0 * n_rows).astype(int), 0, n_rows - 1)
        cols = np.clip(((lons + 180.0) / 360.0 * n_cols).astype(int), 0, n_cols - 1)
        on_land, on_coast = land[rows, cols], coast[rows, cols]
        return np.where(~on_land, 2, np.where(on_coast, 1, 0))

    def _partition_seeds(self, region: Optional[np.ndarray], n: int,
                         rng: np.random.Generator) -> np.ndarray:
        """Category per seed, equal counts, geography-first.

        Hydrological takes a third of the coast-band seeds (all of them,
        topped up from the interior, should the band ever hold fewer than a
        third); military takes everything at sea, then the coastal surplus,
        then interior seeds, up to a third; terrestrial is the rest —
        interior seeds only, as long as sea plus coastal surplus stay below
        a third, which the base weights guarantee (8% + ~18%). The
        permutations consume the RNG only AFTER the base sampler, so the
        seed positions stay the base class's for the same generator state.
        Without an Earth map there is no geography: the split is random and
        equal.
        """
        n_h, n_m = n // 3, n // 3
        category = np.zeros(n, dtype=int)
        if region is None:
            perm = rng.permutation(n)
            category[perm[:n_h]] = 1
            category[perm[n_h:n_h + n_m]] = 2
            return category
        coast = rng.permutation(np.flatnonzero(region == 1))
        sea = rng.permutation(np.flatnonzero(region == 2))
        interior = rng.permutation(np.flatnonzero(region == 0))
        hydro, coast_surplus = coast[:n_h], coast[n_h:]
        short = n_h - hydro.size
        if short > 0:
            hydro = np.concatenate([hydro, interior[:short]])
            interior = interior[short:]
        pool = np.concatenate([sea, coast_surplus, interior])
        category[hydro] = 1
        category[pool[:n_m]] = 2
        return category

    def _seed_weight_cat(self, c: int) -> np.ndarray:
        """Seed density grid for category ``c``, from its weight triple."""
        masks = self._land_masks()
        if masks is None:
            return np.ones((90, 180))
        land, coast = masks
        w = self.category_land_weights[c]
        weight = np.where(land, w["land"], w["ocean"])
        weight[coast] = w["coast"]
        return weight

    def _sample_seeds_n(self, rng: np.random.Generator, n: int, weight: np.ndarray):
        """Base rejection sampler, parameterized by count and density."""
        w_max = weight.max()
        n_rows, n_cols = weight.shape

        lats = np.empty(n)
        lons = np.empty(n)
        n_kept = 0
        while n_kept < n:
            cand_lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 4 * n)))
            cand_lon = rng.uniform(-180.0, 180.0, 4 * n)
            rows = np.clip(
                ((90.0 - cand_lat) / 180.0 * n_rows).astype(int), 0, n_rows - 1
            )
            cols = np.clip(
                ((cand_lon + 180.0) / 360.0 * n_cols).astype(int), 0, n_cols - 1
            )
            keep = rng.uniform(0.0, w_max, cand_lat.size) < weight[rows, cols]
            take = min(int(keep.sum()), n - n_kept)
            lats[n_kept:n_kept + take] = cand_lat[keep][:take]
            lons[n_kept:n_kept + take] = cand_lon[keep][:take]
            n_kept += take

        cores = rng.uniform(*self.core_radius_range, n)
        falloffs = rng.uniform(*self.falloff_range, n)
        peaks = rng.uniform(*self.peak_range, n)
        return lats, lons, cores, falloffs, peaks

    def build_risk_field(self, rng: np.random.Generator) -> None:
        """Dispatch on ``seeding`` (constructor)."""
        if self.seeding == "per_category":
            self._build_per_category(rng)
        else:
            self._build_v1_partition(rng)

    def _note_seed_contribution(self, seed_idx: int, category: int,
                                contribution: np.ndarray) -> None:
        """Per-seed hook of :meth:`_build_v1_partition` (no-op here)."""

    def _build_v1_partition(self, rng: np.random.Generator) -> None:
        """The base field, partitioned into channels by seed label.

        Reproduces :meth:`SweepRiskField.build_risk_field` operation for
        operation — background draw, then the base sampler, then the
        probabilistic union in seed order — so the union is the base field
        bit for bit for the same generator state. Alongside, each channel
        accumulates the union of its own seeds; the channel that dominates a
        cell claims that cell's base value, the others get zero. Labels are
        assigned from the seed positions before the loop (they consume the
        RNG after the base sampler is done, leaving the positions intact).
        """
        res = self.resolution_deg
        self.lats = np.arange(-90.0 + res / 2, 90.0, res)
        self.lons = np.arange(-180.0 + res / 2, 180.0, res)
        lon_g, lat_g = np.meshgrid(self.lons, self.lats)

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
        seed_lats, seed_lons, cores, falloffs, peaks = self._sample_seeds(rng)
        region = self._seed_regions(seed_lats, seed_lons)
        category = self._partition_seeds(region, seed_lats.size, rng)

        miss = 1.0 - risk
        miss_c = [np.ones_like(miss) for _ in range(self.n_categories)]
        for s, (lat0, lon0, core, falloff, peak, c) in enumerate(zip(
            seed_lats, seed_lons, cores, falloffs, peaks, category
        )):
            d = _great_circle_km(lat_g, lon_g, lat0, lon0)
            ramp = 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * np.clip((d - core) / max(falloff - core, 1e-6), 0.0, 1.0)
                )
            )
            factor = 1.0 - peak * ramp
            miss *= factor
            miss_c[c] *= factor
            # Hook for subclasses that attribute cells to spots (the quota
            # field): sees every seed's contribution in the same pass, so
            # no second sweep over the raster is needed. No-op here, and it
            # touches nothing the union depends on.
            self._note_seed_contribution(s, c, peak * ramp)
        union = 1.0 - miss
        strength = np.stack([1.0 - m for m in miss_c])
        winner = np.argmax(strength, axis=0)
        keep = winner[None] == np.arange(self.n_categories)[:, None, None]
        self.risk = np.where(keep, union, 0.0)
        self.seed_lats, self.seed_lons = seed_lats, seed_lons
        self.seed_region, self.seed_category = region, category

    def _build_per_category(self, rng: np.random.Generator) -> None:
        """Build one probabilistic-union channel per category."""
        res = self.resolution_deg
        self.lats = np.arange(-90.0 + res / 2, 90.0, res)
        self.lons = np.arange(-180.0 + res / 2, 180.0, res)
        lon_g, lat_g = np.meshgrid(self.lons, self.lats)

        channels = []
        for c in range(self.n_categories):
            n_c = int(round(self.n_seeds * self.seed_split[c]))
            # Ambient background per channel, as in the base class (off by
            # default: background_max=0).
            coarse = rng.uniform(
                0.0,
                self.background_max,
                (max(self.lats.size // 16, 1), max(self.lons.size // 16, 1)),
            )
            risk = np.clip(
                zoom(
                    coarse,
                    (
                        self.lats.size / coarse.shape[0],
                        self.lons.size / coarse.shape[1],
                    ),
                    order=3,
                    mode="reflect",
                ),
                0.0,
                self.background_max,
            )
            miss = 1.0 - risk
            seeds = self._sample_seeds_n(rng, n_c, self._seed_weight_cat(c))
            for lat0, lon0, core, falloff, peak in zip(*seeds):
                d = _great_circle_km(lat_g, lon_g, lat0, lon0)
                ramp = 0.5 * (
                    1.0
                    + np.cos(
                        np.pi
                        * np.clip((d - core) / max(falloff - core, 1e-6), 0.0, 1.0)
                    )
                )
                miss *= 1.0 - peak * ramp
            channels.append(1.0 - miss)
        self.risk = np.stack(channels)
        if self.exclusive_categories:
            # Winner-take-all per cell: where spots of different categories
            # landed on the same ground, only the dominant one survives.
            # Applied at the raster level rather than during seed placement,
            # so the geographic densities and the RNG draw order stay
            # untouched; the price is a hard edge where two spots met (one
            # ramp is truncated at the crossover line).
            winner = np.argmax(self.risk, axis=0)
            keep = winner[None] == np.arange(self.n_categories)[:, None, None]
            self.risk = np.where(keep, self.risk, 0.0)

    def reset_overwrite_previous(self) -> None:
        """Regenerate the field, then fix shapes and draw the preference."""
        super().reset_overwrite_previous()
        # The base reset sizes `covered` off `risk.shape`, which is now
        # (C, n_lat, n_lon); coverage is a property of the ground, not of a
        # channel, so it must stay 2-D or the swath indexing breaks.
        self.covered = np.zeros(self.risk.shape[1:], dtype=bool)
        if self.fixed_preference is not None:
            p = np.asarray(self.fixed_preference, dtype=float)
            self.preference = p / p.sum()
        else:
            # Global np.random: seeded by the env right before this call
            # (gym reset), so the draw is reproducible per episode seed. All
            # three draws happen on every reset, whichever branch is taken,
            # so the RNG stream after this point is identical for any
            # ``preference_onehot_prob`` — episodes stay paired across
            # settings of it.
            draw = np.random.dirichlet(self.preference_alpha)
            n = draw.size
            use_corner = np.random.uniform() < self.preference_onehot_prob
            corner = np.random.randint(n)
            if use_corner:
                self.preference = np.eye(n)[corner]
            else:
                self.preference = (
                    self.preference_min + (1.0 - n * self.preference_min) * draw
                )
        logger.info(f"Preference drawn: {np.round(self.preference, 3)}")

    def update_coverage(self, satellite) -> np.ndarray:
        """Consume risk under the swath, tallied per category.

        Same swath geometry and idempotency as the base class; the collected
        total is a vector over categories, and a swept cell is zeroed in
        every channel at once (one image captures whatever is there).

        Returns:
            [risk * km^2] Per-category cumulative collected this episode.
        """
        state = self._coverage.setdefault(
            satellite.name,
            {"idx": 1, "collected": np.zeros(self.n_categories)},
        )
        cells = self._new_swath_cells(satellite, state)
        if cells is not None:
            ri, ci, cell_km2 = cells
            state["collected"] += (self.risk[:, ri, ci] * cell_km2).sum(axis=1)
            self.risk[:, ri, ci] = 0.0
            self.covered[ri, ci] = True
        # A copy, not the live array: the DataStore keeps the returned object
        # as its log state, and mutating it in place would make every
        # compare_log_states delta zero — silently killing all reward.
        return state["collected"].copy()

    def sample_risk(self, lat, lon) -> np.ndarray:
        """Bilinear sample of every channel, periodic in longitude.

        Returns:
            Channel-LAST array, shape ``lat.shape + (n_categories,)``. The
            trailing axis keeps ``RiskTokens.rectified_map`` working
            untouched: its ``.mean(axis=2)`` still collapses the
            swath-sample axis, leaving the category axis intact.
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
        lo = risk[:, i0, j0] * (1.0 - wj) + risk[:, i0, j1] * wj
        hi = risk[:, i0 + 1, j0] * (1.0 - wj) + risk[:, i0 + 1, j1] * wj
        return np.moveaxis(lo * (1.0 - wi) + hi * wi, 0, -1)


__doc_title__ = "Categorized Sweep Risk Field"
__all__ = ["CategorizedSweepRiskField", "CATEGORY_NAMES"]
