"""Categorized risk field with replenishing image quotas per category.

Composition as a *resource* rather than a discount: every category owns a
token bucket. A swept cell whose category-``c`` priority exceeds
``image_tau`` **consumes** from budget ``b_c`` and **is paid its priority
only while budget remains**; the bucket refills continuously at ``quota_c``
per ``quota_period_s`` and is capped at one period's worth, with
``quota_c = w_c * total`` — the operator's preference *is* the quota split,
and a zero weight is a zero quota (that category's images are worth
nothing). Earning and consuming are the same event, cell by cell, so
nothing is gained by grazing a spot's edge or dodging its centre.

Two currencies for the buckets (``quota_unit``):

* ``"spots"`` (default) — tokens are *images*: one token per hot spot.
  Every cell is attributed to the spot that dominates it, and a swept cell
  costs the fraction of its spot's priority mass (above ``image_tau``) that
  it carries, so a spot fully imaged costs exactly one token and a fringe
  chord through a large spot costs the fraction of the spot it captured.
  Token cost is flat per spot while spot value varies by more than an
  order of magnitude (peaks 0.1-1, falloff radii 10-500 km), so a policy
  that spends tokens on rich spots and skips faint ones is paid much more
  than a collector that burns them on whatever it crosses first — the
  selection headroom the quota benchmark is built to reward.
* ``"area"`` — tokens are km^2 of category ground: a cell costs its area.
  Kept for comparison; its headroom is small because priority density
  varies only ~2x between fringe and core, so "spend on the denser ground"
  buys little over plain collection.

The threshold is symmetric — cells at or below ``image_tau`` neither pay
nor consume — because most category ground is faint fringe (falloff radii
reach 500 km) that the imager, which never switches off, crosses whatever
the policy does (see ``avs_rl_tools/sweep_imaging/quota_sizing_3.py``).
Within one decision step the swept cells are charged in swath order, the
cell on which a bucket runs dry being paid for the fraction the remaining
budget covers — exact cell-by-cell accounting.

:meth:`update_coverage` returns the cumulative PAID priority per category,
so a :class:`~bsk_rl.data.CategorizedSweptRiskStore` on top of it yields
paid-priority increments and :class:`~bsk_rl.data.QuotaPriorityReward`
just scales them. Unpaid priority, consumed tokens/area and the live
budgets stay on the scenario for the observation and the episode metrics.
"""

import numpy as np

from bsk_rl.scene.categorized_risk_field import CategorizedSweepRiskField
from bsk_rl.scene.risk_field import R_EARTH_KM


class QuotaSweepRiskField(CategorizedSweepRiskField):
    """Categorized field whose categories are budgeted in images or area."""

    def __init__(
        self,
        image_tau: float = 0.2,
        quota_unit: str = "spots",
        quota_total_images: float = 20.0,
        quota_total_km2: float = 1.0e6,
        quota_period_s: float = 6053.0,
        **kwargs,
    ) -> None:
        """Quota field.

        Args:
            image_tau: Risk level a cell must exceed, in its category's
                channel, to count as (part of) an image: it then consumes
                budget and is paid; below it neither. 0 is the pure rule.
            quota_unit: ``"spots"`` (tokens = images, one per hot spot,
                fractional per cell) or ``"area"`` (tokens = km^2).
            quota_total_images: [images] Total quota per period under
                ``"spots"``, split by the preference. Sized against how
                many spot-equivalents the reference policies image per
                orbit so that quotas bind for a pure collector but stay
                reachable by a selective policy.
            quota_total_km2: [km^2] Total quota per period under ``"area"``.
            quota_period_s: [s] Refill period; the default is one orbit of
                the 800 km SSO. Budgets refill continuously at
                ``quota_c / period`` and are capped at ``quota_c``.
            kwargs: Forwarded to :class:`CategorizedSweepRiskField`.
        """
        super().__init__(**kwargs)
        assert image_tau >= 0.0 and quota_period_s > 0.0
        assert quota_unit in ("spots", "area"), quota_unit
        assert quota_total_images > 0.0 and quota_total_km2 > 0.0
        self.image_tau = image_tau
        self.quota_unit = quota_unit
        self.quota_total_images = quota_total_images
        self.quota_total_km2 = quota_total_km2
        self.quota_period_s = quota_period_s
        self.quota = None
        self.budget = None
        self._quota_time = 0.0
        # Spot attribution of the last build (spots mode).
        self._best_val = None
        self._best_seed = None
        self.cell_seed = None
        self.cell_area = None
        self.spot_mass = None

    @property
    def quota_total(self) -> float:
        """Total quota per period in the bucket's own unit."""
        return (self.quota_total_images if self.quota_unit == "spots"
                else self.quota_total_km2)

    # --- build: attribute every cell to its dominant spot -------------

    def _note_seed_contribution(self, seed_idx, category, contribution):
        """Track, per channel, the seed contributing most to each cell."""
        if self._best_val is None or self._best_val.shape[1:] != contribution.shape:
            shape = (self.n_categories,) + contribution.shape
            self._best_val = np.zeros(shape)
            self._best_seed = np.full(shape, -1, dtype=int)
        better = contribution > self._best_val[category]
        self._best_val[category][better] = contribution[better]
        self._best_seed[category][better] = seed_idx

    def _build_v1_partition(self, rng: np.random.Generator) -> None:
        """The parent's build, then the spot attribution and masses."""
        self._best_val = None
        self._best_seed = None
        super()._build_v1_partition(rng)
        union = self.risk.sum(axis=0)                  # == the v1 field
        winner = np.argmax(self.risk, axis=0)
        self.cell_seed = np.take_along_axis(
            self._best_seed, winner[None], axis=0)[0]
        self.cell_seed[union <= 0.0] = -1
        res = self.resolution_deg
        self.cell_area = (
            R_EARTH_KM ** 2 * np.radians(res) ** 2
            * np.cos(np.radians(self.lats))[:, None]
            * np.ones(self.lons.size)[None, :]
        )
        counted = (union > self.image_tau) & (self.cell_seed >= 0)
        self.spot_mass = np.bincount(
            self.cell_seed[counted],
            weights=(union * self.cell_area)[counted],
            minlength=self.seed_category.size,
        )
        # Attribution arrays are only needed for the masses; drop them.
        self._best_val = None
        self._best_seed = None

    def _build_per_category(self, rng: np.random.Generator) -> None:
        raise NotImplementedError(
            "QuotaSweepRiskField needs the v1_partition seeding: spot "
            "attribution relies on its per-seed hook."
        )

    # --- episode -----------------------------------------------------

    def reset_overwrite_previous(self) -> None:
        """Field + preference as the parent, then full buckets."""
        super().reset_overwrite_previous()
        self.quota = np.asarray(self.preference, dtype=float) * self.quota_total
        self.budget = self.quota.copy()
        self._quota_time = 0.0

    @property
    def budget_fraction(self) -> np.ndarray:
        """Remaining budget as a fraction of the quota (0 for a zero quota)."""
        return np.where(self.quota > 0.0,
                        self.budget / np.maximum(self.quota, 1e-12), 0.0)

    def update_coverage(self, satellite) -> np.ndarray:
        """Consume budget and pay priority under the swath.

        Same swath geometry and idempotency as the parent; on top, the
        token buckets are refilled for the elapsed time, then charged, cell
        by cell in swath order, for the category ground above
        ``image_tau`` in the newly imaged cells.

        Returns:
            [risk * km^2] Per-category cumulative PAID priority this
            episode (a copy — the DataStore keeps it as its log state).
        """
        state = self._coverage.setdefault(
            satellite.name,
            {
                "idx": 1,
                "collected": np.zeros(self.n_categories),   # all priority
                "paid": np.zeros(self.n_categories),        # within quota
                "unpaid": np.zeros(self.n_categories),      # beyond quota
                "tokens": np.zeros(self.n_categories),      # consumed
                "area": np.zeros(self.n_categories),        # km^2 counted
            },
        )
        # Refill first: budget accrues over the flight time since the last
        # update, whether or not anything was imaged.
        now = float(satellite.simulator.sim_time)
        dt = max(now - self._quota_time, 0.0)
        self._quota_time = now
        self.budget = np.minimum(
            self.quota, self.budget + self.quota * dt / self.quota_period_s
        )

        cells = self._new_swath_cells(satellite, state)
        if cells is not None:
            ri, ci, cell_km2 = cells
            values = self.risk[:, ri, ci]                      # (C, n)
            km2 = np.broadcast_to(np.asarray(cell_km2, dtype=float),
                                  values.shape[1:])
            state["collected"] += (values * km2).sum(axis=1)
            cat = np.argmax(values, axis=0)                    # (n,)
            v = values[cat, np.arange(values.shape[1])]
            priority = v * km2
            counted = v > self.image_tau
            if self.quota_unit == "spots":
                s = self.cell_seed[ri, ci]
                mass = np.where(s >= 0, self.spot_mass[np.maximum(s, 0)], 0.0)
                counted &= mass > 0.0
                cost = np.where(counted, priority / np.maximum(mass, 1e-12), 0.0)
            else:
                cost = np.where(counted, km2, 0.0)
            for c in range(self.n_categories):
                sel = np.flatnonzero((cat == c) & counted)
                if sel.size == 0:
                    continue
                cum = np.cumsum(cost[sel])
                before = cum - cost[sel]
                frac = np.clip((self.budget[c] - before) / cost[sel], 0.0, 1.0)
                state["paid"][c] += float(np.sum(frac * priority[sel]))
                state["unpaid"][c] += float(np.sum((1.0 - frac) * priority[sel]))
                state["tokens"][c] += float(min(self.budget[c], cum[-1]))
                state["area"][c] += float(np.sum(km2[sel]))
                self.budget[c] = max(0.0, self.budget[c] - cum[-1])
            self.risk[:, ri, ci] = 0.0
            self.covered[ri, ci] = True
        return state["paid"].copy()

    def quota_state(self, satellite_name: str) -> dict:
        """The episode's quota bookkeeping for ``satellite_name`` (copies)."""
        state = self._coverage.get(satellite_name)
        if state is None:
            z = np.zeros(self.n_categories)
            return dict(collected=z, paid=z.copy(), unpaid=z.copy(),
                        tokens=z.copy(), area=z.copy())
        return {k: np.asarray(v, dtype=float).copy()
                for k, v in state.items() if k != "idx"}


__doc_title__ = "Quota Sweep Risk Field"
__all__ = ["QuotaSweepRiskField"]
