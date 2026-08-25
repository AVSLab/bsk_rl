"""Per-category sweep data and its preference-conditioned rewards.

Vector counterpart of :mod:`bsk_rl.data.swept_risk_data` for
:class:`~bsk_rl.scene.CategorizedSweepRiskField`: the collected risk-area is
tracked per category, and the reward is conditioned on the scenario's
per-episode preference vector ``w``. Two readings of ``w`` are implemented:

* :class:`DiminishingSweptRiskReward` — ``w`` is a *budget share*: each
  category has its own saturating value curve and ``w_c`` prices its
  increments, so the marginal rate of a category decays as it is collected.
* :class:`ProportionalSweptRiskReward` — ``w`` is a *required composition*:
  the operator wants the largest possible total risk-area collected *in the
  proportions* ``w``, so the reward pays the increment of a CES blend total
  that discounts whatever is collected off-blend.

Both are concave in the cumulative vector, which is what makes the task
irreducible to sweeping any single reweighted map: the marginal price of a
pixel depends on the agent's own collection history.
"""

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward

# [risk * km^2] Kept in sync with the calibrated envs_3.S_REF (which is
# passed explicitly everywhere; this default is only a fallback).
S_REF_DEFAULT = 1.0e6

# Defaults for ProportionalSweptRiskReward, mirrored by envs_3.RHO / U_REF
# (passed explicitly there; these are only fallbacks).
RHO_DEFAULT = -4.0
U_REF_DEFAULT = 3.0e6


def blend_total(swept, preference, rho: float) -> float:
    """CES blend total of a per-category collected vector.

    .. math::

        U(S) = \\Big(\\sum_{c: w_c > 0} w_c (S_c / w_c)^\\rho\\Big)^{1/\\rho}

    With ``w`` on the simplex and ``rho <= 1``, the generalized-mean
    inequality gives ``U <= sum_c S_c`` with equality exactly when the
    collected shares match ``w`` — so ``U`` is the total collected priority,
    discounted for compositional drift away from the operator's proportions.
    ``rho = 1`` is the plain sum (proportions ignored); ``rho = 0`` is
    evaluated as its Cobb-Douglas limit, the weighted geometric mean
    ``prod_c (S_c / w_c)^(w_c)``; ``rho -> -inf`` is the Leontief
    ``min_c S_c / w_c`` (off-blend surplus worthless). For ``rho <= 0``,
    ``U = 0`` until every required category holds something — a delivery in
    the blend needs all of its components. Zero-weight categories are
    "don't care": excluded from the aggregate, their collection neither
    paid nor punished.
    """
    S = np.asarray(swept, dtype=float)
    w = np.asarray(preference, dtype=float)
    active = w > 0.0
    if not np.any(active):
        return 0.0
    x = S[active] / w[active]
    if rho == 1.0:
        return float(w[active] @ x)
    if rho == 0.0:
        if np.any(x <= 0.0):
            return 0.0
        return float(np.exp(w[active] @ np.log(x)))
    # Factor out the dominant fill level so x^rho never overflows: for
    # rho < 0 the smallest fill dominates and x/m >= 1, for 0 < rho < 1 the
    # largest does and x/m <= 1; either way (x/m)^rho stays in (0, 1].
    m = float(x.min()) if rho < 0.0 else float(x.max())
    if m <= 0.0:
        return 0.0
    z = (x / m) ** rho
    return float(m * (w[active] @ z) ** (1.0 / rho))


def blend_rates(swept, preference, rho: float, x_floor: float = 1.0):
    """Marginal value of one risk-area unit per category, ``dU/dS_c``.

    Differentiating :func:`blend_total` gives ``(U / x_c)^(1 - rho)`` with
    ``x_c = S_c / w_c``: exactly 1 for every active category when the
    collected shares match ``w`` (on-blend, priority trades at face value),
    above 1 for categories lagging their share and below 1 for those ahead.
    Zero for zero-weight categories. Fill levels are floored at ``x_floor``
    [risk * km^2] — negligible against any realistic collection — so the
    empty-category limit is finite and an all-zero start prices every
    active category at exactly face value.
    """
    S = np.asarray(swept, dtype=float)
    w = np.asarray(preference, dtype=float)
    rates = np.zeros_like(w)
    active = w > 0.0
    if not np.any(active):
        return rates
    if rho == 1.0:
        rates[active] = 1.0
        return rates
    x = np.maximum(S[active] / w[active], x_floor)
    if rho == 0.0:
        U = np.exp(w[active] @ np.log(x))
        rates[active] = U / x
        return rates
    m = float(x.min()) if rho < 0.0 else float(x.max())
    z = (x / m) ** rho
    U = m * (w[active] @ z) ** (1.0 / rho)
    rates[active] = (U / x) ** (1.0 - rho)
    return rates


class CategorizedSweptRisk(Data):
    """Risk-area collected by the image swath, per category."""

    def __init__(self, swept=None, n_categories: int = 3) -> None:
        """Per-category collected risk-area.

        Args:
            swept: [risk * km^2] Collected risk-area per category. The
                bare constructor must yield zeros: ``DataStore.__init__``
                and ``Data.__copy__`` both call ``data_type()`` with no
                arguments.
            n_categories: Vector length when ``swept`` is not given.
        """
        self.swept = (
            np.zeros(n_categories) if swept is None
            else np.asarray(swept, dtype=float)
        )

    def __add__(self, other: "CategorizedSweptRisk") -> "CategorizedSweptRisk":
        """Define the combination of two units of data."""
        return self.__class__(self.swept + other.swept)


class CategorizedSweptRiskStore(DataStore):
    """DataStore for per-category risk collected by the image swath.

    The coverage bookkeeping lives in
    :class:`~bsk_rl.scene.CategorizedSweepRiskField.update_coverage`, which
    already returns a *copy* of its per-category cumulative vector — the
    store keeps the returned object as its log state across steps, so a live
    reference would alias old and new states and zero every delta.
    """

    data_type = CategorizedSweptRisk

    def get_log_state(self) -> np.ndarray:
        """Per-category cumulative risk-area collected since episode start."""
        return np.asarray(
            self.satellite.sweep_scene.update_coverage(self.satellite)
        )

    def compare_log_states(self, old_state, new_state) -> "CategorizedSweptRisk":
        """New data from the change in per-category collected risk-area."""
        return CategorizedSweptRisk(np.maximum(new_state - old_state, 0.0))


class DiminishingSweptRiskReward(GlobalReward):
    """Preference-weighted reward with per-category diminishing returns.

    For a step collecting ``delta_c`` on pre-step cumulative ``S_c``:

    .. math::

        r = \\sum_c w_c \\, [f(S_c + \\delta_c) - f(S_c)] / S_{ref},
        \\qquad f(S) = S_{ref} (1 - e^{-S / S_{ref}})

    with ``w`` the scenario's preference vector. The effective per-km² rate
    of category ``c`` is ``w_c * exp(-S_c / S_ref)``: it starts at the
    preference weight and decays as the category saturates, so the optimal
    policy balances categories instead of maximizing one blend. With the
    ``1/S_ref`` normalization the undiscounted episode return is the
    weighted value fraction ``sum_c w_c (1 - exp(-S_c / S_ref))`` in [0, 1).

    Use with :class:`~bsk_rl.scene.CategorizedSweepRiskField` and a sweep
    satellite (:class:`~bsk_rl.sim.dyn.SweepDynModel`,
    :class:`~bsk_rl.sim.fsw.SweepFSWModel`, :class:`~bsk_rl.act.Sweep`).
    Single-satellite only: with several stores contributing in one step the
    curve increments would each be taken from the same pre-step ``S``.
    """

    datastore_type = CategorizedSweptRiskStore

    def __init__(self, s_ref: float = S_REF_DEFAULT) -> None:
        """Diminishing-returns preference reward.

        Args:
            s_ref: [risk * km^2] Saturation scale of the value curve. The
                marginal rate of a category falls to 1/e of its preference
                weight once ``s_ref`` of it has been collected; calibrate
                to roughly half a good policy's per-category haul so the
                curvature binds mid-episode.
        """
        super().__init__()
        self.s_ref = s_ref

    def _f(self, S: np.ndarray) -> np.ndarray:
        return self.s_ref * (1.0 - np.exp(-S / self.s_ref))

    def initial_data(self, satellite) -> "CategorizedSweptRisk":
        """Start each satellite with zero collected risk in every category."""
        return CategorizedSweptRisk()

    def calculate_reward(self, new_data_dict) -> dict[str, float]:
        """Preference-weighted saturating-curve increment for the step.

        ``GlobalReward.reward`` calls this *before* merging the new data
        into ``self.data`` (base.py), so ``self.data.swept`` is exactly the
        pre-step cumulative the curve increment must be taken from.
        """
        w = np.asarray(self.scenario.preference)
        S = self.data.swept
        return {
            sat: float(np.sum(w * (self._f(S + d.swept) - self._f(S))))
            / self.s_ref
            for sat, d in new_data_dict.items()
        }


class ProportionalSweptRiskReward(GlobalReward):
    """Maximize total collected priority in the operator's proportions.

    For a step collecting ``delta_c`` on pre-step cumulative ``S_c``:

    .. math::

        r = [U(S + \\delta) - U(S)] / u_{ref},
        \\qquad U(S) = \\Big(\\sum_{c: w_c > 0} w_c (S_c / w_c)^\\rho
        \\Big)^{1/\\rho}

    with ``w`` the scenario's preference vector, here read as the operator's
    *required composition* of the collection. ``U`` equals the plain sum
    ``sum_c S_c`` exactly when the collected shares match ``w`` and is
    strictly smaller otherwise (see :func:`blend_total`), so the undiscounted
    episode return ``U(S_final) / u_ref`` is the normalized total priority
    net of an off-blend discount — "maximize the sum, respect the
    proportions" in one concave objective. ``rho`` sets how binding the
    proportions are: 1 ignores them, 0 is the Cobb-Douglas geometric-mean
    limit (the mildest curvature that still couples the categories),
    ``-> -inf`` makes them hard (Leontief min), and the default trades
    near face value on-blend while hugging the min under large drift.

    The marginal rate of category ``c`` is ``(U / x_c)^(1 - rho) / u_ref``
    with ``x_c = S_c / w_c`` (:func:`blend_rates`): unity on-blend,
    amplified for lagging categories, crushed for leading ones. The rates
    depend on the cumulative fills, so the optimal policy must anticipate
    the corridor's upcoming composition rather than sweep any fixed
    reweighted map. For ``rho < 0`` the reward stays zero until every
    required category has been touched — the honest blend behavior, and a
    few decision steps at most under any policy on a dense field.

    Use with :class:`~bsk_rl.scene.CategorizedSweepRiskField` and a sweep
    satellite (:class:`~bsk_rl.sim.dyn.SweepDynModel`,
    :class:`~bsk_rl.sim.fsw.SweepFSWModel`, :class:`~bsk_rl.act.Sweep`).
    Single-satellite only: with several stores contributing in one step the
    blend increments would each be taken from the same pre-step ``S``.
    """

    datastore_type = CategorizedSweptRiskStore

    def __init__(
        self, rho: float = RHO_DEFAULT, u_ref: float = U_REF_DEFAULT
    ) -> None:
        """Proportional blend-total reward.

        Args:
            rho: CES exponent, ``rho <= 1`` (concavity); 0 is evaluated as
                the Cobb-Douglas geometric-mean limit. Controls how
                strictly the composition binds — see the class docstring.
            u_ref: [risk * km^2] Return normalization only (no effect on
                the optimal policy). Calibrate to a good policy's blend
                total so episode returns land near 1.
        """
        super().__init__()
        assert rho <= 1.0, rho
        self.rho = rho
        self.u_ref = u_ref

    def blend_total(self, S) -> float:
        """The blend total ``U(S)`` under the episode's preference."""
        return blend_total(S, self.scenario.preference, self.rho)

    def initial_data(self, satellite) -> "CategorizedSweptRisk":
        """Start each satellite with zero collected risk in every category."""
        return CategorizedSweptRisk()

    def calculate_reward(self, new_data_dict) -> dict[str, float]:
        """Normalized blend-total increment for the step.

        ``GlobalReward.reward`` calls this *before* merging the new data
        into ``self.data`` (base.py), so ``self.data.swept`` is exactly the
        pre-step cumulative the increment must be taken from; the merge then
        performs the identical addition, so the increments telescope to
        ``U(S_final) / u_ref`` exactly.
        """
        S = self.data.swept
        return {
            sat: (self.blend_total(S + d.swept) - self.blend_total(S))
            / self.u_ref
            for sat, d in new_data_dict.items()
        }


__doc_title__ = "Categorized Swept Risk"
__all__ = [
    "DiminishingSweptRiskReward",
    "ProportionalSweptRiskReward",
    "CategorizedSweptRiskStore",
    "CategorizedSweptRisk",
    "S_REF_DEFAULT",
    "RHO_DEFAULT",
    "U_REF_DEFAULT",
    "blend_total",
    "blend_rates",
]
