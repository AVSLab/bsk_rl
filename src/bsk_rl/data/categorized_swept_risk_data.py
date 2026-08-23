"""Per-category sweep data and the diminishing-returns preference reward.

Vector counterpart of :mod:`bsk_rl.data.swept_risk_data` for
:class:`~bsk_rl.scene.CategorizedSweepRiskField`: the collected risk-area is
tracked per category, and the reward pays the *increment of a saturating
value curve* per category, weighted by the scenario's per-episode preference
vector. A category's marginal rate therefore decays as it is collected —
"50% military, 50% terrestrial" is a budget share, not a constant price — which is
what makes the task irreducible to sweeping any single reweighted map.
"""

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward

# [risk * km^2] Kept in sync with the calibrated envs_3.S_REF (which is
# passed explicitly everywhere; this default is only a fallback).
S_REF_DEFAULT = 1.0e6


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


__doc_title__ = "Categorized Swept Risk"
__all__ = [
    "DiminishingSweptRiskReward",
    "CategorizedSweptRiskStore",
    "CategorizedSweptRisk",
    "S_REF_DEFAULT",
]
