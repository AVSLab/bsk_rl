"""Shared AMOS target-population samplers for baselines and learned environments.

The completed three-sensor baseline artifacts are immutable.  This module moves
their sampling equations into one reusable location so the new learned pilot can
use the same altitude/eccentricity/inclination distributions without reading or
modifying those artifacts.
"""

from __future__ import annotations

import numpy as np
from Basilisk.utilities import macros, orbitalMotion


R_EARTH_M = 6371e3
ALTITUDE_BANDS_M = {
    "LEO": (400e3, 2000e3),
    "MEO": (2000e3, 35000e3),
    "GEO": (35786e3 - 300e3, 35786e3 + 300e3),
}
TARGET_POPULATIONS = ("all_leo", "mixed_50_30_20")


def exact_regimes(count: int, population: str, seed: int) -> list[str]:
    """Return a seeded target-ID assignment with exact largest-remainder counts.

    For 100 targets, ``mixed_50_30_20`` always contains exactly 50 LEO, 30 MEO,
    and 20 GEO spacecraft.  The independent generator makes regime assignment a
    pure function of episode seed instead of the order of unrelated NumPy draws.
    """
    if population in {"leo", "all_leo"}:  # ``leo`` keeps old manifests readable.
        return ["LEO"] * int(count)
    if population not in {"mixed", "mixed_50_30_20"}:
        raise ValueError(
            "target_population must be all_leo or mixed_50_30_20."
        )
    raw = int(count) * np.asarray([0.5, 0.3, 0.2])
    counts = np.floor(raw).astype(int)
    order = sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[: int(count) - int(counts.sum())]:
        counts[index] += 1
    names = [regime for regime, number in zip(ALTITUDE_BANDS_M, counts) for _ in range(number)]
    return list(np.random.default_rng(int(seed)).permutation(names))


def sample_orbit(regime: str) -> orbitalMotion.ClassicElements:
    """Draw one live target spacecraft orbit from the baseline distribution.

    The caller supplies determinism by seeding NumPy at environment reset.  The
    perigee guard is intentionally identical to the completed baseline campaign.
    """
    if regime not in ALTITUDE_BANDS_M:
        raise ValueError(f"Unknown target regime: {regime!r}")
    orbit = orbitalMotion.ClassicElements()
    orbit.a = R_EARTH_M + np.random.uniform(*ALTITUDE_BANDS_M[regime])
    e_max = {"LEO": 0.02, "MEO": 0.10, "GEO": 0.0015}[regime]
    orbit.e = np.random.uniform(0.0, e_max)
    while orbit.a * (1 - orbit.e) < R_EARTH_M + 400e3:
        orbit.e = np.random.uniform(0.0, e_max)
    orbit.i = (
        np.random.uniform(0.0, {"LEO": 180, "MEO": 120, "GEO": 15}[regime])
        * macros.D2R
    )
    orbit.Omega, orbit.omega, orbit.f = (
        np.random.uniform(0.0, 360.0, 3) * macros.D2R
    )
    return orbit


__all__ = [
    "ALTITUDE_BANDS_M",
    "R_EARTH_M",
    "TARGET_POPULATIONS",
    "exact_regimes",
    "sample_orbit",
]
