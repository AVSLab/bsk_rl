"""Earth-centered passive orbits, with no simulator or policy dependencies.

SI units, inertial Cartesian states. J2 is the unnormalized positive zonal
coefficient. GGM03S uses fully normalized C20=-4.841692638330e-4, so
J2=-sqrt(5)*C20. All other harmonics, Sun acceleration and drag are absent.
The J2 symmetry axis is the Earth-fixed z axis expressed in inertial coordinates.
"""

import numpy as np

# The Basilisk Earth body overwrites the gravity-file header's mu/radius at
# initializeParameters(body). Match the actual Earth body values, not the header.
EARTH_MU = 3.98600436e14
EARTH_RADIUS = 6378136.6
EARTH_C20 = -4.841692638330e-4
EARTH_J2 = -np.sqrt(5.0) * EARTH_C20


def _states(r, v):
    r, v = np.array(r, dtype=float, copy=True), np.array(v, dtype=float, copy=True)
    if r.ndim != 2 or r.shape[1] != 3 or r.shape != v.shape or len(r) == 0:
        raise ValueError("States must be nonempty (N, 3) arrays.")
    if not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("States must be finite.")
    if np.any(np.linalg.norm(r, axis=1) == 0):
        raise ValueError("Position must be nonzero.")
    return r, v


def acceleration(r, mu=EARTH_MU, radius=EARTH_RADIUS, j2=0.0, pole=(0, 0, 1)):
    """Central plus J2 acceleration for rows of inertial positions [m/s^2]."""
    r = np.asarray(r, dtype=float)
    pole = np.asarray(pole, dtype=float)
    norm = np.linalg.norm(r, axis=-1, keepdims=True)
    a = -mu * r / norm**3
    if j2:
        z = np.sum(r * pole, axis=-1, keepdims=True)
        a += (
            1.5
            * j2
            * mu
            * radius**2
            / norm**5
            * ((5 * (z / norm) ** 2 - 1) * r - 2 * z * pole)
        )
    return a


class KeplerBatch:
    """Analytical universal-variable two-body states evaluated from the epoch.

    Bound nonradial Earth orbits only. Evaluation never advances past the
    requested time. State does not depend on callback partition or history.
    """

    def __init__(self, r, v, mu=EARTH_MU):
        """Copy and validate the epoch's bound Cartesian states."""
        self.r0, self.v0 = _states(r, v)
        self.mu = float(mu)
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("mu must be positive and finite.")
        self.radius0 = np.linalg.norm(self.r0, axis=1)
        self.rv0 = np.sum(self.r0 * self.v0, axis=1)
        self.alpha = 2 / self.radius0 - np.sum(self.v0**2, axis=1) / mu
        if np.any(self.alpha <= 0) or np.any(
            np.linalg.norm(np.cross(self.r0, self.v0), axis=1) == 0
        ):
            raise ValueError("Only bound, nonradial elliptic orbits are supported.")

    def at(self, time_s):
        """Return position and velocity at nonnegative seconds since epoch."""
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Time must be finite and nonnegative.")
        if time_s == 0:
            return self.r0.copy(), self.v0.copy()
        root_mu = np.sqrt(self.mu)
        chi = root_mu * self.alpha * time_s
        for _ in range(50):
            z = self.alpha * chi**2
            c, s = _stumpff(z)
            f = (
                self.rv0 / root_mu * chi**2 * c
                + (1 - self.alpha * self.radius0) * chi**3 * s
                + self.radius0 * chi
                - root_mu * time_s
            )
            df = (
                self.rv0 / root_mu * chi * (1 - z * s)
                + (1 - self.alpha * self.radius0) * chi**2 * c
                + self.radius0
            )
            delta = f / df
            chi -= delta
            if np.max(np.abs(delta)) < 1e-8:
                break
        else:
            raise RuntimeError("Universal Kepler solve did not converge.")
        c, s = _stumpff(self.alpha * chi**2)
        fg_f = 1 - chi**2 / self.radius0 * c
        fg_g = time_s - chi**3 / root_mu * s
        r = fg_f[:, None] * self.r0 + fg_g[:, None] * self.v0
        radius = np.linalg.norm(r, axis=1)
        fdot = root_mu / (radius * self.radius0) * (self.alpha * chi**3 * s - chi)
        gdot = 1 - chi**2 / radius * c
        v = fdot[:, None] * self.r0 + gdot[:, None] * self.v0
        return r, v


def _stumpff(z):
    small = np.abs(z) < 1e-5
    # Avoid cancellation and zero division close to epoch.
    safe = np.where(small, 1.0, z)
    root = np.sqrt(safe)
    c = (1 - np.cos(root)) / safe
    s = (root - np.sin(root)) / root**3
    return (
        np.where(small, 0.5 - z / 24 + z**2 / 720 - z**3 / 40320, c),
        np.where(small, 1 / 6 - z / 120 + z**2 / 5040 - z**3 / 362880, s),
    )


class J2Batch:
    """Batched Cartesian RK4, bounded internal steps, no interpolation/lookahead.

    The caller supplies the current ephemeris pole. It is held fixed over the
    elapsed callback interval. Earth precession/nutation within one dynamics
    tick is neglected. Native validation must quantify this approximation.
    """

    def __init__(
        self, r, v, mu=EARTH_MU, radius=EARTH_RADIUS, j2=EARTH_J2, max_step_s=1.0
    ):
        """Copy states and configure an independent Earth central-plus-J2 field."""
        self.r, self.v = _states(r, v)
        KeplerBatch(r, v, mu)  # Same explicit supported initial-state domain.
        if not all(np.isfinite(x) for x in (radius, j2, max_step_s)):
            raise ValueError("Gravity and step parameters must be finite.")
        if radius <= 0 or max_step_s <= 0 or j2 < 0:
            raise ValueError("Invalid radius, J2, or integration step.")
        self.mu, self.radius, self.j2 = mu, radius, j2
        self.max_step_s = max_step_s
        self.time_s = 0.0

    def at(self, time_s, pole=(0, 0, 1)):
        """Advance monotonically to exactly time_s and return independent arrays."""
        if not np.isfinite(time_s) or time_s < self.time_s:
            raise ValueError("Time must be finite and monotonic. Recreate on reset.")
        if time_s == self.time_s:
            # Native Reset may precede the first SPICE UpdateState. No gravity
            # evaluation is needed at the epoch, so no pole is required yet.
            return self.r.copy(), self.v.copy()
        pole = np.asarray(pole, dtype=float)
        if (
            pole.shape != (3,)
            or not np.isfinite(pole).all()
            or not np.isclose(np.linalg.norm(pole), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("Earth pole must be a unit inertial vector.")
        dt = time_s - self.time_s
        steps = int(np.ceil(dt / self.max_step_s))
        if steps:
            h = dt / steps
            for _ in range(steps):
                r, v = self.r, self.v
                a1 = acceleration(r, self.mu, self.radius, self.j2, pole)
                a2 = acceleration(r + h / 2 * v, self.mu, self.radius, self.j2, pole)
                a3 = acceleration(
                    r + h / 2 * v + h**2 / 4 * a1, self.mu, self.radius, self.j2, pole
                )
                a4 = acceleration(
                    r + h * v + h**2 / 2 * a2, self.mu, self.radius, self.j2, pole
                )
                self.r = r + h * v + h**2 / 6 * (a1 + a2 + a3)
                self.v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        self.time_s = float(time_s)
        return self.r.copy(), self.v.copy()
