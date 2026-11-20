"""Bounded numerical unit checks. No Basilisk simulator is constructed."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bsk_rl.utils.passive_orbits import (
    EARTH_C20,
    EARTH_J2,
    EARTH_MU,
    EARTH_RADIUS,
    J2Batch,
    KeplerBatch,
    acceleration,
)


def circular():
    r = np.array([[7e6, 0.0, 0.0], [0.0, 0.0, 42.16e6]])
    v = np.array(
        [[0.0, np.sqrt(EARTH_MU / 7e6), 0.0], [0.0, np.sqrt(EARTH_MU / 42.16e6), 0.0]]
    )
    return r, v


def test_kepler_circular_solution_and_no_history_dependence(record_property):
    r, v = circular()
    model = KeplerBatch(r, v)
    t = 45000.0  # One analytical evaluation, no mission simulation.
    rates = np.sqrt(EARTH_MU / np.linalg.norm(r, axis=1) ** 3)
    expected = (
        r * np.cos(rates * t)[:, None] + v / rates[:, None] * np.sin(rates * t)[:, None]
    )
    np.testing.assert_allclose(model.at(t)[0], expected, atol=1e-6, rtol=0)
    record_property(
        "analytic_snapshot_position_error_m",
        float(np.max(np.linalg.norm(model.at(t)[0] - expected, axis=1))),
    )
    np.testing.assert_array_equal(model.at(0)[0], r)
    model.at(123.0)
    np.testing.assert_array_equal(model.at(t)[0], KeplerBatch(r, v).at(t)[0])


@pytest.mark.parametrize("mode", ["two_body", "j2"])
def test_short_arc_against_independent_potential_gradient_reference(
    mode, record_property
):
    r, v = circular()
    # Noncircular, inclined, rotated J2 pole. Numerical central differences of
    # the scalar potential independently check sign, frame, and normalization.
    v[0] *= 0.97
    pole = np.array([0.12, -0.2, np.sqrt(1 - 0.12**2 - 0.2**2)])
    j2 = EARTH_J2 if mode == "j2" else 0.0

    def potential(x):
        norm = np.linalg.norm(x)
        return (
            EARTH_MU
            / norm
            * (
                1
                - j2
                * (EARTH_RADIUS / norm) ** 2
                * (0.5 * (3 * (x @ pole / norm) ** 2 - 1))
            )
        )

    def rhs(t, y):
        x, vel = y[:3], y[3:]
        h = 10.0
        gradient = [
            (potential(x + np.eye(3)[k] * h) - potential(x - np.eye(3)[k] * h))
            / (2 * h)
            for k in range(3)
        ]
        return np.r_[vel, gradient]

    model = J2Batch(r, v, max_step_s=1) if j2 else KeplerBatch(r, v)
    result = model.at(120.0, pole=pole) if j2 else model.at(120.0)
    for index in range(2):
        ref = solve_ivp(
            rhs,
            (0.0, 120.0),
            np.r_[r[index], v[index]],
            method="DOP853",
            rtol=2e-12,
            atol=1e-9,
        )
        assert ref.success
        record_property(
            f"target_{index}_short_arc_position_error_m",
            float(np.linalg.norm(result[0][index] - ref.y[:3, -1])),
        )
        record_property(
            f"target_{index}_short_arc_velocity_error_m_s",
            float(np.linalg.norm(result[1][index] - ref.y[3:, -1])),
        )
        np.testing.assert_allclose(result[0][index], ref.y[:3, -1], atol=0.005, rtol=0)
        np.testing.assert_allclose(result[1][index], ref.y[3:, -1], atol=1e-5, rtol=0)


def test_j2_axes_and_rotation_covariance():
    assert EARTH_J2 == -np.sqrt(5) * EARTH_C20
    r = np.array([[7e6, 0, 0], [0, 0, 7e6]])
    delta = acceleration(r, j2=EARTH_J2) - acceleration(r)
    scale = 1.5 * EARTH_J2 * EARTH_MU * EARTH_RADIUS**2 / (7e6) ** 4
    np.testing.assert_allclose(delta, [[-scale, 0, 0], [0, 0, 2 * scale]], atol=1e-15)
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]])
    np.testing.assert_allclose(
        acceleration(r @ rotation.T, j2=EARTH_J2, pole=rotation[:, 2]),
        acceleration(r, j2=EARTH_J2) @ rotation.T,
    )


def test_j2_reset_monotonic_time_and_partition_determinism():
    r, v = circular()
    np.testing.assert_array_equal(J2Batch(r, v).at(0, pole=[0, 0, 0])[0], r)
    first, second = J2Batch(r, v), J2Batch(r, v)
    first.at(10.0)
    np.testing.assert_array_equal(first.at(20.0)[0], second.at(20.0)[0])
    with pytest.raises(ValueError, match="monotonic"):
        first.at(0.0)
    np.testing.assert_array_equal(J2Batch(r, v).at(0.0)[0], r)
    with pytest.raises(ValueError, match="unit"):
        second.at(21.0, pole=[0, 0, 2])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_invalid_step(bad):
    with pytest.raises(ValueError):
        J2Batch(*circular(), max_step_s=bad)


def test_state_validation_and_unbound_domain():
    r, v = circular()
    with pytest.raises(ValueError, match="bound"):
        KeplerBatch(r, v * 2)
    with pytest.raises(ValueError, match="finite"):
        KeplerBatch(r * np.nan, v)
    with pytest.raises(ValueError, match="nonnegative"):
        KeplerBatch(r, v).at(-1)
