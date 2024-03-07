"""Attitude dynamics related utilities."""

import numpy as np


def random_tumble(maxSpinRate: float = 0.001):
    """Generate a spacecraft random tumble with uniformly sampled conditions.

    Args:
        maxSpinRate: Maximum spin rate [rad/s].

    Returns:
        sigma_bn: Initial spacecraft attitude [rad].
        omega_bn: Initial spacecraft angular velocity [rad/s].
    """
    sigma_bn = np.random.uniform(
        0,
        1.0,
        [
            3,
        ],
    )
    omega_bn = np.random.uniform(
        -maxSpinRate,
        maxSpinRate,
        [
            3,
        ],
    )

    return sigma_bn, omega_bn
