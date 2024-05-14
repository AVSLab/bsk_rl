"""``bsk_rl.utils.attitude``: Attitude dynamics related utilities."""

import numpy as np


def random_tumble(maxSpinRate: float = 0.001):
    """Generate a spacecraft random tumble with uniformly sampled conditions.

    Args:
        maxSpinRate: [rad/s] Maximum spin rate.

    Returns:
        tuple:
            * **sigma_bn**: [rad] Initial spacecraft attitude.
            * **omega_bn**: [rad/s] Initial spacecraft angular velocity.
    """
    sigma_bn = np.random.uniform(0, 1.0, [3])
    omega_bn = np.random.uniform(-maxSpinRate, maxSpinRate, [3])

    return sigma_bn, omega_bn


__doc_title__ = "Attitude"
__all__ = ["random_tumble"]
