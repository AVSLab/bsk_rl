"""``bsk_rl.utils.attitude``: Attitude dynamics related utilities."""

from typing import Any, Iterable, Optional

import numpy as np
from Basilisk.utilities.RigidBodyKinematics import C2MRP


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


def hill_frame_attitude(
    r_N: Iterable[float],
    v_N: Iterable[float],
    spin_config: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates an initially nadir-pointing attitude with configurable body spin.

    The body frame is initially aligned with the Hill frame and points nadir. Spin
    rates in "spin_config" are specified in degrees per second.

    Args:
        r_N: [m] Inertial position of the spacecraft.
        v_N: [m/s] Inertial velocity of the spacecraft.
        spin_config: Body spin relative to the Hill frame.
            type can be "fixed" with a rate, or "uniform" with low and high rate bounds.
            axis can be "radial", "tangential", "normal", "velocity", "random", or a three-element Hill-frame vector.
            Defaults to zero spin about the radial axis.

    Returns:
        Initial body attitude MRPs and body angular rate (sigma_BN, omega_BN_B).

    Raises:
        ValueError: If an input is not a three-vector, the orbit does not define a
            Hill frame, or the spin configuration is invalid.
    """
    r_N = np.asarray(r_N, dtype=float)
    v_N = np.asarray(v_N, dtype=float)
    if r_N.shape != (3,) or v_N.shape != (3,):
        raise ValueError("r_N and v_N must be three-element vectors")

    r_norm = np.linalg.norm(r_N)
    h_N = np.cross(r_N, v_N)
    h_norm = np.linalg.norm(h_N)
    if np.isclose(r_norm, 0.0) or np.isclose(h_norm, 0.0):
        raise ValueError("r_N and v_N must define a valid Hill frame")

    x = r_N / r_norm
    z = h_N / h_norm
    y = np.cross(z, x)
    HN = np.array([x, y, z])
    BH = np.eye(3)

    config = {"type": "fixed", "rate": 0.0, "axis": "radial"}
    if spin_config is not None:
        unknown_keys = set(spin_config) - {"type", "rate", "low", "high", "axis"}
        if unknown_keys:
            raise ValueError(f"Unknown spin configuration keys: {unknown_keys}")
        config.update(spin_config)

    if config["type"] == "fixed":
        spin_rate = config["rate"]
    elif config["type"] == "uniform":
        if "low" not in config or "high" not in config:
            raise ValueError("Uniform spin requires low and high rate bounds")
        if config["low"] > config["high"]:
            raise ValueError("Uniform spin low bound must not exceed high bound")
        spin_rate = np.random.uniform(config["low"], config["high"])
    else:
        raise ValueError(f"Unknown spin type: {config['type']}")

    axis = config["axis"]
    if isinstance(axis, str):
        named_axes = {
            "radial": np.array([1.0, 0.0, 0.0]),
            "tangential": np.array([0.0, 1.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0]),
            "velocity": HN @ v_N,
        }
        if axis == "random":
            spin_axis_H = np.random.normal(size=3)
        elif axis in named_axes:
            spin_axis_H = named_axes[axis]
        else:
            raise ValueError(f"Unknown spin axis: {axis}")
    else:
        spin_axis_H = np.asarray(axis, dtype=float)

    if spin_axis_H.shape != (3,):
        raise ValueError("Spin axis must be a three-element vector")
    axis_norm = np.linalg.norm(spin_axis_H)
    if np.isclose(axis_norm, 0.0):
        raise ValueError("Spin axis must be nonzero")

    # The instantaneous Hill-frame rate is valid for circular and eccentric orbits.
    spin_rate_rad = np.radians(spin_rate)
    omega_HN_N = h_N / r_norm**2
    omega_BH_H = spin_rate_rad * spin_axis_H / axis_norm
    omega_BN_H = BH @ HN @ omega_HN_N + omega_BH_H

    return C2MRP(BH @ HN), omega_BN_H


__doc_title__ = "Attitude"
__all__ = ["hill_frame_attitude", "random_tumble"]
