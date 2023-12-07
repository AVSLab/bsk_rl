import numpy as np


def random_tumble(maxSpinRate=0.001):
    """
    Simulates a spacecraft in a random tumble with uniformly sampled initial conditions.

    :return: sigma_bn
    :return: omega_bn
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


def static_inertial():
    """
    Simulates a spacecraft in a sidereal stare fixed to the inertial origin.

    :return:
    """

    sigma_bn = np.zeros(
        [
            3,
        ]
    )
    omega_bn = np.zeros(
        [
            3,
        ]
    )

    return sigma_bn, omega_bn
