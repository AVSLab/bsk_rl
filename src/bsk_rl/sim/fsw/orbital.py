"""FSW models for orbital maneuvers."""

import numpy as np

from bsk_rl.sim.fsw import BasicFSWModel, action
from bsk_rl.utils.functional import aliveness_checker, default_args


class MagicOrbitalManeuverFSWModel(BasicFSWModel):
    """Model that allows for instantaneous Delta V maneuvers."""

    def __init__(self, *args, **kwargs) -> None:
        """Model that allows for instantaneous Delta V maneuvers."""
        super().__init__(*args, **kwargs)
        self.setup_fuel(**kwargs)
        self.thrust_count = 0

    @property
    def dv_available(self):
        """Delta-V available for the satellite."""
        return self._dv_available

    @aliveness_checker
    def fuel_remaining(self) -> bool:
        """Check if the satellite has fuel remaining."""
        return self.dv_available > 1e-8

    @default_args(dv_available_init=100.0)
    def setup_fuel(self, dv_available_init: float, **kwargs):
        """Set up available fuel for the satellite.

        Args:
            dv_available_init: [m/s] Initial fuel level.
            kwargs: Passed to other setup functions.
        """
        # TODO: may adjust names for consistency with modelled fuel take in future.
        self._dv_available = dv_available_init

    @action
    def action_impulsive_thrust(self, dv_N: np.ndarray) -> None:
        """Thrust relative to the inertial frame.

        Args:
            dv_N: [m/s] Inertial Delta V.
        """
        if np.linalg.norm(dv_N) > self.dv_available:
            self.satellite.logger.warning(
                f"Maneuver exceeds available Delta V ({np.linalg.norm(dv_N)}/{self.dv_available} m/s)."
            )
            dv_N = dv_N / np.linalg.norm(dv_N) * self.dv_available

        self._dv_available -= np.linalg.norm(dv_N)

        self.dynamics.scObject.dynManager.getStateObject(
            self.dynamics.scObject.hub.nameOfHubVelocity
        ).setState(list(np.array(self.dynamics.v_BN_N) + np.array(dv_N)))

        self.thrust_count += 1
