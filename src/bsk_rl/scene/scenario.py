"""Scenarios define data available for satellites to collect."""

import logging
from abc import ABC
from typing import TYPE_CHECKING

from bsk_rl.utils.functional import Resetable

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class Scenario(ABC, Resetable):
    """Base scenario class."""

    def __init__(self) -> None:
        self.satellites: list["Satellite"]
        self.utc_init: str

    def link_satellites(self, satellites: list["Satellite"]) -> None:
        """Link the environment satellite list to the scenario.

        Args:
            satellites: List of satellites to communicate between.
        """
        self.satellites = satellites

    def validate_satellite_names(self, satellites: list["Satellite"]) -> None:
        """Optionally validate original participant names before automatic renaming.

        The default permits existing environment name normalization. Scenarios with
        explicit identity bindings can reject ambiguous original names.
        """
        pass

    def after_step(self, sim_time: float) -> None:
        """Update scenario conditions after reward and before communication/observation.

        The default does nothing. Scheduled scenarios can override this hook without
        putting scenario-specific logic in the environment.
        """
        pass


class UniformNadirScanning(Scenario):
    """Defines a nadir target center at the center of the planet."""

    def __init__(self, value_per_second: float = 1.0) -> None:
        """Construct uniform data over the surface of the planet.

        Can be used with :class:`~bsk_rl.data.ScanningTimeReward`.

        Args:
            value_per_second: Reward per second for imaging nadir.
        """
        self.value_per_second = value_per_second


__all__ = []
