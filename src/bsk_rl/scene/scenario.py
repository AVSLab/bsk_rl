"""Scenarios define data available for satellites to collect."""

import logging
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class Scenario(ABC):
    """Base scenario class."""

    def reset_pre_sim(self) -> None:  # pragma: no cover
        """Reset the scenario before initializing the simulator."""
        pass

    def initial_data(self, satellite: "Satellite", data_type: type["Data"]) -> "Data":
        """Furnish the :class:`~bsk_rl.data.base.DataStore` with initial data."""
        return data_type()


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
