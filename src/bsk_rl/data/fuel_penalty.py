import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


class FuelData(Data):
    """Data for fuel usage."""

    def __init__(self, fuel_used: float = 0.0) -> None:
        """Construct fuel data.

        Args:
            fuel_used: Amount of fuel used.
        """
        self.fuel_used = fuel_used

    def __add__(self, other: "FuelData") -> "FuelData":
        """Combine two units of fuel data.

        Args:
            other: Another unit of fuel data to combine with this one.

        Returns:
            Combined unit of fuel data.
        """
        fuel_used = self.fuel_used + other.fuel_used
        return FuelData(fuel_used)


class FuelDataStore(DataStore):
    data_type = FuelData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_log_state(self) -> float:
        try:
            return self.satellite.fsw.dv_available  # TODO update for other fuel models
        except AttributeError:
            return 0.0

    def compare_log_states(self, prev_state: np.ndarray, new_state: np.ndarray) -> Data:
        fuel_used = prev_state - new_state
        return FuelData(fuel_used)


class FuelPenalty(GlobalReward):
    """Global penalty for fuel usage."""

    datastore_type = FuelDataStore

    def __init__(self, penalty_weight: float = 1.0) -> None:
        """Construct fuel penalty.

        Args:
            penalty_weight: Scaling factor to apply to fuel penalty.
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self._penalty_weight = penalty_weight

    def reset_pre_sim_init(self):
        if isinstance(self._penalty_weight, Callable):
            self.penalty_weight = self._penalty_weight()
        else:
            self.penalty_weight = self._penalty_weight

        for sat in self.scenario.satellites:
            sat.penalty_weight = self.penalty_weight

    def calculate_reward(self, new_data_dict: dict[str, FuelData]) -> dict[str, float]:
        penalty = {
            sat_name: -data.fuel_used * self.penalty_weight
            for sat_name, data in new_data_dict.items()
        }

        return penalty
