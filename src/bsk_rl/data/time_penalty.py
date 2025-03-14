import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


class TimeData(Data):
    """Data for time usage."""

    def __init__(self, time_expended: float = 0.0) -> None:
        """Construct time data.

        Args:
            fuel_used: Amount of fuel used.
        """
        self.time_expended = time_expended

    def __add__(self, other: "TimeData") -> "TimeData":
        """Combine two units of fuel data.

        Args:
            other: Another unit of fuel data to combine with this one.

        Returns:
            Combined unit of fuel data.
        """
        fuel_used = self.time_expended + other.time_expended
        return TimeData(fuel_used)


class TimeDataStore(DataStore):
    data_type = TimeData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_log_state(self) -> float:
        return self.satellite.simulator.sim_time

    def compare_log_states(self, prev_state: float, new_state: float) -> Data:
        time_passed = new_state - prev_state
        return TimeData(time_passed)


class TimePenalty(GlobalReward):
    """Global penalty for fuel usage."""

    datastore_type = TimeDataStore

    def __init__(self, penalty_weight: float = 1.0) -> None:
        """Construct fuel penalty.

        Args:
            penalty_weight: Scaling factor to apply to time penalty.
        """
        super().__init__()
        self.penalty_weight = penalty_weight

    def calculate_reward(self, new_data_dict: dict[str, TimeData]) -> dict[str, float]:
        penalty = {
            sat_name: -data.time_expended * self.penalty_weight
            for sat_name, data in new_data_dict.items()
        }

        return penalty
