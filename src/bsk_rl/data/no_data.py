"""A data and reward system that does nothing, returning zero reward on every step."""

import logging
from typing import TYPE_CHECKING, Any

from bsk_rl.data.base import Data, DataStore, GlobalReward

logger = logging.getLogger(__name__)


class NoData(Data):
    """Holds no data."""

    def __init__(self, *args, **kwargs):
        """Holds no data."""
        return super().__init__(*args, **kwargs)

    def __add__(self, other):
        """Add nothing to nothing."""
        return self.__class__()


class NoDataStore(DataStore):
    """DataStore for no data."""

    data_type = NoData

    def __init__(self, *args, **kwargs):
        """Stores and generates no data."""
        return super().__init__(*args, **kwargs)

    def compare_log_states(self, old_state, new_state):
        """Always returns no data."""
        return self.data_type()


class NoReward(GlobalReward):
    """GlobalReward for no data."""

    data_store_type = NoDataStore

    def __init__(self, *args, **kwargs):
        """Returns zero reward at every step.

        This reward system is useful for debugging environments, but is not useful for
        training, since reward is always zero for every satellite.
        """
        return super().__init__(*args, **kwargs)

    def calculate_reward(self, new_data_dict):
        """Reward nothing."""
        return {sat: 0.0 for sat in new_data_dict.keys()}


__doc_title__ = "No Data"
__all__ = ["NoReward", "NoDataStore", "NoData"]
