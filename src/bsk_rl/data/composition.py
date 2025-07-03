"""Data composition classes."""

import logging
from typing import TYPE_CHECKING, Optional

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.sats import Satellite
from bsk_rl.scene.scenario import Scenario

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class ComposedData(Data):
    """Data for composed data types."""

    def __init__(self, *data: Data) -> None:
        """Data for composed data types.

        Args:
            data: Data types to compose.
        """
        self.data = data

    def __add__(self, other: "ComposedData") -> "ComposedData":
        """Combine two units of composed data.

        Args:
            other: Another unit of composed data to combine with this one.

        Returns:
            Combined unit of composed data.
        """
        if len(self.data) == 0 and len(other.data) == 0:
            data = []
        elif len(self.data) == 0:
            data = [type(d)() + d for d in other.data]
        elif len(other.data) == 0:
            data = [d + type(d)() for d in self.data]
        elif len(self.data) == len(other.data):
            data = [d1 + d2 for d1, d2 in zip(self.data, other.data)]
        else:
            raise ValueError(
                "ComposedData units must have the same number of data types."
            )
        return ComposedData(*data)

    def __getattr__(self, name: str):
        """Search for an attribute in the datas."""
        for data in self.data:
            if hasattr(data, name):
                return getattr(data, name)
        raise AttributeError(f"No Data in ComposedData has attribute '{name}'")

    def __repr__(self) -> str:
        """String representation of the ComposedData."""
        return f"ComposedData({', '.join(repr(d) for d in self.data)})"


class ComposedDataStore(DataStore):
    data_type = ComposedData

    def pass_data(self) -> None:
        """Pass data to the sub-DataStores.

        :meta private:
        """
        for ds, data in zip(self.data_stores, self.data.data):
            ds.data = data

    def __init__(
        self,
        satellite: "Satellite",
        *data_store_types: type[DataStore],
        initial_data: Optional[ComposedData] = None,
        data_store_kwargs: Optional[list] = None,
    ):
        """DataStore for composed data types.

        Args:
            satellite: Satellite which data is being stored for.
            data_store_types: DataStore types to compose.
            initial_data: Initial data to start the store with. Usually comes from
                :class:`~bsk_rl.data.GlobalReward.initial_data`.
            data_store_kwargs: List of data_store kwargs matching data_store_types.
        """
        self.data: ComposedData
        super().__init__(satellite, initial_data)
        if data_store_kwargs is None:
            data_store_kwargs = [{} for _ in data_store_types]

        if len(data_store_types) != len(data_store_kwargs):
            raise ValueError(
                "data_store_types and data_store_kwargs must have the same length."
            )

        self.data_stores = tuple(
            ds(satellite, **kwargs)
            for ds, kwargs in zip(data_store_types, data_store_kwargs)
        )
        self.pass_data()

    def __getattr__(self, name: str):
        """Search for an attribute in the data_stores."""
        for data_store in self.data_stores:
            if hasattr(data_store, name):
                return getattr(data_store, name)
        raise AttributeError(
            f"No DataStore in ComposedDataStore has attribute '{name}'"
        )

    def get_log_state(self) -> list:
        """Pull information used in determining current data contribution."""
        log_states = [ds.get_log_state() for ds in self.data_stores]
        return log_states

    def compare_log_states(self, prev_state: list, new_state: list) -> Data:
        """Generate a unit of composed data based on previous step and current step logs."""
        data = [
            ds.compare_log_states(prev, new)
            for ds, prev, new in zip(self.data_stores, prev_state, new_state)
        ]
        return ComposedData(*data)

    def update_from_logs(self) -> Data:
        """Update the data store based on collected information."""
        new_data = super().update_from_logs()
        self.pass_data()
        return new_data

    def update_with_communicated_data(self) -> None:
        """Update the data store based on collected information from other satellites."""
        super().update_with_communicated_data()
        self.pass_data()


class ComposedReward(GlobalReward):
    data_store_type = ComposedDataStore

    def pass_data(self) -> Data:
        """Pass data to the sub-rewarders.

        :meta private:
        """
        for rewarder, data in zip(self.rewarders, self.data.data):
            rewarder.data = data

    def __init__(self, *rewarders: GlobalReward) -> None:
        """Rewarder for composed data types.

        This type can be automatically constructed by passing a tuple of rewarders to
        the environment constructor's `reward` argument.

        Args:
            rewarders: Global rewarders to compose.
        """
        super().__init__()
        self.rewarders = rewarders

    def __getattr__(self, name: str):
        """Search for an attribute in the rewarders."""
        for rewarder in self.rewarders:
            if hasattr(rewarder, name):
                return getattr(rewarder, name)
        raise AttributeError(
            f"No GlobalReward in ComposedReward has attribute '{name}'"
        )

    def reset_pre_sim_init(self) -> None:
        """Handle resetting for all rewarders."""
        super().reset_pre_sim_init()
        for rewarder in self.rewarders:
            rewarder.reset_pre_sim_init()

    def reset_post_sim_init(self) -> None:
        """Handle resetting for all rewarders."""
        super().reset_post_sim_init()
        for rewarder in self.rewarders:
            rewarder.reset_post_sim_init()

    def reset_overwrite_previous(self) -> None:
        """Handle resetting for all rewarders."""
        super().reset_overwrite_previous()
        for rewarder in self.rewarders:
            rewarder.reset_overwrite_previous()

    def link_scenario(self, scenario: Scenario) -> None:
        """Link the rewarder to the scenario."""
        super().link_scenario(scenario)
        for rewarder in self.rewarders:
            rewarder.link_scenario(scenario)

    def initial_data(self, satellite: Satellite) -> ComposedData:
        """Furnish the DataStore with :class:`ComposedData`."""
        return ComposedData(
            *[rewarder.initial_data(satellite) for rewarder in self.rewarders]
        )

    def create_data_store(self, satellite: Satellite) -> None:
        """Create a :class:`CompositeDataStore` for a satellite."""
        satellite.data_store = ComposedDataStore(
            satellite,
            *[r.data_store_type for r in self.rewarders],
            initial_data=self.initial_data(satellite),
            data_store_kwargs=[r.data_store_kwargs for r in self.rewarders],
        )
        self.cum_reward[satellite.name] = 0.0
        for rewarder in self.rewarders:
            rewarder.cum_reward[satellite.name] = 0.0

    def calculate_reward(
        self, new_data_dict: dict[str, ComposedData]
    ) -> dict[str, float]:
        """Calculate reward for each data type and combine them."""
        data_len = len(list(new_data_dict.values())[0].data)

        for data in new_data_dict.values():
            assert len(data.data) == data_len

        reward = {}
        if data_len != 0:
            for i, rewarder in enumerate(self.rewarders):
                reward_i = rewarder.calculate_reward(
                    {sat_id: data.data[i] for sat_id, data in new_data_dict.items()}
                )

                # Logging
                nonzero_reward = {k: v for k, v in reward_i.items() if v != 0}
                if len(nonzero_reward) > 0:
                    logger.info(f"{type(rewarder).__name__} reward: {nonzero_reward}")

                for sat_id, sat_reward in reward_i.items():
                    reward[sat_id] = reward.get(sat_id, 0.0) + sat_reward
                    rewarder.cum_reward[sat_id] += sat_reward
        return reward

    def reward(self, new_data_dict: dict[str, ComposedData]) -> dict[str, float]:
        """Return combined reward calculation and update data."""
        reward = super().reward(new_data_dict)
        self.pass_data()
        return reward

    def is_truncated(self, satellite: Satellite) -> bool:
        """Check if the episode is truncated by any rewarder."""
        return any(rewarder.is_truncated(satellite) for rewarder in self.rewarders)

    def is_terminated(self, satellite) -> bool:
        """Check if the episode is terminated by any rewarder."""
        return any(rewarder.is_terminated(satellite) for rewarder in self.rewarders)


__doc_title__ = "Data Composition"
__all__ = ["ComposedReward", "ComposedDataStore", "ComposedData"]
