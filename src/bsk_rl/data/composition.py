"""Data composition classes."""

import logging
from collections.abc import Mapping, Sequence
from copy import copy
from typing import TYPE_CHECKING, Optional

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.sats import Satellite
from bsk_rl.scene import MixedScenario, Scenario

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
        """Link every component to the shared scenario."""
        super().link_scenario(scenario)
        for rewarder in self.rewarders:
            rewarder.link_scenario(scenario)

    def initial_data(self, satellite: Satellite) -> ComposedData:
        """Furnish every component's initial data in a fixed order."""
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


class MixedData(Data):
    """Knowledge indexed by reward channel, independent of local capabilities.

    Channel IDs distinguish independent rewarders even when they use the same data
    class. An absent channel contributes no data; an empty MixedData is the identity.
    """

    def __init__(self, data: Optional[Mapping[str, Data]] = None) -> None:
        """Copy the channel mapping; values follow the Data copying contract."""
        self.data = {} if data is None else dict(data)

    def __add__(self, other: "MixedData") -> "MixedData":
        """Merge matching channels and copy channels present on only one side."""
        if not isinstance(other, MixedData):
            return NotImplemented
        merged = {key: copy(value) for key, value in self.data.items()}
        for key, value in other.data.items():
            if key in self.data:
                if type(self.data[key]) is not type(value):
                    raise TypeError(f"Incompatible data types for channel {key!r}.")
                merged[key] = self.data[key] + value
            else:
                merged[key] = copy(value)
        return MixedData(merged)

    def __getattr__(self, name: str):
        """Forward unambiguous attributes for existing observations and actions."""
        # Avoid recursion when deepcopy probes an incompletely constructed object.
        data = self.__dict__.get("data", {})
        matches = [value for value in data.values() if hasattr(value, name)]
        if len(matches) == 1:
            return getattr(matches[0], name)
        if matches:
            raise AttributeError(
                f"Ambiguous attribute {name!r}; select a MixedData channel explicitly."
            )
        raise AttributeError(f"No Data in MixedData has attribute {name!r}.")

    def __repr__(self) -> str:
        """Represent the channel mapping."""
        return f"MixedData({self.data!r})"


class MixedDataStore(DataStore):
    """Poll local channel stores while retaining any communicated channels."""

    data_type = MixedData

    def __init__(
        self,
        satellite: Satellite,
        data_stores: Mapping[str, DataStore],
        initial_data: Optional[MixedData] = None,
    ) -> None:
        """Initialize explicit channel or scenario bindings."""
        self.data_stores = dict(data_stores)
        if initial_data is None:
            initial_data = MixedData(
                {key: store.data for key, store in self.data_stores.items()}
            )
        super().__init__(satellite, initial_data)
        self.pass_data()

    def pass_data(self) -> None:
        """Synchronize component data with the parent channel mapping."""
        for key, store in self.data_stores.items():
            store.data = self.data.data[key]

    def __getattr__(self, name: str):
        """Forward attributes only when one local component provides them."""
        stores = self.__dict__.get("data_stores", {})
        matches = [store for store in stores.values() if hasattr(store, name)]
        if len(matches) == 1:
            return getattr(matches[0], name)
        if matches:
            raise AttributeError(
                f"Ambiguous attribute {name!r}; select a datastore channel explicitly."
            )
        raise AttributeError(f"No DataStore in MixedDataStore has attribute {name!r}.")

    def get_log_state(self) -> dict:
        """Collect log states from locally active channels."""
        return {key: store.get_log_state() for key, store in self.data_stores.items()}

    def compare_log_states(self, prev_state: dict, new_state: dict) -> MixedData:
        """Generate deltas only for locally active channels."""
        return MixedData(
            {
                key: store.compare_log_states(prev_state[key], new_state[key])
                for key, store in self.data_stores.items()
            }
        )

    def update_from_logs(self) -> MixedData:
        """Update local knowledge and synchronize component stores."""
        self.pass_data()
        new_data = super().update_from_logs()
        self.pass_data()
        return new_data

    def update_with_communicated_data(self) -> None:
        """Merge received knowledge and synchronize local stores."""
        super().update_with_communicated_data()
        self.pass_data()


class MixedReward(GlobalReward):
    """Route named reward channels to selected satellites and scenarios.

    Args:
        rewarders: Channel ID to independent rewarder instance.
        satellite_mapping: Channel ID to names of satellites producing that data.
            Satellites may belong to several channels or none.
        scenario_mapping: Channel ID to scenario ID in MixedScenario. For an ordinary
            Scenario this may be omitted, and every channel uses that scenario.
    """

    data_store_type = MixedDataStore

    def __init__(
        self,
        rewarders: Mapping[str, GlobalReward],
        satellite_mapping: Mapping[str, Sequence[str]],
        scenario_mapping: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Initialize explicit channel or scenario bindings."""
        super().__init__()
        self.rewarders = dict(rewarders)
        self.satellite_mapping = {
            key: frozenset(names) for key, names in satellite_mapping.items()
        }
        self.scenario_mapping = (
            None if scenario_mapping is None else dict(scenario_mapping)
        )
        if self.rewarders.keys() != self.satellite_mapping.keys():
            raise ValueError("Every reward channel must have a satellite mapping.")
        if len({id(r) for r in self.rewarders.values()}) != len(self.rewarders):
            raise ValueError(
                "Each reward channel requires an independent rewarder instance."
            )
        if (
            self.scenario_mapping is not None
            and self.rewarders.keys() != self.scenario_mapping.keys()
        ):
            raise ValueError("Every reward channel must have a scenario mapping.")

    def link_scenario(self, scenario: Scenario) -> None:
        """Bind reward channels to the selected scenario instances."""
        super().link_scenario(scenario)
        if isinstance(scenario, MixedScenario):
            if self.scenario_mapping is None:
                raise ValueError("MixedScenario requires an explicit scenario mapping.")
            known_names = {sat.name for sat in scenario.satellites}
            for key, rewarder in self.rewarders.items():
                scene_id = self.scenario_mapping[key]
                if scene_id not in scenario.scenarios:
                    raise ValueError(
                        f"Unknown scenario {scene_id!r} for channel {key!r}."
                    )
                names = self.satellite_mapping[key]
                if not names <= known_names:
                    raise ValueError(
                        f"Unknown satellites for channel {key!r}: {sorted(names - known_names)}"
                    )
                if not names <= set(scenario.satellite_mapping[scene_id]):
                    raise ValueError(
                        f"Reward channel {key!r} contains satellites outside its scenario."
                    )
                rewarder.link_scenario(scenario.scenarios[scene_id])
        else:
            if self.scenario_mapping is not None:
                raise ValueError("Scenario IDs require a MixedScenario.")
            known_names = {sat.name for sat in scenario.satellites}
            for key, rewarder in self.rewarders.items():
                if not self.satellite_mapping[key] <= known_names:
                    raise ValueError(f"Unknown satellites for channel {key!r}.")
                rewarder.link_scenario(scenario)

    def pass_data(self) -> None:
        """Synchronize component data with the parent channel mapping."""
        for key, rewarder in self.rewarders.items():
            if key not in self.data.data:
                self.data.data[key] = rewarder.data_type()
            rewarder.data = self.data.data[key]

    def reset_overwrite_previous(self) -> None:
        """Clear episode state and reset each component once."""
        super().reset_overwrite_previous()
        for rewarder in self.rewarders.values():
            rewarder.reset_overwrite_previous()
        self.pass_data()

    def reset_pre_sim_init(self) -> None:
        """Prepare each component before simulator construction."""
        self.pass_data()
        for rewarder in self.rewarders.values():
            rewarder.reset_pre_sim_init()

    def reset_during_sim_init(self) -> None:
        """Prepare each component during simulator construction."""
        self.pass_data()
        for rewarder in self.rewarders.values():
            rewarder.reset_during_sim_init()

    def reset_post_sim_init(self) -> None:
        """Finalize component setup after simulator initialization."""
        self.pass_data()
        for rewarder in self.rewarders.values():
            rewarder.reset_post_sim_init()

    def initial_data(self, satellite: Satellite) -> MixedData:
        """Provide initial knowledge for the satellite's assigned channels."""
        return MixedData(
            {
                key: rewarder.initial_data(satellite)
                for key, rewarder in self.rewarders.items()
                if satellite.name in self.satellite_mapping[key]
            }
        )

    def create_data_store(self, satellite: Satellite) -> None:
        """Create assigned stores using each rewarder's setup hooks."""
        stores = {}
        for key, rewarder in self.rewarders.items():
            if satellite.name in self.satellite_mapping[key]:
                # Preserve rewarder-specific setup (e.g. imaging access filters).
                rewarder.create_data_store(satellite)
                stores[key] = satellite.data_store
        satellite.data_store = MixedDataStore(satellite, stores)
        self.cum_reward[satellite.name] = 0.0

    def calculate_reward(self, new_data_dict: dict[str, MixedData]) -> dict[str, float]:
        """Route local deltas and sum rewards against previous global data."""
        self.pass_data()
        reward = {name: 0.0 for name in new_data_dict}
        for key, rewarder in self.rewarders.items():
            channel_data = {
                name: delta.data[key]
                for name, delta in new_data_dict.items()
                if name in self.satellite_mapping[key] and key in delta.data
            }
            if not channel_data:
                continue
            reward_i = rewarder.calculate_reward(channel_data)
            for name, value in reward_i.items():
                if name not in channel_data:
                    raise ValueError(
                        f"Channel {key!r} returned reward for unassigned satellite {name!r}."
                    )
                reward[name] += value
                rewarder.cum_reward[name] += value
        return reward

    def reward(self, new_data_dict: dict[str, MixedData]) -> dict[str, float]:
        """Calculate rewards, merge global deltas, and synchronize child data."""
        reward = super().reward(new_data_dict)
        self.pass_data()
        return reward

    def is_truncated(self, satellite: Satellite) -> bool:
        """Check only rewarders assigned to this satellite."""
        return any(
            r.is_truncated(satellite)
            for key, r in self.rewarders.items()
            if satellite.name in self.satellite_mapping[key]
        )

    def is_terminated(self, satellite: Satellite) -> bool:
        """Check only rewarders assigned to this satellite."""
        return any(
            r.is_terminated(satellite)
            for key, r in self.rewarders.items()
            if satellite.name in self.satellite_mapping[key]
        )


__doc_title__ = "Data Composition"
__all__ = [
    "ComposedReward",
    "ComposedDataStore",
    "ComposedData",
    "MixedReward",
    "MixedDataStore",
    "MixedData",
]
