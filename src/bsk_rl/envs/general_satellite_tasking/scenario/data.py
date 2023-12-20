from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
        Target,
    )
    from bsk_rl.envs.general_satellite_tasking.types import (
        EnvironmentFeatures,
        Satellite,
    )

import numpy as np

LogStateType = Any


class DataType(ABC):
    """Base class for units of satellite data"""

    @abstractmethod  # pragma: no cover
    def __add__(self, other: "DataType") -> "DataType":
        """Define the combination of two units of data"""
        pass


class DataStore(ABC):
    DataType: type[DataType]  # Define the unit of data used by the DataStore

    def __init__(self, data_manager: "DataManager", satellite: "Satellite") -> None:
        """Base class for satellite data logging; one created per satellite

        Args:
            data_manager: Simulation data manager to report back to
            satellite: Satellite's data being stored
        """
        self.satellite = satellite
        self.is_fresh = True
        self.staged_data = []

        self._initialize_knowledge(data_manager.env_features)
        self.data = self.DataType()
        self.new_data = self.DataType()

    def _initialize_knowledge(self, env_features: "EnvironmentFeatures") -> None:
        """Establish knowledge about the world known to the satellite. Defaults to
        knowing everything about the environment."""
        self.env_knowledge = env_features

    def _clear_logs(self) -> None:
        """If necessary, clear any loggers"""
        pass

    def _get_log_state(self) -> LogStateType:
        """Pull information for current data contribution e.g. sensor readings"""
        pass

    @abstractmethod  # pragma: no cover
    def _compare_log_states(
        self, old_state: LogStateType, new_state: LogStateType
    ) -> "DataType":
        """Generate a unit of data based on previous step and current step logs

        Args:
            old_state: A previous result of _get_log_state()
            new_state: A newer result of _get_log_state()

        Returns:
            DataType: Data generated
        """
        pass

    def internal_update(self) -> "DataType":
        """Update the data store based on collected information

        Returns:
            New data from the previous step
        """
        if not hasattr(self, "log_state"):
            self.log_state = self._get_log_state()
            self._clear_logs()
            return self.DataType()
        old_log_state = self.log_state
        self.log_state = self._get_log_state()
        self._clear_logs()
        new_data = self._compare_log_states(old_log_state, self.log_state)
        self.data += new_data
        self.new_data = new_data
        return new_data

    def stage_communicated_data(self, external_data: "DataType") -> None:
        """Prepare data to be added from another source, but don't add it yet

        Args:
            external_data: Data from another satellite to be added
        """
        self.staged_data.append(external_data)

    def communication_update(self) -> None:
        """Update the data store from staged data

        Args:
            external_data (DataType): Data collected by another satellite
        """
        for staged in self.staged_data:
            self.data += staged
        self.staged_data = []


class DataManager(ABC):
    DataStore: type[DataStore]  # type of DataStore managed by the DataManager

    def __init__(self, env_features: Optional["EnvironmentFeatures"] = None) -> None:
        """Base class for simulation-wide data management; handles data recording and
        rewarding.
        TODO: allow for creation/composition of multiple managers

        Args:
            env_features: Information about the environment that can be collected as
                data
        """
        self.env_features = env_features
        self.DataType = self.DataStore.DataType

    def reset(self) -> None:
        self.data = self.DataType()
        self.cum_reward = {}

    def create_data_store(self, satellite: "Satellite") -> None:
        """Create a data store for a satellite"""
        satellite.data_store = self.DataStore(self, satellite)
        self.cum_reward[satellite.id] = 0.0

    @abstractmethod  # pragma: no cover
    def _calc_reward(self, new_data_dict: dict[str, DataType]) -> dict[str, float]:
        """Calculate step reward based on all satellite data from a step

        Args:
            new_data_dict: Satellite-DataType pairs of new data from a step

        Returns:
            Step reward for each satellite
        """
        pass

    def reward(self, new_data_dict: dict[str, DataType]) -> dict[str, float]:
        """Calls _calc_reward and logs cumulative reward"""
        reward = self._calc_reward(new_data_dict)
        for satellite_id, sat_reward in reward.items():
            self.cum_reward[satellite_id] += sat_reward
        return reward


###########
# No Data #
###########
class NoData(DataType):
    def __add__(self, other):
        return self.__class__()


class NoDataStore(DataStore):
    DataType = NoData

    def _compare_log_states(self, old_state, new_state):
        return self.DataType()


class NoDataManager(DataManager):
    DataStore = NoDataStore

    def _calc_reward(self, new_data_dict):
        return {sat: 0.0 for sat in new_data_dict.keys()}


#######################################
# Unique Targets with Constant Values #
#######################################


class UniqueImageData(DataType):
    def __init__(
        self, imaged: Optional[list["Target"]] = None, duplicates: int = 0
    ) -> None:
        """DataType to log unique imaging

        Args:
            imaged: List of targets that are known to be imaged.
            duplicates: Count of target imaging duplication.
        """
        if imaged is None:
            imaged = []

        self.imaged = list(set(imaged))
        self.duplicates = duplicates + len(imaged) - len(self.imaged)

    def __add__(self, other: "UniqueImageData") -> "UniqueImageData":
        imaged = list(set(self.imaged + other.imaged))
        duplicates = (
            self.duplicates
            + other.duplicates
            + len(self.imaged)
            + len(other.imaged)
            - len(imaged)
        )
        return self.__class__(imaged=imaged, duplicates=duplicates)


class UniqueImageStore(DataStore):
    DataType = UniqueImageData

    def _get_log_state(self) -> np.ndarray:
        """Log the instantaneous storage unit state at the end of each step

        Returns:
            array: storedData from satellite storage unit
        """
        return np.array(
            self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
        )

    def _compare_log_states(
        self, old_state: np.ndarray, new_state: np.ndarray
    ) -> UniqueImageData:
        """Checks two storage unit logs for an increase in logged data to identify new
        images

        Args:
            old_state: older storedData from satellite storage unit
            new_state: newer storedData from satellite storage unit

        Returns:
            list: Targets imaged at new_state that were unimaged at old_state
        """
        update_idx = np.where(new_state - old_state > 0)[0]
        imaged = []
        for idx in update_idx:
            message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
            target_id = message.read().storedDataName[int(idx)]
            imaged.append(
                [
                    target
                    for target in self.env_knowledge.targets
                    if target.id == target_id
                ][0]
            )
        return UniqueImageData(imaged=imaged)


class UniqueImagingManager(DataManager):
    DataStore = UniqueImageStore

    def __init__(
        self,
        env_features: Optional["EnvironmentFeatures"] = None,
        reward_fn: Callable = lambda p: p,
    ) -> None:
        """DataManager for rewarding unique images

        Args:
            env_features: DataManager.env_features
            reward_fn: Reward as function of priority.
        """
        super().__init__(env_features)
        self.reward_fn = reward_fn

    def _calc_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        """Reward new each unique image once using self.reward_fn()

        Args:
            new_data_dict: Record of new images for each satellite

        Returns:
            reward: Cumulative reward across satellites for one step
        """
        reward = {}
        imaged_targets = sum(
            [new_data.imaged for new_data in new_data_dict.values()], []
        )
        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                if target not in self.data.imaged:
                    reward[sat_id] += self.reward_fn(
                        target.priority
                    ) / imaged_targets.count(target)

        for new_data in new_data_dict.values():
            self.data += new_data
        return reward


"""Targets with Time-Dependent Rewards"""


# class TimeDepImageData(DataType):
#     def __init__(self, rewards=None) -> None:
#         """DataType to log scalar imaging

#         Args:
#             imaged (dict, optional): Reward obtained from each target.
#         """
#         if rewards is None:
#             rewards = {}
#         self.rewards = rewards

#     def __add__(self, other) -> "TimeDepImageData":
#         rewards = copy(self.rewards)
#         for target, reward in other.rewards.items():
#             if target in rewards:
#                 rewards[target] = max(rewards[target], reward)
#             else:
#                 rewards[target] = reward
#         return self.__class__(rewards=rewards)


# class TimeDepImageStore(DataStore):
#     DataType = TimeDepImageData

#     def _get_log_state(self) -> Iterable[float]:
#         """Log the instaneous storage unit state at the end of each step

#         Returns:
#             array: storedData from satellite storage unit
#         """
#         return np.array(
#           self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
#         )

#     def _compare_log_states(self, old_state, new_state) -> TimeDepImageData:
#         """Checks two storage unit logs for an increase in logged data to identify new
#         images

#         Args:
#             old_state (array): older storedData from satellite storage unit
#             new_state (array): newer storedData from satellite storage unit

#         Returns:
#             list: Targets imaged at new_state that were unimaged at old_state
#         """
#         update_idx = np.where(new_state - old_state > 0)[0]
#         imaged = []
#         for idx in update_idx:
#             message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
#             target_id = message.read().storedDataName[int(idx)]
#             imaged.append(
#                 [
#                     target
#                     for target in self.env_knowledge.targets
#                     if target.id == target_id
#                 ][0]
#             )
#         return self.DataType(imaged=imaged)


# class TimeDepImagingManager(DataManager):
#     DataStore = TimeDepImageStore

#     # def __init__(self, env_features):
#     #     """DataManager for rewarding time-dependent images. Will only give marginal
#     #       reward for reimaging at higher value

#     #     Args:
#     #         env_features (EnvironmentFeatures): DataManager.env_features
#     #         reward_fn (function, optional): Reward as function of priority.
#     #     """
#     #     super().__init__(env_features)

#     def _calc_reward(self, new_data_dict):
#         """Reward each image for additional reward from higher quality images

#         Args:
#             new_data_dict (dict): Record of new images for each satellite

#         Returns:
#             float: Cumulative new reward
#         """
#         reward = 0.0
#         for new_data in new_data_dict.values():
#             for target, reward in new_data.rewards.items():
#                 if target not in self.data.rewards:
#                     reward += reward
#                 elif reward > self.data.rewards[target]:
#                     reward += reward - self.data.rewards[target]
#             self.data += new_data
#         return reward

#################
# Nadir Pointing#
#################


class NadirScanningTimeData(DataType):
    def __init__(self, scanning_time: float = 0.0) -> None:
        """DataType to log data generated scanning nadir

        Args:
            scanning_time: Time spent scanning nadir
        """
        self.scanning_time = scanning_time

    def __add__(self, other: "NadirScanningTimeData") -> "NadirScanningTimeData":
        """Define the combination of two units of data"""
        scanning_time = self.scanning_time + other.scanning_time

        return self.__class__(scanning_time)


class ScanningNadirTimeStore(DataStore):
    DataType = NadirScanningTimeData

    def _get_log_state(self) -> LogStateType:
        """Returns the amount of data stored in the storage unit."""

        storage_unit = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
        stored_amount = storage_unit.storageLevel

        # return amount of data stored
        return stored_amount

    def _compare_log_states(
        self, old_state: float, new_state: float
    ) -> "NadirScanningTimeData":
        """Generate a unit of data based on previous step and current step stored
        data amount.

        Args:
            old_state: Previous amount of data in the storage unit
            new_state: Current amount of data in the storage unit

        Returns:
            DataType: Data generated
        """
        instrument_baudrate = self.satellite.dynamics.instrument.nodeBaudRate

        if new_state > old_state:
            data_generated = (new_state - old_state) / instrument_baudrate
        else:
            data_generated = 0.0

        return NadirScanningTimeData(data_generated)


class NadirScanningManager(DataManager):
    DataStore = ScanningNadirTimeStore  # type of DataStore managed by the DataManager

    def __init__(
        self,
        env_features: Optional["EnvironmentFeatures"] = None,
        reward_fn: Callable = None,
    ) -> None:
        """
        Args:
            env_features: Information about the environment that can be collected as
                data
            reward_fn: Reward as function of time spend pointing nadir
        """
        super().__init__(env_features)
        if reward_fn is None:

            def reward_fn(p):
                # Reward as a function of time send pointing nadir (p) and value
                # per second
                return p * self.env_features.value_per_second

        self.reward_fn = reward_fn

    def _calc_reward(
        self, new_data_dict: dict[str, "NadirScanningTimeData"]
    ) -> dict[str, float]:
        """Calculate step reward based on all satellite data from a step

        Args:
            new_data_dict (dict): Satellite-DataType of new data from a step

        Returns:
            Step reward
        """
        reward = {}
        for sat, scanning_time in new_data_dict.items():
            reward[sat] = self.reward_fn(scanning_time.scanning_time)

        return reward
