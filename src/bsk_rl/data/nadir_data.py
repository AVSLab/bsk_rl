"""Data logging and management for nadir scanning."""

from typing import Callable, Optional

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.sats.satellite import Satellite


class ScanningTime(Data):
    """Data for time spent scanning nadir."""

    def __init__(self, scanning_time: float = 0.0) -> None:
        """Data for time spent scanning nadir.

        Time spent scanning nadir ``scanning_time`` is stored in seconds.

        Args:
            scanning_time: [s] Time spent scanning nadir.
        """
        self.scanning_time = scanning_time

    def __add__(self, other: "ScanningTime") -> "ScanningTime":
        """Define the combination of two units of data."""
        scanning_time = self.scanning_time + other.scanning_time

        return self.__class__(scanning_time)


class ScanningTimeStore(DataStore):
    """DataStore for time spent scanning nadir."""

    data_type = ScanningTime

    def __init__(self, *args, **kwargs) -> None:
        """DataStore for time spent scanning nadir.

        Stores the amount of time spent scanning nadir. Calculates new time spent
        scanning based on baud rate of the instrument and the increase in data stored
        in the buffer.
        """
        super().__init__(*args, **kwargs)

    def get_log_state(self) -> float:
        """Return the amount of data currently stored in the storage unit."""
        storage_unit = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
        stored_amount = storage_unit.storageLevel

        return stored_amount

    def compare_log_states(self, old_state: float, new_state: float) -> "ScanningTime":
        """Generate a unit of data based on change in stored data amount.

        Args:
            old_state: Previous amount of data in the storage unit.
            new_state: Current amount of data in the storage unit.

        Returns:
            Data: Data generated
        """
        instrument_baudrate = self.satellite.dynamics.instrument.nodeBaudRate

        if new_state > old_state:
            data_generated = (new_state - old_state) / instrument_baudrate
        else:
            data_generated = 0.0

        return ScanningTime(scanning_time=data_generated)


class ScanningTimeReward(GlobalReward):
    """GlobalReward for rewarding time spent scanning nadir."""

    data_store_type = ScanningTimeStore  # type of DataStore managed by the GlobalReward

    def __init__(
        self,
        reward_fn: Optional[Callable] = None,
    ) -> None:
        """GlobalReward for rewarding time spent scanning nadir.

        This class should be used with the :class:`~bsk_rl.scene.UniformNadirScanning`
        scenario and a satellite with :class:`~bsk_rl.sim.fsw.ContinuousImagingFSWModel`
        and the :class:`~bsk_rl.act.Scan` action.

        Time is computed based on the amount of data in the satellite's buffer. In the
        basic configuration, this is the amount of time that the :class:`~bsk_rl.act.Scan`
        action is enabled and pointing thresholds are met. However, if other models are
        used to prevent the accumulation of data, the satellite will not be rewarded for
        those times.

        Args:
            reward_fn: Reward as function of time spend pointing nadir. By default,
                is set to the time spent scanning times ``scenario.value_per_second``.
        """
        super().__init__()
        self._reward_fn = reward_fn

    @property
    def reward_fn(self):
        """Function to calculate reward based on time spent scanning nadir.

        :meta private:
        """
        if self._reward_fn is None:
            return lambda t: t * self.scenario.value_per_second
        return self._reward_fn

    def calculate_reward(
        self, new_data_dict: dict[str, "ScanningTime"]
    ) -> dict[str, float]:
        """Calculate reward based on ``reward_fn``."""
        reward = {}
        for sat, scanning_time in new_data_dict.items():
            reward[sat] = self.reward_fn(scanning_time.scanning_time)

        return reward


__doc_title__ = "Nadir Scanning"
__all__ = ["ScanningTimeReward", "ScanningTimeStore", "ScanningTime"]
