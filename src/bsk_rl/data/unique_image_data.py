"""Data system for recording unique images of targets."""

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.utils import vizard

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


class UniqueImageData(Data):
    """Data for unique images of targets."""

    def __init__(
        self,
        imaged: Optional[set["Target"]] = None,
        duplicates: int = 0,
        known: Optional[set["Target"]] = None,
    ) -> None:
        """Construct unit of data to record unique images.

        Keeps track of ``imaged`` targets, a count of ``duplicates`` (i.e. images that
        were not rewarded due to the target already having been imaged), and all
        ``known`` targets in the environment.

        Args:
            imaged: Set of targets that are known to be imaged.
            duplicates: Count of target imaging duplication.
            known: Set of targets that are known to exist (imaged and unimaged).
        """
        if imaged is None:
            imaged = set()
        self.imaged = set(imaged)
        self.duplicates = duplicates + len(imaged) - len(self.imaged)
        if known is None:
            known = set()
        self.known = set(known)

    def __add__(self, other: "UniqueImageData") -> "UniqueImageData":
        """Combine two units of data.

        Args:
            other: Another unit of data to combine with this one.

        Returns:
            Combined unit of data.
        """
        imaged = self.imaged | other.imaged
        duplicates = (
            self.duplicates
            + other.duplicates
            + len(self.imaged)
            + len(other.imaged)
            - len(imaged)
        )
        known = self.known | other.known
        return self.__class__(imaged=imaged, duplicates=duplicates, known=known)


class UniqueImageStore(DataStore):
    """DataStore for unique images of targets."""

    data_type = UniqueImageData

    def __init__(self, *args, **kwargs) -> None:
        """DataStore for unique images.

        Detects new images by watching for an increase in data in each target's corresponding
        buffer.
        """
        super().__init__(*args, **kwargs)

    def get_log_state(self) -> float:
        """Log the instantaneous storage unit state at the end of each step.

        Returns:
            float: storedData from satellite storage unit
        """
        msg = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
        return msg.storedData[0]

    def compare_log_states(
        self, old_state: np.ndarray, new_state: np.ndarray
    ) -> UniqueImageData:
        """Check for an increase in logged data to identify new images.

        Args:
            old_state: Older storedData from satellite storage unit.
            new_state: Newer storedData from satellite storage unit.

        Returns:
            list: Targets imaged at new_state that were unimaged at old_state.
        """
        data_increase = new_state - old_state
        if data_increase <= 0:
            return UniqueImageData()
        else:
            assert self.satellite.latest_target is not None
            self.update_target_colors([self.satellite.latest_target])
            return UniqueImageData(imaged={self.satellite.latest_target})

    @vizard.visualize
    def update_target_colors(self, targets, vizInstance=None, vizSupport=None):
        """Update target colors in Vizard."""
        for target in targets:
            vizSupport.changeLocation(
                vizInstance,
                target.name,
                color=vizSupport.toRGBA255(self.satellite.vizard_color),
            )


class UniqueImageReward(GlobalReward):
    """GlobalReward for rewarding unique images."""

    data_store_type = UniqueImageStore

    def __init__(
        self,
        reward_fn: Callable = lambda p: p,
    ) -> None:
        """GlobalReward for rewarding unique images.

        This data system should be used with the :class:`~bsk_rl.sats.ImagingSatellite` and
        a scenario that generates targets, such as :class:`~bsk_rl.scene.UniformTargets` or
        :class:`~bsk_rl.scene.CityTargets`.

        The satellites all start with complete knowledge of the targets in the scenario.
        Each target can only give one satellite a reward once; if any satellite has imaged
        a target, reward will never again be given for that target. The satellites filter
        known imaged targets from consideration for imaging to prevent duplicates.
        Communication can transmit information about what targets have been imaged in order
        to prevent reimaging.


        Args:
            scenario: GlobalReward.scenario
            reward_fn: Reward as function of priority.
        """
        super().__init__()
        self.reward_fn = reward_fn

    def initial_data(self, satellite: "Satellite") -> "UniqueImageData":
        """Furnish data to the scenario.

        Currently, it is assumed that all targets are known a priori, so the initial data
        given to the data store is the list of all targets.
        """
        return self.data_type(known=self.scenario.targets)

    def create_data_store(self, satellite: "Satellite") -> None:
        """Override the access filter in addition to creating the data store."""
        super().create_data_store(satellite)

        def unique_target_filter(opportunity):
            if opportunity["type"] == "target":
                return opportunity["object"] not in satellite.data_store.data.imaged
            return True

        satellite.add_access_filter(unique_target_filter)

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        """Reward each new unique image once.

        Reward is evaluated based on ``self.reward_fn(target.priority)``.

        Args:
            new_data_dict: Record of new images for each satellite

        Returns:
            reward: Cumulative reward across satellites for one step
        """
        reward = {}
        imaged_counts = {}
        for new_data in new_data_dict.values():
            for target in new_data.imaged:
                if target not in imaged_counts:
                    imaged_counts[target] = 0
                imaged_counts[target] += 1

        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                if target not in self.data.imaged:
                    reward[sat_id] += (
                        self.reward_fn(target.priority) / imaged_counts[target]
                    )
        return reward


__doc_title__ = "Unique Images"
__all__ = ["UniqueImageReward", "UniqueImageStore", "UniqueImageData"]
