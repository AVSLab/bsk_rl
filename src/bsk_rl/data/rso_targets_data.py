"""Data system for recording unique images of targets."""

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward
from collections import OrderedDict

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


class RSOTargetImageData(Data):
    """Data for unique images of targets."""

    def __init__(
        self,
        imaged: Optional[list["RSOTarget"]] = None,
        eclipsed:Optional[list["RSOTarget"]] = None,
        duplicates: int = 0,
        known: Optional[list["RSOTarget"]] = None,
    ) -> None:
        """Construct unit of data to record unique images.

        Keeps track of ``imaged`` targets, a count of ``duplicates`` (i.e. images that
        were not rewarded due to the target already having been imaged), and all
        ``known`` targets in the environment.

        Args:
            imaged: List of targets that are known to be imaged.
            duplicates: Count of target imaging duplication.
            known: List of targets that are known to exist (imaged and unimaged).
        """
        if imaged is None:
            imaged = []
        self.imaged = list(OrderedDict.fromkeys(imaged))  # Preserve order, remove duplicates
        self.duplicates = duplicates + len(imaged) - len(self.imaged)

        if known is None:
            known = []
        self.known = list(OrderedDict.fromkeys(known))  # Preserve order, remove duplicates

    def __add__(self, other: "RSOTargetImageData") -> "RSOTargetImageData":
        """Combine two units of data.

        Args:
            other: Another unit of data to combine with this one.

        Returns:
            Combined unit of data.
        """
        imaged = list(OrderedDict.fromkeys(self.imaged + other.imaged))
        duplicates = (
            self.duplicates
            + other.duplicates
            + len(self.imaged)
            + len(other.imaged)
            - len(imaged)
        )
        known = list(OrderedDict.fromkeys(self.known + other.known))
        return self.__class__(imaged=imaged, duplicates=duplicates, known=known)


class RSOTargetImageStore(DataStore):
    """DataStore for unique images of targets."""

    data_type = RSOTargetImageData

    def __init__(self, *args, **kwargs) -> None:
        """DataStore for unique images.

        Detects new images by watching for an increase in data in each target's corresponding
        buffer.
        """
        super().__init__(*args, **kwargs)
        self.inspection_task_completed = False

    def get_log_state(self) -> np.ndarray:
        """Log the instantaneous storage unit state at the end of each step.

        Returns:
            array: storedData from satellite storage unit
        """
        return np.array(
            self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
        )

    def compare_log_states(
        self, old_state: np.ndarray, new_state: np.ndarray
    ) -> RSOTargetImageData:
        """Check for an increase in logged data to identify new images.

        Args:
            old_state: Older storedData from satellite storage unit.
            new_state: Newer storedData from satellite storage unit.

        Returns:
            list: Targets imaged at new_state that were unimaged at old_state.
        """
        if self.satellite.name == "SS1":
            # Check if all n_targets have non-zero buffer levels
            self.non_zero_buffers = np.count_nonzero(new_state)
            if self.non_zero_buffers >= len(self.satellite.data_store.data.known):
                if self.inspection_task_completed == None:
                    self.inspection_task_completed = False
                self.inspection_task_completed = True
            print('Targets imaged:'+str(self.non_zero_buffers))


            update_idx = np.where(new_state - old_state > 0)[0]
            imaged = []
            eclipsed=[]
            for idx in update_idx:
                message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
                target_id = message.read().storedDataName[int(idx)]
                # self.data.imaged.append( # TODO: DHP check if this should be removed or if this replaces the part right below (what about duplicates?)
                #     [target for target in self.data.known if target.id == target_id][0]
                # )
                imaged.append(
                    [target for target in self.data.known if target.name == target_id][0]
                )
            # eclipse_threshold = 0.6
            # eclipsed.append([target for target in imaged if target.eclipse_status >= eclipse_threshold]) # this can be used to change reward in case the target is in eclipse
            # return RSOTargetImageData(imaged=imaged, eclipsed = eclipsed)
            return RSOTargetImageData(imaged=imaged)
        else:
            return RSOTargetImageData(imaged=[])



class RSOTargetImageReward(GlobalReward):
    """GlobalReward for rewarding unique images."""

    datastore_type = RSOTargetImageStore

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
        self.inspection_task_completed = False

    def initial_data(self, satellite: "Satellite") -> "RSOTargetImageData":
        """Furnish data to the scenario.

        Currently, it is assumed that all targets are known a priori, so the initial data
        given to the data store is the list of all targets.
        """
        return self.data_type(known=self.scenario.target_spacecrafts)


    def calculate_reward(
        self, new_data_dict: dict[str, RSOTargetImageData]
    ) -> dict[str, float]:
        """Reward each new unique image once.

        Reward is evaluated based on ``self.reward_fn(target.priority)``.

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
                # if target not in self.data.imaged and target not in self.data.eclipsed:
                if target not in self.data.imaged:
                    reward[sat_id] += self.reward_fn(
                        target.priority
                    ) / imaged_targets.count(target)

        return reward

    def is_terminated(self) -> bool:
        return self.inspection_task_completed



__doc_title__ = "Unique Images"
__all__ = ["RSOTargetImageReward", "RSOTargetImageStore", "RSOTargetImageData"]
