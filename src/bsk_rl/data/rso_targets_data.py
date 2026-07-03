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
        self.eclipse_threshold_for_imaging = 0.5

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
                # self.inspection_task_completed = True

            update_idx = np.where(new_state - old_state > 0)[0]
            imaged = []
            eclipsed=[]
            for idx in update_idx:
                message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
                target_id = message.read().storedDataName[int(idx)]
                # imaged.append(
                #     [target for target in self.data.known if target.name == target_id and self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor < self.eclipse_threshold_for_imaging][0]
                # )

                for target in self.data.known:
                    if target.name == target_id:
                        if self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor > self.satellite.dynamics.eclipse_threshold_for_imaging:
                            imaged.append(target)
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
        # self.downlink_bonus = 0.5 #deprecated
        # self.imaging_bonus = 0.5 #deprecated
        self.eclipse_threshold_for_reward = 0.5
        self.total_downlinks = 0
        self.useful_downlinks = 0
        self.imaged_illuminated = []
        self.imaged_illuminated_names: set[str] = set()
        self.usefully_downlinked_names: set[str] = set()

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
        if not hasattr(self, "old_state"):
            self.old_state = np.zeros_like(
                self.scenario.satellites[0].dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
            )
        reward = {}
        imaged_targets = sum(
            [new_data.imaged for new_data in new_data_dict.values()], []
        )

        new_state = np.array(self.scenario.satellites[0].dynamics.storageUnit.storageUnitDataOutMsg.read().storedData)
        downlinked_targets = [int(i) for i in np.where(new_state - self.old_state < 0)[0]] # keep track of downlinked targets
        downlinked_idxs = [int(i) for i in np.where(new_state - self.old_state < 0)[0]]
        if downlinked_idxs:
            # Event-level total (unchanged semantics)
            self.total_downlinks += len(downlinked_idxs)

            # Names of downlinked targets this step
            msg = self.scenario.satellites[0].dynamics.storageUnit.storageUnitDataOutMsg
            downlinked_names = [msg.read().storedDataName[idx] for idx in downlinked_idxs]

            print("Downlinked target names:", downlinked_names)
            print('total accumulated rewards SS1: ' + str(self.cum_reward['SS1']))
            print('Targets imaged:' + str(len(self.scenario.satellites[0].data_store.data.imaged)))

            # Reward per event (unchanged)
            for sat_id, _ in new_data_dict.items():
                if sat_id != 'SS1':
                    continue
                for name in downlinked_names:
                    # Reward logic: find target by name and award if it was illuminated
                    tgt = next((t for t in self.scenario.target_spacecrafts if t.name == name), None)
                    if (tgt is not None) and (tgt.name in self.imaged_illuminated_names):
                        # Add to "unique useful downlinked" set (counts at most once per target)
                        if name not in self.usefully_downlinked_names:
                            self.usefully_downlinked_names.add(name)
                            # Keep the metric synchronized with the set size
                            self.useful_downlinks = len(self.usefully_downlinked_names)

                        # Optional per-event downlink reward hook.
                        # reward[sat_id] += self.reward_fn(tgt.priority * self.scenario.satellites[0].dynamics.downlink_bonus)

        if self.scenario.satellites[0].simulator.sim_time >= (self.scenario.satellites[0].simulator.time_limit - 300):
            imaged_illuminated_elevations =[]
            self.scenario.satellites[0].dynamics.imaged_illuminated = len(self.imaged_illuminated)
            # self.scenario.satellites[0].imaged_illuminated_elevation = ()
            self.scenario.satellites[0].dynamics.total_downlinks  = self.total_downlinks
            self.scenario.satellites[0].dynamics.useful_downlinks  = self.useful_downlinks
            print("METRICS")
            print("self.scenario.satellites[0].dynamics.imaged_overall",len(self.scenario.satellites[0].data_store.data.imaged))
            print("self.scenario.satellites[0].dynamics.imaged_illuminated",self.scenario.satellites[0].dynamics.imaged_illuminated)
            print("self.scenario.satellites[0].dynamics.total_downlinks",self.scenario.satellites[0].dynamics.total_downlinks)
            print("self.scenario.satellites[0].dynamics.useful_downlinks", self.scenario.satellites[0].dynamics.useful_downlinks)


        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            if self.scenario.satellites[0].dynamics.penalties == 1:
                if sat_id == 'SS1' and self.scenario.satellites[0].dynamics.battery_charge_fraction < 0.05:
                    reward[sat_id] += self.scenario.satellites[0].dynamics.low_battery_penalty
                elif sat_id == 'SS1' and self.scenario.satellites[0].dynamics.battery_charge_fraction < 0.1:
                    reward[sat_id] += self.scenario.satellites[0].dynamics.low_battery_penalty
                if sat_id == 'SS1' and self.scenario.satellites[0].dynamics.storage_level_fraction > .991:
                    reward[sat_id] += self.scenario.satellites[0].dynamics.full_storage_penalty

            for target in new_data.imaged:
                # if target not in self.data.imaged:
                if sat_id == 'SS1':
                    if target not in self.data.imaged and self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor > self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward:
                        self.imaged_illuminated.append(target)
                        self.imaged_illuminated_names.add(target.name)

            # Adding Downlink Reward
            if len(downlinked_targets) > 0:
                if sat_id == 'SS1':
                    for idx in downlinked_targets:
                        # Do something with downlinked index
                        target_name = self.scenario.satellites[0].dynamics.storageUnit.storageUnitDataOutMsg.read().storedDataName[idx] # this uses the id index for the storage unit not the id of the target itself!
                        target = next((t for t in self.scenario.target_spacecrafts if t.name == target_name), None)
                        if target is not None and target in self.imaged_illuminated:
                            reward[sat_id] += self.reward_fn(target.priority * self.scenario.satellites[0].dynamics.downlink_bonus)
                            # self.useful_downlinks += 1

            shadow_factors=[]
            penumbra_target_id = []
            # Adding Imaging Reward
            for target in new_data.imaged:
                # if target not in self.data.imaged:
                if sat_id == 'SS1':
                    if target not in self.data.imaged and self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor > self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward:
                        reward[sat_id] += self.reward_fn(
                            target.priority * self.scenario.satellites[0].dynamics.imaging_bonus  # full reward above the illumination threshold
                            # target.priority * self.scenario.satellites[0].dynamics.imaging_bonus * (self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor-self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward)/(self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward)  # this reward gives linearly scaled returns based on the actual value
                        ) / imaged_targets.count(target)
                        # self.imaged_illuminated.append(target)
                        # self.imaged_illuminated_dict[sat_id]+=target
                    elif target not in self.data.imaged and self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor < self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward:
                        reward[sat_id] += target.priority * self.scenario.satellites[0].dynamics.eclipsedImagePenalty

                if sat_id == 'SS1':
                    if target is not None and self.scenario.satellites[0].dynamics.print_info:
                        shadow_factors.append(self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor)
                        # Check for non-binary shadow factors
                        penumbra_target_id.append(target.id)
                        non_binary_indices = [i for i, val in enumerate(shadow_factors) if val != 0.0 and val != 1.0]

                        # Print results
                        if non_binary_indices:
                            print("Non-binary shadowFactors found at indices:", non_binary_indices)
                            for i in non_binary_indices:
                                print(f"Penumbra Target: ID {penumbra_target_id[i]}: shadowFactor = {shadow_factors[i]}")
                        # else:
                        #     print("All shadowFactors are either 0.0 or 1.0")




        self.old_state = np.array(self.scenario.satellites[0].dynamics.storageUnit.storageUnitDataOutMsg.read().storedData)

        return reward

    # def is_terminated(self) -> bool:
    #     return self.inspection_task_completed



__doc_title__ = "Unique Images"
__all__ = ["RSOTargetImageReward", "RSOTargetImageStore", "RSOTargetImageData"]
