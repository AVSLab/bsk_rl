"""Data system for recording unique images of targets."""

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import math

from bsk_rl.data.base import Data, DataStore, GlobalReward
from Basilisk.utilities import orbitalMotion
from bsk_rl.utils import vizard
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target,Strip

logger = logging.getLogger(__name__)


class UniqueStripImageData(Data):
    """Data for unique strip images of targets."""

    def __init__(
        self,
        imaged: Optional[list["Strip"]] = None,
        duplicates: int = 0,
        known: Optional[list["Strip"]] = None,
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
        self.imaged = list(set(imaged))
        self.duplicates = duplicates + len(imaged) - len(self.imaged)
        if known is None:
            known = []
        self.known = list(set(known))

    def __add__(self, other: "UniqueStripImageData") -> "UniqueStripImageData":
        """Combine two units of data.

        Args:
            other: Another unit of data to combine with this one.

        Returns:
            Combined unit of data.
        """
        imaged = list(set(self.imaged + other.imaged))
        duplicates = (
            self.duplicates
            + other.duplicates
            + len(self.imaged)
            + len(other.imaged)
            - len(imaged)
        )
        known = list(set(self.known + other.known))
        return self.__class__(imaged=imaged, duplicates=duplicates, known=known)


class UniqueStripImageStore(DataStore):
    """DataStore for unique images of targets."""

    data_type = UniqueStripImageData

    def __init__(self, *args, **kwargs) -> None:
        """DataStore for unique images.

        Detects new images by watching for an increase in data in each target's corresponding
        buffer.
        """
        super().__init__(*args, **kwargs)

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
    ) -> UniqueStripImageData:
        """Check for an increase in logged data to identify new images. To be considered as imaged the strip needs to fullfill this condition: the data-buffer increase is greater than 95% of the data necessary to store the images from the target strip.

        Args:
            old_state: Older storedData from satellite storage unit.
            new_state: Newer storedData from satellite storage unit.

        Returns:
            list: Targets imaged at new_state that were unimaged at old_state.
        """
        update_idx = []
        instrument_baudrate = self.satellite.dynamics.instrument.nodeBaudRate

        for idx in np.where(new_state - old_state > 0)[0]:
            data_generated = (new_state[idx] - old_state[idx]) / instrument_baudrate
            update_idx.append((idx, data_generated))
        
        imaged = []
        for idx in update_idx:
            index=idx[0]
            data_generated_target=idx[1]
            message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
            target_id = message.read().storedDataName[int(index)]
            target = [target for target in self.data.known if target.id == target_id][0]
            # Calculation of the time required to image the strip
            dot_product = np.dot(target.r_LP_P_start / np.linalg.norm(target.r_LP_P_start), target.r_LP_P_end / np.linalg.norm(target.r_LP_P_end))
            theta = np.arccos(np.clip(dot_product, -1.0, 1.0))
            d_strip = theta * orbitalMotion.REQ_EARTH * 1e3  # length of the strip [m]
            t_strip = d_strip / target.aquisition_speed  # Calculation of the time to cover the strip


            if data_generated_target >= math.floor(0.90 * t_strip * 2) / 2: # The strip is considered as imaged if the data buffer-increase is greater than 95% of the data necessary to store the images from the target strip
                imaged.append(target)
            self.update_target_colors(imaged)
        return UniqueStripImageData(imaged=imaged)
    
    @vizard.visualize
    def update_target_colors(self, targets, vizInstance=None, vizSupport=None):
        """Update target colors in Vizard."""
        for target in targets:
            def compute_strip_vertices(r_start, r_end, width=100e3):
                r_start = np.array(r_start).flatten()
                r_end = np.array(r_end).flatten()

                # Direction along the centerline
                d_vec = r_end - r_start
                d_vec /= np.linalg.norm(d_vec)

                # Approximate surface normal at the midpoint
                mid_vec = (r_start + r_end) / 2
                mid_vec /= np.linalg.norm(mid_vec)

                # Perpendicular vector in the tangent plane
                p_vec = np.cross(mid_vec, d_vec)
                p_vec /= np.linalg.norm(p_vec)

                # Compute the four corner points
                half_width = width / 2.0
                v1 = r_start + half_width * p_vec  # top-left
                v2 = r_start - half_width * p_vec  # bottom-left
                v3 = r_end + half_width * p_vec    # top-right
                v4 = r_end - half_width * p_vec    # bottom-right

                vertices = list(v1) + list(v3) + list(v4) + list(v2)
                return vertices
            def get_priority_color(priority, is_imaged=True):
                """
                Maps priority (0 to 1) to an RGBA color using the magma colormap.
                Sets alpha to 255 if imaged, 50 if not.
                """
                priority = np.clip(priority, 0.0, 1.0)

                # Get RGBA from magma colormap (values in 0-1 float)
                cmap = plt.get_cmap('autumn')
                rgba = cmap(priority)  # returns (r, g, b, a) in [0, 1]

                # Convert to 0–255 and override alpha depending on imaging
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                a = 255 

                return [r, g, b, a]
            
            vizSupport.addQuadMap(vizInstance,
                                ID=int(target.name[6:]),
                                parentBodyName="earth",
                                vertices=compute_strip_vertices(target.r_LP_P_start, target.r_LP_P_end, width=100e3),
                                color=get_priority_color(target.priority),  # Use target priority for color
                                )


class UniqueStripImageReward(GlobalReward):
    """GlobalReward for rewarding unique images."""

    datastore_type = UniqueStripImageStore

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

    def initial_data(self, satellite: "Satellite") -> "UniqueStripImageData":
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
        self, new_data_dict: dict[str, UniqueStripImageData]
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
                if target not in self.data.imaged:
                    reward[sat_id] += self.reward_fn(
                        target.priority
                    ) / imaged_targets.count(target)
            if reward[sat_id] == 0.0:
                reward[sat_id] = 0.0
        return reward


__doc_title__ = "Unique Strip Images"
__all__ = ["UniqueStripImageReward", "UniqueStripImageStore", "UniqueStripImageData"]
