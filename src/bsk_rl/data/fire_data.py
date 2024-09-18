"""Data system for recording unique images of targets."""

import logging
from copy import copy
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward

if TYPE_CHECKING:
    from bsk_rl.sats import AccessSatellite, Satellite
    from bsk_rl.scene.fires.fires import Fire

logger = logging.getLogger(__name__)


class FireData(Data):
    """Data for unique images of targets."""

    def __init__(
        self,
        image_times: Optional[dict["Fire", list[tuple[float, float]]]] = None,
    ) -> None:
        if image_times is None:
            image_times = {}
        self.image_times = image_times

    def __add__(self, other: "FireData") -> "FireData":
        """Combine two units of data.

        Args:
            other: Another unit of data to combine with this one.

        Returns:
            Combined unit of data.
        """
        image_times = {}
        for fire in set(list(other.image_times.keys()) + list(self.image_times.keys())):
            if fire in self.image_times and fire in other.image_times:
                image_times[fire] = sorted(
                    list(set(self.image_times[fire] + other.image_times[fire])),
                    key=lambda x: x[0],
                )
            elif fire in self.image_times:
                image_times[fire] = self.image_times[fire]
            elif fire in other.image_times:
                image_times[fire] = other.image_times[fire]

        return self.__class__(image_times=image_times)


class FireStore(DataStore):
    """DataStore for unique images of targets."""

    data_type = FireData

    def __init__(self, *args, only_detect_burning: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.only_detect_burning = only_detect_burning
        self.scan_time = 0.0

    def get_log_state(self) -> tuple[np.ndarray, list]:
        """Log the instantaneous storage unit state at the end of each step.

        Returns:
            array: storedData from satellite storage unit
        """
        data = np.array(
            self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
        )

        scanned_fires_opportunities = self.satellite.opportunities_dict(
            types="fire",
            filter=lambda opportunity: (
                opportunity["object"].start_time
                <= opportunity["window"][0]  # Started before we got there
                and (
                    (opportunity["window"][0] + opportunity["window"][1]) / 2
                    <= self.satellite.simulator.sim_time
                    + self.satellite.scan_ahead_time
                )  # Accessible within scan ahead range
                and (
                    opportunity["object"].end_time >= self.scan_time
                    if self.only_detect_burning
                    else True
                )  # Burning when scanned
            ),
        )
        scanned_fires = list(
            set(scanned_fires_opportunities.keys()) - set(self.data.image_times.keys())
        )
        self.scan_time = self.satellite.simulator.sim_time

        return data, scanned_fires

    def compare_log_states(
        self, old_state: np.ndarray, new_state: np.ndarray
    ) -> FireData:
        """Check for an increase in logged data to identify new images.

        Args:
            old_state: Older storedData from satellite storage unit.
            new_state: Newer storedData from satellite storage unit.

        Returns:
            list: Targets imaged at new_state that were unimaged at old_state.
        """
        old_data, _ = old_state
        new_data, new_scanned_fires = new_state
        update_idx = np.where(new_data - old_data > 0)[0]
        fires = {}
        for idx in update_idx:
            message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
            target_id = message.read().storedDataName[int(idx)]
            fire = [fire for fire in self.data.image_times if fire.id == target_id][0]
            time = self.satellite.simulator.sim_time
            fires[fire] = [(time, fire.burning_area(time))]

        for fire in new_scanned_fires:
            if fire not in fires:
                fires[fire] = []

        return FireData(fires)


class FireReward(GlobalReward):
    """GlobalReward for rewarding unique images."""

    datastore_type = FireStore

    def __init__(
        self,
        only_detect_burning: bool = True,
        filter_burnt: bool = True,
        reimage_post_burn: bool = False,
    ) -> None:
        super().__init__()
        self.only_detect_burning = only_detect_burning
        self.filter_burnt = filter_burnt
        self.reimage_post_burn = reimage_post_burn

    def reset_overwrite_previous(self) -> None:
        super().reset_overwrite_previous()
        self.first_time_images = 0
        self.reimages = 0

    def initial_data(self, satellite) -> "FireData":
        """Furnish the :class:`~bsk_rl.data.base.DataStore` with initial data."""
        data = FireData(
            {
                fire: []
                for fire in self.scenario.fires
                if fire.start_time < -5700.0 and fire.end_time > 0.0
            }
        )
        logger.info(f"Initial fires: {len(data.image_times)}")
        return data

    def create_data_store(self, satellite: "AccessSatellite") -> None:
        """Override the access filter in addition to creating the data store."""
        super().create_data_store(
            satellite, only_detect_burning=self.only_detect_burning
        )

        def fire_appeared_filter(opportunity):
            return satellite.simulator.sim_time >= opportunity["object"].start_time

        satellite.add_access_filter(fire_appeared_filter, types="fire")

        def fire_known_filter(opportunity):
            return opportunity["object"] in satellite.data_store.data.image_times

        satellite.add_access_filter(fire_known_filter, types="fire")

        if self.filter_burnt:

            if self.reimage_post_burn:

                def fire_burnt_filter(opportunity):
                    """Hide if burnt out unless imaged during fire but not after fire."""
                    fire = opportunity["object"]
                    burnt_out = (
                        opportunity["object"].end_time < satellite.simulator.sim_time
                    )
                    if not burnt_out:
                        return True
                    else:
                        if fire in satellite.data_store.data.image_times:
                            image_times = satellite.data_store.data.image_times[fire]
                            if len(image_times) == 0:  # Never imaged
                                return False
                            elif (
                                image_times[-1][0] > fire.end_time
                            ):  # Already imaged post-burn
                                return False
                            else:
                                return True
                        else:
                            return False

                satellite.add_access_filter(fire_burnt_filter, types="fire")

            else:

                def fire_burnt_filter(opportunity):
                    return opportunity["object"].end_time > satellite.simulator.sim_time

                satellite.add_access_filter(fire_burnt_filter, types="fire")

    def calculate_reward(self, new_data_dict: dict[str, FireData]) -> dict[str, float]:
        reward = {}
        min_revisit_time = 60 * 5.0
        # target_spacing = 3600.0
        # max_spacing = 3600.0 * 5.0
        # print({id: new_data.image_times for id, new_data in new_data_dict.items()})
        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for fire, time_values in new_data.image_times.items():
                if len(time_values) > 0:  # Only reward imaged fires

                    fast_revisit_penalty = 1.0
                    if (
                        fire not in self.data.image_times
                        or len(self.data.image_times[fire]) == 0
                    ):  # First image
                        self.first_time_images += 1
                    else:  # Reimage
                        self.reimages += 1
                        revisit_time = (
                            time_values[0][0] - self.data.image_times[fire][-1][0]
                        )
                        if revisit_time < min_revisit_time:
                            fast_revisit_penalty = max(
                                revisit_time / min_revisit_time, 0
                            )

                    reward[sat_id] += time_values[0][1] * fast_revisit_penalty

        return reward


__doc_title__ = "Fire Tip and Cue"
__all__ = ["FireData", "FireStore", "FireReward"]
