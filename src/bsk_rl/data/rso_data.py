"""Data system for recording RSO surface."""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.sats import Satellite
from bsk_rl.scene.rso_points import RSOPoint
from bsk_rl.sim.dyn import RSODynModel, RSOImagingDynModel
from bsk_rl.utils import vizard

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)

RSO = "rso"
OBSERVER = "observer"


class RSOInspectionData(Data):
    def __init__(self, point_inspect_status: Optional[dict[RSOPoint, bool]] = None):
        if point_inspect_status is None:
            point_inspect_status = {}
        self.point_inspect_status = point_inspect_status

    def __add__(self, other: "RSOInspectionData"):
        point_inspect_status = {}
        point_inspect_status.update(self.point_inspect_status)
        for point, access in other.point_inspect_status.items():
            if point not in point_inspect_status:
                point_inspect_status[point] = access
            else:
                point_inspect_status[point] = point_inspect_status[point] or access

        return RSOInspectionData(point_inspect_status)


class RSOInspectionDataStore(DataStore):
    data_type = RSOInspectionData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_access_recorders = []
        self.storage_recorder = None

        if issubclass(self.satellite.dyn_type, RSOImagingDynModel):
            self.role = OBSERVER
        else:
            self.role = RSO

    def set_storage_recorder(self, recorder):
        self.storage_recorder = recorder
        self.satellite.simulator.AddModelToTask(
            self.satellite.dynamics.task_name, recorder, ModelPriority=1000
        )

    def add_point_access_recorder(self, recorder):
        self.point_access_recorders.append(recorder)
        self.satellite.simulator.AddModelToTask(
            self.satellite.dynamics.task_name, recorder, ModelPriority=1000
        )

    def clear_recorders(self):
        if self.storage_recorder:
            self.storage_recorder.clear()
        for recorder in self.point_access_recorders:
            recorder.clear()

    def get_log_state(self) -> list[list[bool]]:
        """Log the storage unit state and point access state for all times in the step.

        Returns:
            todo
        """
        if self.role == RSO:
            return None

        log_len = len(self.storage_recorder.storageLevel)
        if log_len <= 1:
            imaging_req = np.zeros(log_len)
        else:
            imaging_req = np.diff(self.storage_recorder.storageLevel)
            imaging_req = np.concatenate((imaging_req, [imaging_req[-1]]))

        inspected_logs = []
        for recorder in self.point_access_recorders:
            inspected = np.logical_and(imaging_req, recorder.hasAccess)
            inspected_logs.append(list(np.array(inspected)))

        self.clear_recorders()

        return inspected_logs

    def compare_log_states(self, _, inspected_logs) -> Data:
        if self.role == RSO:
            return RSOInspectionData()

        point_inspect_status = {}
        for rso_point, log in zip(
            self.data.point_inspect_status.keys(), inspected_logs
        ):
            if any(log):
                point_inspect_status[rso_point] = True

        self.update_point_colors(
            [
                rso_point
                for rso_point in point_inspect_status
                if point_inspect_status[rso_point]
            ]
        )

        if len(point_inspect_status) > 0:
            self.satellite.logger.info(
                f"Inspected {len(point_inspect_status)} points this step"
            )

        return RSOInspectionData(point_inspect_status)

    @vizard.visualize
    def update_point_colors(self, rso_points, vizInstance=None, vizSupport=None):
        """Update target colors in Vizard."""
        for location in vizInstance.locations:
            if location.stationName in [str(point) for point in rso_points]:
                location.color = vizSupport.toRGBA255("tab:green", alpha=0.5)


class RSOInspectionReward(GlobalReward):
    datastore_type = RSOInspectionDataStore

    def __init__(
        self, inspection_reward_scale: float = 1.0, completion_bonus: float = 0.0
    ):
        super().__init__()
        self.completion_bonus = completion_bonus
        self.inspection_reward_scale = inspection_reward_scale

    def reset_overwrite_previous(self) -> None:
        super().reset_overwrite_previous()
        self.bonus_reward_yielded = False

    def reset_post_sim_init(self) -> None:
        super().reset_post_sim_init()

        for i, observer in enumerate(self.scenario.observers):
            observer.data_store.set_storage_recorder(
                observer.dynamics.storageUnit.storageUnitDataOutMsg.recorder()
            )
            logger.debug(
                f"Logging {len(self.scenario.rso.dynamics.rso_points)} access points"
            )
            for rso_point_model in self.scenario.rso.dynamics.rso_points:
                observer.data_store.add_point_access_recorder(
                    rso_point_model.accessOutMsgs[i].recorder()
                )

    def initial_data(self, satellite: Satellite) -> Data:
        if not issubclass(satellite.dyn_type, RSOImagingDynModel):
            return RSOInspectionData()

        return RSOInspectionData({point: False for point in self.scenario.rso_points})

    def calculate_reward(self, new_data_dict: dict[str, Data]) -> dict[str, float]:
        total_points = len(self.scenario.rso_points)
        reward = {}
        total_new_points = 0
        for satellite_id, data in new_data_dict.items():
            if len(data.point_inspect_status) == 0:
                continue

            new_points = 0
            for point, access in data.point_inspect_status.items():
                if access and not self.data.point_inspect_status.get(point, False):
                    new_points += 1

            if new_points > 0:
                logger.info(f"{satellite_id} inspected {new_points} new points.")

            reward[satellite_id] = (
                new_points / total_points * self.inspection_reward_scale
            )
            total_new_points += new_points
        if (
            sum(self.data.point_inspect_status.values()) + total_new_points
            == len(self.scenario.rso_points)
            and not self.bonus_reward_yielded
        ):
            logger.info("All points inspected! Awarding completion bonus.")
            for satellite_id in self.cum_reward:
                reward[satellite_id] = (
                    reward.get(satellite_id, 0.0) + self.completion_bonus
                )
            self.bonus_reward_yielded = True

        return reward


__doc_title__ = "RSO Inspection"
__all__ = ["RSOInspectionReward", "RSOInspectionDataStore", "RSOInspectionData"]
