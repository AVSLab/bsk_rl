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
    def __init__(
        self,
        point_inspect_status: Optional[dict[RSOPoint, bool]] = None,
        point_illuminate_status: Optional[dict[RSOPoint, bool]] = None,
    ):
        if point_inspect_status is None:
            point_inspect_status = {}
        self.point_inspect_status = point_inspect_status
        if point_illuminate_status is None:
            point_illuminate_status = {}
        self.point_illuminate_status = point_illuminate_status

    def __add__(self, other: "RSOInspectionData"):
        point_inspect_status = {}
        point_inspect_status.update(self.point_inspect_status)
        for point, access in other.point_inspect_status.items():
            if point not in point_inspect_status:
                point_inspect_status[point] = access
            else:
                point_inspect_status[point] = point_inspect_status[point] or access

        point_illuminate_status = {}
        point_illuminate_status.update(self.point_illuminate_status)
        for point, access in other.point_illuminate_status.items():
            if point not in point_illuminate_status:
                point_illuminate_status[point] = access
            else:
                point_illuminate_status[point] = (
                    point_illuminate_status[point] or access
                )

        return RSOInspectionData(point_inspect_status, point_illuminate_status)

    @property
    def num_points_inspected(self):
        """Number of points inspected."""
        return sum(self.point_inspect_status.values())

    @property
    def num_points_illuminated(self):
        """Number of points illuminated."""
        return sum(self.point_illuminate_status.values())


class RSOInspectionDataStore(DataStore):
    data_type = RSOInspectionData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_access_recorders = []
        self.illumination_recorders = []
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

    def add_point_access_recorder(self, recorder, illumination_recorder):
        self.point_access_recorders.append(recorder)
        self.illumination_recorders.append(illumination_recorder)
        self.satellite.simulator.AddModelToTask(
            self.satellite.dynamics.task_name, recorder, ModelPriority=1000
        )
        self.satellite.simulator.AddModelToTask(
            self.satellite.dynamics.task_name,
            illumination_recorder,
            ModelPriority=1000,
        )

    def clear_recorders(self):
        if self.storage_recorder:
            self.storage_recorder.clear()
        for recorder in self.point_access_recorders:
            recorder.clear()
        for recorder in self.illumination_recorders:
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

        illuminated_logs = []
        for recorder in self.illumination_recorders:
            illuminated_logs.append(list(np.array(recorder.hasAccess)))

        self.clear_recorders()

        return inspected_logs, illuminated_logs

    def compare_log_states(self, _, logs) -> Data:
        if self.role == RSO:
            return RSOInspectionData()

        inspected_logs, illuminated_logs = logs

        point_inspect_status = {}
        for rso_point, log in zip(
            self.data.point_inspect_status.keys(), inspected_logs
        ):
            if any(log):
                point_inspect_status[rso_point] = True

        point_illuminate_status = {}
        for rso_point, log in zip(
            self.data.point_illuminate_status.keys(), illuminated_logs
        ):
            if any(log):
                point_illuminate_status[rso_point] = True

        self.update_point_colors(
            self.data.point_illuminate_status.keys(),
            color="gray",
        )
        self.update_point_colors(
            [
                rso_point
                for rso_point in point_illuminate_status
                if point_illuminate_status[rso_point]
            ],
            color="yellow",
        )
        self.update_point_colors(
            [
                rso_point
                for rso_point in point_inspect_status
                if point_inspect_status[rso_point]
            ],
            color="chartreuse",
            permanent=True,
        )

        if len(point_inspect_status) > 0:
            self.satellite.logger.info(
                f"Inspected {len(point_inspect_status)} points this step"
            )

        return RSOInspectionData(point_inspect_status, point_illuminate_status)

    @vizard.visualize
    def update_point_colors(
        self,
        rso_points,
        color,
        alpha=0.5,
        permanent=False,
        vizInstance=None,
        vizSupport=None,
    ):
        """Update target colors in Vizard."""
        if not hasattr(self, "permanent_point_colors"):
            self.permanent_point_colors = []

        for location in vizInstance.locations:
            if (
                location.stationName not in self.permanent_point_colors
                and location.stationName in [str(point) for point in rso_points]
            ):
                if not all(
                    np.equal(location.color, vizSupport.toRGBA255(color, alpha=alpha))
                ):
                    location.color = vizSupport.toRGBA255(color, alpha=alpha)
                    if permanent:
                        self.permanent_point_colors.append(location.stationName)


class RSOInspectionReward(GlobalReward):
    datastore_type = RSOInspectionDataStore

    def __init__(
        self,
        inspection_reward_scale: float = 1.0,
        completion_bonus: float = 0.0,
        completion_threshold=0.90,
        min_illuminated_for_completion=0.4,
        min_time_for_completion=5700,
    ):
        super().__init__()
        self.completion_bonus = completion_bonus
        self.inspection_reward_scale = inspection_reward_scale
        self.completion_threshold = completion_threshold
        self.min_illuminated_for_completion = min_illuminated_for_completion
        self.min_time_for_completion = min_time_for_completion

    def reset_overwrite_previous(self) -> None:
        super().reset_overwrite_previous()
        self.bonus_reward_yielded = False
        self.bonus_reward_time = None

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
                    rso_point_model.accessOutMsgs[i].recorder(),
                    rso_point_model.illuminationOutMsgs[i].recorder(),
                )

    def initial_data(self, satellite: Satellite) -> Data:
        if not issubclass(satellite.dyn_type, RSOImagingDynModel):
            return RSOInspectionData()

        return RSOInspectionData(
            {point: False for point in self.scenario.rso_points},
            {point: False for point in self.scenario.rso_points},
        )

    def calculate_reward(self, new_data_dict: dict[str, Data]) -> dict[str, float]:
        total_points = len(self.scenario.rso_points)
        reward = {}
        total_data = self.data_type() + self.data
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
            total_data += data

        # Check for completion bonus
        min_illuminated_met = (
            total_data.num_points_illuminated
            > (len(self.scenario.rso_points) * self.min_illuminated_for_completion)
        ) or (
            self.scenario.satellites[0].simulator.sim_time
            > self.min_time_for_completion
        )

        imaged_fraction_met = (
            total_data.num_points_inspected
            > self.completion_threshold * total_data.num_points_illuminated
        )

        if (
            min_illuminated_met
            and imaged_fraction_met
            and not self.bonus_reward_yielded
        ):
            logger.info(
                f"{total_data.num_points_inspected} points inspected / {total_data.num_points_illuminated} illuminated! Awarding completion bonus."
            )
            for satellite_id in self.cum_reward:
                reward[satellite_id] = (
                    reward.get(satellite_id, 0.0) + self.completion_bonus
                )
            self.bonus_reward_yielded = True
            self.bonus_reward_time = self.scenario.satellites[0].simulator.sim_time
        return reward

    def is_terminated(self, satellite) -> bool:
        return self.bonus_reward_yielded

    # @vizard.visualize
    # def update_point_colors(self, rso_points, vizInstance=None, vizSupport=None):
    #     """Update target colors in Vizard."""
    #     for location in vizInstance.locations:
    #         if location.stationName in [str(point) for point in rso_points]:
    #             location.color = vizSupport.toRGBA255("tab:green", alpha=0.5)


__doc_title__ = "RSO Inspection"
__all__ = ["RSOInspectionReward", "RSOInspectionDataStore", "RSOInspectionData"]
