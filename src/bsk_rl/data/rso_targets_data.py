"""Data system for recording RSO images with configurable reimaging cooldown."""

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from Basilisk.utilities import orbitalMotion

from bsk_rl.data.base import Data, DataStore, GlobalReward

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


class RSOTargetImageData(Data):
    """Data for target images and cooldown eligibility."""

    def __init__(
        self,
        imaged: Optional[list["RSOTarget"]] = None,
        eclipsed: Optional[list["RSOTarget"]] = None,
        duplicates: int = 0,
        known: Optional[list["RSOTarget"]] = None,
        cooldown_until_by_id: Optional[dict[int, float]] = None,
        pending_image_records_by_id: Optional[dict[int, list[dict[str, Any]]]] = None,
        verified_useful_records: Optional[list[dict[str, Any]]] = None,
        verified_failed_records: Optional[list[dict[str, Any]]] = None,
        hide_pending_targets: bool = True,
    ) -> None:
        """Construct unit of data to record images and reimaging cooldown state.

        Keeps track of ``imaged`` targets, a count of ``duplicates`` (i.e. images that
        were duplicates in this data stream), all ``known`` targets in the environment,
        pending downlink-verification records, and cooldown deadlines for when a target
        becomes image-eligible again.

        Args:
            imaged: List of targets that are known to be imaged.
            duplicates: Count of target imaging duplication.
            known: List of targets that are known to exist (imaged and unimaged).
            cooldown_until_by_id: Mapping from ``target.id`` to cooldown end time [s].
            pending_image_records_by_id: Pending captured images waiting for downlink
                verification, keyed by ``target.id``.
            verified_useful_records: Records acknowledged as useful on downlink.
            verified_failed_records: Records acknowledged as failed/bad on downlink.
            hide_pending_targets: If true, pending targets are not image-eligible.
        """
        if imaged is None:
            imaged = []
        self.imaged = list(OrderedDict.fromkeys(imaged))  # Preserve order, remove duplicates
        self.duplicates = duplicates + len(imaged) - len(self.imaged)

        if known is None:
            known = []
        self.known = list(OrderedDict.fromkeys(known))  # Preserve order, remove duplicates

        if cooldown_until_by_id is None:
            cooldown_until_by_id = {}
        self.cooldown_until_by_id = {
            int(target_id): float(cooldown_until)
            for target_id, cooldown_until in cooldown_until_by_id.items()
        }
        self.pending_image_records_by_id = self._normalize_record_map(
            pending_image_records_by_id
        )
        self.verified_useful_records = self._dedupe_records(
            verified_useful_records or []
        )
        self.verified_failed_records = self._dedupe_records(
            verified_failed_records or []
        )
        self.hide_pending_targets = bool(hide_pending_targets)

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
        cooldown_until_by_id = dict(self.cooldown_until_by_id)
        for target_id, cooldown_until in other.cooldown_until_by_id.items():
            previous = cooldown_until_by_id.get(target_id, -np.inf)
            cooldown_until_by_id[target_id] = max(previous, cooldown_until)
        pending_image_records_by_id = self._merge_record_maps(
            self.pending_image_records_by_id,
            other.pending_image_records_by_id,
        )
        verified_useful_records = self._dedupe_records(
            self.verified_useful_records + other.verified_useful_records
        )
        verified_failed_records = self._dedupe_records(
            self.verified_failed_records + other.verified_failed_records
        )

        return self.__class__(
            imaged=imaged,
            duplicates=duplicates,
            known=known,
            cooldown_until_by_id=cooldown_until_by_id,
            pending_image_records_by_id=pending_image_records_by_id,
            verified_useful_records=verified_useful_records,
            verified_failed_records=verified_failed_records,
            hide_pending_targets=self.hide_pending_targets
            and other.hide_pending_targets,
        )

    @staticmethod
    def _target_id(target: "RSOTarget") -> int:
        """Extract target id regardless of object or raw id input."""
        return int(getattr(target, "id", target))

    @staticmethod
    def _record_key(record: dict[str, Any]) -> str:
        """Return a stable key for de-duplicating image records."""
        if record.get("record_id") is not None:
            return str(record["record_id"])
        return "|".join(
            str(record.get(key))
            for key in ("source_satellite", "target_id", "capture_time", "storage_index")
        )

    @classmethod
    def _dedupe_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Preserve order while removing duplicate records."""
        deduped = OrderedDict()
        for record in records:
            deduped[cls._record_key(record)] = dict(record)
        return list(deduped.values())

    @classmethod
    def _normalize_record_map(
        cls, record_map: Optional[dict[int, list[dict[str, Any]]]]
    ) -> dict[int, list[dict[str, Any]]]:
        """Normalize pending record mapping keys and de-duplicate records."""
        if record_map is None:
            return {}
        return {
            int(target_id): cls._dedupe_records(list(records))
            for target_id, records in record_map.items()
        }

    @classmethod
    def _merge_record_maps(
        cls,
        left: dict[int, list[dict[str, Any]]],
        right: dict[int, list[dict[str, Any]]],
    ) -> dict[int, list[dict[str, Any]]]:
        """Merge pending image records by target id."""
        merged = {int(target_id): list(records) for target_id, records in left.items()}
        for target_id, records in right.items():
            target_id = int(target_id)
            merged[target_id] = cls._dedupe_records(
                merged.get(target_id, []) + list(records)
            )
        return {target_id: records for target_id, records in merged.items() if records}

    def mark_target_cooldown(self, target: "RSOTarget", cooldown_until: float) -> None:
        """Mark target as ineligible until ``cooldown_until`` simulation time [s]."""
        target_id = self._target_id(target)
        previous = self.cooldown_until_by_id.get(target_id, -np.inf)
        self.cooldown_until_by_id[target_id] = max(previous, float(cooldown_until))

    def mark_target_imaged(self, target: "RSOTarget") -> None:
        """Record a target as verified imaged."""
        if target in self.imaged:
            self.duplicates += 1
            return
        self.imaged.append(target)

    def clear_target_cooldown(self, target: "RSOTarget") -> None:
        """Remove any cooldown for ``target`` so it can become eligible immediately."""
        target_id = self._target_id(target)
        self.cooldown_until_by_id.pop(target_id, None)

    def mark_target_pending(
        self, target: "RSOTarget", record: dict[str, Any]
    ) -> None:
        """Mark a captured image as pending downlink verification."""
        target_id = self._target_id(target)
        record = dict(record)
        record.setdefault("target_id", target_id)
        self.pending_image_records_by_id[target_id] = self._dedupe_records(
            self.pending_image_records_by_id.get(target_id, []) + [record]
        )

    def pop_pending_record(
        self, target: "RSOTarget"
    ) -> Optional[dict[str, Any]]:
        """Pop the oldest pending image record for ``target``."""
        target_id = self._target_id(target)
        records = self.pending_image_records_by_id.get(target_id, [])
        if not records:
            return None
        record = records.pop(0)
        if records:
            self.pending_image_records_by_id[target_id] = records
        else:
            self.pending_image_records_by_id.pop(target_id, None)
        return record

    def is_target_pending(self, target: "RSOTarget") -> bool:
        """Return True if ``target`` has pending images awaiting downlink."""
        target_id = self._target_id(target)
        return len(self.pending_image_records_by_id.get(target_id, [])) > 0

    def mark_record_verified(
        self, record: dict[str, Any], useful: bool
    ) -> None:
        """Store a verification result for bookkeeping."""
        if useful:
            self.verified_useful_records = self._dedupe_records(
                self.verified_useful_records + [record]
            )
        else:
            self.verified_failed_records = self._dedupe_records(
                self.verified_failed_records + [record]
            )

    def target_lifecycle_state(self, target: "RSOTarget", sim_time: float) -> str:
        """Return ``eligible``, ``pending_verification``, or ``cooldown``."""
        if self.hide_pending_targets and self.is_target_pending(target):
            return "pending_verification"
        target_id = self._target_id(target)
        if float(sim_time) < self.cooldown_until_by_id.get(target_id, -np.inf):
            return "cooldown"
        return "eligible"

    def is_target_eligible(self, target: "RSOTarget", sim_time: float) -> bool:
        """Return True if target can currently be selected for imaging."""
        return self.target_lifecycle_state(target, sim_time) == "eligible"

    def eligible_targets(
        self,
        sim_time: float,
        targets: Optional[list["RSOTarget"]] = None,
    ) -> list["RSOTarget"]:
        """Return currently eligible targets from ``targets`` or all known targets."""
        if targets is None:
            targets = self.known
        return [target for target in targets if self.is_target_eligible(target, sim_time)]


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
        self.cooldown_duration_s = 0.0
        self.verify_image_quality_on_downlink = False
        self.hide_pending_targets = True
        self.image_quality_threshold: Optional[float] = None

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
            if self.verify_image_quality_on_downlink:
                pending_records_by_id: dict[int, list[dict[str, Any]]] = {}
                for idx in update_idx:
                    # Capture only creates a pending packet. The onboard agent does not
                    # decide quality here; ground verification happens after downlink.
                    target_name = self._target_name_from_storage_index(idx)
                    target = self._target_from_name(target_name)
                    if target is None:
                        continue
                    record = self._pending_capture_record(
                        target, idx, old_state, new_state
                    )
                    pending_records_by_id.setdefault(int(target.id), []).append(record)
                return RSOTargetImageData(
                    pending_image_records_by_id=pending_records_by_id,
                    hide_pending_targets=self.hide_pending_targets,
                )

            imaged = []
            eclipsed = []
            for idx in update_idx:
                target_id = self._target_name_from_storage_index(idx)
                # imaged.append(
                #     [target for target in self.data.known if target.name == target_id and self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor < self.eclipse_threshold_for_imaging][0]
                # )

                for target in self.data.known:
                    if target.name == target_id:
                        if (
                            self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[
                                target.target_spacecraft.dynamics.eclipse_index
                            ]
                            .read()
                            .shadowFactor
                            > self.satellite.dynamics.eclipse_threshold_for_imaging
                        ):
                            imaged.append(target)
            # eclipse_threshold = 0.6
            # eclipsed.append([target for target in imaged if target.eclipse_status >= eclipse_threshold]) # this can be used to change reward in case the target is in eclipse
            # return RSOTargetImageData(imaged=imaged, eclipsed = eclipsed)
            return RSOTargetImageData(imaged=imaged)
        else:
            return RSOTargetImageData(imaged=[])

    def _target_name_from_storage_index(self, idx: int) -> str:
        """Return storage partition name for ``idx``."""
        message = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg
        return str(message.read().storedDataName[int(idx)])

    def _target_from_name(self, target_name: str):
        """Return known target with matching storage partition name."""
        return next((target for target in self.data.known if target.name == target_name), None)

    def _target_shadow_factor(self, target) -> float:
        """Read the current target eclipse shadow factor."""
        return float(
            self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[
                target.target_spacecraft.dynamics.eclipse_index
            ]
            .read()
            .shadowFactor
        )

    def _quality_threshold(self) -> float:
        """Return the image-quality threshold used for downlink verification."""
        if self.image_quality_threshold is not None:
            return float(self.image_quality_threshold)
        return float(self.satellite.dynamics.eclipse_threshold_for_imaging)

    def _pop_staged_capture_metadata(self, target_name: str) -> Optional[dict[str, Any]]:
        """Pop hold-gate metadata staged by the active imaging action."""
        pending_by_name = getattr(
            self.satellite, "_rso_pending_capture_metadata_by_name", None
        )
        if not pending_by_name:
            return None
        records = pending_by_name.get(target_name, [])
        if not records:
            return None
        record = dict(records.pop(0))
        if records:
            pending_by_name[target_name] = records
        else:
            pending_by_name.pop(target_name, None)
        return record

    def _pending_capture_record(
        self,
        target,
        idx: int,
        old_state: np.ndarray,
        new_state: np.ndarray,
    ) -> dict[str, Any]:
        """Build a pending-verification record for a newly captured image."""
        target_name = self._target_name_from_storage_index(idx)
        staged = self._pop_staged_capture_metadata(target_name)
        if staged is None:
            capture_shadow_factor = self._target_shadow_factor(target)
            capture_time = float(self.satellite.simulator.sim_time)
            staged = {
                "record_id": (
                    f"{self.satellite.name}:{target.id}:{capture_time:.9f}:"
                    f"{float(new_state[int(idx)]):.9f}"
                ),
                "target_id": int(target.id),
                "target_name": target_name,
                "capture_time": capture_time,
                "first_capture_time": capture_time,
                "capture_shadow_factor": capture_shadow_factor,
                "mean_hold_shadow_factor": None,
                "hold_valid_time_s": None,
                "source_satellite": self.satellite.name,
            }

        quality_value = staged.get("mean_hold_shadow_factor")
        if quality_value is None:
            quality_value = staged.get("capture_shadow_factor")
        quality_threshold = self._quality_threshold()

        staged.update(
            {
                "target_id": int(target.id),
                "target_name": target_name,
                "storage_index": int(idx),
                "storage_delta_bits": max(
                    0.0, float(new_state[int(idx)]) - float(old_state[int(idx)])
                ),
                "quality_threshold": quality_threshold,
                "quality_passed": (
                    bool(float(quality_value) >= quality_threshold)
                    if quality_value is not None
                    else False
                ),
                "verification_status": "pending",
            }
        )
        return staged



class RSOTargetImageReward(GlobalReward):
    """GlobalReward for rewarding unique images."""

    datastore_type = RSOTargetImageStore

    def __init__(
        self,
        reward_fn: Callable = lambda p: p,
        reimage_cooldown_orbits: float = 2.0,
        fallback_orbit_period_s: float = 95.0 * 60.0,
        verify_image_quality_on_downlink: bool = False,
        hide_pending_targets: bool = True,
        image_quality_threshold: Optional[float] = None,
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
        self.reimage_cooldown_orbits = float(reimage_cooldown_orbits)
        self.fallback_orbit_period_s = float(fallback_orbit_period_s)
        self.verify_image_quality_on_downlink = bool(verify_image_quality_on_downlink)
        self.hide_pending_targets = bool(hide_pending_targets)
        self.image_quality_threshold = image_quality_threshold
        self.orbit_period_s: Optional[float] = None
        self.reimage_cooldown_s: Optional[float] = None
        self.inspection_task_completed = False
        # self.downlink_bonus = 0.5 #deprecated
        # self.imaging_bonus = 0.5 #deprecated
        self.eclipse_threshold_for_reward = 0.5
        self.total_downlinks = 0
        self.useful_downlinks = 0
        self.failed_downlinks = 0
        self.bad_downlinks = 0
        self.pending_images = 0
        self.verified_image_count = 0
        self.reimage_count = 0
        self.verified_counts_by_id: dict[int, int] = {}
        self.imaging_rewarded_record_keys: set[str] = set()
        self.imaged_illuminated = []
        self.imaged_illuminated_names: set[str] = set()
        self.usefully_downlinked_names: set[str] = set()
        self.old_state: Optional[np.ndarray] = None

    def reset_overwrite_previous(self) -> None:
        """Reset episode-level bookkeeping and cooldown initialization."""
        super().reset_overwrite_previous()
        self.orbit_period_s = None
        self.reimage_cooldown_s = None
        self.old_state = None
        self.total_downlinks = 0
        self.useful_downlinks = 0
        self.failed_downlinks = 0
        self.bad_downlinks = 0
        self.pending_images = 0
        self.verified_image_count = 0
        self.reimage_count = 0
        self.verified_counts_by_id = {}
        self.imaging_rewarded_record_keys = set()
        self.imaged_illuminated = []
        self.imaged_illuminated_names = set()
        self.usefully_downlinked_names = set()

    def create_data_store(self, satellite: "Satellite") -> None:
        """Create datastore and attach a target eligibility access filter once."""
        super().create_data_store(satellite)
        satellite.data_store.verify_image_quality_on_downlink = (
            self.verify_image_quality_on_downlink
        )
        satellite.data_store.hide_pending_targets = self.hide_pending_targets
        satellite.data_store.image_quality_threshold = self.image_quality_threshold
        satellite.data_store.data.hide_pending_targets = self.hide_pending_targets

        if (
            hasattr(satellite, "add_access_filter")
            and not getattr(satellite, "_rso_reimage_filter_added", False)
        ):
            def reimage_target_filter(opportunity, sat=satellite):
                if opportunity["type"] != "target":
                    return True

                data_obj = getattr(sat.data_store, "data", None)
                if data_obj is None or not hasattr(data_obj, "is_target_eligible"):
                    return True

                sim_time = (
                    float(sat.simulator.sim_time)
                    if hasattr(sat, "simulator") and sat.simulator is not None
                    else 0.0
                )
                return data_obj.is_target_eligible(opportunity["object"], sim_time)

            satellite.add_access_filter(reimage_target_filter, types="target")
            satellite._rso_reimage_filter_added = True

    def _estimate_orbit_period_s(self) -> float:
        """Estimate scanner orbital period from current inertial state."""
        try:
            scanner = self.scenario.satellites[0]
            r_vec = np.array(scanner.dynamics.r_BN_N, dtype=float)
            v_vec = np.array(scanner.dynamics.v_BN_N, dtype=float)
            r_norm = float(np.linalg.norm(r_vec))
            v_norm = float(np.linalg.norm(v_vec))
            mu = float(orbitalMotion.MU_EARTH * 1e9)

            if r_norm <= 0.0 or mu <= 0.0:
                raise ValueError("Invalid norm(s) for period estimate.")

            specific_energy = 0.5 * (v_norm**2) - mu / r_norm
            if specific_energy >= 0.0:
                raise ValueError("Non-elliptic state cannot provide closed orbital period.")

            semi_major_axis = -mu / (2.0 * specific_energy)
            if semi_major_axis <= 0.0:
                raise ValueError("Invalid semi-major axis for period estimate.")

            return 2.0 * np.pi * np.sqrt((semi_major_axis**3) / mu)
        except Exception as exc:
            logger.warning(
                "Falling back to configured orbit period %.1fs due to: %s",
                self.fallback_orbit_period_s,
                exc,
            )
            return self.fallback_orbit_period_s

    def _ensure_cooldown_configured(self) -> None:
        """Ensure reimage cooldown is initialized and mirrored into each datastore."""
        if self.reimage_cooldown_s is None:
            self.orbit_period_s = self._estimate_orbit_period_s()
            self.reimage_cooldown_s = max(
                0.0, self.reimage_cooldown_orbits * self.orbit_period_s
            )

        for satellite in self.scenario.satellites:
            if hasattr(satellite, "data_store"):
                satellite.data_store.cooldown_duration_s = float(self.reimage_cooldown_s)

    def reset_post_sim_init(self) -> None:
        """Initialize cooldown timing once simulator state is available."""
        self._ensure_cooldown_configured()

    def initial_data(self, satellite: "Satellite") -> "RSOTargetImageData":
        """Furnish data to the scenario.

        Currently, it is assumed that all targets are known a priori, so the initial data
        given to the data store is the list of all targets.
        """
        return self.data_type(
            known=self.scenario.target_spacecrafts,
            hide_pending_targets=self.hide_pending_targets,
        )

    def _target_from_name(self, target_name: str):
        """Return scenario target matching a storage partition name."""
        return next(
            (target for target in self.scenario.target_spacecrafts if target.name == target_name),
            None,
        )

    def _target_from_id(self, target_id: int):
        """Return scenario target matching a target id."""
        return next(
            (
                target
                for target in self.scenario.target_spacecrafts
                if int(target.id) == int(target_id)
            ),
            None,
        )

    def _quality_threshold(self) -> float:
        """Return threshold for ground-side image quality verification."""
        if self.image_quality_threshold is not None:
            return float(self.image_quality_threshold)
        return float(self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward)

    def _record_quality_value(self, record: dict[str, Any]) -> Optional[float]:
        """Return the illumination value used to verify a pending image."""
        value = record.get("mean_hold_shadow_factor")
        if value is None:
            value = record.get("capture_shadow_factor")
        return None if value is None else float(value)

    def _record_quality_passed(self, record: dict[str, Any]) -> bool:
        """Return True if a pending image record passes ground verification."""
        value = self._record_quality_value(record)
        return value is not None and value >= self._quality_threshold()

    def _all_data_objects(self) -> list[RSOTargetImageData]:
        """Return global and satellite datastore objects that track lifecycle state."""
        data_objects = [self.data]
        for satellite in self.scenario.satellites:
            if hasattr(satellite, "data_store"):
                data_objects.append(satellite.data_store.data)
        return data_objects

    def _broadcast_new_pending_records(
        self, new_data_dict: dict[str, RSOTargetImageData]
    ) -> None:
        """Ensure pending capture records are visible to all lifecycle filters."""
        for new_data in new_data_dict.values():
            for target_id, records in new_data.pending_image_records_by_id.items():
                target = self._target_from_id(target_id)
                if target is None:
                    continue
                for data_obj in self._all_data_objects():
                    for record in records:
                        data_obj.mark_target_pending(target, record)
        self.pending_images = sum(
            len(records)
            for records in self.scenario.satellites[
                0
            ].data_store.data.pending_image_records_by_id.values()
        )

    def _reward_new_pending_illuminated_images(
        self,
        reward: dict[str, float],
        new_data_dict: dict[str, RSOTargetImageData],
    ) -> None:
        """Reward the imaging portion when a captured onboard image is already useful."""
        scanner = self.scenario.satellites[0]
        imaging_bonus = float(getattr(scanner.dynamics, "imaging_bonus", 0.0))
        if imaging_bonus == 0.0 or "SS1" not in reward:
            return

        for new_data in new_data_dict.values():
            for target_id, records in new_data.pending_image_records_by_id.items():
                target = self._target_from_id(target_id)
                if target is None:
                    continue
                for record in records:
                    record_key = RSOTargetImageData._record_key(record)
                    if record_key in self.imaging_rewarded_record_keys:
                        continue
                    if not self._record_quality_passed(record):
                        continue
                    reward["SS1"] += self.reward_fn(target.priority * imaging_bonus)
                    self.imaging_rewarded_record_keys.add(record_key)

    def _pop_pending_record_everywhere(
        self, target: "Target"
    ) -> Optional[dict[str, Any]]:
        """Remove the oldest pending image for ``target`` from all lifecycle stores."""
        record = None
        for data_obj in self._all_data_objects():
            popped = data_obj.pop_pending_record(target)
            if record is None and popped is not None:
                record = popped
        return record

    def _mark_verified_everywhere(
        self, target: "Target", record: dict[str, Any], useful: bool
    ) -> None:
        """Store downlink verification result in all lifecycle stores."""
        for data_obj in self._all_data_objects():
            data_obj.mark_record_verified(record, useful)
            if useful:
                data_obj.mark_target_imaged(target)

    @staticmethod
    def _record_capture_time(record: dict[str, Any], fallback_time: float) -> float:
        """Return the timestamp that should anchor reimage cooldown."""
        for key in ("capture_time", "first_capture_time", "start_time"):
            value = record.get(key)
            if value is not None:
                return float(value)
        return float(fallback_time)

    def _cooldown_until_from_capture(self, capture_time: float) -> float:
        """Return cooldown deadline anchored to the image capture time."""
        return float(capture_time) + float(self.reimage_cooldown_s)

    def _start_cooldown_everywhere(
        self, target: "Target", capture_time: float
    ) -> float:
        """Start reimage cooldown for a verified-useful target.

        Cooldown is intentionally anchored to the original capture timestamp, not the
        downlink acknowledgement. If the image was captured long before downlink, the
        cooldown may already be expired and the target will be eligible immediately.
        """
        cooldown_until = self._cooldown_until_from_capture(capture_time)
        for data_obj in self._all_data_objects():
            data_obj.mark_target_cooldown(target, cooldown_until)
        return cooldown_until

    def _clear_cooldown_everywhere(self, target: "Target") -> None:
        """Clear cooldown when downlink verifies that an image was not useful."""
        for data_obj in self._all_data_objects():
            data_obj.clear_target_cooldown(target)

    def _add_operational_penalties(self, reward: dict[str, float]) -> None:
        """Apply battery/storage penalties shared by both reward paths."""
        scanner = self.scenario.satellites[0]
        if scanner.dynamics.penalties != 1:
            return
        for sat_id in reward:
            if sat_id != "SS1":
                continue
            if scanner.dynamics.battery_charge_fraction < 0.05:
                reward[sat_id] += scanner.dynamics.low_battery_penalty
            elif scanner.dynamics.battery_charge_fraction < 0.1:
                reward[sat_id] += scanner.dynamics.low_battery_penalty
            if scanner.dynamics.storage_level_fraction > 0.991:
                reward[sat_id] += scanner.dynamics.full_storage_penalty

    def _publish_terminal_metrics(self) -> None:
        """Mirror rewarder metrics onto dynamics near the end of the episode."""
        scanner = self.scenario.satellites[0]
        if scanner.simulator.sim_time < (scanner.simulator.time_limit - 300):
            return
        scanner.dynamics.imaged_illuminated = len(self.imaged_illuminated)
        scanner.dynamics.total_downlinks = self.total_downlinks
        scanner.dynamics.useful_downlinks = self.useful_downlinks
        scanner.dynamics.failed_downlinks = self.failed_downlinks
        scanner.dynamics.bad_downlinks = self.bad_downlinks
        scanner.dynamics.reimage_count = self.reimage_count

    def _calculate_reward_with_downlink_verification(
        self, new_data_dict: dict[str, RSOTargetImageData]
    ) -> dict[str, float]:
        """Reward only after downlink verifies pending image quality."""
        scanner = self.scenario.satellites[0]
        if self.old_state is None:
            self.old_state = np.zeros_like(
                scanner.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
            )

        reward = {sat_id: 0.0 for sat_id in new_data_dict}
        self._add_operational_penalties(reward)
        self._broadcast_new_pending_records(new_data_dict)
        self._reward_new_pending_illuminated_images(reward, new_data_dict)

        sim_time = float(scanner.simulator.sim_time)
        storage_msg = scanner.dynamics.storageUnit.storageUnitDataOutMsg.read()
        new_state = np.array(storage_msg.storedData)
        # Only partitions whose stored data decreased are considered downlinked.
        # This keeps partial passes from verifying every target still in the buffer.
        downlinked_idxs = [int(i) for i in np.where(new_state - self.old_state < 0)[0]]
        if downlinked_idxs:
            self.total_downlinks += len(downlinked_idxs)

        for idx in downlinked_idxs:
            target_name = str(storage_msg.storedDataName[idx])
            target = self._target_from_name(target_name)
            if target is None:
                continue

            record = self._pop_pending_record_everywhere(target)
            if record is None:
                record = {
                    "record_id": f"unmatched:{target_name}:{sim_time:.9f}:{idx}",
                    "target_id": int(target.id),
                    "target_name": target_name,
                    "storage_index": int(idx),
                    "source_satellite": scanner.name,
                    "verification_status": "unmatched_downlink",
                }

            quality_passed = self._record_quality_passed(record)
            capture_time = self._record_capture_time(record, fallback_time=sim_time)
            cooldown_until = self._cooldown_until_from_capture(capture_time)
            record = dict(record)
            record.update(
                {
                    "capture_time": capture_time,
                    "downlink_time": sim_time,
                    "verification_status": "useful" if quality_passed else "failed",
                    "quality_threshold": self._quality_threshold(),
                    "quality_value": self._record_quality_value(record),
                    "quality_passed": bool(quality_passed),
                    "cooldown_start_time": capture_time if quality_passed else None,
                    "cooldown_until": cooldown_until if quality_passed else None,
                }
            )

            if quality_passed:
                self._mark_verified_everywhere(target, record, useful=True)
                self._start_cooldown_everywhere(target, capture_time)

                previous_count = self.verified_counts_by_id.get(int(target.id), 0)
                if previous_count > 0:
                    self.reimage_count += 1
                self.verified_counts_by_id[int(target.id)] = previous_count + 1

                self.verified_image_count += 1
                self.useful_downlinks += 1
                self.usefully_downlinked_names.add(target.name)
                self.imaged_illuminated.append(target)
                self.imaged_illuminated_names.add(target.name)
                if "SS1" in reward:
                    reward["SS1"] += self.reward_fn(
                        target.priority * scanner.dynamics.downlink_bonus
                    )
            else:
                self._mark_verified_everywhere(target, record, useful=False)
                self._clear_cooldown_everywhere(target)
                self.failed_downlinks += 1
                self.bad_downlinks = self.failed_downlinks

        self.pending_images = sum(
            len(records)
            for records in scanner.data_store.data.pending_image_records_by_id.values()
        )
        self._publish_terminal_metrics()
        self.old_state = np.array(storage_msg.storedData)
        return reward


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
        self._ensure_cooldown_configured()
        if self.verify_image_quality_on_downlink:
            return self._calculate_reward_with_downlink_verification(new_data_dict)

        if self.old_state is None:
            self.old_state = np.zeros_like(
                self.scenario.satellites[0].dynamics.storageUnit.storageUnitDataOutMsg.read().storedData
            )
        reward = {}
        imaged_targets = sum(
            [new_data.imaged for new_data in new_data_dict.values()], []
        )
        target_image_counts: dict[int, int] = {}
        for target in imaged_targets:
            target_id = int(target.id)
            target_image_counts[target_id] = target_image_counts.get(target_id, 0) + 1

        sim_time = float(self.scenario.satellites[0].simulator.sim_time)
        eligible_this_step = {
            int(target.id): self.data.is_target_eligible(target, sim_time)
            for target in imaged_targets
        }

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

                        # Keep your downlink reward per event as before
                        # (If you prefer to reward only once per target, gate on the same condition above.)
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
                    if (
                        eligible_this_step.get(int(target.id), True)
                        and self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[
                            target.target_spacecraft.dynamics.eclipse_index
                        ].read().shadowFactor
                        > self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward
                    ):
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
                target_id = int(target.id)
                if not eligible_this_step.get(target_id, True):
                    continue

                cooldown_until = sim_time + float(self.reimage_cooldown_s)
                new_data.mark_target_cooldown(target, cooldown_until)
                self.data.mark_target_cooldown(target, cooldown_until)
                for satellite in self.scenario.satellites:
                    if hasattr(satellite, "data_store"):
                        satellite.data_store.data.mark_target_cooldown(
                            target, cooldown_until
                        )

                if sat_id == 'SS1':
                    if self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor > self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward:
                        reward[sat_id] += self.reward_fn(
                            target.priority * self.scenario.satellites[0].dynamics.imaging_bonus  # this gives full reward as long as the shadowFactor was smaller than the eclipse_threshold_for_reward
                            # target.priority * self.scenario.satellites[0].dynamics.imaging_bonus * (self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor-self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward)/(self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward)  # this reward gives linearly scaled returns based on the actual value
                        ) / max(1, target_image_counts.get(target_id, 1))
                        # self.imaged_illuminated.append(target)
                        # self.imaged_illuminated_dict[sat_id]+=target
                    elif self.scenario.satellites[0].dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor < self.scenario.satellites[0].dynamics.eclipse_threshold_for_reward:
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
