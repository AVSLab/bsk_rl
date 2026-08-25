"""Discrete actions are indexable by integer."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from gymnasium import spaces

from Basilisk.utilities import macros
from bsk_rl.utils.functional import valid_func_name

from bsk_rl.act.actions import Action, ActionBuilder

from bsk_rl.obs.observations import (
    _angle_to_target,
    _target_distance,
    _target_elevation_angle,
    _target_shadowFactor,
    _target_id_extracted,
    _relative_position_H
)


if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.rso_targets import RSOTarget
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)
R_EARTH_M = 6371e3
LEO_MAX_KM = 2000.0
MEO_MAX_KM = 35000.0
GEO_ALT_KM = 35786.0
GEO_TOL_KM = 300.0

LEO_MIN_KM = 400.0

# Smart-decision thresholds
SUNLIT_TAU = 0.5          # shadowFactor >= this => "illuminated"
UMBRA_TAU  = 0.5          # shadowFactor < this  => "in umbra"
SUN_ALIGN_DOT_TAU = 0.0   # dot(los_H, sun_H) >= this => "sunward"


def _select_highest_priority_candidate(candidate_targets):
    """Return the highest-priority unique target from an action candidate set."""
    unique_candidates = list(
        {candidate.id: candidate for candidate in candidate_targets}.values()
    )
    if not unique_candidates:
        raise RuntimeError("No candidate targets available.")

    def _priority_sort_key(target):
        try:
            priority = float(getattr(target, "priority", float("-inf")))
        except (TypeError, ValueError):
            priority = float("-inf")
        if not np.isfinite(priority):
            priority = float("-inf")
        return (-priority, int(target.id))

    return min(unique_candidates, key=_priority_sort_key)



class DiscreteActionBuilder(ActionBuilder):

    def __init__(self, satellite: "Satellite") -> None:
        """Processes actions for a discrete action space.

        Args:
            satellite: Satellite to create actions for.
        """
        super().__init__(satellite)
        self.prev_action_key = None

    def reset_post_sim_init(self) -> None:
        """Log previous action key."""
        super().reset_post_sim_init()
        self.prev_action_key = None
        self.satellite._current_action_label = ""

    @property
    def action_space(self) -> spaces.Discrete:
        """Discrete action space."""
        return spaces.Discrete(sum([act.n_actions for act in self.action_spec]))

    @property
    def action_description(self) -> list[str]:
        """Return a list of strings corresponding to action names."""
        actions = []
        for act in self.action_spec:
            if act.n_actions == 1:
                actions.append(act.name)
            else:
                actions.extend([f"{act.name}_{i}" for i in range(act.n_actions)])
        return actions

    def _set_current_action_label(self, action_spec: "DiscreteAction") -> None:
        """Expose a compact, visualization-safe name for the active FSW mode."""
        action_name = str(getattr(action_spec, "name", "")).lower()
        fsw_action = str(getattr(action_spec, "fsw_action", "")).lower()
        combined = f"{action_name} {fsw_action}"
        if "image" in combined:
            label = "Imaging"
        elif "downlink" in combined:
            label = "Downlink"
        elif "desat" in combined:
            label = "Desat"
        elif "charge" in combined:
            label = "Charge"
        elif "drift" in combined:
            label = "Drift"
        else:
            label = str(getattr(action_spec, "name", "Action"))
        self.satellite._current_action_label = label

    def set_action(self, action: int) -> None:
        """Sets the action based on the integer index.

        If the action is not an integer, the satellite will attempt to call ``set_action_override``
        for each action, in order, until one works.
        """
        active_downlink_action = getattr(self.satellite, "_active_downlink_action", None)
        if active_downlink_action is not None and hasattr(
            active_downlink_action, "_disable_downlink_empty_event"
        ):
            active_downlink_action._disable_downlink_empty_event()

        self.satellite.disable_timed_terminal_event()
        if not np.issubdtype(type(action), np.integer):
            logger.warning(
                f"Action '{action}' is not an integer. Will attempt to use compatible set_action_override method."
            )
            for act in self.action_spec:
                try:
                    self.prev_action_key = act.set_action_override(
                        action, prev_action_key=self.prev_action_key
                    )
                    self._set_current_action_label(act)
                    return
                except AttributeError:
                    pass
                except TypeError:
                    pass
            else:
                raise ValueError(
                    f"Action '{action}' is not an integer and no compatible set_action_override method found."
                )
        index = 0
        for act in self.action_spec:
            if index + act.n_actions > action:
                self.prev_action_key = act.set_action(
                    action - index, prev_action_key=self.prev_action_key
                )
                self._set_current_action_label(act)
                return
            index += act.n_actions
        else:
            raise ValueError(f"Action index {action} out of range.")


class DiscreteAction(Action):
    builder_type = DiscreteActionBuilder

    def __init__(self, name: str = "discrete_act", n_actions: int = 1):
        """Base class for discrete, integer-indexable actions.

        A discrete action may represent multiple indexed actions of the same type.

        Optionally, discrete actions may have a ``set_action_override`` function defined.
        If the action passed to the satellite is not an integer, the satellite will iterate
        over the ``action_spec`` and attempt to call ``set_action_override`` on each action
        until one is successful.

        Args:
            name: Name of the action.
            n_actions: Number of actions available.
        """
        super().__init__(name=name)
        self.n_actions = n_actions

    @abstractmethod
    def set_action(self, action: int, prev_action_key=None) -> str:
        """Activate an action by local index."""
        pass


class DiscreteFSWAction(DiscreteAction):
    def __init__(
        self,
        fsw_action,
        name=None,
        duration: Optional[float] = None,
        reset_task: bool = False,
    ):
        """Discrete action to task a flight software action function.

        This action executes a function of a :class:`~bsk_rl.env.simulation.fsw.FSWModel`
        instance that takes no arguments, typically decorated with ``@action``.

        Args:
            fsw_action: Name of the flight software function to task.
            name: Name of the action. If not specified, defaults to the ``fsw_action`` name.
            duration: Duration of the action in seconds. Defaults to a large value so that
                the :class:`~bsk_rl.env.gym_env.GeneralSatelliteTasking` ``max_step_duration``
                controls step length.
            reset_task: If true, reset the action if the previous action was the same.
                Generally, this parameter should be false to ensure realistic, continuous
                operation of satellite modes; however, some Basilisk modules may require
                frequent resetting for normal operation.
        """
        if name is None:
            name = fsw_action
        super().__init__(name=name, n_actions=1)
        self.fsw_action = fsw_action
        self.reset_task = reset_task
        if duration is None:
            duration = 1e9
        self.duration = duration

    def set_action(self, action: int, prev_action_key=None) -> str:
        """Activate the ``fsw_action`` function.

        Args:
            action: Should always be ``1``.
            prev_action_key: Previous action key.

        Returns:
            The name of the activated action.
        """
        assert action == 0
        self.satellite.logger.info(f"{self.name} tasked for {self.duration} seconds")
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration, info=f"for {self.fsw_action}"
        )

        if self.reset_task or prev_action_key != self.fsw_action:
            getattr(self.satellite.fsw, self.fsw_action)()

        return self.fsw_action


class Charge(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to enter a sun-pointing charging mode (:class:`~bsk_rl.env.simulation.fsw.BasicFSWModel.action_charge`).

        Charging will only occur if the satellite is in sunlight.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_charge", name=name, duration=duration)


class Drift(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to disable all FSW tasks (:class:`~bsk_rl.env.simulation.fsw.BasicFSWModel.action_drift`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_drift", name=name, duration=duration)


class BroadcastIntent(DiscreteAction):
    """Spend a finite action interval broadcasting typed catalog metadata.

    The associated communication method consumes only compact metadata explicitly
    staged by the sender. It never copies the sender's complete Python datastore.
    """

    def __init__(
        self,
        name: str = "action_broadcast_intent",
        duration: float = 30.0,
    ) -> None:
        if float(duration) <= 0.0:
            raise ValueError("Broadcast duration must be positive.")
        super().__init__(name=name, n_actions=1)
        self.duration = float(duration)

    def reset_post_sim_init(self) -> None:
        super().reset_post_sim_init()
        self.broadcast_pending = False

    def set_action(self, action: int, prev_action_key=None) -> str:
        assert action == 0
        self.broadcast_pending = True
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration,
            info="for metadata broadcast",
        )
        # Broadcasting occupies the sensor instead of silently continuing an imaging
        # or downlink mode from the previous command.
        if hasattr(self.satellite.fsw, "action_drift"):
            self.satellite.fsw.action_drift()
        return self.name


class Desat(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to desaturate reaction wheels (:class:`~bsk_rl.env.simulation.fsw.BasicFSWModel.action_desat`).

        This action must be called repeatedly to fully desaturate the reaction wheels.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(
            fsw_action="action_desat", name=name, duration=duration, reset_task=True
        )


class Downlink(DiscreteFSWAction):
    def __init__(
        self,
        name: Optional[str] = None,
        duration: Optional[float] = None,
        variable_duration_downlink: bool = False,
        empty_storage_threshold_bits: float = 0.1,
    ):
        """Action to transmit data from the data buffer (:class:`~bsk_rl.env.simulation.fsw.ImagingFSWModel.action_downlink`).

        If not in range of a ground station (defined in
        :class:`~bsk_rl.env.world.GroundStationWorldModel`), no data will
        be downlinked.

        Args:
            name: Action name.
            duration: Maximum time to task action, in seconds.
            variable_duration_downlink: If True, stop early once storage is empty.
            empty_storage_threshold_bits: Storage level treated as empty for early stop.
        """
        super().__init__(fsw_action="action_downlink", name=name, duration=duration)
        if empty_storage_threshold_bits < 0.0:
            raise ValueError("empty_storage_threshold_bits must be non-negative.")
        self.variable_duration_downlink = bool(variable_duration_downlink)
        self.empty_storage_threshold_bits = float(empty_storage_threshold_bits)
        self._downlink_empty_event_name = None
        # Prevent a zero-duration loop when downlink starts with empty storage.
        # The action must advance at least one FSW/sim cadence before it can stop early.
        self._downlink_earliest_stop_time = None

    def _storage_level_bits(self) -> float:
        """Return the current onboard storage level in bits."""
        try:
            return float(self.satellite.dynamics.storage_level)
        except Exception:
            msg = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
            stored_data = np.array(msg.storedData, dtype=float)
            return float(np.sum(np.maximum(stored_data, 0.0)))

    def _storage_empty(self) -> bool:
        """Return True when the satellite storage buffer is effectively empty."""
        return self._storage_level_bits() <= self.empty_storage_threshold_bits

    def _disable_downlink_empty_event(self) -> None:
        """Disable prior downlink-empty terminal event if present."""
        if (
            self._downlink_empty_event_name is not None
            and self._downlink_empty_event_name in self.simulator.eventMap
        ):
            self.simulator.delete_event(self._downlink_empty_event_name)
        self._downlink_empty_event_name = None
        self._downlink_earliest_stop_time = None
        if getattr(self.satellite, "_active_downlink_action", None) is self:
            self.satellite._active_downlink_action = None

    def _clear_downlink_state(self) -> None:
        """Clear the active downlink pointer after an early-stop event fires."""
        if getattr(self.satellite, "_active_downlink_action", None) is self:
            self.satellite._active_downlink_action = None
        self._downlink_earliest_stop_time = None

    def _downlink_empty_event_ready(self) -> bool:
        """Return True when empty-storage downlink termination is safe."""
        earliest_stop = self._downlink_earliest_stop_time
        if earliest_stop is None:
            return False
        return (
            float(self.simulator.sim_time) >= float(earliest_stop)
            and self._storage_empty()
        )

    def _enable_downlink_empty_event(self) -> None:
        """Terminate variable-duration downlink once storage reaches zero."""
        self._disable_downlink_empty_event()
        self.satellite._active_downlink_action = self
        min_duration_s = max(
            float(getattr(self.satellite.fsw, "fsw_rate", 0.0)),
            float(getattr(self.simulator, "sim_rate", 0.0)),
            # Avoid rapid retasking loops when storage is already empty.
            10.0,
        )
        # Even if storage is already empty, wait one cadence so env.step advances time.
        self._downlink_earliest_stop_time = (
            float(self.simulator.sim_time) + min_duration_s
        )
        self._downlink_empty_event_name = valid_func_name(
            f"downlink_empty_{self.satellite.name}_{self.simulator.sim_time}"
        )
        self.simulator.createNewEvent(
            self._downlink_empty_event_name,
            macros.sec2nano(self.satellite.fsw.fsw_rate),
            True,
            [
                f"{self.satellite._satellite_command}._active_downlink_action is not None and "
                f"{self.satellite._satellite_command}._active_downlink_action._downlink_empty_event_ready()"
            ],
            [
                self.satellite._info_command("downlink storage empty"),
                self.satellite._satellite_command + ".requires_retasking = True",
                f"[{self.satellite._satellite_command}._active_downlink_action._clear_downlink_state() "
                f"if {self.satellite._satellite_command}._active_downlink_action is not None else None]",
            ],
            terminal=self.satellite.variable_interval,
        )

    def set_action(self, action: int, prev_action_key=None) -> str:
        """Activate downlink and optionally stop early when storage is empty."""
        start_storage_level = self._storage_level_bits()
        self.satellite.dynamics.last_downlink_start_storage_level = start_storage_level
        self.satellite.dynamics.last_downlink_started_empty = (
            start_storage_level <= self.empty_storage_threshold_bits
        )
        action_key = super().set_action(action, prev_action_key=prev_action_key)
        if self.variable_duration_downlink:
            self._enable_downlink_empty_event()
        return action_key


class Scan(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to collect data from a :class:`~bsk_rl.scene.UniformNadirScanning` (:class:`~bsk_rl.sim.fsw.ContinuousImagingFSWModel.action_nadir_scan`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_nadir_scan", name=name, duration=duration)


class Image(DiscreteAction):
    def __init__(
        self,
        n_ahead_image: int,
        name: str = "action_image",
    ):
        """Actions to image upcoming target (:class:`~bsk_rl.env.simulation.fsw.ImagingFSWModel.action_image`).

        Adds ``n_ahead_image`` actions to the action space, corresponding to the next
        ``n_ahead_image`` unimaged targets. The action may be unsuccessful if the target
        exits the satellite's field of regard before the satellite settles on the target
        and takes an image. The action with stop as soon as the image is successfully
        taken, or when the the target exits the field of regard.

        This action implements a ``set_action_override`` that allows a target to be tasked
        based on the target's ID string or the Target object.

        Args:
            name: Action name.
            n_ahead_image: Number of unimaged, along-track targets to consider.
        """
        from bsk_rl.sats import ImagingSatellite

        self.satellite: "ImagingSatellite"
        super().__init__(name=name, n_actions=n_ahead_image)

    def image(
        self, target: Union[int, "Target", str], prev_action_key: Optional[str] = None
    ) -> str:
        """Task or retask a satellite for imaging a target.

        Args:
            target: Target to image.
            prev_action_key: Previous action key.

        :meta private:
        """
        target = self.satellite.parse_target_selection(target)
        if target.id != prev_action_key:
            self.satellite.task_target_for_imaging(target)
        else:
            self.satellite.enable_target_window(target)

        return target.id

    def set_action(self, action: int, prev_action_key: Optional[str] = None) -> str:
        """Image a target by local index.

        Args:
            action: Index of the target to image.
            prev_action_key: Previous action key.

        :meta_private:
        """
        self.satellite.logger.info(f"target index {action} tasked")
        return self.image(action, prev_action_key)

    def set_action_override(
        self, action: Union["Target", str], prev_action_key: Optional[str] = None
    ) -> str:
        """Image a target by target index, Target, or ID.

        Args:
            action: Target to image in the form of a Target object, target ID, or target index.
            prev_action_key: Previous action key.

        :meta_private:
        """
        return self.image(action, prev_action_key)


class ImageRSO(DiscreteAction):
    def __init__(
        self,
        n_ahead_image: int,
        name: str = "action_imageRSO",
        duration: Optional[float] = None,
        variable_duration_imaging: bool = True,
        min_pointing_hold_s: float = 10.0,
        hold_mode: str = "cumulative",
        require_illumination_during_hold: bool = True,
        hold_illumination_threshold: Optional[float] = None,
    ):
        """Actions to image upcoming target (:class:`~bsk_rl.env.simulation.fsw.ImagingFSWModel.action_image`).

        Adds ``n_ahead_image`` actions to the action space, corresponding to the next
        ``n_ahead_image`` unimaged targets. The action may be unsuccessful if the target
        exits the satellite's field of regard before the satellite settles on the target
        and takes an image. The action with stop as soon as the image is successfully
        taken, or when the the target exits the field of regard.

        This action implements a ``set_action_override`` that allows a target to be tasked
        based on the target's ID string or the Target object.

        Args:
            name: Action name.
            n_ahead_image: Number of unimaged, along-track targets to consider.
            duration: Maximum action duration [s].
            variable_duration_imaging: If True, imaging can terminate early on image
                success or target window close. If False, imaging uses fixed duration.
            min_pointing_hold_s: Minimum valid pointing hold time before image success [s].
            hold_mode: ``"cumulative"`` or ``"continuous"`` hold accumulation behavior.
            require_illumination_during_hold: Require target illumination while accumulating hold.
            hold_illumination_threshold: Override illumination threshold for hold checks.
        """
        # from bsk_rl.sats import ImagingSatellite
        #
        # self.satellite: "ImagingSatellite"
        super().__init__(name=name, n_actions=n_ahead_image)
        if duration is None:
            duration = 6000  # 1e9
        if min_pointing_hold_s < 0.0:
            raise ValueError("min_pointing_hold_s must be non-negative.")
        if hold_mode not in {"cumulative", "continuous"}:
            raise ValueError("hold_mode must be 'cumulative' or 'continuous'.")

        self.duration = float(duration)
        self.variable_duration_imaging = bool(variable_duration_imaging)
        self.min_pointing_hold_s = float(min_pointing_hold_s)
        self.hold_mode = hold_mode
        self.require_illumination_during_hold = bool(require_illumination_during_hold)
        self.hold_illumination_threshold = hold_illumination_threshold

        self._image_event_name = None
        self._hold_target = None
        self._hold_data_index = None
        self._hold_initial_data_level = None
        self._hold_last_eval_time = None
        self._hold_valid_time_s = 0.0
        self._hold_shadow_time_integral = 0.0
        self._capture_observed = False
        self._capture_passed_illum = False
        self._attempt_start_time = None
        self._attempt_target_id = None
        self._attempt_target_priority = None
        self._first_capture_time = None
        self._first_capture_shadow_factor = None
        self._attempt_recorded = False
        self._attempt_result_reason = None

        self.slew_time_success_s = []
        self.slew_time_unsuccessful_s = []
        self.imaging_attempt_records = []

        self.ever_visible=[]
        self.initial_angular_error = []
        self.chosen_target_distance = []
        self.chosen_target_elevation_angle = []
        self.chosen_target_illumination_status = []
        self.chosen_target_ids = []
        self.chosen_target_priority = []
        self.chosen_target_rel_pos_H = []
        self.chosen_target_rel_direction = []  # "ahead", "left", etc.
        self.chosen_target_alt_km = []
        self.chosen_target_orbit_regime = []
        self.chosen_target_azimuth = []
        self.chosen_target_elevation_local = []
        self.imaging_times = []

        # Sun / eclipse-alignment metrics (Hill frame)
        self.sun_azimuth = []
        self.sun_elevation_local = []
        self.sun_target_dot = []
        self.sun_target_sep_deg = []
        self.sun_target_daz_deg = []
        self.scanner_shadowFactor = []

        # "Smart vs regular" counts (only when scanner is in umbra)
        self.umbra_imaging_decisions = 0
        self.umbra_smart_decisions = 0
        self.umbra_regular_decisions = 0
        self.umbra_smart_reason_counts = {"illum_target": 0, "high_regime": 0, "sunward_leo": 0}



    def image_rso(
        self, target: Union[int, "RSOTarget", str], prev_action_key: Optional[str] = None
    ) -> str:
        """Task or retask a satellite for imaging a target.

        Args:
            target: Target to image.
            prev_action_key: Previous action key.

        :meta private:
        """
        self.satellite.fsw.action_image_rso_target(target)  #  TODO: here the data_name of the buffer should also be passsed  add ,target.target_spacecraft.name

        return target.id

    def _disable_image_success_event(self) -> None:
        """Disable prior image-success event if present."""
        if self._attempt_start_time is not None and not self._attempt_recorded:
            self._record_imaging_attempt(False, "event_disabled")
        if (
            self._image_event_name is not None
            and self._image_event_name in self.simulator.eventMap
        ):
            self.simulator.delete_event(self._image_event_name)
        self._image_event_name = None
        self._clear_hold_state()

    def _target_shadow_factor(self, target) -> float:
        return float(
            self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[
                target.target_spacecraft.dynamics.eclipse_index
            ]
            .read()
            .shadowFactor
        )

    def _hold_illum_threshold(self) -> float:
        if self.hold_illumination_threshold is not None:
            return float(self.hold_illumination_threshold)
        return float(self.satellite.dynamics.eclipse_threshold_for_imaging)

    def _reset_hold_state(self, target, data_index: int, initial_data_level: float) -> None:
        self._hold_target = target
        self._hold_data_index = int(data_index)
        self._hold_initial_data_level = float(initial_data_level)
        self._hold_last_eval_time = float(self.simulator.sim_time)
        self._hold_valid_time_s = 0.0
        self._hold_shadow_time_integral = 0.0
        self._capture_observed = False
        self._capture_passed_illum = False
        self._attempt_start_time = float(self.simulator.sim_time)
        self._attempt_target_id = target.id
        self._attempt_target_priority = float(getattr(target, "priority", np.nan))
        self._first_capture_time = None
        self._first_capture_shadow_factor = None
        self._attempt_recorded = False
        self._attempt_result_reason = None

        self.satellite._active_image_rso_action = self

    def _clear_hold_state(self) -> None:
        self._hold_target = None
        self._hold_data_index = None
        self._hold_initial_data_level = None
        self._hold_last_eval_time = None
        self._hold_valid_time_s = 0.0
        self._hold_shadow_time_integral = 0.0
        self._capture_observed = False
        self._capture_passed_illum = False
        self._attempt_start_time = None
        self._attempt_target_id = None
        self._attempt_target_priority = None
        self._first_capture_time = None
        self._first_capture_shadow_factor = None
        self._attempt_recorded = False
        self._attempt_result_reason = None

        if getattr(self.satellite, "_active_image_rso_action", None) is self:
            self.satellite._active_image_rso_action = None

    def _mean_hold_shadow_factor(self) -> Optional[float]:
        """Return average target illumination over valid hold time."""
        if self._hold_valid_time_s <= 0.0:
            return None
        return self._hold_shadow_time_integral / max(self._hold_valid_time_s, 1e-9)

    def _record_imaging_attempt(self, success: bool, reason: str = "") -> Optional[dict]:
        if self._attempt_recorded:
            return None
        if self._attempt_start_time is None:
            return None

        end_time = float(self.simulator.sim_time)
        if success and self._first_capture_time is not None:
            slew_time = max(0.0, self._first_capture_time - self._attempt_start_time)
        else:
            slew_time = max(0.0, end_time - self._attempt_start_time)

        mean_hold_shadow_factor = self._mean_hold_shadow_factor()
        capture_shadow_factor = self._first_capture_shadow_factor
        if self._hold_target is not None:
            try:
                if capture_shadow_factor is None:
                    capture_shadow_factor = self._target_shadow_factor(self._hold_target)
            except Exception:
                capture_shadow_factor = None
        quality_shadow_factor = (
            mean_hold_shadow_factor
            if mean_hold_shadow_factor is not None
            else capture_shadow_factor
        )
        quality_threshold = self._hold_illum_threshold()

        record = {
            "target_id": self._attempt_target_id,
            "target_priority": self._attempt_target_priority,
            "success": bool(success),
            "reason": reason,
            "start_time": float(self._attempt_start_time),
            "end_time": end_time,
            "first_capture_time": self._first_capture_time,
            "slew_time_s": float(slew_time),
            "capture_shadow_factor": capture_shadow_factor,
            "mean_hold_shadow_factor": mean_hold_shadow_factor,
            "hold_valid_time_s": float(self._hold_valid_time_s),
            "quality_threshold": quality_threshold,
            "quality_passed": (
                bool(quality_shadow_factor >= quality_threshold)
                if quality_shadow_factor is not None
                else False
            ),
        }
        self.imaging_attempt_records.append(record)
        if success:
            self.slew_time_success_s.append(float(slew_time))
        else:
            self.slew_time_unsuccessful_s.append(float(slew_time))

        self._attempt_recorded = True
        self._attempt_result_reason = reason
        return record

    def _stage_capture_metadata(
        self,
        data_index: int,
        initial_data_level: float,
        current_data_level: float,
    ) -> None:
        """Stage hold metadata for the datastore to attach to the captured image."""
        record = self._record_imaging_attempt(True, "hold_gate_success")
        if record is None or self._hold_target is None:
            return

        target_name = self._hold_target.target_spacecraft.name
        capture_time = (
            float(self._first_capture_time)
            if self._first_capture_time is not None
            else float(self.simulator.sim_time)
        )
        metadata = {
            **record,
            "record_id": (
                f"{self.satellite.name}:{self._hold_target.id}:"
                f"{capture_time:.9f}:{float(current_data_level):.9f}"
            ),
            "target_name": target_name,
            "capture_time": capture_time,
            "storage_index": int(data_index),
            "initial_data_level": float(initial_data_level),
            "current_data_level": float(current_data_level),
            "storage_delta_bits": max(
                0.0, float(current_data_level) - float(initial_data_level)
            ),
            "source_satellite": self.satellite.name,
        }

        # This handoff bridges the flight-software action and the data store. The
        # packet is pending verification until its named storage partition downlinks.
        pending_by_name = getattr(
            self.satellite, "_rso_pending_capture_metadata_by_name", None
        )
        if pending_by_name is None:
            pending_by_name = {}
            self.satellite._rso_pending_capture_metadata_by_name = pending_by_name
        pending_by_name.setdefault(target_name, []).append(metadata)

    def _pointing_constraints_ok(self, target) -> tuple[bool, float]:
        access_ok = False
        try:
            access_msg = self.satellite.dynamics.targetLocation.accessOutMsgs[target.id].read()
            access_ok = bool(access_msg.hasAccess)
        except Exception:
            access_ok = False

        att_ok = False
        rate_ok = True
        try:
            att_guid = self.satellite.fsw.attGuidMsg.read()
            sigma_norm = np.linalg.norm(np.array(att_guid.sigma_BR, dtype=float))
            att_ok = sigma_norm <= float(self.satellite.fsw.insControl.attErrTolerance)

            if bool(getattr(self.satellite.fsw.insControl, "useRateTolerance", 0)):
                omega_norm = np.linalg.norm(np.array(att_guid.omega_BR_B, dtype=float))
                rate_ok = omega_norm <= float(
                    self.satellite.fsw.insControl.rateErrTolerance
                )
        except Exception:
            att_ok = False
            rate_ok = False

        shadow_factor = self._target_shadow_factor(target)
        illum_ok = shadow_factor >= self._hold_illum_threshold()
        if not self.require_illumination_during_hold:
            illum_ok = True

        return (access_ok and att_ok and rate_ok and illum_ok), shadow_factor

    def _update_hold_timer(self) -> None:
        if self._hold_target is None:
            return

        now = float(self.simulator.sim_time)
        if self._hold_last_eval_time is None:
            self._hold_last_eval_time = now
            return

        dt = max(0.0, now - self._hold_last_eval_time)
        self._hold_last_eval_time = now
        if dt <= 0.0:
            return

        valid_now, shadow_factor = self._pointing_constraints_ok(self._hold_target)
        if valid_now:
            self._hold_valid_time_s += dt
            self._hold_shadow_time_integral += shadow_factor * dt
        elif self.hold_mode == "continuous":
            self._hold_valid_time_s = 0.0
            self._hold_shadow_time_integral = 0.0

    def _image_success_with_hold_gate(
        self, target_id, data_index: int, initial_data_level: float
    ) -> bool:
        if self._hold_target is None:
            return False
        if str(self._hold_target.id) != str(target_id):
            return False

        self._update_hold_timer()

        current_level = float(
            self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData[
                int(data_index)
            ]
        )
        if (current_level > float(initial_data_level)) and (not self._capture_observed):
            self._capture_observed = True
            self._first_capture_time = float(self.simulator.sim_time)
            self._first_capture_shadow_factor = self._target_shadow_factor(
                self._hold_target
            )
            self._capture_passed_illum = (
                self._first_capture_shadow_factor >= self._hold_illum_threshold()
            )

        if not self._capture_observed:
            return False
        if self.require_illumination_during_hold and not self._capture_passed_illum:
            return False

        if self.min_pointing_hold_s <= 0.0:
            self._stage_capture_metadata(data_index, initial_data_level, current_level)
            return True
        if self._hold_valid_time_s < self.min_pointing_hold_s:
            return False

        avg_shadow = self._mean_hold_shadow_factor()
        quality_passed = (
            avg_shadow is not None and avg_shadow >= self._hold_illum_threshold()
        )
        if self.require_illumination_during_hold and not quality_passed:
            return False

        self._stage_capture_metadata(data_index, initial_data_level, current_level)
        return True

    def _enable_image_success_event(self, target) -> None:
        """Terminate when image is captured and hold-gate criteria are satisfied."""
        self._disable_image_success_event()

        msg = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
        data_names = np.array(list(msg.storedDataName))
        data_name = target.target_spacecraft.name
        match_idx = np.where(data_names == data_name)[0]
        if len(match_idx) == 0:
            raise ValueError(
                f"Could not find storage buffer partition '{data_name}' for target {target}."
            )

        data_index = int(match_idx[0])
        current_data_level = float(msg.storedData[data_index])
        self._reset_hold_state(target, data_index, current_data_level)
        target_id_literal = repr(target.id)

        self._image_event_name = valid_func_name(
            f"image_rso_{self.satellite.name}_{target.id}"
        )
        self.simulator.createNewEvent(
            self._image_event_name,
            macros.sec2nano(self.satellite.fsw.fsw_rate),
            True,
            [
                f"{self.satellite._satellite_command}._active_image_rso_action is not None and "
                f"{self.satellite._satellite_command}._active_image_rso_action."
                f"_image_success_with_hold_gate({target_id_literal}, {data_index}, {current_data_level})"
            ],
            [
                self.satellite._info_command(f"imaged {target}"),
                self.satellite._satellite_command + ".requires_retasking = True",
                f"[{self.satellite._satellite_command}._active_image_rso_action._clear_hold_state() "
                f"if {self.satellite._satellite_command}._active_image_rso_action is not None else None]",
            ],
            terminal=self.satellite.variable_interval,
        )


    @staticmethod
    def elevation_angle(sat_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """
        Compute the elevation angle of a target relative to the local horizontal
        of the satellite.

        Args:
            sat_pos: Position of the satellite in inertial frame.
            target_pos: Position of the target in inertial frame.

        Returns:
            Elevation angle in degrees.
        """
        los_vector = target_pos - sat_pos
        los_unit = los_vector / np.linalg.norm(los_vector)

        # Local zenith (up) is aligned with position vector of satellite
        zenith = sat_pos / np.linalg.norm(sat_pos)

        # Elevation is angle between LOS and local horizontal (i.e., 90 - angle to zenith)
        cos_angle = np.clip(np.dot(los_unit, zenith), -1.0, 1.0)
        elevation_rad = np.arcsin(cos_angle)
        return np.degrees(elevation_rad)


    def sun_hat_chief(self) -> np.ndarray:
        """Unit sun direction vector expressed in the H frame (same convention as los_H)."""
        r_SN_N = (
            self.satellite.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
                self.satellite.simulator.world.sun_index
            ]
            .read()
            .PositionVector
        )
        sat_pos = np.asarray(self.satellite.dynamics.r_BN_N, dtype=float)
        r_SN_N = np.asarray(r_SN_N, dtype=float)

        # Inertial unit vector from spacecraft to sun
        r_SB_N = r_SN_N - sat_pos
        sun_N_hat = r_SB_N / np.linalg.norm(r_SB_N)

        # Use the same Hill triad you use for target LOS projection
        sat_vel = np.asarray(self.satellite.dynamics.v_BN_N, dtype=float)
        r_hat = sat_pos / np.linalg.norm(sat_pos)
        v_hat = sat_vel / np.linalg.norm(sat_vel)

        x_hat = r_hat
        z_hat = np.cross(r_hat, v_hat)
        z_hat /= np.linalg.norm(z_hat)
        y_hat = np.cross(z_hat, x_hat)

        sun_H = np.array([np.dot(sun_N_hat, x_hat), np.dot(sun_N_hat, y_hat), np.dot(sun_N_hat, z_hat)], dtype=float)
        sun_H /= np.linalg.norm(sun_H)
        return sun_H


    @staticmethod
    def az_el_from_H(u_H: np.ndarray) -> tuple[float, float]:
        """Return (az_deg, el_deg) from a unit vector in H frame."""
        u_H = np.asarray(u_H, dtype=float)
        u_H = u_H / np.linalg.norm(u_H)
        x, y, z = u_H
        el = float(np.degrees(np.arcsin(np.clip(x, -1.0, 1.0))))
        az = float(np.degrees(np.arctan2(z, y)) % 360.0)
        return az, el

    @staticmethod
    def wrap_deg180(a: float) -> float:
        """Wrap degrees to [-180, 180)."""
        return (a + 180.0) % 360.0 - 180.0


    @staticmethod
    def classify_regime_from_alt_km(alt_km: float) -> str:
        """Classify using your altitude-band definitions."""
        if (LEO_MIN_KM <= alt_km < LEO_MAX_KM):
            return "LEO"
        if (LEO_MAX_KM <= alt_km < MEO_MAX_KM):
            return "MEO"
        # GEO ring OR anything above MEO ceiling treated as GEO-like
        if abs(alt_km - GEO_ALT_KM) <= GEO_TOL_KM or alt_km >= MEO_MAX_KM:
            return "GEO"
        return "UNKNOWN"

    def _eligible_targets_now(self, known_targets):
        """Return currently image-eligible targets based on datastore lifecycle state."""
        data_obj = self.satellite.data_store.data
        sim_time = float(self.satellite.simulator.sim_time)
        if hasattr(data_obj, "eligible_targets"):
            return data_obj.eligible_targets(sim_time, known_targets)

        # Backward-compatible fallback if running with legacy data objects.
        imaged_targets = getattr(data_obj, "imaged", [])
        imaged_ids = {tgt.id for tgt in imaged_targets}
        return [tgt for tgt in known_targets if tgt.id not in imaged_ids]

    def set_action(self, action: int, prev_action_key: Optional[str] = None) -> str:
        """
        Image an unimaged target based on elevation angle from local horizontal.

        Args:
            action: Index of the target in elevation-filtered list.
            prev_action_key: Previous action key.

        Returns:
            Action result string.
        """
        scanner_pos = np.array(self.satellite.dynamics.r_BN_N)
        known_targets = self.satellite.data_store.data.known
        eligible_targets = self._eligible_targets_now(known_targets)
        eligible_ids = {tgt.id for tgt in eligible_targets}
        ever_imaged_ids = {
            tgt.id for tgt in getattr(self.satellite.data_store.data, "imaged", [])
        }

        # Always update ever_visible with currently visible targets
        for target in known_targets:
            target_pos = np.array(target.target_spacecraft.dynamics.r_BN_N)
            elev = self.elevation_angle(scanner_pos, target_pos)
            if -21.0 <= elev <= 90.0:
                if target.id not in self.ever_visible:
                    self.ever_visible.append(target.id)

        # Compute elevation angles for currently eligible targets
        target_elevations = []
        for target in eligible_targets:
            target_pos = np.array(target.target_spacecraft.dynamics.r_BN_N)
            los_vector = target_pos - scanner_pos
            los_unit = los_vector / np.linalg.norm(los_vector)

            zenith = scanner_pos / np.linalg.norm(scanner_pos)
            cos_angle = np.clip(np.dot(los_unit, zenith), -1.0, 1.0)
            elevation_rad = np.arcsin(cos_angle)
            elev = np.degrees(elevation_rad)
            target_elevations.append((target, elev))

        visible_eligible_targets = [
            (tgt, elev) for tgt, elev in target_elevations
            if -21.0 <= elev <= 90.0 and tgt.id in eligible_ids
        ]

        visible_eligible_targets.sort(key=lambda x: x[1])

        num_actions = self.n_actions
        final_targets = [tgt for tgt, _ in visible_eligible_targets[:num_actions]]

        if len(final_targets) < num_actions:
            remaining = num_actions - len(final_targets)
            selected_ids = {tgt.id for tgt in final_targets}
            remaining_eligible = [tgt for tgt in eligible_targets if tgt.id not in selected_ids]
            remaining_eligible.sort(
                key=lambda tgt: np.linalg.norm(np.array(tgt.target_spacecraft.dynamics.r_BN_N) - scanner_pos)
            )
            final_targets += remaining_eligible[:remaining]  # pad with closest eligible targets

        if len(final_targets) < num_actions:
            if len(final_targets) < 1:
                print("no eligible targets available!")
            try:
                final_targets += [final_targets[-1]] * (num_actions - len(final_targets))
            except IndexError:
                print("No eligible targets available; using closest known targets fallback")
                sorted_fallback = sorted(
                    known_targets,
                    key=lambda tgt: np.linalg.norm(
                        np.array(tgt.target_spacecraft.dynamics.r_BN_N) - scanner_pos
                    )
                )
                final_targets = sorted_fallback[:self.n_actions]
                if not final_targets:
                    raise RuntimeError("No targets available.")

        new_target = final_targets[action]
        policy_target = new_target

        # Heuristic target selection must happen before the commanded-target metrics
        # below are recorded.  Previously the heuristic override happened after those
        # metrics were appended, so heuristic plots described candidate slot zero even
        # when a different target was actually commanded.
        run_heuristic_policy = self.satellite.dynamics.use_heuristic
        if run_heuristic_policy:
            if self.satellite.dynamics.print_info:
                print("using HEURISTIC POLICY")

            mode = getattr(self.satellite.dynamics, "heuristic_mode", "distance")
            top_k = int(getattr(self.satellite.dynamics, "heuristic_top_k", 10))

            def _dist_to(tgt):
                tgt_pos = np.array(tgt.target_spacecraft.dynamics.r_BN_N)
                return np.linalg.norm(tgt_pos - scanner_pos)

            def _elev_of(tgt):
                tgt_pos = np.array(tgt.target_spacecraft.dynamics.r_BN_N)
                return self.elevation_angle(scanner_pos, tgt_pos)

            def _angle_err_of(tgt):
                return float(_angle_to_target(self.satellite, {"object": tgt}))

            if mode == "distance":
                distances = [(tgt, _dist_to(tgt)) for tgt in eligible_targets]
                distances.sort(key=lambda x: x[1])

                if not distances:
                    sorted_fallback = sorted(known_targets, key=_dist_to)
                    if not sorted_fallback:
                        raise RuntimeError("No targets available.")
                    heuristic_target = sorted_fallback[0]
                else:
                    top_list = [t for t, _ in distances[: max(1, top_k)]]
                    visible_candidates = [
                        (t, _dist_to(t))
                        for t in top_list
                        if -21.0 <= _elev_of(t) <= 90.0
                    ]
                    if visible_candidates:
                        visible_candidates.sort(key=lambda x: x[1])
                        heuristic_target = visible_candidates[0][0]
                    else:
                        heuristic_target = distances[0][0]

            elif mode in {"angle", "priority_angle"}:

                def _selection_score(tgt, angle_error):
                    if mode == "priority_angle":
                        priority = max(float(getattr(tgt, "priority", 0.0)), 1e-6)
                        return angle_error / priority
                    return angle_error

                visible_eligible = []
                for tgt in eligible_targets:
                    if -21.0 <= _elev_of(tgt) <= 90.0:
                        try:
                            aerr = _angle_err_of(tgt)
                        except Exception:
                            aerr = float("inf")
                        visible_eligible.append(
                            (tgt, _selection_score(tgt, aerr), aerr)
                        )

                if visible_eligible:
                    visible_eligible.sort(key=lambda x: (x[1], x[2], x[0].id))
                    heuristic_target = visible_eligible[0][0]
                else:
                    angle_list = []
                    for tgt in eligible_targets:
                        try:
                            aerr = _angle_err_of(tgt)
                            angle_list.append(
                                (tgt, _selection_score(tgt, aerr), aerr)
                            )
                        except Exception:
                            pass

                    if angle_list:
                        angle_list.sort(key=lambda x: (x[1], x[2], x[0].id))
                        heuristic_target = angle_list[0][0]
                    else:
                        known_angles = []
                        for tgt in known_targets:
                            try:
                                aerr = _angle_err_of(tgt)
                                known_angles.append(
                                    (tgt, _selection_score(tgt, aerr), aerr)
                                )
                            except Exception:
                                pass
                        if known_angles:
                            known_angles.sort(key=lambda x: (x[1], x[2], x[0].id))
                            heuristic_target = known_angles[0][0]
                        else:
                            heuristic_target = min(known_targets, key=_dist_to)

            elif mode == "candidate_priority":
                # Use exactly the same changing candidate set represented by the
                # imaging actions available to the RL policy.  Duplicate padding is
                # removed while preserving the candidate-set membership.
                heuristic_target = _select_highest_priority_candidate(final_targets)
            else:
                raise ValueError(
                    f"Unknown heuristic_mode '{mode}'. Use 'distance', 'angle', "
                    "'priority_angle', or 'candidate_priority'."
                )

            new_target = heuristic_target
            self.satellite.dynamics.target_selection.append(policy_target)
            if policy_target.id == heuristic_target.id:
                self.satellite.dynamics.target_selection_comparison.append(
                    heuristic_target.id
                )
            else:
                if self.satellite.dynamics.print_info:
                    print(f"heuristic ({mode}) chose target: {heuristic_target.name}")
                self.satellite.dynamics.target_selection_comparison.append(False)
            self.satellite.dynamics.last_policy_target_id = int(policy_target.id)
            self.satellite.dynamics.last_heuristic_target_id = int(
                heuristic_target.id
            )
            self.satellite.dynamics.last_imaging_selection_mode = "heuristic"
        else:
            self.satellite.dynamics.last_policy_target_id = int(policy_target.id)
            self.satellite.dynamics.last_heuristic_target_id = None
            self.satellite.dynamics.last_imaging_selection_mode = "policy"

        if self.satellite.dynamics.print_info:
            if len(visible_eligible_targets) !=0 and action < len(visible_eligible_targets):
                print(f'chosen target elevation {visible_eligible_targets[action][1]} and shadowFactor {self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[new_target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor}')
            else:
                print(f"currently no visible eligible targets--> chosen target shadowFactor {self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[new_target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor}")
        opp = {"object": new_target}

        # Append target ID (as integer)
        self.chosen_target_ids.append(new_target.id)

        # Append target priority if available
        priority = getattr(new_target, "priority", np.nan)
        self.chosen_target_priority.append(priority)

        # Append angular error (angle to target)
        self.initial_angular_error.append(_angle_to_target(self.satellite, opp))

        # Append target distance
        self.chosen_target_distance.append(_target_distance(self.satellite, opp))

        # Append elevation angle
        self.chosen_target_elevation_angle.append(_target_elevation_angle(self.satellite, opp))
        if self.satellite.dynamics.print_info:
            print(
                "_target_elevation_angle(self.satellite, opp): ",
                _target_elevation_angle(self.satellite, opp),
            )

        # Append illumination (eclipse shadow factor)
        try:
            self.chosen_target_illumination_status.append(_target_shadowFactor(self.satellite, opp))
        except Exception as e:
            print(f"Could not get shadow factor for target {new_target.id}: {e}")
            self.chosen_target_illumination_status.append(np.nan)

        # Append relative position in H frame
        rel_pos_H = _relative_position_H(self.satellite, opp)
        self.chosen_target_rel_pos_H.append(rel_pos_H)

        # Determine relative direction based on rel_pos_H and c_hat (body-frame instrument forward vector)
        c_hat_H = self.satellite.fsw.c_hat_P  # Assume this is in H frame (or transform if needed)

        # Normalize vectors
        c_hat_H = c_hat_H / np.linalg.norm(c_hat_H)
        rel_dir_unit = rel_pos_H / np.linalg.norm(rel_pos_H)

        # Compute dot and cross
        dot = np.dot(rel_dir_unit, c_hat_H)
        cross = np.cross(c_hat_H, rel_dir_unit)

        # Define thresholds for classification
        if dot > 0.7:
            direction = "ahead"
        elif dot < -0.7:
            direction = "behind"
        elif cross[2] > 0:
            direction = "left"
        else:
            direction = "right"

        self.chosen_target_rel_direction.append(direction)

        # LOS vector from satellite to target in inertial frame
        sat_pos = np.array(self.satellite.dynamics.r_BN_N)
        sat_vel = np.array(self.satellite.dynamics.v_BN_N)
        target_pos = np.array(new_target.target_spacecraft.dynamics.r_BN_N)

        r_norm = float(np.linalg.norm(target_pos))
        alt_km = (r_norm - R_EARTH_M) / 1000.0

        if alt_km < LEO_MAX_KM:
            regime = "LEO"
        elif alt_km < MEO_MAX_KM:
            regime = "MEO"
        else:
            regime = "GEO"

        self.chosen_target_alt_km.append(alt_km)
        self.chosen_target_orbit_regime.append(regime)

        r_hat = sat_pos / np.linalg.norm(sat_pos)
        v_hat = sat_vel / np.linalg.norm(sat_vel)

        x_hat = r_hat  # +zenith
        z_hat = np.cross(r_hat, v_hat)
        z_hat /= np.linalg.norm(z_hat)
        y_hat = np.cross(z_hat, x_hat)

        los = target_pos - sat_pos
        los /= np.linalg.norm(los)

        los_H = np.array([
            np.dot(los, x_hat),
            np.dot(los, y_hat),
            np.dot(los, z_hat),
        ])

        x, y, z = los_H

        elevation = np.degrees(np.arcsin(x))        # zenith-based elevation
        azimuth = np.degrees(np.arctan2(z, y)) % 360


        if self.satellite.dynamics.print_info:
            print(f"azimuth and elevation: {azimuth}, {elevation}")
        self.chosen_target_azimuth.append(azimuth)
        self.chosen_target_elevation_local.append(elevation)

        self.imaging_times.append(self.satellite.simulator.sim_time)

        # -----------------------------
        # Sun alignment + "smart decision" counting (only when scanner is in umbra)
        # -----------------------------

        # Scanner shadowFactor
        try:
            sc_idx = getattr(self.satellite.dynamics, "eclipse_index", None)
            sc_sf = float(self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[sc_idx].read().shadowFactor) if sc_idx is not None else float("nan")
        except Exception:
            sc_sf = float("nan")

        # Target shadowFactor at decision time
        try:
            tgt_sf = float(self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[
                new_target.target_spacecraft.dynamics.eclipse_index
            ].read().shadowFactor)
        except Exception:
            tgt_sf = float("nan")

        # Sun direction in H frame + az/el
        sun_H = self.sun_hat_chief()
        sun_az, sun_el = self.az_el_from_H(sun_H)

        # LOS already projected into H as los_H; normalize for safety
        los_H_u = los_H / np.linalg.norm(los_H)
        sun_H_u = sun_H / np.linalg.norm(sun_H)

        dot_sun = float(np.clip(np.dot(los_H_u, sun_H_u), -1.0, 1.0))
        sep_deg = float(np.degrees(np.arccos(dot_sun)))
        daz = self.wrap_deg180(sun_az - azimuth)

        # Store per-imaging-event sun metrics
        self.sun_azimuth.append(sun_az)
        self.sun_elevation_local.append(sun_el)
        self.sun_target_dot.append(dot_sun)
        self.sun_target_sep_deg.append(sep_deg)
        self.sun_target_daz_deg.append(daz)
        self.scanner_shadowFactor.append(sc_sf)

        # Regime classification (your altitude-band definition)
        regime = self.classify_regime_from_alt_km(alt_km)

        # SMART criteria during scanner umbra:
        #   (1) target illuminated, OR (2) target is MEO/GEO, OR (3) LEO but sunward
        if np.isfinite(sc_sf) and sc_sf < UMBRA_TAU:
            self.umbra_imaging_decisions += 1

            illum_target = np.isfinite(tgt_sf) and (tgt_sf >= SUNLIT_TAU)
            high_regime = (regime in {"MEO", "GEO"})
            sunward_leo = (regime == "LEO") and (dot_sun >= SUN_ALIGN_DOT_TAU)

            smart = bool(illum_target or high_regime or sunward_leo)
            if illum_target:
                self.umbra_smart_reason_counts["illum_target"] += 1
            if high_regime:
                self.umbra_smart_reason_counts["high_regime"] += 1
            if sunward_leo:
                self.umbra_smart_reason_counts["sunward_leo"] += 1

            if smart:
                self.umbra_smart_decisions += 1
            else:
                self.umbra_regular_decisions += 1

            if self.satellite.dynamics.print_info:
                print(
                    f"[UMBRA] smart={smart} "
                    f"scanner_sf={sc_sf:.3f} target_sf={tgt_sf:.3f} regime={regime} "
                    f"sun_az/el=({sun_az:.1f},{sun_el:.1f}) tgt_az/el=({azimuth:.1f},{elevation:.1f}) "
                    f"dot={dot_sun:.3f} sep_deg={sep_deg:.1f} dAz={daz:.1f}"
                )


        # sun_direction  = sun_hat_chief(self) #what is the azimuth of the sun direction  and how aligned is it with the choices made when in and around eclipse... I want to measure that if it is choosing a target when the scanning sc is around eclipse it should either be looking towards the sunny side of the eclipse or look at much higher up targets and others RSOs who have a shadowfactor of 1 either when choosing them or when aquiring their image.help me print these metrcis here or generate the metrics for the main script somehow.




        print_status = self.satellite.dynamics.print_info
        if print_status:
            frequency_to_print = 15
            if round(self.satellite.simulator.sim_time, 9) % (frequency_to_print * 300) < 0.1:
                currently_visible_ids = []
                for target in known_targets:
                    target_pos = np.array(target.target_spacecraft.dynamics.r_BN_N)
                    elev = self.elevation_angle(scanner_pos, target_pos)
                    if -21.0 <= elev <= 90.0:
                        currently_visible_ids.append(target.id)

                currently_visible_ids.sort()
                currently_visible_ids_eclipsed=[]
                currently_visible_ids_eclipsed_elevation=[]
                visible_eligible_targets.sort(key=lambda x: x[0].id)
                for target, elev in visible_eligible_targets:
                    if self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor < self.satellite.dynamics.eclipse_threshold_for_imaging:
                        currently_visible_ids_eclipsed.append(target.id)
                        currently_visible_ids_eclipsed_elevation.append(elev)

                all_ids = {target.id for target in known_targets}
                seen_ids = set(self.ever_visible)
                eligible_ids_now = {target.id for target in eligible_targets}
                data_obj = self.satellite.data_store.data
                pending_verification_ids = set()
                cooling_down_ids = set()
                if hasattr(data_obj, "target_lifecycle_state"):
                    sim_time = float(self.satellite.simulator.sim_time)
                    for target in known_targets:
                        state = data_obj.target_lifecycle_state(target, sim_time)
                        if state == "pending_verification":
                            pending_verification_ids.add(target.id)
                        elif state == "cooldown":
                            cooling_down_ids.add(target.id)
                else:
                    pending_by_id = getattr(data_obj, "pending_image_records_by_id", {})
                    pending_verification_ids = {
                        int(target_id)
                        for target_id, records in pending_by_id.items()
                        if records
                    }
                    cooldown_until_by_id = getattr(data_obj, "cooldown_until_by_id", {})
                    sim_time = float(self.satellite.simulator.sim_time)
                    cooling_down_ids = {
                        int(target_id)
                        for target_id, cooldown_until in cooldown_until_by_id.items()
                        if sim_time < float(cooldown_until)
                    }
                temporarily_ineligible_ids = all_ids - eligible_ids_now
                unexpected_cooldown_ids = cooling_down_ids - ever_imaged_ids
                never_seen = sorted(list(all_ids - seen_ids))

                print(f"\nSimulation Timestep: {self.satellite.simulator.sim_time}")
                print(f"Seen targets so far ({len(seen_ids)}): {sorted(seen_ids)}")
                print(f"Currently Visible targets ({len(currently_visible_ids)}): {currently_visible_ids}")
                if len(currently_visible_ids_eclipsed) != 0:
                    print(f"Currently Visible but Eclipse targets ({len(currently_visible_ids_eclipsed)}): {currently_visible_ids_eclipsed}")
                print(f"Ever-imaged targets: ({len(ever_imaged_ids)}): {sorted(ever_imaged_ids)}")
                print(f"Eligible targets now: ({len(eligible_ids_now)}): {sorted(eligible_ids_now)}")
                print(f"Pending-verification targets: ({len(pending_verification_ids)}): {sorted(pending_verification_ids)}")
                print(f"Cooling-down targets: ({len(cooling_down_ids)}): {sorted(cooling_down_ids)}")
                print(f"Temporarily ineligible targets: ({len(temporarily_ineligible_ids)}): {sorted(temporarily_ineligible_ids)}")
                if unexpected_cooldown_ids:
                    print(
                        "WARNING: true cooldown contains targets not in verified imaged list: "
                        f"{sorted(unexpected_cooldown_ids)}"
                    )
                print(f"Never seen targets ({len(never_seen)}): {never_seen} \n")

        # action_satid = new_target.id
        # self.satellite.logger.info(f"target index {action_satid} tasked: {new_target.name}")
        # self.satellite.update_timed_terminal_event(
        #     self.simulator.sim_time + self.duration, info=""
        # )
        # prev_action_key = action_satid
        #
        # return self.image_rso(new_target, prev_action_key)

        action_satid = new_target.id
        self.satellite.dynamics.last_imaging_target_id = int(action_satid)
        self.satellite.logger.info(f"target index {action_satid} tasked: {new_target.name}")

        # Remove stale success event from previous imaging task (if any)
        self._disable_image_success_event()

        # Task FSW for RSO imaging
        action_key = self.image_rso(new_target, action_satid)

        # Early finish condition (optional): end action as soon as selected target buffer increases
        if self.variable_duration_imaging:
            self._enable_image_success_event(new_target)

        # Fallback finish condition.
        # Fixed mode: (A) action duration only.
        # Variable mode: whichever is earlier between (A) duration and (B) target LOS window close.
        timeout_time = self.simulator.sim_time + self.duration
        timeout_info = f"for image_rso timeout ({self.duration:.1f}s)"

        if self.variable_duration_imaging:
            try:
                target_type = getattr(self.satellite, "target_types", "target")
                next_windows = self.satellite.next_opportunities_dict(
                    types=target_type,
                    filter=self.satellite.default_access_filter,
                )
                if new_target in next_windows:
                    window_close = next_windows[new_target][1]
                    if window_close < timeout_time:
                        timeout_time = window_close
                        timeout_info = f"for {new_target} window"
            except Exception:
                # Keep duration timeout fallback if windows are unavailable
                pass

        self.satellite.update_timed_terminal_event(
            timeout_time,
            info=timeout_info,
            extra_actions=[
                f"[getattr({self.satellite._satellite_command}, '_active_image_rso_action', None)._record_imaging_attempt(False, 'timeout_or_window') "
                f"if getattr({self.satellite._satellite_command}, '_active_image_rso_action', None) is not None else None]",
                f"[getattr({self.satellite._satellite_command}, '_active_image_rso_action', None)._clear_hold_state() "
                f"if getattr({self.satellite._satellite_command}, '_active_image_rso_action', None) is not None else None]",
            ],
        )

        return action_key



__doc_title__ = "Discrete Backend"
__all__ = ["DiscreteActionBuilder"]
