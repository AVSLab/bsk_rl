"""Discrete actions are indexable by integer."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from gymnasium import spaces

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

    def set_action(self, action: int) -> None:
        """Sets the action based on the integer index.

        If the action is not an integer, the satellite will attempt to call ``set_action_override``
        for each action, in order, until one works.
        """
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
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to transmit data from the data buffer (:class:`~bsk_rl.env.simulation.fsw.ImagingFSWModel.action_downlink`).

        If not in range of a ground station (defined in
        :class:`~bsk_rl.env.world.GroundStationWorldModel`), no data will
        be downlinked.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_downlink", name=name, duration=duration)


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
        # from bsk_rl.sats import ImagingSatellite
        #
        # self.satellite: "ImagingSatellite"
        super().__init__(name=name, n_actions=n_ahead_image)
        if duration is None:
            duration = 6000 #1e9
        self.duration = duration
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
        self.satellite.fsw.action_image_rso_target(target)

        return target.id


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
        imaged_targets = self.satellite.data_store.data.imaged

        imaged_ids = {tgt.id for tgt in imaged_targets}
        unimaged_targets = [tgt for tgt in known_targets if tgt.id not in imaged_ids]

        # Always update ever_visible with currently visible targets
        for target in known_targets:
            target_pos = np.array(target.target_spacecraft.dynamics.r_BN_N)
            elev = self.elevation_angle(scanner_pos, target_pos)
            if -21.0 <= elev <= 90.0:
                if target.id not in self.ever_visible:
                    self.ever_visible.append(target.id)

        # Compute elevation angles for unimaged targets
        target_elevations = []
        for target in unimaged_targets:
            target_pos = np.array(target.target_spacecraft.dynamics.r_BN_N)
            los_vector = target_pos - scanner_pos
            los_unit = los_vector / np.linalg.norm(los_vector)

            zenith = scanner_pos / np.linalg.norm(scanner_pos)
            cos_angle = np.clip(np.dot(los_unit, zenith), -1.0, 1.0)
            elevation_rad = np.arcsin(cos_angle)
            elev = np.degrees(elevation_rad)
            target_elevations.append((target, elev))

        visible_unimaged_targets = [
            (tgt, elev) for tgt, elev in target_elevations
            if -21.0 <= elev <= 90.0 and tgt.id not in imaged_ids
        ]

        visible_unimaged_targets.sort(key=lambda x: x[1])

        num_actions = self.n_actions
        final_targets = [tgt for tgt, _ in visible_unimaged_targets[:num_actions]]

        if len(final_targets) < num_actions:
            remaining = num_actions - len(final_targets)
            selected_ids = {tgt.id for tgt in final_targets}
            remaining_unimaged = [tgt for tgt in unimaged_targets if tgt.id not in selected_ids]
            remaining_unimaged.sort(
                key=lambda tgt: np.linalg.norm(np.array(tgt.target_spacecraft.dynamics.r_BN_N) - scanner_pos)
            )
            final_targets += remaining_unimaged[:remaining] # padding the array with the closest unimaged targets

        if len(final_targets) < num_actions:
            if len(final_targets) < 1:
                print("no new targets available!")
            try:
                final_targets += [final_targets[-1]] * (num_actions - len(final_targets))
            except IndexError:
                print('All targets imaged... No unimaged targets remaining')
                sorted_fallback = sorted(
                    known_targets,
                    key=lambda tgt: np.linalg.norm(
                        np.array(tgt.target_spacecraft.dynamics.r_BN_N) - scanner_pos
                    )
                )
                final_targets = sorted_fallback[:self.n_actions]
                self.simulator.terminate = True

        new_target = final_targets[action]
        policy_target = new_target
        if self.satellite.dynamics.print_info:
            if len(visible_unimaged_targets) !=0 and action < len(visible_unimaged_targets):
                print(f'chosen target elevation {visible_unimaged_targets[action][1]} and shadowFactor {self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[new_target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor}')
            else:
                print(f"currently no visible unimaged targets--> chosen target shadowFactor {self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[new_target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor}")
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
        print("_target_elevation_angle(self.satellite, opp): ",_target_elevation_angle(self.satellite, opp))

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
            print(
                    f"[UMBRA] smart={smart} "
                    f"scanner_sf={sc_sf:.3f} target_sf={tgt_sf:.3f} regime={regime} "
                    f"sun_az/el=({sun_az:.1f},{sun_el:.1f}) tgt_az/el=({azimuth:.1f},{elevation:.1f}) "
                    f"dot={dot_sun:.3f} sep_deg={sep_deg:.1f} dAz={daz:.1f}"
                )

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
                visible_unimaged_targets.sort(key=lambda x: x[0].id)
                for target, elev in visible_unimaged_targets:
                    if self.satellite.dynamics.world.eclipseObject.eclipseOutMsgs[target.target_spacecraft.dynamics.eclipse_index].read().shadowFactor < self.satellite.dynamics.eclipse_threshold_for_imaging:
                        currently_visible_ids_eclipsed.append(target.id)
                        currently_visible_ids_eclipsed_elevation.append(elev)

                all_ids = set(range(len(known_targets)))
                seen_ids = set(self.ever_visible)
                unimaged_ids = all_ids - imaged_ids
                never_seen = sorted(list(all_ids - seen_ids))

                print(f"\nSimulation Timestep: {self.satellite.simulator.sim_time}")
                print(f"Seen targets so far ({len(seen_ids)}): {sorted(seen_ids)}")
                print(f"Currently Visible targets ({len(currently_visible_ids)}): {currently_visible_ids}")
                if len(currently_visible_ids_eclipsed) != 0:
                    print(f"Currently Visible but Eclipse targets ({len(currently_visible_ids_eclipsed)}): {currently_visible_ids_eclipsed}")
                print(f"Imaged targets: ({len(imaged_ids)}): {sorted(imaged_ids)}")
                print(f"Unimaged targets: ({len(unimaged_ids)}): {sorted(unimaged_ids)}")
                print(f"Never seen targets ({len(never_seen)}): {never_seen} \n")

        run_heuristic_policy = self.satellite.dynamics.use_heuristic
        if run_heuristic_policy:
            print("using HEURISTIC POLICY")

            # Config knobs (with sane defaults)
            mode = getattr(self.satellite.dynamics, "heuristic_mode", "distance")  # 'distance' or 'angle'
            top_k = int(getattr(self.satellite.dynamics, "heuristic_top_k", 10))

            def _dist_to(tgt):
                tgt_pos = np.array(tgt.target_spacecraft.dynamics.r_BN_N)
                return np.linalg.norm(tgt_pos - scanner_pos)

            def _elev_of(tgt):
                tgt_pos = np.array(tgt.target_spacecraft.dynamics.r_BN_N)
                return self.elevation_angle(scanner_pos, tgt_pos)

            def _angle_err_of(tgt):
                # Use the same "initial angular error" metric you already record
                return float(_angle_to_target(self.satellite, {"object": tgt}))

            imaged_ids = []
            for i in range(len(self.simulator.satellites[0].data_store.data.imaged)):
                imaged_ids.append(self.simulator.satellites[0].data_store.data.imaged[i].id)

            # ---- Heuristic A: by distance (your current behavior) ----
            if mode == "distance":
                distances = [(tgt, _dist_to(tgt)) for tgt in unimaged_targets]
                distances.sort(key=lambda x: x[1])

                if not distances:
                    # Fall back to nearest known target (nothing left unimaged)
                    sorted_fallback = sorted(known_targets, key=_dist_to)
                    if not sorted_fallback:
                        raise RuntimeError("No targets available.")
                    heuristic_target = sorted_fallback[0]
                else:
                    top_list = [t for t, _ in distances[:max(1, top_k)]]
                    visible_candidates = [(t, _dist_to(t)) for t in top_list if -21.0 <= _elev_of(t) <= 90.0]
                    if visible_candidates:
                        visible_candidates.sort(key=lambda x: x[1])
                        heuristic_target = visible_candidates[0][0]
                    else:
                        heuristic_target = distances[0][0]

                new_target = heuristic_target
            # ---- Heuristic B: by current angle (smallest initial angular error) ----
            elif mode == "angle":
                # 1) visible + unimaged first
                visible_unimaged = []
                for tgt in unimaged_targets:
                    elev = _elev_of(tgt)  # LOS check via elevation
                    if -21.0 <= elev <= 90.0:
                        try:
                            aerr = _angle_err_of(tgt)
                        except Exception:
                            aerr = float("inf")
                        visible_unimaged.append((tgt, aerr))

                if visible_unimaged:
                    # Pick the visible unimaged target with the smallest angle
                    visible_unimaged.sort(key=lambda x: x[1])
                    heuristic_target = visible_unimaged[0][0]
                else:
                    # 2) none in LOS → pick overall smallest-angle *unimaged* target (even if not visible)
                    angle_list = []
                    for tgt in unimaged_targets:
                        try:
                            angle_list.append((tgt, _angle_err_of(tgt)))
                        except Exception:
                            pass

                    if angle_list:
                        angle_list.sort(key=lambda x: x[1])
                        heuristic_target = angle_list[0][0]
                    else:
                        # 3) no unimaged at all → fallback: best angle among known targets
                        known_angles = []
                        for tgt in known_targets:
                            try:
                                known_angles.append((tgt, _angle_err_of(tgt)))
                            except Exception:
                                pass
                        if known_angles:
                            known_angles.sort(key=lambda x: x[1])
                            heuristic_target = known_angles[0][0]
                        else:
                            # last resort: nearest by distance
                            heuristic_target = min(known_targets, key=_dist_to)

                new_target = heuristic_target
            else:
                raise ValueError(f"Unknown heuristic_mode '{mode}'. Use 'distance' or 'angle'.")

            # Keep your comparison & logging exactly as before
            self.satellite.dynamics.target_selection.append(policy_target)
            if policy_target.id == heuristic_target.id:
                self.satellite.dynamics.target_selection_comparison.append(heuristic_target.id)
            else:
                print(f"heuristic ({mode}) chose target: {heuristic_target.name}")
                self.satellite.dynamics.target_selection_comparison.append(False)

        action_satid = new_target.id
        self.satellite.logger.info(f"target index {action_satid} tasked: {new_target.name}")
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration, info=""
        )
        prev_action_key = action_satid

        return self.image_rso(new_target, prev_action_key)


__doc_title__ = "Discrete Backend"
__all__ = ["DiscreteActionBuilder"]
