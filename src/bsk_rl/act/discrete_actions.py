"""Discrete actions are indexable by integer."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from gymnasium import spaces

from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


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
            if -14.0 <= elev <= 90.0:
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
            if -14.0 <= elev <= 90.0 and tgt.id not in imaged_ids
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


        print_status = False
        if print_status:
            frequency_to_print = 0.1
            if round(self.satellite.simulator.sim_time, 9) % (frequency_to_print * 300) < 0.1:
                currently_visible_ids = []
                for target in known_targets:
                    target_pos = np.array(target.target_spacecraft.dynamics.r_BN_N)
                    elev = self.elevation_angle(scanner_pos, target_pos)
                    if -14.0 <= elev <= 90.0:
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
        run_heuristic_policy = False
        if run_heuristic_policy:
            imaged_ids = []
            for i in range(len(self.simulator.satellites[0].data_store.data.imaged)):
                imaged_ids.append(self.simulator.satellites[0].data_store.data.imaged[i].id)

            # Ensure final_targets has exactly n_actions entries
            # If still fewer (e.g., fewer total unimaged than n_actions), repeat last
            if len(final_targets) < num_actions:
                try:
                    final_targets += [final_targets[-1]] * (num_actions - len(final_targets))
                except IndexError:
                    sorted_fallback = sorted(
                    known_targets,
                    key=lambda tgt: np.linalg.norm(
                        np.array(tgt.target_spacecraft.dynamics.r_BN_N) - scanner_pos
                    )
                )
                    final_targets = sorted_fallback[:self.n_actions]
                    self.simulator.terminate = True

            new_target = final_targets[0] # always choosing the first of the targets in the observation space

        action_satid = new_target.id
        self.satellite.logger.info(f"target index {action_satid} tasked: {new_target.name}")
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration, info=""
        )
        prev_action_key = action_satid

        return self.image_rso(new_target, prev_action_key)


__doc_title__ = "Discrete Backend"
__all__ = ["DiscreteActionBuilder"]
