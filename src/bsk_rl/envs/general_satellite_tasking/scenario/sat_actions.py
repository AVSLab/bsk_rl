"""Satellite action types can be used to add actions to the agents."""

from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
from gymnasium import spaces

from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import Target
from bsk_rl.envs.general_satellite_tasking.scenario.satellites import (
    ImagingSatellite,
    Satellite,
)
from bsk_rl.envs.general_satellite_tasking.utils.functional import bind, configurable


class SatAction(Satellite):
    """Base satellite subclass for composing actions."""

    pass


class DiscreteSatAction(SatAction):
    """Base satellite subclass for composing discrete actions."""

    def __init__(self, *args, **kwargs) -> None:
        """Construct satellite with discrete actions.

        Actions are added to the satellite for each DiscreteSatAction subclass, and can
        be accessed by index in order added.
        """
        super().__init__(*args, **kwargs)
        self.action_list = []
        self.action_map = {}

    def reset_pre_sim(self) -> None:
        """Reset the previous action key."""
        self.prev_action_key = None  # Used to avoid retasking of BSK tasks
        return super().reset_pre_sim()

    def add_action(
        self, act_fn, act_name: Optional[str] = None, n_actions: Optional[int] = None
    ):
        """Add an action to the action map.

        Args:
            act_fn: Function to call when selecting action. Takes as a keyword
                prev_action_key, used to avoid retasking of BSK models. Can accept an
                integer argument.
            act_name: String to refer to action.
            n_actions: If not none, add action n_actions times, calling it with an
                increasing integer argument for each subsequent action.
        """
        if act_name is None:
            act_name = act_fn.__name__

        if n_actions is None:
            self.action_map[f"{len(self.action_list)}"] = act_name
            self.action_list.append(act_fn)
        else:
            self.action_map[
                f"{len(self.action_list)}-{len(self.action_list)+n_actions-1}"
            ] = act_name
            for i in range(n_actions):
                act_i = self.generate_indexed_action(act_fn, i)
                act_i.__name__ = f"act_{act_fn.__name__}_{i}"
                self.action_list.append(bind(self, deepcopy(act_i)))

    def generate_indexed_action(self, act_fn, index: int):
        """Create an indexed action function.

        Makes an indexed action function from an action function that takes an index
        as an argument.

        Args:
            act_fn: Action function to index.
            index: Index to pass to act_fn.
        """

        def act_i(self, prev_action_key=None) -> Any:
            return getattr(self, act_fn.__name__)(
                index, prev_action_key=prev_action_key
            )

        return act_i

    def set_action(self, action: int):
        """Call action function my index."""
        self._disable_timed_terminal_event()
        self.prev_action_key = self.action_list[action](
            prev_action_key=self.prev_action_key
        )  # Update prev action data to avoid retasking

    @property
    def action_space(self) -> spaces.Discrete:
        """Infer action space."""
        return spaces.Discrete(len(self.action_list))


def fsw_action_gen(fsw_action: str, action_duration: float = 1e9) -> type:
    """Generate an action class for a FSW @action.

    Args:
        fsw_action: Function name of FSW action.
        action_duration: Time to task action for.

    Returns:
        Satellite action class with fsw_action action.
    """

    @configurable
    class FSWAction(DiscreteSatAction):
        def __init__(
            self, *args, action_duration: float = action_duration, **kwargs
        ) -> None:
            """Discrete action to perform a fsw action.

            Typically this is includes a function decorated by @action.

            Args:
                action_duration: Time to act when action selected. [s]
                args: Passed through to satellite
                kwargs: Passed through to satellite

            """
            super().__init__(*args, **kwargs)
            setattr(self, fsw_action + "_duration", action_duration)

            def act(self, prev_action_key=None) -> str:
                """Activate action.

                Returns:
                    action key
                """
                duration = getattr(self, fsw_action + "_duration")
                self.log_info(f"{fsw_action} tasked for {duration} seconds")
                self._disable_timed_terminal_event()
                self._update_timed_terminal_event(
                    self.simulator.sim_time + duration, info=f"for {fsw_action}"
                )
                if prev_action_key != fsw_action:
                    getattr(self.fsw, fsw_action)()
                return fsw_action

            act.__name__ = f"act_{fsw_action}"

            self.add_action(
                bind(self, act),
                act_name=fsw_action,
            )

    return FSWAction


# Charges the satellite
ChargingAction = fsw_action_gen("action_charge")

# Disables all actuators and control
DriftAction = fsw_action_gen("action_drift")

# Points in a specified direction while firing desat thrusters and desaturating wheels
DesatAction = fsw_action_gen("action_desat")

# Points nadir while downlinking data
DownlinkAction = fsw_action_gen("action_downlink")


@configurable
class ImagingActions(DiscreteSatAction, ImagingSatellite):
    """Satellite subclass to add upcoming target imaging to action space."""

    def __init__(self, *args, n_ahead_act=10, **kwargs) -> None:
        """Discrete action to image upcoming targets.

        Args:
            n_ahead_act: Number of actions to include in action space.
            args: Passed through to satellite
            kwargs: Passed through to satellite
        """
        super().__init__(*args, **kwargs)
        self.add_action(self.image, n_actions=n_ahead_act, act_name="image")

    def image(self, target: Union[int, Target, str], prev_action_key=None) -> str:
        """Activate imaging action.

        Args:
            target: Target, in terms of upcoming index, Target, or ID,
            prev_action_key: Previous action key

        Returns:
            Target ID
        """
        if np.issubdtype(type(target), np.integer):
            self.log_info(f"target index {target} tasked")

        target = self.parse_target_selection(target)
        if target.id != prev_action_key:
            self.task_target_for_imaging(target)
        else:
            self.enable_target_window(target)

        return target.id

    def set_action(self, action: Union[int, Target, str]):
        """Allow the satellite to be tasked by Target or target id.

        Allows for additional tasking modes in addition to action index-based tasking.
        """
        self._disable_image_event()
        if isinstance(action, (Target, str)):
            self.prev_action_key = self.image(action, self.prev_action_key)
        else:
            super().set_action(action)


NadirImagingAction = fsw_action_gen("action_nadir_scan")
