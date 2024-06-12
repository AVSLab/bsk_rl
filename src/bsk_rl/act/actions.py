"""Satellite action types can be used to add actions to an agent."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from bsk_rl.utils.functional import AbstractClassProperty

if TYPE_CHECKING:  # pragma: no cover
    from gymnasium import spaces

    from bsk_rl.sats import Satellite
    from bsk_rl.sim import Simulator


def select_action_builder(satellite: "Satellite") -> "ActionBuilder":
    """Identify the proper action builder based on a satellite's action spec.

    Args:
        satellite: Satellite to build actions for.

    Returns:
        action builder of the appropriate type
    """
    builder_types = [spec.builder_type for spec in satellite.action_spec]
    if all([builder_type == builder_types[0] for builder_type in builder_types]):
        return builder_types[0](satellite)
    else:
        raise NotImplementedError("Heterogenous action builders not supported.")


class ActionBuilder(ABC):

    def __init__(self, satellite: "Satellite") -> None:
        """Base class for all action builders.

        Args:
            satellite: Satellite to build actions for.
        """
        self.satellite = satellite
        self.simulator: "Simulator"
        self.action_spec = deepcopy(self.satellite.action_spec)
        for act in self.action_spec:
            act.link_satellite(self.satellite)

    def reset_post_sim(self) -> None:
        """Perform any once-per-episode setup."""
        self.simulator = self.satellite.simulator  # already a proxy
        for act in self.action_spec:
            act.link_simulator(self.simulator)  # already a proxy
            act.reset_post_sim()

    @property
    @abstractmethod
    def action_space(self) -> "spaces.Space":
        """Return the action space."""
        pass

    @property
    @abstractmethod
    def action_description(self) -> Any:
        """Return a description of the action space."""
        pass

    @abstractmethod
    def set_action(self, action: Any) -> None:
        """Set the action to be taken."""
        pass


class Action(ABC):
    builder_type: type[ActionBuilder] = AbstractClassProperty()  #: :meta private:

    def __init__(self, name: str = "act") -> None:
        """Base class for all actions.

        Args:
            name: Name of the action.
        """
        self.name = name
        self.satellite: "Satellite"
        self.simulator: "Simulator"

    def link_satellite(self, satellite: "Satellite") -> None:
        """Link the action to a satellite.

        Args:
            satellite: Satellite to link to

        :meta private:
        """
        self.satellite = satellite  # already a proxy

    def link_simulator(self, simulator: "Simulator") -> None:
        """Link the action to a simulator.

        Args:
            simulator: Simulator to link to

        :meta private:
        """
        self.simulator = simulator  # already a proxy

    def reset_post_sim(self) -> None:  # pragma: no cover
        """Perform any once-per-episode setup."""
        pass

    @abstractmethod
    def set_action(self, action: Any) -> None:  # pragma: no cover
        """Execute code to perform an action."""
        pass


__doc_title__ = "Backend"
__all__ = ["ActionBuilder"]
