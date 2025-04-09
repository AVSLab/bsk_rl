"""Continuous actions set satellite behavior based on some continuous value."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from gymnasium import spaces

from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class ContinuousActionBuilder(ActionBuilder):
    def __init__(self, satellite: "Satellite") -> None:
        """Processes actions for a continuous action space.

        Args:
            satellite: Satellite to create actions for.
        """
        self.action_spec: list[ContinuousAction]
        super().__init__(satellite)
        assert len(self.action_spec) == 1, "Only one continuous action is supported."

    @property
    def _action(self) -> "ContinuousAction":
        return self.action_spec[0]

    @property
    def action_space(self) -> spaces.Box:
        """Continuous action space."""
        return self._action.space

    @property
    def action_description(self) -> list[str]:
        """Return a human-readable description of the continuous action space."""
        return self._action.action_description()

    def set_action(self, action: np.ndarray) -> None:
        """Activate the action by setting the continuous value."""
        self._action.set_action(action)


class ContinuousAction(Action):
    builder_type = ContinuousActionBuilder

    def __init__(self, name: str = "discrete_act") -> None:
        """Base class for actions with a continuous action space.

        Args:
            name: Name of the action.
        """
        super().__init__(name=name)

    @property
    @abstractmethod
    def space(self) -> spaces.Box:
        """Return the action space."""
        pass

    @property
    @abstractmethod
    def action_description(self) -> list[str]:
        """Return a description of the action space."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Activate an action by a continuous value."""
        pass
