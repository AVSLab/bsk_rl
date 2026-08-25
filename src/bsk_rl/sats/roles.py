"""Explicit spacecraft roles used by role-aware tasking environments."""

from enum import Enum


class SpacecraftRole(str, Enum):
    """Operational role of a propagated spacecraft.

    Roles are explicit metadata. Core tasking logic must not infer a role from a
    spacecraft name or its position in the simulator's satellite list.
    """

    SENSING_AGENT = "sensing_agent"
    PASSIVE_TARGET = "passive_target"


__all__ = ["SpacecraftRole"]
