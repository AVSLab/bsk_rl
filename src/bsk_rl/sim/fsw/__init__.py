"""Basilisk flight software models (FSW) are given in ``bsk_rl.sim.fsw``.

Flight software models serve as the interface between the operation of the satellite in
simulation and the Gymnasium environment. While some FSW models add additional
functionality to the satellite, such as imaging instrument control in :class:`ImagingFSWModel`,
others replace the default control laws with a more complex algorithms, such as :class:`SteeringFSWModel`
vis a vis :class:`BasicFSWModel`.

Actions
-------

Each FSW model has a number of actions that can be called to task the satellite. These
actions are decorated with the :func:`~bsk_rl.sim.fsw.action` decorator, which performs
housekeeping tasks before the action is executed. These actions are the primary way to
control the satellite simulation from other parts of the Gymnasium environment.

Properties
----------

The FSW model provides a number of properties for easy access to the satellite state.
These can be accessed directly from the dynamics model instance, or in the observation
via the :class:`~bsk_rl.obs.SatProperties` observation.

Aliveness Checking
------------------

Certain functions in the FSW models are decorated with the :func:`~bsk_rl.utils.functional.aliveness_checker`
decorator. These functions are called at each step to check if the satellite is still
operational, returning true if the satellite is still alive.
"""

from bsk_rl.sim.fsw.base import BasicFSWModel, FSWModel, SteeringFSWModel, Task, action
from bsk_rl.sim.fsw.ground_imaging import (
    ContinuousImagingFSWModel,
    ImagingFSWModel,
    SteeringImagerFSWModel,
)
from bsk_rl.sim.fsw.orbital import MagicOrbitalManeuverFSWModel
from bsk_rl.sim.fsw.rso_inspection import RSOInspectorFSWModel

__doc_title__ = "FSW Sims"
__all__ = [
    "action",
    "FSWModel",
    "BasicFSWModel",
    "ImagingFSWModel",
    "ContinuousImagingFSWModel",
    "SteeringFSWModel",
    "SteeringImagerFSWModel",
    "MagicOrbitalManeuverFSWModel",
    "RSOInspectorFSWModel",
]
