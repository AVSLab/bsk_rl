"""Basilisk dynamics models are given in ``bsk_rl.sim.dyn``.

The dynamics model is the core of the satellite simulation, representing the physical
properties of the satellite and its interactions with the environment. The dynamics model
can be customized to represent different satellite configurations, actuator models, and
instrument models.

The dynamics model is selected using the ``dyn_type`` class property of the
:class:`~bsk_rl.sats.Satellite`. Certain environment elements may require specific
dynamics models, such as :class:`~bsk_rl.comm.LOSCommunication` requiring a dynamics
model that inherits from :class:`~bsk_rl.sim.dyn.LOSCommDynModel` or :class:`~bsk_rl.sats.ImagingSatellite`
requiring a dynamics model that inherits from :class:`~bsk_rl.sim.dyn.ImagingDynModel`.

Setting Parameters
------------------

Customization of the dynamics model parameters is achieved through the ``sat_args``
dictionary passed to the :class:`~bsk_rl.sats.Satellite` constructor. This dictionary is
passed on to the dynamics model setup functions, which are called each time the simulator
is reset.

Properties
----------

The dynamics model provides a number of properties for easy access to the satellite state.
These can be accessed directly from the dynamics model instance, or in the observation
via the :class:`~bsk_rl.obs.SatProperties` observation.


Aliveness Checking
------------------

Certain functions in the dynamics model are decorated with the :func:`~bsk_rl.utils.functional.aliveness_checker`
decorator. These functions are called at each step to check if the satellite is still
operational, returning true if the satellite is still alive.

"""

from deprecated import deprecated

from bsk_rl.sim.dyn.base import BasicDynamicsModel, DynamicsModel
from bsk_rl.sim.dyn.ground_imaging import (
    ContinuousImagingDynModel,
    GroundStationDynModel,
    ImagingDynModel,
)
from bsk_rl.sim.dyn.relative_motion import (
    ConjunctionDynModel,
    LOSCommDynModel,
    MaxRangeDynModel,
)
from bsk_rl.sim.dyn.rso_inspection import RSODynModel, RSOInspectorDynModel


class FullFeaturedDynModel(GroundStationDynModel, LOSCommDynModel):
    """Convenience class for a satellite with ground station and line-of-sight comms."""

    @deprecated(
        "Create your own subclass of BasicDynamicsModel instead of using FullFeaturedDynModel."
    )
    def __init__(self, *args, **kwargs) -> None:
        """Convenience class for an imaging satellite with ground stations and line-of-sight communication."""
        super().__init__(*args, **kwargs)


__doc_title__ = "Dynamics Sims"
__all__ = [
    "DynamicsModel",
    "BasicDynamicsModel",
    "LOSCommDynModel",
    "ImagingDynModel",
    "ContinuousImagingDynModel",
    "GroundStationDynModel",
    "ConjunctionDynModel",
    "MaxRangeDynModel",
    "FullFeaturedDynModel",
    "RSODynModel",
    "RSOInspectorDynModel",
]
