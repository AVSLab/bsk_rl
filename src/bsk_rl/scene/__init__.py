"""``bsk_rl.scene`` provides scenarios, or the underlying environment in which the satellite can collect data.

Scenarios typically correspond to certain type(s) of :ref:`bsk_rl.data` systems. The
following scenarios have been implemented:

* :class:`UniformTargets`: Uniformly distributed targets to be imaged by an :class:`~bsk_rl.sats.ImagingSatellite`.
* :class:`CityTargets`: Targets distributed near population centers.
* :class:`UniformNadirScanning`: Uniformly desireable data over the surface of the Earth.
"""

from bsk_rl.scene.scenario import Scenario, UniformNadirScanning
from bsk_rl.scene.targets import CityTargets, UniformTargets

__doc_title__ = "Scenario"
__all__ = [
    "Scenario",
    "UniformTargets",
    "CityTargets",
    "UniformNadirScanning",
]
