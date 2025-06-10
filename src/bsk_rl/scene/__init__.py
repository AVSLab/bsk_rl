"""``bsk_rl.scene`` provides scenarios, or the underlying environment in which the satellite can collect data.

Scenarios typically correspond to certain type(s) of :ref:`bsk_rl.data` systems.

For Earth observation, the following scenarios have been implemented:

* :class:`UniformTargets`: Uniformly distributed targets to be imaged by an :class:`~bsk_rl.sats.ImagingSatellite`.
* :class:`CityTargets`: Targets distributed near population centers.
* :class:`UniformNadirScanning`: Uniformly desireable data over the surface of the Earth.

For RSO Inspection tasks, the following scenario has been implemented:

* :class:`SphericalRSO`: A RSO with spherical points and radial normals.

These RSO scenarios can be used with :class:`RSOInspectionReward`.
"""

from bsk_rl.scene.rso_points import RSOPoints, SphericalRSO
from bsk_rl.scene.scenario import Scenario, UniformNadirScanning
from bsk_rl.scene.targets import CityTargets, UniformTargets

__doc_title__ = "Scenario"
__all__ = [
    "Scenario",
    "UniformTargets",
    "CityTargets",
    "UniformNadirScanning",
    "RSOPoints",
    "SphericalRSO",
]
