"""``bsk_rl.sim`` is a package for the underlying Basilisk simulation.

The simulation is divided into three categories of Basilisk models:

* :ref:`bsk_rl.sim.world`, capturing elements of the simulation environment common to
  all satellites. This includes things such as gravity and atmosphere models, the epoch,
  and ground station locations. While the world model can be specified in the :class:`~bsk_rl.GeneralSatelliteTasking`
  constructor, it is generally automatically inferred from the satellite requirements.
* :ref:`bsk_rl.sim.dyn`, representing the dynamics model for each satellite. This is
  specified on a per-satellite basis by the :class:`~bsk_rl.sats.Satellite` type definition.
  The dynamics model captures the properties of the satellite, such as physical configurations,
  actuators models, instrument models, the power system, and storage resources.
* :ref:`bsk_rl.sim.fsw`, representing the flight software models for each satellite. As
  with flight software, this specified by the :class:`~bsk_rl.sats.Satellite`. The flight
  software model represents the low-level algorithms used for actuator and instrument
  control.

Generally, this can be thought of as a hierarchy of models, with dynamics models acting
in the world model, and flight software models controlling the dynamics models, and
other parts of ``bsk_rl`` controlling the flight software models. This hierarchy
contributes to the realism of the simulation, as the satellite is being controlled
through its flight software.

The :class:`~bsk_rl.sim.Simulator` is the main class for the simulation environment,
subclassing from the `Basilisk SimBaseClass <http://hanspeterschaub.info/basilisk/Documentation/utilities/SimulationBaseClass.html?highlight=simbaseclass#SimulationBaseClass.SimBaseClass>`_.
At each environment reset, the simulator and the associated flight software, dynamics,
and world models are deleted and reconstructed, generating a fresh Basilisk simulation.
"""

from bsk_rl.sim.simulator import Simulator

__doc_title__ = "Simulation (BSK)"
__all__ = [
    "Simulator",
]
