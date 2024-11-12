"""``bsk_rl.comm`` provides methods of communication for multisatellite environments.

Communication methods are used to induce collaborative behavior between satellites.
While the :class:`~bsk_rl.data.GlobalReward` acts as a global critic for the environment,
individual satellites may not have complete environmental knowledge; for example, in a
target imaging scenario, individual satellites do not know what requests have already
been fulfilled by other satellites. With communication, satellites can share data to
improve decision-making.

Communication works by sharing data between satellites, updating each other's local
knowledge of the scenario. After each environment step,
:class:`CommunicationMethod.communication_pairs` is evaluated to determine which pairs
of satellites should share data. Then, each local :class:`~bsk_rl.data.base.DataStore` is
updated with the other satellite's data.

Configuration
-------------

Communication methods can be configured by passing an instance of :class:`CommunicationMethod`
to the ``communicator`` field of the environment constructor.

.. code-block:: python

    env = ConstellationTasking(
        ...,
        communicator=LOSMultiCommunication(),
        ...
    )


Types of Communication
----------------------

* :class:`NoCommunication`: No communication between satellites.
* :class:`FreeCommunication`: Free communication between all satellites. This method is
  cheap to evaluate, and in scenarios with many satellites or tightly clustered
  satellites, it is often functionally equivalent to more complex models.
* :class:`LOSCommunication`: Line-of-sight communication between satellites. This method
  evaluates whether a direct line of sight exists between two satellites. If so, they
  can communicate.
* :class:`MultiDegreeCommunication`: This allows for "paths" of communication between
  satellites linked by some other method.
* :class:`LOSMultiCommunication`: A combination of :class:`LOSCommunication` and
  :class:`MultiDegreeCommunication` communication. This method allows for instantaneous
  communication by satellites that are obscured from each other but have a path of
  connected satellites acting as relays between them.
"""

from bsk_rl.comm.communication import (
    BroadcastCommunication,
    CommunicationMethod,
    FreeCommunication,
    LOSCommunication,
    LOSMultiCommunication,
    MultiDegreeCommunication,
    NoCommunication,
)

__doc_title__ = "Communication"
__all__ = [
    "CommunicationMethod",
    "NoCommunication",
    "FreeCommunication",
    "LOSCommunication",
    "MultiDegreeCommunication",
    "LOSMultiCommunication",
    "BroadcastCommunication",
]
