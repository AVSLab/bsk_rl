"""Actions ``bsk_rl.act`` can be used to add actions to an agent.

To configure the observation, set the ``action_spec`` attribute of a :class:`~bsk_rl.sats.Satellite`
subclass. For example:

.. code-block:: python

    class MyActionSatellite(Satellite):
        action_spec = [
            Charge(duration=60.0),
            Desat(duration=30.0),
            Downlink(duration=60.0),
            Image(n_ahead_image=10),
        ]

Actions in an ``action_spec`` should all be of the same subclass of :class:`Action`. The
following actions are currently available:

Discrete Actions
----------------

Use :class:`DiscreteAction` for integer-indexable, discrete actions.

+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| **Action**                 |**Count**| **Description**                                                                                       |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`DiscreteFSWAction` | 1       | Call an arbitrary ``@action`` decorated function in the :class:`~bsk_rl.sim.fsw.FSWModel`.            |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Charge`            | 1       | Point the solar panels at the sun.                                                                    |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Drift`             | 1       | Do nothing.                                                                                           |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Desat`             | 1       | Desaturate the reaction wheels with RCS thrusters. Needs to be called multiple times.                 |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Downlink`          | 1       | Downlink data to any ground station that is in range.                                                 |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Image`             | ≥1      | Image one of the next ``N`` upcoming, unimaged targets once in range.                                 |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Scan`              | 1       | Scan nadir, collecting data when pointing within a threshold.                                         |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+

Continuous Actions
-------------------

Use :class:`ContinuousAction` for actions with a continuous action space. Currently, satellites
can only have a single continuous action in their ``action_spec``.

+-----------------------------+-------------+-------------------------------------------------------------------------------------------------------+
| **Action**                  |**Dimension**| **Description**                                                                                       |
+-----------------------------+-------------+-------------------------------------------------------------------------------------------------------+
| :class:`ImpulsiveThrust`    | 4           | Instantaneously change the satellite's velocity, and drift for some duration.                         |
+-----------------------------+-------------+-------------------------------------------------------------------------------------------------------+
| :class:`ImpulsiveThrustHill`| 4           | Like :class:`ImpulsiveThrust`, but specified in the Hill frame of another satellite.                  |
+-----------------------------+-------------+-------------------------------------------------------------------------------------------------------+



"""

from bsk_rl.act.actions import Action
from bsk_rl.act.continuous_actions import (
    ContinuousAction,
    ImpulsiveThrust,
    ImpulsiveThrustHill,
)
from bsk_rl.act.discrete_actions import (
    Broadcast,
    Charge,
    Desat,
    DiscreteAction,
    DiscreteFSWAction,
    Downlink,
    Drift,
    Image,
    Scan,
)

__doc_title__ = "Actions"
__all__ = [
    "Action",
    "DiscreteAction",
    "DiscreteFSWAction",
    "Charge",
    "Drift",
    "Desat",
    "Downlink",
    "Image",
    "Scan",
    "Broadcast",
    "ContinuousAction",
    "ImpulsiveThrust",
    "ImpulsiveThrustHill",
]
