"""Data collection and reward calculation is given in ``bsk_rl.data``.

Reward System Components
------------------------

The reward system has three main components: :class:`GlobalReward`, :class:`~bsk_rl.data.base.DataStore`,
and :class:`~bsk_rl.data.base.Data`.

:class:`GlobalReward` acts as a global critic for the environment, rewarding each
agent's performance. Has full knowledge of the environment and can provide rewards
based on the global state of the environment, even if the agent does not have access
to that information; for example, you may not want to reward an agent for imaging a
target that has already been imaged by another agent, even if the agent does not know
that the target has been imaged. Reward is generally calculated by processing the
dictionary of new :class:`~bsk_rl.data.base.Data` per-satellite generated at each step
with the :class:`GlobalReward.calculate_reward` method.

The :class:`~bsk_rl.data.base.DataStore` handles each satellite's local knowledge of the
scenario and the data it generates. The data store gains data in three ways:

1. On environment reset, the :class:`~bsk_rl.data.GlobalReward` calls
   :class:`~bsk_rl.data.GlobalReward.initial_data` to provide the initial knowledge of the
   scenario for each satellite. This may be empty or may contain some a priori knowledge,
   such as a list of targets that are desired to be imaged.
2. At the end of each step, the result of :class:`~bsk_rl.data.base.DataStore.get_log_state`
   is compared to the previous step's result via :class:`~bsk_rl.data.base.DataStore.compare_log_states`.
   A unit of :class:`~bsk_rl.data.base.Data` is returned. For example, the log state may
   be the level of each target's buffer partition in the storage unit, so a change in
   a certain buffer level leads to a unit of data that indicates the corresponding target
   has been imaged.
3. At the end of each step, satellites communicate based on the :ref:`bsk_rl.comm`
   system being used. Satellites merge the contents of their data stores with any other
   satellite's data store that they have communicated with.

Finally, :class:`~bsk_rl.data.base.Data` can represent data generated by the satellite
towards some goal (e.g. images of targets, time spend in a desireable mode, etc.) as well
as information about the environment that is useful toward completing its mission (e.g.
desired targets to image, what targets have already been imaged, etc.).

Implementing Data & Reward Types
================================

See :ref:`bsk_rl.data.base` for full documentation of the reward system components to
when implementing custom data and reward types.

Reward System Types
-------------------

A variety of reward systems are available for use in the environment. The following table
provides a summary of the available reward systems:

+-----------------------------+-------------------------------------------------------------------------+---------------------------------------------------------------------+
| **Type**                    | **Purpose**                                                             | **Compatibility**                                                   |
+-----------------------------+-------------------------------------------------------------------------+---------------------------------------------------------------------+
| :class:`NoReward`           | Returns zero reward for every agent at every step.                      |                                                                     |
+-----------------------------+-------------------------------------------------------------------------+---------------------------------------------------------------------+
| :class:`UniqueImageReward`  | Returns reward corresponding to target priority the                     | Should be used with :class:`~bsk_rl.sats.ImagingSatellite` and a    |
|                             | first time a target is imaged by any agent. Causes                      | :class:`~bsk_rl.scene.Target`-based scenario.                       |
|                             | satellites to filter targets that are known to have                     |                                                                     |
|                             | been imaged already.                                                    |                                                                     |
+-----------------------------+-------------------------------------------------------------------------+---------------------------------------------------------------------+
| :class:`ScanningTimeReward` | Returns reward based on time spend in the nadir-pointing scanning mode. | Should be used with the :class:`~bsk_rl.scene.UniformNadirScanning` |
|                             |                                                                         | scenario.                                                           |
+-----------------------------+-------------------------------------------------------------------------+---------------------------------------------------------------------+

To select a reward system to use, pass an instance of :class:`GlobalReward` to the ``data``
field of the environment constructor:

.. code-block:: python

    env = ConstellationTasking(
        ...,
        data=ScanningTimeReward(),
        ...
    )

"""

from bsk_rl.data.base import GlobalReward
from bsk_rl.data.nadir_data import ScanningTimeReward
from bsk_rl.data.no_data import NoReward
from bsk_rl.data.unique_image_data import UniqueImageReward

__doc_title__ = "Data & Reward"
__all__ = [
    "GlobalReward",
    "NoReward",
    "UniqueImageReward",
    "ScanningTimeReward",
]
