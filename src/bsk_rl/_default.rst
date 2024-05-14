API Reference
=============

``bsk_rl`` is a framework for creating satellite tasking reinforcement learning environments. 
Three base environment classes are provided for configuring new environments:

+-------------------------------------+------------+---------------+--------------------------------------------------------------------+
| **Environment**                     | **API**    |**Agent Count**| **Purpose**                                                        |
+-------------------------------------+------------+---------------+--------------------------------------------------------------------+
| :class:`SatelliteTasking`           | Gymnasium  | 1             | Single-agent training; compatible with most RL libraries.          |
+-------------------------------------+------------+---------------+--------------------------------------------------------------------+
| :class:`GeneralSatelliteTasking`    | Gymnasium  | ≥1            | Multi-agent testing; actions and observations are given in tuples. |
+-------------------------------------+------------+---------------+--------------------------------------------------------------------+
| :class:`ConstellationTasking`       | PettingZoo | ≥1            | Multi-agent training; compatible with multiagent RL libraries.     |
+-------------------------------------+------------+---------------+--------------------------------------------------------------------+

Environments are customized by passing keyword arguments to the environment constructor.
When using ``gym.make``, the syntax looks like this:

.. code-block:: python

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=Satellite(...),
        scenario=UniformTargets(...),
        ...
    )

In some cases (e.g. the multiprocessed Gymnasium vector environment), it is necessary
for compatibility to instead register a new environment using the GeneralSatelliteTasking
class and a kwargs dict.

See the :ref:`examples` for more information on environment configuration arguments.

.. automodule:: bsk_rl
   :members:
   :show-inheritance:

.. toctree::
   :maxdepth: 1
   :caption: Modules
   :hidden:

   sats/index
   obs/index
   act/index
   scene/index
   data/index
   comm/index
   sim/index
   utils/index
