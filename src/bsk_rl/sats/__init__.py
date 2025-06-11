"""``bsk_rl.sats`` defines satellite agents for the environments.

A reference for working with satellites is given below. For a step-by-step guide, see
`this example <../../examples/satellite_configuration.ipynb>`_.

Configuring a Satellite
-----------------------

A subclass of a :class:`Satellite` type must be defined before can be used as an agent.
Two fields (``observation_spec`` and ``action_spec``) must be specified: these define
the observation and action spaces for the satellite. :ref:`bsk_rl.act` and :ref:`bsk_rl.obs`
provide more information on specifying the observation and action spaces.

Two other fields (``dyn_model`` and ``fsw_model``) may be specified to select the
underlying dynamics and FSW models used by the Basilisk simulation. Some actions,
communication methods, or other environment configurations may necessitate the use of a
specific dynamics or FSW model. See :ref:`bsk_rl.sim.fsw` and :ref:`bsk_rl.sim.dyn` for
more information on selecting these models.

In practice, configuring a satellite and passing it to an environment is straightforward:

.. code-block:: python

    class MySatellite(Satellite):
        observation_spec = [obs.Time(), ...]  # list of observations
        action_spec = [act.Drift(), ...]  # list of actions
        dyn_model = MyDynamicsModel  # dynamics model type
        fsw_model = MyFSWModel  # FSW model type

    my_sat = MySatellite(name="my_satellite")
    env = gym.make("SatelliteTasking-v1", satellite=my_sat, ...)


Setting Satellite Parameters
----------------------------

To specify satellite parameters such as physical properties and controller gains, a
``sat_args`` dictionary can be passed to the satellite constructor, which in turn is
used when initializing the FSW and dynamics simulations. Call the class method
:class:`~Satellite.default_sat_args` to list what parameters are available:

.. code-block:: python

    >>> MySatellite.default_sat_args()

    {'mass': 100.0, 'Kp': 0.1, 'Ki': 0.01, 'Kd': 0.01, ...}


These parameters are documented in :ref:`bsk_rl.sim.fsw` and :ref:`bsk_rl.sim.dyn`. To
override the default parameters, pass a dictionary with the desired values. Parameters
can be set by value, or by passing a function that returns a value. In the latter case,
the randomizer function will be called each time the simulation is reset. For example:

.. code-block:: python

    >>> my_sat = MySatellite(
            name="my_satellite",
            sat_args={"mass": lambda: np.random.uniform(95.0, 105.0), "Kp": 0.3}
        )
    >>> env = gym.make("SatelliteTasking-v1", satellite=my_sat, ...)
    >>> env.reset()
    >>> my_sat.sat_args

    {'mass': 98.372, 'Kp': 0.3, 'Ki': 0.01, 'Kd': 0.01, ...}

If one attempts to set a parameter that is not recognized, an error will be raised.

Helpful Methods for Debugging
-----------------------------
A variety of methods are available for debugging and introspection:

* :class:`~Satellite.observation_description` - Returns a human-interpretable description
  of the observation. For array-type observations, this can be useful to map indices to
  specific observation elements.
* :class:`~Satellite.action_description` - Returns a human-interpretable description of
  the actions. For discrete actions, this will be a list of action names.
* :class:`~Satellite.observation_space` - Returns the observation space for the single agent.
* :class:`~Satellite.action_space` - Returns the action space for the single agent.
* :class:`~Satellite.is_alive` - Returns whether the satellite is still operational based on
  ``@aliveness_checker`` s in the FSW and dynamics simulators.

Helpful Methods for Extending
-----------------------------
When extending the satellite class, certain convenience methods are available:

* :class:`~Satellite.reset_pre_sim_init` - Called on reset before the simulation is constructed.
* :class:`~Satellite.reset_post_sim_init` - Called on reset  after the simulation is constructed.
* ``logger.info/warning/debug`` - Logs a message to ``INFO``, associating it with the satellite.

Satellite Varieties
-------------------
* :class:`AccessSatellite` - Provides methods for determining when a satellite has
  access to a ground location based on elevation angle. Can return ordered lists of
  upcoming opportunities.
* :class:`ImagingSatellite` - Extends :class:`AccessSatellite` to provide methods for
  interacting with :class:`bsk_rl.scene.targets.Target` objects.
"""

from bsk_rl.sats.access_satellite import AccessSatellite, ImagingSatellite
from bsk_rl.sats.satellite import Satellite

__doc_title__ = "Satellites"
__all__ = ["Satellite", "AccessSatellite", "ImagingSatellite"]
