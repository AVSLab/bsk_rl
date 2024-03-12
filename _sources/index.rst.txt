BSK-RL: Environments and Algorithms for Spacecraft Planning and Scheduling
==========================================================================

.. toctree::
    :hidden:

    install
    Examples/index
    API Reference/index
    publications
    citation
    GitHub <https://github.com/AVSLab/bsk_rl/>


.. note::

    BSK-RL and its documentation are under active development. Please continue to check back for updates. 

.. note::
    
    New environments should be built using the :ref:`general satellite tasking framework <bsk_rl.envs.general_satellite_tasking>`; legacy environments are in the process of being ported to this framework.


**BSK-RL** (`Basilisk <https://hanspeterschaub.info/basilisk>`_ + `Reinforcement Learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_) is a Python package for constructing `Gymnasium <https://gymnasium.farama.org/index.html>`_ environments for spacecraft tasking problems. It is built on top of `Basilisk <https://hanspeterschaub.info/basilisk>`_, a modular and fast spacecraft simulation framework, making the simulation environments high-fidelity and computationally efficient. BSK-RL also includes a collection of agents, training scripts, and examples for working with these environments.

Quickstart
----------
Installation
^^^^^^^^^^^^
Complete installation instructions and common troubleshooting tips can be found :doc:`here <install>`. To install BSK-RL:

#. Install the `Basilisk <https://hanspeterschaub.info/basilisk>`_ spacecraft simulation framework.
#. Clone BSK-RL.

    .. code-block:: console

        $ git clone git@github.com:AVSLab/bsk_rl.git && cd bsk_rl

#. Install BSK-RL in the same virtual environment as Basilisk.

    .. code-block:: console

        (.venv) $ python -m pip install -e . && finish_install

#. Test the installation.

    .. code-block:: console

        (.venv) $ pytest ./tests/examples

Construct an Environment
^^^^^^^^^^^^^^^^^^^^^^^^
TODO: Add more detail to this example

.. code-block:: python

    import gymnasium as gym

    from bsk_rl.envs.general_satellite_tasking.scenario import data
    from bsk_rl.envs.general_satellite_tasking.scenario import satellites as sats
    from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import StaticTargets
    from bsk_rl.envs.general_satellite_tasking.simulation import environment
    from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=sats.FullFeaturedSatellite(
            "EO1", 
            sats.FullFeaturedSatellite.default_sat_args(oe=random_orbit), n_ahead_observe=30, 
            n_ahead_act=15
        ),
        env_type=environment.GroundStationEnvModel,
        env_args=environment.GroundStationEnvModel.default_env_args(),
        env_features=StaticTargets(n_targets=1000),
        data_manager=data.UniqueImagingManager,
        max_step_duration=600.0,
        time_limit=5700.0,
        terminate_on_time_limit=True,
    )

Train an Agent
^^^^^^^^^^^^^^
Show RLLib or SB3 configs here. 


Acknowledgements
----------------
BSK-RL is developed by the `Autonomous Vehicle Systems (AVS) Lab <https://hanspeterschaub.info/AVSlab.html>`_ at the University of Colorado Boulder. The AVS Lab is part of the `Colorado Center for Astrodynamics Research (CCAR) <https://www.colorado.edu/ccar>`_ and the `Department of Aerospace Engineering Sciences <https://www.colorado.edu/aerospace/>`_.

Development has been supported by NASA Space Technology Graduate Research Opportunity (NSTGRO) grants, 80NSSC20K1162 and 80NSSC23K1182. This work has also been supported by Air Force Research Lab grant FA9453-22-2-0050. 

Development of this software has utilized the Alpine high performance computing resource at the University of Colorado Boulder. Alpine is jointly funded by the University of Colorado Boulder, the University of Colorado Anschutz, and Colorado State University.