Examples
========

Environments
------------
.. toctree::
   :maxdepth: 1

   simple_environment
   satellite_configuration
   multiagent_envs
   fault_environment
   communication_action
   cloud_environment

Earth Observation
~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   cloud_environment
   cloud_environment_with_reimaging
   aeos

RSO Inspection
~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   rso_inspection

Training
--------
.. toctree::
   :maxdepth: 1

   rllib_training
   time_discounted_gae
   async_multiagent_training
   curriculum_learning
   training_with_shield

Benchmarks
----------

BSK-RL includes benchmark environments that can be used for training and evaluating RL
algorithms. These can be found in the ``benchmarks`` directory and trained using PPO
with

.. code-block:: bash

   python bsk_rl/benchmarks/benchmark.py -o results_dir -e nadir_science:nadir_science

To see a full list of options for the training script, run

.. code-block:: bash

   python bsk_rl/benchmarks/benchmark.py -h

Environments are specified in the format ``[file_name]:[env_name]``, where ``file_name``
is the name of a Python file in the ``benchmarks`` directory and ``env_name`` is the name
of an environment defined in that file. The environment includes both simulation and
training settings. The following environments are currently available:

* ``nadir_science:nadir_science``: A simple science environment with resource constraints.
* ``aeos:aeos_single``: A single-satellite agile Earth observation environment.
* ``aeos:aeos_constellation``: A multi-satellite agile Earth observation environment.
* ``rso_inspection:rso_inspection``: An RSO inspection environment with imaging constraints
  and safety constraints.