Release Notes
=============

Development - |version|
-----------------------
.. *Release Date: MMM. DD, YYYY*
* Add an example script where reward is based on the probability of successfully observing targets covered 
  by clouds in the `Cloud Environment with Re-imaging <examples/cloud_environment_with_reimaging.ipynb>`_ example.
* Add a conjunction checking dynamics model in :class:`~bsk_rl.dynamics.ConjunctionDynModel`.
* Add utilities for relative motion state setup, :class:`~bsk_rl.utils.orbital.cd2hill`, :class:`~bsk_rl.utils.orbital.hill2cd`,
  and :class:`~bsk_rl.utils.orbital.relative_to_chief`.
* Add a ``dtype`` argument to the environment (or individual satellites) and sets the default
  dtype to ``np.float64``.
* Add support for continuous action spaces (e.g. for control problems) with :class:`~bsk_rl.act.ContinuousAction`.
* Add models and action for impulsive thrust and drift with a continuous action space (:class:`~bsk_rl.act.ImpulsiveThrust`).
* Changed inconsistent uses of ``datastore`` to ``data_store``.
* Added property ``data_store_kwargs`` to :class:`~bsk_rl.data.GlobalReward` that is unpacked in the
  :class:`~bsk_rl.data.DataStore` constructor.
* Implemented :class:`~bsk_rl.data.ResourceReward` to reward based on the level of a property in the satellite
  multiplied by some coefficient.
* Allow rewarders to mark a satellite as truncated or terminated with the ``is_truncated`` and ``is_terminated``
  methods.
* Added example script for using curriculum learning with RLlib in
  `Curriculum Learning <examples/curriculum_learning.ipynb>`_ example.
* Updated the list of `publications <publications.rst>`_
* Added the option to compute value with sMDP rewards at the start of the step in the
  RLlib configuration.
* Add the ability to observe remaining time in :class:`~bsk_rl.obs.Time`.
* Allow for the ``time_limit`` to be randomized.


Version 1.1.0
-------------
*Release Date: Feb. 26, 2025*


* Add ability in :class:`~bsk_rl.obs.SatProperties` to define new observations with
  a custom function.
* Add ``deepcopy`` to mutable inputs to the environment so that an environment argument
  dictionary can be copied without being affected by things that happen in the environment.
  This fixes compatibility with RLlib 2.33.0+. Note that this means that the satellite
  object passed to the environment is not the same object as the one used in the environment,
  as is the case for rewarders and communication objects.
* Add additional observation properties for satellites and opportunities.
* Add connectors for multiagent semi-MDPs, as demonstrated in a new `single agent <examples/time_discounted_gae.ipynb>`_
  and `multiagent <examples/async_multiagent_training.ipynb>`_ example.
* Add a ``min_period`` option to :class:`~bsk_rl.comm.CommunicationMethod`.
* Cache ``agents`` in the :class:`~bsk_rl.ConstellationTasking` environment to improve 
  performance.
* Add option to ``generate_obs_retasking_only`` to prevent computing observations for
  satellites that are continuing their current action.
* Allow for :class:`~bsk_rl.sats.ImagingSatellite` to default to a different type of
  opportunity than ``target``. Also allows for access filters to include an opportunity
  type.
* Improve performance of :class:`~bsk_rl.obs.Eclipse` observations by about 95%.
* Logs a warning if the initial battery charge or buffer level is incompatible with its capacity.
* Optimize communication when all satellites are communicating with each other.
* Enable Vizard visualization of the environment by setting the ``vizard_dir`` and ``vizard_settings``
  options in the environment.
* Allow for the specification of multiple rewarders in the environment.



Version 1.0.1
-------------
*Release Date: Aug. 29, 2024*

* Change the :class:`~bsk_rl.ConstellationTasking` environment info dictionary to include
  all non-agent information in ``info['__common__']``, which is expected by RLlib's 
  multiagent interfaces.
* Rewarder, communication, scenario, and satellites all have standardized ``reset_overwrite_previous``,
  ``reset_pre_sim_init``, and ``reset_post_sim_init`` methods to all for more complex
  initialization dependencies.
* Replace ``get_access_filter`` with :class:`~bsk_rl.sats.AccessSatellite.add_access_filter`,
  which uses boolean functions to determine which opportunity windows to consider.
* Changed the initial data generation to be defined in :class:`~bsk_rl.data.GlobalReward` 
  instead of :class:`~bsk_rl.scene.Scenario`.
* Added a new :ref:`examples` script that demonstrates how to include
  a targets with cloud coverage and a rewarder that accounts for cloud cover.
* Reformat the info dictionary to be more consistent across environments. All satellites now
  have a ``requires_retasking`` key, as opposed to a global list of satellites that require retasking.
  Each satellite also gets ``d_ts`` in its info dictionary. Info and warning messages are no longer
  saved in the info dict.
* ``log_info`` and ``log_warning`` are deprecated by :class:`~bsk_rl.sats.Satellite`, in favor of
  ``logger.info`` and ``logger.warning``.
* Add ability to correlate ``sat_args`` between satellites with the ``sat_arg_randomizer``
  option in :class:`~bsk_rl.GeneralSatelliteTasking`.  This is demonstrated in the setup
  of a constellation in the `multiagent example <examples/multiagent_envs.ipynb>`_.
* The default solar panel normal direction is now the negative z-axis, which is antiparallel
  to the default instrument direction.


Version 1.0.0
-------------
*Release Date: Jun. 12, 2024*

First major release of BSK-RL. 

* Refactored the repository to prioritize use of the :class:`~bsk_rl.GeneralSatelliteTasking` 
  environment. The general environment is now at the base level of ``bsk_rl``.
* Renamed various elements of the environment for simplicity and clarity. See the 
  :ref:`bsk_rl` for further details.
* Refactored the satellite :ref:`bsk_rl.obs` and :ref:`bsk_rl.act` specification 
  to be more clear and avoid conflicting variable names.
* Rewrote the documentation and added useful :ref:`examples`.
* Deprecated one-off environments and training scripts. These are still accessible
  in the `git history of the repository <https://github.com/AVSLab/bsk_rl/>`_.
