Release Notes
=============

Version 1.0.1
-------------
*Release Date: MMM. DD, YYYY*

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
  option in :class:`~bsk_rl.GeneralSatelliteTasking`.


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