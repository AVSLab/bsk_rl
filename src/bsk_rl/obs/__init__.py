"""Observations are found at ``bsk_rl.obs``.

Satellite observation types can be used to add information to the observation.
:class:`Observation` provides an interface for creating new observation types. To
configure the observation, set the ``observation_spec`` attribute of a
:class:`~bsk_rl.sats.Satellite` subclass. For example:

.. code-block:: python

    class MyObservationSatellite(Satellite):
        observation_spec = [
            SatProperties(
                dict(prop="r_BN_P", module="dynamics", norm=REQ_EARTH * 1e3),
                dict(prop="v_BN_P", module="dynamics", norm=7616.5),
            ),
            obs.OpportunityProperties(
                dict(prop="priority"),
                dict(prop="r_LP_P", norm=REQ_EARTH * 1e3),
                n_ahead_observe=16,
            ),
            obs.Time(),
        ]

The format of the observation can setting the ``obs_type`` attribute of the
:class:`~bsk_rl.sats.Satellite`. The default is ``np.ndarray``, but
it can also be set to a human-readable ``dict`` or a ``list``.

Some commonly used observations are provided:

* :class:`SatProperties` - Add arbitrary ``dynamics`` and ``fsw`` properties.
* :class:`RelativeProperties` - Add arbitrary properties relative to some other satellite.
* :class:`Time` - Add simulation time to the observation.
* :class:`OpportunityProperties` - Add information about upcoming targets or other ground access points to the observation.
* :class:`Eclipse` - Add a tuple of the next orbit start and end.
* :class:`ResourceRewardWeight` - Reports the weights of any randomized :class:`ResourceReward`.
"""

from bsk_rl.obs.observations import (
    Eclipse,
    Observation,
    OpportunityProperties,
    ResourceRewardWeight,
    SatProperties,
    Time,
)
from bsk_rl.obs.relative_observations import RelativeProperties

__doc_title__ = "Observations"
__all__ = [
    "Observation",
    "SatProperties",
    "RelativeProperties",
    "Time",
    "OpportunityProperties",
    "Eclipse",
    "ResourceRewardWeight",
]
