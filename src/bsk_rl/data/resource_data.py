"""The ResourceReward system tracks the change in an arbitrary resource level (e.g. fuel, power, time, etc.).

It can be used to penalize the use of resources or reward a gain. The reward type is
configured by setting the ``resource_fn`` argument to the desired function of the satellite.
"""

import logging
from typing import Callable, Union

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.obs import ResourceRewardWeight

logger = logging.getLogger(__name__)


class ResourceData(Data):
    """Data for tracking a change in some resource."""

    def __init__(self, resource_accumulated: float = 0.0) -> None:
        """Construct resource data.

        Args:
            resource_accumulated: Amount of resource change accumulated.
        """
        self.resource_accumulated = resource_accumulated

    def __add__(self, other: "ResourceData") -> "ResourceData":
        """Combine two units of ResourceData.

        Args:
            other: Another unit of ResourceData to add with this one.
        """
        total_resource = self.resource_accumulated + other.resource_accumulated
        return ResourceData(total_resource)

    def __repr__(self) -> str:
        """String representation of the ResourceData."""
        return f"ResourceData(resource_accumulated={self.resource_accumulated})"


class ResourceDataStore(DataStore):
    data_type = ResourceData

    def __init__(self, *args, resource_fn: Callable = lambda sat: 0.0, **kwargs):
        """DataStore for tracking a change in some resource."""
        super().__init__(*args, **kwargs)
        self.resource_fn = resource_fn

    def get_log_state(self) -> float:
        """Gets the current state of the resource."""
        return self.resource_fn(self.satellite)

    def compare_log_states(self, prev_state: float, new_state: float) -> ResourceData:
        """Compare the previous and new state of the resource."""
        delta_resource = new_state - prev_state
        return ResourceData(delta_resource)


class ResourceReward(GlobalReward):
    """Rewards for an arbitrary resource."""

    data_store_type = ResourceDataStore

    def __init__(
        self,
        reward_weight: Union[float, Callable] = 1.0,
        resource_fn: Callable = lambda sat: 0.0,
    ) -> None:
        """Rewards for an arbitrary resource.

        For example, to penalize the use of fuel, set

        .. code-block:: python

            resource_fn = lambda sat: sat.dynamics.dv_available  # or equivalent
            reward_weight = 0.1  # is positive, because the fuel changes are negative


        To penalize the use of time, set

        .. code-block:: python

            resource_fn = lambda sat: sat.simulator.sim_time
            reward_weight = -1e-3  # is negative, because time increases over time

        Args:
            reward_weight: [reward/resource] Scaling factor to apply to changes in resource
                level to yield reward. Can be a float or a function that randomizes the reward
                weight per-episode.
            resource_fn: Function to call to get the resource level for each satellite.
        """
        super().__init__()
        self._reward_weight = reward_weight
        self.resource_fn = resource_fn
        self.data_store_kwargs = dict(resource_fn=resource_fn)

    def reset_pre_sim_init(self) -> None:
        """Reset the reward weight before simulation initialization."""
        if callable(self._reward_weight):
            self.reward_weight = self._reward_weight()
        else:
            self.reward_weight = self._reward_weight
        return super().reset_pre_sim_init()

    def reset_post_sim_init(self) -> None:
        """Add the reward weight to each satellite's observation spec."""
        for satellite in self.scenario.satellites:
            for obs in satellite.observation_builder.observation_spec:
                if isinstance(obs, ResourceRewardWeight):
                    obs.weight_vector.append(self.reward_weight)
        return super().reset_post_sim_init()

    def calculate_reward(
        self, new_data_dict: dict[str, ResourceData]
    ) -> dict[str, float]:
        """Calculate the reward based on the change in resource level times the weight."""
        penalty = {
            sat_name: self.reward_weight * data.resource_accumulated
            for sat_name, data in new_data_dict.items()
        }

        return penalty


__doc_title__ = "Resource Data"
__all__ = ["ResourceData", "ResourceDataStore", "ResourceReward"]
