from typing import Callable, Dict, Tuple, Type

import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from bsk_rl.utilities.sb3 import custom_sb3_policies, shields


class CustomActorCriticShieldedPolicy(ActorCriticPolicy):
    """
    Custom actor critic policy with a shield. Made for AgileEOS environment.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Dict = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # Initialize the network parameters
        self.num_states = observation_space.shape[0]
        self.num_actions = action_space.n
        self.width = net_arch["width"]
        self.depth = net_arch["depth"]
        self.dropout = net_arch["dropout"]
        self.activation_function = activation_fn
        self.alpha = net_arch["alpha"]

        self.features_dim = self.width

        # Create the shield
        # self.shield = shields.AgileEOSShield()

        super(CustomActorCriticShieldedPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = custom_sb3_policies.CustomNetwork(
            self.num_states,
            self.num_actions,
            self.width,
            self.depth,
            self.dropout,
            self.activation_fn,
            self.alpha,
        )

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions = self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )
        actions = self.shield.shield_actions(observation, actions)
        return actions

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        # print('old actions: ', actions)

        # Override the actions here
        actions = self.shield.shield_actions(obs, actions)

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob


class CustomActorCriticShieldedAgileEOSPolicy(CustomActorCriticShieldedPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Dict = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # Create the shield
        self.shield = shields.AgileEOSShield()

        super(CustomActorCriticShieldedAgileEOSPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


class CustomActorCriticShieldedMultiSensorEOSPolicy(CustomActorCriticShieldedPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Dict = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # Create the shield
        self.shield = shields.MultiSensorEOSShield()

        super(CustomActorCriticShieldedMultiSensorEOSPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
