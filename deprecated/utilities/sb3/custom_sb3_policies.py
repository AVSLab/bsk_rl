from typing import Callable, Dict, Tuple, Type

import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor
        (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy
        network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value
        network
    """

    def __init__(
        self,
        num_states=12,
        num_actions=4,
        width=100,
        depth=1,
        dropout=None,
        activation_function=nn.LeakyReLU,
        alpha=None,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = width
        self.latent_dim_vf = width

        layers = []

        # Add layers
        for layer in range(0, depth):
            if layer == 0:
                layers.append(nn.Linear(num_states, width))
            else:
                layers.append(nn.Linear(width, width))

            if activation_function == nn.Tanh:
                layers.append(nn.Tanh())
            elif activation_function == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(negative_slope=alpha))

            # Add dropout layers
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout, inplace=True))

        # Policy network
        self.policy_net = nn.Sequential(*layers)
        # Value network
        self.value_net = nn.Sequential(*layers)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified
            network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # print('Feature shape: ', features.shape)
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
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

        super(CustomActorCriticPolicy, self).__init__(
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
        self.mlp_extractor = CustomNetwork(
            self.num_states,
            self.num_actions,
            self.width,
            self.depth,
            self.dropout,
            self.activation_fn,
            self.alpha,
        )
