"""Masked policy implementations for the AMOS architecture comparison.

The attention model is intentionally named a *target-set attention policy*.
It has no graph, edge set, adjacency matrix, or message-passing graph operator.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .config import ArchitectureConfig, EnvironmentConfig
from .environment import (
    GLOBAL_FEATURE_COUNT,
    TARGET_FEATURE_COUNT,
    TARGET_MASK_INDEX,
)

INVALID_ACTION_LOGIT = -1.0e9


@dataclass(frozen=True)
class ObservationLayout:
    """Flattened observation and action layout shared by both policies."""

    global_features: int = GLOBAL_FEATURE_COUNT
    target_features: int = TARGET_FEATURE_COUNT
    target_capacity: int = 10
    target_mask_index: int = TARGET_MASK_INDEX
    non_target_actions: int = 3

    @property
    def observation_size(self) -> int:
        return self.global_features + self.target_capacity * self.target_features

    @property
    def action_size(self) -> int:
        return self.target_capacity + self.non_target_actions

    def validate(self) -> None:
        if self.global_features < 1 or self.target_features < 2:
            raise ValueError("observation feature dimensions must be positive")
        if self.target_capacity < 1 or self.non_target_actions < 1:
            raise ValueError("action dimensions must be positive")
        if not 0 <= self.target_mask_index < self.target_features:
            raise ValueError("target mask index is invalid")

    def split(self, observation: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return global values, physical target values, and a Boolean mask."""

        self.validate()
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[-1] != self.observation_size:
            raise ValueError(
                f"expected [batch, {self.observation_size}] observation, got "
                f"{tuple(observation.shape)}"
            )
        global_values = observation[:, : self.global_features]
        rows = observation[:, self.global_features :].reshape(
            observation.shape[0], self.target_capacity, self.target_features
        )
        valid = rows[:, :, self.target_mask_index] > 0.5
        physical = torch.cat(
            [
                rows[:, :, : self.target_mask_index],
                rows[:, :, self.target_mask_index + 1 :],
            ],
            dim=-1,
        )
        # This is the second line of defense against padded-feature leakage.
        physical = physical * valid.unsqueeze(-1).to(physical.dtype)
        return global_values, physical, valid


def layout_from_environment(config: EnvironmentConfig) -> ObservationLayout:
    return ObservationLayout(
        target_capacity=config.candidate_count,
        non_target_actions=config.non_target_actions,
    )


def _activation(name: str) -> type[nn.Module]:
    return {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[name]


def make_mlp(
    input_dim: int,
    hidden_widths: tuple[int, ...],
    output_dim: int,
    *,
    activation: str,
    layer_norm: bool,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    activation_type = _activation(activation)
    for width in hidden_widths:
        layers.append(nn.Linear(current, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation_type())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def mask_target_logits(
    target_logits: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if target_logits.shape != valid.shape:
        raise ValueError("target logits and validity mask must have the same shape")
    return target_logits.masked_fill(~valid, INVALID_ACTION_LOGIT)


class FixedInputMonolithicActorCritic(nn.Module):
    """Historical-style large MLP with fixed 400-target input and output."""

    def __init__(self, layout: ObservationLayout, architecture: ArchitectureConfig):
        super().__init__()
        layout.validate()
        self.layout = layout
        self.architecture = architecture
        self.actor = make_mlp(
            layout.observation_size,
            architecture.hidden_widths,
            layout.action_size,
            activation=architecture.activation,
            layer_norm=architecture.layer_norm,
            dropout=architecture.dropout,
        )
        self.critic = make_mlp(
            layout.observation_size,
            architecture.hidden_widths,
            1,
            activation=architecture.activation,
            layer_norm=architecture.layer_norm,
            dropout=architecture.dropout,
        )

    def _sanitized_observation(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_values, physical, valid = self.layout.split(observation)
        rows = torch.cat([physical, valid.unsqueeze(-1).to(physical.dtype)], dim=-1)
        # The configured mask is last in this study.  Keep this explicit check so
        # a future layout change cannot silently alter the flattened contract.
        if self.layout.target_mask_index != self.layout.target_features - 1:
            raise ValueError("monolithic model currently requires a trailing mask")
        clean = torch.cat([global_values, rows.flatten(start_dim=1)], dim=-1)
        return clean, valid

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clean, valid = self._sanitized_observation(observation)
        raw_logits = self.actor(clean)
        target_logits = mask_target_logits(
            raw_logits[:, : self.layout.target_capacity], valid
        )
        logits = torch.cat(
            [target_logits, raw_logits[:, self.layout.target_capacity :]], dim=-1
        )
        return logits, self.critic(clean).squeeze(-1)


class MaskedSelfAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention with strict key/query masking."""

    def __init__(
        self,
        embedding_dim: int,
        heads: int,
        feed_forward_width: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(embedding_dim)
        self.feed_forward = make_mlp(
            embedding_dim,
            (feed_forward_width,),
            embedding_dim,
            activation=activation,
            layer_norm=False,
            dropout=dropout,
        )
        self.norm_feed_forward = nn.LayerNorm(embedding_dim)

    def forward(self, latent: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # PyTorch cannot softmax a row whose every key is masked.  Supply one
        # synthetic zero key only for that edge case, then zero all invalid rows.
        safe_valid = valid.clone()
        all_invalid = ~safe_valid.any(dim=1)
        if torch.any(all_invalid):
            safe_valid[all_invalid, 0] = True
        attended, _ = self.attention(
            latent,
            latent,
            latent,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        output = self.norm_attention(latent + attended)
        output = self.norm_feed_forward(output + self.feed_forward(output))
        return output * valid.unsqueeze(-1).to(output.dtype)


class TargetSetBackbone(nn.Module):
    """Shared encoder and attention blocks for an unordered masked target set."""

    def __init__(self, layout: ObservationLayout, architecture: ArchitectureConfig):
        super().__init__()
        self.layout = layout
        embed = architecture.embedding_dim
        physical_features = layout.target_features - 1
        self.target_encoder = make_mlp(
            physical_features,
            architecture.hidden_widths,
            embed,
            activation=architecture.activation,
            layer_norm=architecture.layer_norm,
            dropout=architecture.dropout,
        )
        self.global_encoder = make_mlp(
            layout.global_features,
            architecture.hidden_widths,
            embed,
            activation=architecture.activation,
            layer_norm=architecture.layer_norm,
            dropout=architecture.dropout,
        )
        self.attention_blocks = nn.ModuleList(
            [
                MaskedSelfAttentionBlock(
                    embed,
                    architecture.attention_heads,
                    architecture.feed_forward_width,
                    architecture.dropout,
                    architecture.activation,
                )
                for _ in range(architecture.attention_blocks)
            ]
        )

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_values, physical, valid = self.layout.split(observation)
        latent = self.target_encoder(physical)
        latent = latent * valid.unsqueeze(-1).to(latent.dtype)
        for block in self.attention_blocks:
            latent = block(latent, valid)

        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(latent.dtype)
        mean = latent.sum(dim=1) / count
        max_input = latent.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        maximum = max_input.max(dim=1).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return latent, self.global_encoder(global_values), mean, maximum


class TargetSetAttentionActorCritic(nn.Module):
    """Variable-cardinality target-set attention actor and separate critic."""

    def __init__(self, layout: ObservationLayout, architecture: ArchitectureConfig):
        super().__init__()
        layout.validate()
        self.layout = layout
        self.architecture = architecture
        embed = architecture.embedding_dim
        self.actor_backbone = TargetSetBackbone(layout, architecture)
        self.target_head = make_mlp(
            4 * embed,
            (architecture.feed_forward_width,),
            1,
            activation=architecture.activation,
            layer_norm=architecture.layer_norm,
            dropout=architecture.dropout,
        )
        self.non_target_head = make_mlp(
            3 * embed,
            (architecture.feed_forward_width,),
            layout.non_target_actions,
            activation=architecture.activation,
            layer_norm=architecture.layer_norm,
            dropout=architecture.dropout,
        )
        # A separate value backbone matches the historical non-shared critic.
        self.critic_backbone = TargetSetBackbone(layout, architecture)
        self.value_head = make_mlp(
            3 * embed,
            (architecture.feed_forward_width,),
            1,
            activation=architecture.activation,
            layer_norm=architecture.layer_norm,
            dropout=architecture.dropout,
        )

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        targets, global_latent, mean, maximum = self.actor_backbone(observation)
        _, _, valid = self.layout.split(observation)
        context = torch.cat([global_latent, mean, maximum], dim=-1)
        expanded_context = context.unsqueeze(1).expand(-1, targets.shape[1], -1)
        target_logits = self.target_head(
            torch.cat([targets, expanded_context], dim=-1)
        ).squeeze(-1)
        target_logits = mask_target_logits(target_logits, valid)
        non_target_logits = self.non_target_head(context)

        _, critic_global, critic_mean, critic_maximum = self.critic_backbone(
            observation
        )
        value = self.value_head(
            torch.cat([critic_global, critic_mean, critic_maximum], dim=-1)
        ).squeeze(-1)
        # Action order is identical to the environment: target slots, then the
        # charge/downlink/desaturate actions.
        return torch.cat([target_logits, non_target_logits], dim=-1), value


def build_actor_critic(
    architecture: ArchitectureConfig, layout: ObservationLayout
) -> nn.Module:
    architecture.validate()
    if architecture.name == "fixed_input_monolithic_mlp":
        return FixedInputMonolithicActorCritic(layout, architecture)
    if architecture.name == "target_set_attention":
        return TargetSetAttentionActorCritic(layout, architecture)
    raise ValueError(f"unsupported architecture {architecture.name}")


def parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad or not trainable_only
        )
    )


def benchmark_inference(
    model: nn.Module,
    observation_size: int,
    *,
    batch_size: int = 1,
    warmup: int = 25,
    repeats: int = 200,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure deterministic actor+critic forward latency on a named device."""

    model = model.to(device).eval()
    observation = torch.zeros(batch_size, observation_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(observation)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(observation)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "device": device,
        "batch_size": float(batch_size),
        "repeats": float(repeats),
        "mean_forward_ms": 1000.0 * elapsed / repeats,
        "mean_sample_ms": 1000.0 * elapsed / (repeats * batch_size),
    }


def model_metadata(model: nn.Module) -> dict[str, Any]:
    architecture = getattr(model, "architecture", None)
    layout = getattr(model, "layout", None)
    return {
        "class": type(model).__name__,
        "architecture": None if architecture is None else asdict(architecture),
        "layout": None if layout is None else asdict(layout),
        "trainable_parameters": parameter_count(model),
    }


# RLlib is optional for pure model/unit-test use.  The cluster environment used
# for training provides Ray 2.35, whose PPO RLModule contract is wrapped below.
try:  # pragma: no cover - exercised by integration/training tests
    from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
    from ray.rllib.core import Columns
    from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
    from ray.rllib.core.models.configs import RecurrentEncoderConfig
    from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
    from ray.rllib.models.torch.torch_distributions import TorchCategorical
    from ray.rllib.utils.annotations import override
    from ray.rllib.utils.typing import TensorType

    class MaskedStudyPPOTorchRLModule(PPOTorchRLModule, nn.Module):
        """RLlib adapter shared by the two masked study networks."""

        architecture_name: str

        def setup(self) -> None:
            catalog = self.config.get_catalog()
            is_stateful = isinstance(
                catalog.actor_critic_encoder_config.base_encoder_config,
                RecurrentEncoderConfig,
            )
            if is_stateful:
                self.config.inference_only = False
            if self.config.inference_only and self.framework == "torch":
                catalog.actor_critic_encoder_config.inference_only = True
            self.encoder = lambda x: {ENCODER_OUT: {ACTOR: x, CRITIC: x}}

            config = dict(self.config.model_config_dict)
            layout = ObservationLayout(
                global_features=int(config["global_features"]),
                target_features=int(config["target_features"]),
                target_capacity=int(config["target_capacity"]),
                target_mask_index=int(config["target_mask_index"]),
                non_target_actions=int(config["non_target_actions"]),
            )
            architecture = ArchitectureConfig(
                name=self.architecture_name,
                hidden_widths=tuple(config["hidden_widths"]),
                activation=config.get("activation", "relu"),
                layer_norm=bool(config.get("layer_norm", False)),
                embedding_dim=int(config.get("embedding_dim", 128)),
                attention_heads=int(config.get("attention_heads", 2)),
                attention_blocks=int(config.get("attention_blocks", 1)),
                feed_forward_width=int(config.get("feed_forward_width", 128)),
                dropout=float(config.get("dropout", 0.0)),
                separate_value_network=True,
            )
            self.core = build_actor_critic(architecture, layout)
            self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)
            self._inference_only_state_dict_keys = {}

        def vf(self, encoded: dict[str, TensorType] | TensorType) -> TensorType:
            """Value head adapter required by PPO's ``compute_values`` API."""

            observation = (
                encoded[Columns.OBS]
                if isinstance(encoded, dict) and Columns.OBS in encoded
                else encoded
            )
            _, value = self.core(observation)
            return value.unsqueeze(-1)

        def _policy_outputs(
            self, batch: dict[str, TensorType], deterministic: bool
        ) -> dict[str, TensorType]:
            observation = batch[Columns.OBS]
            if self.config.inference_only:
                with torch.inference_mode():
                    logits, _ = self.core(observation)
            else:
                logits, _ = self.core(observation)
            distribution = TorchCategorical.from_logits(logits)
            if deterministic:
                distribution = distribution.to_deterministic()
            action = distribution.sample()
            return {
                Columns.ACTION_LOGP: distribution.logp(action),
                Columns.ACTION_DIST_INPUTS: logits,
                Columns.ACTIONS: action,
            }

        @override(TorchRLModule)
        def _forward_inference(
            self, batch: dict[str, TensorType]
        ) -> dict[str, TensorType]:
            return self._policy_outputs(batch, deterministic=True)

        @override(TorchRLModule)
        def _forward_exploration(
            self, batch: dict[str, TensorType], **kwargs: Any
        ) -> dict[str, TensorType]:
            return self._policy_outputs(batch, deterministic=False)

        @override(TorchRLModule)
        def _forward_train(self, batch: dict[str, TensorType]) -> dict[str, TensorType]:
            outputs = self._policy_outputs(batch, deterministic=False)
            _, value = self.core(batch[Columns.OBS])
            outputs[Columns.VF_PREDS] = value
            return outputs

    class FixedInputMonolithicPPOModule(MaskedStudyPPOTorchRLModule):
        architecture_name = "fixed_input_monolithic_mlp"

    class TargetSetAttentionPPOModule(MaskedStudyPPOTorchRLModule):
        architecture_name = "target_set_attention"

except ImportError:  # pragma: no cover
    MaskedStudyPPOTorchRLModule = None
    FixedInputMonolithicPPOModule = None
    TargetSetAttentionPPOModule = None


__all__ = [
    "FixedInputMonolithicActorCritic",
    "FixedInputMonolithicPPOModule",
    "INVALID_ACTION_LOGIT",
    "ObservationLayout",
    "TargetSetAttentionActorCritic",
    "TargetSetAttentionPPOModule",
    "benchmark_inference",
    "build_actor_critic",
    "layout_from_environment",
    "mask_target_logits",
    "model_metadata",
    "parameter_count",
]
