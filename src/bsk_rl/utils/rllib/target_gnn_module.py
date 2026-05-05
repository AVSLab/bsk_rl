"""RLlib PPO module for a target-wise imaging architecture.

The module expects a flat observation with all spacecraft/global features first,
followed by `n_targets` equally sized target feature chunks.  Each target chunk is
encoded by the same small network, so the policy can score a variable-looking list
of imaging candidates without feeding the entire target list through one giant MLP.
"""

from typing import Dict

try:
    from ray.rllib.core.columns import Columns
except (ImportError, ModuleNotFoundError):  # pragma: no cover - older RLlib
    from ray.rllib.core import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
)
import math
from ray.rllib.core.models.configs import RecurrentEncoderConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

torch, nn = try_import_torch()

# rl_module_args = dict(
#     model_config_dict={
#         "n_targets": 32,
#         "obs_sat": 38,
#         "width_f": 256,
#         "depth_f": 2,
#         "block_f": False,
#         "width_g": 128,
#         "depth_g": 4,
#         "tgt_encoded_dim": 128,
#         "attention_depth": 1,
#         "num_heads": 2,
#         "attention_dim": 128,
#         "dropout": 0,
#         "act_function": "ReLU",
#         "critic_tgt_encoded_dim": 128,
#         "critic_width_f": 256,
#         "critic_depth_f": 2,
#         "critic_block_f": False,
#         "critic_width_g": 64,
#         "critic_depth_g": 3,
#         "critic_block_g": False,
#         "critic_pooling_std": False,
#         "non_imaging_actions": 1,
#     },
#     rl_module_spec=RLModuleSpec(module_class=GNNModule),
# )

# Suggested training args
# training_args = dict(
#     lr=[
#         [0, 0.00033003435881682255],
#         [40000, 0.00033003435881682255 / 16.749479444886223],
#     ],
#     gamma=0.9915045428565076,
#     train_batch_size=int(300 * 10 * 3),  # TO MATCH CLUSTER
#     num_sgd_iter=30,
#     lambda_=0.8713548569911232,
#     use_kl_loss=False,
#     clip_param=0.14701727973480344,
#     grad_clip=0.3104924935285628,
#     entropy_coeff=0.023694512589767867,
# )


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.net(x)


class AttentionHead(nn.Module):
    def __init__(self, d_in: int, embed_dim: int):
        super().__init__()
        self.d_in = d_in
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(d_in, embed_dim)
        self.W_k = nn.Linear(d_in, embed_dim)
        self.W_v = nn.Linear(d_in, embed_dim)

    def forward(self, x, y=None):
        # x: (B, n_tgts, d_in)
        q = self.W_q(x)  # (B, N_x, embed_dim)
        if y is None:
            y = x
        k = self.W_k(y)  # (B, N_y, embed_dim)
        v = self.W_v(y)  # (B, N_y, embed_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, N_x, N_y)
        attn_output = torch.matmul(attn_weights, v)  # (B, N_x, embed_dim)

        return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_in = d_in
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attention_heads = nn.ModuleList(
            [AttentionHead(d_in, self.head_dim) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(embed_dim, d_in)

    def forward(self, x, y=None):
        head_outputs = [head(x, y) for head in self.attention_heads]
        concat_heads = torch.cat(head_outputs, dim=-1)  # (B, n_tgts, embed_dim)
        output = self.out_proj(concat_heads)  # (B, n_tgts, d_in)
        return output


class GNNCritic(nn.Module):

    def __init__(
        self,
        inputs: int,
        width_f: int = 32,
        depth_f: int = 4,
        block_f: bool = False,
        tgt_encoded_dim: int = 16,
        width_g: int = 32,
        depth_g: int = 4,
        pooling_std: bool = False,
        n_tgts: int = 32,
        obs_sat: int = 38,
        dropout: float = 0.0,
    ):
        super(GNNCritic, self).__init__()

        self.obs_sat = obs_sat
        act_function = nn.ReLU

        self.n_tgts = n_tgts
        target_inputs = inputs - self.obs_sat
        if target_inputs <= 0 or target_inputs % n_tgts != 0:
            raise ValueError(
                "GNN critic observation layout must be "
                "[spacecraft features][equal target chunks]. "
                f"Received inputs={inputs}, obs_sat={self.obs_sat}, n_tgts={n_tgts}."
            )
        self.features_per_tgt = target_inputs // n_tgts
        self.block_f = block_f

        layers_f = []
        layers_f.append(nn.Linear(self.features_per_tgt, width_f))
        layers_f.append(act_function())
        if dropout > 0:
            layers_f.append(nn.Dropout(dropout))
        if block_f:
            self.blocks_f = nn.ModuleList(
                [ResidualMLPBlock(width_f) for _ in range(depth_f)]
            )
        else:
            for _ in range(depth_f - 1):
                layers_f.append(nn.Linear(width_f, width_f))
                layers_f.append(act_function())
            if dropout > 0:
                layers_f.append(nn.Dropout(dropout))

        self.model_f = nn.Sequential(*layers_f)
        self.out_layer_f = nn.Linear(width_f, tgt_encoded_dim)

        n_dim = 2
        self.pooling_std = pooling_std
        if self.pooling_std:
            n_dim += 1

        layers_g = []
        layers_g.append(nn.Linear(obs_sat + n_dim * tgt_encoded_dim, width_g))
        layers_g.append(act_function())
        if dropout > 0:
            layers_g.append(nn.Dropout(dropout))

        for _ in range(depth_g - 1):
            layers_g.append(nn.Linear(width_g, width_g))
            layers_g.append(act_function())
            if dropout > 0:
                layers_g.append(nn.Dropout(dropout))

        layers_g.append(nn.Linear(width_g, 1))
        self.model_g = nn.Sequential(*layers_g)

    def forward(self, x):
        if isinstance(x, dict) and "obs" in x:
            x = x["obs"]

        B = x.shape[0]
        if self.obs_sat > 0:
            x_sat = x[:, : self.obs_sat]
            x_tgts = x[:, self.obs_sat :]
        else:
            x_sat = x.new_zeros((B, 0))
            x_tgts = x

        # Allows changes in the number of targets during runtime without changing internal variables as long as the input dimension is consistent with the number of targets
        n_tgts = x_tgts.shape[1] // self.features_per_tgt
        x_tgts = x_tgts.view(
            B, n_tgts, self.features_per_tgt
        )  # (B, n_tgts, features_per_tgt)
        latent_tgts = self.model_f(x_tgts)  # (B, n_tgts, width_f)
        if self.block_f:
            for block_f in self.blocks_f:
                latent_tgts = block_f(latent_tgts)  # (B, n_tgts, width_f)
        latent_tgts = self.out_layer_f(latent_tgts)  # (B, n_tgts, tgt_encoded_dim)

        if self.pooling_std:
            latent = torch.cat(
                [
                    x_sat,
                    torch.mean(latent_tgts, dim=1),
                    torch.max(latent_tgts, dim=1).values,
                    torch.std(latent_tgts, dim=1),
                ],
                dim=-1,
            )
        else:
            latent = torch.cat(
                [
                    x_sat,
                    torch.mean(latent_tgts, dim=1),
                    torch.max(latent_tgts, dim=1).values,
                ],
                dim=-1,
            )

        critic_value = self.model_g(latent).squeeze(-1)  # (B,)

        return critic_value


class GNNActor(nn.Module):

    def __init__(
        self,
        inputs: int,
        width_f: int = 32,
        depth_f: int = 4,
        block_f: bool = False,
        tgt_encoded_dim: int = 16,
        attention_depth: int = 1,
        num_heads: int = 2,
        attention_dim: int = 32,
        width_g: int = 32,
        depth_g: int = 2,
        n_tgts: int = 32,
        obs_sat: int = 38,
        non_imaging_actions: int = 1,
        dropout: float = 0.0,
    ):
        super(GNNActor, self).__init__()

        self.obs_sat = obs_sat
        act_function = nn.ReLU
        self.n_tgts = n_tgts
        target_inputs = inputs - self.obs_sat
        if target_inputs <= 0 or target_inputs % n_tgts != 0:
            raise ValueError(
                "GNN actor observation layout must be "
                "[spacecraft features][equal target chunks]. "
                f"Received inputs={inputs}, obs_sat={self.obs_sat}, n_tgts={n_tgts}."
            )
        self.features_per_tgt = target_inputs // n_tgts
        self.block_f = block_f

        layers_f = []
        layers_f.append(nn.Linear(self.features_per_tgt, width_f))
        layers_f.append(act_function())
        if dropout > 0:
            layers_f.append(nn.Dropout(dropout))
        if block_f:
            self.blocks_f = nn.ModuleList(
                [ResidualMLPBlock(width_f) for _ in range(depth_f)]
            )
        else:
            for _ in range(depth_f - 1):
                layers_f.append(nn.Linear(width_f, width_f))
                layers_f.append(act_function())
                if dropout > 0:
                    layers_f.append(nn.Dropout(dropout))

        self.model_f = nn.Sequential(*layers_f)
        self.out_layer_f = nn.Linear(width_f, tgt_encoded_dim)

        self.attention_depth = attention_depth

        layers_attention = []
        layers_normalization = []
        sequential_g = []
        for i in range(attention_depth):
            layers_attention.append(
                MultiHeadAttention(tgt_encoded_dim, attention_dim, num_heads)
            )
            layers_normalization.append(nn.LayerNorm(tgt_encoded_dim))

            layers_g = []
            layers_g.append(nn.Linear(tgt_encoded_dim, width_g))
            layers_g.append(act_function())
            if dropout > 0:
                layers_g.append(nn.Dropout(dropout))
            for _ in range(depth_g - 1):
                layers_g.append(nn.Linear(width_g, width_g))
                layers_g.append(act_function())
                if dropout > 0:
                    layers_g.append(nn.Dropout(dropout))
            layers_g.append(nn.Linear(width_g, tgt_encoded_dim))
            # Ensure normalization is applied after every attention layer except the last one mimicking the structure of a Transformer block
            if i < attention_depth - 1:
                layers_g.append(nn.LayerNorm(tgt_encoded_dim))
            sequential_g.append(nn.Sequential(*layers_g))

        self.attention_layers = nn.ModuleList(layers_attention)
        self.normalization_layers = nn.ModuleList(layers_normalization)
        self.models_g = nn.ModuleList(sequential_g)

        self.out_layer_g = nn.Linear(tgt_encoded_dim, 1)

        self.non_imaging_actions = non_imaging_actions
        if self.non_imaging_actions > 0:
            self.no_action_head = nn.Linear(tgt_encoded_dim, non_imaging_actions)

    def forward(self, x):
        if isinstance(x, dict) and "obs" in x:
            x = x["obs"]

        B = x.shape[0]
        if self.obs_sat > 0:
            x_sat = x[:, : self.obs_sat]
            x_tgts = x[:, self.obs_sat :]
        else:
            x_sat = None
            x_tgts = x
        n_tgts = x_tgts.shape[1] // self.features_per_tgt
        x_tgts = x_tgts.view(
            B, n_tgts, self.features_per_tgt
        )  # (B, n_tgts, features_per_tgt)
        x_tgts = self.model_f(x_tgts)  # (B, n_tgts, width)
        if self.block_f:
            for block_f in self.blocks_f:
                x_tgts = block_f(x_tgts)  # (B, n_tgts, width)

        latent_tgts = self.out_layer_f(x_tgts)  # (B, n_tgts, tgt_encoded_dim)

        for i in range(self.attention_depth):
            attention_out = self.attention_layers[i](
                latent_tgts
            )  # (B, n_tgts, tgt_encoded_dim)

            latent_tgts = self.normalization_layers[i](
                latent_tgts + attention_out
            )  # (B, n_tgts, tgt_encoded_dim)

            latent_tgts = self.models_g[i](latent_tgts)  # (B, n_tgts, tgt_encoded_dim)

        logits_tgts = self.out_layer_g(latent_tgts).squeeze(-1)  # (B, n_tgts)

        if self.non_imaging_actions == 0:
            return logits_tgts

        # Right now this is intended for padding the non-imaging actions. Otherwise this should be conditioned on the x_sat vector as well
        no_action_logit = self.no_action_head(
            torch.mean(latent_tgts, dim=1)
        )  # (B, non_imaging_actions)

        return torch.cat(
            [no_action_logit, logits_tgts], dim=1
        )  # (B, n_tgts + non_imaging_actions)


class GNNModule(PPOTorchRLModule, nn.Module):
    def setup(self):
        # __sphinx_doc_begin__
        catalog = self.config.get_catalog()
        # If we have a stateful model, states for the critic need to be collected
        # during sampling and `inference-only` needs to be `False`. Note, at this
        # point the encoder is not built, yet and therefore `is_stateful()` does
        # not work.
        is_stateful = isinstance(
            catalog.actor_critic_encoder_config.base_encoder_config,
            RecurrentEncoderConfig,
        )
        if is_stateful:
            self.config.inference_only = False
        # If this is an `inference_only` Module, we'll have to pass this information
        # to the encoder config as well.
        if self.config.inference_only and self.framework == "torch":
            catalog.actor_critic_encoder_config.inference_only = True

        model_config = self.config.model_config_dict
        dropout = model_config.get("dropout", model_config.get("dropout_rate", 0.0))

        self.encoder = lambda x: {ENCODER_OUT: {ACTOR: x, CRITIC: x}}

        self.pi_head = GNNActor(
            inputs=self.config.observation_space.shape[0],
            n_tgts=model_config["n_targets"],
            obs_sat=model_config["obs_sat"],
            width_f=model_config["width_f"],
            depth_f=model_config["depth_f"],
            block_f=model_config["block_f"],
            tgt_encoded_dim=model_config["tgt_encoded_dim"],
            attention_depth=model_config["attention_depth"],
            num_heads=model_config["num_heads"],
            attention_dim=model_config["attention_dim"],
            width_g=model_config["width_g"],
            depth_g=model_config["depth_g"],
            dropout=dropout,
            non_imaging_actions=model_config.get(
                "non_imaging_actions", 1
            ),
        )

        # Only build the critic network when this is a learner module.
        if not self.config.inference_only or self.framework != "torch":
            self.vf = GNNCritic(
                inputs=self.config.observation_space.shape[0],
                n_tgts=model_config["n_targets"],
                obs_sat=model_config["obs_sat"],
                width_f=model_config["critic_width_f"],
                depth_f=model_config["critic_depth_f"],
                block_f=model_config["critic_block_f"],
                tgt_encoded_dim=model_config["critic_tgt_encoded_dim"],
                width_g=model_config["critic_width_g"],
                depth_g=model_config["critic_depth_g"],
                pooling_std=model_config["critic_pooling_std"],
                dropout=dropout,
            )
            # Holds the parameter names to be removed or renamed when synching
            # from the learner to the inference module.
            self._inference_only_state_dict_keys = {}

        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)

    def pi(
        self, batch: Dict[str, TensorType], inference: bool = False
    ) -> Dict[str, TensorType]:
        pi_outs = {}

        logits = self.pi_head(batch)

        if inference:
            discrete_action_dist = TorchCategorical.from_logits(
                logits
            ).to_deterministic()
            discrete_action = discrete_action_dist.sample()
        else:
            discrete_action_dist = TorchCategorical(probs=torch.softmax(logits, dim=-1))
            discrete_action = discrete_action_dist.rsample().argmax(dim=-1)

        discrete_action_logp = discrete_action_dist.logp(discrete_action)

        pi_outs[Columns.ACTION_LOGP] = discrete_action_logp
        pi_outs[Columns.ACTION_DIST_INPUTS] = logits

        pi_outs[Columns.ACTIONS] = discrete_action
        return pi_outs

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        return self.pi(batch, inference=True)

    @override(TorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        return self.pi(batch, inference=False)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        outs = {}
        outs.update(self.pi(batch))
        vf_out = self.vf(batch)
        outs[Columns.VF_PREDS] = vf_out.squeeze(-1)
        return outs
