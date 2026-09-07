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
        """Configure shared encoders, attention and output dimensions."""
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        """Encode the item set and preserve its declared masking and ordering."""
        return x + self.net(x)


class AttentionHead(nn.Module):
    def __init__(self, d_in: int, embed_dim: int):
        """Configure shared encoders, attention and output dimensions."""
        super().__init__()
        self.d_in = d_in
        self.embed_dim = embed_dim

        self.W_q = nn.Linear(d_in, embed_dim)
        self.W_k = nn.Linear(d_in, embed_dim)
        self.W_v = nn.Linear(d_in, embed_dim)

    def forward(self, x, y=None):
        # x: (B, n_tgts, d_in)
        """Encode the item set and preserve its declared masking and ordering."""
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
        """Configure shared encoders, attention and output dimensions."""
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
        """Encode the item set and preserve its declared masking and ordering."""
        head_outputs = [head(x, y) for head in self.attention_heads]
        concat_heads = torch.cat(head_outputs, dim=-1)  # (B, n_tgts, embed_dim)
        output = self.out_proj(concat_heads)  # (B, n_tgts, d_in)
        return output


class FastSelfAttention(nn.Module):
    """Self-attention block using PyTorch's fused kernel when available."""

    def __init__(self, d_in, d_model, n_heads=4):
        """Configure shared encoders, attention and output dimensions."""
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_in, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_in, bias=False)

    def forward(self, x, valid=None):
        """Encode the item set and preserve its declared masking and ordering."""
        B, N, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Exclude padding keys before softmax. Give empty rows one zero dummy
        # key so both fused and fallback kernels remain finite; zero them below.
        mask = None
        if valid is not None:
            safe = valid.clone()
            safe[~safe.any(dim=1), 0] = True
            mask = safe[:, None, None, :]
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        else:  # pragma: no cover - compatibility with older PyTorch
            scale = math.sqrt(self.d_head)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            if mask is not None:
                scores = scores.masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, dim=-1)
            attn = torch.matmul(weights, v)

        attn = attn.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out(attn)
        return output if valid is None else output * valid.unsqueeze(-1)


def masked_pool(tokens, valid):
    """Mean, maximum and population deviation with a zero empty-set identity."""
    if valid is None:
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    weights = valid.unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1)
    mean = (tokens * weights).sum(dim=1) / count
    maximum = tokens.masked_fill(~weights, -torch.inf).max(dim=1).values
    maximum = torch.where(valid.any(dim=1, keepdim=True), maximum, 0.0)
    variance = ((tokens - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
    std = torch.where(
        valid.any(dim=1, keepdim=True), torch.sqrt(variance.clamp_min(1e-12)), 0.0
    )
    return mean, maximum, std


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
        attention: bool = False,
        attention_dim: int = 32,
        num_heads: int = 2,
        valid_feature: int | None = None,
    ):
        """Configure shared encoders, attention and output dimensions."""
        super(GNNCritic, self).__init__()
        self.valid_feature = valid_feature

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

        self.attention = attention
        if self.attention:
            self.attention_layer = FastSelfAttention(
                tgt_encoded_dim, attention_dim, num_heads
            )

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
        """Encode the item set and preserve its declared masking and ordering."""
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
        valid = (
            None
            if self.valid_feature is None
            else x_tgts[:, :, self.valid_feature] > 0.5
        )
        if valid is not None:
            x_tgts = x_tgts.masked_fill(~valid.unsqueeze(-1), 0.0)
        latent_tgts = self.model_f(x_tgts)  # (B, n_tgts, width_f)
        if self.block_f:
            for block_f in self.blocks_f:
                latent_tgts = block_f(latent_tgts)  # (B, n_tgts, width_f)
        latent_tgts = self.out_layer_f(latent_tgts)  # (B, n_tgts, tgt_encoded_dim)

        if self.attention:
            attention = self.attention_layer(latent_tgts, valid)
            latent_tgts = latent_tgts + attention

        mean, maximum, std = masked_pool(latent_tgts, valid)
        pooled = [x_sat, mean, maximum]
        if self.pooling_std:
            pooled.append(std)
        latent = torch.cat(pooled, dim=-1)

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
        condition_on_spacecraft: bool = False,
        valid_feature: int | None = None,
    ):
        """Configure shared encoders, attention and output dimensions."""
        super(GNNActor, self).__init__()
        self.valid_feature = valid_feature

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
        self.condition_on_spacecraft = bool(condition_on_spacecraft)

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
        if self.condition_on_spacecraft:
            if self.obs_sat <= 0:
                raise ValueError(
                    "condition_on_spacecraft requires at least one global feature."
                )
            self.spacecraft_context_encoder = nn.Sequential(
                nn.Linear(self.obs_sat, tgt_encoded_dim),
                act_function(),
                nn.Linear(tgt_encoded_dim, tgt_encoded_dim),
            )
            self.spacecraft_context_normalization = nn.LayerNorm(tgt_encoded_dim)

        self.attention_depth = attention_depth

        layers_attention = []
        layers_normalization = []
        sequential_g = []
        for i in range(attention_depth):
            layers_attention.append(
                FastSelfAttention(tgt_encoded_dim, attention_dim, num_heads)
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
        """Encode the item set and preserve its declared masking and ordering."""
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
        valid = (
            None
            if self.valid_feature is None
            else x_tgts[:, :, self.valid_feature] > 0.5
        )
        if valid is not None:
            x_tgts = x_tgts.masked_fill(~valid.unsqueeze(-1), 0.0)
        x_tgts = self.model_f(x_tgts)  # (B, n_tgts, width)
        if self.block_f:
            for block_f in self.blocks_f:
                x_tgts = block_f(x_tgts)  # (B, n_tgts, width)

        latent_tgts = self.out_layer_f(x_tgts)  # (B, n_tgts, tgt_encoded_dim)
        if self.condition_on_spacecraft:
            context = self.spacecraft_context_encoder(x_sat).unsqueeze(1)
            latent_tgts = self.spacecraft_context_normalization(latent_tgts + context)

        for i in range(self.attention_depth):
            attention_out = self.attention_layers[i](
                latent_tgts, valid
            )  # (B, n_tgts, tgt_encoded_dim)

            latent_tgts = self.normalization_layers[i](
                latent_tgts + attention_out
            )  # (B, n_tgts, tgt_encoded_dim)

            latent_tgts = self.models_g[i](latent_tgts)  # (B, n_tgts, tgt_encoded_dim)

        logits_tgts = self.out_layer_g(latent_tgts).squeeze(-1)  # (B, n_tgts)

        if self.non_imaging_actions == 0:
            return logits_tgts

        # Right now this is intended for padding the non-imaging actions. Otherwise this should be conditioned on the x_sat vector as well
        pooled, _, _ = masked_pool(latent_tgts, valid)
        # Operational choices must still see own resources when no target exists.
        if self.condition_on_spacecraft:
            pooled = pooled + context.squeeze(1)
        no_action_logit = self.no_action_head(pooled)  # (B, non_imaging_actions)

        return torch.cat(
            [no_action_logit, logits_tgts], dim=1
        )  # (B, n_tgts + non_imaging_actions)


class PeerSetHead(nn.Module):
    """Shared peer scorer and invariant peer context for the target-set network.

    Only declared contact beacons and local ACK history enter these rows. Peer
    order permutes transmission logits and leaves imaging/operational values fixed.
    """

    def __init__(self, base, base_inputs, own_features, peer_features, critic=False):
        """Configure shared encoders, attention and output dimensions."""
        super().__init__()
        self.base, self.base_inputs, self.own_features = base, base_inputs, own_features
        self.peer_features, self.critic = peer_features, critic
        self.encode = nn.Sequential(
            nn.Linear(peer_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.context = nn.Linear(64, own_features, bias=False)
        self.score = nn.Sequential(
            nn.Linear(64 + own_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, batch):
        """Encode the item set and preserve its declared masking and ordering."""
        x = batch["obs"] if isinstance(batch, dict) else batch
        base = x[:, : self.base_inputs]
        peers = x[:, self.base_inputs :].reshape(x.shape[0], -1, self.peer_features)
        valid = peers[:, :, -1] > 0.5
        peers = peers.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.encode(peers)
        pooled, _, _ = masked_pool(encoded, valid)
        own = base[:, : self.own_features]
        conditioned = torch.cat(
            [own + self.context(pooled), base[:, self.own_features :]], dim=-1
        )
        result = self.base(conditioned)
        if self.critic:
            return result
        context = own.unsqueeze(1).expand(-1, peers.shape[1], -1)
        peer_logits = self.score(torch.cat([encoded, context], dim=-1)).squeeze(-1)
        return torch.cat([result, peer_logits], dim=-1)


class GNNModule(PPOTorchRLModule, nn.Module):
    def setup(self):
        # __sphinx_doc_begin__
        """Build the actor and local critic for the versioned observation layout."""
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

        n_peers = model_config.get("n_peers", 0)
        peer_features = model_config.get("peer_features", 12)
        base_inputs = self.config.observation_space.shape[0] - n_peers * peer_features
        valid_feature = 16 if model_config.get("completion_mask", False) else None
        self.pi_head = GNNActor(
            inputs=base_inputs,
            valid_feature=valid_feature,
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
            non_imaging_actions=model_config.get("non_imaging_actions", 1),
            condition_on_spacecraft=model_config.get("condition_on_spacecraft", False),
        )

        # Only build the critic network when this is a learner module.
        if not self.config.inference_only or self.framework != "torch":
            self.vf = GNNCritic(
                inputs=base_inputs,
                valid_feature=valid_feature,
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
                attention=model_config.get("critic_attention", False),
                attention_dim=model_config.get("critic_attention_dim", 32),
                num_heads=model_config.get("critic_num_heads", 2),
            )
            # Holds the parameter names to be removed or renamed when synching
            # from the learner to the inference module.
            self._inference_only_state_dict_keys = {}

        if n_peers:
            self.pi_head = PeerSetHead(
                self.pi_head, base_inputs, model_config["obs_sat"], peer_features
            )
            if hasattr(self, "vf"):
                self.vf = PeerSetHead(
                    self.vf,
                    base_inputs,
                    model_config["obs_sat"],
                    peer_features,
                    critic=True,
                )
        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)

    def pi(
        self, batch: Dict[str, TensorType], inference: bool = False
    ) -> Dict[str, TensorType]:
        """Return masked logits, actions and their categorical log probabilities."""
        pi_outs = {}

        logits = self.pi_head(batch)
        if self.config.model_config_dict.get("completion_mask", False):
            from bsk_rl.obs.completion_observations import (
                GLOBAL_FEATURES,
                TARGET_FEATURES,
                NON_IMAGING_ACTIONS,
                CONTINUE_ACTION,
                CONTINUE_VALID_FEATURE,
                VALID_TARGET_FEATURE,
                OBSERVATION_VERSION,
            )

            if (
                self.config.model_config_dict.get("observation_version")
                != OBSERVATION_VERSION
            ):
                raise ValueError("Completion observation/checkpoint version mismatch.")
            features = batch[Columns.OBS]
            count = self.config.model_config_dict["n_targets"]
            target_end = GLOBAL_FEATURES + count * TARGET_FEATURES
            targets = features[:, GLOBAL_FEATURES:target_end].reshape(
                features.shape[0], count, TARGET_FEATURES
            )
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, NON_IMAGING_ACTIONS : NON_IMAGING_ACTIONS + count] = (
                targets[:, :, VALID_TARGET_FEATURE] > 0.5
            )
            if self.config.model_config_dict.get("n_peers", 0):
                peers = features[:, target_end:].reshape(features.shape[0], -1, 12)
                mask[:, NON_IMAGING_ACTIONS + count :] = peers[:, :, -1] > 0.5
                mask[:, 3] = False  # Directed mode reserves baseline index 3.
            if self.config.model_config_dict.get("communication_mode") == "directed":
                mask[:, 3] = False
            if self.config.model_config_dict.get("information_case") in {
                "independent",
                "ideal_completion",
            }:
                mask[:, 3] = False
                mask[:, NON_IMAGING_ACTIONS + count :] = False
            mask[:, CONTINUE_ACTION] = features[:, CONTINUE_VALID_FEATURE] > 0.5
            # Finite sentinels keep categorical entropy/KL numerically well behaved.
            logits = logits.masked_fill(~mask, -1e9)

        if inference:
            discrete_action_dist = TorchCategorical.from_logits(
                logits
            ).to_deterministic()
            discrete_action = discrete_action_dist.sample()
        else:
            discrete_action_dist = TorchCategorical.from_logits(logits)
            discrete_action = discrete_action_dist.sample()

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
