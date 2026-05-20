from typing import Dict

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

# Suggested parameters:
# training_args = dict(
#     lr=[
#         [0, 0.00033003435881682255],
#         [40000, 0.00033003435881682255 / 16.749479444886223],
#     ],
#     gamma=0.999,
#     train_batch_size=int(300 * 10 * 3),  # TO MATCH CLUSTER
#     num_sgd_iter=30,
#     lambda_=0.8713548569911232,
#     use_kl_loss=False,
#     clip_param=0.14701727973480344,
#     grad_clip=0.3104924935285628,
#     entropy_coeff=0.023694512589767867,
# )

# rl_module_args = dict(
#     model_config_dict={
#         "n_targets": 40,
#         "obs_sat": 38,
#         "non_imaging_actions": 1,
#         "width_f": 256,
#         "depth_f": 2,
#         "width_g": 128,
#         "depth_g": 4,
#         "tgt_encoded_dim": 128,
#         "attention_depth": 1,
#         "num_heads": 2,
#         "attention_dim": 128,
#         "width_f_sat": 256,
#         "depth_f_sat": 2,
#         "width_g_sat": 128,
#         "depth_g_sat": 4,
#         "sat_attention_dim": 128,
#         "sat_attention_heads": 2,
#         "sat_encoded_dim": 128,
#         "act_function": "ReLU",
#         "critic_tgt_encoded_dim": 128,
#         "critic_width_f": 256,
#         "critic_depth_f": 2,
#         "critic_width_g": 64,
#         "critic_depth_g": 3,
#         "dropout": 0.1,
#         "post_self_attention": True,
#     },
#     rl_module_spec=RLModuleSpec(module_class=GATModule),
# )


class FastSelfAttention(nn.Module):
    def __init__(self, d_in, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_in, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_in, bias=False)

    def forward(self, x):
        # x: (B, N, d_in)
        B, N, _ = x.shape

        qkv = self.qkv(x)  # (B, N, 3*d_model)
        qkv = qkv.view(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d_head)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )  # (B, H, N, d_head)

        attn = attn.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out(attn)


class FastMultiHeadAttention(nn.Module):
    def __init__(self, d_in_q, d_in_kv, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_in_q, d_model, bias=False)
        self.k_proj = nn.Linear(d_in_kv, d_model, bias=False)
        self.v_proj = nn.Linear(d_in_kv, d_model, bias=False)
        self.out = nn.Linear(d_model, d_in_q, bias=False)

    def forward(self, x, y=None):
        # x: (B, N_q, d_in_q)
        if y is None:
            y = x

        B, N_q, _ = x.shape
        N_kv = y.shape[1]

        q = self.q_proj(x).view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(y).view(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(y).view(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )  # (B, H, N_q, d_head)

        attn = attn.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.out(attn)


class GATCritic(nn.Module):

    def __init__(
        self,
        inputs: int,
        width_f: int = 32,
        depth_f: int = 4,
        tgt_encoded_dim: int = 16,
        width_g: int = 32,
        depth_g: int = 4,
        n_tgts: int = 32,
        obs_sat: int = 38,
        dropout: float = 0.0,
        concat_sat_f: bool = False,
        use_sat_f: bool = False,
    ):
        super(GATCritic, self).__init__()

        self.obs_sat = obs_sat
        act_function = nn.ReLU

        self.n_tgts = n_tgts
        self.features_per_tgt = (inputs - self.obs_sat) // n_tgts
        self.concat_sat_f = concat_sat_f
        self.use_sat_f = use_sat_f

        input_size_f = self.features_per_tgt
        if concat_sat_f:
            if use_sat_f:
                input_size_f += tgt_encoded_dim
            else:
                input_size_f += self.obs_sat

        layers_f = []
        layers_f.append(nn.Linear(input_size_f, width_f))
        layers_f.append(act_function())
        if dropout > 0:
            layers_f.append(nn.Dropout(dropout))
        for _ in range(depth_f - 1):
            layers_f.append(nn.Linear(width_f, width_f))
            layers_f.append(act_function())
            if dropout > 0:
                layers_f.append(nn.Dropout(dropout))
        layers_f.append(nn.Linear(width_f, tgt_encoded_dim))
        self.model_f = nn.Sequential(*layers_f)

        if use_sat_f:
            layers_f_sat = []
            layers_f_sat.append(nn.Linear(self.obs_sat, width_f))
            layers_f_sat.append(act_function())
            if dropout > 0:
                layers_f_sat.append(nn.Dropout(dropout))
            for _ in range(depth_f - 1):
                layers_f_sat.append(nn.Linear(width_f, width_f))
                layers_f_sat.append(act_function())
                if dropout > 0:
                    layers_f_sat.append(nn.Dropout(dropout))
            layers_f_sat.append(nn.Linear(width_f, tgt_encoded_dim))
            self.model_f_sat = nn.Sequential(*layers_f_sat)

        n_dim = 2
        input_size_g = n_dim * tgt_encoded_dim
        if use_sat_f:
            input_size_g += tgt_encoded_dim
        else:
            input_size_g += self.obs_sat

        layers_g = []
        layers_g.append(nn.Linear(input_size_g, width_g))
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
        x_sat = x[:, : self.obs_sat]
        x_tgts = x[:, self.obs_sat :]

        # Allows changes in the number of targets during runtime without changing internal variables as long as the input dimension is consistent with the number of targets
        n_tgts = x_tgts.shape[1] // self.features_per_tgt
        x_tgts = x_tgts.view(
            B, n_tgts, self.features_per_tgt
        )  # (B, n_tgts, features_per_tgt)

        if self.use_sat_f:
            x_sat = self.model_f_sat(x_sat)

        if self.concat_sat_f:
            model_f_input = torch.cat(
                [x_tgts, x_sat.unsqueeze(1).expand(-1, n_tgts, -1)], dim=-1
            )
        else:
            model_f_input = x_tgts

        latent_tgts = self.model_f(model_f_input)  # (B, n_tgts, tgt_encoded_dim)

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


class GATActorM32(nn.Module):

    def __init__(
        self,
        inputs: int,
        width_f: int = 32,
        depth_f: int = 4,
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
        width_g_sat: int = 32,
        depth_g_sat: int = 2,
        width_f_sat: int = 32,
        depth_f_sat: int = 2,
        sat_attention_dim: int = 32,
        sat_attention_heads: int = 2,
        sat_encoded_dim: int = 32,
        post_self_attention: bool = False,
        hierarchical: bool = False,
    ):
        super(GATActorM32, self).__init__()

        self.obs_sat = obs_sat
        act_function = nn.ReLU
        self.n_tgts = n_tgts
        self.features_per_tgt = (inputs - self.obs_sat) // n_tgts
        self.attention_depth = attention_depth
        self.non_imaging_actions = non_imaging_actions
        self.post_self_attention = post_self_attention
        self.hierarchical = hierarchical

        self.model_f = self._build_f_model(
            input_dim=self.features_per_tgt,
            width=width_f,
            depth=depth_f,
            output_dim=tgt_encoded_dim,
            act_function=act_function,
            dropout=dropout,
        )

        layers_attention = []
        layers_normalization = []
        sequential_g = []
        for i in range(attention_depth):
            layers_attention.append(
                FastSelfAttention(tgt_encoded_dim, attention_dim, num_heads)
            )
            layers_normalization.append(nn.LayerNorm(tgt_encoded_dim))

            sequential_g.append(
                self._build_g_model(
                    3 * tgt_encoded_dim + sat_encoded_dim,
                    width=width_g,
                    depth=depth_g,
                    output_dim=tgt_encoded_dim,
                    act_function=act_function,
                    dropout=dropout,
                    add_norm=i < attention_depth - 1,
                )
            )

        self.attention_layers = nn.ModuleList(layers_attention)
        self.normalization_layers = nn.ModuleList(layers_normalization)
        self.models_g = nn.ModuleList(sequential_g)
        self.out_layer_g = nn.Linear(tgt_encoded_dim, 1)

        self.model_f_sat = self._build_f_model(
            input_dim=self.obs_sat,
            width=width_f_sat,
            depth=depth_f_sat,
            output_dim=sat_encoded_dim,
            act_function=act_function,
            dropout=dropout,
        )

        sat_attention_layers = []
        sat_normalization_layers = []
        sequential_g_sat = []
        for i in range(attention_depth):
            sat_attention_layers.append(
                FastMultiHeadAttention(
                    sat_encoded_dim,
                    tgt_encoded_dim,
                    sat_attention_dim,
                    sat_attention_heads,
                )
            )
            sat_normalization_layers.append(nn.LayerNorm(sat_encoded_dim))

            sequential_g_sat.append(
                self._build_g_model(
                    2 * tgt_encoded_dim + sat_encoded_dim,
                    width=width_g_sat,
                    depth=depth_g_sat,
                    output_dim=sat_encoded_dim,
                    act_function=act_function,
                    dropout=dropout,
                    add_norm=i < attention_depth - 1,
                )
            )

        self.sat_attention_layers = nn.ModuleList(sat_attention_layers)
        self.sat_normalization_layers = nn.ModuleList(sat_normalization_layers)
        self.models_g_sat = nn.ModuleList(sequential_g_sat)

        out_dim_g_sat = 1 + non_imaging_actions
        self.out_layer_g_sat = nn.Linear(sat_encoded_dim, out_dim_g_sat)

    def _build_f_model(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        act_function,
        dropout: float,
    ):
        layers_f = []
        layers_f.append(nn.Linear(input_dim, width))
        layers_f.append(act_function())
        if dropout > 0:
            layers_f.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers_f.append(nn.Linear(width, width))
            layers_f.append(act_function())
            if dropout > 0:
                layers_f.append(nn.Dropout(dropout))

        layers_f.append(nn.Linear(width, output_dim))
        return nn.Sequential(*layers_f)

    def _build_g_model(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        act_function,
        dropout: float,
        add_norm: bool,
    ):
        layers_g = []
        layers_g.append(nn.Linear(input_dim, width))
        layers_g.append(act_function())
        if dropout > 0:
            layers_g.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers_g.append(nn.Linear(width, width))
            layers_g.append(act_function())
            if dropout > 0:
                layers_g.append(nn.Dropout(dropout))
        layers_g.append(nn.Linear(width, output_dim))
        layers_g.append(act_function())
        if add_norm:
            layers_g.append(nn.LayerNorm(output_dim))
        return nn.Sequential(*layers_g)

    def forward(self, x):
        if isinstance(x, dict) and "obs" in x:
            x = x["obs"]

        B = x.shape[0]

        x_sat = x[:, : self.obs_sat]
        x_tgts = x[:, self.obs_sat :]

        n_tgts = x_tgts.shape[1] // self.features_per_tgt
        x_tgts = x_tgts.view(
            B, n_tgts, self.features_per_tgt
        )  # (B, n_tgts, features_per_tgt)
        latent_tgts = self.model_f(x_tgts)  # (B, n_tgts, tgt_encoded_dim)
        latent_sat = self.model_f_sat(x_sat)  # (B, sat_encoded_dim)

        for i in range(self.attention_depth):
            attention_out_self = self.attention_layers[i](
                latent_tgts
            )  # (B, n_tgts, tgt_encoded_dim)

            latent_tgts_self = self.normalization_layers[i](
                latent_tgts + attention_out_self
            )

            latent_tgts = self.models_g[i](
                torch.cat(
                    [
                        latent_tgts_self,
                        torch.mean(latent_tgts_self, dim=1)
                        .unsqueeze(1)
                        .expand(-1, n_tgts, -1),
                        torch.max(latent_tgts_self, dim=1)
                        .values.unsqueeze(1)
                        .expand(-1, n_tgts, -1),
                        latent_sat.unsqueeze(1).expand(-1, n_tgts, -1),
                    ],
                    dim=-1,
                )
            )  # (B, n_tgts, tgt_encoded_dim)

            if self.post_self_attention:
                latent_tgts_self = latent_tgts

            # Cross attention with satellite features
            sat_attention_out = self.sat_attention_layers[i](
                latent_sat.unsqueeze(1), latent_tgts_self
            )  # (B, 1, sat_attention_dim)

            latent_sat = self.sat_normalization_layers[i](
                latent_sat + sat_attention_out.squeeze(1)
            )  # (B, sat_encoded_dim)

            latent_sat = self.models_g_sat[i](
                torch.cat(
                    [
                        latent_sat,
                        torch.mean(latent_tgts_self, dim=1),
                        torch.max(latent_tgts_self, dim=1).values,
                    ],
                    dim=-1,
                )
            )  # (B, non_imaging_actions)

        logits_tgts = self.out_layer_g(latent_tgts).squeeze(-1)  # (B, n_tgts)

        g_sat_out = self.out_layer_g_sat(latent_sat)  # (B, non_imaging_actions)

        img_modulated = g_sat_out[:, 0:1]
        non_img_logit = g_sat_out[:, 1:]

        if self.hierarchical:
            return torch.cat([non_img_logit, img_modulated, logits_tgts], dim=1)

        return torch.cat([non_img_logit, logits_tgts + img_modulated], dim=1)


class GATModule(PPOTorchRLModule, nn.Module):
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

        self.encoder = lambda x: {ENCODER_OUT: {ACTOR: x, CRITIC: x}}

        config_dict = self.config.model_config_dict

        self.pi_head = GATActorM32(
            inputs=self.config.observation_space.shape[0],
            n_tgts=config_dict["n_targets"],
            obs_sat=config_dict["obs_sat"],
            width_f=config_dict["width_f"],
            depth_f=config_dict["depth_f"],
            tgt_encoded_dim=config_dict["tgt_encoded_dim"],
            attention_depth=config_dict["attention_depth"],
            num_heads=config_dict["num_heads"],
            attention_dim=config_dict["attention_dim"],
            width_g=config_dict["width_g"],
            depth_g=config_dict["depth_g"],
            dropout=config_dict.get("dropout", 0.0),
            non_imaging_actions=config_dict.get("non_imaging_actions", 1),
            width_g_sat=config_dict["width_g_sat"],
            depth_g_sat=config_dict["depth_g_sat"],
            sat_attention_dim=config_dict["sat_attention_dim"],
            sat_attention_heads=config_dict["sat_attention_heads"],
            width_f_sat=config_dict["width_f_sat"],
            depth_f_sat=config_dict["depth_f_sat"],
            sat_encoded_dim=config_dict["sat_encoded_dim"],
            post_self_attention=config_dict["post_self_attention"],
        )

        # Only build the critic network when this is a learner module.
        if not self.config.inference_only or self.framework != "torch":
            self.vf = GATCritic(
                inputs=self.config.observation_space.shape[0],
                n_tgts=config_dict["n_targets"],
                obs_sat=config_dict["obs_sat"],
                width_f=config_dict["critic_width_f"],
                depth_f=config_dict["critic_depth_f"],
                tgt_encoded_dim=config_dict["critic_tgt_encoded_dim"],
                width_g=config_dict["critic_width_g"],
                depth_g=config_dict["critic_depth_g"],
                dropout=config_dict.get("dropout", 0.1),
                use_sat_f=config_dict.get("critic_use_sat_f", False),
                concat_sat_f=config_dict.get("critic_concat_sat_f", False),
            )
            # Holds the parameter names to be removed or renamed when synching
            # from the learner to the inference module.
            self._inference_only_state_dict_keys = {}

        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)
        # It uses inference onlt to collect data without having the critic (rollouts from EnvRunners), which means that I cannot really use the eval mode of the actor

    def pi(
        self, batch: Dict[str, TensorType], inference: bool = False
    ) -> Dict[str, TensorType]:
        pi_outs = {}

        if self.config.inference_only:
            with torch.inference_mode():
                logits = self.pi_head(batch)
        else:
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


class GATModuleH(PPOTorchRLModule, nn.Module):
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

        self.encoder = lambda x: {ENCODER_OUT: {ACTOR: x, CRITIC: x}}

        config_dict = self.config.model_config_dict

        self.non_imaging_actions = config_dict.get("non_imaging_actions", 1)

        self.pi_head = GATActorM32(
            inputs=self.config.observation_space.shape[0],
            n_tgts=config_dict["n_targets"],
            obs_sat=config_dict["obs_sat"],
            width_f=config_dict["width_f"],
            depth_f=config_dict["depth_f"],
            tgt_encoded_dim=config_dict["tgt_encoded_dim"],
            attention_depth=config_dict["attention_depth"],
            num_heads=config_dict["num_heads"],
            attention_dim=config_dict["attention_dim"],
            width_g=config_dict["width_g"],
            depth_g=config_dict["depth_g"],
            dropout=config_dict.get("dropout", 0.1),
            non_imaging_actions=config_dict.get("non_imaging_actions", 1),
            width_g_sat=config_dict["width_g_sat"],
            depth_g_sat=config_dict["depth_g_sat"],
            sat_attention_dim=config_dict["sat_attention_dim"],
            sat_attention_heads=config_dict["sat_attention_heads"],
            width_f_sat=config_dict["width_f_sat"],
            depth_f_sat=config_dict["depth_f_sat"],
            sat_encoded_dim=config_dict["sat_encoded_dim"],
            post_self_attention=config_dict["post_self_attention"],
            hierarchical=True,
        )

        # Only build the critic network when this is a learner module.
        if not self.config.inference_only or self.framework != "torch":
            self.vf = GATCritic(
                inputs=self.config.observation_space.shape[0],
                n_tgts=config_dict["n_targets"],
                obs_sat=config_dict["obs_sat"],
                width_f=config_dict["critic_width_f"],
                depth_f=config_dict["critic_depth_f"],
                tgt_encoded_dim=config_dict["critic_tgt_encoded_dim"],
                width_g=config_dict["critic_width_g"],
                depth_g=config_dict["critic_depth_g"],
                dropout=config_dict.get("dropout", 0.1),
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

        logits_actions = logits[:, : self.non_imaging_actions + 1]
        logits_tgts = logits[:, self.non_imaging_actions + 1 :]

        image_action_idx = self.non_imaging_actions

        logp_actions = torch.log_softmax(logits_actions, dim=-1)
        logp_tgts = torch.log_softmax(logits_tgts, dim=-1)

        # Final flat logits/log-probs over the environment action space:
        #
        # non-imaging action k:
        #   log p(a = k) = log p(branch = k)
        #
        # imaging target i:
        #   log p(a = image_target_i)
        #       = log p(branch = image) + log p(target = i | image)
        logits_non_image = logp_actions[:, : self.non_imaging_actions]

        logits_image_tgts = (
            logp_actions[:, image_action_idx : image_action_idx + 1] + logp_tgts
        )

        # Shape: (B, non_imaging_actions + n_tgts)
        hierarchical_logits = torch.cat(
            [logits_non_image, logits_image_tgts],
            dim=-1,
        )

        if inference:
            discrete_action_dist = TorchCategorical.from_logits(
                hierarchical_logits
            ).to_deterministic()
            discrete_action = discrete_action_dist.sample()
        else:
            discrete_action_dist = TorchCategorical.from_logits(hierarchical_logits)
            discrete_action = discrete_action_dist.sample()

        discrete_action_logp = discrete_action_dist.logp(discrete_action)

        pi_outs[Columns.ACTION_LOGP] = discrete_action_logp
        pi_outs[Columns.ACTION_DIST_INPUTS] = hierarchical_logits
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
