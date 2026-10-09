"""Explicit PPO settings and loss diagnostics for completion-policy experiments."""

from dataclasses import asdict, dataclass
import math

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from bsk_rl.utils.rllib.discounting import TimeDiscountedGAEPPOTorchLearner


@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float | list[list[float]] = 3e-5
    entropy_coeff: float = 0.02
    grad_clip: float = 1.0
    num_sgd_iter: int = 1
    minibatch_size: int | None = None
    vf_clip_param: float = 10.0
    vf_loss_coeff: float = 1.0
    clip_param: float = 0.3
    use_kl_loss: bool = True

    def __post_init__(self):
        schedule = self.learning_rate
        if isinstance(schedule, (int, float)):
            schedule = [[0, schedule]]
        if not schedule or schedule[0][0] != 0:
            raise ValueError("Learning-rate schedule must start at timestep zero.")
        previous = -1
        for timestep, rate in schedule:
            if (
                not math.isfinite(timestep)
                or timestep <= previous
                or not math.isfinite(rate)
                or rate <= 0
            ):
                raise ValueError(
                    "Learning-rate points must be finite, ordered, and positive."
                )
            previous = timestep
        for name in ("grad_clip", "vf_clip_param", "clip_param"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("entropy_coeff", "vf_loss_coeff"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.num_sgd_iter < 1 or (
            self.minibatch_size is not None and self.minibatch_size < 8
        ):
            raise ValueError(
                "Require positive epochs and minibatches of at least eight."
            )

    @classmethod
    def from_profile(cls, name):
        if name == "legacy_pilot":
            return cls()
        if name == "value_repair":
            return cls(vf_clip_param=10000.0)
        if name == "learning_recovery":
            # Controlled candidate: preserve LR/KL/entropy, repair the value cap,
            # and allow ten PPO passes instead of the pilot's single pass.
            return cls(vf_clip_param=10000.0, num_sgd_iter=10, minibatch_size=128)
        if name == "amos_best":
            # Exact training overrides in the selected fixed-mixed alpha=0.1
            # attention run 4osb53df (paper checkpoint 000119). Unspecified PPO
            # value/minibatch defaults are made explicit for pinned Ray 2.35.
            return cls(
                learning_rate=[[0, 1e-5], [40000, 5.970334799302134e-7]],
                entropy_coeff=0.023694512589767867,
                grad_clip=0.3104924935285628,
                num_sgd_iter=30,
                minibatch_size=128,
                vf_clip_param=10.0,
                vf_loss_coeff=1.0,
                clip_param=0.14701727973480344,
                use_kl_loss=False,
            )
        raise ValueError(f"Unknown PPO profile: {name}")

    def training_kwargs(self, train_batch_size):
        values = asdict(self)
        values["lr"] = values.pop("learning_rate")
        values["sgd_minibatch_size"] = values.pop("minibatch_size") or max(
            8, train_batch_size // 2
        )
        return values


def actual_ppo_settings(config):
    """Read the effective algorithm, including settings restored from checkpoint."""
    return {
        key: getattr(config, key)
        for key in (
            "lr",
            "entropy_coeff",
            "grad_clip",
            "num_sgd_iter",
            "sgd_minibatch_size",
            "vf_clip_param",
            "vf_loss_coeff",
            "clip_param",
            "use_kl_loss",
            "train_batch_size",
            "gamma",
            "lambda_",
            "learner_config_dict",
            "grad_clip_by",
            "count_steps_by",
        )
    }


class CompletionDiagnosticLearner(TimeDiscountedGAEPPOTorchLearner):
    """Leave Ray's PPO loss intact and retain measured loss-scale diagnostics."""

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        directory = self.config.learner_config_dict.get("batch_evidence_directory")
        if directory:
            from examples.multiagent_imaging.learner_batch_evidence import (
                CaptureLearnerBatch,
            )

            self._learner_connector.append(CaptureLearnerBatch(directory))

    def _compute_gae_from_episodes(self, *, episodes=None):
        directory = self.config.learner_config_dict.get("batch_evidence_directory")
        if directory:
            from examples.multiagent_imaging.learner_batch_evidence import (
                save_raw_episodes,
            )

            save_raw_episodes(
                directory,
                list(
                    self._learner_connector.single_agent_episode_iterator(
                        episodes, agents_that_stepped_only=False
                    )
                ),
            )
        return super()._compute_gae_from_episodes(episodes=episodes)

    def _compute_values(self, batch):
        values = super()._compute_values(batch)
        directory = self.config.learner_config_dict.get("batch_evidence_directory")
        if directory:
            from examples.multiagent_imaging.learner_batch_evidence import (
                save_numeric_batch,
            )

            save_numeric_batch(
                directory, "bootstrap-values.npz", {Columns.VF_PREDS: values["imager"]}
            )
        return values

    def compute_gradients(self, loss_per_module, **kwargs):
        gradients = super().compute_gradients(loss_per_module, **kwargs)
        for module_id, module in self.module.items():
            norms = {"actor": 0.0, "critic": 0.0}
            for name, parameter in module.named_parameters():
                gradient = gradients.get(self.get_param_ref(parameter))
                if gradient is not None:
                    norms["critic" if name.startswith("vf.") else "actor"] += float(
                        gradient.square().sum()
                    )
            total = math.sqrt(sum(norms.values()))
            clip = self.config.get_config_for_module(module_id).grad_clip
            scale = min(1.0, clip / (total + 1e-6)) if clip else 1.0
            for kind, squared in norms.items():
                self.metrics.log_value(
                    (module_id, f"{kind}_gradient_l2_before_clip"),
                    math.sqrt(squared),
                    window=1,
                )
                self.metrics.log_value(
                    (module_id, f"{kind}_gradient_l2_after_shared_clip"),
                    math.sqrt(squared) * scale,
                    window=1,
                )
            self.metrics.log_value(
                (module_id, "shared_global_clip_scale"), scale, window=1
            )
        return gradients

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        with torch.no_grad():
            targets = batch[Postprocessing.VALUE_TARGETS]
            values = fwd_out[Columns.VF_PREDS]
            keep = batch.get(
                Columns.LOSS_MASK, torch.ones_like(targets, dtype=torch.bool)
            ).bool()
            residual = (values - targets).square()[keep]
            logits = fwd_out[Columns.ACTION_DIST_INPUTS][keep]
            probabilities = torch.softmax(logits, dim=-1)
            valid_count = (logits > -1e8).sum(dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(
                dim=-1
            )
            normalized = torch.where(
                valid_count > 1,
                entropy / valid_count.float().log().clamp_min(1e-9),
                1.0,
            )
            module = self.module[module_id].unwrapped()
            distribution = module.get_train_action_dist_cls().from_logits(
                fwd_out[Columns.ACTION_DIST_INPUTS]
            )
            ratio = torch.exp(
                distribution.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
            )[keep]
            # Ray defaults reduce="mean", window=None to an EMA. Retain sums
            # and counts instead so unequal minibatches are weighted correctly.
            metrics = {
                "vf_clamped_presentations": (residual > config.vf_clip_param).sum(),
                "vf_squared_error_sum": residual.sum(),
                "normalized_entropy_sum": normalized.sum(),
                "ppo_ratio_clipped_presentations": (
                    (ratio - 1).abs() > config.clip_param
                ).sum(),
            }
            for key, value in metrics.items():
                self.metrics.log_value(
                    (module_id, key),
                    float(value),
                    reduce="sum",
                    window=None,
                    clear_on_reduce=True,
                )
            self.metrics.log_value(
                # A distinct key is required when resuming older checkpoints:
                # Ray saves each metric's reduction method with algorithm state.
                (module_id, "value_target_abs_max_observed"),
                float(targets[keep].abs().max()),
                reduce="max",
                window=None,
                clear_on_reduce=True,
            )
            self.metrics.log_value(
                (module_id, "valid_minibatch_presentations"),
                int(keep.sum()),
                reduce="sum",
                window=None,
                clear_on_reduce=True,
            )
        return loss
