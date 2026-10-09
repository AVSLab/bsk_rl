"""Policy export runs actual weights; PPO resume retains optimizer and seed progress."""

from dataclasses import replace
import numpy as np
import pytest
import ray
import torch
from ray.rllib.algorithms.ppo import PPO

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.train import build_ppo_config
from examples.multiagent_imaging.checkpoints import (
    load_policy,
    save_checkpoint,
    resume_checkpoint,
    optimizer_digest,
    policy_callable,
)
from examples.multiagent_imaging.evaluate import run_rollout
from examples.multiagent_imaging.training_audit import parameter_change


@pytest.mark.parametrize("amos_parameters", [False, True])
def test_checkpoint_outputs_optimizer_resume_and_seed_progress(
    tmp_path, amos_parameters
):
    torch.set_num_threads(1)
    config = MultiAgentImagingConfig(
        n_sensors=4,
        n_targets=4,
        n_candidates=2,
        episode_duration_s=360,
        max_step_duration_s=120,
        imaging_duration_s=120,
        information_case="completion",
        communication_mode="directed",
        target_population="mixed_50_30_20",
        sensor_constellation="walker_delta",
        walker_planes=2,
        walker_phasing=1,
        seed=18,
        **(
            dict(
                discount_per_s=0.9915045428565076,
                gae_lambda_per_decision=0.8713548569911232,
                policy_model_profile="amos_attention",
            )
            if amos_parameters
            else {}
        ),
    )
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    algorithm = None
    try:
        from examples.multiagent_imaging.optimization import PPOSettings

        algorithm = PPO(
            build_ppo_config(
                config,
                train_batch_size=128 if amos_parameters else 16,
                ppo_settings=PPOSettings.from_profile("amos_best")
                if amos_parameters
                else None,
                batch_evidence_directory=str(tmp_path / "batch_evidence")
                if amos_parameters
                else None,
            )
        )
        initial_result = algorithm.train()
        assert initial_result["learners"]["imager"]["actor_gradient_l2_before_clip"] > 0
        assert (
            initial_result["learners"]["imager"]["critic_gradient_l2_before_clip"] > 0
        )
        if amos_parameters:
            batch = np.load(tmp_path / "batch_evidence" / "learner-batch.npz")
            assert np.all(batch["actions"] >= 0)
            assert np.all(batch["actions"] < batch["action_dist_inputs"].shape[-1])
            assert np.isfinite(batch["value_targets"]).all()
            assert len(batch["obs"]) == len(batch["value_targets"])
            assert (tmp_path / "batch_evidence" / "raw-episodes.json.gz").exists()
            assert (
                np.load(tmp_path / "batch_evidence" / "bootstrap-values.npz")[
                    "vf_preds"
                ].size
                > 0
            )
            from examples.multiagent_imaging.fixed_batch_diagnostic import (
                independently_recompute,
            )

            _, target_audit = independently_recompute(
                tmp_path / "batch_evidence", config
            )
            assert target_audit["passed"]
        env = algorithm.env_runner.env.unwrapped.par_env
        probes = np.asarray(
            [s.get_obs() for s in env.sensing_satellites], dtype=np.float32
        )
        before = algorithm.get_weights(["imager"])["imager"]
        expected_optimizer = optimizer_digest(algorithm)
        next_episode = env._episode_seed_index
        # Older diagnostic checkpoints stored this misleadingly named metric
        # with Ray's default EMA reduction. Preserve it during telemetry migration.
        algorithm.metrics.log_value(
            ("learners", "imager", "value_target_abs_max"), 8.0, reduce="mean"
        )
        save_checkpoint(algorithm, config, tmp_path, probes)
        module, evidence = load_policy(tmp_path, config)
        assert evidence["max_logit_error"] == 0
        assert evidence["matched_actions"]
        with pytest.raises(ValueError, match="schema mismatch"):
            load_policy(tmp_path, replace(config, n_candidates=3))
        # Run the exported policy through an actual simulator episode, with a held-
        # out seed. This path cannot silently fall back to the heuristic.
        calls = []
        actual_policy = policy_callable(module)

        def counted_policy(obs):
            result = actual_policy(obs)
            calls.append(result)
            return result

        rollout = run_rollout(replace(config, seed=19), policy=counted_policy)
        assert calls and len(calls) == sum(
            rollout["coordination"]["policy_decisions"].values()
        )
        assert rollout["sim_time_s"] == config.episode_duration_s
        algorithm.stop()
        algorithm = resume_checkpoint(tmp_path, config)
        from bsk_rl.utils.rllib.discounting import CondenseMultiStepActions

        condensers = [
            c
            for c in algorithm.learner_group._learner._learner_connector.connectors
            if isinstance(c, CondenseMultiStepActions)
        ]
        assert len(condensers) == 1 and condensers[0].gamma == config.discount_per_s
        assert algorithm.training_iteration == 1
        assert optimizer_digest(algorithm) == expected_optimizer
        assert (
            algorithm.env_runner.env.unwrapped.par_env._episode_seed_index
            == next_episode
        )
        result = algorithm.train()
        assert algorithm.training_iteration == 2
        change = parameter_change(before, algorithm.get_weights(["imager"])["imager"])
        assert change["l2"] > 0
        assert np.isfinite(result["learners"]["imager"]["total_loss"])
        assert result["learners"]["imager"]["value_target_abs_max_observed"] > 0
        assert (
            algorithm.env_runner.env.unwrapped.par_env._episode_seed_index
            > next_episode
        )
        assert optimizer_digest(algorithm) != expected_optimizer
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
