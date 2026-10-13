"""Shared target-wise PPO training for sensing agents only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

try:
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
except (ImportError, ModuleNotFoundError):
    from ray.rllib.core.rl_module.marl_module import (
        MultiAgentRLModuleSpec as MultiRLModuleSpec,
    )

from bsk_rl.utils.rllib.discounting import (
    CondenseMultiStepActions,
    ContinuePreviousAction,
    MakeAddedStepActionValid,
)
from bsk_rl.utils.rllib.target_gnn_module import GNNModule

from examples.multiagent_imaging.config import (
    GLOBAL_FEATURES,
    MultiAgentImagingConfig,
    NON_IMAGING_ACTIONS,
    OBSERVATION_VERSION,
)
from examples.multiagent_imaging.environment import build_environment
from examples.multiagent_imaging.optimization import (
    CompletionDiagnosticLearner,
    PPOSettings,
    actual_ppo_settings,
)
from bsk_rl.obs.completion_observations import (
    candidate_snapshot,
    can_continue,
    peer_snapshot,
)


SHARED_POLICY_ID = "imager"
PPO_ENTROPY_COEFF = 0.02
PPO_GRAD_CLIP = 1.0
ACTION_HEAD_INIT_GAIN = 0.01
INITIAL_IMAGING_LOGIT_BIAS = 0.01


@dataclass(frozen=True)
class LauncherResources:
    """Single-node Ray layout; one Basilisk environment per rollout process."""

    num_env_runners: int = 0
    cpus_per_env_runner: int = 1
    num_learners: int = 0
    gpus_per_learner: float = 0.0
    ray_cpus: int = 2
    torch_threads: int = 1
    sample_timeout_s: float = 1800.0


def worker_seed(base_seed, worker_index=0, vector_index=0):
    """Stable independent 32-bit streams keyed by experiment, worker and vector IDs."""
    return int(
        np.random.SeedSequence(
            [int(base_seed), int(worker_index), int(vector_index), 731]
        ).generate_state(1)[0]
    )


def make_rllib_env(env_config):
    config = dict(env_config)
    config["seed"] = worker_seed(
        config["seed"],
        getattr(env_config, "worker_index", 0),
        getattr(env_config, "vector_index", 0),
    )
    experiment = MultiAgentImagingConfig(**config)
    import os
    if os.environ.get("BSK_REQUIRE_EXTERNAL_PSUTIL") == "1":
        from examples.multiagent_imaging.cluster.effective_imports import require_effective_imports
        from examples.multiagent_imaging.readiness import write_json
        runtime=require_effective_imports()
        folder=os.environ.get("BSK_STUDY_RUNTIME_OUTPUT")
        if folder: write_json(Path(folder)/f"worker-{getattr(env_config,'worker_index',0)}-{runtime['pid']}.json", runtime)
    wrapped = CompletionPettingZooEnv(build_environment(experiment))
    if experiment.is_study:
        # Ray 2.35's checker unconditionally takes an otherwise unaccounted
        # physical step. Our focused mask/slot tests plus space-only check avoid
        # that partial startup rollout. Actual sampled readiness remains gated.
        from examples.multiagent_imaging.readiness import schema
        import ray.rllib.env.multi_agent_env_runner as runner_module
        wrapped._study_contract=schema(experiment)
        runner_module.check_multiagent_environments=check_study_static_contract
    import os
    if (os.environ.get("BSK_ZERO_UPDATE_RESTORE") == "1"
            or os.environ.get("BSK_STARTUP_ONLY_GATE") == "1"):
        # Ray 2.35's startup checker otherwise calls env.step even on restoration.
        # Restore/startup-only processes validate spaces without advancing time.
        from examples.multiagent_imaging.readiness import schema
        import ray.rllib.env.multi_agent_env_runner as runner_module
        wrapped._restore_contract = schema(experiment)
        wrapped._restore_only_guard = True
        wrapped._restore_step_calls = 0
        runner_module.check_multiagent_environments = check_restore_only_contract
    return wrapped


def check_study_static_contract(env):
    contract=env._study_contract
    for agent in env.get_agent_ids():
        assert env.observation_space[agent].shape == (contract["observation"]["size"],)
        assert env.action_space[agent].n == contract["actions"]["size"]


def check_restore_only_contract(env):
    assert env._restore_only_guard, "Restore-only environment guard missing"
    contract = env._restore_contract
    for agent in env.get_agent_ids():
        assert env.observation_space[agent].shape == (contract["observation"]["size"],)
        assert env.action_space[agent].n == contract["actions"]["size"]


class CompletionPettingZooEnv(ParallelPettingZooEnv):
    """Let RLlib's API checker sample valid state-dependent discrete actions."""

    def step(self, action_dict):
        if getattr(self, "_restore_only_guard", False):
            self._restore_step_calls += 1
            raise RuntimeError("Environment stepping is forbidden in a zero-update restore gate")
        return super().step(action_dict)

    def action_space_sample(self, agent_ids=None):
        """Sample the same candidate/continue mask used by the policy actor."""
        selected = set(self.par_env.possible_agents if agent_ids is None else agent_ids)
        result = {}
        for sensor in self.par_env.sensing_satellites:
            if sensor.name not in selected:
                continue
            if getattr(sensor,"completion_study",False):
                from bsk_rl.obs.completion_observations import private_action_mask
                result[sensor.name]=sensor.action_space.sample(mask=private_action_mask(sensor))
                continue
            count = sensor.completion_n_candidates
            snapshot = candidate_snapshot(sensor, count)
            mask = np.asarray(
                [
                    1,
                    int(
                        sensor.dynamics.storage_level_fraction
                        > sensor.completion_empty_storage_mask_fraction
                        and sensor.dynamics.battery_charge_fraction
                        > sensor.completion_minimum_task_battery_fraction
                    ),
                    int(
                        sensor.dynamics.battery_charge_fraction
                        > sensor.completion_minimum_task_battery_fraction
                    ),
                    int(
                        sensor.completion_communication_mode == "broadcast"
                        and sensor.dynamics.battery_charge_fraction
                        > sensor.completion_minimum_task_battery_fraction
                    ),
                    int(can_continue(sensor)),
                    *(
                        int(
                            t is not None
                            and sensor.dynamics.battery_charge_fraction
                            > sensor.completion_minimum_task_battery_fraction
                        )
                        for t in snapshot.targets
                    ),
                    *(
                        [
                            int(
                                p is not None
                                and sensor.completion_communicator.transmit_eligible(sensor, p)
                                and sensor.dynamics.battery_charge_fraction
                                > sensor.completion_minimum_task_battery_fraction
                            )
                            for p in peer_snapshot(sensor).peers
                        ]
                        if sensor.completion_communication_mode == "directed"
                        else []
                    ),
                ],
                dtype=np.int8,
            )
            result[sensor.name] = sensor.action_space.sample(mask=mask)
        return result


def make_shared_policy_mapping(sensor_ids: set[str]):
    """Map only the explicitly configured sensing IDs to one shared module."""
    allowed = frozenset(sensor_ids)

    def mapping(agent_id, *args, **kwargs):
        if agent_id not in allowed:
            raise KeyError(f"Non-sensing agent {agent_id!r} reached policy mapping.")
        return SHARED_POLICY_ID

    return mapping


def target_attention_config(config: MultiAgentImagingConfig) -> dict:
    model = {
        "n_targets": config.n_candidates,
        "n_peers": config.n_peers,
        "communication_mode": config.communication_mode,
        "peer_features": config.peer_features,
        "target_features": config.target_features,
        "peer_observation_valid_feature": 16 if config.is_study else 11,
        "information_case": config.information_case,
        "obs_sat": GLOBAL_FEATURES,
        "width_f": 64,
        "depth_f": 2,
        "block_f": False,
        "width_g": 64,
        "depth_g": 2,
        "tgt_encoded_dim": 64,
        "attention_depth": 1,
        "num_heads": 2,
        "attention_dim": 64,
        "dropout": 0.0,
        "critic_tgt_encoded_dim": 64,
        "critic_width_f": 64,
        "critic_depth_f": 2,
        "critic_block_f": False,
        "critic_width_g": 64,
        "critic_depth_g": 2,
        "critic_pooling_std": False,
        "non_imaging_actions": NON_IMAGING_ACTIONS,
        "condition_on_spacecraft": True,
        "completion_mask": True,
        "observation_version": config.observation_version,
        "empty_storage_mask_fraction": config.empty_storage_mask_fraction,
        "minimum_task_battery_fraction": config.minimum_task_battery_fraction,
        "action_head_init_gain": ACTION_HEAD_INIT_GAIN,
        "initial_imaging_logit_bias": INITIAL_IMAGING_LOGIT_BIAS,
    }
    if config.policy_model_profile == "amos_attention":
        # Match the common target-attention/critic capacity of AMOS run 4osb53df.
        # Private peer inputs and the completion action/mask contract remain
        # the multi-agent architecture; AMOS had no completion-sharing peer branch.
        model.update(
            width_f=256,
            depth_f=2,
            width_g=128,
            depth_g=4,
            tgt_encoded_dim=128,
            attention_depth=1,
            num_heads=2,
            attention_dim=128,
            dropout=0.1,
            critic_tgt_encoded_dim=128,
            critic_width_f=256,
            critic_depth_f=2,
            critic_width_g=64,
            critic_depth_g=3,
        )
    return model


def build_ppo_config(
    experiment: MultiAgentImagingConfig,
    *,
    train_batch_size: int,
    resources: LauncherResources | None = None,
    ppo_settings: PPOSettings | None = None,
    batch_evidence_directory: str | None = None,
) -> PPOConfig:
    env_name = "MultiAgentImaging-RLlib"
    resources = resources or LauncherResources()
    ppo_settings = ppo_settings or PPOSettings()
    learner_options = {
        "reward_time": "step_start",
        "lambda_time": experiment.gae_lambda_time,
    }
    if batch_evidence_directory:
        learner_options["batch_evidence_directory"] = batch_evidence_directory
    register_env(env_name, make_rllib_env)
    sensor_ids = {f"sensor_{index}" for index in range(experiment.n_sensors)}
    policy_mapping = make_shared_policy_mapping(sensor_ids)

    from examples.multiagent_imaging.training_audit import CompletionAuditCallbacks

    config = (
        PPOConfig()
        .callbacks(CompletionAuditCallbacks)
        .environment(env=env_name, env_config=experiment.to_dict())
        .framework("torch")
        .debugging(seed=experiment.seed)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=resources.num_env_runners,
            num_cpus_per_env_runner=resources.cpus_per_env_runner,
            num_envs_per_env_runner=1,
            sample_timeout_s=resources.sample_timeout_s,
            rollout_fragment_length="auto",
            # A physical task may span many peer events. Complete episodes avoid
            # dropping its start/log-probability across rollout-fragment boundaries.
            batch_mode="complete_episodes",
            module_to_env_connector=lambda env: (ContinuePreviousAction(),),
        )
        .resources(num_gpus=0)
        .learners(
            num_learners=resources.num_learners,
            num_gpus_per_learner=resources.gpus_per_learner,
        )
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=policy_mapping,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={
                    SHARED_POLICY_ID: RLModuleSpec(
                        module_class=GNNModule,
                        model_config_dict=target_attention_config(experiment),
                    )
                }
            )
        )
        .training(
            gamma=experiment.discount_per_s,
            lambda_=experiment.effective_gae_lambda,
            # The old pilot inherited RLlib's entropy_coeff=0.0.  Four complete
            # mission episodes per update were too little evidence to let a tiny
            # operational-head preference become a trustworthy argmax.  Keep
            # categorical exploration explicit during this recovery pilot.
            train_batch_size=train_batch_size,
            **ppo_settings.training_kwargs(train_batch_size),
            learner_connector=lambda obs_space, act_space: (
                MakeAddedStepActionValid(expected_train_batch_size=train_batch_size),
                CondenseMultiStepActions(gamma=experiment.discount_per_s),
            ),
            learner_class=CompletionDiagnosticLearner,
            learner_config_dict=learner_options,
        )
    )
    return config


def real_initial_observation_batch(experiment: MultiAgentImagingConfig) -> np.ndarray:
    """Build the exact mission reset observed by all sensing agents.

    This is intentionally a real Basilisk reset rather than a shape-only tensor:
    it catches missing candidate slots, wrong feature order, and mask/action drift
    before a cluster allocation is spent collecting a PPO rollout.
    """
    environment = build_environment(experiment)
    observations, _ = environment.reset(seed=experiment.seed)
    batch = np.asarray(
        [observations[f"sensor_{index}"] for index in range(experiment.n_sensors)],
        dtype=np.float32,
    )
    close = getattr(environment, "close", None)
    if callable(close):
        close()
    return batch


def train_run(
    experiment,
    *,
    iterations=1,
    train_batch_size=64,
    output,
    resources=None,
    resume=None,
    ppo_settings=None,
    wandb_mode="disabled",
    require_wandb=False,
    save_initial=False,
    capture_learner_batch=False,
    max_true_decisions=None,
    max_sampled_true_decisions=None,
    max_training_episodes=None,
    max_wall_seconds=None,
):
    """Bounded updates with actual sampler, optimizer, checkpoint and cost evidence."""
    import json
    import os
    import torch
    from dataclasses import asdict
    from examples.multiagent_imaging.checkpoints import (
        save_checkpoint,
        resume_checkpoint,
    )
    from examples.multiagent_imaging.readiness import (
        measure,
        provenance,
        write_json,
        save_reproduction_bundle,
    )
    from examples.multiagent_imaging.training_audit import (
        initial_policy_gate,
        parameter_change,
        policy_distribution_audit,
        take_audits,
    )
    from examples.multiagent_imaging.wandb_logging import TrainingLogger

    resources = resources or LauncherResources()
    if resume and iterations == 0:
        os.environ["BSK_ZERO_UPDATE_RESTORE"] = "1"
    torch.set_num_threads(resources.torch_threads)
    output = Path(output).resolve()
    if (output / "updates.json").exists():
        raise FileExistsError(
            "Use a fresh output directory; existing update evidence must not be overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "provenance.json", provenance(experiment))
    write_json(output / "launcher.json", asdict(resources))
    save_reproduction_bundle(output)
    from examples.multiagent_imaging.memory_profile import MemoryProfile, ray_memory_options
    ray_options = ray_memory_options()
    memory_profile = MemoryProfile(output)
    memory_profile.start()
    # Include the initial probe fixture in the driver memory measurements.
    # A zero-update gate uses saved probes; do not construct a new mission fixture.
    try:
        initial_observations = (
            None if resume and iterations == 0 else real_initial_observation_batch(experiment)
        )
        memory_profile.mark("after-initial-probes")
    except Exception:
        memory_profile.close()
        raise
    if experiment.is_study:
        from examples.multiagent_imaging.sampling_budget import configure_environment
        write_json(output / "sampling-budget.json", configure_environment(
            max_training_episodes, max_sampled_true_decisions, resources.num_env_runners))
        os.environ["BSK_STUDY_EPISODE_SPOOL"] = str(output/"episode-spool")
        os.environ["BSK_STUDY_RUNTIME_OUTPUT"]=str(output/"effective-worker-runtime")
    if os.environ.get("BSK_RL_RAY_TMPDIR"):
        ray_options["_temp_dir"] = os.environ["BSK_RL_RAY_TMPDIR"]
    try:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=resources.ray_cpus,
            include_dashboard=False,
            **ray_options,
        )
    except Exception:
        memory_profile.close()
        raise
    algorithm = None
    logger = None
    failed = True
    records = []
    try:
        with measure() as startup:
            algorithm = (
                resume_checkpoint(resume, experiment, resources=resources)
                if resume
                else PPO(
                    build_ppo_config(
                        experiment,
                        train_batch_size=train_batch_size,
                        resources=resources,
                        ppo_settings=ppo_settings,
                        batch_evidence_directory=str(output / "learner_batch")
                        if capture_learner_batch
                        else None,
                    )
                )
            )
        write_json(output / "startup.json", startup)
        memory_profile.mark("after-startup")
        if experiment.is_study and resume is None:
            from examples.multiagent_imaging.worker_initial_evidence import initialize_shared_policy
            initialize_shared_policy(algorithm, output)
        effective = actual_ppo_settings(algorithm.config)
        if resume and ppo_settings is not None:
            different = {
                key: (effective[key], value)
                for key, value in ppo_settings.training_kwargs(train_batch_size).items()
                if effective[key] != value
            }
            if different:
                raise ValueError(
                    f"Requested PPO settings differ from saved optimizer configuration: {different}"
                )
        write_json(output / "training_hyperparameters.json", effective)
        logger = TrainingLogger(
            output,
            {
                "mission": experiment.to_dict(),
                "ppo": effective,
                "resources": asdict(resources),
            },
            mode=wandb_mode,
            resume=resume,
            required=require_wandb,
        )
        logger.iteration=algorithm.training_iteration
        if experiment.is_study and iterations>0: logger.start_heartbeat()
        import signal,time
        tranche_start=time.monotonic(); stop_requested=[]
        def stop_at_boundary(signum, frame): stop_requested.append(signum)
        prior_handlers={}
        for signum in (signal.SIGTERM,signal.SIGUSR1):
            prior_handlers[signum]=signal.signal(signum,stop_at_boundary)
        policy_decisions_lifetime = 0
        if resume:
            policy_decisions_lifetime = json.loads(
                (Path(resume) / "manifest.json").read_text()
            ).get("policy_decisions_lifetime", 0)
        starting_true_decisions=policy_decisions_lifetime
        if experiment.is_study:
            algorithm.study_lifetime_state=logger.checkpoint_state()
            if not (resume and iterations==0): algorithm.study_accounting_state=logger.accounting_state()
        if resume is None:
            write_json(
                output / "initial_policy_gate.json",
                initial_policy_gate(
                    algorithm.get_module(SHARED_POLICY_ID),
                    initial_observations,
                    experiment,
                ),
            )
            if save_initial:
                initial_checkpoint = output / "checkpoint_0000"
                save_checkpoint(
                    algorithm, experiment, initial_checkpoint, initial_observations
                )
                initial_meta=json.loads((initial_checkpoint/"manifest.json").read_text())
                initial_meta.update(policy_decisions_lifetime=0,training_hyperparameters=effective)
                write_json(initial_checkpoint/"manifest.json",initial_meta)
                logger.save_identity(initial_checkpoint)
        accounted_episode_ids = set(logger.checkpoint_state()['episode_hashes']) if experiment.is_study else set()
        for _ in range(iterations):
            logger.check_background()
            # Stop only at a completed update/checkpoint boundary. Budget an
            # entire worst-case collection+optimization interval before starting.
            if stop_requested or (max_wall_seconds is not None and time.monotonic()-tranche_start>=max_wall_seconds): break
            new_decisions = policy_decisions_lifetime - starting_true_decisions
            if max_true_decisions is not None and new_decisions >= max_true_decisions: break
            before = algorithm.get_weights([SHARED_POLICY_ID])[SHARED_POLICY_ID]
            memory_profile.mark("before-sampling-and-learning")
            with measure() as measurement:
                result = algorithm.train()
            memory_profile.mark("after-sampling-and-learning")
            # Preserve evidence and fail before an incomplete worker set can be
            # accepted as a checkpoint. This readback never samples or restores.
            from examples.multiagent_imaging.checkpoint_topology import collect_all_workers, WorkerTopologyError
            from examples.multiagent_imaging.checkpoints import _runner_seed_state
            try:
                _, topology = collect_all_workers(algorithm, lambda runner: _runner_seed_state(runner, include_pending=False),
                    phase="after-update", require_accounting=False)
                write_json(output / f"worker-topology-{algorithm.training_iteration:04d}.json", topology)
            except WorkerTopologyError as error:
                write_json(output / "worker-topology-failed.json", error.evidence)
                raise
            after = algorithm.get_weights([SHARED_POLICY_ID])[SHARED_POLICY_ID]
            change = parameter_change(before, after)
            audits = [
                item
                for worker in algorithm.env_runner_group.foreach_worker(take_audits)
                for item in worker
            ]
            episode_accounting = None
            previous_true_decisions = policy_decisions_lifetime
            if experiment.is_study:
                from examples.multiagent_imaging.episode_accounting import reconcile_episode_audits
                audits, episode_accounting = reconcile_episode_audits(
                    audits, os.environ.get('BSK_STUDY_EPISODE_SPOOL'), accounted_episode_ids
                )
            if not audits:
                raise AssertionError("No complete episodes were collected.")
            partial = [
                a
                for a in audits
                if not np.isclose(
                    a["simulated_seconds"],
                    experiment.episode_duration_s,
                    rtol=0,
                    atol=1e-6,
                )
            ]
            for audit in audits: logger.ingest_episode(audit)
            if partial:
                write_json(output / "failed_horizon_episodes.json", partial)
                raise AssertionError(
                    "Training episodes ended before the requested horizon; failed evidence saved."
                )
            learners = result["learners"]
            losses = {
                key: float(learners[SHARED_POLICY_ID][key])
                for key in ("total_loss", "policy_loss", "vf_loss")
            }
            if not all(np.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Nonfinite PPO losses.")
            gradients = learners["__all_modules__"]
            learner_metrics = {}
            for key, value in learners[SHARED_POLICY_ID].items():
                array = np.asarray(value)
                if array.ndim == 0 and np.issubdtype(array.dtype, np.number):
                    # Ray can return NaN for an obsolete, empty metric after
                    # checkpoint migration, or an undefined explained variance.
                    # Preserve unavailability as null, never as a measured zero.
                    learner_metrics[key] = float(array) if np.isfinite(array) else None
            presentations = learner_metrics["valid_minibatch_presentations"]
            if presentations is None or presentations <= 0:
                raise FloatingPointError("Missing valid learner presentations.")
            for ratio, total in (
                ("vf_clamp_fraction", "vf_clamped_presentations"),
                ("vf_mse", "vf_squared_error_sum"),
                ("normalized_entropy", "normalized_entropy_sum"),
                ("ppo_ratio_clip_fraction", "ppo_ratio_clipped_presentations"),
            ):
                if learner_metrics[total] is None:
                    raise FloatingPointError(f"Nonfinite required diagnostic: {total}")
                learner_metrics[ratio] = learner_metrics[total] / presentations
            sim_seconds = sum(a["simulated_seconds"] for a in audits)
            measurement.update(
                simulated_seconds=sim_seconds,
                sim_seconds_per_wall_second=sim_seconds / measurement["wall_time_s"],
            )
            record = dict(
                iteration=algorithm.training_iteration,
                measurement=measurement,
                losses=losses,
                learner_metrics=learner_metrics,
                gradient_l2=float(gradients["gradient_l2"]),
                finite_gradients=float(gradients["finite_gradients"]),
                parameter_change=change,
                requested_minimum_batch=train_batch_size,
                actual_env_steps=sum(a["env_steps"] for a in audits),
                actual_agent_steps=sum(a["agent_steps"] for a in audits),
                actual_policy_decisions=sum(
                    sum(a["policy_decisions"].values()) for a in audits
                ),
                complete_episode_count=len(audits),
                episodes=audits,
                episode_accounting=episode_accounting,
                condensed_learner_rows=int(learner_metrics['num_module_steps_trained']),
                repeated_learner_presentations=int(presentations),
                initial_observation_policy_distribution=policy_distribution_audit(
                    algorithm.get_module(SHARED_POLICY_ID),
                    initial_observations,
                    experiment,
                    sample_count=1024,
                    seed=1700 + algorithm.training_iteration,
                ),
            )
            if experiment.is_study:
                algorithm.study_lifetime_state = logger.checkpoint_state()
                policy_decisions_lifetime = algorithm.study_lifetime_state['true_decisions']
                if policy_decisions_lifetime - previous_true_decisions != record['actual_policy_decisions']:
                    write_json(output/'failed_episode_accounting.json',record['episode_accounting'])
                    raise AssertionError('Completed episode evidence and checkpoint lifetime delta disagree')
                accounted_episode_ids.update(a['lifetime_episode_id'] for a in audits)
                algorithm.study_accounting_state = logger.accounting_state()
            else:
                policy_decisions_lifetime += record["actual_policy_decisions"]
            record["policy_decisions_lifetime"] = policy_decisions_lifetime
            record["common_true_decision_milestones_crossed"] = [m for m in range(50000,policy_decisions_lifetime+1,50000)
                if previous_true_decisions < m <= policy_decisions_lifetime]
            record["milestone_budget_semantics"] = "First completed-update checkpoint crossing milestone; exact observed decisions saved, overshoot is not called matched experience"
            probes = np.asarray(
                [o for a in audits for o in a["probe_observations"]], dtype=np.float32
            )
            checkpoint = output / f"checkpoint_{algorithm.training_iteration:04d}"
            record["restore_validation"] = save_checkpoint(
                algorithm, experiment, checkpoint, probes
            )
            metadata_path = checkpoint / "manifest.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.update(
                training_hyperparameters=effective,
                policy_decisions_lifetime=policy_decisions_lifetime,
            )
            write_json(metadata_path, metadata)
            logger.save_identity(checkpoint)
            record["checkpoint"] = str(checkpoint)
            records.append(record)
            write_json(output / "updates.json", records)
            memory_profile.mark("after-checkpoint")
            logger.log_update(record)
            print(
                json.dumps(
                    {key: value for key, value in record.items() if key != "episodes"}
                ),
                flush=True,
            )
        if not (resume and iterations == 0):
            write_json(output/"tranche-stop.json",dict(reason="signal" if stop_requested else "budget_or_update_limit",
                training_iteration=algorithm.training_iteration,policy_decisions_lifetime=policy_decisions_lifetime,
                checkpoint=str(checkpoint) if records else str(resume) if resume else None))
        if resume and iterations == 0:
            from examples.multiagent_imaging.checkpoints import zero_update_resume_evidence
            evidence = zero_update_resume_evidence(algorithm, resume)
            evidence["policy_decisions_lifetime"] = policy_decisions_lifetime
            if logger.run is None or logger.metadata["id"] != evidence["wandb_id"]:
                raise AssertionError("Zero-update gate did not resume the saved online W&B run")
            evidence["wandb"] = dict(logger.metadata, requested_resume="must")
            write_json(output / "zero-update-resume.json", evidence)
        failed = False
    except Exception as error:
        from examples.multiagent_imaging.checkpoint_topology import WorkerTopologyError
        write_json(output / "training-failed.json", dict(passed=False, error_type=type(error).__name__,
            error=str(error)[:2000], worker_topology=error.evidence if isinstance(error,WorkerTopologyError) else None,
            training_iteration=getattr(algorithm, "training_iteration", None)))
        raise
    finally:
        if 'prior_handlers' in locals():
            for signum,handler in prior_handlers.items(): signal.signal(signum,handler)
        if logger is not None:
            logger.finish(failed=failed)
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
        memory_profile.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs" / "smoke.json"
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument(
        "--checkpoint-dir",
        "--output",
        dest="output",
        type=Path,
        default=Path("results/multiagent_imaging/training"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--cpus-per-env-runner", type=int, default=1)
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--gpus-per-learner", type=float, default=0)
    parser.add_argument("--ray-cpus", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--sample-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--ppo-profile",
        choices=["legacy_pilot", "value_repair", "learning_recovery", "amos_best"],
    )
    parser.add_argument(
        "--wandb-mode", choices=["disabled", "offline", "online"], default="disabled"
    )
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--save-initial", action="store_true")
    parser.add_argument("--capture-learner-batch", action="store_true")
    parser.add_argument("--max-true-decisions", type=int)
    parser.add_argument("--max-sampled-true-decisions", type=int)
    parser.add_argument("--max-training-episodes", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args()
    for value in (args.max_true_decisions,args.max_sampled_true_decisions,args.max_training_episodes,args.max_wall_seconds):
        if value is not None and value <= 0: parser.error("Study budgets must be positive")
    if args.iterations < 0 or (args.iterations == 0 and args.resume is None):
        parser.error("Require at least one update, or zero updates with --resume.")
    if args.train_batch_size < 8:
        parser.error("Require a minimum batch of eight.")
    if args.iterations == 0 and not all(
        (args.resume / name).exists() for name in ("manifest.json", "policy.pt", "algorithm")
    ):
        parser.error("Zero-update resume requires an existing complete checkpoint.")
    resources = LauncherResources(
        args.num_env_runners,
        args.cpus_per_env_runner,
        args.num_learners,
        args.gpus_per_learner,
        args.ray_cpus,
        args.torch_threads,
        args.sample_timeout_s,
    )
    train_run(
        MultiAgentImagingConfig.from_json(args.config),
        iterations=args.iterations,
        train_batch_size=args.train_batch_size,
        output=args.output,
        resources=resources,
        resume=args.resume,
        ppo_settings=PPOSettings.from_profile(args.ppo_profile)
        if args.ppo_profile
        else None,
        wandb_mode=args.wandb_mode,
        require_wandb=args.require_wandb,
        save_initial=args.save_initial,
        capture_learner_batch=args.capture_learner_batch,
        max_true_decisions=args.max_true_decisions,
        max_sampled_true_decisions=args.max_sampled_true_decisions,
        max_training_episodes=args.max_training_episodes,
        max_wall_seconds=args.max_wall_seconds,
    )


if __name__ == "__main__":
    main()
