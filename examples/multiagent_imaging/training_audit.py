"""Episode and update evidence, independent of the reward/policy observations."""

from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.columns import Columns

from bsk_rl.obs.completion_observations import (
    GLOBAL_FEATURES,
    PEER_FEATURES,
    STORAGE_FRACTION_FEATURE,
    TARGET_FEATURES,
)


def frozen_policy_logits(module, observations):
    """Measure deployment logits with dropout off, preserving learner mode."""
    training = module.training
    try:
        module.eval()
        with torch.no_grad():
            return module.forward_inference({Columns.OBS: observations})[
                Columns.ACTION_DIST_INPUTS
            ].cpu()
    finally:
        module.train(training)


def policy_distribution_audit(
    module,
    observations,
    config,
    *,
    sample_count=4096,
    seed=1701,
):
    """Measure categorical probabilities, masks, argmaxes and actual samples.

    Deterministic argmax is not an initialization test: a one-millilogit random
    difference can select the same action everywhere while the distribution is
    otherwise uniform.  This audit therefore records both the exact probability
    vector and empirical categorical samples for every sensing agent.
    """
    batch = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    logits = frozen_policy_logits(module, batch)
    probabilities = torch.softmax(logits, dim=-1)
    valid = logits > -1e8
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    labels = [
        "charge",
        "downlink",
        "desat",
        "broadcast",
        "continue",
        *[f"image_{index}" for index in range(config.n_candidates)],
        *[f"transmit_{index}" for index in range(config.n_peers)],
    ]
    rows = []
    for index, (row_logits, row_probabilities, row_valid) in enumerate(
        zip(logits, probabilities, valid)
    ):
        valid_probabilities = row_probabilities[row_valid]
        valid_count = int(row_valid.sum())
        entropy = float(
            -(valid_probabilities * valid_probabilities.clamp_min(1e-30).log()).sum()
        )
        normalized_entropy = (
            entropy / float(np.log(valid_count)) if valid_count > 1 else 1.0
        )
        counts = torch.bincount(
            torch.multinomial(
                row_probabilities,
                num_samples=sample_count,
                replacement=True,
                generator=generator,
            ),
            minlength=len(labels),
        )
        frequencies = counts.float() / sample_count
        rows.append(
            dict(
                sensor_index=index,
                logits=[float(value) for value in row_logits],
                probabilities=[float(value) for value in row_probabilities],
                valid=[bool(value) for value in row_valid],
                valid_action_labels=[
                    label for label, keep in zip(labels, row_valid) if keep
                ],
                deterministic_argmax=labels[int(row_probabilities.argmax())],
                normalized_entropy=normalized_entropy,
                maximum_to_minimum_valid_probability_ratio=float(
                    valid_probabilities.max() / valid_probabilities.min()
                ),
                empirical_frequencies={
                    label: float(frequency)
                    for label, frequency in zip(labels, frequencies)
                    if frequency > 0
                },
                maximum_empirical_probability_error=float(
                    (frequencies - row_probabilities).abs().max()
                ),
            )
        )
    return dict(sample_count=sample_count, action_labels=labels, rows=rows)


def initial_policy_gate(module, observations, config):
    """Reject biased initialization or a broken real-observation action mask."""
    empty = policy_distribution_audit(module, observations, config)
    with_products = np.asarray(observations, dtype=np.float32).copy()
    with_products[:, STORAGE_FRACTION_FEATURE] = max(
        0.1, 10 * config.empty_storage_mask_fraction
    )
    filled = policy_distribution_audit(module, with_products, config, seed=1702)
    with_products_and_peers = with_products.copy()
    if config.n_peers:
        peer_start = GLOBAL_FEATURES + config.n_candidates * config.target_features
        peer_end = peer_start + config.n_peers * config.peer_features
        peers = with_products_and_peers[:, peer_start:peer_end].reshape(
            len(with_products_and_peers), config.n_peers, config.peer_features
        )
        peers[:, :, -1] = 1.0
        if config.is_study: peers[:, :, 16] = 1.0
    filled_and_connected = policy_distribution_audit(
        module, with_products_and_peers, config, seed=1703
    )
    failures = []
    for label, audit in (
        ("empty", empty),
        ("with_products", filled),
        ("with_products_and_peers", filled_and_connected),
    ):
        for row in audit["rows"]:
            if row["normalized_entropy"] < 0.999:
                failures.append(
                    f"{label} sensor {row['sensor_index']} normalized entropy "
                    f"{row['normalized_entropy']:.6f} < 0.999"
                )
            if row["maximum_to_minimum_valid_probability_ratio"] > 1.10:
                failures.append(
                    f"{label} sensor {row['sensor_index']} max/min probability "
                    f"ratio {row['maximum_to_minimum_valid_probability_ratio']:.6f} > 1.10"
                )
            if row["maximum_empirical_probability_error"] > 0.03:
                failures.append(
                    f"{label} sensor {row['sensor_index']} categorical samples "
                    "do not match its logits"
                )
    target_slice = slice(
        GLOBAL_FEATURES,
        GLOBAL_FEATURES + config.n_candidates * config.target_features,
    )
    targets = np.asarray(observations)[:, target_slice].reshape(
        -1, config.n_candidates, config.target_features
    )
    if not np.all(targets[:, :, 16] > 0.5):
        failures.append("Not all ten real initial candidate slots were valid.")
    if any(row["valid"][1] for row in empty["rows"]):
        failures.append("Empty-storage downlink remained valid.")
    if not all(row["valid"][1] for row in filled["rows"]):
        failures.append("Downlink did not become valid with a local physical product.")
    if config.n_peers and config.information_case != "geometry_only" and not all(
        all(row["valid"][-config.n_peers :]) for row in filled_and_connected["rows"]
    ):
        failures.append("Eligible directed peer actions did not become valid.")
    result = dict(
        passed=not failures,
        failures=failures,
        empty=empty,
        filled=filled,
        filled_and_connected=filled_and_connected,
    )
    if failures:
        raise AssertionError("Initial policy gate failed: " + "; ".join(failures))
    return result


class CompletionAuditCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, env_runner, **kwargs):
        import os
        from examples.multiagent_imaging.sampling_budget import worker_quota
        runners=int(os.environ.get("BSK_STUDY_NUM_ENV_RUNNERS", "1"))
        declared=os.environ.get("BSK_STUDY_MAX_EPISODES")
        cap=worker_quota(int(declared) if declared else None, env_runner.worker_index, runners)
        env=env_runner.env.unwrapped.par_env
        # Ray eagerly resets after a completed mission. A reset with no learned
        # action is not a newly sampled mission; enforce before its first action.
        env.study_remaining_episode_budget=cap-getattr(env_runner,"study_sampled_episodes",0) if cap is not None else None
        decision_cap=os.environ.get("BSK_STUDY_MAX_TRUE_DECISIONS")
        quota=worker_quota(int(decision_cap) if decision_cap else None, env_runner.worker_index, runners)
        env.study_remaining_true_budget=quota-getattr(env_runner,"study_new_true_decisions",0) if quota is not None else None
        env_runner.completion_resource_history = []
        from examples.multiagent_imaging.baseline_monte_carlo import initial_conditions, digest
        env_runner.completion_initial_hash = digest(initial_conditions(env_runner.env.unwrapped.par_env))
        self._record_resources(env_runner)
        self._publish_progress(env_runner)

    def on_episode_step(self, *, env_runner, **kwargs):
        self._record_resources(env_runner)
        self._publish_progress(env_runner)

    @staticmethod
    def _publish_progress(runner):
        """Worker writes local evaluation telemetry; only driver publishes W&B.

        Current-episode counters reset with the environment. They never enter
        lifetime completed-episode aggregation or policy input dictionaries.
        """
        import os,json
        from pathlib import Path
        spool=os.environ.get("BSK_STUDY_EPISODE_SPOOL")
        if not spool or runner.worker_index==0: return
        env=runner.env.unwrapped.par_env
        if not any(getattr(s,"completion_study",False) for s in env.sensing_satellites): return
        row=dict(worker_index=runner.worker_index,seed=env.seed,sim_time_s=float(env.simulator.sim_time),
            next_episode_index=env._episode_seed_index,reward_by_agent=dict(env.episode_reward_by_agent),
            true_decisions=dict(env.decision_counts),agents={})
        for sensor in env.sensing_satellites:
            row["agents"][sensor.name]=dict(storage_fraction=float(sensor.dynamics.storage_level_fraction),
                battery_fraction=float(sensor.dynamics.battery_charge_fraction),
                reward=env.episode_reward_by_agent[sensor.name],
                true_decisions=env.decision_counts[sensor.name],
                actions={label:env.action_selection_counts[sensor.name].get(label,0) for label in sensor.action_description})
        folder=Path(spool)/"progress";folder.mkdir(parents=True,exist_ok=True)
        path=folder/f"worker-{runner.worker_index}.json"
        temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(row,allow_nan=False)+'\n');temporary.replace(path)

    @staticmethod
    def _record_resources(runner):
        """Event-boundary resource history; do not add peer data to policy inputs."""
        env = runner.env.unwrapped.par_env
        history = getattr(runner, "completion_resource_history", [])
        for sensor in env.sensing_satellites:
            history.append(
                dict(
                    sensor=sensor.name,
                    time_s=float(env.simulator.sim_time),
                    alive=bool(sensor.is_alive()),
                    battery_fraction=float(sensor.dynamics.battery_charge_fraction),
                    storage_fraction=float(sensor.dynamics.storage_level_fraction),
                    wheel_speed_fraction=np.asarray(
                        sensor.dynamics.wheel_speeds_fraction
                    ).tolist(),
                )
            )
        runner.completion_resource_history = history

    def on_episode_end(self, *, env_runner, episode, **kwargs):
        env = env_runner.env.unwrapped.par_env
        resources, products = {}, {}
        for sensor in env.sensing_satellites:
            values = np.asarray(
                [
                    sensor.dynamics.battery_charge_fraction,
                    sensor.dynamics.storage_level_fraction,
                    *sensor.dynamics.wheel_speeds_fraction,
                ]
            )
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite spacecraft resources.")
            resources[sensor.name] = values.tolist()
            products[sensor.name] = len(sensor.data_store.products)
            for product in sensor.data_store.products:
                if (
                    product.storage_owner != sensor.name
                    or product.source_sensor != sensor.name
                ):
                    raise AssertionError(
                        "Completion metadata transferred physical product ownership."
                    )
        probes = [sensor.get_obs().tolist() for sensor in env.sensing_satellites]
        captures = list(env.rewarder._team_accounting.capture_attempts)
        services = list(env.rewarder.service_entries)
        from examples.multiagent_imaging.baseline_monte_carlo import (
            coverage_metrics,
            duplicate_product_metrics,
        )
        from examples.multiagent_imaging.evaluate import _useful_revisits

        coverage = coverage_metrics(
            [target.rso_target.id for target in env.passive_satellites],
            captures,
            services,
            env.rewarder.quality_threshold,
            sensor_names=[sensor.name for sensor in env.sensing_satellites],
        )
        duplicate_products = duplicate_product_metrics(
            captures,
            services,
            env.rewarder.quality_threshold,
            float(env.simulator.sim_time),
        )
        useful_revisits = _useful_revisits(
            captures, env.rewarder.quality_threshold, env.rewarder.reimage_cooldown_s
        )
        acquisition = (1.0 - env.rewarder.alpha) * sum(
            metrics["acquisition_credit"]
            for metrics in env.rewarder.per_sensor_metrics.values()
        )
        delivery = env.rewarder.alpha * sum(
            metrics["delivery_credit"]
            for metrics in env.rewarder.per_sensor_metrics.values()
        )
        communication_adjustment = -env.communication_cost_per_s * sum(
            env.communication_time.values()
        )
        completed_holds = [
            entry
            for entry in env.communicator.transmission_history
            if entry.get("outcome") == "hold_complete"
        ]
        total_reward = float(episode.get_return())
        summary = dict(
            episode_id=str(episode.id_),
            initial_conditions_sha256=getattr(env_runner, "completion_initial_hash", None),
            episode_seed=int(env.seed),
            reward_by_agent=dict(env.episode_reward_by_agent),
            reward_events=list(env.reward_events),
            decision_records=list(env.decision_records),
            radio_power_intervals=(env.communicator.radio_audit(float(env.simulator.sim_time)) if hasattr(env.communicator,"radio_audit") else []),
            agent_live_until_s={s.name:env.agent_live_until_s.get(s.name,float(env.simulator.sim_time)) for s in env.sensing_satellites},
            first_receipts={s.name:dict(s.data_store.catalog.received_at) for s in env.sensing_satellites},
            terminated=episode.is_terminated,
            truncated=episode.is_truncated,
            alive_sensors={s.name: bool(s.is_alive()) for s in env.sensing_satellites},
            simulated_seconds=float(env.simulator.sim_time),
            env_steps=len(episode),
            agent_steps=episode.agent_steps(),
            policy_decisions=dict(env.decision_counts),
            action_selection_counts={
                sensor: dict(counts)
                for sensor, counts in env.action_selection_counts.items()
            },
            action_labels={s.name:list(getattr(s,"action_description",env.action_selection_counts[s.name])) for s in env.sensing_satellites},
            resources=resources,
            resource_history=getattr(env_runner, "completion_resource_history", []),
            products=products,
            capture_products=[asdict(product) for product in captures],
            delivered_products=[asdict(entry) for entry in services],
            constellation_reward=total_reward,
            cooperative_tasking=(env.rewarder.task_summary() if hasattr(env.rewarder, "task_summary") else None),
            reward_decomposition=dict(
                acquisition_90_percent_component=acquisition,
                ground_delivery_10_percent_component=delivery,
                communication_time_adjustment=communication_adjustment,
                other_operational_penalties_or_adjustments=total_reward
                - acquisition
                - delivery
                - communication_adjustment,
            ),
            team_summary=dict(env.rewarder.team_summary),
            coverage=coverage,
            product_duplicates=duplicate_products,
            useful_post_cooldown_revisits=dict(
                count=len(useful_revisits), record_ids=useful_revisits
            ),
            coordination=env.coordination_metrics(),
            packets=list(env.communicator.delivery_history),
            transmissions=list(env.communicator.transmission_history),
            communication_summary=dict(
                radio_occupancy_s=sum(env.communication_time.values()),
                recipient_selection_counts=dict(
                    Counter(entry["receiver"] for entry in completed_holds)
                ),
                peer_slot_selection_counts=dict(
                    Counter(str(entry["peer_slot"]) for entry in completed_holds)
                ),
                payload_records_attempted=sum(
                    entry.get("records", 0) for entry in completed_holds
                ),
                payload_bytes_attempted=sum(
                    entry.get("payload_bytes", 0) for entry in completed_holds
                ),
                packet_outcome_counts=dict(
                    Counter(
                        packet["outcome"]
                        for packet in env.communicator.delivery_history
                    )
                ),
            ),
            acknowledged_versions_by_link={
                f"{sender}->{receiver}": dict(sorted(versions.items()))
                for (sender, receiver), versions in sorted(
                    env.communicator.acknowledged.items()
                )
            },
            target_regime_counts=dict(
                Counter(
                    target.target_population_regime for target in env.passive_satellites
                )
            ),
            reimage_cooldown_s=float(env.rewarder.reimage_cooldown_s),
            probe_observations=probes,
            transport_ack_revision_pairs=sorted({(p["sender"],p["receiver"],record_id,int(version))
                for p in env.communicator.delivery_history if p["outcome"]=="accepted"
                for record_id,version in p.get("record_versions",{}).items()}),
            next_episode_index=env._episode_seed_index,
            worker_index=env_runner.worker_index,
        )
        from examples.multiagent_imaging.study_metrics import mission_metrics
        summary["mission_metrics"] = mission_metrics(summary, env.time_limit)
        import os
        summary["source_release_id"]=os.environ.get("BSK_STUDY_RELEASE_ID","local-offline-fixture")
        summary["lifetime_episode_id"] = f"{summary['source_release_id']}:{env.communicator.information_case}:{env.default_seed}:{env_runner.worker_index}:{env._episode_seed_index}"
        # Shared scratch spool is append-only evidence, not an observer input.
        # The driver alone reads it and calls W&B, including during long sampling.
        import os, hashlib
        from pathlib import Path
        spool = os.environ.get("BSK_STUDY_EPISODE_SPOOL")
        if spool:
            from examples.multiagent_imaging.readiness import write_json
            folder=Path(spool);folder.mkdir(parents=True,exist_ok=True)
            name=hashlib.sha256(summary["lifetime_episode_id"].encode()).hexdigest()
            target=folder/(name+".json")
            if target.exists():
                import json
                if json.loads(target.read_text()) != summary: raise ValueError("Conflicting episode spool replay")
            else:
                temporary=folder/(name+".tmp")
                write_json(temporary,summary); temporary.replace(target)
        import hashlib,json
        callback_state=getattr(env_runner,"study_audit_state",dict(episode_hashes={},true_decisions=0,reward=0.))
        metric_hash=hashlib.sha256(json.dumps(summary["mission_metrics"],sort_keys=True,allow_nan=False).encode()).hexdigest()
        previous=callback_state["episode_hashes"].get(summary["lifetime_episode_id"])
        if previous:
            if previous != metric_hash: raise ValueError("Conflicting worker callback replay")
            return
        callback_state["episode_hashes"][summary["lifetime_episode_id"]]=metric_hash
        callback_state["true_decisions"]+=sum(summary["policy_decisions"].values())
        callback_state["reward"]+=summary["constellation_reward"]
        env_runner.study_audit_state=callback_state
        if not hasattr(env_runner, "completion_audits"):
            env_runner.completion_audits = []
        env_runner.completion_audits.append(summary)
        env_runner.study_new_true_decisions=getattr(env_runner,"study_new_true_decisions",0)+sum(summary["policy_decisions"].values())
        if sum(summary["policy_decisions"].values()):
            env_runner.study_sampled_episodes=getattr(env_runner,"study_sampled_episodes",0)+1


def take_audits(runner):
    audits = getattr(runner, "completion_audits", [])
    runner.completion_audits = []
    return audits


def parameter_change(before, after):
    squared, count, largest = 0.0, 0, 0.0
    by_branch = {"actor": 0.0, "critic": 0.0}
    for key in before:
        delta = np.asarray(after[key]) - np.asarray(before[key])
        if not np.isfinite(delta).all():
            raise FloatingPointError("Nonfinite policy parameter update.")
        squared += float(np.square(delta).sum())
        by_branch["critic" if key.startswith("vf.") else "actor"] += float(
            np.square(delta).sum()
        )
        largest = max(largest, float(np.max(np.abs(delta))))
        count += int(np.count_nonzero(delta))
    if not count:
        raise AssertionError("PPO returned without changing any policy weights.")
    return dict(
        l2=float(np.sqrt(squared)),
        max_absolute=largest,
        changed_elements=count,
        actor_l2=float(np.sqrt(by_branch["actor"])),
        critic_l2=float(np.sqrt(by_branch["critic"])),
    )
