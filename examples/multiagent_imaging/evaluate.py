"""Run one deterministic bounded two-sensor rollout and write metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

from bsk_rl import NO_ACTION

from examples.multiagent_imaging.config import (
    MultiAgentImagingConfig,
    GLOBAL_FEATURES,
    NON_IMAGING_ACTIONS,
    TARGET_FEATURES,
    OBSERVATION_VERSION,
)
from examples.multiagent_imaging.environment import build_environment


def _target_action(
    observation: np.ndarray,
    config: MultiAgentImagingConfig,
    controller: str = "priority",
) -> int:
    """Apply one shared deterministic target rule to every sensing agent."""
    target_features = np.asarray(
        observation[
            GLOBAL_FEATURES : GLOBAL_FEATURES + config.n_candidates * TARGET_FEATURES
        ],
        dtype=float,
    ).reshape(config.n_candidates, TARGET_FEATURES)
    priorities = target_features[:, 0]
    unavailable = target_features[:, 16] < 0.5
    if np.all(unavailable):
        return 0
    if controller == "priority":
        scores = np.where(unavailable, -np.inf, priorities)
        return NON_IMAGING_ACTIONS + int(np.argmax(scores))
    if controller == "closest_angle":
        angles = target_features[:, 7]
        scores = np.where(unavailable | ~np.isfinite(angles), np.inf, angles)
        if np.all(np.isinf(scores)):
            return 0  # charge until a target becomes eligible
        return NON_IMAGING_ACTIONS + int(np.argmin(scores))
    raise ValueError(f"Unknown deterministic controller: {controller!r}")


def _shared_knowledge_blocks_target(sensor, target_id: int, sim_time: float) -> bool:
    """Report whether available teammate metadata would suppress one private target."""
    return not sensor.data_store.catalog.is_eligible(target_id, sim_time)


def run_rollout(
    config: MultiAgentImagingConfig,
    *,
    controller: str = "priority",
    policy=None,
    vizard_dir: str | None = None,
    vizard_settings: dict | None = None,
) -> dict:
    """Run one matched deterministic shared-policy rollout with diagnostics."""
    env = build_environment(
        config,
        vizard_dir=vizard_dir,
        vizard_settings=vizard_settings,
    )
    observations, infos = env.reset(seed=config.seed)
    initial_conditions = {
        "sensors": {
            sensor.name: {
                "position_N_m": list(map(float, sensor.dynamics.r_BN_N)),
                "velocity_N_m_s": list(map(float, sensor.dynamics.v_BN_N)),
            }
            for sensor in env.sensing_satellites
        },
        "targets": {
            target.name: {
                "position_N_m": list(map(float, target.dynamics.r_BN_N)),
                "velocity_N_m_s": list(map(float, target.dynamics.v_BN_N)),
                "priority": float(target.rso_target.priority),
                "target_id": int(target.rso_target.id),
            }
            for target in env.passive_satellites
        },
    }
    step = 0
    action_counts = {agent: {} for agent in env.possible_agents}
    cumulative_reward = {agent: 0.0 for agent in env.possible_agents}
    reward_history = {agent: [] for agent in env.possible_agents}
    decision_count = {agent: 0 for agent in env.possible_agents}
    resource_history = {agent: [] for agent in env.possible_agents}
    omission_diagnostics = {
        agent: {
            "decision_samples": 0,
            "target_samples_omitted_by_local_knowledge": 0,
            "target_samples_omitted_by_shared_knowledge": 0,
            "local_omitted_target_ids": set(),
            "shared_omitted_target_ids": set(),
        }
        for agent in env.possible_agents
    }
    concurrent_target_conflict_events = 0
    concurrent_target_conflict_time_s = 0.0
    active_conflicts: set[int] = set()
    prior_sim_time = float(env.simulator.sim_time)

    while env.agents:
        actions = {}
        for sensor in env.sensing_satellites:
            if sensor.name not in env.agents:
                continue
            if not sensor.requires_retasking:
                action = NO_ACTION
            else:
                sim_time = float(env.simulator.sim_time)
                omission = omission_diagnostics[sensor.name]
                omission["decision_samples"] += 1
                for target_id in sensor.data_store.catalog.targets:
                    if not sensor.data_store.catalog.is_privately_eligible(
                        target_id, sim_time
                    ):
                        omission["target_samples_omitted_by_local_knowledge"] += 1
                        omission["local_omitted_target_ids"].add(target_id)
                    elif _shared_knowledge_blocks_target(sensor, target_id, sim_time):
                        omission["target_samples_omitted_by_shared_knowledge"] += 1
                        omission["shared_omitted_target_ids"].add(target_id)

                task = getattr(sensor, "completion_task", None)
                if policy is not None:
                    action = int(policy(observations[sensor.name]))
                elif sensor.dynamics.battery_charge_fraction < 0.3:
                    action = 0
                elif max(abs(np.asarray(sensor.dynamics.wheel_speeds_fraction))) > 0.7:
                    action = 2
                elif (
                    task
                    and not task.finished
                    and task.mode in {"broadcast", "downlink"}
                ):
                    action = 4  # Keep finite operations running across peer events.
                elif (
                    sensor.dynamics.storage_level_fraction > 0.0
                    and decision_count[sensor.name] % 4 == 3
                ):
                    action = 1  # success still requires a ground-station window
                elif (
                    config.information_case == "completion"
                    and env.communicator.backlog(sensor.name) > 0
                    and step > 0
                    and decision_count[sensor.name] % 5 == 4
                ):
                    if config.communication_mode == "directed":
                        from bsk_rl.obs.completion_observations import peer_snapshot

                        peers = peer_snapshot(sensor).peers
                        valid = [i for i, p in enumerate(peers) if p is not None]
                        action = (
                            NON_IMAGING_ACTIONS + config.n_candidates + valid[0]
                            if valid
                            else _target_action(
                                observations[sensor.name], config, controller
                            )
                        )
                    else:
                        action = 3  # Configurable finite omnidirectional baseline.
                else:
                    action = _target_action(
                        observations[sensor.name], config, controller=controller
                    )
            actions[sensor.name] = action
            if action != NO_ACTION:
                decision_count[sensor.name] += 1
            label = (
                "continue"
                if action == NO_ACTION
                else str(sensor.action_description[action])
            )
            action_counts[sensor.name][label] = (
                action_counts[sensor.name].get(label, 0) + 1
            )

        observations, reward, terminated, truncated, infos = env.step(actions)
        current_sim_time = float(env.simulator.sim_time)
        global_dt = current_sim_time - prior_sim_time
        prior_sim_time = current_sim_time
        for agent in env.possible_agents:
            if agent not in infos:
                continue
            cumulative_reward[agent] += float(reward.get(agent, 0.0))
            reward_history[agent].append(
                {
                    "time_s": current_sim_time,
                    "reward": float(reward.get(agent, 0.0)),
                    "cumulative_reward": cumulative_reward[agent],
                }
            )
        targets_by_sensor = {
            sensor.name: (
                sensor.completion_task.target_id
                if sensor.completion_task and sensor.completion_task.mode == "image"
                else None
            )
            for sensor in env.sensing_satellites
        }
        target_counts: dict[int, int] = {}
        for target_id in targets_by_sensor.values():
            if target_id is not None:
                target_counts[target_id] = target_counts.get(target_id, 0) + 1
        conflicts = {
            target_id for target_id, count in target_counts.items() if count > 1
        }
        concurrent_target_conflict_events += len(conflicts - active_conflicts)
        if conflicts:
            concurrent_target_conflict_time_s += global_dt
        active_conflicts = conflicts
        for sensor in env.sensing_satellites:
            resource_history[sensor.name].append(
                {
                    "time_s": float(env.simulator.sim_time),
                    "battery_fraction": float(sensor.dynamics.battery_charge_fraction),
                    "storage_fraction": float(sensor.dynamics.storage_level_fraction),
                    "wheel_speed_fraction": list(
                        map(float, sensor.dynamics.wheel_speeds_fraction)
                    ),
                }
            )
        step += 1
        if all(terminated.values()) or all(truncated.values()):
            break

    # Physical operation durations come from task lifetimes. In continuous mode
    # a new policy decision or ContinueTask does not start a new physical operation.
    coordination = env.coordination_metrics()
    action_time_s = {agent: {} for agent in env.possible_agents}
    accumulated_d_ts = {agent: [] for agent in env.possible_agents}
    for task in coordination["task_history"]:
        duration = task["end"] - task["start"]
        times = action_time_s[task["sensor"]]
        times[task["mode"]] = times.get(task["mode"], 0.0) + duration
        if task["reason"] == "action_end":
            accumulated_d_ts[task["sensor"]].append(duration)

    message_dispositions = {}
    message_ages_s = {agent: [] for agent in env.possible_agents}
    for entry in env.communicator.delivery_history:
        key = entry["outcome"]
        message_dispositions[key] = message_dispositions.get(key, 0) + 1
        message_ages_s[entry["receiver"]].append(
            entry["received_at"] - entry["sent_at"]
        )

    omission_output = {}
    for agent, values in omission_diagnostics.items():
        omission_output[agent] = {
            **{
                key: value
                for key, value in values.items()
                if not key.endswith("_target_ids")
            },
            "local_omitted_target_ids": sorted(values["local_omitted_target_ids"]),
            "shared_omitted_target_ids": sorted(values["shared_omitted_target_ids"]),
        }

    result = {
        "seed": config.seed,
        "controller": controller,
        "config": config.to_dict(),
        "initial_conditions": initial_conditions,
        "sim_time_s": float(env.simulator.sim_time),
        "pettingzoo_agents": list(env.possible_agents),
        "passive_target_count": len(env.passive_satellites),
        "cumulative_reward": cumulative_reward,
        "action_counts": action_counts,
        "completed_action_d_ts": accumulated_d_ts,
        "action_time_s": action_time_s,
        "broadcast_time_s": {
            agent: sum(
                duration
                for label, duration in times.items()
                if "broadcast" in label.lower() or "transmit" in label.lower()
            )
            for agent, times in action_time_s.items()
        },
        "reward_history": reward_history,
        "resource_history": resource_history,
        "concurrent_target_conflicts": {
            "event_count": concurrent_target_conflict_events,
            "time_s": concurrent_target_conflict_time_s,
        },
        "message_diagnostics": {
            "disposition_counts": message_dispositions,
            "packet_latency_s": message_ages_s,
            "delivery_history": env.communicator.delivery_history,
            "transmission_history": env.communicator.transmission_history,
        },
        "target_omission_diagnostics": omission_output,
        "per_sensor_metrics": env.rewarder.per_sensor_metrics,
        "team_summary": env.rewarder.team_summary,
        "coordination": coordination,
        "observation_version": OBSERVATION_VERSION,
        "completion_records": {
            s.name: [asdict(r) for r in s.data_store.catalog.records.values()]
            for s in env.sensing_satellites
        },
        "catalog_receipts": {
            s.name: dict(
                first_receipt=s.data_store.catalog.received_at,
                version_receipts=s.data_store.catalog.version_received_at,
            )
            for s in env.sensing_satellites
        },
        "team_service_history": [
            {
                "record_id": entry.product.record_id,
                "source_sensor": entry.product.source_sensor,
                "target_id": entry.product.target_id,
                "capture_time": entry.product.capture_time,
                "completion_time": entry.product.completion_time,
                "request_epoch": entry.product.request_epoch,
                "delivery_time": entry.product.delivery_time,
                "quality": entry.product.quality,
                "storage_owner": entry.product.storage_owner,
                "unique_service": entry.unique_service,
                "successful_duplicate": entry.successful_duplicate,
                "credited_value": entry.credited_value,
            }
            for entry in env.rewarder.service_entries
        ],
        "local_catalogs": {
            sensor.name: {
                str(target_id): {
                    "latest_acquisition_time": state.latest_acquisition_time,
                    "latest_delivery_time": state.latest_delivery_time,
                    "cooldown_until": state.cooldown_until,
                    "pending_record_ids": list(state.pending_record_ids),
                    "remote_pending_sources": list(state.remote_pending_sources),
                    "last_update_time": state.last_update_time,
                    "last_update_source": state.last_update_source,
                }
                for target_id, state in sensor.data_store.catalog.targets.items()
            }
            for sensor in env.sensing_satellites
        },
        "onboard_products": {
            sensor.name: [
                {
                    "record_id": product.record_id,
                    "source_sensor": product.source_sensor,
                    "target_id": product.target_id,
                    "capture_time": product.capture_time,
                    "delivery_time": product.delivery_time,
                    "quality": product.quality,
                    "storage_owner": product.storage_owner,
                }
                for product in sensor.data_store.products
            ]
            for sensor in env.sensing_satellites
        },
    }
    env.close()

    # Unknown legacy summary timestamps use infinities internally. Export null,
    # preserving valid standards-compliant JSON rather than Infinity/NaN literals.
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    return json_safe(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "smoke.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plots-dir", type=Path)
    args = parser.parse_args()
    config = MultiAgentImagingConfig.from_json(args.config)
    if args.seed is not None:
        config = MultiAgentImagingConfig(**{**config.to_dict(), "seed": args.seed})
    policy = None
    if args.checkpoint:
        from examples.multiagent_imaging.checkpoints import load_policy, policy_callable

        module, restore_evidence = load_policy(args.checkpoint, config)
        policy = policy_callable(module)
    result = run_rollout(config, policy=policy)
    if args.checkpoint:
        result["controller"] = "restored-target-set-attention"
        result["checkpoint"] = str(args.checkpoint.resolve())
        result["restore_validation"] = restore_evidence
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(args.output.resolve())
    if args.plots_dir is not None:
        from examples.multiagent_imaging.plot_evaluation import plot_evaluation

        for path in plot_evaluation(result, args.plots_dir):
            print(path.resolve())


if __name__ == "__main__":
    main()
