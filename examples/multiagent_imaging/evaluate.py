"""Run one deterministic bounded two-sensor rollout and write metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bsk_rl import NO_ACTION
from bsk_rl.comm.teammate_state import current_target_id

from examples.multiagent_imaging.config import (
    MultiAgentImagingConfig,
    GLOBAL_FEATURES,
    NON_IMAGING_ACTIONS,
    TARGET_FEATURES,
)
from examples.multiagent_imaging.environment import build_environment


def _target_action(observation: np.ndarray, config: MultiAgentImagingConfig) -> int:
    """Apply one shared deterministic target-score rule to every sensing agent."""
    target_features = np.asarray(observation[GLOBAL_FEATURES:], dtype=float).reshape(
        config.n_candidates, TARGET_FEATURES
    )
    priorities = target_features[:, 0]
    cooldown = target_features[:, 10]
    pending = target_features[:, 11]
    teammate_intent = target_features[:, 12]
    unavailable = (cooldown > 0.0) | (pending > 0.0) | (teammate_intent > 0.0)
    scores = priorities - 100.0 * unavailable.astype(float)
    return NON_IMAGING_ACTIONS + int(np.argmax(scores))


def _shared_knowledge_blocks_target(sensor, target_id: int, sim_time: float) -> bool:
    """Report whether available teammate metadata would suppress one private target."""
    case = getattr(sensor, "information_case", "independent")
    if case == "centralized_information":
        snapshot = sensor.centralized_information_view.target_snapshot(target_id)
        return bool(
            snapshot["pending_anywhere"] or float(snapshot["cooldown_until"]) > sim_time
        )
    if case == "intent_status":
        return not sensor.local_catalog.is_eligible(target_id, sim_time)
    return False


def run_rollout(config: MultiAgentImagingConfig) -> dict:
    """Run one matched deterministic shared-policy rollout with diagnostics."""
    env = build_environment(config)
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
    accumulated_d_ts = {agent: [] for agent in env.possible_agents}
    active_elapsed = {agent: 0.0 for agent in env.possible_agents}
    active_action = {agent: None for agent in env.possible_agents}
    action_time_s = {agent: {} for agent in env.possible_agents}
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
    intent_conflict_events = 0
    intent_conflict_time_s = 0.0
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
                for target_id in sensor.local_catalog.targets:
                    if not sensor.local_catalog.is_privately_eligible(
                        target_id, sim_time
                    ):
                        omission["target_samples_omitted_by_local_knowledge"] += 1
                        omission["local_omitted_target_ids"].add(target_id)
                    elif _shared_knowledge_blocks_target(sensor, target_id, sim_time):
                        omission["target_samples_omitted_by_shared_knowledge"] += 1
                        omission["shared_omitted_target_ids"].add(target_id)

                if (
                    sensor.dynamics.storage_level_fraction > 0.0
                    and decision_count[sensor.name] % 4 == 3
                ):
                    action = 1  # success still requires a ground-station window
                elif (
                    config.information_case == "intent_status"
                    and not config.perfect_metadata_delivery
                    and step > 0
                    and decision_count[sensor.name] % 5 == 4
                ):
                    action = 3  # finite metadata broadcast
                else:
                    action = _target_action(observations[sensor.name], config)
            actions[sensor.name] = action
            if action != NO_ACTION:
                decision_count[sensor.name] += 1
                active_action[sensor.name] = sensor.action_description[action]
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
            active_elapsed[agent] += float(infos[agent]["d_ts"])
            cumulative_reward[agent] += float(reward.get(agent, 0.0))
            reward_history[agent].append(
                {
                    "time_s": current_sim_time,
                    "reward": float(reward.get(agent, 0.0)),
                    "cumulative_reward": cumulative_reward[agent],
                }
            )
            if infos[agent]["requires_retasking"]:
                accumulated_d_ts[agent].append(active_elapsed[agent])
                label = active_action[agent] or "unknown"
                action_time_s[agent][label] = (
                    action_time_s[agent].get(label, 0.0) + active_elapsed[agent]
                )
                active_elapsed[agent] = 0.0
                active_action[agent] = None

        targets_by_sensor = {
            sensor.name: current_target_id(sensor) for sensor in env.sensing_satellites
        }
        target_counts: dict[int, int] = {}
        for target_id in targets_by_sensor.values():
            if target_id is not None:
                target_counts[target_id] = target_counts.get(target_id, 0) + 1
        conflicts = {
            target_id for target_id, count in target_counts.items() if count > 1
        }
        intent_conflict_events += len(conflicts - active_conflicts)
        if conflicts:
            intent_conflict_time_s += global_dt
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

    for agent, elapsed in active_elapsed.items():
        if elapsed > 0.0:
            label = active_action[agent] or "incomplete_unknown"
            action_time_s[agent][label] = action_time_s[agent].get(label, 0.0) + elapsed

    message_dispositions = {}
    message_ages_s = {agent: [] for agent in env.possible_agents}
    expired_latest_messages = {agent: 0 for agent in env.possible_agents}
    for sensor in env.sensing_satellites:
        inbox = sensor.intent_status_inbox
        for _, disposition in inbox.history:
            key = disposition.value
            message_dispositions[key] = message_dispositions.get(key, 0) + 1
        for message in inbox.latest_intent_by_sender.values():
            age = max(0.0, float(env.simulator.sim_time) - message.creation_time)
            message_ages_s[sensor.name].append(age)
            if message.expiry_time < float(env.simulator.sim_time):
                expired_latest_messages[sensor.name] += 1

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
                if "broadcast" in label.lower()
            )
            for agent, times in action_time_s.items()
        },
        "reward_history": reward_history,
        "resource_history": resource_history,
        "intent_conflicts": {
            "event_count": intent_conflict_events,
            "time_s": intent_conflict_time_s,
        },
        "message_diagnostics": {
            "disposition_counts": message_dispositions,
            "latest_message_ages_s": message_ages_s,
            "expired_latest_message_count": expired_latest_messages,
        },
        "target_omission_diagnostics": omission_output,
        "per_sensor_metrics": env.rewarder.per_sensor_metrics,
        "team_summary": env.rewarder.team_summary,
        "team_service_ledger": [
            {
                "record_id": entry.product.record_id,
                "source_sensor": entry.product.source_sensor,
                "target_id": entry.product.target_id,
                "capture_time": entry.product.capture_time,
                "delivery_time": entry.product.delivery_time,
                "quality": entry.product.quality,
                "storage_owner": entry.product.storage_owner,
                "unique_service": entry.unique_service,
                "successful_duplicate": entry.successful_duplicate,
                "credited_value": entry.credited_value,
            }
            for entry in env.rewarder.team_ledger.entries
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
                for target_id, state in sensor.local_catalog.targets.items()
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
                for product in sensor.physical_product_store.products
            ]
            for sensor in env.sensing_satellites
        },
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "smoke.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = MultiAgentImagingConfig.from_json(args.config)
    if args.seed is not None:
        config = MultiAgentImagingConfig(**{**config.to_dict(), "seed": args.seed})
    result = run_rollout(config)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(args.output.resolve())


if __name__ == "__main__":
    main()
