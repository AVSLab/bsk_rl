"""Run one deterministic bounded two-sensor rollout and write metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bsk_rl import NO_ACTION

from examples.multiagent_imaging.config import (
    MultiAgentImagingConfig,
    NON_IMAGING_ACTIONS,
)
from examples.multiagent_imaging.environment import build_environment


def run_rollout(config: MultiAgentImagingConfig) -> dict:
    env = build_environment(config)
    observations, infos = env.reset(seed=config.seed)
    step = 0
    action_counts = {agent: {} for agent in env.possible_agents}
    accumulated_d_ts = {agent: [] for agent in env.possible_agents}
    active_elapsed = {agent: 0.0 for agent in env.possible_agents}
    cumulative_reward = {agent: 0.0 for agent in env.possible_agents}
    decision_count = {agent: 0 for agent in env.possible_agents}
    resource_history = {agent: [] for agent in env.possible_agents}
    sensor_index = {
        sensor.name: index for index, sensor in enumerate(env.sensing_satellites)
    }

    while env.agents:
        actions = {}
        for sensor in env.sensing_satellites:
            if sensor.name not in env.agents:
                continue
            if not sensor.requires_retasking:
                action = NO_ACTION
            elif (
                sensor.dynamics.storage_level_fraction > 0.0
                and decision_count[sensor.name] % 4 == 3
            ):
                action = 1  # downlink; success still requires a ground-station window
            elif (
                config.information_case == "intent_status"
                and not config.perfect_metadata_delivery
                and step > 0
                and decision_count[sensor.name] % 5 == 4
            ):
                action = 3  # finite metadata broadcast
            else:
                action = NON_IMAGING_ACTIONS + (
                    (step + sensor_index[sensor.name]) % config.n_candidates
                )
            actions[sensor.name] = action
            if action != NO_ACTION:
                decision_count[sensor.name] += 1
            label = "continue" if action == NO_ACTION else str(action)
            action_counts[sensor.name][label] = (
                action_counts[sensor.name].get(label, 0) + 1
            )

        observations, reward, terminated, truncated, infos = env.step(actions)
        for agent in env.possible_agents:
            if agent not in infos:
                continue
            active_elapsed[agent] += float(infos[agent]["d_ts"])
            cumulative_reward[agent] += float(reward.get(agent, 0.0))
            if infos[agent]["requires_retasking"]:
                accumulated_d_ts[agent].append(active_elapsed[agent])
                active_elapsed[agent] = 0.0
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

    result = {
        "seed": config.seed,
        "sim_time_s": float(env.simulator.sim_time),
        "pettingzoo_agents": list(env.possible_agents),
        "passive_target_count": len(env.passive_satellites),
        "cumulative_reward": cumulative_reward,
        "action_counts": action_counts,
        "completed_action_d_ts": accumulated_d_ts,
        "resource_history": resource_history,
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
