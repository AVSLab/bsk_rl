"""Replay saved native decisions to independently repair physical audit history.

No policy sampling, learner, gradient, or PPO update. Preserve the original
training journal and compare every primitive API reward and physical capture.
"""

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from bsk_rl import NO_ACTION
from bsk_rl.obs.completion_observations import private_action_mask
from examples.multiagent_imaging.baseline_monte_carlo import (
    coverage_metrics,
    duplicate_product_metrics,
    initial_conditions,
    digest,
)
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.environment import build_environment
from examples.multiagent_imaging.memory_profile import MemoryProfile
from examples.multiagent_imaging.readiness import write_json
from examples.multiagent_imaging.study_metrics import mission_metrics
from examples.multiagent_imaging.update_evidence import UpdateRecords


def replay(training, iteration, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    row = next(r for r in UpdateRecords(training) if r["iteration"] == iteration)
    assert len(row["episodes"]) == 1
    original = row["episodes"][0]
    config = MultiAgentImagingConfig.from_json(
        Path(__file__).with_name("configs") / "cooperative_v1_shared_catalog_tx.json"
    )
    config = replace(config, seed=original["episode_seed"])
    expected = {
        (d["sensor"], round(d["time_s"], 6)): d for d in original["decision_records"]
    }
    assert len(expected) == len(original["decision_records"])
    started, cpu_started = time.monotonic(), time.process_time()
    memory = MemoryProfile(output, interval_s=10)
    memory.start()
    env = None
    result = dict(
        passed=False,
        new_ppo_updates=0,
        kind="Saved-action native replay; not a new training or policy-evaluation episode",
        iteration=iteration,
    )
    try:
        env = build_environment(config)
        env.reset(seed=config.seed)
        assert digest(initial_conditions(env)) == original["initial_conditions_sha256"]
        consumed = set()
        while env.simulator.sim_time < config.episode_duration_s - 1e-6:
            if time.monotonic() - started > 1800:
                raise RuntimeError("Replay wall cap exceeded")
            now = float(env.simulator.sim_time)
            actions = {}
            for sensor in env.sensing_satellites:
                if not sensor.requires_retasking:
                    actions[sensor.name] = NO_ACTION
                    continue
                key = (sensor.name, round(now, 6))
                decision = expected[key]
                assert private_action_mask(sensor).tolist() == decision["action_mask"]
                assert [
                    None if t is None else int(t.id)
                    for t in sensor.completion_candidates.targets
                ] == decision["candidate_ids"]
                assert [
                    None if p is None else p.name for p in sensor.completion_peers.peers
                ] == decision["peer_ids"]
                actions[sensor.name] = decision["action"]
                consumed.add(key)
            env.step(actions)
        assert consumed == set(expected)
        captures = [asdict(p) for p in env.rewarder._team_accounting.capture_attempts]
        assert captures == original["capture_products"]
        assert len(env.reward_events) == len(original["reward_events"])
        for actual, saved in zip(env.reward_events, original["reward_events"]):
            assert (
                actual["sensor"] == saved["sensor"]
                and actual["time_s"] == saved["time_s"]
            )
            np.testing.assert_allclose(
                actual["reward"], saved["reward"], rtol=0, atol=1e-12
            )
        np.testing.assert_allclose(
            list(env.episode_reward_by_agent.values()),
            list(original["reward_by_agent"].values()),
            rtol=0,
            atol=1e-10,
        )
        services = list(env.rewarder.service_entries)
        corrected = copy.deepcopy(original)
        corrected["delivered_products"] = [asdict(e) for e in services]
        corrected["coverage"] = coverage_metrics(
            [t.rso_target.id for t in env.passive_satellites],
            env.rewarder._team_accounting.capture_attempts,
            services,
            env.rewarder.quality_threshold,
            sensor_names=[s.name for s in env.sensing_satellites],
        )
        corrected["duplicate_products"] = duplicate_product_metrics(
            env.rewarder._team_accounting.capture_attempts,
            services,
            env.rewarder.quality_threshold,
            env.simulator.sim_time,
        )
        corrected["mission_metrics"] = mission_metrics(
            corrected, config.episode_duration_s
        )
        result.update(
            passed=True,
            simulated_seconds=float(env.simulator.sim_time),
            seed=config.seed,
            true_decisions=len(consumed),
            captures_exact=True,
            all_primitive_rewards_exact=True,
            all_private_masks_candidates_peers_exact=True,
            physically_delivered_products=len(services),
            useful_delivered_requests=env.rewarder.team_summary["unique_service_count"],
            uncredited_physical_deliveries=sum(e.credited_value == 0 for e in services),
            task_summary=env.rewarder.task_summary(),
            physical_remaining_products={
                s.name: len(s.data_store.products) for s in env.sensing_satellites
            },
        )
        assert result["physical_remaining_products"] == original["products"]
        assert len(services) + sum(original["products"].values()) == len(captures)
        write_json(output / "corrected-episode.json", corrected)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if env is not None:
            env.close()
        memory.close()
        result.update(
            wall_s=time.monotonic() - started,
            driver_cpu_s=time.process_time() - cpu_started,
        )
        write_json(output / "replay-verification.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--iteration", type=int, choices=[1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.training, args.iteration, args.output)


if __name__ == "__main__":
    main()
