"""Local full-mission recorder parity and repeated-reset memory measurements.

Run each arm/recorder setting in a fresh process. Identical seeds and a fixed
observation-based controller permit stepwise science-trace comparison. No Ray,
PPO, W&B, native rebuild, or cluster access. Outputs are machine-readable only.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from examples.multiagent_imaging.update_evidence import peak_rss_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["catalog_tx", "geometry_only"], required=True)
    parser.add_argument("--recorders", choices=["on", "off"], required=True)
    parser.add_argument("--missions", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=2400)
    args = parser.parse_args()
    if not 1 <= args.missions <= 4:
        parser.error("Use one to four complete missions")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["BSK_RL_TARGET_RECORDERS"] = "1" if args.recorders == "on" else "0"
    import Basilisk
    import numpy as np
    import psutil
    from bsk_rl import NO_ACTION
    from bsk_rl.obs.completion_observations import GLOBAL_FEATURES, TARGET_FEATURES, VALID_TARGET_FEATURE, private_action_mask
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.environment import build_environment

    config_path = Path(__file__).with_name("configs") / f"walker4_t200_cd1_{args.arm}.json"
    config = MultiAgentImagingConfig.from_json(config_path)
    assert config.episode_duration_s == 45000 and config.reward_credit_mode == "owned"
    process = psutil.Process()
    start = time.monotonic()
    memory = []
    results = []
    def sample(label, mission, sim_time):
        full = process.memory_full_info()
        memory.append(dict(label=label, mission=mission, sim_time_s=sim_time,
                           rss_bytes=full.rss, uss_bytes=full.uss))

    sample("before-build", -1, 0)
    env = build_environment(config)
    try:
        for mission in range(args.missions):
            observations, _ = env.reset(seed=3105)
            sample("reset", mission, 0)
            trace = hashlib.sha256()
            def add_trace(task, current_observations, actions, rewards):
                trace.update(json.dumps(dict(time=float(task.simulator.sim_time), actions=actions,
                    rewards=rewards), sort_keys=True, allow_nan=False).encode())
                arrays = [np.asarray(v, dtype="<f8") for _, v in sorted(current_observations.items())]
                arrays.append(np.asarray([[*s.dynamics.r_BN_N, *s.dynamics.v_BN_N]
                    for s in task.satellites], dtype="<f8"))
                arrays.append(np.asarray([[s.dynamics.battery_charge_fraction,
                    s.dynamics.storage_level_fraction, *s.dynamics.wheel_speeds_fraction]
                    for s in task.sensing_satellites], dtype="<f8"))
                for array in arrays:
                    if not np.isfinite(array).all():
                        raise ValueError("Nonfinite scientific trace")
                    array = array.copy()
                    array[array == 0] = 0  # Signed zero has identical numeric meaning.
                    trace.update(array.tobytes())
            add_trace(env, observations, {}, {})
            steps, next_sample = 0, 900
            with (args.output / f"trace-{mission}.jsonl").open("x") as stream:
                while env.agents:
                    if steps > 40000 or time.monotonic() - start > args.wall_seconds:
                        raise RuntimeError("Local validation wall/step cap exceeded")
                    actions = {}
                    for sat in env.sensing_satellites:
                        if sat.name not in env.agents:
                            continue
                        if not sat.requires_retasking:
                            actions[sat.name] = NO_ACTION
                            continue
                        row = np.asarray(observations[sat.name])
                        targets = row[GLOBAL_FEATURES:GLOBAL_FEATURES + config.n_candidates * TARGET_FEATURES].reshape(config.n_candidates, TARGET_FEATURES)
                        valid = np.flatnonzero(targets[:, VALID_TARGET_FEATURE] > .5)
                        # Fixed controller exercises imaging, storage/downlink,
                        # and explicit directed sends; no optimizer is involved.
                        if sat.dynamics.storage_level_fraction > .5:
                            action = 1
                        elif args.arm == "catalog_tx" and env.decision_counts[sat.name] % 17 == 16:
                            action = 5 + config.n_candidates + (steps % config.n_peers)
                        else:
                            action = 5 + int(valid[0]) if len(valid) else 0
                        mask = private_action_mask(sat)
                        if not mask[action]:
                            eligible = [5 + int(i) for i in valid if mask[5 + int(i)]]
                            action = eligible[0] if eligible else 0
                        assert mask[action]
                        actions[sat.name] = action
                    observations, rewards, _, _, _ = env.step(actions)
                    add_trace(env, observations, actions, rewards)
                    steps += 1
                    stream.write(json.dumps(dict(time_s=float(env.simulator.sim_time),
                        step=steps, science_sha256=trace.hexdigest())) + "\n")
                    if env.simulator.sim_time >= next_sample:
                        sample("mission", mission, float(env.simulator.sim_time))
                        next_sample += 900
                assert env.simulator.sim_time == 45000, "Incomplete mission"
            recorders = [getattr(s.dynamics, key) for s in env.passive_satellites
                for key in ("target_state_recorder", "target_eclipse_recorder")]
            final_science = dict(rewards=env.episode_reward_by_agent,
                decisions=env.decision_counts, actions=env.action_selection_counts,
                captures=[asdict(p) for p in env.rewarder._team_accounting.capture_attempts],
                deliveries=[asdict(p) for p in env.rewarder.service_entries],
                reward_events=env.reward_events, transmissions=env.communicator.transmission_history,
                packets=env.communicator.delivery_history, summary=env.rewarder.team_summary)
            with (args.output / f"science-{mission}.json").open("x") as stream:
                json.dump(final_science, stream, sort_keys=True, allow_nan=False)
            sample("mission-end", mission, 45000)
            results.append(dict(mission=mission, steps=steps, simulated_seconds=45000,
                science_trace_sha256=trace.hexdigest(), decisions=dict(env.decision_counts),
                reward=dict(env.episode_reward_by_agent),
                recorder_count=sum(r is not None for r in recorders),
                recorder_samples=sum(int(r.size()) for r in recorders if r is not None)))
            # Do not let the validation harness itself retain recorder/native refs.
            del recorders, final_science, observations, sat, targets, row
            gc.collect()
            print(json.dumps(results[-1]), flush=True)
    finally:
        env.close()
    del env
    gc.collect()
    sample("after-close", args.missions, 0)
    result = dict(arm=args.arm, recorders=args.recorders, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        native_build=Basilisk.getBuildInfo(), missions=results, memory=memory,
        max_rss_bytes=peak_rss_bytes(),
        wall_s=time.monotonic()-start, new_PPO_updates=0, cluster_jobs=0,
        scope="Local full native missions; identical fixed controller, no learning or native-version comparison")
    with (args.output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
