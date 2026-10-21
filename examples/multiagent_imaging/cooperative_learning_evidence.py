"""Link actual shared primitive rewards to transmitted-action PPO rows."""

import gzip
import json
from pathlib import Path

import numpy as np

from bsk_rl import NO_ACTION
from examples.multiagent_imaging.fixed_batch_diagnostic import independently_recompute


def sender_credit_evidence(directory, config, episodes):
    directory = Path(directory)
    batch, audit = independently_recompute(directory, config)
    with gzip.open(directory / "raw-episodes.json.gz", "rt") as stream:
        raw = json.load(stream)
    by_episode = {e["episode_id"]: e for e in episodes}
    traces = []
    offset = 0
    for row in raw:
        starts = [i for i, a in enumerate(row["actions"]) if a != NO_ACTION]
        episode = by_episode[row["multi_agent_episode_id"]]
        times = np.cumsum([i["d_ts"] for i in row["infos"]])
        events = [
            e
            for e in episode["reward_events"]
            if e["sensor"] == row["agent_id"]
            and e.get("components", {}).get("credited_science", 0) > 0
            and e["components"].get("owned_science", 0) == 0
        ]
        for j, start in enumerate(starts):
            if row["actions"][start] < 5 + config.n_candidates:
                continue
            later = [e for e in events if e["time_s"] > times[start]]
            if not later:
                continue
            event = later[0]
            index = offset + j
            traces.append(
                dict(
                    agent=row["agent_id"],
                    episode_id=row["multi_agent_episode_id"],
                    selected_transmit_action=row["actions"][start],
                    action_start_s=float(times[start]),
                    downstream_shared_event=event,
                    delay_s=float(event["time_s"] - times[start]),
                    discount_to_event=float(
                        config.discount_per_s ** (event["time_s"] - times[start])
                    ),
                    condensed_reward=float(batch["rewards"][index]),
                    normalized_ppo_advantage=float(batch["advantages"][index]),
                    value_target=float(batch["value_targets"][index]),
                    semantics="Actual PPO row and later shared science reward; temporal association is not counterfactual causal attribution",
                )
            )
        offset += len(starts)
    return dict(
        passed=True,
        independent_target_reconstruction=audit,
        sender_transmit_rows_with_later_shared_science=len(traces),
        traces=traces,
        causal_credit_claim=False,
    )
