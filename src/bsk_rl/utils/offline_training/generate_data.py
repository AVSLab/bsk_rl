import os
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

from ray.rllib.core.columns import Columns

from bsk_rl.bc_environment import make_env_and_args_for_bc as setup_env

import pathlib
import ray

from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


def test_policy(
        time_discount: bool = True,
):
    
    env, env_args = setup_env()

    obs, _ = env.reset()

    samples = []
    list_rewards = []
    sim_time = []

    step_count = 0

    while True:

        action = 0

        obs_prime, reward, terminated, truncated, _ = env.step(action)

        list_rewards.append(reward)
        sim_time.append(env.unwrapped.simulator.sim_time)

        samples.append(
            ({
                Columns.OBS: obs,
                Columns.NEXT_OBS: obs_prime,
                Columns.REWARDS: float(reward),
                Columns.ACTIONS: int(action),
                Columns.TRUNCATEDS: bool(truncated),
                Columns.TERMINATEDS: bool(terminated),
            })
        )

        obs = obs_prime

        step_count += 1


        if terminated or truncated or step_count >= 4:
            break

    v_f = []
    # gamma: float = train_config_dict["discount_factor"] # type: ignore
    gamma = 0.997
    for i, r in enumerate(list_rewards):
        v = r
        time_i = sim_time[i]
        for j in range(i + 1, len(list_rewards)):
            if not time_discount:
                v += gamma ** j * list_rewards[j]
            else:
                v += gamma ** (sim_time[j] - time_i) * list_rewards[j]
        v_f.append(v)
        samples[i][Columns.VF_PREDS] = float(v)
    return samples


if __name__ == "__main__":


    experience_length = 4096

    collected_experiences = 0

    while collected_experiences < experience_length:

        samples = test_policy()

        if collected_experiences == 0:
            experiences = samples
        else:
            experiences.extend(samples)

        collected_experiences += len(samples)

    print(f"Collected {collected_experiences} experiences")

    output_dir = pathlib.Path(__file__).parent / 'data'
    os.makedirs(output_dir, exist_ok=True)

    dataset = ray.data.from_items(samples)
    dataset = dataset.repartition(1).materialize()
    print(dataset)
    dataset.write_parquet(pathlib.Path().absolute() / "data", overwrite=True)

    print("Data saved successfully.")


