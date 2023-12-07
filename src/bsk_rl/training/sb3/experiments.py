import os

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from bsk_rl.utilities.sb3.custom_sb3_policies import CustomActorCriticPolicy
from bsk_rl.utilities.sb3.shielded_policies import (
    CustomActorCriticShieldedAgileEOSPolicy,
    CustomActorCriticShieldedMultiSensorEOSPolicy,
)

SEP = os.path.sep


def create_ppo_kwargs_list(**kwargs):
    activation_functions = kwargs.get("activation_functions", [nn.LeakyReLU])
    dropouts = kwargs.get("dropouts", [None])
    alphas = kwargs.get("alphas", [0.1])
    learning_rates = kwargs.get("learning_rates", [3e-4])
    clip_ranges = kwargs.get("clip_ranges", [0.1])
    entropy_coeffs = kwargs.get("entropy_coeffs", [0.01])
    widths = kwargs.get("widths", [20])
    depths = kwargs.get("depths", [4])
    batch_sizes = kwargs.get("batch_sizes", [64])
    n_epochs = kwargs.get("n_epochs", [50])
    max_grad_norms = kwargs.get("max_grad_norms", [0.5])
    n_experiments = kwargs.get("n_experiments", 1)

    kwargs_list = []

    # Set the experiment ID to zero
    experiment_ID = -1

    # Get the cartesian product of hyperparameter values
    for activation_fn in activation_functions:
        for dropout in dropouts:
            for alpha in alphas:
                for lr in learning_rates:
                    for clip_range in clip_ranges:
                        for ent_coef in entropy_coeffs:
                            for width in widths:
                                for depth in depths:
                                    for batch_size in batch_sizes:
                                        for epoch in n_epochs:
                                            for max_grad_norm in max_grad_norms:
                                                # Update the experiment ID
                                                experiment_ID += 1
                                                for experiment_num in range(
                                                    n_experiments
                                                ):
                                                    kwargs = {
                                                        "activation_fn": activation_fn,
                                                        "dropout": dropout,
                                                        "alpha": alpha,
                                                        "lr": lr,
                                                        "clip_range": clip_range,
                                                        "ent_coef": ent_coef,
                                                        "width": width,
                                                        "depth": depth,
                                                        "batch_size": batch_size,
                                                        "epoch": epoch,
                                                        "max_grad_norm": max_grad_norm,
                                                        "experiment_ID": experiment_ID,
                                                    }
                                                    kwargs_list.append(kwargs)

        return kwargs_list


def create_a2c_kwargs_list(**kwargs):
    activation_functions = kwargs.get("activation_functions", [nn.LeakyReLU])
    widths = kwargs.get("widths", [20])
    depths = kwargs.get("depths", [4])
    dropouts = kwargs.get("dropouts", [None])
    alphas = kwargs.get("alphas", [0.1])
    learning_rates = kwargs.get("learning_rates", [3e-4])
    entropy_coeffs = kwargs.get("entropy_coeffs", [0.01])
    n_steps_ = kwargs.get("n_steps_", [45])
    n_experiments = kwargs.get("n_experiments", 1)

    kwargs_list = []

    experiment_ID = -1

    for activation_fn in activation_functions:
        for width in widths:
            for depth in depths:
                for dropout in dropouts:
                    for alpha in alphas:
                        for lr in learning_rates:
                            for ent_coef in entropy_coeffs:
                                for n_steps in n_steps_:
                                    experiment_ID += 1
                                    for experiment_num in range(n_experiments):
                                        kwargs = {
                                            "width": width,
                                            "depth": depth,
                                            "dropout": dropout,
                                            "alpha": alpha,
                                            "lr": lr,
                                            "ent_coef": ent_coef,
                                            "n_steps": n_steps,
                                            "activation_fn": activation_fn,
                                            "experiment_ID": experiment_ID,
                                        }
                                        kwargs_list.append(kwargs)

    return kwargs_list


def create_dqn_kwargs_list(**kwargs):
    learning_rates = kwargs.get("learning_rates", [1e-4])
    depths = kwargs.get("depths", [4])
    widths = kwargs.get("widths", [20])
    batch_sizes = kwargs.get("batch_sizes", [64])
    buffer_sizes = kwargs.get("buffer_sizes", [5e5])
    n_experiments = kwargs.get("n_experiments", 1)

    kwargs_list = []

    experiment_ID = -1

    for lr in learning_rates:
        for depth in depths:
            for width in widths:
                for batch_size in batch_sizes:
                    for buffer_size in buffer_sizes:
                        experiment_ID += 1
                        for experiment_num in range(n_experiments):
                            kwargs = {
                                "lr": lr,
                                "depth": depth,
                                "width": width,
                                "batch_size": batch_size,
                                "buffer_size": buffer_size,
                                "experiment_ID": experiment_ID,
                            }
                            kwargs_list.append(kwargs)

    return kwargs_list


def ppo_experiment(
    policy_kwargs,
    learning_rate,
    clip_range,
    ent_coef,
    batch_size,
    epoch,
    max_grad_norm,
    idx,
    agent_dir,
    n_its=10,
    base_steps=1020,
    shielded=False,
    env_name="MultiSensorEOS-v0",
    n_steps=90,
    num_cores=4,
):
    """
    Run a PPO experiment with the given hyperparameters

    :param policy_kwargs: (dict) Policy kwargs
    :param learning_rate: (float) Learning rate
    :param clip_range: (float) Clip range
    :param ent_coef: (float) Entropy coefficient
    :param batch_size: (int) Batch size
    :param epoch: (int) Number of epochs
    :param idx: (int) Index of the experiment
    :param agent_dir: (str) Directory where to save the agent
    :param n_its: (int) Number of iterations
    :param shielded: (bool) Whether to use the shielded policy
    :param env_name: (str) Name of the environment
    :param n_steps: (int) Number of steps.
    """
    network_dir = agent_dir + "/network_" + str(idx) + SEP

    validation_reward = []

    total_step = int(base_steps * n_its * n_steps)

    print("Base steps: ", base_steps)
    print("Iterations: ", n_its)
    print("Total steps: ", total_step)

    os.makedirs(network_dir + "logs/", exist_ok=True)
    os.makedirs(network_dir + "logger/", exist_ok=True)
    multienv = SubprocVecEnv(
        [lambda: gym.make(env_name) for _ in range(num_cores)], start_method="spawn"
    )
    multienv = VecMonitor(multienv, network_dir + "logs/")

    # set up logger
    new_logger = configure(network_dir + "logger/", ["stdout", "csv", "tensorboard"])

    if shielded:
        if env_name == "MultiSensorEOS-v0":
            model = PPO(
                CustomActorCriticShieldedMultiSensorEOSPolicy,
                multienv,
                device="cuda",
                verbose=2,
                tensorboard_log=network_dir,
                n_epochs=epoch,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_grad_norm=max_grad_norm,
                clip_range=clip_range,
                ent_coef=ent_coef,
                gamma=0.999,
                policy_kwargs=policy_kwargs,
            )
        elif env_name == "AgileEOS-v0":
            model = PPO(
                CustomActorCriticShieldedAgileEOSPolicy,
                multienv,
                device="cuda",
                verbose=2,
                tensorboard_log=network_dir,
                n_epochs=epoch,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_grad_norm=max_grad_norm,
                clip_range=clip_range,
                ent_coef=ent_coef,
                gamma=0.999,
                policy_kwargs=policy_kwargs,
            )
        else:
            raise ValueError("Shielding not implemented for this environment", env_name)
    else:
        model = PPO(
            CustomActorCriticPolicy,
            multienv,
            device="cuda",
            verbose=2,
            tensorboard_log=network_dir,
            n_epochs=epoch,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            clip_range=clip_range,
            ent_coef=ent_coef,
            gamma=0.999,
            policy_kwargs=policy_kwargs,
        )

    # Set new logger
    model.set_logger(new_logger)

    for i in range(0, n_its):
        model.set_env(multienv)
        model.learn(total_timesteps=int(total_step / n_its), reset_num_timesteps=False)

        multienv.close()
        del multienv

        os.rename(
            network_dir + "logs/monitor.csv",
            network_dir + "logs/monitor_" + str(i) + ".csv",
        )

        multienv = SubprocVecEnv(
            [lambda: gym.make(env_name) for _ in range(num_cores)], start_method="spawn"
        )
        multienv = VecMonitor(multienv, network_dir + "logs/")

    model.save(network_dir + "final_network_" + str(idx))

    # run environment
    for _ in range(0, 3):
        obs = multienv.reset()
        reward_sum = np.zeros(num_cores)
        for _ in range(0, n_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = multienv.step(action)
            reward_sum = np.add(reward_sum, np.array(rewards))

        for r in reward_sum:
            validation_reward.append(r)

    print("Average Reward: ", np.average(validation_reward))

    multienv.close()
    del multienv
    del model

    return validation_reward, "final_network_" + str(idx)


def a2c_experiment(
    agent_dir,
    policy_kwargs,
    learning_rate=0.0007,
    ent_coef=0.01,
    idx=0,
    n_its=10,
    base_steps=1020,
    env_name="MultiSensorEOS-v0",
    max_steps=90,
    n_steps=45,
    num_cores=4,
):
    """
    Run an A2C experiment with the given hyperparameters

    :param policy_kwargs: (dict) Policy kwargs
    :param learning_rate: (float) Learning rate
    :param ent_coef: (float) Entropy coefficient
    :param idx: (int) Index of the experiment
    :param agent_dir: (str) Directory where to save the agent
    :param n_its: (int) Number of iterations
    :param env_name: (str) Name of the environment
    :param max_steps: (int) Maximum number of steps in the environment
    :param n_steps: (int) Number of steps before update
    :param num_cores: (int) Number of cores.
    """

    network_dir = agent_dir + "/network_" + str(idx) + SEP

    validation_reward = []

    total_step = int(base_steps * n_its * max_steps)

    print("Base steps: ", base_steps)
    print("Iterations: ", n_its)
    print("Total steps: ", total_step)

    os.makedirs(network_dir + "/logs/", exist_ok=True)
    os.makedirs(network_dir + "/logger/", exist_ok=True)
    multienv = SubprocVecEnv(
        [lambda: gym.make(env_name) for _ in range(num_cores)], start_method="spawn"
    )
    multienv = VecMonitor(multienv, network_dir + "/logs/")

    # set up logger
    new_logger = configure(network_dir + "/logger/", ["stdout", "csv", "tensorboard"])

    model = A2C(
        CustomActorCriticPolicy,
        multienv,
        device="cpu",
        verbose=2,
        tensorboard_log=network_dir,
        n_steps=n_steps,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        gamma=0.999,
        policy_kwargs=policy_kwargs,
    )

    # Set new logger
    model.set_logger(new_logger)

    for i in range(0, n_its):
        model.set_env(multienv)
        model.learn(
            total_timesteps=int(total_step / n_its),
            reset_num_timesteps=False,
            log_interval=10,
        )

        multienv.close()
        del multienv

        os.rename(
            network_dir + "/logs/monitor.csv",
            network_dir + "/logs/monitor_" + str(i) + ".csv",
        )

        multienv = SubprocVecEnv(
            [lambda: gym.make(env_name) for _ in range(num_cores)], start_method="spawn"
        )
        multienv = VecMonitor(multienv, network_dir + "/logs/")

    model.save(network_dir + "/final_network_" + str(idx))

    # run environment
    for _ in range(0, 3):
        obs = multienv.reset()
        reward_sum = np.zeros(num_cores)
        for _ in range(0, max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = multienv.step(action)
            reward_sum = np.add(reward_sum, np.array(rewards))

        for r in reward_sum:
            validation_reward.append(r)

    print("Average Reward: ", np.average(validation_reward))

    multienv.close()
    del multienv
    del model

    return validation_reward, "final_network_" + str(idx)


def dqn_experiment(
    agent_dir,
    policy_kwargs,
    idx,
    n_its,
    base_steps,
    learning_rate=1e-4,
    batch_size=64,
    buffer_size=5e5,
    num_cores=4,
    n_steps=90,
    env_name="MultiSensorEOS-v0",
):
    network_dir = agent_dir + "/network_" + str(idx) + SEP

    validation_reward = []

    total_step = int(base_steps * n_its * n_steps)

    print("Policy Kwargs: ", policy_kwargs)
    print("Learning Rate: ", learning_rate)
    print("Batch Size: ", batch_size)
    print("Buffer Size: ", buffer_size)
    print("Num Cores: ", num_cores)
    print("Num Steps: ", total_step)

    os.makedirs(network_dir + "logs/", exist_ok=True)
    os.makedirs(network_dir + "logger/", exist_ok=True)
    multienv = SubprocVecEnv(
        [lambda: gym.make(env_name) for _ in range(num_cores)], start_method="spawn"
    )
    multienv = VecMonitor(multienv, network_dir + "logs/")

    # set up logger
    new_logger = configure(network_dir + "logger/", ["stdout", "csv", "tensorboard"])

    model = DQN(
        "MlpPolicy",
        multienv,
        verbose=2,
        tensorboard_log=network_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=0.999,
        buffer_size=buffer_size,
        policy_kwargs=policy_kwargs,
        train_freq=1,
        learning_starts=1,
        target_update_interval=1,
    )

    # Set new logger
    model.set_logger(new_logger)

    for i in range(0, n_its):
        model.set_env(multienv)
        model.learn(
            total_timesteps=int(total_step / n_its),
            reset_num_timesteps=False,
            log_interval=100,
        )

        multienv.close()
        del multienv

        os.rename(
            network_dir + "logs/monitor.csv",
            network_dir + "logs/monitor_" + str(i) + ".csv",
        )

        multienv = SubprocVecEnv(
            [lambda: gym.make(env_name) for _ in range(num_cores)], start_method="spawn"
        )
        multienv = VecMonitor(multienv, network_dir + "logs/")

    model.save(network_dir + "final_network_" + str(idx))

    # run environment
    for _ in range(0, 3):
        obs = multienv.reset()
        reward_sum = np.zeros(num_cores)
        for _ in range(0, n_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = multienv.step(action)
            reward_sum = np.add(reward_sum, np.array(rewards))

        for r in reward_sum:
            validation_reward.append(r)

    print("Average Reward: ", np.average(validation_reward))

    multienv.close()
    del multienv
    del model

    return validation_reward, "final_network_" + str(idx)


def run_ppo_experiments(
    agent_dir,
    kwargs_list,
    n_its=10,
    base_steps=1020,
    index=None,
    env_name="MultiSensorEOS-v0",
    n_steps=45,
    num_cores=4,
    shielded=False,
):
    """
    Run PPO experiments with the given hyperparameters

    :param agent_dir: (str) Directory where to save the agent
    :param n_its: (int) Number of iterations
    :param kwargs_list: (list) List of dictionaries containing hyperparameters
    :param index: (int) Index of the hyperparameter dictionary to use
    :param env_name: (str) Name of the environment
    :param n_steps: (int) Number of steps
    :param num_cores: (int) Number of cores
    :param shielded: (bool) Whether to use shielded policy].
    """
    results = {}
    idx = 0

    if index is not None:
        kwargs_list = [kwargs_list[index]]
        idx = index

    for kwargs in kwargs_list:
        print(kwargs)
        activation_fn = kwargs.get("activation_fn", nn.LeakyReLU)
        width = kwargs.get("width", 20)
        depth = kwargs.get("depth", 4)
        dropout = kwargs.get("dropout", None)
        alpha = kwargs.get("alpha", 0.1)
        lr = kwargs.get("lr", 3e-4)
        clip_range = kwargs.get("clip_range", 0.1)
        ent_coef = kwargs.get("ent_coef", 0.01)
        epoch = kwargs.get("epoch", 50)
        batch_size = kwargs.get("batch_size", 64)
        max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        experiment_ID = kwargs.get("experiment_ID", None)

        # Create the policy kwargs
        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch={
                "width": width,
                "depth": depth,
                "dropout": dropout,
                "alpha": alpha,
            },
        )

        # Call the training function
        reward, model_name = ppo_experiment(
            policy_kwargs,
            lr,
            clip_range,
            ent_coef,
            batch_size,
            epoch,
            max_grad_norm,
            idx,
            agent_dir,
            n_its,
            base_steps,
            env_name=env_name,
            n_steps=n_steps,
            num_cores=num_cores,
            shielded=shielded,
        )

        # Append to the results
        results.update(
            {
                model_name: {
                    "validation_reward": reward,
                    "policy_kwargs": policy_kwargs,
                    "lr": lr,
                    "clip_range": clip_range,
                    "ent_coef": ent_coef,
                    "batch_size": batch_size,
                    "n_epochs": epoch,
                    "max_grad_norm": max_grad_norm,
                    "experiment_ID": experiment_ID,
                }
            }
        )

        # Intermittently save the results
        np.save(agent_dir + SEP + "results_" + str(idx) + ".npy", results)

        # Increment the index
        idx += 1


def run_a2c_experiments(
    agent_dir,
    kwargs_list,
    n_its=10,
    base_steps=1020,
    index=None,
    env_name="MultiSensorEOS-v0",
    max_steps=90,
    num_cores=4,
):
    results = {}
    idx = 0

    if index is not None:
        kwargs_list = [kwargs_list[index]]
        idx = index

    for kwargs in kwargs_list:
        print(kwargs)
        activation_fn = kwargs.get("activation_fn", nn.LeakyReLU)
        width = kwargs.get("width", 20)
        depth = kwargs.get("depth", 4)
        dropout = kwargs.get("dropout", None)
        alpha = kwargs.get("alpha", 0.1)
        lr = kwargs.get("lr", 3e-4)
        ent_coef = kwargs.get("ent_coef", 0.01)
        n_steps = kwargs.get("n_steps", 5)

        # Create the policy kwargs
        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch={
                "width": width,
                "depth": depth,
                "dropout": dropout,
                "alpha": alpha,
            },
        )

        # Call the training function
        reward, model_name = a2c_experiment(
            agent_dir,
            policy_kwargs,
            lr,
            ent_coef,
            idx,
            n_its,
            base_steps,
            env_name=env_name,
            max_steps=max_steps,
            n_steps=n_steps,
            num_cores=num_cores,
        )

        # Append to the results
        results.update(
            {
                model_name: {
                    "validation_reward": reward,
                    "policy_kwargs": policy_kwargs,
                    "lr": lr,
                    "ent_coef": ent_coef,
                    "n_steps": n_steps,
                }
            }
        )

        # Intermittently save the results
        np.save(agent_dir + SEP + "results_" + str(idx) + ".npy", results)

        # Increment the index
        idx += 1


def run_dqn_experiments(
    agent_dir,
    kwargs_list,
    n_its=10,
    base_steps=1020,
    index=None,
    env_name="MultiSensorEOS-v0",
    max_steps=90,
    num_cores=4,
):
    results = {}
    idx = 0

    if index is not None:
        kwargs_list = [kwargs_list[index]]
        idx = index

    for kwargs in kwargs_list:
        print(kwargs)
        width = kwargs.get("width", 20)
        depth = kwargs.get("depth", 4)
        lr = kwargs.get("lr", 1e-4)
        batch_size = kwargs.get("batch_size", 64)
        buffer_size = kwargs.get("buffer_size", 5e5)

        sizes = [width] * depth

        # Create the policy kwargs
        policy_kwargs = dict(net_arch=sizes)

        # Call the training function
        reward, model_name = dqn_experiment(
            agent_dir,
            policy_kwargs,
            idx,
            n_its,
            base_steps,
            learning_rate=lr,
            batch_size=batch_size,
            buffer_size=buffer_size,
            num_cores=num_cores,
            n_steps=max_steps,
            env_name=env_name,
        )

        # Append to the results
        results.update(
            {
                model_name: {
                    "validation_reward": reward,
                    "policy_kwargs": policy_kwargs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                }
            }
        )

        # Intermittently save the results
        np.save(agent_dir + SEP + "results_" + str(idx) + ".npy", results)

        # Increment the index
        idx += 1
