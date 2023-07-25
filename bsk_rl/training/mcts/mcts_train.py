import os
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential

from bsk_rl.agents.mcts import MCTS
from bsk_rl.agents.state_machine import StateMachine
from bsk_rl.utilities.mcts.rollout_policies import (
    AgileEOSRolloutPolicy,
    MultiSensorEOSRolloutPolicy,
    SimpleEOSRolloutPolicy,
    SmallBodyRolloutPolicy,
)

plt.style.use("tableau-colorblind10")
plt.style.use("seaborn-colorblind")


def mcts_batch(
    data_directory,
    data_indicator,
    c=50,
    num_sims=50,
    initial_conditions=None,
    result_list=None,
    render=False,
    env_name="SimpleEOS-v0",
    num_steps=45,
    t_final=270.0,
):
    """
    The function performs a single run of MCTS over a planning horizon, generating \
    performance and training data
    :param data_directory: Data directory to store training data
    :param data_indicator: Data indicator to append to filename
    :param c: MCTS exploration constant
    :param num_sims: number of simulations-per-step
    :param initial_conditions: Dictionary of initial conditions
    :param result_list: Results list
    :param render: T/F to render BSK sim using Vizard
    :param env_name: environment name
    :return: N/A
    """
    np.random.seed(int(data_indicator.split("_")[-1]))
    # Create env
    env = gym.make(env_name)
    env.max_length = t_final
    env.max_steps = num_steps

    env.render = render

    # Run an MCTS episode, returning Q, N, reward sum, and action history
    Q, N, reward, act_hist, exec_time, final_info, metrics = run_episode(
        env, num_steps, t_final, c, num_sims, initial_conditions, env_name
    )

    # Grab the performance metrics from the environment (downlink utilization, etc)
    performance_metrics = metrics
    performance_metrics["reward_sum"] = reward
    performance_metrics["action_history"] = act_hist
    performance_metrics["exec_time"] = exec_time

    # Grab the states and value functions after the training cycle
    states = np.array(list(Q.keys()))
    state_action_output = []
    for state_actions in Q.values():
        state_action_output.append(list(state_actions.values()))
    state_action_output = np.array(state_action_output)

    num_visited_output = []
    for num_visited in N.values():
        num_visited_output.append(list(num_visited.values()))
    num_visited_output = np.array(num_visited_output)

    # Add to the data dictionary
    data = {
        "states": states,
        "action_value": state_action_output,
        "num_visited": num_visited_output,
        "ic": env.simulator.initial_conditions,
        "info": final_info,
    }

    # Save the training data
    os.makedirs(data_directory + "/Training Data/", exist_ok=True)
    os.makedirs(data_directory + "/Validation Plots/", exist_ok=True)
    np.save(
        data_directory
        + "/Training Data/"
        + "action_value_all_info_"
        + data_indicator.replace(".", "__")
        + ".npy",
        data,
    )

    if result_list is not None:
        # return performance_metrics as a dictionary
        result_list.append(
            {(c, num_sims, int(data_indicator.split("_")[-1])): performance_metrics}
        )
        return
    else:
        return {(c, num_sims, int(data_indicator.split("_")[-1])): performance_metrics}


def run_episode(env, num_steps, t_final, c, num_sims, initial_conditions, env_name):
    """
    Runs an episode of MCTS.
    :param env: Gym environment
    :param num_steps: number of steps to take
    :param t_final: Final time
    :param c: Exploration constant
    :param num_sims: Number of simulations-per-step
    :param initial_conditions: Dictionary of initial conditions
    :param env_name: environment name
    :return Q: state-action value estimates
    :return N: Number of times the state-action pairs were visited
    :return reward: Reward sum
    :return actHist: history of actions
    :return exec_time: execution time
    :return final_info: final infor from env
    """

    # Call reset on the environment
    ob, info = env.reset()

    # Create rollout policies automatically
    state_machine = StateMachine()
    abs_path = os.path.dirname(os.path.abspath(__file__))
    if "AgileEOS" in env_name:
        state_machine.loadTransferConditions(
            os.path.join(abs_path, "../../utilities/state_machine/agile_eos_ops.adv")
        )
        rollout_policy = AgileEOSRolloutPolicy(env=env, state_machine=state_machine)
    elif "SimpleEOS" in env_name:
        # Get the path relative to directory
        state_machine.loadTransferConditions(
            os.path.join(abs_path, "../../utilities/state_machine/simple_eos_ops.adv")
        )
        rollout_policy = SimpleEOSRolloutPolicy(state_machine=state_machine)
    elif "SmallBodyScience" in env_name:
        rollout_policy = SmallBodyRolloutPolicy(env=env, state_machine=state_machine)
    elif "MultiSensorEOS" in env_name:
        state_machine.loadTransferConditions(
            os.path.join(
                abs_path, "../../utilities/state_machine/multisensor_eos_ops.adv"
            )
        )
        rollout_policy = MultiSensorEOSRolloutPolicy(
            env=env, state_machine=state_machine
        )
    else:
        print(
            "Environment name "
            + env_name
            + " not found while instantiating state machine"
        )

    # Create an MCTS agent
    MCTS_Agent = MCTS(c=c, num_sims=num_sims, rollout_policy=rollout_policy)

    # Set the initial conditions if they don't exist
    if initial_conditions is not None:
        env.initial_conditions = initial_conditions

    # Reset with the initial conditions
    ob, info = env.reset(options={"initial_conditions": env.initial_conditions})
    ob_hist = [ob]

    # Set the type of environment in the MCTS agent
    MCTS_Agent.setEnv(
        env_name, env.initial_conditions, max_steps=num_steps, max_length=t_final
    )

    actHist = []

    # Total reward
    reward_sum = 0.0

    # Start the execution timer
    start_time = time.time()

    # Loop through each time step
    for ind in range(0, num_steps):
        # Set the depth to be the end of the environment
        d = num_steps - ind

        # Select new action
        act = MCTS_Agent.selectAction(ob, d, actHist)

        # Reset with the initial conditions
        env.reset(options={"initial_conditions": env.initial_conditions})

        # Step through env to take it to current place
        env.return_obs = False
        for stepAct in actHist:
            _, _, _, _, _ = env.step(stepAct)
        env.return_obs = True

        # Append last action to action list
        actHist.append(act)

        print("Actual Action Taken: ", act)
        print("Real Environment Step: ", ind)

        # Take the step in the environment
        ob, reward, episode_over, _, info = env.step(act)
        ob_hist.append(ob)
        print("Reward: ", reward)

        # Sum the reward, add to rewards array
        reward_sum = reward_sum + reward

        # Append to the trajectory
        MCTS_Agent.trajectory.append(
            [tuple(ob_hist[-2].reshape(1, -1)[0]), act, reward, reward_sum, info]
        )

        # If the episode is over, end the simulation
        if episode_over:
            print("episode over")
            break

    # End the execution timer
    end_time = time.time()
    exec_time = end_time - start_time

    # Backup the trajectory along the main tree
    MCTS_Agent.backup_tree()

    # Append Q
    Q = MCTS_Agent.Q_main
    N = MCTS_Agent.N_main
    final_info = MCTS_Agent.info
    reward = reward_sum

    return Q, N, reward, actHist, exec_time, final_info, info["metrics"]


def data_number(x):
    """
    Splits the data indicator string to return the data number
    :param x: data indicator string
    :return: data number
    """
    return int(((x.rsplit("_")[-1]).rsplit("."))[0])


def create_model(
    hidden_layer_num, net_size, activation, num_states, dropout, alpha, num_actions
):
    """
    Creates a feedforward neural network subject to various hyperparameters.
    :param hidden_layer_num: Number of hidden layers
    :param net_size: Widths of hidden layers
    :param activation: activation function, either Leaky ReLU or tanh
    :param num_states: Number of states
    :param dropout: Dropout rate
    :param alpha: alpha-parameter for Leaky ReLU activation function
    :param num_actions: number of actions
    :return: model
    """
    # Create model
    model = Sequential()

    # Add layers
    for layer in range(0, hidden_layer_num):
        if layer == 0:
            if activation == "tanh":
                model.add(
                    Dense(net_size, input_shape=(num_states,), activation=activation)
                )
            elif activation == "LeakyReLU":
                model.add(Dense(net_size, input_shape=(num_states,)))
                model.add(LeakyReLU(alpha=alpha))
        else:
            if activation == "tanh":
                model.add(Dense(net_size, activation=activation))
            elif activation == "LeakyReLU":
                model.add(Dense(net_size))
                model.add(LeakyReLU(alpha=alpha))

        # Add dropout layers
        if dropout is not None:
            model.add(Dropout(dropout))

    # Add the output layer and compile
    model.add(Dense(num_actions, activation="linear"))
    model.compile(
        loss="mse",
        optimizer="Adam",
        metrics=["mean_squared_error", "mean_absolute_error"],
    )
    model.summary()

    return model


def load_and_modify_data(data_directory, modified_states=[]):
    """
    Loads AND modifies the training data
    :param data_directory: data directory to load data from
    :param modified_states: modified states. First entry is a list of indices to keep.\
    Next entries are dictionary keys
    for info.
    :return: train-test split of data
    """
    data = []

    # Load the data
    for count, filename in enumerate(
        sorted(
            Path(data_directory + "/Training Data/").iterdir(), key=os.path.getmtime
        ),
        start=0,
    ):
        print(count, filename)
        # if filename.startswith('AlphaZero'):
        data.append(np.load(filename, allow_pickle=True))

    for idx in range(0, len(data)):
        data[idx] = data[idx].item()

    states = None

    # Load the data
    for idx2 in range(0, len(data)):
        if all(act_val < 0 for act_val in data[idx2]["action_value"][1, :]):
            print(data[idx2]["action_value"][1, :])
            continue
        elif states is None:
            if len(modified_states) > 0:
                for idx3, elem in enumerate(modified_states):
                    if idx3 == 0:
                        states = data[idx2]["states"][:, elem]
                    else:
                        states_to_add = [
                            list(data[idx2]["info"].values())[0][elem].reshape(
                                list(data[idx2]["info"].values())[0][elem].size
                            )
                        ]
                        for idx4 in range(1, len(list(data[idx2]["info"].values()))):
                            states_to_add = np.concatenate(
                                (
                                    states_to_add,
                                    [
                                        list(data[idx2]["info"].values())[idx4][
                                            elem
                                        ].reshape(
                                            list(data[idx2]["info"].values())[idx4][
                                                elem
                                            ].size
                                        )
                                    ],
                                )
                            )
                        states = np.concatenate((states, states_to_add), axis=1)
                        print(states.shape)
            else:
                states = data[idx2]["states"]
            action_value = data[idx2]["action_value"]
        else:
            if len(modified_states) > 0:
                for idx3, elem in enumerate(modified_states):
                    if idx3 == 0:
                        states_2 = data[idx2]["states"][:, elem]
                    else:
                        states_to_add = [
                            list(data[idx2]["info"].values())[0][elem].reshape(
                                list(data[idx2]["info"].values())[0][elem].size
                            )
                        ]
                        for idx4 in range(1, len(list(data[idx2]["info"].values()))):
                            states_to_add = np.concatenate(
                                (
                                    states_to_add,
                                    [
                                        list(data[idx2]["info"].values())[idx4][
                                            elem
                                        ].reshape(
                                            list(data[idx2]["info"].values())[idx4][
                                                elem
                                            ].size
                                        )
                                    ],
                                )
                            )
                            print(elem)
                            print(states_to_add)
                        states_2 = np.concatenate((states_2, states_to_add), axis=1)
                        print("States to add: ", states_2.shape)
                states = np.concatenate((states, states_2))
                print("Full states: ", states.shape)
            else:
                states = np.concatenate((states, data[idx2]["states"]))

            action_value = np.concatenate((action_value, data[idx2]["action_value"]))

    return train_test_split(states, action_value, test_size=0.1, random_state=42)


def load_data(data_directory):
    """
    Loads the data to train with.
    :param data_directory: Data directory to load data from.
    :return: train-test split of training data.
    """
    data = []

    # Load the data
    for count, filename in enumerate(
        sorted(
            Path(data_directory + "/Training Data/").iterdir(), key=os.path.getmtime
        ),
        start=0,
    ):
        print(count, filename)
        # if filename.startswith('AlphaZero'):
        data.append(np.load(filename, allow_pickle=True))

    for idx in range(0, len(data)):
        data[idx] = data[idx].item()
        print(data[idx])

    states = None

    # Load the data
    for idx2 in range(0, len(data)):
        # print(data[idx2]['action_value'][1,:])
        if all(act_val < 0.1 for act_val in data[idx2]["action_value"][1, :]):
            # print(data[idx2]['action_value'][1,:])
            continue
        elif states is None:
            states = data[idx2]["states"]
            action_value = data[idx2]["action_value"]
        else:
            states = np.concatenate((states, data[idx2]["states"]))
            action_value = np.concatenate((action_value, data[idx2]["action_value"]))

    # split the training and test data, shuffle around the data to decorrelate
    return train_test_split(states, action_value, test_size=0.1, random_state=42)


def run_experiment(data_directory, parameters, modified_states=[], batch_sizes=None):
    """
    Runs a hyperparameter search over neural network hyperparameters
    :param data_directory: Data directory to load data from and save networks, \
    training plots.
    :param parameters: dictionary of network hyperparameters.
    :param modified_states (optional): Modified states
    :return:
    """
    net_sizes = parameters["net_sizes"]
    layers = parameters["layers"]
    activations = parameters["activations"]
    dropouts = parameters["dropouts"]
    alphas = parameters["alphas"]
    epoch_num = parameters["epoch_num"]

    (
        states_train,
        states_test,
        action_value_train,
        action_value_test,
    ) = load_and_modify_data(data_directory, modified_states)

    master_data = []
    num_states = len(states_train[0, :])

    if batch_sizes is None:
        batch_sizes = [len(states_train)]

    net_idx = 0

    # Search loop
    for hidden_layer_num in layers:
        for net_size in net_sizes:
            for activation in activations:
                for dropout in dropouts:
                    for batch_size in batch_sizes:
                        for epoch in epoch_num:
                            for alpha_idx, alpha in enumerate(alphas):
                                # Break out of the loop if on second alpha for tanh
                                if alpha_idx > 0 and activation == "tanh":
                                    break
                                print(
                                    "Hyperparam combination: ",
                                    [
                                        hidden_layer_num,
                                        net_size,
                                        activation,
                                        dropout,
                                        alpha,
                                    ],
                                )

                                model = create_model(
                                    hidden_layer_num,
                                    net_size,
                                    activation,
                                    num_states,
                                    dropout,
                                    alpha,
                                    len(action_value_test[0, :]),
                                )

                                history = model.fit(
                                    states_train,
                                    action_value_train,
                                    epochs=epoch,
                                    batch_size=batch_size,
                                    verbose=2,
                                    validation_data=(states_test, action_value_test),
                                )
                                os.makedirs(
                                    data_directory + "/Network History", exist_ok=True
                                )
                                np.save(
                                    data_directory
                                    + "/Network History/"
                                    + "network_hist_"
                                    + str(net_idx)
                                    + ".npy",
                                    history.history,
                                )

                                os.makedirs(
                                    data_directory + "/Networks/", exist_ok=True
                                )
                                model.save(
                                    data_directory
                                    + "/Networks/"
                                    + "network_"
                                    + str(net_idx)
                                )

                                prediction = model.predict(states_test)

                                final_mse_mean = np.average(
                                    [
                                        mean_squared_error(
                                            action_value_test[:, i], prediction[:, i]
                                        )
                                        for i in range(0, len(action_value_test[0, :]))
                                    ]
                                )
                                final_mae_mean = np.average(
                                    [
                                        mean_absolute_error(
                                            action_value_test[:, i], prediction[:, i]
                                        )
                                        for i in range(0, len(action_value_test[0, :]))
                                    ]
                                )

                                master_data.append(
                                    {
                                        "file_name": "network_" + str(net_idx),
                                        "hidden_layer_num": hidden_layer_num,
                                        "net_size": net_size,
                                        "activation": activation,
                                        "dropout": dropout,
                                        "alpha": alpha,
                                        "batch_size": batch_size,
                                        "epochs": epoch,
                                        "final_mse_mean": final_mse_mean,
                                        "final_mae_mean": final_mae_mean,
                                    }
                                )

                                os.makedirs(
                                    data_directory + "/Training Plots/", exist_ok=True
                                )

                                plt.rc("xtick", labelsize=14)
                                plt.rc("ytick", labelsize=14)

                                plt.figure(figsize=(8, 6))
                                plt.plot(
                                    history.history["mean_squared_error"],
                                    label="Training Set",
                                )
                                plt.plot(
                                    history.history["val_mean_squared_error"],
                                    label="Validation Set",
                                )
                                plt.legend(loc="upper right", fontsize=16)
                                plt.grid(which="both", linestyle="dotted")
                                plt.minorticks_on
                                plt.xlabel("Epochs", fontsize=16)
                                plt.ylabel("Mean Squared Error", fontsize=16)
                                plt.yticks(fontsize=14)
                                plt.xticks(fontsize=14)
                                plt.yscale("log")
                                plt.savefig(
                                    data_directory
                                    + "/Training Plots"
                                    + "/network_"
                                    + str(net_idx)
                                    + "_mse.pdf",
                                    dpi=300,
                                    format="pdf",
                                )

                                plt.figure(figsize=(8, 6))
                                plt.plot(
                                    history.history["mean_absolute_error"],
                                    label="Training Set",
                                )
                                plt.plot(
                                    history.history["val_mean_absolute_error"],
                                    label="Validation Set",
                                )
                                plt.xlabel("Epochs", fontsize=16)
                                plt.ylabel("Mean Absolute Error", fontsize=16)
                                plt.yticks(fontsize=14)
                                plt.xticks(fontsize=14)
                                plt.legend(loc="upper right", fontsize=16)
                                plt.grid(linestyle="dotted")
                                plt.yscale("log")
                                plt.savefig(
                                    data_directory
                                    + "/Training Plots"
                                    + "/network_"
                                    + str(net_idx)
                                    + "_mae.pdf",
                                    dpi=300,
                                    format="pdf",
                                )

                                net_idx += 1

    np.save(data_directory + "/network_results.npy", master_data)
