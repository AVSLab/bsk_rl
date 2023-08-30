import os
import time
from multiprocessing import Manager, Process
from pathlib import Path

import gymnasium as gym
import numpy as np

from bsk_rl.envs.agile_eos.gym_env import AgileEOS  # noqa: F401; needed for gym

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SEP = os.path.sep


def modify_observation(obs, info, modified_states):
    """This function modifies the observation from the environment if this is required.
    :param obs: environment observation
    :param info: info that may be add to the observation
    :param modified states: a list of state indices to keep and dictionary keys
    :return obs: modified observation
    """
    # Loop through each element in modified_states
    for idx3, elem in enumerate(modified_states):
        # If its the first element, this is the array of indices to keep
        if idx3 == 0:
            obs = obs[elem]
        # If it's not the first element, it's a dictionary key
        else:
            obs_to_add = info[elem].reshape(info[elem].size)
            obs = np.concatenate((obs, obs_to_add))

    return obs


def run_simulator(
    network,
    reward_specific,
    success_specific,
    downlink_util_specific,
    exec_time_specific,
    imaged_specific,
    downlinked_specific,
    modified_states,
    ic,
):
    """
    This function is utilized to a.) load the network, b.) run the network in the
    environment, and c.) update performance metrics
    :param network: File name of neural network
    :param reward_specific: List of reward metrics
    :param success_specific: List of success metrics
    :param downlink_util_specific: List of downlink utilization metrics
    :param exec_time_specific: List of execution time metrics
    :param imaged_specific: List of image metric
    :param downlinked_specific: List of downlinked metric
    :param modified_states: List containing indices of states to keep and entries of
    the info dictionary
    :param ic: initial conditions
    """

    # Import keras here
    import tensorflow.keras as keras

    # Create env
    env = gym.make("AgileEOS-v0")

    # Set num steps, max length
    tFinal = float(270)
    numSteps = int(tFinal / (env.step_duration / 60.0))
    env.max_length = tFinal
    env.max_steps = numSteps

    # Initialize action history
    actHist = []

    # Total reward
    reward_sum = 0.0
    exec_time_sum = 0.0

    # Reset the environment to initial conditions
    ob, info = env.reset(options={"initial_conditions": ic})

    # Load model
    model = keras.models.load_model(network, custom_objects=None, compile=True)

    # Loop through each time step
    for ind in range(0, numSteps):
        # Start the clock
        start = time.perf_counter()

        # Select action
        action_value = model.predict(ob.reshape(1, -1))

        # End the clock
        end = time.perf_counter()

        # Add to exec_time
        exec_time_sum += end - start

        # Select new action
        act = np.argmax(action_value)

        # Append last action to action list
        actHist.append(act)

        # Print out some information on the action and environment step
        print("Actual Action Taken: ", act)
        print("Real Environment Step: ", ind)

        # Take the step in the environment
        ob, reward, episode_over, _, info = env.step(act)

        # Print reward
        print("Reward: ", reward)

        # Sum the reward, add to rewards array
        reward_sum = reward_sum + reward

        # If the episode is over, end the simulation
        if episode_over:
            print("episode over")
            break

    # Pull the total access
    total_access = env.simulator.total_access

    # Pull the utilized access
    utilized_access = env.simulator.utilized_access

    # Update the performance metrics
    reward_specific.append(reward_sum)
    downlink_util_specific.append(100 * utilized_access / total_access)
    exec_time_specific.append(exec_time_sum)
    if env.simulator.simTime / 60 >= 270 and reward >= -0.00001:
        success_specific.append(1)
    else:
        success_specific.append(0)

    # Update the number of imaged and downlinked performance metrics
    num_imaged = 0
    num_downlinked = 0
    for idx in range(len(env.simulator.imaged_targets)):
        if env.simulator.imaged_targets[idx] >= 1.0:
            num_imaged += 1

        if env.simulator.downlinked_targets[idx] >= 1.0:
            num_downlinked += 1
    imaged_specific.append(num_imaged)
    downlinked_specific.append(num_downlinked)


def run_experiment(data_directory, ident, initial_conditions_list, N, batch_num):
    # Load the network history
    net_hist = np.load(data_directory + "/network_results.npy", allow_pickle=True)
    # net_hist = net_hist.item()
    print(net_hist)

    # Initialize lists for each metric
    reward_master = []
    success_master = []
    downlink_util_master = []
    exec_time_master = []
    imaged_master = []
    downlinked_master = []

    # Create master dictionary
    data_master = {}

    # Initialize manager for creating lists
    manager = Manager()

    # Initialize a dictionary for each model
    models = {}

    # Initialize another results dictionary
    results = {}

    # Load the networks
    for count, filename in enumerate(
        sorted(Path(data_directory + "/Networks/").iterdir(), key=os.path.getmtime),
        start=0,
    ):
        print(str(filename))
        if "network" in str(filename):
            print(count, filename)
            models.update({count: filename})

    # Loop through each model
    for key, model in models.items():
        # Initialize lists for each metric using the manager
        reward_specific = manager.list()
        success_specific = manager.list()
        downlink_util_specific = manager.list()
        exec_time_specific = manager.list()
        imaged_specific = manager.list()
        downlinked_specific = manager.list()

        # Set the initial condition index
        ic_idx = 0

        # Loop through the batches
        for j in range(batch_num):
            # Initialize the lists of processes
            processes = []

            # Loop through N
            for idx in range(0, N):
                # Grab the initial condition
                ic = initial_conditions_list[str(ic_idx)]
                # ic = env.simulator.initial_conditions
                ic_idx += 1
                # Append process to processes
                processes.append(
                    Process(
                        target=run_simulator,
                        args=(
                            model,
                            reward_specific,
                            success_specific,
                            downlink_util_specific,
                            exec_time_specific,
                            imaged_specific,
                            downlinked_specific,
                            [],
                            ic,
                        ),
                    )
                )
            # Start each process
            for idx in range(0, N):
                processes[idx].start()
            # Join each process
            for process in processes:
                process.join()

        # Append the specific data to the master metrics
        reward_master.append(list(reward_specific))
        imaged_master.append(list(imaged_specific))
        downlinked_master.append(list(downlinked_specific))
        downlink_util_master.append(list(downlink_util_specific))
        exec_time_master.append(list(exec_time_specific))
        success_master.append(100 * np.average(success_specific))

        # Update the master data dictionary with the metrics
        data_master.update(
            {
                key: {
                    "reward": reward_master[-1],
                    "downlink": downlink_util_master[-1],
                    "exec_time": exec_time_master[-1],
                    "success": success_master[-1],
                    "downlinked": downlinked_master[-1],
                    "imaged": imaged_master[-1],
                }
            }
        )

        f_name = "network_" + str(key)
        policy_kwargs = dict(
            activation_fn=net_hist[key]["activation"],
            net_arch={
                "width": net_hist[key]["net_size"],
                "depth": net_hist[key]["hidden_layer_num"],
                "dropout": net_hist[key]["dropout"],
                "alpha": net_hist[key]["alpha"],
            },
        )

        # Update the master results
        results.update(
            {
                f_name: {
                    "validation_reward": reward_master[-1],
                    "policy_kwargs": policy_kwargs,
                    "batch_size": net_hist[key]["batch_size"],
                    "n_epochs": net_hist[key]["epochs"],
                }
            }
        )

    # Print the averages of each metric
    print("Averages: ")
    for key, value in data_master.items():
        print(
            key,
            np.average(data_master[key]["reward"]),
            np.average(data_master[key]["downlink"]),
            np.average(data_master[key]["exec_time"]),
        )

    # Uncomment to save data
    np.save(data_directory + "/validation_results_" + ident + ".npy", data_master)
    np.save(data_directory + "/bar_plot_validation_results_" + ident + ".npy", results)


if __name__ == "__main__":
    """This script is used to validate and benchmark trained networks for the
    multiTgtEarthEnvironment in terms of average reward, downlink utilization, success,
    and execution time. This implementation does utilize multi-processing. Before
    running this script, ensure that agents have been trained and placed in the
    'data_directory.' Furthermore, be sure to specify a dictionary of initial
    conditions.

    A bar plot of these validation results can be generated using the plotting tools
    in `utilities/plotting_tools/`
    """

    # Set the data directory
    data_directory = (
        ".."
        + SEP
        + ".."
        + SEP
        + "bsk_rl"
        + SEP
        + "results"
        + SEP
        + "mcts"
        + SEP
        + "AgileEOS"
        + SEP
        + "MCTS-Train Test"
        + SEP
    )

    data_directory = str(Path(__file__).parent.resolve() / data_directory)

    # Set the number of simulations and number of batches
    N = 10
    batch_num = 1

    # Data identifier
    ident = "benchmark_3_tgt"

    # Load initial conditions
    initial_conditions_list = np.load(
        str(Path(__file__).parent.resolve() / "initial_conditions.npy"),
        allow_pickle=True,
    )
    initial_conditions_list = initial_conditions_list.item()

    # Run the experiment
    run_experiment(data_directory, ident, initial_conditions_list, N, batch_num)
