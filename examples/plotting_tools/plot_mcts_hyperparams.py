import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
from pathlib import Path
import os

SEP = os.path.sep

plt.style.use("seaborn-colorblind")
matplotlib.rcParams["lines.linewidth"] = 2


def plot_hyperparam_reward(
    num_sim_values, c_values, master_data_dict, lb=None, ub=None, num_samples=30
):
    return_fig = plt.figure(figsize=(8, 6))

    data = []
    cis = []
    data_type = "reward_sum"

    # Loop through c values
    for c in c_values:
        c_temp = []
        ci_temp = []
        # Loop through num_sim_values
        for num_sim in num_sim_values:
            # num_sim_temp = []
            vals_for_averaging = []
            # Find all initial conditions in our data set with these values
            for idx in range(0, num_samples):
                if (c, num_sim, idx) in master_data_dict.keys():
                    vals_for_averaging.append(
                        master_data_dict[(c, num_sim, idx)][data_type]
                    )
            c_temp.append(np.average(np.array(vals_for_averaging)))
            ci_temp.append(
                list(
                    st.t.interval(
                        alpha=0.95,
                        df=len(vals_for_averaging) - 1,
                        loc=np.mean(vals_for_averaging),
                        scale=st.sem(vals_for_averaging),
                    )
                )
            )
            print("mean: ", np.average(np.array(vals_for_averaging)))
            print("std: ", np.std(np.array(vals_for_averaging)))
        data.append(c_temp)
        cis.append(ci_temp)

    print(np.array(cis).shape)

    for idx, point in enumerate(data):
        plt.plot(
            num_sim_values,
            point,
            label=c_values[idx],
        )
        plt.fill_between(
            num_sim_values,
            np.array(cis)[idx, :, 0],
            np.array(cis)[idx, :, 1],
            alpha=0.1,
        )

    legend = plt.legend(
        title="Exploration Constant",
        loc="upper center",
        ncol=3,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.22),
    )
    # plt.setp(legend.get_title(), fontsize=16)
    # fig.subplots_adjust(top=0.25)
    plt.xlabel("Simulations-Per-Step", fontsize=16)
    plt.ylabel("Average Reward", fontsize=16)
    plt.grid(which="both", linestyle="--")
    plt.xticks(num_sim_values, fontsize=14)
    # plt.yticks([0, 100, 200, 300, 400, 500], fontsize=14)
    if lb is not None:
        plt.ylim([lb, ub])
    plt.yticks(fontsize=14)
    plt.xlim([num_sim_values[0], num_sim_values[-1]])

    return return_fig


def plot_hyperparam_failures(num_sim_values, c_values, master_data_dict):
    return_fig = plt.figure(figsize=(8, 6))

    data = []
    data_type = "sim_length"

    # Loop through c values
    for c in c_values:
        c_temp = []
        # Loop through num_sim_values
        for num_sim in num_sim_values:
            # num_sim_temp = []
            vals_for_averaging = []
            # Find all initial conditions in our data set with these values
            for idx in range(0, 10):
                vals_for_averaging.append(
                    master_data_dict[(c, num_sim, idx)][data_type] >= 270
                )
            c_temp.append(100 * np.average(np.array(vals_for_averaging)))
        data.append(c_temp)

    for idx, point in enumerate(data):
        plt.plot(num_sim_values, point, label=c_values[idx])

    legend = plt.legend(
        title="Exploration Constant",
        loc="upper center",
        ncol=3,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.3),
    )
    plt.setp(legend.get_title(), fontsize=16)
    # fig.subplots_adjust(top=0.25)
    plt.xlabel("Simulations-Per-Step", fontsize=16)
    plt.ylabel("Success Rate (%)", fontsize=16)
    plt.grid(which="both", linestyle="--")
    plt.xticks(num_sim_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([num_sim_values[0], num_sim_values[-1]])

    return return_fig


def plot_hyperparam_downlink_utilization(num_sim_values, c_values, master_data_dict):
    return_fig = plt.figure(figsize=(8, 6))

    data = []
    cis = []

    # Loop through c values
    for c in c_values:
        c_temp = []
        ci_temp = []
        # Loop through num_sim_values
        for num_sim in num_sim_values:
            # num_sim_temp = []
            vals_for_averaging = []
            # Find all initial conditions in our data set with these values
            for idx in range(0, 10):
                utilized_access = master_data_dict[(c, num_sim, idx)]["utilized_access"]
                total_access = master_data_dict[(c, num_sim, idx)]["total_access"]
                if total_access > 0:
                    vals_for_averaging.append(100 * utilized_access / total_access)
                else:
                    vals_for_averaging.append(0)
            c_temp.append(np.average(np.array(vals_for_averaging)))
            ci_temp.append(
                list(
                    st.t.interval(
                        alpha=0.95,
                        df=len(vals_for_averaging) - 1,
                        loc=np.mean(vals_for_averaging),
                        scale=st.sem(vals_for_averaging),
                    )
                )
            )
        data.append(c_temp)
        cis.append(ci_temp)

    for idx, point in enumerate(data):
        plt.plot(num_sim_values, point, label=c_values[idx])
        plt.fill_between(
            num_sim_values,
            np.array(cis)[idx, :, 0],
            np.array(cis)[idx, :, 1],
            alpha=0.1,
        )

    # legend = plt.legend(title='Exploration Constant', loc='upper center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, 1.3))
    # plt.setp(legend.get_title(), fontsize=16)
    # return_fig.subplots_adjust(top=0.9)
    plt.xlabel("Simulations-Per-Step", fontsize=16)
    plt.ylabel("Downlink Utilization (%)", fontsize=16)
    plt.grid(which="both", linestyle="--")
    plt.xticks(num_sim_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([num_sim_values[0], num_sim_values[-1]])
    plt.ylim([0, 100])

    return return_fig


def plot_exec_time(num_sim_values, c_values, master_data_dict):
    return_fig = plt.figure(figsize=(8, 6))

    data = []

    # Loop through c values
    for c in c_values:
        c_temp = []
        # Loop through num_sim_values
        for num_sim in num_sim_values:
            # num_sim_temp = []
            vals_for_averaging = []
            # Find all initial conditions in our data set with these values
            for idx in range(0, 10):
                vals_for_averaging.append(
                    master_data_dict[(c, num_sim, idx)]["exec_time"]
                )
            c_temp.append(np.average(np.array(vals_for_averaging)))
        data.append(c_temp)
        print(data)

    for idx, point in enumerate(data):
        plt.plot(num_sim_values, point, label=c_values[idx])

    legend = plt.legend(
        title="Exploration Constant",
        loc="upper center",
        ncol=3,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.3),
    )
    plt.setp(legend.get_title(), fontsize=16)
    # return_fig.subplots_adjust(top=0.9)
    plt.xlabel("Simulations-Per-Step", fontsize=16)
    plt.ylabel("Execution Time (s)", fontsize=16)
    plt.grid(which="both", linestyle="--")
    plt.xticks(num_sim_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([num_sim_values[0], num_sim_values[-1]])

    return return_fig


def construct_master_data_dict(results_directories):
    master_data_array = np.array([])

    for directory in results_directories:
        data = np.load(directory, allow_pickle=True)
        master_data_array = np.concatenate((master_data_array, data))

    master_data_dict = {}

    for key_val_pair in master_data_array:
        master_data_dict.update(
            {list(key_val_pair.keys())[0]: list(key_val_pair.values())[0]}
        )

    return master_data_dict


if __name__ == "__main__":
    base_directory = (
        ".."
        + SEP
        + ".."
        + SEP
        + "bsk_rl"
        + SEP
        + "results"
        + SEP
        + "mcts - old"
        + SEP
        + "uct_hyperparameter_search"
        + SEP
        + "random_rollout"
        + SEP
    )
    base_directory = str(Path(__file__).parent.resolve() / base_directory)

    results_directories = [
        base_directory + "/results_5_to_50_sims.npy",
        base_directory + "/results_75_sims.npy",
    ]

    master_data_dict = construct_master_data_dict(results_directories)

    num_sim_values = [5, 10, 25, 50, 75]
    c_values = [0, 100, 200, 300, 400, 500]

    fig = plot_hyperparam_reward(num_sim_values, c_values, master_data_dict)
    plt.savefig(
        base_directory + "reward_sum_random.pdf",
        dpi=300,
        pad_inches=0.1,
        bbox_inches="tight",
    )

    fig2 = plot_hyperparam_downlink_utilization(
        num_sim_values, c_values, master_data_dict
    )
    plt.savefig(
        base_directory + "downlink_utilization_random.pdf",
        dpi=300,
        pad_inches=0.1,
        bbox_inches="tight",
    )

    fig3 = plot_hyperparam_failures(num_sim_values, c_values, master_data_dict)
    plt.savefig(
        base_directory + "resource_management_success_rate_random.pdf",
        dpi=300,
        pad_inches=0.1,
        bbox_inches="tight",
    )

    fig4 = plot_exec_time(num_sim_values, c_values, master_data_dict)
    plt.savefig(
        base_directory + "exec_time.pdf",
        dpi=300,
        pad_inches=0.1,
        bbox_inches="tight",
    )

    plt.show()
