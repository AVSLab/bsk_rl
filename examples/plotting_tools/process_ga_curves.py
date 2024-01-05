import os
import pickle
from pathlib import Path

from deap import base, creator
from matplotlib import pyplot as plt

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

SEP = os.path.sep


def process_ga_curve(run_dir, env_name):
    """
    Generates plots of GA convergence, performance, and final population vector behavior
    :param checkpoint_names: list of checkpoint names, assumed to be pickle files.
    :return:
    """
    #   Pull data out of the pickles
    gen_list = []
    min_reward = []
    max_reward = []
    mean_reward = []

    # Find all of pkl files
    n_gens = len([name for name in os.listdir(run_dir) if name.endswith(".pkl")])

    checkpoint_names = [run_dir + env_name + f"_{gen}.pkl" for gen in range(1, n_gens)]

    for name in checkpoint_names:
        with open(name, "rb") as file:
            result = pickle.load(file)
        gen_list.append(result["generation"])
        tmp_log = result["logbook"]
        mean_reward.append(tmp_log[-1]["avg"])
        min_reward.append(tmp_log[-1]["min"])
        max_reward.append(tmp_log[-1]["max"])

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(gen_list, mean_reward, label="Mean Reward")
    plt.plot(gen_list, max_reward, label="Max Reward")
    plt.plot(gen_list, min_reward, label="Min Reward", alpha=0.4)
    plt.grid()
    plt.ylabel("Reward", fontsize=16)
    plt.xlabel("Generation", fontsize=16)
    plt.ylim([-1, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(
        run_dir + env_name + "_curve.pdf",
        dpi=300,
        pad_inches=0.1,
        bbox_inches="tight",
        transparent=True,
    )


if __name__ == "__main__":
    # Define the directory
    ga_dir = (
        ".."
        + SEP
        + ".."
        + SEP
        + "bsk_rl"
        + SEP
        + "results"
        + SEP
        + "GA"
        + SEP
        + "MultiSensorEOS-v0"
        + SEP
        + "GA Test"
        + SEP
    )
    ga_dir = str(Path(__file__).parent.resolve() / ga_dir)

    # Make a list of every folder in the directory
    folders = os.listdir(ga_dir)

    # Loop through each folder
    for folder in folders:
        # Check if the folder is a directory
        if os.path.isdir(ga_dir + SEP + folder):
            process_ga_curve(ga_dir + SEP + folder + SEP, "MultiSensorEOS-v0")
