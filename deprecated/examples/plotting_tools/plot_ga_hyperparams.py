import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

SEP = os.path.sep


def concatenate_results(ga_dir, dict_entries=["n_gens", "gen_size"]):
    # Find all of the files that start with results_
    full_results = {}

    for file in os.listdir(ga_dir):
        if file.startswith("results_"):
            # Load the results
            result = np.load(ga_dir + SEP + file, allow_pickle=True)
            result = result.item()

            # Add the results to the dictionary
            full_results.update(result)

    # Count the number of unique hyperparameters
    results = {}
    hp_idx = 0

    for key, value in full_results.items():
        dict_values = []
        for dict_entry in dict_entries:
            if dict_entry not in value:
                print("Error: ", dict_entry, " not in ", key)
                return
            dict_values.append(value[dict_entry])

        # Check if the hyperparameters have been used before
        # If the results dictionary is empty
        if not results:
            results[hp_idx] = value
            results[hp_idx]["max_reward"] = [results[hp_idx]["max_reward"]]
            hp_idx += 1
        else:
            found_match = False
            for key2, value2 in results.items():
                if all(
                    [
                        dict_value == value2[dict_entry]
                        for dict_value, dict_entry in zip(dict_values, dict_entries)
                    ]
                ):
                    value2["max_reward"] += [value["max_reward"]]
                    found_match = True
            if not found_match:
                results[hp_idx] = value
                results[hp_idx]["max_reward"] = [results[hp_idx]["max_reward"]]
                hp_idx += 1

    return results


def plot_gen_pop(results):
    # Initialize batches and epochs
    generations = []
    populations = []
    reward = []

    # Collect the independent variables and reward
    for key, value in results.items():
        generations.append(value["n_gens"])
        populations.append(value["gen_size"])
        reward.append(np.average(value["max_reward"]))

    # Check if the maximum reward is greater than 1, if so, set the z-limit to the
    # maximum reward
    if max(reward) > 1:
        z_lim = 1.8
    else:
        z_lim = 1

    # Grab the unique batches and unique epochs
    unique_generations = np.unique(np.array(generations)).tolist()
    unique_generations.sort()

    unique_populations = np.unique(np.array(populations)).tolist()
    unique_populations.sort()

    # Define the deltas
    dx = np.ones_like(generations) * 1
    dy = np.ones_like(populations) * 1

    # Initialize x and y
    x = []
    y = []

    # Construct x and y
    for gen, pop in zip(generations, populations):
        x.append(unique_generations.index(gen))
        y.append(unique_populations.index(pop))

    # Create the colormap
    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=z_lim)
    colors = cmap(norm(reward))

    # Plot the figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    ax.bar3d(x, y, np.zeros(len(reward)), dx, dy, reward, color=colors)
    ax.set_xlabel("Generations", labelpad=16, fontsize=16)
    ax.set_ylabel("Population Size", labelpad=12, fontsize=16)
    ax.set_zlabel("Average Maximum Reward", labelpad=10, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.set_xticks(np.unique(x) + 0.5, unique_generations)
    ax.set_yticks(np.unique(y) + 0.5, unique_populations)
    ax.set_zlim([0, z_lim])

    return fig, ax


def plot_results(ga_dir):
    # Check if the results.npy file exists
    if os.path.isfile(ga_dir + "results.npy"):
        results = np.load(ga_dir + "/results.npy", allow_pickle=True)
        results = results.item()
    else:
        results = concatenate_results(ga_dir)

    # Loop through the results, report the gens, pop size, and reward
    for key, value in results.items():
        print(
            "Gens: ",
            value["n_gens"],
            " Pop Size: ",
            value["gen_size"],
            " Reward: ",
            np.average(value["max_reward"]),
        )

    fig1, ax1 = plot_gen_pop(results)
    plt.savefig(
        ga_dir + "/hp_bar_plot.pdf",
        dpi=300,
        pad_inches=0.1,
        bbox_inches="tight",
    )

    plt.show()


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
    plot_results(ga_dir)
