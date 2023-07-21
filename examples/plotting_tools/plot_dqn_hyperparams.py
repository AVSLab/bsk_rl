import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path

SEP = os.path.sep


def concatenate_results(
    agent_dir, dict_entries=["policy_kwargs", "lr", "batch_size", "buffer_size"]
):
    # Find all of the files that start with results_
    full_results = {}

    net_idx = 0
    for file in os.listdir(agent_dir):
        if file.startswith("results_"):
            # Load the results
            result = np.load(agent_dir + SEP + file, allow_pickle=True)
            result = result.item()

            full_results.update(result)

            net_idx += 1

    # Count the number of unique hyperparameters
    results = {}
    net_idx = 0

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
            results["final_network_" + str(net_idx)] = value
            net_idx += 1
        else:
            found_match = False
            for key2, value2 in results.items():
                if all(
                    [
                        dict_value == value2[dict_entry]
                        for dict_value, dict_entry in zip(dict_values, dict_entries)
                    ]
                ):
                    value2["validation_reward"] += value["validation_reward"]
                    found_match = True
            if not found_match:
                results["final_network_" + str(net_idx)] = value
                net_idx += 1

    return results


def plot_batch_buffer(results):
    # Initialize batches and epochs
    batches = []
    epochs = []
    reward = []

    # Collect the independent variables and reward
    for key, value in results.items():
        batches.append(value["batch_size"])
        epochs.append(value["buffer_size"])
        reward.append(np.average(value["validation_reward"]))

    # Grab the unique batches and unique epochs
    unique_batches = np.unique(np.array(batches)).tolist()
    unique_batches.sort()

    unique_epochs = np.unique(np.array(epochs)).tolist()
    unique_epochs.sort()

    # Define the deltas
    dx = np.ones_like(batches) * 1
    dy = np.ones_like(epochs) * 1

    # Initialize x and y
    x = []
    y = []

    # Construct x and y
    for batch, epoch in zip(batches, epochs):
        x.append(unique_batches.index(batch))
        y.append(unique_epochs.index(epoch))

    # Create the colormap
    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(reward))

    # Plot the figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    ax.bar3d(x, y, np.zeros(len(reward)), dx, dy, reward, color=colors)
    ax.set_xlabel("Batch Size", labelpad=16, fontsize=16)
    ax.set_ylabel("Buffer Size", labelpad=12, fontsize=16)
    ax.set_zlabel("Average Reward", labelpad=10, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.set_xticks(np.unique(x) + 0.5, unique_batches)
    ax.set_yticks(np.unique(y) + 0.5, unique_epochs)
    ax.set_zlim([0, 1.0])

    return fig, ax


def plot_size(results):
    # Initialize batches and epochs
    layers = []
    nodes = []
    reward = []

    # Collect the independent variables and reward
    for key, value in results.items():
        nodes.append(value["policy_kwargs"]["net_arch"][0])
        layers.append(len(value["policy_kwargs"]["net_arch"]))
        reward.append(np.maximum(0, np.average(value["validation_reward"])))

    # Grab the unique batches and unique epochs
    unique_nodes = np.unique(np.array(nodes)).tolist()
    unique_nodes.sort()

    unique_layers = np.unique(np.array(layers)).tolist()
    unique_layers.sort()

    # Define the deltas
    dx = np.ones_like(nodes) * 1
    dy = np.ones_like(layers) * 1

    # Initialize x and y
    x = []
    y = []

    # Construct x and y
    for node, layer in zip(nodes, layers):
        x.append(unique_nodes.index(node))
        y.append(unique_layers.index(layer))

    # Create the colormap
    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(reward))

    # Plot the figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    ax.bar3d(x, y, np.zeros(len(reward)), dx, dy, reward, color=colors)
    ax.set_xlabel("Nodes", labelpad=16, fontsize=16)
    ax.set_ylabel("Layers", labelpad=12, fontsize=16)
    ax.set_zlabel("Average Reward", labelpad=10, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.set_xticks(np.unique(x) + 0.5, unique_nodes)
    ax.set_yticks(np.unique(y) + 0.5, unique_layers)
    ax.set_zlim([0, 1.0])

    return fig, ax


def plot_results(agent_dir):
    # Check if the results.npy file exists
    if os.path.isfile(agent_dir + "results.npy"):
        results = np.load(agent_dir + "results.npy", allow_pickle=True)
        results = results.item()
    else:
        results = concatenate_results(agent_dir)

    for key, value in results.items():
        print("---------------------------------------------------------------------")
        print("Network: ", key)
        print("Network Params: ", value["policy_kwargs"])
        print("LR: ", value["lr"])
        print("Batch Size: ", value["batch_size"])
        print("Buffer Size: ", value["buffer_size"])
        print("Reward: ", np.average(value["validation_reward"]))

    fig1, ax1 = plot_batch_buffer(results)
    plt.savefig(
        agent_dir + "/batch_buffer_bar_plot.pdf",
        dpi=300,
        pad_inches=0.1,
        bbox_inches="tight",
    )

    fig2, ax2 = plot_size(results)
    plt.savefig(
        agent_dir + "/size_bar_plot.pdf", dpi=300, pad_inches=0.1, bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    agent_dir = (
        ".."
        + SEP
        + ".."
        + SEP
        + "bsk_rl"
        + SEP
        + "results"
        + SEP
        + "SB3"
        + SEP
        + "DQN"
        + SEP
        + "MultiSensorEOS-v0"
        + SEP
        + "DQN Test"
        + SEP
    )

    agent_dir = str(Path(__file__).parent.resolve() / agent_dir)

    plot_results(agent_dir=agent_dir)
