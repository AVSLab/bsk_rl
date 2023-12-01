import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from .plot_dqn_hyperparams import concatenate_results

SEP = os.path.sep


def plot_n_steps_lr(results, alpha=None, dropout=None):
    # Initialize batches and epochs
    n_steps = []
    lrs = []
    reward = []

    # Collect the independent variables and reward
    for key, value in results.items():
        if (
            value["policy_kwargs"]["net_arch"]["alpha"] == alpha
            and value["policy_kwargs"]["net_arch"]["dropout"] == dropout
        ):
            n_steps.append(value["n_steps"])
            lrs.append(value["lr"])
            reward.append(np.maximum(0, np.average(value["validation_reward"])))

    # Grab the unique batches and unique epochs
    unique_steps = np.unique(np.array(n_steps)).tolist()
    unique_steps.sort()

    unique_lrs = np.unique(np.array(lrs)).tolist()
    unique_lrs.sort()

    # Define the deltas
    dx = np.ones_like(n_steps) * 1
    dy = np.ones_like(lrs) * 1

    # Initialize x and y
    x = []
    y = []

    # Construct x and y
    for n_step, lr in zip(n_steps, lrs):
        x.append(unique_steps.index(n_step))
        y.append(unique_lrs.index(lr))

    # Create the colormap
    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(reward))

    # Plot the figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    ax.bar3d(x, y, np.zeros(len(reward)), dx, dy, reward, color=colors)
    ax.set_xlabel("Steps Before Update", labelpad=16, fontsize=16)
    ax.set_ylabel("Learning Rate", labelpad=12, fontsize=16)
    ax.set_zlabel("Average Reward", labelpad=10, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.set_xticks(np.unique(x) + 0.5, unique_steps)
    ax.set_yticks(np.unique(y) + 0.5, unique_lrs)
    ax.set_zlim([0, 1.0])

    return fig, ax


def plot_size(results, alpha=None, dropout=None):
    # Initialize batches and epochs
    layers = []
    nodes = []
    reward = []

    # Collect the independent variables and reward
    for key, value in results.items():
        if (
            value["policy_kwargs"]["net_arch"]["alpha"] == alpha
            and value["policy_kwargs"]["net_arch"]["dropout"] == dropout
        ):
            nodes.append(value["policy_kwargs"]["net_arch"]["width"])
            layers.append(value["policy_kwargs"]["net_arch"]["depth"])
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
    if os.path.isfile(agent_dir + "/results.npy"):
        results = np.load(agent_dir + "/results.npy", allow_pickle=True)
        results = results.item()
    else:
        results = concatenate_results(
            agent_dir, dict_entries=["policy_kwargs", "lr", "n_steps"]
        )

    alphas = []
    dropouts = []

    for key, value in results.items():
        print("---------------------------------------------------------------------")
        print("Network: ", key)
        print("Network Params: ", value["policy_kwargs"])
        print("LR: ", value["lr"])
        print("N Steps: ", value["n_steps"])
        print("Reward: ", np.average(value["validation_reward"]))
        alphas.append(value["policy_kwargs"]["net_arch"]["alpha"])
        if value["policy_kwargs"]["net_arch"]["dropout"] is not None:
            dropouts.append(value["policy_kwargs"]["net_arch"]["dropout"])

    alphas = np.unique(np.array(alphas)).tolist()

    dropouts = np.unique(np.array(dropouts)).tolist()
    if dropouts == []:
        dropouts.append(None)

    idx = 0
    for alpha in alphas:
        for dropout in dropouts:
            fig, ax = plot_n_steps_lr(results, alpha=alpha, dropout=dropout)
            drop_ident = "None" if dropout is None else str(int(100 * dropout))
            plt.savefig(
                agent_dir
                + "/n_steps_lr_bar_plot_"
                + drop_ident
                + "_drop_"
                + str(int(100 * alpha))
                + "_alpha.pdf",
                dpi=300,
                pad_inches=0.1,
                bbox_inches="tight",
            )
            idx += 1

    idx = 0
    for alpha in alphas:
        for dropout in dropouts:
            fig, ax = plot_size(results, alpha=alpha, dropout=dropout)
            drop_ident = "None" if dropout is None else str(int(100 * dropout))
            plt.savefig(
                agent_dir
                + "/size_bar_plot_"
                + drop_ident
                + "_drop_"
                + str(int(100 * alpha))
                + "_alpha.pdf",
                dpi=300,
                pad_inches=0.1,
                bbox_inches="tight",
            )
            idx += 1

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
        + "A2C"
        + SEP
        + "MultiSensorEOS-v0"
        + SEP
        + "A2C Test"
        + SEP
    )
    agent_dir = str(Path(__file__).parent.resolve() / agent_dir)

    plot_results(agent_dir)
