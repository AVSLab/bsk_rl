import csv
import os
from pathlib import Path

from matplotlib import pyplot as plt

SEP = os.path.sep


def process_monitor_reward(subfolder):
    # Process all of the monitors
    monitor_reward = []
    monitor_ep_len = []
    monitor_folders = [
        name for name in os.listdir(subfolder + SEP + "logs") if "monitor_" in name
    ]
    for idx in range(0, len(monitor_folders)):
        with open(
            subfolder + SEP + "logs" + SEP + "monitor_" + str(idx) + ".csv", mode="r"
        ) as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=["r", "l", "t"])
            line_count = 0
            for row in csv_reader:
                if line_count == 0 or line_count == 1:
                    line_count += 1
                else:
                    monitor_reward.append(float(row["r"]))
                    monitor_ep_len.append(float(row["l"]))

    reward_avg_x, reward_avg_y = moving_avg(50, monitor_reward)
    len_avg_x, len_avg_y = moving_avg(50, monitor_ep_len)

    return reward_avg_x, reward_avg_y, len_avg_x, len_avg_y


def moving_avg(N, data):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    return [i + N for i in range(0, len(moving_aves))], moving_aves


def plot_reward_curve(steps, reward, x_label="Total Timesteps"):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(steps, reward)
    # plt.legend(loc='upper right', fontsize=18)
    plt.grid(which="both", linestyle="dotted")
    # plt.minorticks_on
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Average Reward", fontsize=16)
    # plt.ylim((-1, 1))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    return fig


def plot_loss(steps, loss):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(steps, loss)
    # plt.legend(loc='upper right', fontsize=18)
    plt.grid(which="both", linestyle="dotted")
    # plt.minorticks_on
    plt.xlabel("Total Timesteps", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    # plt.ylim((-0.1, 0.1))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    return fig


def plot_ep_len(steps, ep_len, x_label="Total Timesteps"):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(steps, ep_len)
    # plt.legend(loc='upper right', fontsize=18)
    plt.grid(which="both", linestyle="dotted")
    # plt.minorticks_on
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Average Episode Length", fontsize=16)
    # plt.ylim((0, 90))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    return fig


def plot_results(agent_dir):
    subfolders = [f.path for f in os.scandir(agent_dir) if f.is_dir()]

    for subfolder in subfolders:
        # Process all of the progress csv files
        reward = []
        episode_length = []
        loss = []
        timesteps = []
        with open(
            subfolder + SEP + "logger" + SEP + "progress.csv", mode="r"
        ) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    reward.append(float(row["rollout/ep_rew_mean"]))
                    episode_length.append(float(row["rollout/ep_len_mean"]))
                    if "A2C" not in agent_dir:
                        loss.append(float(row["train/loss"]))
                    timesteps.append(int(row["time/total_timesteps"]))

        if "A2C" not in agent_dir:
            plot_loss(timesteps, loss)
            plt.savefig(subfolder + SEP + "loss.pdf", dpi=300, format="pdf")

        plot_reward_curve(timesteps, reward)
        plt.savefig(subfolder + SEP + "reward.pdf", dpi=300, format="pdf")

        plot_ep_len(timesteps, episode_length)
        plt.savefig(subfolder + SEP + "episode_length.pdf", dpi=300, format="pdf")

        reward_avg_x, reward_avg_y, len_avg_x, len_avg_y = process_monitor_reward(
            subfolder
        )

        plot_reward_curve(reward_avg_x, reward_avg_y, "Episodes")
        plt.savefig(subfolder + SEP + "monitor_reward.pdf", dpi=300, format="pdf")

        plot_ep_len(len_avg_x, len_avg_y, "Episodes")
        plt.savefig(
            subfolder + SEP + "monitor_episode_length.pdf", dpi=300, format="pdf"
        )


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
        + "PPO"
        + SEP
        + "MultiSensorEOS-v0"
        + SEP
        + "PPO Test"
        + SEP
    )
    agent_dir = str(Path(__file__).parent.resolve() / agent_dir)

    plot_results(agent_dir)
