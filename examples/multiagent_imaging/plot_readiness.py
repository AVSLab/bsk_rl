"""Diagnostic resource and physical-discount figures; no policy ranking plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_readiness(profile, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
    for sensor, records in profile["resource_history"].items():
        hours = [r["time_s"] / 3600 for r in records]
        axes[0].plot(hours, [r["battery_fraction"] for r in records], label=sensor)
        axes[1].plot(hours, [r["storage_fraction"] for r in records], label=sensor)
    for axis, label in zip(axes, ["Battery fraction", "Storage fraction"]):
        axis.set(ylabel=label, ylim=(-0.02, 1.02))
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[0].set_title("Mission-scale heuristic profile: physical resources, seed 0")
    axes[1].set_xlabel("Simulation time (hours)")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"mission_resources.{extension}", dpi=160)
    plt.close(fig)

    seconds = np.linspace(0, 45000, 1000)
    fig, axis = plt.subplots(figsize=(8, 3), constrained_layout=True)
    axis.plot(
        seconds / 3600,
        np.exp(np.log(0.5) * seconds / 45000),
        label="45,000-second half-life",
    )
    axis.plot(seconds / 3600, 0.999**seconds, label="Previous gamma = 0.999 per second")
    axis.set(
        xlabel="Reward delay (hours)", ylabel="Discount weight", ylim=(-0.02, 1.02)
    )
    axis.legend()
    axis.grid(alpha=0.2)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"physical_discount.{extension}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_readiness(json.loads(args.profile.read_text()), args.output)
