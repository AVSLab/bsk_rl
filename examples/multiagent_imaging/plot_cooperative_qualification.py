"""Two constructed native counterexamples; no learned-policy performance claim."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence

    def science(case):
        record = json.loads((root / case / "fixture.json").read_text())
        return sum(
            sum(step["owned_science_reward"].values()) for step in record["steps"]
        )

    summary = json.loads((root / "qualification.json").read_text())
    groups = [
        (
            "Original unfavorable opportunity",
            [
                science("negative_blocked"),
                science("negative_delivered"),
                science("negative_no_comm_image"),
            ],
        ),
        (
            "Declared urgent-request opportunity",
            [
                science("blocked"),
                science("delivered"),
                summary["best_alternative_science"],
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, (title, values) in zip(axes, groups):
        bars = ax.bar(
            ["Packet lost", "Packet delivered", "Best tested\nno-communication"],
            values,
            color=["#94a3b8", "#2563eb", "#d97706"],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_ylabel("Unshaped science value (duplicate fees excluded)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Constructed native fixtures · 2400 s · fixed private controllers", fontsize=12
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
