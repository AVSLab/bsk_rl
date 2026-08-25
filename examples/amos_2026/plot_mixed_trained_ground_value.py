"""Plot mixed-regime-trained delivered ground value across reward weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


ALPHAS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
ALPHA_LABELS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0"]
VALUE_COLUMN = "delivered_ground_value_100d00i_mixed"
PLASMA = plt.get_cmap("plasma")
ALPHA_NORM = Normalize(0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="paired_per_run.csv from the mixed-training comparison campaign",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the vector PDF and high-resolution PNG",
    )
    parser.add_argument(
        "--stem",
        default="training_environment_value_mixed_only",
        help="Output filename stem",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path).sort_values(["alpha", "seed"])
    required = {"alpha", "seed", VALUE_COLUMN}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    observed_alphas = np.sort(runs["alpha"].unique())
    if not np.allclose(observed_alphas, ALPHAS):
        raise ValueError(
            f"Expected alpha sweep {ALPHAS.tolist()}, found {observed_alphas.tolist()}"
        )
    seed_counts = runs.groupby("alpha")["seed"].nunique()
    if not seed_counts.eq(100).all():
        raise ValueError(f"Expected 100 seeds per alpha, found {seed_counts.to_dict()}")
    return runs


def plot(runs: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.1, 4.15), constrained_layout=True)
    rng = np.random.default_rng(20260803)

    for alpha in ALPHAS:
        values = runs.loc[runs["alpha"].eq(alpha), VALUE_COLUMN].to_numpy(
            dtype=float
        )
        color = PLASMA(ALPHA_NORM(alpha))

        ax.boxplot(
            [values],
            positions=[alpha],
            widths=0.040,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            manage_ticks=False,
            zorder=2,
            boxprops={
                "facecolor": color,
                "edgecolor": "0.16",
                "linewidth": 0.9,
                "alpha": 0.72,
            },
            medianprops={"color": "0.08", "linewidth": 1.1},
            whiskerprops={"color": "0.22", "linewidth": 0.8},
            capprops={"color": "0.22", "linewidth": 0.8},
        )

        jitter = rng.uniform(-0.015, 0.015, size=len(values))
        ax.scatter(
            alpha + jitter,
            values,
            s=12,
            marker="o",
            color=color,
            edgecolors="none",
            alpha=0.28,
            zorder=1,
        )
        ax.scatter(
            [alpha],
            [float(np.mean(values))],
            s=46,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=4,
        )

    ax.set_ylim(175, 425)
    ax.set_xlim(-0.055, 1.055)
    ax.set_xticks(ALPHAS, ALPHA_LABELS, rotation=45, ha="right")
    ax.set_xlabel(r"Downlink reward weight, $\alpha$")
    ax.set_ylabel("Delivered ground value")
    ax.grid(axis="y", color="0.86", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", which="major", length=4.5, width=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.15")
        spine.set_linewidth(0.9)

    mixed_handle = Line2D(
        [0],
        [0],
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        color="none",
        linestyle="none",
        markersize=6.5,
        label="Mixed-regime-trained",
    )
    ax.legend(
        handles=[mixed_handle],
        frameon=True,
        edgecolor="0.25",
        loc="lower center",
    )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=ALPHA_NORM, cmap=PLASMA),
        ax=ax,
        pad=0.025,
        fraction=0.055,
    )
    colorbar.set_label(r"Downlink reward weight, $\alpha$")
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    return fig


def main() -> None:
    args = parse_args()
    configure_style()
    runs = load_runs(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot(runs)
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    png_path = args.output_dir / f"{args.stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    means = runs.groupby("alpha")[VALUE_COLUMN].mean()
    print(f"Saved vector PDF: {pdf_path}")
    print(f"Saved preview PNG: {png_path}")
    print("Means by alpha:")
    for alpha, value in means.items():
        print(f"  alpha={alpha:g}: {value:.2f}")


if __name__ == "__main__":
    main()
