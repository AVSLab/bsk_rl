#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# USER CONFIGURATION
# =========================

# SUMMARY_CSV = Path("results/overall_summary_by_alpha_allPolicies_20260115_201523.csv")
# SUMMARY_CSV = Path("results/overall_summary_by_alpha_allPolicies_20260115_173753.csv")

# or hardcode the exact file:
SUMMARY_CSV = Path("results/overall_summary_by_alpha_allPolicies_20260116_150922.csv")

COLORMAP = "plasma"   # or "viridis"

SAVE_DIR = Path("plots")
SAVE_DIR.mkdir(exist_ok=True)

# per-metric y-limits you can tweak; key must match the y argument used below
YLIMITS = {
    # Example: zoom reward to [80, 100]
    "Total reward_mean": (80, 100),
    # other examples (disabled): "Useful downlinks (est)_mean": (0, 200)
}

# =========================
# LOAD & CLEAN
# =========================

df = pd.read_csv(SUMMARY_CSV)

# Extract mean value from "mean ± std"
def extract_mean(x):
    if isinstance(x, str) and "±" in x:
        try:
            return float(x.split("±")[0])
        except Exception:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def extract_std(x):
    """Return the std value parsed from 'mean ± std' or nan if missing."""
    if isinstance(x, str) and "±" in x:
        try:
            right = x.split("±")[1]
            return float(right)
        except Exception:
            return np.nan
    # No std information
    return np.nan

metrics = [
    "Total reward",
    "Illuminated images",
    "Useful downlinks (est)",
    "Downlink actions",
    "Imaging actions",
]

# create _mean and _std columns for each metric
for m in metrics:
    mean_col = m + "_mean"
    std_col = m + "_std"
    df[mean_col] = df[m].apply(extract_mean)
    df[std_col] = df[m].apply(extract_std)

# Only mixed env if present
df = df[df["Env"] == "MIXED"].copy()

# if alpha column is missing or NaN, try to coerce from strings
if "alpha" not in df.columns or df["alpha"].isna().all():
    # attempt to parse numeric alpha when it might be stored as text
    try:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    except Exception:
        pass

# drop rows without valid alpha
df = df[df["alpha"].notna()].copy()

# =========================
# PLOTTING
# =========================

cmap = plt.get_cmap(COLORMAP)
norm = plt.Normalize(df["alpha"].min(), df["alpha"].max())

def alpha_color(a):
    return cmap(norm(a))

def make_illum_and_downlink_plot(
    df: pd.DataFrame,
    illum_mean_col: str,
    illum_std_col: str,
    downlink_mean_col: str,
    downlink_std_col: str,
    ylabel: str,
    fname: str,
):
    df_sorted = df.sort_values("alpha").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7,5))

    # Scatter & line for illuminated images
    ax.errorbar(
        df_sorted["alpha"],
        df_sorted[illum_mean_col],
        yerr=df_sorted[illum_std_col],
        fmt='o',
        color='tab:blue',
        label="Illuminated images",
        capsize=4,
        markersize=6,
    )
    ax.plot(df_sorted["alpha"], df_sorted[illum_mean_col], linestyle='--', color='tab:blue', alpha=0.6)

    # Scatter & line for downlinked data amount
    ax.errorbar(
        df_sorted["alpha"],
        df_sorted[downlink_mean_col],
        yerr=df_sorted[downlink_std_col],
        fmt='s',
        color='tab:orange',
        label="Downlinked data amount",
        capsize=4,
        markersize=6,
    )
    ax.plot(df_sorted["alpha"], df_sorted[downlink_mean_col], linestyle='--', color='tab:orange', alpha=0.6)

    # Annotate each point with "mean ± std"
    for i, row in df_sorted.iterrows():
        # Illuminated images
        if not np.isnan(row[illum_mean_col]):
            ax.text(
                row["alpha"], row[illum_mean_col] + 0.5,
                f"{row[illum_mean_col]:.1f}±{row[illum_std_col]:.1f}",
                color='tab:blue', fontsize=8, ha='center'
            )
        # Downlinked amount
        if not np.isnan(row[downlink_mean_col]):
            ax.text(
                row["alpha"], row[downlink_mean_col] + 0.5,
                f"{row[downlink_mean_col]:.1f}±{row[downlink_std_col]:.1f}",
                color='tab:orange', fontsize=8, ha='center'
            )

    ax.set_xlabel(r"Downlink reward weight $\alpha$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAVE_DIR / fname, dpi=300)
    plt.close(fig)



# def make_plot(y_mean_col, y_std_col, ylabel, fname):
#     # sort by alpha for consistent lines
#     df_sorted = df.sort_values("alpha").reset_index(drop=True)
#
#     fig, ax = plt.subplots(figsize=(6,4))
#
#     # scatter with colored markers
#     for _, r in df_sorted.iterrows():
#         ax.scatter(
#             r["alpha"],
#             r[y_mean_col],
#             color=alpha_color(r["alpha"]),
#             s=60,
#             edgecolor="k",
#             zorder=3,
#         )
#
#     # line connecting means (sorted)
#     ax.plot(df_sorted["alpha"], df_sorted[y_mean_col], linestyle="--", alpha=0.6, zorder=2)
#
#     # add error bars (std). If std is nan, it will be ignored automatically.
#     yerr = df_sorted[y_std_col].to_numpy()
#     # Replace NaNs with zeros for plotting style but mask them in alpha for visibility
#     yerr_filled = np.nan_to_num(yerr, nan=0.0)
#     ax.errorbar(
#         df_sorted["alpha"],
#         df_sorted[y_mean_col],
#         yerr=yerr_filled,
#         fmt='none',
#         ecolor='gray',
#         elinewidth=1.2,
#         capsize=4,
#         alpha=0.8,
#         zorder=1,
#     )
#
#     ax.set_xlabel(r"Downlink reward weight $\alpha$")
#     ax.set_ylabel(ylabel)
#     ax.grid(True, alpha=0.3)
#
#     # optional y-limits
#     if y_mean_col in YLIMITS:
#         ymin, ymax = YLIMITS[y_mean_col]
#         ax.set_ylim(ymin, ymax)
#
#     # colorbar: explicit mappable
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, label=r"$\alpha$")
#
#     fig.tight_layout()
#     fig.savefig(SAVE_DIR / fname, dpi=300)
#     plt.close(fig)

def make_plot(y_mean_col, y_std_col, ylabel, fname):
    # sort by alpha for consistent lines
    df_sorted = df.sort_values("alpha").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6,4))

    # scatter with colored markers
    for _, r in df_sorted.iterrows():
        ax.scatter(
            r["alpha"],
            r[y_mean_col],
            color=alpha_color(r["alpha"]),
            s=60,
            edgecolor="k",
            zorder=3,
        )
        # Annotate each point with mean ± std
        if not np.isnan(r[y_mean_col]):
            ax.text(
                r["alpha"],
                r[y_mean_col] + 0.5,  # small offset above marker
                f"{r[y_mean_col]:.1f} ± {r[y_std_col]:.1f}" if not np.isnan(r[y_std_col]) else f"{r[y_mean_col]:.1f}",
                color="black",
                fontsize=8,
                ha="center",
            )

    # line connecting means (sorted)
    ax.plot(df_sorted["alpha"], df_sorted[y_mean_col], linestyle="--", alpha=0.6, zorder=2)

    # add error bars (std). If std is nan, it will be ignored automatically.
    yerr = df_sorted[y_std_col].to_numpy()
    yerr_filled = np.nan_to_num(yerr, nan=0.0)
    ax.errorbar(
        df_sorted["alpha"],
        df_sorted[y_mean_col],
        yerr=yerr_filled,
        fmt='none',
        ecolor='gray',
        elinewidth=1.2,
        capsize=4,
        alpha=0.8,
        zorder=1,
    )

    ax.set_xlabel(r"Downlink reward weight $\alpha$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # optional y-limits
    if y_mean_col in YLIMITS:
        ymin, ymax = YLIMITS[y_mean_col]
        ax.set_ylim(ymin, ymax)

    # colorbar: explicit mappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$\alpha$")

    fig.tight_layout()
    fig.savefig(SAVE_DIR / fname, dpi=300)
    plt.close(fig)


# =========================
# GENERATE FIGURES
# =========================

make_plot(
    "Total reward_mean",
    "Total reward_std",
    "Total episode reward",
    "reward_vs_alpha.png",
)

make_plot(
    "Illuminated images_mean",
    "Illuminated images_std",
    "Illuminated images",
    "images_vs_alpha.png",
)

make_plot(
    "Useful downlinks (est)_mean",
    "Useful downlinks (est)_std",
    "Useful downlinks",
    "useful_downlinks_vs_alpha.png",
)

make_plot(
    "Downlink actions_mean",
    "Downlink actions_std",
    "Downlink actions",
    "downlink_actions_vs_alpha.png",
)

make_plot(
    "Imaging actions_mean",
    "Imaging actions_std",
    "Imaging actions",
    "imaging_actions_vs_alpha.png",
)

make_illum_and_downlink_plot(
    df,
    illum_mean_col="Illuminated images_mean",
    illum_std_col="Illuminated images_std",
    downlink_mean_col="Downlink actions_mean",  # or your column with avg downlinked data
    downlink_std_col="Downlink actions_std",
    ylabel="Counts / Data amount",
    fname="updated_illum_and_downlink_vs_alpha.png"
)

print(f"Saved plots to {SAVE_DIR}/")
