#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sympy.abc import alpha

# =========================
# USER CONFIGURATION
# =========================

# SUMMARY_CSV = Path("results/overall_summary_by_alpha_allPolicies_20260115_201523.csv")
# SUMMARY_CSV = Path("results/overall_summary_by_alpha_allPolicies_20260115_173753.csv")

# or hardcode the exact file:
SUMMARY_CSV = Path("results/overall_summary_by_alpha_allPolicies_20260116_150922.csv")

COLORMAP = "plasma"   # or "viridis"
PLOT_COLORBAR = False  # set False if you don't want colorbars in the saved figures


SAVE_DIR = Path("plots")
SAVE_DIR.mkdir(exist_ok=True)

# per-metric y-limits you can tweak; key must match the y argument used below
YLIMITS = {
    # Example: zoom reward to [80, 100]
    "Total reward_mean": (80, 100),
    # other examples (disabled): "Useful downlinks (est)_mean": (0, 200)
}

# =========================
# VISUAL STYLE
# =========================

# Keep these in one place so you can quickly tweak aesthetics.
LINEWIDTH = 2.5
FILL_ALPHA = 0.10
ERRORBAR_ALPHA = 0.5
CAPSIZE = 3
GRID_ALPHA = 0.25

# If you don't yet have these derived quantities in your CSV, the script will compute
# them from the summary table columns it already has:
#   - images_per_downlink = useful_downlinks / downlink_actions
#   - (optionally) downlink_fraction = downlink_actions / (downlink_actions + imaging_actions)


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

# sort once (all plots use the same ordering)
df = df.sort_values("alpha").reset_index(drop=True)

# =========================
# DERIVED QUANTITIES
# =========================

# Images-per-downlink-action (mean) computed from means.
u = df["Useful downlinks (est)_mean"].to_numpy(dtype=float)
u_std = df["Useful downlinks (est)_std"].to_numpy(dtype=float)
d = df["Downlink actions_mean"].to_numpy(dtype=float)
d_std = df["Downlink actions_std"].to_numpy(dtype=float)

with np.errstate(divide="ignore", invalid="ignore"):
    images_per_downlink_mean = u / d
    # first-order error propagation for ratio; assumes independence
    images_per_downlink_std = np.abs(images_per_downlink_mean) * np.sqrt(
        (u_std / np.where(u == 0, np.nan, u)) ** 2 + (d_std / np.where(d == 0, np.nan, d)) ** 2
    )

# Optional: downlink fraction (actions) for later use if you want to plot it.
i = df["Imaging actions_mean"].to_numpy(dtype=float)
i_std = df["Imaging actions_std"].to_numpy(dtype=float)
total_actions = d + i
with np.errstate(divide="ignore", invalid="ignore"):
    downlink_fraction_mean = d / total_actions
    downlink_fraction_std = np.sqrt(
        # propagate through d/(d+i) with partial derivatives (approx)
        (i / np.where(total_actions == 0, np.nan, total_actions) ** 2) ** 2 * (d_std ** 2)
        + (d / np.where(total_actions == 0, np.nan, total_actions) ** 2) ** 2 * (i_std ** 2)
    )

# =========================
# PLOTTING HELPERS
# =========================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


def _finish_axis(ax):
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlabel(r"Downlink reward weight $\alpha$")


def _save(fig, fname: str):
    # bbox_inches="tight" prevents the colorbar/legend from being cut off
    fig.savefig(SAVE_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)



# =========================
# GENERATE FIGURES (coded one-by-one)
# =========================

x = df["alpha"].to_numpy(dtype=float)
cmap = plt.get_cmap(COLORMAP)
norm = Normalize(vmin=np.nanmin(x), vmax=np.nanmax(x))


# (a) Total reward
fig, ax = plt.subplots(figsize=(6.2, 4.2))
y = df["Total reward_mean"].to_numpy(dtype=float)
ys = df["Total reward_std"].to_numpy(dtype=float)
# shaded uncertainty band (keep this neutral so the plasma line reads clearly)
ax.fill_between(x, y - ys, y + ys, alpha=FILL_ALPHA, zorder=1)

# --- plasma gradient line (colored by alpha) ---
pts = np.array([x, y]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

lc = LineCollection(segs, cmap=cmap, norm=norm)
lc.set_array(x[:-1])              # color each segment by its alpha (left-to-right transition)
lc.set_linewidth(LINEWIDTH)
lc.set_zorder(3)
ax.add_collection(lc)

# optional: colored markers at each alpha
ax.scatter(x, y, c=x, cmap=cmap, norm=norm, s=35, edgecolor="k", linewidth=0.6, zorder=4)
# error bars ON TOP of the band (keep them neutral so the plasma markers remain readable)
ax.errorbar(x, y, yerr=ys, fmt="none", ecolor="0.25", elinewidth=1.2,
            capsize=CAPSIZE, zorder=2)

# colorbar for alpha (on the right)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r"$\alpha$")
ax.set_xlim(np.nanmin(x), np.nanmax(x))

ax.set_ylabel("Total episode reward")
if "Total reward_mean" in YLIMITS:
    ax.set_ylim(*YLIMITS["Total reward_mean"])
_finish_axis(ax)
_save(fig, "reward_vs_alpha.png")


#######################PLOT (b) Illuminated images
fig, ax = plt.subplots(figsize=(6.2, 4.2))
y = df["Illuminated images_mean"].to_numpy(dtype=float)
ys = df["Illuminated images_std"].to_numpy(dtype=float)
# shaded uncertainty band (keep this neutral so the plasma line reads clearly)
ax.fill_between(x, y - ys, y + ys, alpha=FILL_ALPHA, zorder=1)

# --- plasma gradient line (colored by alpha) ---
pts = np.array([x, y]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

lc = LineCollection(segs, cmap=cmap, norm=norm)
lc.set_array(x[:-1])              # color each segment by its alpha (left-to-right transition)
lc.set_linewidth(LINEWIDTH)
lc.set_zorder(3)
ax.add_collection(lc)

# optional: colored markers at each alpha
ax.scatter(x, y, c=x, cmap=cmap, norm=norm, s=35, edgecolor="k", linewidth=0.6, zorder=4)
ax.errorbar(x, y, yerr=ys, fmt="none", ecolor="0.25", elinewidth=1.2,
            capsize=CAPSIZE, zorder=2,alpha=ERRORBAR_ALPHA)

# colorbar for alpha (on the right)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r"$\alpha$")

ax.set_ylabel("Illuminated images")
_finish_axis(ax)
_save(fig, "images_vs_alpha.png")


#######################PLOT (c) Useful downlinks + images-per-downlink (secondary y)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
# --- DownMean (secondary axis) values (mean ± std) ---
downmean_map = {
    1.0: (0.16, 0.02),
    0.9: (0.17, 0.03),
    0.8: (0.16, 0.03),
    0.7: (0.16, 0.03),
    0.6: (0.17, 0.03),
    0.5: (0.15, 0.03),
    0.4: (0.19, 0.03),
    0.3: (0.20, 0.04),
    0.2: (0.24, 0.05),
    0.1: (0.32, 0.07),
    0.0: (0.85, 0.13),
}

# align DownMean to your x array (assumes x contains those alphas)
down_mean = np.array([downmean_map[float(a)][0] for a in x], dtype=float)
down_std  = np.array([downmean_map[float(a)][1] for a in x], dtype=float)

y = df["Useful downlinks (est)_mean"].to_numpy(dtype=float)
ys = df["Useful downlinks (est)_std"].to_numpy(dtype=float)
# shaded uncertainty band (keep this neutral so the plasma line reads clearly)
# --- Primary axis: Useful downlinks (mean ± std) ---
ax.fill_between(x, y - ys, y + ys, alpha=FILL_ALPHA, zorder=1)

pts = np.array([x, y]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

lc = LineCollection(segs, cmap=cmap, norm=norm)
lc.set_array(x[:-1])
lc.set_linewidth(LINEWIDTH)
lc.set_zorder(3)
ax.add_collection(lc)

ax.scatter(x, y, c=x, cmap=cmap, norm=norm,
           s=35, edgecolor="k", linewidth=0.6, zorder=4,
           label="Downlinked images",marker="v")

ax.errorbar(x, y, yerr=ys, fmt="none", ecolor="0.25", elinewidth=1.2,
            capsize=CAPSIZE, zorder=2,alpha=ERRORBAR_ALPHA)


ax.set_ylabel("Downlinked Images")
ax.set_xlim(np.nanmin(x), np.nanmax(x))
ax.set_ylim(70, 97.5)

# # optional: colored markers at each alpha
# ax.scatter(x, y, c=x, cmap=cmap, norm=norm, s=35, edgecolor="k", linewidth=0.6, zorder=4)

# # colorbar for alpha (on the right)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# fig.colorbar(sm, ax=ax, label=r"$\alpha$")

#
# ax2 = ax.twinx()
# ax2.plot(x, images_per_downlink_mean, linewidth=LINEWIDTH, linestyle="--", label="Images per downlink")
# ax2.fill_between(
#     x,
#     images_per_downlink_mean - images_per_downlink_std,
#     images_per_downlink_mean + images_per_downlink_std,
#     alpha=FILL_ALPHA,
# )
# ax2.set_ylabel("Useful images per downlink action")
ax2 = ax.twinx()

# --- Secondary axis: DownMean (mean ± std), plasma-colored ---
ax2.fill_between(x, down_mean - down_std, down_mean + down_std,
                 alpha=FILL_ALPHA, zorder=1)

pts2 = np.array([x, down_mean]).T.reshape(-1, 1, 2)
segs2 = np.concatenate([pts2[:-1], pts2[1:]], axis=1)

lc2 = LineCollection(segs2, cmap=cmap, norm=norm)
lc2.set_array(x[:-1])
lc2.set_linewidth(LINEWIDTH)   # you can make this thinner if you want, e.g. LINEWIDTH*0.9
lc2.set_zorder(3)
ax2.add_collection(lc2)

ax2.scatter(x, down_mean, c=x, cmap=cmap, norm=norm,
            s=35, marker="s", edgecolor="k", linewidth=0.6, zorder=4,
            label="Mean Downlink Fraction")

# ax2.errorbar(x, down_mean, yerr=down_std, fmt="none", ecolor="0.25", elinewidth=1.2,
#              capsize=CAPSIZE, zorder=2)
ax2.errorbar(x, down_mean, yerr=down_std, fmt="none", ecolor="0.2",
             elinewidth=1.6, capsize=CAPSIZE, zorder=5)

ax2.set_ylabel("Mean Downlink Fraction")


_finish_axis(ax)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2,
          loc="upper center", bbox_to_anchor=(0.5, 0.75),
          ncol=1, framealpha=0.9)

if PLOT_COLORBAR:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # attach to BOTH axes and push outward so it won't overlap ax2 label
    fig.colorbar(sm, ax=[ax, ax2], pad=0.12, fraction=0.05, label=r"$\alpha$")

ax.relim()
ax.autoscale_view()

_save(fig, "useful_downlinks_vs_alpha.png")


#######################PLOT (d)+(e) Combine Imaging actions (left y, shaded) with Downlink actions (right y, stars + error bars)
fig, ax = plt.subplots(figsize=(8.2, 4.2))

# y_img = df["Imaging actions_mean"].to_numpy(dtype=float)
# ys_img = df["Imaging actions_std"].to_numpy(dtype=float)
# ax.plot(x, y_img, linewidth=LINEWIDTH, label="Imaging actions")
# ax.fill_between(x, y_img - ys_img, y_img + ys_img, alpha=FILL_ALPHA)
# ax.set_ylabel("Imaging actions")
#
# ax2 = ax.twinx()
# y_dl = df["Downlink actions_mean"].to_numpy(dtype=float)
# ys_dl = df["Downlink actions_std"].to_numpy(dtype=float)
# ax2.errorbar(
#     x,
#     y_dl,
#     yerr=ys_dl,
#     fmt="*",
#     markersize=10,
#     capsize=CAPSIZE,
#     linewidth=1.5,
#     label="Downlink actions",
# )
# ax2.plot(x, y_dl, linewidth=1.8, linestyle="--")
# ax2.set_ylabel("Downlink actions")
y_img = df["Imaging actions_mean"].to_numpy(dtype=float)
ys_img = df["Imaging actions_std"].to_numpy(dtype=float)

# imaging: band + neutral errorbars + plasma-colored STAR markers
ax.fill_between(x, y_img - ys_img, y_img + ys_img, alpha=FILL_ALPHA, zorder=1)
ax.errorbar(x, y_img, yerr=ys_img, fmt="none", ecolor="0.25", elinewidth=1.2,
            capsize=CAPSIZE, zorder=2)
ax.scatter(x, y_img, c=x, cmap=cmap, norm=norm,
           marker="*", s=110, edgecolor="k", linewidth=0.6, zorder=4,
           label="Imaging actions")
ax.set_ylabel("Imaging actions")

ax2 = ax.twinx()
y_dl = df["Downlink actions_mean"].to_numpy(dtype=float)
ys_dl = df["Downlink actions_std"].to_numpy(dtype=float)

# downlink: band + neutral errorbars + plasma-colored TRIANGLE_DOWN markers
ax2.fill_between(x, y_dl - ys_dl, y_dl + ys_dl, alpha=FILL_ALPHA, zorder=1)
ax2.errorbar(x, y_dl, yerr=ys_dl, fmt="none", ecolor="0.25", elinewidth=1.2,
             capsize=CAPSIZE, zorder=2)
ax2.scatter(x, y_dl, c=x, cmap=cmap, norm=norm,
            marker="v", s=70, edgecolor="k", linewidth=0.6, zorder=4,
            label="Downlink actions")
ax2.set_ylabel("Downlink actions")

# colorbar (optional but consistent)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# fig.colorbar(sm, ax=[ax, ax2], pad=0.02, fraction=0.05, label=r"$\alpha$")


_finish_axis(ax)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2,
          loc="upper left", bbox_to_anchor=(0.1, 0.98 ),
          ncol=1, framealpha=0.9)

_save(fig, "actions_vs_alpha_combined.png")

############ACTIONS Plot version 2
fig, ax = plt.subplots(figsize=(8.2, 4.2))

# --- imaging ---
y_img = df["Imaging actions_mean"].to_numpy(dtype=float)
ys_img = df["Imaging actions_std"].to_numpy(dtype=float)

ax.fill_between(
    x, y_img - ys_img, y_img + ys_img,
    alpha=FILL_ALPHA, zorder=1
)
ax.errorbar(
    x, y_img, yerr=ys_img,
    fmt="none", ecolor="0.25",
    elinewidth=1.2, capsize=CAPSIZE, zorder=2
)
ax.scatter(
    x, y_img, c=x, cmap=cmap, norm=norm,
    marker="*", s=60, edgecolor="k",
    linewidth=0.2, zorder=4,
    label="Imaging actions"
)

# --- downlink ---
y_dl = df["Downlink actions_mean"].to_numpy(dtype=float)
ys_dl = df["Downlink actions_std"].to_numpy(dtype=float)

ax.fill_between(
    x, y_dl - ys_dl, y_dl + ys_dl,
    alpha=0.6 * FILL_ALPHA, zorder=1
)
ax.errorbar(
    x, y_dl, yerr=ys_dl,
    fmt="none", ecolor="0.25",
    elinewidth=1.2, capsize=CAPSIZE, zorder=2
)
ax.scatter(
    x, y_dl, c=x, cmap=cmap, norm=norm,
    marker="v", s=45, edgecolor="k",
    linewidth=0.15, zorder=4,
    label="Downlink actions"
)

# --- axis formatting ---
ax.set_ylabel("Action count")

y_max = max(
    np.max(y_img + ys_img),
    np.max(y_dl + ys_dl)
)
ax.set_ylim(0, 160)

_finish_axis(ax)

ax.legend(
    loc="center left",
    bbox_to_anchor=(0.1, 0.48),
    framealpha=0.9
)
# --- save ---
_save(fig, "actions_vs_alpha_combined_oneaxis.pdf")





# (sixth) Combined: Total reward (o), Illuminated images (*), Useful downlinks (v)
fig, ax = plt.subplots(figsize=(7.2, 4.2))

y_r  = df["Total reward_mean"].to_numpy(dtype=float)
ys_r = df["Total reward_std"].to_numpy(dtype=float)

y_i  = df["Illuminated images_mean"].to_numpy(dtype=float)
ys_i = df["Illuminated images_std"].to_numpy(dtype=float)

y_u  = df["Useful downlinks (est)_mean"].to_numpy(dtype=float)
ys_u = df["Useful downlinks (est)_std"].to_numpy(dtype=float)

# shaded bands
alpha_factor = 0.6
ax.fill_between(x, y_r - ys_r, y_r + ys_r, alpha=FILL_ALPHA*alpha_factor, zorder=1)
ax.fill_between(x, y_i - ys_i, y_i + ys_i, alpha=FILL_ALPHA*alpha_factor, zorder=1)
ax.fill_between(x, y_u - ys_u, y_u + ys_u, alpha=FILL_ALPHA*alpha_factor, zorder=1)

# error bars on top
ax.errorbar(x, y_r, yerr=ys_r, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
ax.errorbar(x, y_i, yerr=ys_i, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
ax.errorbar(x, y_u, yerr=ys_u, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)

# plasma-coded markers
ax.scatter(x, y_r, c=x, cmap=cmap, norm=norm,
           marker="o", s=60, edgecolor="k", linewidth=0.8, zorder=6,
           label="Reward")

ax.scatter(x, y_i, c=x, cmap=cmap, norm=norm,
           marker="*", s=140, edgecolor="k", linewidth=0.8, zorder=6,
           label="Illuminated images")

ax.scatter(x, y_u, c=x, cmap=cmap, norm=norm,
           marker="v", s=80, edgecolor="k", linewidth=0.8, zorder=6,
           label="Downlinked images")

ax.set_ylabel("Count")
ax.set_ylim(70, 100)
ax.set_xlim(np.nanmin(x), np.nanmax(x))

ax.minorticks_on()
ax.grid(True, alpha=GRID_ALPHA)
ax.grid(True, which="minor", alpha=GRID_ALPHA * 0.6)

_finish_axis(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, framealpha=0.9)

if PLOT_COLORBAR:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.05, label=r"$\alpha$")
# ---- inset zoom: alpha in [0.1, 1.0], y in [90, 96] ----
axins = inset_axes(ax, width="85%", height="60%", loc="lower right", borderpad=1.2)
axins.set_xlim(0.05, 1.0)
axins.set_ylim(91, 96.5)

# lighter grid in inset
axins.minorticks_on()
axins.grid(True, alpha=GRID_ALPHA)
axins.grid(True, which="minor", alpha=GRID_ALPHA * 0.6)

# re-plot the same data in the inset (bands + errorbars + plasma markers)
axins.fill_between(x, y_r - ys_r, y_r + ys_r, alpha=FILL_ALPHA*alpha_factor, zorder=1)
axins.fill_between(x, y_i - ys_i, y_i + ys_i, alpha=FILL_ALPHA*alpha_factor, zorder=1)
axins.fill_between(x, y_u - ys_u, y_u + ys_u, alpha=FILL_ALPHA*alpha_factor, zorder=1)

axins.errorbar(x, y_r, yerr=ys_r, fmt="none", ecolor="0.2", elinewidth=1.3, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
axins.errorbar(x, y_i, yerr=ys_i, fmt="none", ecolor="0.2", elinewidth=1.3, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
axins.errorbar(x, y_u, yerr=ys_u, fmt="none", ecolor="0.2", elinewidth=1.3, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)

axins.scatter(x, y_r, c=x, cmap=cmap, norm=norm,
              marker="o", s=35, edgecolor="k", linewidth=0.7, zorder=6)
axins.scatter(x, y_i, c=x, cmap=cmap, norm=norm,
              marker="*", s=85, edgecolor="k", linewidth=0.7, zorder=6)
axins.scatter(x, y_u, c=x, cmap=cmap, norm=norm,
              marker="v", s=45, edgecolor="k", linewidth=0.7, zorder=6)

# hide inset tick labels if you want it cleaner (optional)
# axins.tick_params(labelleft=False, labelbottom=False)

# draw connectors + rectangle showing zoomed region
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", lw=1.0)


_save(fig, "reward_images_downlinks_combined.png")




# (seventh) Combined: Total reward (o), Illuminated images (*), Useful downlinks (v)
# ---------------- Font size controls ----------------
TICK_FONTSIZE      = 13
LABEL_FONTSIZE     = 15
LEGEND_FONTSIZE    = 14
LEGEND_TITLE_SIZE  = 14
COLORBAR_FONTSIZE  = 13
INSET_TICK_SIZE    = 13

fig, ax = plt.subplots(figsize=(10.0, 7.2))
ax.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_FONTSIZE
)
ax.tick_params(
    axis="both",
    which="minor",
    labelsize=TICK_FONTSIZE - 2
)

y_r  = df["Total reward_mean"].to_numpy(dtype=float)
ys_r = df["Total reward_std"].to_numpy(dtype=float)

y_i  = df["Illuminated images_mean"].to_numpy(dtype=float)
ys_i = df["Illuminated images_std"].to_numpy(dtype=float)

y_u  = df["Useful downlinks (est)_mean"].to_numpy(dtype=float)
ys_u = df["Useful downlinks (est)_std"].to_numpy(dtype=float)

# shaded bands
alpha_factor = 0.6
ax.fill_between(x, y_r - ys_r, y_r + ys_r, alpha=FILL_ALPHA*alpha_factor, zorder=1)
ax.fill_between(x, y_i - ys_i, y_i + ys_i, alpha=FILL_ALPHA*alpha_factor, zorder=1)
ax.fill_between(x, y_u - ys_u, y_u + ys_u, alpha=FILL_ALPHA*alpha_factor, zorder=1)

# error bars on top
ax.errorbar(x, y_r, yerr=ys_r, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
ax.errorbar(x, y_i, yerr=ys_i, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
ax.errorbar(x, y_u, yerr=ys_u, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)

# plasma-coded markers
ax.scatter(x, y_r, c=x, cmap=cmap, norm=norm,
           marker="o", s=40, edgecolor="k", linewidth=0.2, zorder=6,
           label="Reward")
ax.scatter(x[0], y_r[0], c=x[0], cmap=cmap, norm=norm,
           marker="o", s=40, edgecolor="k", linewidth=0.2, zorder=6)
ax.scatter(x, y_i, c=x, cmap=cmap, norm=norm,
           marker="*", s=60, edgecolor="k", linewidth=0.2, zorder=6,
           label="Illuminated images")

ax.scatter(x, y_u, c=x, cmap=cmap, norm=norm,
           marker="v", s=40, edgecolor="k", linewidth=0.2, zorder=6,
           label="Downlinked images")
ax.scatter(x[0], y_u[0], c=x[0], cmap=cmap, norm=norm,
           marker="v", s=50, edgecolor="k", linewidth=0.2, zorder=6)

ax.set_ylabel("Count", fontsize=LABEL_FONTSIZE)
ax.set_ylim(78, 97)
ax.set_xlim(np.nanmin(x), np.nanmax(x))

ax.minorticks_on()
ax.grid(True, alpha=GRID_ALPHA)
ax.grid(True, which="minor", alpha=GRID_ALPHA * 0.6)

ax.grid(True, alpha=GRID_ALPHA)
ax.set_xlabel(r"Downlink reward weight $\alpha$", fontsize=LABEL_FONTSIZE)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, framealpha=0.95,fontsize=LEGEND_FONTSIZE)



# ---- inset zoom: alpha in [0.1, 1.0], y in [90, 96] ----
# axins = inset_axes(ax, width="85%", height="55%", loc="lower right", borderpad=2.4)
axins = inset_axes(
    ax,
    width="90%",
    height="55%",
    loc="lower right",
    bbox_to_anchor=(-0.03, 0.07, 1.0, 1.0),
    bbox_transform=ax.transAxes,
    borderpad=0
)
axins.tick_params(
    axis="both",
    which="major",
    labelsize=INSET_TICK_SIZE
)
axins.tick_params(
    axis="both",
    which="minor",
    labelsize=INSET_TICK_SIZE - 2
)

axins.set_xlim(0.05, 1.05)
axins.set_ylim(91, 96)

# lighter grid in inset
axins.minorticks_on()
axins.grid(True, alpha=GRID_ALPHA)
axins.grid(True, which="minor", alpha=GRID_ALPHA * 0.6)
alpha_factor=0.2
# re-plot the same data in the inset (bands + errorbars + plasma markers)
axins.fill_between(x, y_r - ys_r, y_r + ys_r, alpha=FILL_ALPHA*alpha_factor, zorder=1)
axins.fill_between(x, y_i - ys_i, y_i + ys_i, alpha=FILL_ALPHA*alpha_factor, zorder=1)
axins.fill_between(x, y_u - ys_u, y_u + ys_u, alpha=FILL_ALPHA*alpha_factor, zorder=1)

axins.errorbar(x, y_r, yerr=ys_r, fmt="none", ecolor="0.2", elinewidth=1.3, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
axins.errorbar(x, y_i, yerr=ys_i, fmt="none", ecolor="0.2", elinewidth=1.3, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
axins.errorbar(x, y_u, yerr=ys_u, fmt="none", ecolor="0.2", elinewidth=1.3, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)

axins.scatter(x, y_r, c=x, cmap=cmap, norm=norm,
              marker="o", s=35, edgecolor="k", linewidth=0.7, zorder=6)
axins.scatter(x, y_i, c=x, cmap=cmap, norm=norm,
              marker="*", s=85, edgecolor="k", linewidth=0.7, zorder=6)
axins.scatter(x, y_u, c=x, cmap=cmap, norm=norm,
              marker="v", s=45, edgecolor="k", linewidth=0.7, zorder=6)

# hide inset tick labels if you want it cleaner (optional)
# axins.tick_params(labelleft=False, labelbottom=False)

# draw connectors + rectangle showing zoomed region
mark_inset(ax, axins, loc1=2, loc2=1 , fc="none", ec="0.3", lw=0.0)

if PLOT_COLORBAR:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=[ax],
        pad=0.03,
        fraction=0.05
    )

    cbar.set_label(r"$\alpha$", fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE)

plt.show()

_save(fig, "reward_images_downlinks_simple.png")



# (eight) Combined: Total reward (o), Illuminated images (*), Useful downlinks (v)
fig, ax = plt.subplots(figsize=(7.2, 4.2))

y_r  = df["Total reward_mean"].to_numpy(dtype=float)
ys_r = df["Total reward_std"].to_numpy(dtype=float)

y_i  = df["Illuminated images_mean"].to_numpy(dtype=float)
ys_i = df["Illuminated images_std"].to_numpy(dtype=float)

y_u  = df["Useful downlinks (est)_mean"].to_numpy(dtype=float)
ys_u = df["Useful downlinks (est)_std"].to_numpy(dtype=float)

# shaded bands
alpha_factor = 0.6
ax.fill_between(x, y_r - ys_r, y_r + ys_r, alpha=FILL_ALPHA*alpha_factor, zorder=1)
ax.fill_between(x, y_i - ys_i, y_i + ys_i, alpha=FILL_ALPHA*alpha_factor, zorder=1)
ax.fill_between(x, y_u - ys_u, y_u + ys_u, alpha=FILL_ALPHA*alpha_factor, zorder=1)

# error bars on top
ax.errorbar(x, y_r, yerr=ys_r, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
ax.errorbar(x, y_i, yerr=ys_i, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)
ax.errorbar(x, y_u, yerr=ys_u, fmt="none", ecolor="0.2", elinewidth=1.6, capsize=CAPSIZE, zorder=5, alpha=ERRORBAR_ALPHA)

# plasma-coded markers
ax.scatter(x, y_r, c=x, cmap=cmap, norm=norm,
           marker="o", s=60, edgecolor="k", linewidth=0.8, zorder=6,
           label="Reward")

ax.scatter(x, y_i, c=x, cmap=cmap, norm=norm,
           marker="*", s=140, edgecolor="k", linewidth=0.8, zorder=6,
           label="Illuminated images")

ax.scatter(x, y_u, c=x, cmap=cmap, norm=norm,
           marker="v", s=80, edgecolor="k", linewidth=0.8, zorder=6,
           label="Downlinked images")

ax.set_ylabel("Count")
ax.set_ylim(80, 100)
ax.set_xlim(np.nanmin(x), np.nanmax(x))

ax.minorticks_on()
ax.grid(True, alpha=GRID_ALPHA)
ax.grid(True, which="minor", alpha=GRID_ALPHA * 0.6)

_finish_axis(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, framealpha=0.9)

if PLOT_COLORBAR:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.05, label=r"$\alpha$")

_save(fig, "reward_images_downlinks_default.pdf")


print(f"Saved plots to {SAVE_DIR}/")
