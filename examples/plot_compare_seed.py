#!/usr/bin/env python3
"""
Compare one RL run vs one heuristic run and generate paper-quality plots.

Usage (from examples/):
  python plot_compare_seed.py \
    --rl_dir   data/RL_seed20_20260108_205129 \
    --heur_dir data/HEUR_ANGLE_seed20_20260108_204759 \
    --out_dir  plots/compare_seed20

Outputs:
  - timeline_actions.pdf
  - lookahead_by_phase.pdf
  - acquisition_cdf.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


UMBRA_TAU = 0.05       # shadowFactor <= 0.05 ~ umbra (adjust if needed)
SUNLIT_TAU = 0.95      # shadowFactor >= 0.95 ~ sunlit (adjust if needed)


def _load_run(run_dir: str):
    steps_path = os.path.join(run_dir, "steps.csv")
    images_path = os.path.join(run_dir, "images.csv")

    if not os.path.isfile(steps_path):
        raise FileNotFoundError(f"Missing steps.csv in {run_dir}")
    if not os.path.isfile(images_path):
        raise FileNotFoundError(f"Missing images.csv in {run_dir}")

    steps = pd.read_csv(steps_path)
    images = pd.read_csv(images_path)

    # sanity checks
    for col in ["t_cmd", "action_id", "sat_shadow_cmd"]:
        if col not in steps.columns:
            raise ValueError(f"{steps_path} missing required column: {col}")

    for col in ["t_cmd", "look_ahead", "phase", "dt_acq", "acq_success", "sat_shadow_cmd"]:
        if col not in images.columns:
            raise ValueError(f"{images_path} missing required column: {col}")

    # normalize time to start at 0 for nice aligned plots
    t0 = float(min(steps["t_cmd"].min(), images["t_cmd"].min()))
    steps = steps.copy()
    images = images.copy()
    steps["t_rel_hr"] = (steps["t_cmd"] - t0) / 3600.0
    images["t_rel_hr"] = (images["t_cmd"] - t0) / 3600.0

    return steps, images


def _ensure_outdir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _shade_umbra(ax, t_hr, shadow, alpha=0.12):
    """Shade umbra intervals where shadowFactor <= UMBRA_TAU."""
    t_hr = np.asarray(t_hr, dtype=float)
    shadow = np.asarray(shadow, dtype=float)

    in_umbra = shadow <= UMBRA_TAU
    if in_umbra.size == 0:
        return

    # find contiguous segments
    idx = np.where(in_umbra)[0]
    if idx.size == 0:
        return

    # segment boundaries
    starts = [idx[0]]
    ends = []
    for k in range(1, idx.size):
        if idx[k] != idx[k - 1] + 1:
            ends.append(idx[k - 1])
            starts.append(idx[k])
    ends.append(idx[-1])

    ymin, ymax = ax.get_ylim()
    for s, e in zip(starts, ends):
        ax.axvspan(t_hr[s], t_hr[e], alpha=alpha)


def plot_timeline_actions(steps_rl, imgs_rl, steps_h, imgs_h, out_dir: str):
    """
    Two-panel timeline:
      - eclipse (umbra) shading from sat_shadow_cmd
      - action markers: imaging actions (0-9) vs non-imaging (10-12)
      - optional regime color for imaging if available
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    def _panel(ax, steps, imgs, title):
        ax.set_title(title)
        ax.set_ylabel("Action markers")

        # set default y-limits first so shading spans correct region
        ax.set_ylim(-0.2, 1.2)

        # shade umbra from steps
        _shade_umbra(ax, steps["t_rel_hr"].values, steps["sat_shadow_cmd"].values)

        # imaging actions from steps (0-9)
        is_imaging = steps["action_id"].between(0, 9)
        is_other = ~is_imaging

        ax.scatter(
            steps.loc[is_imaging, "t_rel_hr"],
            np.ones(is_imaging.sum()) * 0.8,
            s=10,
            marker="o",
            label="Imaging cmd (step)",
        )

        # non-imaging actions: charge=10, downlink=11, desat=12
        for aid, lab, y, mk in [(10, "Charge", 0.25, "s"),
                               (11, "Downlink", 0.45, "D"),
                               (12, "Desat", 0.65, "^")]:
            m = steps["action_id"] == aid
            if m.any():
                ax.scatter(steps.loc[m, "t_rel_hr"], np.ones(m.sum()) * y, s=18, marker=mk, label=lab)

        # (optional) overlay actual imaging commands from images.csv as ticks
        # This helps show command decisions even if env.step logging differs slightly.
        ax.scatter(
            imgs["t_rel_hr"].values,
            np.ones(len(imgs)) * 1.05,
            s=6,
            marker="|",
            label="Imaging cmd (images.csv)",
        )

        # annotate umbra % at command time for context
        umbra_pct = float((steps["sat_shadow_cmd"].values <= UMBRA_TAU).mean() * 100.0)
        ax.text(0.01, 0.02, f"Umbra fraction (step samples): {umbra_pct:.1f}%",
                transform=ax.transAxes)

        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)

    _panel(axes[0], steps_rl, imgs_rl, "RL (seed-aligned) — actions with umbra shading")
    _panel(axes[1], steps_h, imgs_h, "Heuristic (seed-aligned) — actions with umbra shading")
    axes[1].set_xlabel("Time since start [hours]")

    path_pdf = os.path.join(out_dir, "timeline_actions.pdf")
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")

def plot_lookahead_by_phase(imgs_rl, imgs_h, out_dir: str):
    phases = ["sunlit", "entering", "umbra", "exiting", "penumbra_flat"]

    def _summ(df):
        out = []
        for ph in phases:
            d = df[df["phase"] == ph]
            if len(d) == 0:
                out.append((ph, np.nan, np.nan, 0))
                continue
            m = float(np.nanmean(d["look_ahead"]))
            se = float(np.nanstd(d["look_ahead"], ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            out.append((ph, m, se, len(d)))
        return out

    rl_all = _summ(imgs_rl)
    h_all  = _summ(imgs_h)

    def _vals(rows):
        means = np.array([r[1] for r in rows], dtype=float)
        ses   = np.array([r[2] for r in rows], dtype=float)
        ns    = [r[3] for r in rows]
        return means, ses, ns

    rl_m, rl_se, rl_n = _vals(rl_all)
    h_m,  h_se,  h_n  = _vals(h_all)

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width/2, rl_m, width, yerr=rl_se, capsize=3, label="RL")
    ax.bar(x + width/2, h_m,  width, yerr=h_se,  capsize=3, label="Heuristic")

    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0.0, linewidth=1)
    ax.axhline(1.0, linewidth=1, alpha=0.25)
    ax.axhline(-1.0, linewidth=1, alpha=0.25)

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("look_ahead = cos(azimuth): +1 ahead, 0 sideways, -1 behind")
    ax.set_title("Look direction vs eclipse/transition bin (command-time)")

    for i in range(len(phases)):
        y_rl = rl_m[i] if np.isfinite(rl_m[i]) else 0.0
        y_h  = h_m[i]  if np.isfinite(h_m[i])  else 0.0
        ax.text(i - width/2, y_rl + 0.05, f"n={rl_n[i]}", ha="center", fontsize=8)
        ax.text(i + width/2, y_h  + 0.05, f"n={h_n[i]}", ha="center", fontsize=8)

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path_pdf = os.path.join(out_dir, "lookahead_by_phase.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")


def _ecdf(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([]), np.array([])
    a = np.sort(a)
    y = np.arange(1, a.size + 1) / a.size
    return a, y


def plot_acquisition_cdf(imgs_rl, imgs_h, out_dir: str):
    """
    Plot ECDFs of acquisition time dt_acq for successful acquisitions.
    Includes both:
      - overall (solid)
      - conditioned on umbra at command (dashed)
    """
    fig, ax = plt.subplots(figsize=(11, 4.8))

    def _select(df, umbra_only=False):
        d = df[df["acq_success"] == 1]
        if umbra_only:
            d = d[d["sat_shadow_cmd"] <= UMBRA_TAU]
        return d["dt_acq"].values

    for label, df in [("RL", imgs_rl), ("Heuristic", imgs_h)]:
        x, y = _ecdf(_select(df, umbra_only=False))
        if x.size:
            ax.plot(x, y, label=f"{label} (all)", linewidth=2)

        x2, y2 = _ecdf(_select(df, umbra_only=True))
        if x2.size:
            ax.plot(x2, y2, label=f"{label} (umbra@cmd)", linestyle="--", linewidth=2)

    ax.set_xlabel("Acquisition time dt_acq [s]")
    ax.set_ylabel("ECDF")
    ax.set_title("Acquisition-time distribution (successes only)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path_pdf = os.path.join(out_dir, "acquisition_cdf.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")

def plot_lookahead_vs_shadow_scatter(imgs_rl, imgs_h, out_dir: str, nbins: int = 12):
    """
    Scatter of look_ahead vs sat_shadow_cmd with binned mean trendlines.
    Works well even for a single seed/run.

    look_ahead = cos(azimuth):
      +1 = ahead, 0 = sideways, -1 = behind
    sat_shadow_cmd in [0,1]:
      0 = deep eclipse (umbra-ish), 1 = full sun
    """
    fig, ax = plt.subplots(figsize=(11, 4.8))

    def _plot_one(df, label):
        x = df["sat_shadow_cmd"].to_numpy(dtype=float)
        y = df["look_ahead"].to_numpy(dtype=float)

        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            return

        # scatter (lightweight)
        ax.scatter(x, y, s=8, alpha=0.20, label=f"{label} samples")

        # binned mean + stderr
        edges = np.linspace(0.0, 1.0, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(nbins, np.nan)
        ses = np.full(nbins, np.nan)

        for i in range(nbins):
            b = (x >= edges[i]) & (x < edges[i + 1] if i < nbins - 1 else x <= edges[i + 1])
            yy = y[b]
            yy = yy[np.isfinite(yy)]
            if yy.size == 0:
                continue
            means[i] = float(np.mean(yy))
            ses[i] = float(np.std(yy, ddof=1) / np.sqrt(yy.size)) if yy.size > 1 else 0.0

        ax.errorbar(
            centers, means, yerr=ses, linewidth=2, marker="o", capsize=3,
            label=f"{label} binned mean ± SE"
        )

    _plot_one(imgs_rl, "RL")
    _plot_one(imgs_h, "Heuristic")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0.0, linewidth=1)
    ax.axhline(1.0, linewidth=1, alpha=0.25)
    ax.axhline(-1.0, linewidth=1, alpha=0.25)

    ax.set_xlabel("sat_shadow_cmd (0=eclipse, 1=sunlit)")
    ax.set_ylabel("look_ahead = cos(azimuth): +1 ahead, 0 sideways, -1 behind")
    ax.set_title("Where the satellite points vs eclipse state (command time)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    path_pdf = os.path.join(out_dir, "lookahead_vs_shadow_scatter.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")


def plot_regime_fraction_vs_shadow_bin(imgs_rl, imgs_h, out_dir: str, nbins: int = 10):
    """
    For each shadowFactor bin, compute the fraction of chosen targets by regime (LEO/MEO/GEO).
    Plots RL and Heuristic as separate panels.

    Requires images.csv column: target_regime (strings like LEO/MEO/GEO).
    """
    required = {"sat_shadow_cmd", "target_regime"}
    for df, name in [(imgs_rl, "RL"), (imgs_h, "Heuristic")]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} images missing columns: {missing}. "
                             f"Did you populate target_regime in evaluation/images.csv?")

    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    regimes_order = ["LEO", "MEO", "GEO"]

    def _compute(df):
        x = df["sat_shadow_cmd"].to_numpy(dtype=float)
        reg = df["target_regime"].astype(str).to_numpy()

        out = {r: np.full(nbins, np.nan) for r in regimes_order}
        n_bin = np.zeros(nbins, dtype=int)

        for i in range(nbins):
            b = (x >= edges[i]) & (x < edges[i + 1] if i < nbins - 1 else x <= edges[i + 1])
            rr = reg[b]
            rr = rr[np.isfinite(x[b])] if rr.size else rr
            n = rr.size
            n_bin[i] = n
            if n == 0:
                continue
            for r in regimes_order:
                out[r][i] = float(np.mean(rr == r))
        return out, n_bin

    rl_frac, rl_n = _compute(imgs_rl)
    h_frac, h_n = _compute(imgs_h)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    def _panel(ax, frac, n_bin, title):
        for r in regimes_order:
            ax.plot(centers, frac[r], marker="o", linewidth=2, label=r)

        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Fraction of chosen targets")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=9, loc="upper right")

        # annotate sample counts per bin at the bottom
        for i, c in enumerate(centers):
            if n_bin[i] > 0:
                ax.text(c, 0.02, f"n={n_bin[i]}", ha="center", fontsize=7, rotation=0)

    _panel(axes[0], rl_frac, rl_n, "RL: target regime selection vs eclipse state")
    _panel(axes[1], h_frac, h_n, "Heuristic: target regime selection vs eclipse state")

    axes[1].set_xlabel("sat_shadow_cmd bin center (0=eclipse, 1=sunlit)")
    fig.tight_layout()

    path_pdf = os.path.join(out_dir, "regime_fraction_vs_shadow_bin.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")


def _pick_latest_run(data_dir: str, prefix: str) -> str:
    """
    Picks the latest run folder by timestamp in the name: *_YYYYMMDD_HHMMSS
    Example: RL_seed20_20260108_205129
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    cands = []
    for name in os.listdir(data_dir):
        full = os.path.join(data_dir, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith(prefix):
            continue
        parts = name.split("_")
        # expect last two parts to be YYYYMMDD and HHMMSS (or combined like 205129)
        # your naming uses ..._20260108_205129
        if len(parts) < 3:
            continue
        date_str = parts[-2]
        time_str = parts[-1]
        if not (date_str.isdigit() and time_str.isdigit()):
            continue
        stamp = date_str + time_str
        cands.append((stamp, full))

    if not cands:
        raise FileNotFoundError(f"No folders found in {data_dir} starting with {prefix}")
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]

def _wrap_deg_180(a_deg):
    a = np.asarray(a_deg, dtype=float)
    a = (a + 180.0) % 360.0 - 180.0
    return a

def plot_azimuth_altitude_by_phase(imgs_rl, imgs_h, out_dir: str):
    """
    2x3 grid:
      rows: RL / Heuristic
      cols: entering / umbra / exiting

    Scatter: x=azimuth_deg (wrapped to [-180, 180]), y=target_alt_km
    Points are all imaging commands in each condition.
    """
    phases = ["entering", "umbra", "exiting"]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=True)

    def _panel(ax, df, title, phase):
        d = df[df["phase"] == phase].copy()
        if len(d) == 0:
            ax.set_title(f"{title}\n({phase}) n=0")
            ax.grid(True, alpha=0.25)
            return

        x = _wrap_deg_180(d["azimuth_deg"].to_numpy(dtype=float))
        y = d["target_alt_km"].to_numpy(dtype=float)

        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

        ax.scatter(x, y, s=10, alpha=0.25)
        ax.set_title(f"{title}\n({phase}) n={len(x)}")
        ax.grid(True, alpha=0.25)

    for j, ph in enumerate(phases):
        _panel(axes[0, j], imgs_rl, "RL", ph)
        _panel(axes[1, j], imgs_h, "Heuristic", ph)

    for ax in axes[-1, :]:
        ax.set_xlabel("Azimuth [deg] (wrapped to [-180,180]; +0 ≈ ahead, ±180 ≈ behind)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Target altitude [km]")

    fig.suptitle("Azimuth vs altitude under eclipse-related conditions", y=1.02)
    fig.tight_layout()

    path_pdf = os.path.join(out_dir, "azimuth_altitude_by_phase.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")


def plot_azimuth_altitude_early_late(imgs_rl, imgs_h, out_dir: str):
    """
    2x2 grid:
      rows: RL / Heuristic
      cols: early half / late half

    Scatter: x=azimuth_deg (wrapped), y=target_alt_km
    This supports the claim “policy clears one regime first” or shifts strategy over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True, sharey=True)

    def _split(df):
        t = df["t_cmd"].to_numpy(dtype=float)
        tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
        mid = 0.5 * (tmin + tmax)
        return df[df["t_cmd"] <= mid], df[df["t_cmd"] > mid]

    def _panel(ax, df, title):
        x = _wrap_deg_180(df["azimuth_deg"].to_numpy(dtype=float))
        y = df["target_alt_km"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        ax.scatter(x, y, s=10, alpha=0.25)
        ax.set_title(f"{title} n={len(x)}")
        ax.grid(True, alpha=0.25)

    rl_early, rl_late = _split(imgs_rl)
    h_early, h_late = _split(imgs_h)

    _panel(axes[0, 0], rl_early, "RL early half")
    _panel(axes[0, 1], rl_late,  "RL late half")
    _panel(axes[1, 0], h_early,  "Heuristic early half")
    _panel(axes[1, 1], h_late,   "Heuristic late half")

    for ax in axes[-1, :]:
        ax.set_xlabel("Azimuth [deg] (wrapped to [-180,180]; +0 ≈ ahead, ±180 ≈ behind)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Target altitude [km]")

    fig.suptitle("Azimuth vs altitude strategy drift over the episode", y=1.02)
    fig.tight_layout()

    path_pdf = os.path.join(out_dir, "azimuth_altitude_early_late.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")

def plot_dt_vs_target_shadow(imgs_rl, imgs_h, out_dir: str, use_acq_shadow: bool = False):
    """
    Scatter: x = dt_acq (successes only), y = target shadow (cmd or acq)
    Two panels: RL vs Heuristic

    If use_acq_shadow=True, expects target_shadow_acq column (recommended once you add it).
    Else uses target_shadow_cmd.
    """
    ycol = "target_shadow_acq" if use_acq_shadow else "target_shadow_cmd"
    if ycol not in imgs_rl.columns or ycol not in imgs_h.columns:
        print(f"Skipping dt vs shadow plot: missing column {ycol}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True)

    def _panel(ax, df, title):
        d = df[df["acq_success"] == 1].copy()
        x = d["dt_acq"].to_numpy(dtype=float)
        y = d[ycol].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        ax.scatter(x, y, s=10, alpha=0.25)
        ax.set_title(f"{title} (successes) n={len(x)}")
        ax.grid(True, alpha=0.25)

        # median marker
        if len(x):
            ax.scatter([np.median(x)], [np.median(y)], s=70, marker="x")

    _panel(axes[0], imgs_rl, "RL")
    _panel(axes[1], imgs_h,  "Heuristic")

    axes[0].set_ylabel(f"{ycol} (0=eclipse, 1=sunlit)")
    for ax in axes:
        ax.set_xlabel("Acquisition time dt_acq [s]")

    fig.suptitle("Acquisition-time vs illumination tradeoff", y=1.02)
    fig.tight_layout()

    name = "dt_vs_target_shadow_acq.pdf" if use_acq_shadow else "dt_vs_target_shadow_cmd.pdf"
    path_pdf = os.path.join(out_dir, name)
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_pdf}")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_dir", default=None, help="RL run folder (default: latest RL_* in examples/data)")
    ap.add_argument("--heur_dir", default=None, help="Heuristic run folder (default: latest HEUR_* in examples/data)")
    ap.add_argument("--out_dir", default="plots/compare_latest", help="Output directory for plots")
    args = ap.parse_args()

    # If user didn't pass dirs, auto-detect from examples/data
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")

    rl_dir = args.rl_dir or _pick_latest_run(data_dir, "RL_")
    heur_dir = args.heur_dir or _pick_latest_run(data_dir, "HEUR_")

    out_dir = _ensure_outdir(os.path.join(here, args.out_dir))

    steps_rl, imgs_rl = _load_run(rl_dir)
    steps_h, imgs_h = _load_run(heur_dir)

    plot_timeline_actions(steps_rl, imgs_rl, steps_h, imgs_h, out_dir)
    plot_lookahead_by_phase(imgs_rl, imgs_h, out_dir)
    plot_acquisition_cdf(imgs_rl, imgs_h, out_dir)
    plot_lookahead_vs_shadow_scatter(imgs_rl, imgs_h, out_dir)
    plot_regime_fraction_vs_shadow_bin(imgs_rl, imgs_h, out_dir)
    plot_azimuth_altitude_by_phase(imgs_rl, imgs_h, out_dir)
    plot_azimuth_altitude_early_late(imgs_rl, imgs_h, out_dir)
    plot_dt_vs_target_shadow(imgs_rl, imgs_h, out_dir, use_acq_shadow=False)  # works now
    # plot_dt_vs_target_shadow(imgs_rl, imgs_h, out_dir, use_acq_shadow=True)




    print("\nDone.")
    print("RL dir  :", rl_dir)
    print("Heur dir:", heur_dir)
    print("Plots   :", os.path.abspath(out_dir))



if __name__ == "__main__":
    main()
