#!/usr/bin/env python3
# mc_behaviour_analysis_v2.py

from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # IMPORTANT: no blocking plot windows
import matplotlib.pyplot as plt


# -------------------------
# Config (edit if you want)
# -------------------------
UMBRA_TAU = 0.05
SUNLIT_TAU = 0.95

DOWNLINK_DROP_TOL = 1e-6
# Storage fraction occupied by a single image (given by user)
IMAGE_STORAGE_FRAC = 0.02



TIME_BINS = [
    (0.0, 15000.0, "T1_0_15000"),
    (15000.0, 30000.0, "T2_15000_30000"),
    (30000.0, 45000.0, "T3_30000_45000"),
]

ENV_COLORS = {
    "LEO":   "#1f77b4",  # blue
    "MIXED": "#ff7f0e",  # orange
}

ACTION_NAMES = ["Imaging", "Charge", "Downlink", "Desat"]


def find_runs(env_dir: Path) -> list[Path]:
    """
    Find run folders that contain steps.csv and images.csv.
    Looks one level down (typical structure), but also supports nested.
    """
    env_dir = Path(env_dir)
    runs = []
    for p in env_dir.rglob("steps.csv"):
        run_dir = p.parent
        if (run_dir / "images.csv").is_file():
            runs.append(run_dir)
    # de-dup & sort
    runs = sorted(set(runs))
    return runs


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_metrics_json(run_dir: Path) -> dict:
    # metrics_*.json is optional
    cands = list(run_dir.glob("metrics_*.json"))
    if not cands:
        return {}
    try:
        with open(cands[0], "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _nested_get(d: dict, path: tuple[str, ...]):
    """Best-effort nested getter; returns None if any level missing."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def circmean_deg(deg: np.ndarray) -> float:
    """Circular mean for angles in degrees."""
    deg = np.asarray(deg, dtype=float)
    deg = deg[np.isfinite(deg)]
    if deg.size == 0:
        return float("nan")
    rad = np.deg2rad(deg)
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    return float((np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0)


def summarize_one_run(run_dir: Path, env_name: str) -> dict:
    steps = safe_read_csv(run_dir / "steps.csv")
    imgs  = safe_read_csv(run_dir / "images.csv")
    mj    = load_metrics_json(run_dir)

    out = {
        "env": env_name,
        "run_dir": str(run_dir),
    }

    # -------------------------
    # Total reward (from steps)
    # -------------------------
    if "reward_cum" in steps.columns and len(steps):
        out["total_reward"] = float(pd.to_numeric(steps["reward_cum"], errors="coerce").dropna().iloc[-1])
    else:
        out["total_reward"] = float("nan")

    # -------------------------
    # Action distribution
    # -------------------------
    if "action_id" in steps.columns and len(steps):
        a = pd.to_numeric(steps["action_id"], errors="coerce").dropna().astype(int)
        n = max(len(a), 1)

        imaging = ((a >= 0) & (a <= 9)).sum()
        charge  = (a == 10).sum()
        down    = (a == 11).sum()
        desat   = (a == 12).sum()

        out["frac_imaging"]  = imaging / n
        out["frac_charge"]   = charge  / n
        out["frac_downlink"] = down    / n
        out["frac_desat"]    = desat   / n

        out["n_downlink_actions"] = int(down)
    else:
        out.update({k: float("nan") for k in ["frac_imaging","frac_charge","frac_downlink","frac_desat"]})
        out["n_downlink_actions"] = 0

    # -------------------------
    # Downlink effectiveness
    #
    # Count a downlink as successful ONLY when storage actually goes down.
    # Amount downlinked per (successful) downlink action is:
    #   drop = storage_frac_cmd - storage_frac_after
    #
    # Also report the equivalent number of images, where 1 image occupies
    # IMAGE_STORAGE_FRAC of storage (default: 0.02).
    # -------------------------
    out["downlink_success_rate"] = float("nan")        # frac of valid downlinks with drop > tol
    out["downlink_amount_mean"] = float("nan")         # mean drop (storage fraction) over successful downlinks
    out["downlink_amount_std"]  = float("nan")
    out["downlink_amount_sum"]  = float("nan")
    out["downlink_amount_n"]    = 0

    out["downlink_images_mean"] = float("nan")         # mean drop converted to image-count units
    out["downlink_images_sum"]  = float("nan")

    if {"action_id", "storage_frac_cmd", "storage_frac_after"}.issubset(steps.columns):
        d = steps[pd.to_numeric(steps["action_id"], errors="coerce") == 11].copy()
        if len(d):
            cmd  = pd.to_numeric(d["storage_frac_cmd"], errors="coerce").to_numpy(dtype=float)
            aft  = pd.to_numeric(d["storage_frac_after"], errors="coerce").to_numpy(dtype=float)
            drop = cmd - aft

            valid = np.isfinite(drop)
            succ  = valid & (drop > DOWNLINK_DROP_TOL)

            denom = int(valid.sum())
            out["downlink_success_rate"] = float(succ.sum() / denom) if denom else float("nan")

            if succ.any():
                vals = drop[succ]
                out["downlink_amount_mean"] = float(vals.mean())
                out["downlink_amount_std"]  = float(vals.std(ddof=1) if vals.size > 1 else 0.0)
                out["downlink_amount_sum"]  = float(vals.sum())
                out["downlink_amount_n"]    = int(vals.size)

                if IMAGE_STORAGE_FRAC > 0:
                    out["downlink_images_mean"] = float(out["downlink_amount_mean"] / IMAGE_STORAGE_FRAC)
                    out["downlink_images_sum"]  = float(out["downlink_amount_sum"]  / IMAGE_STORAGE_FRAC)

            else:
                # No successful downlinks (storage never decreased)
                out["downlink_amount_mean"] = 0.0
                out["downlink_amount_std"]  = 0.0
                out["downlink_amount_sum"]  = 0.0
                out["downlink_amount_n"]    = 0
                out["downlink_images_mean"] = 0.0
                out["downlink_images_sum"]  = 0.0



    # -------------------------
    # Illuminated images taken (success + illuminated at acquisition time)
    # -------------------------
    # IMPORTANT:
    # Prefer the authoritative value from metrics_*.json (produced by the MC runner)
    # rather than re-deriving it from images.csv columns, which can drift depending
    # on logging schema / thresholds.
    out["illum_images_taken"] = float("nan")
    out["acq_success_rate_img_cmds"] = float("nan")
    out["mean_dt_acq_success"] = float("nan")

    # 1) metrics json (preferred)
    n_illum = None
    # common locations
    for path in [
        ("data", "illuminated_images"),
        ("illuminated_images",),
        ("summary", "illuminated_images"),
    ]:
        v = _nested_get(mj, path) if len(path) > 1 else (mj.get(path[0]) if isinstance(mj, dict) else None)
        if v is not None:
            n_illum = v
            break
    if n_illum is not None:
        try:
            out["illum_images_taken"] = int(float(n_illum))
        except Exception:
            pass

    if len(imgs):
        # success rate over imaging commands
        if "acq_success" in imgs.columns:
            s = pd.to_numeric(imgs["acq_success"], errors="coerce").fillna(0).astype(int).to_numpy()
            out["acq_success_rate_img_cmds"] = float(s.mean())

        # dt_acq for successes
        if {"acq_success","dt_acq"}.issubset(imgs.columns):
            succ = pd.to_numeric(imgs["acq_success"], errors="coerce").fillna(0).astype(int) == 1
            dt   = pd.to_numeric(imgs["dt_acq"], errors="coerce")
            vals = dt[succ].dropna().to_numpy()
            if vals.size:
                out["mean_dt_acq_success"] = float(vals.mean())

        # 2) fallback: approximate from images.csv if json didn't have it
        if not np.isfinite(out["illum_images_taken"]):
            illum_col = "target_shadow_acq" if "target_shadow_acq" in imgs.columns else "target_shadow_cmd"
            if {"acq_success", illum_col}.issubset(imgs.columns):
                succ = pd.to_numeric(imgs["acq_success"], errors="coerce").fillna(0).astype(int) == 1
                illum = pd.to_numeric(imgs[illum_col], errors="coerce")
                out["illum_images_taken"] = int(((illum >= SUNLIT_TAU) & succ).sum())

    # -------------------------
    # Illuminated images downlinked (prefer metrics json if available)
    # (Your rewarder tracks useful_downlinks as "unique illuminated downlinked")
    # -------------------------
    out["illum_images_downlinked"] = float("nan")
    for key in ["useful_downlinks", "usefully_downlinked", "usefully_downlinked_count"]:
        if key in mj:
            try:
                out["illum_images_downlinked"] = float(mj[key])
            except Exception:
                pass

    # Sometimes nested:
    if np.isnan(out["illum_images_downlinked"]) and isinstance(mj, dict):
        # try common nested spots
        for k in ["summary", "data", "metrics"]:
            if k in mj and isinstance(mj[k], dict) and "useful_downlinks" in mj[k]:
                try:
                    out["illum_images_downlinked"] = float(mj[k]["useful_downlinks"])
                except Exception:
                    pass

    # -------------------------
    # Regime fractions overall + successes
    # -------------------------
    for suffix, mask in [("all", None), ("succ", "acq_success")]:
        out[f"frac_{suffix}_LEO"] = float("nan")
        out[f"frac_{suffix}_MEO"] = float("nan")
        out[f"frac_{suffix}_GEO"] = float("nan")

    if len(imgs) and "target_regime" in imgs.columns:
        reg = imgs["target_regime"].astype(str)

        def _frac(series: pd.Series) -> dict:
            vc = series.value_counts(dropna=False)
            tot = float(vc.sum()) if len(vc) else 0.0
            return {k: float(v)/tot for k, v in vc.items()} if tot > 0 else {}

        f_all = _frac(reg)
        out["frac_all_LEO"] = f_all.get("LEO", 0.0)
        out["frac_all_MEO"] = f_all.get("MEO", 0.0)
        out["frac_all_GEO"] = f_all.get("GEO", 0.0)

        if "acq_success" in imgs.columns:
            succ = pd.to_numeric(imgs["acq_success"], errors="coerce").fillna(0).astype(int) == 1
            f_s = _frac(reg[succ])
            out["frac_succ_LEO"] = f_s.get("LEO", 0.0)
            out["frac_succ_MEO"] = f_s.get("MEO", 0.0)
            out["frac_succ_GEO"] = f_s.get("GEO", 0.0)

    return out


def summarize_time_thirds(imgs: pd.DataFrame, env: str, run_dir: str) -> list[dict]:
    """
    For each third: regime fractions + success + dt_acq by regime.
    """
    rows = []
    if imgs.empty or "t_cmd" not in imgs.columns:
        return rows

    t = pd.to_numeric(imgs["t_cmd"], errors="coerce")
    reg = imgs["target_regime"].astype(str) if "target_regime" in imgs.columns else pd.Series(["UNK"]*len(imgs))
    succ = pd.to_numeric(imgs["acq_success"], errors="coerce").fillna(0).astype(int) if "acq_success" in imgs.columns else pd.Series([0]*len(imgs))
    dt   = pd.to_numeric(imgs["dt_acq"], errors="coerce") if "dt_acq" in imgs.columns else pd.Series([np.nan]*len(imgs))

    for (t0, t1, label) in TIME_BINS:
        m = (t >= t0) & (t < t1)
        if not m.any():
            continue
        reg_m = reg[m]
        succ_m = succ[m].to_numpy().astype(int)
        dt_m = dt[m]

        vc = reg_m.value_counts()
        tot = float(vc.sum()) if len(vc) else 0.0

        def frac(name: str) -> float:
            return float(vc.get(name, 0.0))/tot if tot > 0 else float("nan")

        row = {
            "env": env,
            "run_dir": run_dir,
            "time_bin": label,
            "N_cmds": int(m.sum()),
            "frac_LEO": frac("LEO"),
            "frac_MEO": frac("MEO"),
            "frac_GEO": frac("GEO"),
            "acq_success_rate": float(succ_m.mean()) if succ_m.size else float("nan"),
        }

        # dt_acq mean for successes by regime
        for rname in ["LEO","MEO","GEO"]:
            mr = (reg_m == rname)
            ms = mr.to_numpy() & (succ_m == 1)
            vals = dt_m[ms].dropna().to_numpy()
            row[f"dt_acq_mean_succ_{rname}"] = float(vals.mean()) if vals.size else float("nan")

        rows.append(row)

    return rows


def summarize_eclipse_pointing(imgs: pd.DataFrame, env: str, run_dir: str) -> dict:
    """
    Pointing stats entering / umbra / exiting.
    Uses:
      - entering: phase_slope == entering OR phase == entering
      - umbra: sat_shadow_cmd <= UMBRA_TAU OR phase_state == umbra
      - exiting: phase_slope == exiting OR phase == exiting
    """
    out = {"env": env, "run_dir": run_dir}
    if imgs.empty:
        return out

    def col(name: str) -> pd.Series:
        return imgs[name] if name in imgs.columns else pd.Series([np.nan]*len(imgs))

    sat_sf = pd.to_numeric(col("sat_shadow_cmd"), errors="coerce")
    phase = col("phase").astype(str)
    phase_state = col("phase_state").astype(str)
    phase_slope = col("phase_slope").astype(str)

    az = pd.to_numeric(col("azimuth_deg"), errors="coerce")
    el = pd.to_numeric(col("elevation_local_deg"), errors="coerce")
    la = pd.to_numeric(col("look_ahead"), errors="coerce")

    succ = pd.to_numeric(col("acq_success"), errors="coerce").fillna(0).astype(int)
    dt   = pd.to_numeric(col("dt_acq"), errors="coerce")
    reg  = col("target_regime").astype(str)

    masks = {
        "entering": (phase_slope == "entering") | (phase == "entering"),
        "umbra":    (sat_sf <= UMBRA_TAU) | (phase_state == "umbra"),
        "exiting":  (phase_slope == "exiting") | (phase == "exiting"),
    }

    for k, m in masks.items():
        m = m.to_numpy()
        out[f"N_{k}"] = int(m.sum())

        out[f"az_circmean_{k}"] = circmean_deg(az[m].to_numpy())
        out[f"el_mean_{k}"] = float(el[m].mean()) if np.isfinite(el[m]).any() else float("nan")
        out[f"lookahead_mean_{k}"] = float(la[m].mean()) if np.isfinite(la[m]).any() else float("nan")

        # success + dt_acq in this segment
        succ_m = succ[m].to_numpy()
        out[f"acq_success_rate_{k}"] = float(succ_m.mean()) if succ_m.size else float("nan")
        vals = dt[m][succ[m] == 1].dropna().to_numpy()
        out[f"dt_acq_mean_succ_{k}"] = float(vals.mean()) if vals.size else float("nan")

        # regime split in this segment
        if "target_regime" in imgs.columns:
            vc = reg[m].value_counts()
            tot = float(vc.sum()) if len(vc) else 0.0
            out[f"frac_LEO_{k}"] = float(vc.get("LEO", 0.0))/tot if tot > 0 else float("nan")
            out[f"frac_MEO_{k}"] = float(vc.get("MEO", 0.0))/tot if tot > 0 else float("nan")
            out[f"frac_GEO_{k}"] = float(vc.get("GEO", 0.0))/tot if tot > 0 else float("nan")

    return out


def mean_pm_std(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "—"
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def make_action_overlay_plot(df: pd.DataFrame, out_pdf: Path, title: str):
    """
    Bar plot: mean action fractions with std error bars.
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    width = 0.35
    xs = np.arange(len(ACTION_NAMES))

    for j, env in enumerate(["LEO", "MIXED"]):
        d = df[df["env"] == env]
        means = np.array([
            pd.to_numeric(d["frac_imaging"], errors="coerce").mean(),
            pd.to_numeric(d["frac_charge"], errors="coerce").mean(),
            pd.to_numeric(d["frac_downlink"], errors="coerce").mean(),
            pd.to_numeric(d["frac_desat"], errors="coerce").mean(),
        ])
        stds = np.array([
            pd.to_numeric(d["frac_imaging"], errors="coerce").std(ddof=1),
            pd.to_numeric(d["frac_charge"], errors="coerce").std(ddof=1),
            pd.to_numeric(d["frac_downlink"], errors="coerce").std(ddof=1),
            pd.to_numeric(d["frac_desat"], errors="coerce").std(ddof=1),
        ])

        ax.bar(xs + (j-0.5)*width, means, width=width,
               yerr=stds, capsize=3,
               label=env, color=ENV_COLORS[env], alpha=0.55 if env=="MIXED" else 0.80)

    ax.set_xticks(xs)
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_ylabel("Fraction of steps")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_downlink_amount_overlay(df: pd.DataFrame, out_pdf: Path, title: str):
    """
    Distribution of downlink amount (storageFrac drop) for successful downlinks.
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.6))

    for env in ["LEO", "MIXED"]:
        d = df[df["env"] == env]
        # use the per-seed mean amount; keeps it clean (100 points rather than 100k events)
        vals = pd.to_numeric(d["downlink_amount_mean"], errors="coerce").dropna().to_numpy()
        if vals.size:
            ax.hist(vals, bins=18, alpha=0.55 if env=="MIXED" else 0.75,
                    label=env, color=ENV_COLORS[env])

    ax.set_xlabel("Mean storage fraction drop per successful downlink (per seed)")
    ax.set_ylabel("Count (seeds)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
def write_overall_latex_table(env_summary: pd.DataFrame, out_tex: Path, caption: str):
    """
    Small interpretable table for the paper.
    Only writes rows for environments that exist in env_summary.
    """
    rows = []
    for env in ["LEO", "MIXED"]:
        if env not in env_summary.index:
            continue
        r = env_summary.loc[env]
        print(f"DownMean is {r['DownMean']}")
        rows.append(
            f"{env} & {int(r['N'])} & {r['IllumTaken']} & {r['IllumDown']} & {r['Reward']} & "
            f"{r['MeanDtAcq']} & {r['DownMean']} \\\\"
        )

    if not rows:
        print(f"⚠️ No environments found in env_summary for LaTeX table {out_tex}")
        return

    tex = r"""\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\small
\begin{tabular}{l r r r r r r}
\toprule
Env & $N$ & Illum. images taken & Illum. images downlinked & Total reward & Mean $dt_{acq}$ (succ) [s] & Downlink frac \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_tex.write_text(tex)
#
# def write_overall_latex_table(env_summary: pd.DataFrame, out_tex: Path, caption: str):
#     """
#     Small interpretable table for the paper.
#     """
#     # expect env_summary index env with columns already formatted as strings
#     rows = []
#     for env in ["LEO", "MIXED"]:
#         r = env_summary.loc[env]
#         rows.append(
#             f"{env} & {int(r['N'])} & {r['IllumTaken']} & {r['IllumDown']} & {r['Reward']} & "
#             f"{r['MeanDtAcq']} & {r['DownFrac']} \\\\"
#         )
#
#     tex = r"""\begin{table}[t]
# \centering
# \caption{""" + caption + r"""}
# \small
# \begin{tabular}{l r r r r r r}
# \toprule
# Env & $N$ & Illum. images taken & Illum. images downlinked & Total reward & Mean $dt_{acq}$ (succ) [s] & Downlink frac \\
# \midrule
# """ + "\n".join(rows) + r"""
# \bottomrule
# \end{tabular}
# \end{table}
# """
#     out_tex.write_text(tex)


def main():
    # import argparse
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--policy_group", type=str, default="RL50d50i")
    # ap.add_argument("--leo_dir", type=str, required=True)
    # ap.add_argument("--mixed_dir", type=str, required=True)
    # ap.add_argument("--out_root", type=str, default="analysis_out")
    # args = ap.parse_args()
    #
    # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_dir = Path(args.out_root) / args.policy_group / ts
    # out_dir.mkdir(parents=True, exist_ok=True)
    #
    # leo_runs = find_runs(Path(args.leo_dir))
    # mix_runs = find_runs(Path(args.mixed_dir))
    #
    # print(f"Found {len(leo_runs)} runs for env=LEO in {args.leo_dir}")
    # print(f"Found {len(mix_runs)} runs for env=MIXED in {args.mixed_dir}")

    # -------------------------
    # HARDCODED CONFIG (no CLI needed)
    # -------------------------
    # policy_group = "RL50d50i"   # or any other alpha label like "RL00d100i"
    # leo_dir = Path("examples/data/RL50d50i_LEO")
    # mixed_dir = Path("examples/data/RL50d50i_mixed")
    # out_root = Path("analysis_out")  # results will go here
    #
    #
    # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_dir = out_root / policy_group / ts
    # out_dir.mkdir(parents=True, exist_ok=True)
    #
    # leo_runs = find_runs(leo_dir)
    # mix_runs = find_runs(mixed_dir)
    #
    # print(f"Found {len(leo_runs)} runs for env=LEO in {leo_dir}")
    # print(f"Found {len(mix_runs)} runs for env=MIXED in {mixed_dir}")




    # # Per-run metrics
    # rows = []
    # thirds_rows = []
    # eclipse_rows = []
    #
    # for env_name, runs in [("LEO", leo_runs), ("MIXED", mix_runs)]:
    #     for run in runs:
    #         r = summarize_one_run(run, env_name)
    #         rows.append(r)
    #
    #         imgs = safe_read_csv(Path(run) / "images.csv")
    #         thirds_rows += summarize_time_thirds(imgs, env_name, str(run))
    #         eclipse_rows.append(summarize_eclipse_pointing(imgs, env_name, str(run)))
    #
    # df = pd.DataFrame(rows)
    # df_thirds = pd.DataFrame(thirds_rows)
    # df_ecl = pd.DataFrame(eclipse_rows)
    #
    # # Save raw per-seed metrics
    # df.to_csv(out_dir / "seed_metrics.csv", index=False)
    # df_thirds.to_csv(out_dir / "phase_regime_summary.csv", index=False)
    # df_ecl.to_csv(out_dir / "eclipse_pointing_summary.csv", index=False)
    #
    # # Summaries for paper (mean ± std)
    # env_summary = []
    # for env in ["LEO", "MIXED"]:
    #     d = df[df["env"] == env]
    #     env_summary.append({
    #         "env": env,
    #         "N": len(d),
    #         "IllumTaken": mean_pm_std(d["illum_images_taken"]),
    #         "IllumDown": mean_pm_std(d["illum_images_downlinked"]),
    #         "Reward": mean_pm_std(d["total_reward"]),
    #         "MeanDtAcq": mean_pm_std(d["mean_dt_acq_success"]),
    #         "DownFrac": mean_pm_std(d["frac_downlink"]),
    #     })
    #
    # env_summary = pd.DataFrame(env_summary).set_index("env")
    # env_summary.to_csv(out_dir / "env_summary.csv")
    #
    # # LaTeX table
    # write_overall_latex_table(
    #     env_summary,
    #     out_dir / "table_overall.tex",
    #     caption="Policy robustness: trained in LEO, evaluated in LEO vs mixed target regimes. Values are mean $\\pm$ std across Monte Carlo seeds."
    # )


    # # Plots
    # make_action_overlay_plot(
    #     df,
    #     out_dir / "action_distribution_overlay.pdf",
    #     title=f"{args.policy_group}: action distribution (mean±std across seeds)"
    # )
    #
    # make_downlink_amount_overlay(
    #     df,
    #     out_dir / "downlink_amount_overlay.pdf",
    #     title=f"{args.policy_group}: downlink amount per successful downlink"
    # )
    # make_action_overlay_plot(
    # df,
    # out_dir / "action_distribution_overlay.pdf",
    # title=f"{policy_group}: action distribution (mean±std across seeds)"
    # )
    #
    # make_downlink_amount_overlay(
    #     df,
    #     out_dir / "downlink_amount_overlay.pdf",
    #     title=f"{policy_group}: downlink amount per successful downlink"
    # )
    #
    #
    # print("\nWrote analysis outputs to:")
    # print(f"  {out_dir}")
    # =========================
    # HARDCODED POLICY DIRECTORIES
    # =========================
    # Only mixed policies
    policy_dirs = {
        "RL100d0i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL100d0i_mixed",
        "RL90d10i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL90d10i_mixed",
        "RL80d20i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL80d20i_mixed",
        "RL70d30i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL70d30i_mixed",
        "RL60d40i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL60d40i_mixed",
        "RL50d50i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL50d50i_mixed",
        "RL40d60i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL40d60i_mixed",
        "RL30d70i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL30d70i_mixed",
        "RL20d80i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL20d80i_mixed",
        "RL10d90i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL10d90i_mixed",
        "RL0d100i": "/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data/RL0d100i_mixed",
    }

    out_root = Path("analysis_out")  # all results will go here
    # for policy_group, mixed_dir in policy_dirs.items():
    #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     out_dir = out_root / policy_group / ts
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # find runs
    #     mix_runs = find_runs(Path(mixed_dir))
    #     print(f"Found {len(mix_runs)} runs for env=MIXED in {mixed_dir}")
    #
    #     # per-run metrics
    #     rows = []
    #     thirds_rows = []
    #     eclipse_rows = []
    #
    #     for run in mix_runs:
    #         r = summarize_one_run(run, "MIXED")
    #         rows.append(r)
    #
    #         imgs = safe_read_csv(Path(run) / "images.csv")
    #         thirds_rows += summarize_time_thirds(imgs, "MIXED", str(run))
    #         eclipse_rows.append(summarize_eclipse_pointing(imgs, "MIXED", str(run)))
    #
    #     df = pd.DataFrame(rows)
    #     df_thirds = pd.DataFrame(thirds_rows)
    #     df_ecl = pd.DataFrame(eclipse_rows)
    #
    #     # save CSVs
    #     df.to_csv(out_dir / "seed_metrics.csv", index=False)
    #     df_thirds.to_csv(out_dir / "phase_regime_summary.csv", index=False)
    #     df_ecl.to_csv(out_dir / "eclipse_pointing_summary.csv", index=False)
    #
    #     # compute mean ± std summary
    #     env_summary = [{
    #         "env": "MIXED",
    #         "N": len(df),
    #         "IllumTaken": mean_pm_std(df["illum_images_taken"]),
    #         "IllumDown": mean_pm_std(df["illum_images_downlinked"]),
    #         "Reward": mean_pm_std(df["total_reward"]),
    #         "MeanDtAcq": mean_pm_std(df["mean_dt_acq_success"]),
    #         "DownFrac": mean_pm_std(df["frac_downlink"]),
    #     }]
    #     env_summary = pd.DataFrame(env_summary).set_index("env")
    #     env_summary.to_csv(out_dir / "env_summary.csv")
    #
    #     # LaTeX table
    #     write_overall_latex_table(
    #         env_summary,
    #         out_dir / "table_overall.tex",
    #         caption=f"Policy {policy_group}: evaluated in MIXED regime. Mean ± std across Monte Carlo seeds."
    #     )
    #
    #     # plots
    #     make_action_overlay_plot(
    #         df,
    #         out_dir / "action_distribution_overlay.pdf",
    #         title=f"{policy_group}: action distribution (mean±std across seeds)"
    #     )
    #     make_downlink_amount_overlay(
    #         df,
    #         out_dir / "downlink_amount_overlay.pdf",
    #         title=f"{policy_group}: downlink amount per successful downlink"
    #     )
    #
    #     print(f"Wrote analysis outputs to: {out_dir}\n")
    for policy_group, mixed_dir in policy_dirs.items():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / policy_group / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        mixed_path = Path(mixed_dir)
        if not mixed_path.exists():
            print(f"⚠️ Directory {mixed_path} does not exist, skipping {policy_group}")
            continue

        mix_runs = find_runs(mixed_path)
        if not mix_runs:
            print(f"⚠️ No runs found for {policy_group} in {mixed_path}, skipping.")
            continue

        print(f"Found {len(mix_runs)} runs for env=MIXED in {mixed_path}")

        rows = []
        thirds_rows = []
        eclipse_rows = []

        for run in mix_runs:
            r = summarize_one_run(run, "MIXED")
            rows.append(r)

            imgs = safe_read_csv(Path(run) / "images.csv")
            thirds_rows += summarize_time_thirds(imgs, "MIXED", str(run))
            eclipse_rows.append(summarize_eclipse_pointing(imgs, "MIXED", str(run)))

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"⚠️ No valid data for {policy_group}, skipping summary and plots.")
            continue
        df = pd.DataFrame(rows)
        df_thirds = pd.DataFrame(thirds_rows)
        df_ecl = pd.DataFrame(eclipse_rows)

        # Save raw per-seed metrics
        df.to_csv(out_dir / "seed_metrics.csv", index=False)
        df_thirds.to_csv(out_dir / "phase_regime_summary.csv", index=False)
        df_ecl.to_csv(out_dir / "eclipse_pointing_summary.csv", index=False)

        # compute mean ± std summary
        env_summary = [{
            "env": "MIXED",
            "N": len(df),
            "IllumTaken": mean_pm_std(df["illum_images_taken"]),
            "IllumDown": mean_pm_std(df["illum_images_downlinked"]),
            "Reward": mean_pm_std(df["total_reward"]),
            "MeanDtAcq": mean_pm_std(df["mean_dt_acq_success"]),
            "DownMean": mean_pm_std(df["downlink_amount_mean"]),
            "DownAmtImgs": mean_pm_std(df["downlink_images_mean"]),

        }]
        env_summary = pd.DataFrame(env_summary).set_index("env")
        env_summary.to_csv(out_dir / "env_summary.csv")

        # LaTeX table
        write_overall_latex_table(
            env_summary,
            out_dir / "table_overall.tex",
            caption=f"Policy {policy_group}: evaluated in MIXED regime. Mean ± std across Monte Carlo seeds."
        )

        # plots
        make_action_overlay_plot(
            df,
            out_dir / "action_distribution_overlay.pdf",
            title=f"{policy_group}: action distribution (mean±std across seeds)"
        )
        make_downlink_amount_overlay(
            df,
            out_dir / "downlink_amount_overlay.pdf",
            title=f"{policy_group}: downlink amount per successful downlink"
        )


if __name__ == "__main__":
    main()
