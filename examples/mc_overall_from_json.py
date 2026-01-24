#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd

# =========================
# USER CONFIGURATION
# =========================

# # DATA_ROOT = Path("data")      # examples/data
# OUTPUT_DIR = Path("results")
# OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 0.02

# =========================
# USER CONFIGURATION
# =========================
POLICY_DIRS = {
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
# Fallback for helpers
DATA_ROOT = Path("/Users/dahu1128/Repositories/bsk_rl/examples/data/GNC26_data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# HELPERS
# =========================

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_alpha_from_name(name: str) -> float | None:
    """
    'RL70d30i_mixed' or policy_name containing '70d30i' → alpha = 0.7
    """
    m = re.search(r"(\d{1,3})d(\d{1,3})i", name)
    if not m:
        return None
    return float(m.group(1)) / 100.0

def is_mixed_policy(path: Path) -> bool:
    return "_mixed" in path.as_posix().lower()


def infer_env_from_path(p: Path) -> str:
    s = p.as_posix().lower()
    if "leo" in s:
        return "LEO"
    if "mixed" in s:
        return "MIXED"
    return "UNKNOWN"

def find_metric_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("metrics_*.json"))

def mean_pm_std(x: pd.Series, fmt="{:.2f}") -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "—"
    return f"{fmt.format(x.mean())} ± {fmt.format(x.std(ddof=1))}"

def get_frac(d: dict, key: str) -> float:
    if not isinstance(d, dict):
        return 0.0
    return float(d.get(key, 0.0))



def estimate_useful_downlinks(
    R: float,
    N_illum: float,
    alpha: float | None,
    run_dir: str,
    img_size: float,
) -> float:
    if alpha is None or not np.isfinite(alpha):
        return float("nan")

    if alpha > 0.0:
        return (R - (1.0 - alpha) * N_illum) / alpha

    # alpha == 0 case
    s_final = final_storage_frac_after(run_dir)
    if not np.isfinite(s_final):
        return float("nan")

    n_remaining = s_final / img_size
    return max(0.0, N_illum - n_remaining)

# =========================
# LOAD ONE JSON
# =========================
def resolve_run_dir_from_json(json_path: Path, meta_run_dir: str | None) -> Path:
    """
    Resolve the true run folder (the directory that contains steps.csv) for a given metrics_*.json file.
    Strategy:
      1) If meta_run_dir is provided and (meta_run_dir / steps.csv) exists -> use it.
      2) If json_path.parent / steps.csv exists -> use json_path.parent
      3) Try DATA_ROOT / basename(meta_run_dir) (useful if meta_run_dir was recorded without the policy parent)
      4) Try json_path.parent.parent / basename(meta_run_dir)
      5) Otherwise return json_path.parent (best-effort fallback)
    """
    candidates: list[Path] = []

    if meta_run_dir:
        candidates.append(Path(meta_run_dir))
    # The directory that actually contains the json is the best fallback
    candidates.append(json_path.parent)
    # try sane guesses relative to DATA_ROOT
    if meta_run_dir:
        candidates.append(DATA_ROOT / Path(meta_run_dir).name)
        # candidate: policy folder + meta basename
        if json_path.parent.parent.exists():
            candidates.append(json_path.parent.parent / Path(meta_run_dir).name)

    # Also include the json's parent.parent (policy folder) as a last try (in case steps.csv is one level up)
    if json_path.parent.parent.exists():
        candidates.append(json_path.parent.parent)

    for c in candidates:
        try:
            if (c / "steps.csv").exists():
                return c.resolve()
        except Exception:
            continue

    # final fallback: return the json parent (even if steps.csv missing) so other code won't break
    return json_path.parent.resolve()


def final_storage_frac_after(run_dir: Path | str) -> float:
    """
    Read steps.csv and return final storage_frac_after.
    Accepts a Path or string. Returns nan if missing/unreadable.
    """
    p = Path(run_dir)
    steps = p / "steps.csv"
    if not steps.exists():
        return float("nan")
    try:
        df = pd.read_csv(steps)
        if "storage_frac_after" not in df.columns or len(df) == 0:
            return float("nan")
        return float(df["storage_frac_after"].iloc[-1])
    except Exception:
        return float("nan")

def load_one(path: Path) -> dict:
    """
    Load a single metrics_*.json Path -> dict of extracted metrics.
    This function now resolves the correct run_dir using resolve_run_dir_from_json.
    """
    j = json.loads(path.read_text())

    meta = j.get("meta", {}) or {}
    data = j.get("data", {}) or {}
    summ = j.get("summary", {}) or {}

    # policy_name: prefer meta, fallback to parent policy folder name
    policy_name = meta.get("policy_name") or path.parent.parent.name if path.parent.parent.name else path.parent.name
    alpha = parse_alpha_from_name(policy_name)
    if alpha == 0.8:
        print("reading 80d20i file")
    env = infer_env_from_path(path)

    regime_metrics = summ.get("regime_metrics", {}) or {}
    frac_all = regime_metrics.get("frac_target_regime_all", {}) or {}

    R = data.get("cumulativeRewardSS1", np.nan)
    N = data.get("illuminated_images", np.nan)

    # Resolve the run_dir robustly
    meta_run_dir = meta.get("run_dir", None)
    resolved_run_dir = resolve_run_dir_from_json(path, meta_run_dir)

    # Prepare outputs
    out = {
        "env": env,
        "alpha": alpha,
        "policy_name": policy_name,
        "seed": meta.get("seed"),

        "total_reward": R,
        "illuminated_images": N,

        "target_imaging_count": summ.get("target_imaging_count", np.nan),
        "downlink_action_count": summ.get("downlink_action_count", np.nan),
        "acq_success_rate": summ.get("acq_success_rate", np.nan),
        "avg_acquisition_time_sec": summ.get("avg_acquisition_time_sec", np.nan),

        "frac_all_LEO": get_frac(frac_all, "LEO"),
        "frac_all_MEO": get_frac(frac_all, "MEO"),
        "frac_all_GEO": get_frac(frac_all, "GEO"),
    }

    # estimate useful downlinks using the resolved run_dir
    try:
        r_val = float(R) if R is not None else float("nan")
    except Exception:
        r_val = float("nan")
    try:
        n_val = float(N) if N is not None else float("nan")
    except Exception:
        n_val = float("nan")

    out["resolved_run_dir"] = str(resolved_run_dir)
    if np.isfinite(r_val) and np.isfinite(n_val) and alpha is not None:
        out["useful_downlinks_est"] = estimate_useful_downlinks(r_val, n_val, alpha, str(resolved_run_dir), IMG_SIZE)
    else:
        out["useful_downlinks_est"] = np.nan

    return out


# =========================
# SUMMARIZATION
# =========================

def summarize_by_alpha_env(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alpha, env), d in df.groupby(["alpha", "env"]):
        rows.append({
            "alpha": alpha,
            "Env": env,
            "N": len(d),
            "Total reward": mean_pm_std(d["total_reward"]),
            "Illuminated images": mean_pm_std(d["illuminated_images"]),
            "Downlink actions": mean_pm_std(d["downlink_action_count"]),
            "Imaging actions": mean_pm_std(d["target_imaging_count"]),
            "Acq success rate": mean_pm_std(d["acq_success_rate"], "{:.3f}"),
            "Mean dt_acq [s]": mean_pm_std(d["avg_acquisition_time_sec"]),
            "Useful downlinks (est)": mean_pm_std(d["useful_downlinks_est"]),
        })
    return pd.DataFrame(rows).sort_values(["alpha", "Env"])

def to_latex_table(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    header = " & ".join(cols) + r" \\"
    body = "\n".join(
        " & ".join(str(v) for v in row) + r" \\"
        for row in df.values
    )
    return (
        r"\begin{table}[t]\centering\small"
        "\n"
        r"\begin{tabular}{" + "l" * len(cols) + "}"
        "\n\\toprule\n"
        + header + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

# =========================
# MAIN
# =========================

# def main():
#     # metric_files = find_metric_jsons(DATA_ROOT) #takes in all folder of that form...
#     metric_files = [
#         p for p in find_metric_jsons(DATA_ROOT)
#         if is_mixed_policy(p)
#     ]
#
#     if not metric_files:
#         raise RuntimeError("No metrics_*.json files found under data/")
def main():
    metric_files = []
    for policy_tag, path_str in POLICY_DIRS.items():
        p = Path(path_str)
        if p.exists():
            # This finds all json files within YOUR specific folders
            found = list(p.rglob("metrics_*.json"))
            print(f"Policy {policy_tag}: Found {len(found)} files")
            metric_files.extend(found)
        else:
            print(f"Warning: Path not found for {policy_tag}")

    if not metric_files:
        raise RuntimeError("No metrics_*.json files found. Check your absolute paths!")

    rows = [load_one(p) for p in metric_files]
    df = pd.DataFrame(rows)

    rows = [load_one(p) for p in metric_files]
    df = pd.DataFrame(rows)

    # FIX: Remove duplicate seeds for the same alpha (the "200 runs" fix)
    # We round alpha to 2 decimals to ensure grouping works correctly
    df['alpha'] = df['alpha'].round(2)
    df = df.drop_duplicates(subset=["alpha", "env", "seed"], keep="first")
    #
    # rows = [load_one(p) for p in metric_files]
    # df = pd.DataFrame(rows)
    #
    # # FIX: Remove duplicate seeds for the same policy/alpha (the "200 runs" fix)
    # df = df.sort_values("total_reward").drop_duplicates(subset=["alpha", "env", "seed"], keep="last")

    tag = f"allPolicies_{timestamp()}"

    per_seed_csv = OUTPUT_DIR / f"per_seed_metrics_{tag}.csv"
    summary_csv  = OUTPUT_DIR / f"overall_summary_by_alpha_{tag}.csv"
    summary_tex  = OUTPUT_DIR / f"overall_summary_by_alpha_{tag}.tex"

    summary = summarize_by_alpha_env(df)

    df.to_csv(per_seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_tex.write_text(to_latex_table(summary))

    print("Wrote:")
    print(f"  {per_seed_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_tex}")

if __name__ == "__main__":
    main()

