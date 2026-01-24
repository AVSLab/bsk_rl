from pathlib import Path
import pandas as pd
import numpy as np
import re

# =========================
# CONFIG (hardcoded)
# =========================
ANALYSIS_ROOT = Path("analysis_out")   # adjust if needed
OUT_DIR = ANALYSIS_ROOT / "eclipse_alpha_analysis"
OUT_DIR.mkdir(exist_ok=True)

POLICY_RE = re.compile(r"RL(?P<d>\d+)d(?P<i>\d+)i")

def parse_alpha_from_path(p: Path):
    m = POLICY_RE.search(str(p))
    if not m:
        return None
    d = int(m.group("d"))
    i = int(m.group("i"))
    return d / (d + i)

def mean_pm_std(x, fmt="{:.3f}"):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "—"
    return f"{fmt.format(x.mean())} ± {fmt.format(x.std(ddof=1))}"

# =========================
# Collect umbra metrics
# =========================
rows = []

for csv_path in ANALYSIS_ROOT.rglob("eclipse_pointing_summary.csv"):
    alpha = parse_alpha_from_path(csv_path)
    if alpha is None:
        continue

    df = pd.read_csv(csv_path)

    # Each CSV is one run; extract umbra columns directly
    rows.append({
        "alpha": alpha,
        "N_umbra": df["N_umbra"].iloc[0],
        "acq_success_rate": df["acq_success_rate_umbra"].iloc[0],
        "dt_acq_mean": df["dt_acq_mean_succ_umbra"].iloc[0],
        "lookahead_mean": df["lookahead_mean_umbra"].iloc[0],
        "el_mean": df["el_mean_umbra"].iloc[0],
        "frac_LEO": df["frac_LEO_umbra"].iloc[0],
        "frac_MEO": df["frac_MEO_umbra"].iloc[0],
        "frac_GEO": df["frac_GEO_umbra"].iloc[0],
    })

eclipse_df = pd.DataFrame(rows)

# =========================
# Aggregate across seeds by alpha
# =========================
summary = []
for alpha, g in eclipse_df.groupby("alpha"):
    summary.append({
        "alpha": alpha,
        "N seeds": len(g),
        "Umbra acq. success": mean_pm_std(g["acq_success_rate"]),
        "Umbra mean lookahead": mean_pm_std(g["lookahead_mean"]),
        "Umbra mean elevation [deg]": mean_pm_std(g["el_mean"], "{:.2f}"),
        "Frac LEO (umbra)": mean_pm_std(g["frac_LEO"]),
        "Frac MEO (umbra)": mean_pm_std(g["frac_MEO"]),
        "Frac GEO (umbra)": mean_pm_std(g["frac_GEO"]),
        "Mean dt_acq succ [s]": mean_pm_std(g["dt_acq_mean"], "{:.1f}"),
    })

summary_df = pd.DataFrame(summary).sort_values("alpha")
summary_df.to_csv(OUT_DIR / "eclipse_alpha_summary.csv", index=False)

# =========================
# Write LaTeX table
# =========================
def to_latex_table(df, caption, label):
    cols = df.columns.tolist()
    header = " & ".join(cols) + r" \\"
    body = "\n".join(
        " & ".join(str(v) for v in row) + r" \\"
        for row in df.values
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{" + "l" + "c"*(len(cols)-1) + "}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

latex = to_latex_table(
    summary_df,
    caption=(
        "Eclipse (umbra) behavior vs downlink reward weight $\\alpha$. "
        "Values are mean $\\pm$ std across Monte Carlo seeds."
    ),
    label="tab:eclipse_alpha",
)

(OUT_DIR / "eclipse_alpha_summary.tex").write_text(latex)

print("Wrote:")
print(" -", OUT_DIR / "eclipse_alpha_summary.csv")
print(" -", OUT_DIR / "eclipse_alpha_summary.tex")
