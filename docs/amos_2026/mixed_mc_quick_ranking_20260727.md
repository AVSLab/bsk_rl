# AMOS 2026 Mixed-Orbit MC Quick Ranking

Created: 2026-07-27

Cluster root audited:

```text
/scratch/alpine/dahu1128/amos2026_mc/gat_full_actions_eval_100d00i_mixed_50LEO30MEO20GEO_200targets_45000s_HIO5_SHIO3_20260611T212640Z
```

Configuration observed from `mc_status.json`:

- Target environment: `mixed`
- Target count: `200`
- Episode duration: `45000.0` seconds
- Evaluation reward mix: `100d00i`
- Completed runs: `800`
- Completed policies: `00d100i`, `10d90i`, `20d80i`, `30d70i`, `40d60i`, `50d50i`, `75d25i`, `100d00i`
- Seeds per completed policy: `100`

Quick score ranking from copied terminal output:

| Rank | Policy | Alpha | Runs | Mean score | Std. dev. | 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `10d90i` | 0.10 | 100 | 293.781 | 19.3607 | 3.7947 |
| 2 | `20d80i` | 0.20 | 100 | 288.471 | 20.6012 | 4.03783 |
| 3 | `30d70i` | 0.30 | 100 | 283.073 | 18.6389 | 3.65322 |
| 4 | `40d60i` | 0.40 | 100 | 268.391 | 17.8716 | 3.50282 |
| 5 | `50d50i` | 0.50 | 100 | 260.665 | 17.9662 | 3.52138 |
| 6 | `00d100i` | 0.00 | 100 | 258.829 | 26.3048 | 5.15575 |
| 7 | `75d25i` | 0.75 | 100 | 241.028 | 18.1263 | 3.55276 |
| 8 | `100d00i` | 1.00 | 100 | 69.5413 | 36.7294 | 7.19896 |

Interpretation:

The best observed policy in this mixed LEO/MEO/GEO + HIO/SHIO campaign is `10d90i`, i.e., training with downlink reward weight alpha `0.1` and imaging reward weight `0.9`, scored under the common `100d00i` ground-value evaluation reward. The neighboring `20d80i` and `30d70i` policies are the nearest competitors. This supports the working AMOS storyline that a small but nonzero downlink reward during training improves ground-value delivery relative to pure imaging or pure downlink reward training.

Copy-back status:

The cluster tarball was created successfully at:

```text
/scratch/alpine/dahu1128/amos2026_mc/amos2026_cluster_audit_20260727.tgz
```

Size reported by Alpine:

```text
41M
```

Direct `scp` from this Codex process failed because the Mac has no authorized non-interactive credential for `dahu1128@login-ci5.rc.colorado.edu`.

The tarball was then copied interactively from a Mac Terminal session and unpacked locally at:

```text
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727
```

Local extracted counts:

- `mc_status.json`: `800`
- `metrics_*.json`: `800`
- `steps.csv`: `800`
- `images.csv`: `800`

Local analysis artifacts:

```text
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727/local_analysis/mixed_mc_per_run_compact.csv
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727/local_analysis/mixed_mc_summary_by_policy.csv
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727/local_analysis/mixed_mc_paired_score_diffs_vs_10d90i.csv
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727/local_analysis/mixed_mc_paper_table.csv
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727/local_analysis/mixed_mc_paper_table.md
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727/local_analysis/mixed_mc_paper_table.tex
/Users/dahu1128/Downloads/AMOS2026_cluster_results_20260727/local_analysis/mixed_reward_sweep_score_vs_alpha.svg
```

Seed-paired score comparisons against `10d90i`:

| Comparison | Mean difference | 95% CI | Win rate |
| --- | ---: | ---: | ---: |
| `10d90i - 00d100i` | 34.953 | 5.302 | 92% |
| `10d90i - 20d80i` | 5.310 | 2.316 | 66% |
| `10d90i - 30d70i` | 10.708 | 2.058 | 86% |
| `10d90i - 40d60i` | 25.390 | 2.155 | 96% |
| `10d90i - 50d50i` | 33.116 | 2.690 | 99% |
| `10d90i - 75d25i` | 52.753 | 2.812 | 100% |
| `10d90i - 100d00i` | 224.240 | 8.481 | 100% |

Additional interpretation:

The best policy is not merely highest by aggregate mean; paired by identical seeds, `10d90i` beats `20d80i` on 66% of seeds and beats every other completed policy on at least 86% of seeds. The paper should describe the optimum as a small but nonzero downlink training reward, with `10d90i` best among the completed sweep and `20d80i`/`30d70i` as the nearest competitors.
