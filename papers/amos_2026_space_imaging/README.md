# AMOS 2026 Space Imaging Paper

This folder contains the working AMOS 2026 conference paper scaffold using the official AMOS LaTeX class.

## Current status

- `main.tex` contains the first manuscript draft structure aligned with the submitted abstract draft.
- `references.bib` contains seed references from the abstract draft and local project context.
- `amos.cls` was copied from the local AMOS LaTeX template.

## Data still needed before paper freeze

- Full reward-sweep Monte Carlo summary: `/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i/analysis/summary_by_policy.csv`.
- Detailed Monte Carlo analysis: `/scratch/alpine/$USER/amos2026_mc/gat_full_actions_eval_100d00i/analysis_detailed/`.
- Curriculum final-policy Monte Carlo results from `submit_gat_curriculum_alpha1p0_mc_200targets_45000s_0to99.sh`.
- Full heuristic campaign using the variable-duration imaging and early-downlink architecture.
- Priority-weighted heuristic and optional short receding-horizon heuristic for the scheduling-plan discussion.

## Build

From this folder:

```sh
latexmk -pdf main.tex
```

or use the Codex LaTeX skill:

```sh
python3 /Users/dahu1128/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py /Users/dahu1128/Repositories/bsk_rl/papers/amos_2026_space_imaging/main.tex
```
