# AMOS/JAS Public Snapshot

This branch preserves a public reference point for the AMOS 2025 / Journal of the Astronautical Sciences (JAS) research workflow based on the late-September 2025 code state.

## Provenance

- Public branch name: `amos-journal-public`
- Public branch URL: <https://github.com/AVSLab/bsk_rl/tree/amos-journal-public>
- Baseline branch: `SSA_behavioral_cloning`
- Baseline commit: `07e0bf7` (`2025-09-29`, `Add function to generate experience tuples`)

This keeps the AMOS-era Polaris training, evaluation, and analysis scripts close to the state used during the August and September 2025 research push, while adding a short guide for outside users.

## Recommended Entry Points

For readers of the journal paper, these are the main scripts worth starting from:

- `examples/train_Polaris.py`
- `examples/updated_train_Polaris.py`
- `examples/policy_evaluation.py`
- `examples/updated_policy_evaluation.py`
- `examples/load_policy.py`
- `examples/batch_policy_evaluation.py`
- `examples/batch_heuristic_evaluation.py`
- `examples/results_analysis.py`
- `examples/bc_environment.py`

## What This Snapshot Is

- A public, citable branch that reflects the AMOS-era workflow and code organization.
- A research snapshot, not a polished end-user product release.
- A useful reference for the training and evaluation setup described in the AMOS journal paper.

## Important Public-Use Notes

- Several AMOS-era evaluation scripts still contain hard-coded local checkpoint paths from the original research environment.
- Those path catalogs are preserved intentionally for provenance, but you should replace them with your own local checkpoint locations before running the scripts.
- The trained policy checkpoint artifacts are not stored in this branch; the branch documents the simulation, training, evaluation, and analysis code used to produce the reported results.
- The main package install and core environment APIs still follow the standard BSK-RL documentation in the root `README.md`.

## Typical Workflow

1. Install BSK-RL and Basilisk using the main project instructions in `README.md`.
2. Choose the script closest to your use case from the list above.
3. Update any local checkpoint or output paths in that script.
4. Run training or evaluation from the repository root.
5. Use the analysis scripts to summarize rollouts, Monte Carlo batches, or policy comparisons.

## Why This Branch Exists

The journal paper benefits from a stable public branch that points to the research-era codebase directly, rather than to a later branch with additional 2026 experiments and exploratory edits layered on top.

## Suggested Paper Wording

Example wording for a data/code availability statement:

> The BSK-RL code snapshot used for the AMOS 2025/JAS space-to-space imaging experiments is available on the public `amos-journal-public` branch of the AVSLab BSK-RL repository: <https://github.com/AVSLab/bsk_rl/tree/amos-journal-public>.
