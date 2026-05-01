# AMOS 2025 Policy Artifact for JAS Evaluation

This directory contains the single policy artifact retained for the public JAS reproduction branch.

Current policy folder:

```text
wGAE_balance0d100i_largepenalties_smallbatch_obs2
```

Included checkpoint subset:

- `checkpoint_best/`
- `checkpoint_000515/`
- `progress.csv`

The public evaluator resolves `--policy_key amos2025_seed184` to this policy folder. Other local checkpoints can still be evaluated with `--policy_path`.
