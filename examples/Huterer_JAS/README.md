# JAS Reproduction Workspace

This directory is the public reproduction workspace for the Journal of Astronautical Sciences paper experiments. It contains the BSK-RL source configuration used by the study, one evaluation entrypoint, and the selected policy artifact used for the paper.

## What Is Included

- `eval_amos2025_jas.py`: the public single-run evaluator for policy, heuristic, or random baselines.
- `load_policy.py`: local RLlib checkpoint loader used by the evaluator.
- `policies/amos2025/wGAE_balance0d100i_largepenalties_smallbatch_obs2/`: selected policy artifact used for the paper.
- `JAS_REPRO.md`: compact reproduction notes and branch-scope statement.

## Quick Start

Run from the repository root after installing the package dependencies:

```bash
python examples/Huterer_JAS/eval_amos2025_jas.py \
  --mode policy \
  --policy_key amos2025_seed184 \
  --policy_mode best \
  --seed 184 \
  --target_env mixed
```

For a lightweight heuristic comparison:

```bash
python examples/Huterer_JAS/eval_amos2025_jas.py \
  --mode heuristic \
  --heuristic_mode angle \
  --seed 184 \
  --target_env mixed
```

Outputs are written under `examples/Huterer_JAS/outputs/` by default and are ignored by git. Each run writes a `summary.json`, `config.json`, and `step_metrics.csv`.

## Policy Artifact

The default `--policy_key amos2025_seed184` resolves to the policy directory committed in this branch:

```text
examples/Huterer_JAS/policies/amos2025/wGAE_balance0d100i_largepenalties_smallbatch_obs2
```

You can also evaluate a different local checkpoint directory with `--policy_path /path/to/policy/run`.

## Source Layout

The source-side dynamics, flight software, data, reward, observation, and action code remain in the normal package locations under `src/bsk_rl/`.
