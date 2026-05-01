# JAS Repro Scope

This branch is a compact public snapshot for reproducing the JAS paper simulations from a stable code state. It preserves the simulation implementation, the selected policy artifact, and a clear evaluator that others can run.

## Included

- Core BSK-RL source code under `src/bsk_rl/`, including the dynamics, FSW, action, observation, data, and reward classes used by the experiments.
- One public evaluator: `examples/Huterer_JAS/eval_amos2025_jas.py`.
- One policy artifact: `examples/Huterer_JAS/policies/amos2025/wGAE_balance0d100i_largepenalties_smallbatch_obs2/`.
- Minimal reproduction notes and output conventions.

## Outputs

The evaluator writes run artifacts under `examples/Huterer_JAS/outputs/` by default. That output directory is ignored by git so repeated local runs do not change the repository state.

## Suggested Paper Link

Use the branch URL for the public artifact and a fixed commit URL for exact reproducibility once the paper version is final.

```text
https://github.com/AVSLab/bsk_rl/tree/SBSS_research_setup
```

After finalizing the paper snapshot, cite the exact commit hash in the paper or repository note so readers can reproduce the frozen version even if the branch receives later documentation edits.
