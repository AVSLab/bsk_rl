# Space-To-Space Imaging Public Snapshot

This branch is the public-facing snapshot for the imaging-only BSK-RL setup used as the basis for the late-summer 2025 AMOS journal-paper experiments.

## Scope

This snapshot is intended to reflect the space-to-space imaging regime that was in use during August and September 2025:

- space-to-space imaging only
- imaging rewards only (`imaging_bonus = 1.0`, `downlink_bonus = 0.0`)
- no later fast target-switching additions
- optional built-in heuristic baseline, with `angle` as the default historical mode

The closest committed base for this public snapshot is commit `d0bcc54` from September 26, 2025, on the `IA_Polaris_imaging_june10` development line. That cutoff preserves the imaging-only reward structure and the angle-based heuristic path while avoiding later mixed-regime additions.

## Recommended Entry Point

Use [examples/space_to_space_imaging_evaluation.py](examples/space_to_space_imaging_evaluation.py) as the public evaluation script.

Example policy evaluation:

```bash
python examples/space_to_space_imaging_evaluation.py \
  --mode policy \
  --policy-path /path/to/policy_directory \
  --policy-mode best \
  --obs-version 7
```

Example heuristic baseline:

```bash
python examples/space_to_space_imaging_evaluation.py \
  --mode heuristic \
  --heuristic-mode angle \
  --obs-version 7
```

## What The Evaluator Saves

Each run writes outputs under `examples/public_outputs/...` by default:

- `summary.json`: episode configuration, reward, action counts, shield interventions, and telemetry
- `inference_summary.json`: mean, standard deviation, median, and percentile timing stats for policy calls
- `inference_timings.csv`: one inference-time sample per policy call
- `inference_time_histogram.png`: a quick visual distribution of policy latency
- `model_summary.json`: checkpoint metadata and parameter counts for the loaded RL module

## Historical Notes

Older scripts such as `examples/updated_policy_evaluation.py` are kept in the branch for provenance, but they still contain local research workflow structure and hard-coded policy catalogs that are not appropriate as the public entry point. The new evaluator is the cleaner script to reference in the paper and for outside users.
