# Breckenridge 2026 LEO-Any Local Training Notes

This branch is for old-network GNC/Breckenridge experiments only. It is
based on `origin/IA_Polaris_SSA` from January 2026, before the AMOS 2026
variable-duration imaging/downlink and priority/HIO/SHIO changes.

## Why alpha 0.1 is the default

The imaging-vs-downlink paper does not identify a single overwhelmingly best
alpha. It identifies an intermediate band. In `/Users/dahu1128/Downloads/main-17.tex`:

- Lines 528-538, Table `alpha_sweep_metrics`: `alpha=0.1` has the highest
  mean total reward, `93.79 +/- 2.44`.
- Lines 625-626: the best-performing policies are in the intermediate range
  `alpha approx 0.1--0.4`.
- Lines 682-684: in the seed-99 rollout, the highest total reward occurs at
  `alpha=0.1`.

The matching repo summary is
`/Users/dahu1128/Repositories/bsk_rl/examples/results/overall_summary_by_alpha_allPolicies_20260116_150922.csv`,
which also ranks `alpha=0.1` first by mean total reward. If the metric is
strictly useful downlinked images instead of total reward, `alpha=0.3` to `0.4`
are slightly stronger, but the default here follows total reward.

## Old policy training setup found

The October 2025 GNC alpha sweep was trained on the cluster, not locally, using
scripts such as `examples/train_Polaris2110.py` from the archived cluster branch.
For the alpha-0.1 policy, the matching old label is `10d90i`.

The archived run params at
`/Users/dahu1128/rllib_results/october_results/oct14rllib_results/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty_1761114479.911475/oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty.out_0_params_oct14.txt`
show:

- `downlink_bonus: 0.1`, `imaging_bonus: 0.9`
- `train_batch_size: 4992`
- `lr: 1e-5`
- `gamma: 0.9997`
- `clip_param: 0.1`
- `grad_clip: 1.0`
- `num_sgd_iter: 10`
- `lambda_: 0.95`
- `failure_penalty: -10.0`
- `low_battery_penalty: -0.5`
- fixed actions: image 300 s, charge 300 s, downlink 300 s, desat 150 s

The progress log for that run ended at about 149 PPO iterations, 743808 sampled
environment steps, and 55269 s wall time. The script requested
`total_timesteps=20_000_000`, but the cluster runs effectively trained until the
48 hour wall-time limit rather than reaching that timestep target.

Note: the paper appendix table lists representative values `lr=1e-6` and
`clip=0.15`. The archived scripts and run names for the policies used in the
figures indicate `1e-5` and `clip=0.1`, so the local reproduction script follows
the archived training artifacts.

## Local mixed-training script

Use:

```bash
cd /Users/dahu1128/Repositories/bsk_rl
python3 examples/train_Breckenridge2026_LEOAny_oldnet_local.py
```

Defaults:

- old inspector network: `[2048, 2048]`
- old target network: `[2, 2]`
- alpha/downlink bonus: `0.1`
- mixed target catalog: `LEO=0.5,MEO=0.3,GEO=0.2`
- batch size: `4992`
- training stop: `750,000` sampled steps (the final batch can overshoot slightly)
- target count: `100`
- targets ahead: `10`
- fixed action durations, no fast/variable action stopping
- uniform target priorities from the January `RandomSatellites` implementation
- umbra smart-decision metrics exported through the episode callback

Recommended first smoke test:

```bash
cd /Users/dahu1128/Repositories/bsk_rl
python3 examples/train_Breckenridge2026_LEOAny_oldnet_local.py --smoke-test --n-envs 2
```

Full local run, keeping exact old batch size:

```bash
cd /Users/dahu1128/Repositories/bsk_rl
python3 examples/train_Breckenridge2026_LEOAny_oldnet_local.py \
  --downlink-bonus 0.1 \
  --mix-weights LEO=0.5,MEO=0.3,GEO=0.2 \
  --train-batch-size 4992
```

If local memory is tight, reduce `--n-envs`; keep `--train-batch-size 4992` for
the closest reproduction of the old training setup.

The completed June 2026 mixed-trained run reached 813,696 sampled steps. Its
latest saved numeric checkpoint is `checkpoint_000160`.

## Umbra smart-decision training metrics

During each imaging decision, the `ImageRSO` action tracks whether the inspector
spacecraft is in umbra. When it is, a decision is counted as smart if the chosen
target is illuminated, is in a higher regime (MEO/GEO), or is a LEO target on the
sunward side according to the Hill-frame sun/target dot product.

The training callback logs both cumulative and per-episode metrics:

- `umbra_imaging_decisions`
- `umbra_smart_decisions`
- `umbra_not_smart_decisions`
- `umbra_smart_fraction`
- `episode_umbra_imaging_decisions`
- `episode_umbra_smart_decisions`
- `episode_umbra_not_smart_decisions`
- `episode_umbra_smart_fraction`
- `umbra_smart_reason_illum_target`
- `umbra_smart_reason_high_regime`
- `umbra_smart_reason_sunward_leo`
- `episode_umbra_smart_reason_illum_target`
- `episode_umbra_smart_reason_high_regime`
- `episode_umbra_smart_reason_sunward_leo`

For TensorBoard, the main curve to watch is
`episode_umbra_smart_fraction`, alongside
`episode_umbra_imaging_decisions` so empty-denominator episodes are easy to spot.
