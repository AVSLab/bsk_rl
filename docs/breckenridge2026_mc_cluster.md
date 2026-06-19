# Breckenridge 2026 Mixed-Trained Monte Carlo Campaign

The LEO-trained policy has already been evaluated in LEO and mixed target
environments. Do not rerun those completed Monte Carlos. This campaign fills
only the new mixed-trained row:

| Training environment | LEO evaluation | Mixed evaluation |
|---|---:|---:|
| Existing LEO-trained policy | use archived results | use archived results |
| June 2026 mixed-trained policy | 100 seeds | 100 seeds |

The two new cells use seeds `0` through `99`, matching the archived campaigns.
The two Slurm arrays are independent and have no job dependencies.

## Important alpha provenance

The final GNC/Breckenridge paper contains a paired 100-seed robustness table
for one LEO-trained policy evaluated in both LEO and mixed environments. That
paired policy is alpha 0.5 (`50d50i`), not alpha 0.1.

The same paper's alpha sweep separately contains the alpha-0.1 (`10d90i`)
October LEO-trained policy evaluated in the mixed environment for seeds 0--99.
That is the clean existing baseline for comparing the new alpha-0.1
mixed-trained policy in the mixed environment.

Therefore:

- no existing LEO-trained campaign should be rerun just to replace it in the
  comparison;
- mixed-trained alpha 0.1 to mixed directly tests whether matching the training
  distribution improves mixed-environment deployment;
- mixed-trained alpha 0.1 to LEO measures reverse transfer;
- a complete alpha-0.1 2x2 table requires a verified October `10d90i`
  LEO-to-LEO archive. The later `examples/Huterer_JAS/data` runs should not be
  substituted silently: although their folder names contain `10d90i`, their
  saved metadata identifies an AMOS legacy `wGAE_balance0d100i...` policy.

## Evaluation settings recovered from the GNC campaign

The executable source of truth is `examples/policy_evaluation_2026.py`, which
created the existing `examples/data/GNC26_data/RL10d90i_mixed` seed outputs.

- 100 targets and 10 candidates in the policy observation
- 45,000 second horizon: `1.5 * 100 * 300`
- mixed target weights: LEO 0.5, MEO 0.3, GEO 0.2
- LEO altitude 400--2,000 km, eccentricity 0--0.02, inclination 0--180 deg
- MEO altitude 2,000--35,000 km, eccentricity 0--0.10, inclination 0--120 deg
- GEO altitude 35,486--36,086 km, eccentricity 0--0.0015, inclination 0--15 deg
- LEO/MEO samples reject perigees below 400 km altitude
- fixed actions: image 300 s, charge 300 s, downlink 180 s, desat 150 s
- safety shield enabled at storage fraction above 0.99 or battery below 0.20
- no fast imaging/downlink stopping, HIO/SHIO, dynamic priorities, or priority
  distributions
- uniform target priority 1

The new campaign freezes the explicit numeric checkpoint in a JSON manifest so
a later checkpoint cannot silently change the comparison.

## Bundled mixed-trained checkpoint

The completed local run reached 813,696 sampled environment steps over 163
training iterations. Its `checkpoint_000160` inspector inference module is
committed directly to this branch under:

```text
policies/breckenridge2026_mixed_10d90i/checkpoint_000160
```

The bundle contains the loadable RL module weights and metadata plus the
training run configuration, RLlib parameters, progress CSV, and SHA-256
checksums. The optimizer and learner state are deliberately omitted because
they are not needed for policy evaluation.

## Pull the branch

On Alpine:

```bash
cd /projects/dahu1128/bsk_rl
git fetch origin
git switch breckenridge2026-leo-any-local
git pull --ff-only
source /projects/dahu1128/.venv/bin/activate

test -f \
  policies/breckenridge2026_mixed_10d90i/checkpoint_000160/learner_group/learner/rl_module/inspector/module_state.pt \
  && echo "Bundled checkpoint ready"
```

## Submit the two missing arrays

```bash
cd /projects/dahu1128/bsk_rl
source /projects/dahu1128/.venv/bin/activate

bash examples/breckenridge2026/submit_2x2_mc.sh 10
```

The submit script first loads the bundled checkpoint through RLlib using
`/projects/$USER/.venv/bin/python`. It submits no jobs if that preflight fails.
It uses Alpine's installed `/curc/sw/install/gcc/14.2.0` runtime directly,
without relying on an Lmod module name, and verifies that `libstdc++.so.6`
provides `GLIBCXX_3.4.29`, which the installed Basilisk wheel requires. If that
path changes, the helper scans the other installed GCC roots. The same Basilisk
runtime preflight runs again on every compute node.

The final argument limits each cell to 10 concurrent seeds. Because there are
two independent arrays, up to 20 seed evaluations can run at once if the
account and QOS allow it. Change `10` to a lower number if desired.

Monitor:

```bash
squeue -u "$USER"
```

No `--dependency` options are used. Re-running the same campaign with the same
`BRECK_MC_OUTPUT_ROOT` skips completed seeds.

## Audit and summarize

Use the output root printed by the submit script:

```bash
python3 examples/breckenridge2026/audit_mc_campaign.py \
  --input-root /scratch/alpine/$USER/breckenridge2026_mc/mixed_trained_row_10d90i_<campaign-id>

python3 examples/breckenridge2026/summarize_2x2_mc.py \
  --input-root /scratch/alpine/$USER/breckenridge2026_mc/mixed_trained_row_10d90i_<campaign-id>
```

The audit must report `PASS` and `200 / 200` validated seed rows. It verifies
the manifest, cells, seeds, completed statuses, one metrics JSON per seed,
Git commit, policy identity and checkpoint hash, target environment and mix,
target counts, action durations, 45,000-second horizon, uniform priority,
safety shield, required finite metrics, and final episode time.

This writes a per-seed CSV and a two-row mean/std summary CSV for the new
mixed-trained evaluations. Join those rows to the archived LEO-trained
baselines during paper analysis. The optional independent reproduction below
checks that the archived baseline can still be regenerated exactly.

## Reproduce the October alpha-0.1 baseline

The exact October LEO-trained `10d90i` inference checkpoint is bundled at
`policies/breckenridge2026_leo_trained_10d90i/checkpoint_000145`. Submit its
LEO and mixed evaluations as two independent arrays:

```bash
bash examples/breckenridge2026/submit_leo_baseline_mc.sh 10
```

After completion, use the output root printed by that command:

```bash
python3 examples/breckenridge2026/audit_mc_campaign.py \
  --input-root /scratch/alpine/$USER/breckenridge2026_mc/leo_trained_baseline_10d90i_<campaign-id>

python3 examples/breckenridge2026/summarize_2x2_mc.py \
  --input-root /scratch/alpine/$USER/breckenridge2026_mc/leo_trained_baseline_10d90i_<campaign-id>
```

For the LEO-trained-to-mixed cell, the audit compares all 100 seeds and five
paper metrics against the archived alpha-0.1 per-seed data at a tolerance of
`1e-6`. A successful exact reproduction reports zero mismatches. Summary
standard deviations use the paper's sample convention (`ddof=1`).
