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

## Copy the mixed-trained checkpoint to Alpine

The completed local run reached 813,696 sampled environment steps. Its latest
saved numeric checkpoint is `checkpoint_000160`.

On the Mac:

```bash
MIXED_CHECKPOINT=$(find \
  "$HOME/rllib_results/breckenridge2026_leo_any_oldnet" \
  -type d -name checkpoint_000160 -path '*4992batch*' | head -1)

ssh dahu1128@login.rc.colorado.edu \
  'mkdir -p /projects/dahu1128/breckenridge2026_policies'

rsync -avh --progress "$MIXED_CHECKPOINT" \
  dahu1128@login.rc.colorado.edu:/projects/dahu1128/breckenridge2026_policies/
```

The checkpoint is about 101 MB.

## Pull the branch

On Alpine:

```bash
cd /projects/dahu1128/bsk_rl
git fetch origin
git switch breckenridge2026-leo-any-local
git pull --ff-only
source /projects/dahu1128/.venv/bin/activate

MIXED_CHECKPOINT=/projects/dahu1128/breckenridge2026_policies/checkpoint_000160
printf 'Mixed checkpoint: %s\n' "$MIXED_CHECKPOINT"
test -d "$MIXED_CHECKPOINT"
```

## Submit the two missing arrays

```bash
cd /projects/dahu1128/bsk_rl
source /projects/dahu1128/.venv/bin/activate

bash examples/breckenridge2026/submit_2x2_mc.sh \
  "$MIXED_CHECKPOINT" \
  10
```

The final argument limits each cell to 10 concurrent seeds. Because there are
two independent arrays, up to 20 seed evaluations can run at once if the
account and QOS allow it. Change `10` to a lower number if desired.

Monitor:

```bash
squeue -u "$USER"
```

No `--dependency` options are used. Re-running the same campaign with the same
`BRECK_MC_OUTPUT_ROOT` skips completed seeds.

## Summarize

Use the output root printed by the submit script:

```bash
python3 examples/breckenridge2026/summarize_2x2_mc.py \
  --input-root /scratch/alpine/$USER/breckenridge2026_mc/mixed_trained_row_10d90i_<campaign-id>
```

This writes a per-seed CSV and a two-row mean/std summary CSV for the new
mixed-trained evaluations. Join those rows to the archived LEO-trained
baselines during paper analysis; do not rerun the old campaigns.
