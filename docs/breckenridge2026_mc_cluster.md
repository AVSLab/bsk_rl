# Breckenridge 2026 2x2 Monte Carlo Campaign

This campaign compares the alpha-0.1 (`10d90i`) policies in a 2x2 grid:

| Training environment | LEO evaluation | Mixed evaluation |
|---|---:|---:|
| October 2025 LEO-trained policy | 100 seeds | 100 seeds |
| June 2026 mixed-trained policy | 100 seeds | 100 seeds |

Every cell uses seeds `0` through `99`. The four Slurm arrays are independent
and have no job dependencies.

## Evaluation settings recovered from the GNC paper campaign

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

The original policy Monte Carlo used the latest numeric October checkpoint.
The new campaign freezes explicit numeric checkpoints in a JSON manifest so a
later checkpoint cannot silently change the comparison.

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

## Pull the branch and locate both checkpoints

On Alpine:

```bash
cd /projects/dahu1128/bsk_rl
git fetch origin
git switch breckenridge2026-leo-any-local
git pull --ff-only
source /projects/dahu1128/.venv/bin/activate
```

Locate the October LEO-trained alpha-0.1 checkpoint:

```bash
LEO_CHECKPOINT=$(find \
  /projects/dahu1128 /scratch/alpine/dahu1128 \
  -type d -name checkpoint_000145 \
  -path '*oct14*10d90i*' 2>/dev/null | head -1)

MIXED_CHECKPOINT=/projects/dahu1128/breckenridge2026_policies/checkpoint_000160

printf 'LEO checkpoint:   %s\n' "$LEO_CHECKPOINT"
printf 'Mixed checkpoint: %s\n' "$MIXED_CHECKPOINT"
test -d "$LEO_CHECKPOINT"
test -d "$MIXED_CHECKPOINT"
```

If the October checkpoint is not on Alpine, copy the local
`checkpoint_000145` to the same `breckenridge2026_policies` directory.

## Submit all four independent arrays

```bash
cd /projects/dahu1128/bsk_rl
source /projects/dahu1128/.venv/bin/activate

bash examples/breckenridge2026/submit_2x2_mc.sh \
  "$LEO_CHECKPOINT" \
  "$MIXED_CHECKPOINT" \
  10
```

The final argument limits each cell to 10 concurrent seeds. Because there are
four independent arrays, up to 40 seed evaluations can run at once if the
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
  --input-root /scratch/alpine/$USER/breckenridge2026_mc/leo_vs_mixed_10d90i_<campaign-id>
```

This writes a per-seed CSV and a four-row mean/std summary CSV.
