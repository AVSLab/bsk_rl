# BSK-RL Breckenridge 2026 Publication Snapshot

This branch is a public reproduction snapshot for the LEO-to-any
imaging-versus-downlink study. It contains the BSK-RL source state, evaluation
scripts, cluster submission helpers, and trained policy checkpoints used for
the paper.

The branch intentionally does not include generated Monte Carlo output tables,
raw rollout folders, or publication figures. Those artifacts can be regenerated
from the bundled scripts and checkpoints.

## Included Policy Checkpoints

The LEO-trained alpha-sweep policies are stored in:

```text
policies/breckenridge2026_alpha_sweep
```

The directory contains one checkpoint for each reward weight:

```text
0d100i, 10d90i, 20d80i, 30d70i, 40d60i, 50d50i,
60d40i, 70d30i, 80d20i, 90d10i, 100d00i
```

The label `XdYi` denotes X% downlink reward weight and Y% image-acquisition
reward weight. These checkpoints come from the October 14, 2025
observation-version-7, batch-5000 LEO-only training runs. The exact selected
checkpoint for each alpha value is listed in:

```text
policies/breckenridge2026_alpha_sweep/MANIFEST.csv
```

The mixed-trained alpha-0.1 policy used for the training-distribution ablation
is stored in:

```text
policies/breckenridge2026_mixed_10d90i/checkpoint_000160
```

SHA-256 checksums for the policy weight files are provided in the policy
directories.

## Main Scripts

The main evaluation and training entry points are:

```text
examples/policy_evaluation_2026.py
examples/train_breckenridge2026_leo_any.py
examples/breckenridge2026/submit_alpha_sweep_mc.sh
examples/breckenridge2026/submit_2x2_mc.sh
examples/breckenridge2026/submit_leo_baseline_mc.sh
examples/breckenridge2026/audit_mc_campaign.py
examples/breckenridge2026/summarize_mc_campaign.py
```

The Slurm submission helpers assume the repository is checked out at
`/projects/$USER/bsk_rl` and that the Python environment is available at
`/projects/$USER/.venv`.

## Reproducing the Alpha Sweep

On Alpine:

```bash
cd /projects/$USER/bsk_rl
git fetch origin
git switch breckenridge2026-publication
git pull --ff-only
source /projects/$USER/.venv/bin/activate

bash examples/breckenridge2026/submit_alpha_sweep_mc.sh 10
```

This submits independent 100-seed mixed-catalog Monte Carlo arrays for all 11
LEO-trained alpha policies.

## Reproducing the Training-Distribution Ablation

To run the mixed-trained alpha-0.1 policy in both LEO-only and mixed-catalog
evaluation environments:

```bash
bash examples/breckenridge2026/submit_2x2_mc.sh 10
```

To rerun the LEO-trained alpha-0.1 baseline in the same two evaluation
environments:

```bash
bash examples/breckenridge2026/submit_leo_baseline_mc.sh 10
```

Each submission script prints the campaign output directory. After the arrays
finish, audit and summarize a campaign with:

```bash
python3 examples/breckenridge2026/audit_mc_campaign.py --input-root <campaign-root>
python3 examples/breckenridge2026/summarize_mc_campaign.py --input-root <campaign-root>
```

## Recreating the Mixed-Trained Policy

The mixed-trained alpha-0.1 policy can be retrained locally with:

```bash
python examples/train_breckenridge2026_leo_any.py \
  --downlink-bonus 0.1 \
  --mix-weights LEO=0.5,MEO=0.3,GEO=0.2 \
  --train-batch-size 4992
```

The committed checkpoint is the policy used in the paper. Retraining is useful
for replication, but exact neural-network weights are not expected because PPO
training is stochastic.

## Installation

Install BSK-RL and Basilisk using the standard project instructions at:

```text
https://avslab.github.io/bsk_rl/
```

This branch is a research reproduction snapshot, not a general software
release.
