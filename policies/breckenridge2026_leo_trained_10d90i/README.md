# Breckenridge October 2025 LEO-Trained Policy

This directory contains the alpha-0.1 (`10d90i`) policy trained on a LEO-only
target catalog and used in the GNC/Breckenridge paper's mixed-regime alpha
sweep.

The paper evaluations selected the latest numeric checkpoint from the original
trial. The available numeric checkpoints are `000135`, `000140`, and `000145`,
so this bundle explicitly preserves `checkpoint_000145`. It is not the distinct
`checkpoint_best` artifact.

Training provenance:

- original trial: `oct14_restrictedResources_obsv7_1e-5lr_batch5000_gamma9997_10d90i_reducedFailurePenalty_lowBatPenalty`
- training environment: LEO-only
- reward mix: 0.9 imaging, 0.1 downlink
- sampled environment steps: 743,808
- training iterations: 149
- selected checkpoint: `checkpoint_000145`

The committed checkpoint is an inference bundle containing the inspector
RLModule weights and metadata required by `RLModule.from_checkpoint`.

`reference/paper_mixed_per_seed.csv` preserves the 100 archived mixed-regime
seed results used by the paper. The campaign audit compares a new
LEO-trained-to-mixed rerun against these values seed by seed.
