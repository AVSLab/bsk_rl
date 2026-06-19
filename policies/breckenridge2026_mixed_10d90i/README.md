# Breckenridge 2026 Mixed-Trained Policy

This directory contains the alpha-0.1 (`10d90i`) policy trained locally in a
mixed LEO/MEO/GEO target environment for the Breckenridge follow-on study.

Training setup:

- target distribution: 50% LEO, 30% MEO, 20% GEO
- 100 targets, 10 observed/actionable candidates
- reward mix: 0.9 imaging, 0.1 downlink
- PPO train batch size: 4,992
- inspector network: 2 hidden layers with 2,048 units each
- sampled environment steps: 813,696
- training iterations: 163
- selected checkpoint: `checkpoint_000160`

The committed checkpoint is an inference bundle. It contains the inspector
RLModule weights, constructor metadata, and module metadata required by
`RLModule.from_checkpoint`. Optimizer and learner state are excluded because
the Monte Carlo campaign only performs inference.

The original training configuration and progress history are preserved under
`training_artifacts/`.
