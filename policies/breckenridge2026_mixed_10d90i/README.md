# Breckenridge 2026 Mixed-Trained Alpha-0.1 Policy

This directory contains the alpha-0.1 (`10d90i`) policy trained directly in the mixed LEO/MEO/GEO target environment for the training-distribution ablation.

Training setup: 50% LEO, 30% MEO, 20% GEO targets; 100 targets; 10 candidate targets; two hidden layers with 2,048 units each; PPO train batch size 4,992. The selected checkpoint is `checkpoint_000160`, preserved as an inference bundle for `RLModule.from_checkpoint`.
