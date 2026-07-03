# Breckenridge 2026 LEO-Trained Alpha-Sweep Policies

This directory contains the LEO-trained policy checkpoints used for the imaging-versus-downlink alpha sweep. Each label `XdYi` denotes X% downlink reward weight and Y% image-acquisition reward weight.

The checkpoints were selected from the October 14, 2025 `batch5000`, observation-version-7 training runs listed in `MANIFEST.csv`. For each alpha, the latest numeric RLlib checkpoint was preserved as an inference bundle containing only the inspector RLModule files required by `RLModule.from_checkpoint`.
