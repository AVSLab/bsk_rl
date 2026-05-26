#!/usr/bin/env python3
"""50d50i entrypoint for AMOS 2026 GAT full-action training."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ["BSK_RL_DOWNLINK_BONUS"] = "0.5"
os.environ["BSK_RL_WANDB_GROUP"] = "polaris-gat-full-actions-obs-v9-50d50i"

runpy.run_path(
    str(Path(__file__).resolve().with_name("train_Polaris_gat_full_actions_wandb.py")),
    run_name="__main__",
)
