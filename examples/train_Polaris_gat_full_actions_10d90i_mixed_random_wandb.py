#!/usr/bin/env python3
"""10d90i mixed-random-regime entrypoint for AMOS 2026 GAT training."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ["BSK_RL_DOWNLINK_BONUS"] = "0.1"
os.environ["BSK_RL_TARGET_ENV"] = "mixed"
os.environ["BSK_RL_RANDOMIZE_MIX_WEIGHTS"] = "1"
os.environ["BSK_RL_REWARD_SPLIT_TAG"] = "10d90iMixedRandom"
os.environ["BSK_RL_ALPHA_TAG"] = "alpha0p1_mixedRandom"
os.environ["BSK_RL_WANDB_GROUP"] = (
    "polaris-gat-full-actions-obs-v9-10d90i-mixed-random"
)

runpy.run_path(
    str(Path(__file__).resolve().with_name("train_Polaris_gat_full_actions_wandb.py")),
    run_name="__main__",
)
