#!/usr/bin/env python3
"""Curriculum 00d100i -> 100d00i entrypoint for AMOS 2026 GAT full-action training."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

# Start with image-only reward, then ramp the downlink reward alpha to 1.0.
os.environ["BSK_RL_DOWNLINK_BONUS"] = "0.0"
os.environ["BSK_RL_ALPHA_CURRICULUM"] = "1"
os.environ["BSK_RL_ALPHA_CURRICULUM_START"] = "0.0"
os.environ["BSK_RL_ALPHA_CURRICULUM_END"] = "1.0"
os.environ.setdefault("BSK_RL_ALPHA_CURRICULUM_POWER", "1.0")
os.environ.setdefault("BSK_RL_ALPHA_CURRICULUM_RAMP_STEPS", "20000000")
os.environ["BSK_RL_REWARD_SPLIT_TAG"] = "curriculum00d100iTo100d00i"
os.environ["BSK_RL_ALPHA_TAG"] = "curriculum0to1"
os.environ["BSK_RL_WANDB_GROUP"] = "polaris-gat-full-actions-obs-v9-curriculum-00to100d00i"

runpy.run_path(
    str(Path(__file__).resolve().with_name("train_Polaris_gat_full_actions_wandb.py")),
    run_name="__main__",
)
