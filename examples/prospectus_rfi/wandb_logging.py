"""Dedicated Weights & Biases namespace for Research Focus I runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = "amos2025-architecture-comparison"
FINAL_GROUP = "rfi-alpha0-100s-candidate-sweep"
TUNING_GROUP = "rfi-alpha0-100s-architecture-tuning"
RUN_PREFIX = "amos2025-rfi-alpha0-100s"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def wandb_settings(
    run_name: str,
    output_root: Path,
    *,
    tuning: bool,
) -> dict[str, Any]:
    """Resolve a deterministic, overrideable W&B identity for one run."""

    project = os.environ.get("BSK_RL_WANDB_PROJECT", DEFAULT_PROJECT)
    default_group = TUNING_GROUP if tuning else FINAL_GROUP
    group = os.environ.get("BSK_RL_WANDB_GROUP", default_group)
    display_name = f"{RUN_PREFIX}__{run_name}"
    run_id = f"{RUN_PREFIX}-{run_name}".replace("_", "-")
    default_key = Path(__file__).resolve().parents[1] / "wandb_key.txt"
    key_path = Path(os.environ.get("BSK_RL_WANDB_KEY_PATH", default_key)).expanduser()
    local_dir = Path(
        os.environ.get(
            "WANDB_DIR",
            output_root.resolve() / "wandb" / project,
        )
    ).expanduser()
    return {
        "enabled": _env_bool("BSK_RL_USE_WANDB", True),
        "required": _env_bool("BSK_RL_REQUIRE_WANDB", False),
        "project": project,
        "group": group,
        "run_name": display_name,
        "run_id": run_id,
        "local_dir": str(local_dir.resolve()),
        "key_path": str(key_path.resolve()),
        "phase": "tuning" if tuning else "candidate_sweep",
    }


def public_wandb_metadata(settings: dict[str, Any]) -> dict[str, Any]:
    """Return settings safe to persist with run metadata."""

    return {key: value for key, value in settings.items() if key != "key_path"}


def maybe_init_wandb(settings: dict[str, Any], config: dict[str, Any]):
    """Initialize the repository W&B logger, or explicitly report why it is off."""

    if not settings["enabled"]:
        print("W&B disabled via BSK_RL_USE_WANDB=0", flush=True)
        return None

    key_path = Path(settings["key_path"])
    if not key_path.exists():
        message = f"W&B key file not found: {key_path}"
        if settings["required"]:
            raise FileNotFoundError(message)
        print(f"{message}; continuing without W&B", flush=True)
        return None

    local_dir = Path(settings["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(local_dir)
    os.environ["WANDB_RUN_ID"] = str(settings["run_id"])

    try:
        from examples.wandb_config import WandbLogger

        logger = WandbLogger(
            project_name=str(settings["project"]),
            run_name=str(settings["run_name"]),
            config=config,
            key_path=key_path,
            group=str(settings["group"]),
        )
    except Exception as error:
        if settings["required"]:
            raise
        print(f"W&B initialization failed: {error}; continuing without W&B", flush=True)
        return None
    print(
        f"W&B project={settings['project']} group={settings['group']} "
        f"run={settings['run_name']} id={settings['run_id']}",
        flush=True,
    )
    return logger


__all__ = [
    "DEFAULT_PROJECT",
    "FINAL_GROUP",
    "RUN_PREFIX",
    "TUNING_GROUP",
    "maybe_init_wandb",
    "public_wandb_metadata",
    "wandb_settings",
]
