"""Small optional Weights & Biases logger for BSK-RL training scripts.

Local default key path:
    examples/wandb_key.txt

Cluster default key path:
    /projects/$USER/bsk_rl/examples/wandb_key.txt

You can override either with:
    export BSK_RL_WANDB_KEY_PATH=/path/to/wandb_key.txt
"""

from __future__ import annotations

import pathlib
import numbers
from collections.abc import Mapping
from typing import Any

try:
    import wandb
except ImportError:  # pragma: no cover - depends on optional local install
    wandb = None

try:
    from flatten_dict import flatten as _flatten
except ImportError:  # pragma: no cover - keep W&B optional and lightweight
    _flatten = None


def _fallback_flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_fallback_flatten(value, flat_key))
        else:
            flat[flat_key] = value
    return flat


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if _flatten is not None:
        return _flatten(metrics, reducer="path")
    return _fallback_flatten(metrics)


def _safe_config(value: Any) -> Any:
    """Convert run configuration objects into W&B-serializable values."""
    if isinstance(value, Mapping):
        return {str(k): _safe_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_config(v) for v in value]
    if isinstance(value, (str, bool, type(None))):
        return value
    if isinstance(value, numbers.Number):
        return value.item() if hasattr(value, "item") else value
    return repr(value)


def _safe_metric(value: Any) -> Any:
    """Keep scalar RLlib metrics and drop arrays/objects that W&B cannot log."""
    if isinstance(value, (str, bool, type(None))):
        return value
    if isinstance(value, numbers.Number):
        return value.item() if hasattr(value, "item") else value
    return None


class WandbLogger:
    """Thin W&B wrapper that tolerates optional dependencies and noisy RLlib dicts."""

    def __init__(
        self,
        project_name: str,
        run_name: str,
        config: dict[str, Any],
        key_path: pathlib.Path,
        group: str | None = None,
    ):
        if wandb is None:
            raise ImportError("wandb is not installed. Install wandb or disable W&B.")
        if not key_path.exists():
            raise FileNotFoundError(f"W&B key file not found: {key_path}")

        self.project_name = project_name
        self.run_name = run_name
        self.config = _safe_config(config)
        self.key_path = key_path
        self.group = group
        self.run = None

        self._login()
        self._init()

    def _login(self) -> None:
        wandb.login(key=self.key_path.read_text().strip())

    def _init(self) -> None:
        if self.run is not None:
            return
        self.run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            group=self.group,
            resume="allow",
        )

    def log(self, metrics: Mapping[str, Any]) -> None:
        self._init()
        flat_metrics = _flatten_metrics(metrics)
        loggable = {}
        for key, value in flat_metrics.items():
            safe_value = _safe_metric(value)
            if safe_value is not None:
                loggable[key] = safe_value
        if loggable:
            self.run.log(loggable)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
