"""Validate release packaging without Ray initialization or mission sampling."""

import hashlib
from pathlib import Path


STUDY_CONFIGS = {
    arm: f"examples/multiagent_imaging/configs/walker4_t200_cd1_{arm}.json"
    for arm in ("catalog_tx", "geometry_only")
}


def validate_study_manifest(root, manifest):
    """Fail before runtime work if the producer/consumer contract is incomplete."""
    root = Path(root)
    for field in ("release_id", "source_sha256", "config_sha256"):
        if field not in manifest:
            raise ValueError(f"Release manifest missing required field: {field}")
    if not isinstance(manifest["release_id"], str) or not manifest["release_id"]:
        raise ValueError("Release manifest requires a nonempty release_id")
    sources = manifest["source_sha256"]
    configs = manifest["config_sha256"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Release manifest requires nonempty source_sha256")
    if not isinstance(configs, dict) or set(configs) != set(STUDY_CONFIGS):
        raise ValueError("Release config hashes must name both study arms exactly")
    for name, expected in sources.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe release source path: {name}")
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source mismatch: {name}")
    for arm, name in STUDY_CONFIGS.items():
        if name not in sources or configs[arm] != sources[name]:
            raise ValueError(f"Release config/source hash mismatch: {arm}")
    return manifest
