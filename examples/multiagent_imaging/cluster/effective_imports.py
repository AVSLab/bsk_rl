"""Installed metadata and actual import provenance; no Ray initialization."""
import hashlib
import importlib
import os
from pathlib import Path
import sys

CORE = {"ray": "2.35.0", "torch": "2.4.1", "numpy": "1.26.4",
        "gymnasium": "0.29.1", "pettingzoo": "1.24.3", "pyarrow": "19.0.1",
        "scipy": "1.14.1", "psutil": "6.1.0", "wandb": "0.26.1"}

def effective_import_record():
    # Match the training imports before recording the actually cached modules.
    import ray
    from ray.rllib.algorithms.ppo import PPO
    from examples.multiagent_imaging.cluster.audit_runtime import distribution_record
    from Basilisk.simulation import spacecraft, simpleNav
    from Basilisk.fswAlgorithms import locationPointing
    import Basilisk
    import bsk_rl
    errors, modules = [], {}
    for name, expected in CORE.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        version = str(module.__version__)
        native = {}
        for loaded_name, loaded in tuple(sys.modules.items()):
            file = getattr(loaded, "__file__", None)
            if loaded_name.startswith(name + ".") and file and Path(file).suffix in {".so", ".pyd", ".dylib"}:
                f = Path(file).resolve()
                native[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        modules[name] = {"version": version, "path": str(path),
                         "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "native_sha256": native, "installed": distribution_record(name)}
        if version.split("+")[0] != expected:
            errors.append(f"Loaded {name}: expected {expected}, found {version}")
        if name == "psutil" and "/ray/thirdparty_files/" in str(path):
            errors.append("psutil resolves to Ray's private vendor")
    ray_vendor = Path(ray.__file__).parent / "thirdparty_files/psutil"
    vendor = {"path": str(ray_vendor), "metadata": [x for x in distribution_record("psutil")["matches"] if x["ray_vendored"]],
              "files_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ray_vendor.glob("*")) if f.is_file() and f.suffix in {".py", ".so"}}}
    bsk_path = Path(bsk_rl.__file__).resolve()
    expected_root = os.environ.get("BSK_PROJECT_ROOT")
    if expected_root and not bsk_path.is_relative_to(Path(expected_root).resolve() / "src"):
        errors.append("Actual worker/driver bsk_rl is not the new reviewed source")
    native_basilisk = {}
    for module in (spacecraft, simpleNav, locationPointing):
        for f in Path(module.__file__).parent.glob("_"+module.__name__.split(".")[-1]+".*"):
            if f.is_file() and f.suffix in {".so", ".pyd", ".dylib"}:
                native_basilisk[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return {"passed": not errors, "errors": errors, "pid": os.getpid(), "python": sys.version,
            "executable": sys.executable, "modules": modules, "ray_private_psutil": vendor,
            "basilisk": {"path": str(Basilisk.__path__[0]), "version": getattr(Basilisk,"__version__",None), "native_sha256": native_basilisk},
            "bsk_rl": {"path": str(bsk_path), "file_sha256": hashlib.sha256(bsk_path.read_bytes()).hexdigest()},
            "per_job_bootstrap_enabled": os.environ.get("BSK_REQUIRE_EXTERNAL_PSUTIL") == "1"}

def require_effective_imports():
    record = effective_import_record()
    if not record["passed"]:
        raise RuntimeError("Effective import gate failed: " + "; ".join(record["errors"]))
    return record
