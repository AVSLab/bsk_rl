"""Approval/allocation and static no-sampling boundaries; no Ray initialization."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from examples.multiagent_imaging.cluster import startup_only


def allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("BSK_STARTUP_ONLY_APPROVED", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setenv("BSK_REQUIRE_EXTERNAL_PSUTIL", "1")
    monkeypatch.setenv("BSK_OUTPUT", str(tmp_path))
    (tmp_path / "allocation-gate.json").write_text(json.dumps(dict(allocated_cpus=56, time_limit_s=900, reserved_cpu_h=14)))
    (tmp_path / "memory-allocation-profile.json").write_text(json.dumps(dict(passed=True, effective_limit_bytes=214748364800)))


def test_approval_missing_fails_before_output_or_ray_import(tmp_path, monkeypatch):
    monkeypatch.delenv("BSK_STARTUP_ONLY_APPROVED", raising=False)
    with pytest.raises(RuntimeError, match="approval"):
        startup_only.run(tmp_path / "no-config.json", tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


@pytest.mark.parametrize("field,value", [("allocated_cpus", 57), ("time_limit_s", 901), ("reserved_cpu_h", 15)])
def test_exact_allocation_not_substitutable(tmp_path, monkeypatch, field, value):
    allocation(tmp_path, monkeypatch)
    path = tmp_path / "allocation-gate.json"
    record = json.loads(path.read_text());record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="allocation differs"):
        startup_only.require_allocation()


def test_readable_200gib_and_external_pin_required(tmp_path, monkeypatch):
    allocation(tmp_path, monkeypatch)
    startup_only.require_allocation()
    monkeypatch.delenv("BSK_REQUIRE_EXTERNAL_PSUTIL")
    with pytest.raises(RuntimeError, match="runtime gate"):
        startup_only.require_allocation()


def test_startup_module_and_script_have_no_train_sample_step_or_checkpoint_calls():
    tree = ast.parse(Path(startup_only.__file__).read_text())
    prohibited = {"train", "sample", "step", "save_checkpoint", "save_to_path", "backward"}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in prohibited
                   for n in ast.walk(tree))
    script = Path(startup_only.__file__).with_name("fresh_startup_15m.slurm").read_text()
    assert "#SBATCH --time=00:15:00" in script and "#SBATCH --cpus-per-task=56" in script
    assert "startup_only" in script and "sbatch " not in script and "examples.multiagent_imaging.train" not in script


def test_restore_or_startup_flag_keeps_existing_no_step_guard():
    from examples.multiagent_imaging import train
    tree = ast.parse(Path(train.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_rllib_env")
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If) and "BSK_STARTUP_ONLY_GATE" in ast.unparse(n.test)]
    assert len(guards) == 1 and "BSK_ZERO_UPDATE_RESTORE" in ast.unparse(guards[0].test)
    assert "_restore_only_guard = True" in ast.unparse(guards[0])


@pytest.mark.parametrize("exit_code", [0, 7])
def test_exact_exit_trap_retains_failure_status_logs_and_provisional_cost(tmp_path, exit_code):
    source = Path(startup_only.__file__).with_name("fresh_startup_15m.slurm").read_text()
    function = source[source.index("startup_exit() {"):source.index("trap startup_exit EXIT")]
    root = Path(__file__).resolve().parents[3]
    ray_tmp = tmp_path / "empty-ray-logs";ray_tmp.mkdir()
    env = dict(os.environ, BSK_RL_PYTHON=sys.executable, BSK_RL_RAY_TMPDIR=str(ray_tmp),
               BSK_OUTPUT=str(tmp_path), BSK_JOB_START_UNIX=str(int(time.time())-1),
               BSK_ALLOCATED_CPUS="56", PYTHONPATH=str(root))
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + function +
        f"trap startup_exit EXIT\nexit {exit_code}\n"], env=env, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    record = json.loads((tmp_path / "job-exit.json").read_text())
    assert record["exit_code"] == exit_code and record["actual_allocated_cpu_h"] > 0
    assert (tmp_path / "ray-evidence/manifest.json").is_file()
