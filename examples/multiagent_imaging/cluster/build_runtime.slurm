#!/usr/bin/env bash
#SBATCH --job-name=bsk-runtime
#SBATCH --account=ucb550_asc2
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bsk-runtime-%j.log

# Optional, explicitly authorized build job. Never upgrades the AMOS environment.
set -euo pipefail
: "${SLURM_JOB_ID:?Do not build Basilisk on the login node}"
: "${BSK_PROJECT_ROOT:?Set the deployed completion source directory}"
: "${BASILISK_SOURCE_ROOT:?Set the separate source checkout to build}"
: "${BSK_RL_PYTHON:?Set the NEW virtual environment Python path}"
BSK_BOOTSTRAP_PYTHON="${BSK_BOOTSTRAP_PYTHON:-/usr/bin/python3.11}"
BSK_ENV_DIRECTORY="$(dirname "$(dirname "$BSK_RL_PYTHON")")"
BSK_BUILD_RECORD="$BSK_PROJECT_ROOT/results/multiagent_imaging/runtime-build-${SLURM_JOB_ID}"
mkdir -p "$BSK_BUILD_RECORD"
test "$(git -C "$BASILISK_SOURCE_ROOT" rev-parse HEAD)" = 8fcb54b2fb28388efb711786630501944fddec28
test -z "$(git -C "$BASILISK_SOURCE_ROOT" status --porcelain --untracked-files=no)"
if [[ -e "$BSK_ENV_DIRECTORY" ]]; then
  echo "Refusing to change an existing environment: $BSK_ENV_DIRECTORY" >&2
  exit 1
fi
"$BSK_BOOTSTRAP_PYTHON" -m venv "$BSK_ENV_DIRECTORY"
export PATH="$BSK_ENV_DIRECTORY/bin:$PATH"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK}"
# Keep Conan cache/build profiles isolated from other projects.
export CONAN_HOME="$BSK_ENV_DIRECTORY/conan2"
export CONAN_CPU_COUNT="${SLURM_CPUS_PER_TASK}"
cd "$BASILISK_SOURCE_ROOT"
git rev-parse HEAD > "$BSK_BUILD_RECORD/basilisk-source.txt"
# Install only the CPU wheel from the Torch index. Resolve its dependencies in
# the following pinned PyPI installation: the Torch index is not a complete
# build-backend registry (e.g. a typing_extensions sdist requires flit_core).
"$BSK_RL_PYTHON" -m pip install --no-deps torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
"$BSK_RL_PYTHON" -m pip install -r "$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/requirements.txt"
"$BSK_RL_PYTHON" -m pip install -r requirements.txt \
  -c "$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/requirements.txt"
# The exact Basilisk checkout declares its native build-tool compatibility ranges.
# Resolve once, record all versions, and retain that build lock for future rebuilds.
# Avoid its broad development/test extra, which conflicts with bsk_rl's test pins.
"$BSK_RL_PYTHON" - <<'PY'
import subprocess, sys, tomllib
with open('pyproject.toml', 'rb') as f:
    requirements = tomllib.load(f)['build-system']['requires']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *requirements])
PY
"$BSK_RL_PYTHON" -m pip freeze > "$BSK_BUILD_RECORD/build-tools.lock.txt"
"$BSK_RL_PYTHON" conanfile.py --buildType Release --buildTesting False \
  --vizInterface True --opNav False --mujoco False --rustModules False
"$BSK_RL_PYTHON" -m pip install --no-deps --no-build-isolation -e .
"$BSK_RL_PYTHON" -m pip install -e "$BSK_PROJECT_ROOT" \
  -c "$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/requirements.txt"
"$BSK_RL_PYTHON" -m pip freeze > "$BSK_BUILD_RECORD/runtime.lock.txt"
cc --version > "$BSK_BUILD_RECORD/compiler.txt"
cd "$BSK_PROJECT_ROOT"
"$BSK_RL_PYTHON" examples/multiagent_imaging/cluster/audit_runtime.py \
  --require-allocation --output "$BSK_BUILD_RECORD/runtime.json"
