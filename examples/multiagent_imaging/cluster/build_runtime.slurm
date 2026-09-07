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
  # Resume only our recorded, pre-install native build. This is not permission
  # to upgrade an arbitrary existing environment such as AMOS.
  : "${BSK_RESUME_BUILD_FROM:?Refusing an existing environment without a recorded failed build ID}"
  [[ "$BSK_RESUME_BUILD_FROM" =~ ^[0-9]+$ ]]
  BSK_PREVIOUS_RECORD="$BSK_PROJECT_ROOT/results/multiagent_imaging/runtime-build-$BSK_RESUME_BUILD_FROM"
  test "$(cat "$BSK_PREVIOUS_RECORD/basilisk-source.txt")" = 8fcb54b2fb28388efb711786630501944fddec28
  "$BSK_RL_PYTHON" -m pip freeze > "$BSK_BUILD_RECORD/resume-before.lock.txt"
  diff -u "$BSK_PREVIOUS_RECORD/build-tools.lock.txt" "$BSK_BUILD_RECORD/resume-before.lock.txt"
  printf '%s\n' "$BSK_RESUME_BUILD_FROM" > "$BSK_BUILD_RECORD/resumed-build-job.txt"
else
  "$BSK_BOOTSTRAP_PYTHON" -m venv "$BSK_ENV_DIRECTORY"
fi
export PATH="$BSK_ENV_DIRECTORY/bin:$PATH"
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Generated SWIG wrappers can each use several GiB under GCC 14. Keep one
# compiler active inside the 32-GiB allocation; the eight CPUs remain available
# to later Ray validation and do not need to be saturated during this build.
export CMAKE_BUILD_PARALLEL_LEVEL="${BSK_BUILD_PARALLELISM:-1}"
[[ "$CMAKE_BUILD_PARALLEL_LEVEL" =~ ^[1-8]$ ]]
printf '%s\n' "$CMAKE_BUILD_PARALLEL_LEVEL" > "$BSK_BUILD_RECORD/build-parallelism.txt"
# Keep Conan cache/build profiles isolated from other projects.
export CONAN_HOME="$BSK_ENV_DIRECTORY/conan2"
export CONAN_CPU_COUNT="${SLURM_CPUS_PER_TASK}"
# The base compute image exposes GCC 8.5, whose standard library cannot link
# Basilisk's C++17 filesystem probe without legacy flags. Alpine provides GCC
# 14.2 on its shared software filesystem; select and record that toolchain.
BSK_GCC_ROOT="${BSK_GCC_ROOT:-/curc/sw/install/gcc/14.2.0}"
export CC="${BSK_C_COMPILER:-$BSK_GCC_ROOT/bin/gcc}"
export CXX="${BSK_CXX_COMPILER:-$BSK_GCC_ROOT/bin/g++}"
export LD_LIBRARY_PATH="$BSK_GCC_ROOT/lib64:$BSK_GCC_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -x "$CC" -a -x "$CXX"
test -f "$BSK_GCC_ROOT/lib64/libstdc++.so.6"
"$CC" --version > "$BSK_BUILD_RECORD/c-compiler.txt"
"$CXX" --version > "$BSK_BUILD_RECORD/cxx-compiler.txt"
# Alpine's login image has python3.11-devel while its compute image does not.
# Use a recorded copy of those exact-version headers without changing /usr or
# patching the clean Basilisk source. Conan forwards this artifact path to CMake.
if [[ -n "${BSK_PYTHON_HEADERS:-}" ]]; then
  "$BSK_RL_PYTHON" - <<'PY'
import hashlib, json, os, pathlib, re, sys
headers = pathlib.Path(os.environ['BSK_PYTHON_HEADERS']).resolve()
assert (headers / 'Python.h').is_file(), 'Missing Python.h'
text = (headers / 'patchlevel.h').read_text()
version = re.search(r'#define\s+PY_VERSION\s+"([^"]+)"', text).group(1)
assert version == '.'.join(map(str, sys.version_info[:3])), (version, sys.version)
key = 'tools.cmake.cmaketoolchain:extra_variables'
conf = pathlib.Path(os.environ['CONAN_HOME']) / 'global.conf'
conf.parent.mkdir(parents=True, exist_ok=True)
lines = conf.read_text().splitlines() if conf.exists() else []
settings = {}
for line in lines:
    if line.startswith(key + '='):
        settings.update(json.loads(line.split('=', 1)[1]))
settings['Python3_INCLUDE_DIR'] = str(headers)
conf.write_text('\n'.join([line for line in lines if not line.startswith(key + '=')]
                         + [key + '=' + json.dumps(settings)]) + '\n')
record = dict(python=sys.version, header_version=version, path=str(headers),
              sha256={str(p.relative_to(headers)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(headers.rglob('*')) if p.is_file()})
path = pathlib.Path(os.environ['BSK_PROJECT_ROOT']) / 'results/multiagent_imaging' / (
    'runtime-build-' + os.environ['SLURM_JOB_ID']) / 'python-headers.json'
path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
PY
fi
# Conan otherwise reads the host's 128 CPUs and appends `-j128`, overriding
# CMAKE_BUILD_PARALLEL_LEVEL. Set its documented build-jobs configuration too.
"$BSK_RL_PYTHON" - <<'PY'
import os, pathlib
key = 'tools.build:jobs'
conf = pathlib.Path(os.environ['CONAN_HOME']) / 'global.conf'
lines = conf.read_text().splitlines() if conf.exists() else []
value = os.environ['CMAKE_BUILD_PARALLEL_LEVEL']
conf.write_text('\n'.join([line for line in lines if not line.startswith(key + '=')]
                         + [key + '=' + value]) + '\n')
PY
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
# Refresh the isolated profile after selecting compilers. A prior failed build
# may have detected the compute image's GCC 8.5 profile.
"$BSK_RL_PYTHON" -m conans.conan profile detect --force
bsk_conan_args=(--buildType Release --buildTesting False --vizInterface True
                --opNav False --mujoco False --rustModules False)
if [[ "${BSK_CLEAN_NATIVE_BUILD:-0}" == 1 ]]; then
  # Remove only Basilisk's marker-validated generated build folder. Conan's
  # downloaded package cache and the clean source checkout are preserved.
  bsk_conan_args+=(--clean)
fi
"$BSK_RL_PYTHON" conanfile.py "${bsk_conan_args[@]}"
"$BSK_RL_PYTHON" -m pip install --no-deps --no-build-isolation -e .
"$BSK_RL_PYTHON" -m pip install -e "$BSK_PROJECT_ROOT" \
  -c "$BSK_PROJECT_ROOT/examples/multiagent_imaging/cluster/requirements.txt"
"$BSK_RL_PYTHON" -m pip freeze > "$BSK_BUILD_RECORD/runtime.lock.txt"
cc --version > "$BSK_BUILD_RECORD/compiler.txt"
cd "$BSK_PROJECT_ROOT"
"$BSK_RL_PYTHON" examples/multiagent_imaging/cluster/audit_runtime.py \
  --require-allocation --output "$BSK_BUILD_RECORD/runtime.json"
