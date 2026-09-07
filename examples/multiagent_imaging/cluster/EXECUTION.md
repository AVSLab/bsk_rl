# Authorized Alpine deployment and validation

Code deployment uses Git, as requested. The desktop pushed
`134f1b9f7812002da32c65344f11880e452b6221` to
`AVSLab/bsk_rl:multi-agent-space-imaging-2026`; the separate cluster checkout at
`/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026` fetched and verified that
exact HEAD through the authenticated Termius session on `login-ci3`.

Basilisk was fetched into `/projects/dahu1128/basilisk-completion-v2` at clean
detached commit `8fcb54b2fb28388efb711786630501944fddec28`. The new environment path
is `/projects/dahu1128/.venv-completion-v2`; bootstrap Python is
`/usr/bin/python3.11` (3.11.13). The existing AMOS checkout and environment were
not modified. No source archive was uploaded.

## Build record

| Job | Status/evidence | Action |
|---|---|---|
| 32173090 | FAILED, exit 1:0, elapsed 1 s, batch MaxRSS 3008 K | Stopped before environment creation because `BSK_PROJECT_ROOT` was absent inside Slurm. |
| 32173134 | FAILED, exit 1:0, elapsed 19 s, batch MaxRSS 880844 K | Explicit exports worked. Torch's CPU index lacked the `flit_core` backend needed by the downloaded `typing_extensions` source distribution. |
| 32173261 | FAILED during dependency resolution | CPU Torch installation succeeded. Ray's optional `rllib` extra required Gymnasium 0.28.1, conflicting with the tested 0.29.1 pin. |
| 32204475 | FAILED, exit 1:0, elapsed 9:59, batch MaxRSS 9322588 K | All Python/build dependencies installed and CSPICE built. CMake could not find Python headers on the compute image. |
| 32207880 | FAILED, exit 1:0, elapsed 2:06, batch MaxRSS 2190164 K | Matching 3.11.13 headers were found and CMake configured. GCC 8.5 could not link the C++17 `std::filesystem` build-info probe. |

The build correction downloads only the CPU Torch wheel from its dedicated index
(`--no-deps`), then resolves all runtime dependencies together from PyPI under the
existing pins. This keeps the selected CPU learner and avoids asking the Torch
wheel index to serve unrelated build tools. The failed new environment will be
renamed with its job ID before a fresh build; the AMOS environment remains intact.

The second dependency correction installs `ray==2.35.0` with explicit RLlib
dependency pins from the saved working preflight environment, including
Gymnasium 0.29.1. It avoids requesting the incompatible optional `rllib` extra.
The same Ray implementation is installed; runtime and PPO validation still must
pass on Linux. No tested primary package version or mission setting is changed.

The compute image omits `/usr/include/python3.11`, although the login image has
the matching `python3.11-devel-3.11.13` package. Those headers are copied into a
separate project path, version-checked against the runtime, hashed in the build
record, and passed through Conan's documented CMake toolchain variables. The
compute image's GCC 8.5 also fails Basilisk's C++17 filesystem probe. The resumed
build selects Alpine's shared GCC 14.2 installation, refreshes only its isolated
Conan profile, and cleans only Basilisk's marker-validated generated build folder.

The interactive shell reports `sbatch is aliased to sbatch --export=NONE`.
The retry explicitly supplied `--export=ALL` and the four reviewed path variables.
Future runbook launches include `--export=ALL`. Build logs are
`bsk-runtime-<jobid>.log`; successful builds write their audit and dependency locks
under `results/multiagent_imaging/runtime-build-<jobid>/`.

Both submissions request account `ucb550_asc2`, partition `acpu`, QoS `cpu-normal`,
one node, eight CPUs per task, 32 GiB, and four hours. Slurm accounting reported nine
allocated CPUs for the first failed job; this differs from the eight requested
CPUs and should be retained when reporting charged resources.

## Remaining authorized work

Only after the runtime audit passes: submit baseline tasks 0 and 100 (LEO, seed
zero, independent and centralized full-state greedy) and the one-worker directed
PPO validation stage. Preserve all mission settings and the implemented two-orbit
cooldown. Compare matched initial-condition hashes and separate distinct-target
capture coverage from full ground-delivery coverage. A single pair cannot supply
Monte Carlo uncertainty or demonstrate reliable 100% coverage.

The remaining 198 baseline episodes, four-worker learning, and the broad six-cell
study remain outside the current submission authorization.
