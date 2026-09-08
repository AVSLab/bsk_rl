# Authorized Alpine deployment and validation

Code deployment uses Git, as requested. The desktop pushed
`006b71c8d0c938d7fcaed250769f9de85559f17e` to
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
| 32208040 | FAILED, exit 1:0, elapsed 5:03, batch MaxRSS 6265936 K | GCC 14 configured and compiled Basilisk sources, but generated `protoc` loaded the system GCC 8 `libstdc++`, missing `GLIBCXX_3.4.32`. |
| 32208367 | Failed during native compilation at 67% | Compiler/runtime pairing passed. Eight simultaneous GCC 14 SWIG-wrapper compilations exceeded the 32-GiB allocation and two `cc1plus` processes were killed. |
| 32208594 | Failed during native compilation at 65% | Two simultaneous large wrapper compilations still exceeded 32 GiB. |
| 32208739 | OUT_OF_MEMORY, exit 0:125, elapsed 3:02, batch MaxRSS 33551916 K | `CMAKE_BUILD_PARALLEL_LEVEL=1` was insufficient because Conan explicitly appended `-j128`. Slurm detected 26 OOM-kill events. |
| 32208877 | FAILED audit after a completed 29:12 serialized build, batch MaxRSS 4232544 K | The native build stayed inside memory, but the audit caught a stale undersized `_vizInterface.so` instead of accepting a damaged runtime. |
| 32282851 | FAILED before build | The resume lock changed only because the editable project Git HEAD changed; third-party dependency comparison now deliberately ignores that project-owned lock line. |
| 32282983 | FAILED audit after a clean 1:06:43 serialized rebuild, batch MaxRSS 7349684 K | Native modules passed the stale-artifact checks. BSK-RL still expected the legacy `Basilisk` distribution name; the pinned source correctly installs current metadata as `bsk`. |
| 32288526 | **COMPLETED**, exit 0:0, elapsed 4:02, batch MaxRSS 819024 K | Incremental retry passed the runtime and support-data audits. The runtime JSON has `passed: true`, no errors, the separate checkout, pinned source, package lock, native hashes and five verified support files. |

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
The build and both simulation launchers also prepend that toolchain's `lib64` and
`lib` directories to `LD_LIBRARY_PATH`, pairing compiled programs and Basilisk
modules with the C++ runtime that built them.
Native build parallelism is reduced to one inside the same 32-GiB allocation;
the eight requested CPUs remain the reviewed resource shape for Ray validation.
The script sets Conan's documented `tools.build:jobs=1` configuration as well,
so its explicit build-system flag cannot override the memory limit.

The interactive shell reports `sbatch is aliased to sbatch --export=NONE`.
The retry explicitly supplied `--export=ALL` and the four reviewed path variables.
Future runbook launches include `--export=ALL`. Build logs are
`bsk-runtime-<jobid>.log`; successful builds write their audit and dependency locks
under `results/multiagent_imaging/runtime-build-<jobid>/`.

Build submissions request account `ucb550_asc2`, partition `acpu`, QoS `cpu-normal`,
one node, eight CPUs per task, 32 GiB, and four hours. Slurm accounting reported nine
allocated CPUs for the first failed job; this differs from the eight requested
CPUs and should be retained when reporting charged resources.

The current audit accepts the distribution name `bsk` used by the pinned Basilisk
source and the legacy name `Basilisk`. It also resolves dependency metadata from the
environment itself and records, but does not mistake for the runtime, Ray's vendored
`ray/thirdparty_files` metadata. Diagnostic job 32289996 proved that the imported
`psutil` module is environment version 6.1.0 even though Ray prepends a vendored
6.0.0 metadata directory during RLlib import.

## Authorized baseline pair

Baseline array job 32289452 ran only tasks 0 and 100. Both used LEO target geometry,
seed zero, two sensing spacecraft, 100 passive Basilisk spacecraft, ten candidates,
a 45,000-second horizon, and the already implemented 11,960.807123947805-second
cooldown. The independent and centralized-full-state episodes have the identical
initial-condition hash
`990f3807aa774525dba94d3946ea0f22be8070c91745f4d6fce890e16cef7ade` and manifest
hash `5166467eabfc8e33f3986c24add4388660f919aa7c909a2581cae4e0396fb93a`.

| Case/task | Coverage (capture / ground) | Reward | Duplicate sensor-s | Unique capture / ground services | Wall / peak RSS |
|---|---:|---:|---:|---:|---:|
| Independent / 0 | 88/100 / 88/100 | 237.8861 | 26,565 | 236 / 211 | 303.96 s / 1,972,326,400 B |
| Centralized full state / 100 | 88/100 / 88/100 | 250.9969 | 0 | 249 / 237 | 275.48 s / 1,971,191,808 B |

Both episodes reached the horizon with both sensors alive and zero radio occupancy.
Centralized information removed all 164 duplicate attempts and all 157 successful
duplicates recorded in the independent episode. It reduced the wasted-time fraction
from 0.29517 to zero and raised ground-service count by 26, but did not change distinct
coverage in this one pair. The centralized-minus-independent reward difference was
+13.1107. Task 0 completed in 5:21 with batch MaxRSS 1,911,048 K; task 100 completed
in 4:52 with batch MaxRSS 1,953,920 K. Although the script requests one CPU per task,
Slurm charged two allocated CPUs per task on this partition.

Aggregation intentionally reports `complete_campaign: false`, two completed episodes,
and one validated information pair. This is a smoke comparison, not a Monte Carlo
estimate: it supplies neither uncertainty nor evidence of reliable 100% coverage.
Tasks 1-99 and 101-199 have not been submitted.

## Authorized one-worker directed PPO validation

Job 32291244 ran the validation stage only in
`results/multiagent_imaging/cluster-one-worker-20260908-006b71c`. It completed in
29:56 with exit 0:0 and batch MaxRSS 4,709,828 K. The script requested eight CPUs
and 32 GiB; Slurm reported nine allocated CPUs. Its runtime audit passed at source
commit `006b71c` and Basilisk commit
`8fcb54b2fb28388efb711786630501944fddec28`.

Each directed finite-completion mode collected one complete 45,000-second episode,
saved a checkpoint, restored the full Ray PPO/Adam and deterministic worker seed
state in a second process lifetime, collected another complete episode, and saved
iteration 2. All four updates had finite losses and gradients, nonzero parameter
changes, exact restored logits (`max_logit_error: 0`) and matching actions.

| Mode/update | Env / agent steps | Decisions | Total / policy / value loss | Gradient L2 | Parameter change L2 |
|---|---:|---:|---:|---:|---:|
| Conflict 1 | 507 / 1014 | 519 | 7.79553 / -1.10835 / 8.90117 | 0.85641 | 0.04101 |
| Conflict resumed 2 | 441 / 882 | 446 | 1.76676 / 1.39809 / 0.62695 | 0.24040 | 0.02916 |
| Continuous 1 | 375 / 750 | 750 | 8.81772 / -0.05657 / 7.95984 | 1.30056 | 0.03987 |
| Continuous resumed 2 | 303 / 606 | 606 | 7.58672 / 0.69269 / 6.89403 | 1.96897 | 0.04768 |

Every update exceeded the requested 64-step minimum through complete episodes. The
changed-element counts were respectively 67,664, 71,645, 72,071 and 74,197. Physical
task histories covered both sensors for the full horizon, product metadata never
changed physical storage ownership, and event-boundary resource histories remained
finite. All sensors stayed alive. Across the four episodes the minimum sampled battery
fraction was 0.86279, maximum sampled storage fraction was 1.0, and maximum absolute
wheel-speed fraction was about 0.1388.

Directed communication was exercised in every sampled episode. Conflict updates
recorded 29 and 19 accepted packets, 58 and 38 pointing/transmission records, and
290 and 190 radio-seconds. Each conflict episode paired every `hold_complete` outcome
with one outcome-less start record. Continuous updates recorded 29 and 28 accepted
packets, 92 and 60 records, and 310 and 293 radio-seconds. Their outcome counts were
29/28 `hold_complete`, 17/2 `policy_switch`, and 46/30 outcome-less start records.
The saved records retain receiver, payload bytes and physical start/end times.

The final restored checkpoints were actually called for every policy decision on
held-out seed 10000 and compared with the closest-angle heuristic under identical
initial conditions. Conflict returned -6357.5790 versus heuristic 172.2807;
continuous returned -6329.1932 versus heuristic 146.9725. This is expected for two
updates from random initialization and proves execution/restore, not learning quality.
The gate is `passed: true`, `workers: 1`, `validated_workers: 1`.

## Submission boundary

The runtime build, tasks 0 and 100, and one-worker validation are complete. The
remaining 198 baseline episodes, four-worker learning, and the broad six-cell study
remain outside the current submission authorization and were not submitted.
