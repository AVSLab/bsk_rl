# Reviewed Alpine launch preparation — 2026-09-06

The completion branch is deployed through Git. The user authorized the runtime
build, then (only after its audit passes) baseline tasks **0 and 100** and the
**one-worker validation** stage. See [EXECUTION.md](EXECUTION.md) for live job
records. The remaining 198 baseline episodes and four-worker learning still
require separate authorization. No cluster learning/coverage result is claimed
until its recorded job completes and its checks pass.
The immediate priority is the [200-episode baseline campaign](../BASELINE_MONTE_CARLO.md).
The directed-completion learning pilot is a separate, bounded workflow.

## Verified live

Read-only commands ran through the user's Termius connection to
`dahu1128@login.rc.colorado.edu` (`login-ci4`). Noninteractive desktop SSH was
denied, so no unattended file-transfer path is assumed.

| Item | Observed on the cluster |
|---|---|
| Existing project | `/projects/dahu1128/bsk_rl`, branch `amos-2026-cluster-uncommitted-20260903` |
| Existing commit | `047131844e99d3a06af9f4a6c70604d92875b14b` |
| Completion destination | `/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026` **absent** |
| Existing interpreter | `/projects/dahu1128/.venv/bin/python`, Python 3.10.2 |
| Candidate interpreter | `/usr/bin/python3.11` exists; venv/build capability remains to be tested |
| Existing packages | Ray 2.35.0, Torch 2.0.1, NumPy 1.26.4, Gymnasium 0.28.1, PettingZoo 1.24.3, PyArrow 20.0.0, SciPy 1.15.2; psutil missing |
| Existing Basilisk | 2.3.29 wheel in the old Python 3.10 environment |
| Account association | `ucb550_asc2` on Alpine, including `cpu-normal` QOS |
| CPU partition | `acpu`, UP, 420 nodes, maximum one day; historical `amilan` query returned nothing |
| QOS | `cpu-normal`, maximum one day, reported max nodes/user 128 |
| User queue | No jobs shown at audit time |

Account eligibility is verified, not a reservation or remaining service-unit
balance. No allocation was requested. The existing AMOS checkout/environment stays
intact. The separate destinations below are proposed paths, not installed runtimes.

## Deployment and runtime

**Git deployment:** use the committed
`multi-agent-space-imaging-2026` branch. Create a separate repository and fetch only
that branch; verify its HEAD against the reported desktop commit before building:

```bash
export BSK_PROJECT_ROOT=/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026
export BSK_RL_PYTHON=/projects/dahu1128/.venv-completion-v2/bin/python
export BASILISK_SOURCE_ROOT=/projects/dahu1128/basilisk-completion-v2
export BSK_BOOTSTRAP_PYTHON=/usr/bin/python3.11
test ! -e "$BSK_PROJECT_ROOT"
git init "$BSK_PROJECT_ROOT"
git -C "$BSK_PROJECT_ROOT" remote add origin https://github.com/AVSLab/bsk_rl.git
git -C "$BSK_PROJECT_ROOT" fetch --depth=1 origin multi-agent-space-imaging-2026
git -C "$BSK_PROJECT_ROOT" switch -c multi-agent-space-imaging-2026 FETCH_HEAD
git -C "$BSK_PROJECT_ROOT" rev-parse HEAD
```

This replaces the originally prepared SFTP release transfer. The archived release
in `results/multiagent_imaging/cluster_preparation/release/` preserves the earlier
uncommitted source as a reproducibility artifact; it is not needed for deployment.
AMOS is not switched, pulled, or modified. Fetch the exact separate Basilisk source
on the login node (these commands do not simulate or submit):

```bash
export BSK_PROJECT_ROOT=/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026
export BSK_RL_PYTHON=/projects/dahu1128/.venv-completion-v2/bin/python
export BASILISK_SOURCE_ROOT=/projects/dahu1128/basilisk-completion-v2
export BSK_BOOTSTRAP_PYTHON=/usr/bin/python3.11
test ! -e "$BASILISK_SOURCE_ROOT"
git init "$BASILISK_SOURCE_ROOT"
git -C "$BASILISK_SOURCE_ROOT" remote add origin https://github.com/AVSLab/basilisk.git
git -C "$BASILISK_SOURCE_ROOT" fetch --depth=1 origin 8fcb54b2fb28388efb711786630501944fddec28
git -C "$BASILISK_SOURCE_ROOT" checkout --detach FETCH_HEAD
cd "$BSK_PROJECT_ROOT"
```

The recorded Basilisk source [commit is available upstream](https://github.com/AVSLab/basilisk/commit/8fcb54b2fb28388efb711786630501944fddec28).
Build it natively for Linux; neither the old wheel nor the desktop ARM binary is a
validated replacement. A source HEAD alone does not establish when an existing
binary was built. The fresh build records source, compiler, resolved tools,
dependencies, and hashes of the native modules actually loaded.

The live interactive shell aliases `sbatch` to `sbatch --export=NONE`. Every launch
below explicitly overrides that alias with `--export=ALL`; otherwise the job loses
the reviewed project/interpreter paths and fails before setup. The authorized build:

```bash
sbatch --export=ALL examples/multiagent_imaging/cluster/build_runtime.slurm
```

The build requests one node/eight CPUs/32 GiB/four hours; refuses an existing venv;
builds Vizard support; and runs the runtime gate. It resolves the native build-tool
ranges declared by the exact Basilisk checkout and saves a build lock. These native
tools have not yet been resolved or tested on Alpine. Runtime pins are in
`requirements.txt`; each run also saves the full transitive dependency freeze.
The old psutil pin was corrected from 6.0.0 to **6.1.0**, matching the actual saved
preflight freeze. No existing environment was modified.

A failed build may leave its new venv for inspection; the script refuses a blind
overwrite. Do not proceed after any build, native import, source association, or
package check failure. Once built, this read-only gate is safe on the login node:

```bash
export PYTHONPATH="$BSK_PROJECT_ROOT/src:$BSK_PROJECT_ROOT"
"$BSK_RL_PYTHON" examples/multiagent_imaging/cluster/audit_runtime.py \
  --output results/multiagent_imaging/runtime-login-check.json
```

## Baselines: exactly 200 episodes

Before concurrent simulations, fetch just the five mission support-data files
through Basilisk's data resolver and verify their hashes against the successful
desktop preflight. This is file preparation, safe on the login node. Use a separate
cache so completion runs do not write to the AMOS support-data cache:

```bash
export BSK_SUPPORT_DATA_CACHE="$BASILISK_SOURCE_ROOT/.support-data-cache"
"$BSK_RL_PYTHON" examples/multiagent_imaging/cluster/prepare_support_data.py \
  --output results/multiagent_imaging/support-data.json
```

Keep that cache variable exported for all baseline and PPO submissions. Do not
launch after a missing file or hash mismatch.

The [baseline runbook](../BASELINE_MONTE_CARLO.md) defines information, controllers,
metrics, and aggregation. All four cells use seeds 0–49, three sensors, 100 targets,
ten candidates, 45,000 seconds, and the **existing cooldown unchanged**: two median
initial sensor orbits, 11,834.835756586714 seconds for the 700/800/700-km team,
plus the own pending-ground-product restriction. Mixed targets are 50 LEO/30 MEO/
20 GEO. Controllers are independent local greedy and centralized-full-state joint
greedy, with no radio action. The centralized controller reads every live sensor's
full mission state on each asynchronous decision boundary. This is the maximum-
information baseline, though the greedy scheduler is not an optimality proof.

The v2 output adds two coverage-waste definitions: stale ground deliveries whose
capture timestamp is older than another sensor's delivered product, and simultaneous
cross-sensor physical ownership of the same qualified target product. The latter is
reported as affected targets/products, redundant acquisitions, and redundant
sensor-seconds. Ground-confirmed coverage remains separate from the existing
capture-anchored cooldown.

After runtime success, generate the final manifest; submit only once authorized:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BSK_MC_ROOT="$BSK_PROJECT_ROOT/results/multiagent_imaging/baseline-mc-3sensor-v2"
export BSK_MC_MANIFEST="$BSK_MC_ROOT/manifest.json"
export BSK_MC_OUTPUT="$BSK_MC_ROOT/episodes"
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.baseline_monte_carlo manifest \
  --config examples/multiagent_imaging/configs/baseline_mc.json --output "$BSK_MC_MANIFEST"
sbatch --export=ALL --array=0,100%2 examples/multiagent_imaging/cluster/baseline_mc.slurm
```

Review this first LEO information pair: initial-state fingerprints must match,
runtime gates must pass, horizon completion or resource failure must be explicit,
capture and ground coverage must be separate, physical ownership must hold, and
radio occupancy must be zero. These two episodes count toward 200. After reviewing
and authorizing expansion, submit the remaining 198 IDs:

```bash
sbatch --export=ALL --array=1-99,101-199%8 examples/multiagent_imaging/cluster/baseline_mc.slurm
"$BSK_RL_PYTHON" -m examples.multiagent_imaging.aggregate_baseline_monte_carlo \
  --manifest "$BSK_MC_MANIFEST" --episodes-dir "$BSK_MC_OUTPUT" \
  --output-dir "$BSK_MC_ROOT/report" --plots
```

Run aggregation only after jobs complete. Each array task requests one CPU/4 GiB/
one hour; maximum eight tasks concurrently. No Ray or learner is started.

## Directed learning: one worker before four

Use `pilot.slurm` for the staged learning workflow. It requests one eight-CPU/32-GiB
node, CPU learner, one Torch/BLAS thread per process, 1800-second rollout timeout.
It preserves completion-v2 shared target-set attention, two sensors/100 passive
RSOs/ten candidates/45,000-second complete episodes, 45,000-second reward half-life,
6000-second GAE trace half-life, directed LOS SimpleNav pointing, minimum hold 10
seconds, 64 kbit/s metadata, and 300-second attempt deadline.

After explicit authorization, one-worker validation runs both retasking modes
serially with training seed zero. Each mode performs one update, saves and checks
logits, restores PPO/Adam/worker seeds, performs a second update, then evaluates
restored weights and the heuristic on matched held-out seed 10000:

```bash
export BSK_STAGE=validate
export BSK_OUTPUT="$BSK_PROJECT_ROOT/results/multiagent_imaging/cluster-one-worker"
sbatch --export=ALL examples/multiagent_imaging/cluster/pilot.slurm
```

The gate is written only after both modes pass finite losses/gradients, weight
changes, complete physical task durations, and actual restored-policy calls at
every evaluation decision. Training saves event-boundary resource histories,
physical ownership checks, task records, and packet outcomes.

After reviewing the evidence and explicitly authorizing four workers:

```bash
export BSK_STAGE=learn
export BSK_VALIDATION_GATE="$BSK_PROJECT_ROOT/results/multiagent_imaging/cluster-one-worker/validation_gate.json"
export BSK_OUTPUT="$BSK_PROJECT_ROOT/results/multiagent_imaging/cluster-four-worker-pilot"
export BSK_UPDATES=8
sbatch --export=ALL examples/multiagent_imaging/cluster/pilot.slurm
```

Four-worker learning starts fresh matched seed-zero policies. It does not claim
an exact one-to-four-worker resume; resume requires the saved topology. The two
validation updates plus eight learning updates keep total work at **ten per mode**.
Only finite-directed completion/conflict and completion/continuous are included.
Final checkpoints are restored and compared against the heuristic on seeds
10000–10004. Actual batch sizes/decisions, losses, gradients, parameter changes,
resource histories, services/duplicates/waste, radio/packet records, checkpoints,
source/dependency snapshots and diagnostic learning curves are saved. One training
seed cannot establish robustness across trained policies or justify statistical
superiority. No broad six-cell study is launched.

## Estimates and current evidence

Saved desktop 100-RSO/45,000-second episodes took roughly 222–249 seconds, with
single-process PPO peaks 1.15–1.35 GiB. These are not Alpine measurements. Complete
episodes can exceed the 64-step minimum by hundreds of rows per worker. Planning
estimates, excluding queues/build time: baseline campaign 13–20 aggregate CPU-hours
(200 × roughly 4–6 minutes), about 2–3 hours at eight slots if speed is similar;
one-worker learning validation 30–60 minutes; four-worker eight-update/two-mode
pilot plus evaluation roughly 2–4 hours on its eight-CPU node. Slurm bounds are one
CPU-hour per baseline task and 32 allocated CPU-hours per learning/build job.
Revise from saved cluster timing/RSS and `sacct` Elapsed/MaxRSS/AllocCPUS.

Local preparation checks include 12 baseline tests; four saved 1800-second/six-target
episodes with matched initial states; pilot gate tests; and a real one-remote-worker
save/restore/resumed update. These small cases do not replace mission-scale cluster
validation or establish 100% coverage. The earlier full-mission evidence remains
in [CLUSTER_READINESS.md](../CLUSTER_READINESS.md). A larger multi-seed study is not
justified by the new preparation work alone.
