# Multi-agent space imaging 2026 — completion sharing

Option 2 is implemented on the existing AMOS-derived `multi-agent-space-imaging-2026`
branch: persistent time-tagged completion catalogs, conflict/continuous retasking,
directed physical metadata transmissions (with a broadcast baseline), and shared target-set
attention PPO. [CLUSTER_READINESS.md](CLUSTER_READINESS.md) records the current preflight. Only sensing spacecraft are RL
agents. Every passive RSO remains a real propagated Basilisk spacecraft and a spacecraft
in Vizard's scene.

For the prepared Alpine deployment and staged learning commands, read
[cluster/README.md](cluster/README.md). The separate 200-episode independent versus
centralized full-state deterministic baselines are documented in
[BASELINE_MONTE_CARLO.md](BASELINE_MONTE_CARLO.md); their omniscient joint controller
does not change the completion-only policy observation/payload contract below.

Read [ARCHITECTURE.md](ARCHITECTURE.md) for event order, source-file responsibilities,
catalog merge rules, and channel assumptions; [OBSERVATION_VALIDATION.md](OBSERVATION_VALIDATION.md)
for the exact observation/action contract; and [PROSPECTUS_MAPPING.md](PROSPECTUS_MAPPING.md)
for the slide-to-code mapping. The earlier branch/upstream review is retained in
[RETASKING_COMPLETION_REVIEW.md](RETASKING_COMPLETION_REVIEW.md).

## Cases and retasking

| Information case | What reaches a sensor's catalog |
|---|---|
| `independent` | Its own captures and ground deliveries |
| `ideal_completion` | Automatic all-to-all completed metadata at event boundaries |
| `completion` | Completed metadata received after finite broadcasts, with configurable LOS, latency, loss, and TTL |

None shares planned actions, intentions, or private peer-sensor state. Ground image
products stay with their source spacecraft until downlink. Catalogs merge per-exposure
facts, then derive the newest relevant target summaries. Packet expiry does not erase
facts already learned. Acquisition freshness and ground-delivered information age both
use acquisition timestamps, with request generation and revisit cooldown handled separately.

Set `retasking_mode="conflict"` to reevaluate on own action end or a locally known
completion of the current imaging request. Set `"continuous"` to reevaluate all live
sensors at every environment boundary. Policy action 4 deliberately continues an unfinished
task. Selecting the same current target or operational mode also preserves the original
progress and deadline. It does not restart the action.

The actor uses `completion-v2`: **26 own/global + 17 per target candidate + 12 per peer**
in directed mode. There are **K+5+P actions**, with P=N_sensors−1. Broadcast mode omits
peer rows/actions. Padding is excluded inside attention and pooling, and all-empty target
sets preserve resource-sensitive operational choices. Old checkpoints require retraining.
Critic inputs remain local. See the exact contract linked above.

Production preflight configurations are `mission_preflight_{conflict,continuous}.json`:
**two sensors, 100 passive RSOs, ten candidates, 45,000 seconds**, directed LOS transmission,
10-second minimum hold, optional 64 kbit/s airtime, and 45,000-second discount half-life.
Tiny configurations remain regression tools. Full-scale one-update preflight is a plumbing
and cost check; it does not establish policy convergence or a comparison between methods.

## Local verification and matched experiments

Run from this worktree (the Python path below refers to the existing shared local venv):

```bash
export PYTHONPATH=src:.
PYTHON=/Users/dahu1128/Repositories/bsk_rl/.venv/bin/python

$PYTHON -m pytest -q tests/unittest/multiagent
$PYTHON -m pytest -q tests/integration/multiagent

$PYTHON -m examples.multiagent_imaging.run_matched_validation \
  --config examples/multiagent_imaging/configs/completion_conflict.json \
  --duration 1200 --seed 0 \
  --output-dir results/multiagent_imaging/completion_validation
```

The runner crosses three information cases with two retasking modes and checks identical
initial spacecraft states/priorities. JSON includes team reward, unique/duplicate service,
duplicate and interrupted sensor-seconds, wasted-time fraction, communication duration,
policy decision counts, packet outcomes, and task history. The deterministic shared
controller is a plumbing baseline; it does not establish a learned-policy advantage.

Run an individual rollout or a short PPO iteration:

```bash
$PYTHON -m examples.multiagent_imaging.evaluate \
  --config examples/multiagent_imaging/configs/completion_conflict.json \
  --output results/multiagent_imaging/completion_seed0.json \
  --plots-dir results/multiagent_imaging/completion_seed0_plots

$PYTHON -m examples.multiagent_imaging.train \
  --config examples/multiagent_imaging/configs/completion_continuous.json \
  --iterations 1 --train-batch-size 64
```

New explicit case files are `{independent,ideal_completion,completion}_{conflict,continuous}.json`.
Existing `smoke.json` now selects ideal completion; `smoke_los_broadcast.json` selects
finite LOS completion. Older filenames containing `centralized_information` or `intent`
are retained as migrated launch aliases: inspect their `information_case` rather than
inferring semantics from the filename. Old schema fields such as
`perfect_metadata_delivery` are no longer accepted. `full*.json` are launch specifications;
the dedicated mission preflight configurations and evidence are described in CLUSTER_READINESS.md.

A quick three-sensor plot demonstration is also available:

```bash
$PYTHON -m examples.multiagent_imaging.run_quick_demo \
  --n-sensors 3 --n-targets 12 --n-candidates 4 --duration-s 1200 \
  --information-case completion --los-broadcast \
  --output-dir results/multiagent_imaging/quick_completion_demo
```

This writes per-sensor plots and, for multiple sensors, a team overview as PDFs/PNGs.
Existing result JSON can be plotted using `examples.multiagent_imaging.plot_evaluation`.

## Vizard: spacecraft roles do not change physics

```bash
$PYTHON -m examples.multiagent_imaging.run_vizard_demo \
  --information-case ideal_completion --retasking-mode conflict \
  --output-dir results/multiagent_imaging/vizard_completion
```

Defaults are six sensors, 100 passive LEO spacecraft, 3000 simulated seconds, and a frame
every two seconds. The shared closest-angle heuristic and original pre-Walker staggered
sensor orbits are retained. The output is a canonical `*_UnityViz.bin` playback plus
`rollout.json`; use a new directory or explicit `--overwrite` for an existing recording.

All 106 spacecraft are supplied to Vizard with their state messages. Sensors retain their
attitude-driven CAD models. Targets use native spacecraft model/sprite transitions and
priority-colored distant sprites (34 light, 33 medium, 33 dark blue for 100 targets).
There are no moving RSO `Location` substitutes. Ground stations still use ground locations
and cones. Imaging lines refer to the actual target spacecraft and change from yellow to
green during valid pointing hold. Optional orbit histories start hidden for responsiveness.
The tiny integration recording verifies scene membership and propagated target state;
interactive Vizard rendering is a separate visual check.

## Scope and next work

The metadata link is a finite-duration, boundary-sampled omnidirectional channel, with
ideal ACK bookkeeping. It has no finite-byte bandwidth, antenna pointing, RF budget, or
metadata-radio power model. Broadcasts occupy time and optionally incur
`communication_cost_per_s`. These limits are explicit in the architecture note.

Next, run paired learned-policy pilot experiments across all six cells with multiple
matched seeds; inspect learning stability, source ownership, and waste/communication
tradeoffs before scaling sensor/target count or launching cluster training. Keep peer
intent and peer-state observations absent for this completion-sharing study. See
[NEXT_PROMPT.md](NEXT_PROMPT.md) for a ready-to-use follow-up request and
[TEST_RESULTS.md](TEST_RESULTS.md) for validation evidence.

## Restored policy evaluation and cluster launcher

```bash
$PYTHON -m examples.multiagent_imaging.evaluate \
  --config examples/multiagent_imaging/configs/mission_preflight_conflict.json \
  --checkpoint results/multiagent_imaging/cluster_readiness/conflict/checkpoint_0001 \
  --seed 101 --output results/multiagent_imaging/restored_evaluation.json
```

The evaluator strictly validates the saved formulation/schema, loads actual weights and
checks saved logits before rollout. Training supports `--resume CHECKPOINT`, resource
flags, complete-episode batches and per-update measurement/checkpoints. The prepared
`cluster/train.slurm` defaults to a CPU learner and four rollout processes on one
8-CPU/32-GiB node. Set `BSK_RL_PYTHON` to the validated cluster interpreter. No jobs have
been submitted. Use [NEXT_PROMPT.md](NEXT_PROMPT.md) for the recommended next task.
