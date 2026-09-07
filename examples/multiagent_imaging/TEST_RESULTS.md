# Cluster and baseline preparation — September 6, 2026

The final focused suite passed **17 tests** (pilot gates, schema/worker seeds, and
12 baseline unit/native-simulator tests). Log: `/tmp/completion-cluster-final-tests.log`.
An earlier six-test run also passed the full policy/Adam/seed resume integration
test: `/tmp/completion-cluster-preparation-tests.log`.

A separate real **one-remote-worker** regression used three sensors, four passive
RSOs, two candidate slots and complete 360-second episodes. Initial update collected
20 environment steps/60 agent rows/24 policy decisions from two complete episodes;
resumed update collected 20/60/32 from three complete episodes. Both had finite
losses/gradients, changed 62,520 and 65,383 weight elements, and restored with zero
probe-logit error. PPO advanced from iteration one to two with Adam/seed validation.
Event-boundary resource histories were present. Restored weights and the heuristic
ran matched held-out seed 10000. Artifacts:
`results/multiagent_imaging/cluster_preparation/one_worker/`;
log `/tmp/completion-one-worker-check.log`. This is a regression, not a mission-scale
learning result. The short restored policy performed poorly; no convergence claim
is made from two updates.

Four saved 1800-second/six-target baseline episodes completed and paired correctly:
LEO capture coverage 4/6 in both information cases; mixed capture coverage 5/6 in
both. These short episodes had no ground deliveries. They used no radio actions;
all passive spacecraft propagated. Artifacts:
`results/multiagent_imaging/baseline_mc_validation/`. The production manifest uses
100 targets and 200 tasks; regenerate it after final deployment/source changes.

The runtime audit passes locally after correcting psutil's pin to the saved 6.1.0.
Ruff and all cluster shell syntax checks pass. Live Alpine audit, incompatible old
runtime, reviewed commands and remaining native-build validation are documented in
[cluster/README.md](cluster/README.md). **No cluster jobs were submitted.**

# Earlier completion-v2 readiness validation

**190 tests passed**: 85 multi-agent unit/integration checks plus 105 core checks in a
separate process. Full 100-RSO, 45,000-second profile and one PPO update in each retasking
mode passed. Checkpoint logits, optimizer resume, actual updates and two-worker seed
streams were validated. See [CLUSTER_READINESS.md](CLUSTER_READINESS.md) for measured
batches, timings, memory, coverage and remaining limits.

Logs: `/tmp/completion-readiness-final-suite.log`, `/tmp/completion-core-regression.log`,
`/tmp/completion-preflight-conflict.log`, `/tmp/completion-preflight-continuous.log`,
`/tmp/completion-worker-preflight.log`, `/tmp/completion-full-restore.log`.

The results below are historical validation of the preceding implementation.

---

# Option 2 implementation validation — September 5, 2026

Validated in the `multi-agent-space-imaging-2026` worktree using the existing
`/Users/dahu1128/Repositories/bsk_rl/.venv/bin/python` and `PYTHONPATH=src:.`.
The implementation preserves the AMOS-derived base and existing local Vizard/FSW work,
while correcting the passive-target moving-Location substitution. No rebase or push was
performed. Old intent observation checkpoints require retraining for completion-v1.

## Executed checks

| Checks | Result |
|---|---:|
| `tests/unittest/multiagent` (catalog/channel/products/masks/timing/legacy roles and messaging) | 55 passed |
| `tests/integration/multiagent` (real Basilisk and RLlib, final run) | 19 passed |
| Existing `tests/unittest/act`, `obs`, `test_gym_env.py`, `sim/test_simulator.py` | 105 passed |
| Scoped Ruff checks on new components and changed example/test files | Passed |
| `git diff --check` | Passed |

**179 passing tests across these groups.** The integrated suite includes one real PPO
training iteration in each retasking mode, with finite total, policy, and value losses.
It also checks implicit reset-seed reproducibility. This is training-pipeline validation,
not a convergence or learned-performance claim. Existing Basilisk deprecation and
Gymnasium wrapper/registry warnings remain.

Behavioral regressions include:

- receiver-local conflict interruption in both modes, unrelated tasks continuing, and
  global reward truth being unable to retask an agent;
- preserving active hold identity, attempt start, and deadline for deliberate continue
  and repeated selection of the same imaging target;
- a 30-second broadcast staying undelivered at 5-second boundaries, then a seven-second
  packet delay producing reception at t=37; one physical broadcast counted once;
- out-of-order exposure/delivery facts, durable knowledge beyond transport expiry,
  packet loss/retry, LOS interruption, no failed-exposure completion broadcasts, and
  source/ownership preservation;
- no capture credit from raw image bits before hold completion; no complete-product
  delivery credit from partial downlink, including native Basilisk SWIG index handling;
- empty candidates explicitly masked, fixed observation size across sensor counts,
  and training/inference/exploration masks;
- physical-time reward aggregation, two-pass PPO idempotence, complete-episode bootstrap
  handling, and real final observations for busy truncated agents;
- all 106 scene sprite entries for six sensors/100 passive targets; a real small Vizard
  recording contains every spacecraft and changing target state, with only sensors in
  the RL agent list. Interactive Vizard rendering/full video playback was not reviewed.

## Six matched deterministic rollouts

Ran `run_matched_validation` with the smoke physical settings, seed 0, two sensors,
eight targets, four candidates, and a 1200-second horizon. The runner verified identical
initial spacecraft states and priorities in all six cells. Each cell produced two unique
qualified acquisitions, total agent reward **32.474679**, zero ground deliveries, and
zero duplicate acquisitions. The short horizon therefore does not establish a ground
service or duplicate-avoidance benefit.

| Case | Policy decisions, summed | Broadcast sensor-seconds | Wasted imaging-time fraction |
|---|---:|---:|---:|
| Independent / conflict | 15 | 0 | 0 |
| Independent / continuous | 20 | 0 | 0.070833 |
| Ideal completion / conflict | 15 | 0 | 0 |
| Ideal completion / continuous | 20 | 0 | 0.070833 |
| Finite completion / conflict | 16 | 30 | 0 |
| Finite completion / continuous | 24 | 60 | 0.070000 |

The nonzero waste here is nonduplicate policy interruption under the heuristic. It is
not evidence that a trained continuous policy must perform worse. The finite LOS cases
accepted one and two packets respectively; the ideal cases each accepted two.

Results are in `results/multiagent_imaging/completion_implementation_validation/`:
`summary.json`, six cell JSON files, and per-sensor/team PDF/PNG
plots under `plots/`. JSON exports unknown timestamps as null. Results are local ignored
artifacts; launch configurations and documentation are versionable. Historical results
below describe the earlier intent/status implementation and do not validate completion-v1.

# Historical validation before completion-v1

# Verification record

Date: 2026-08-25
AMOS base commit: `0a05f2bd72872dc8272da673b550b3f1c9daafab`

All commands below were run from the isolated
`bsk_rl-multi-agent-space-imaging-2026` worktree with:

```bash
export PYTHONPATH=src:.
PYTHON=/Users/dahu1128/Repositories/bsk_rl/.venv/bin/python
```

## Focused multi-agent verification

```bash
$PYTHON -m pytest -q \
  tests/unittest/multiagent \
  tests/integration/multiagent/test_two_sensor_rollout.py \
  tests/integration/multiagent/test_rllib_smoke.py
```

Result after adding multi-agent evaluation plotting: **36 passed**. Six warnings are upstream
Ray/Gymnasium deprecation warnings.

This includes role/passive exclusion, independent access and storage, local-knowledge
separation, global-truth non-leakage, message ordering and expiry, deterministic reward
credit, asynchronous `d_ts` condensation, all four information/delivery cases,
deterministic two-sensor Basilisk rollout, freshness-weighted same-target intent,
peer-message-order invariance, fixed shape for one/two/three sensors, strict
information-case separation, actor target-permutation equivariance, shared-policy mapping,
and one short RLlib PPO update.

The focused suite also verifies that every sensing spacecraft receives its own diagnostic
PDF/PNG and that the combined catalog overview is generated only when more than one
sensing agent is present.

## Complete unit regression

```bash
$PYTHON -m pytest -q tests/unittest
```

Result: **516 passed, 1 skipped**.

## Integration regression

```bash
$PYTHON -m pytest -q tests/integration \
  --deselect tests/integration/act/test_int_actions.py::TestDesatAction::test_desat_action_power_draw \
  --deselect tests/integration/scene/test_int_scenarios.py::TestCityTargets::test_city_distribution
```

Result: **61 passed, 1 skipped, 2 deselected**.

The two deselected tests also fail at the untouched AMOS base commit. The desaturation
test does not drain the battery under the installed Basilisk runtime, and the city test
lacks the optional untracked `worldcities.csv` asset. They are inherited environment/test
limitations rather than differences introduced by this branch.

## Existing AMOS runtime validation

```bash
$PYTHON examples/amos_2026/validate_profile_speed_flags.py
```

Result: **passed** (`steps=2`, `seed=123`, `sim_time=190.000`). Existing passive-target
battery warnings remain unchanged.

## Deterministic bounded rollouts

```bash
$PYTHON examples/multiagent_imaging/evaluate.py \
  --config examples/multiagent_imaging/configs/smoke.json \
  --output /tmp/multiagent_imaging_smoke_seed0.json

$PYTHON examples/multiagent_imaging/evaluate.py \
  --config examples/multiagent_imaging/configs/smoke_los_broadcast.json \
  --output /tmp/multiagent_imaging_smoke_los_seed0.json
```

Both reached 1,200 seconds with exactly two PettingZoo agents and eight passive RSO
spacecraft. The perfect-metadata run produced 3 and 5 captures without requiring a
broadcast. The LOS-broadcast run executed two finite broadcasts per sensor and produced
directional remote-pending knowledge at both receivers.

## Phase-two matched information cases

```bash
$PYTHON examples/multiagent_imaging/run_matched_validation.py \
  --output-dir results/multiagent_imaging/matched_validation_simplified_observation
```

Result: **passed**. The runner confirmed identical initial sensor states, target states,
priorities, seeds, reward settings, and 1,800-second horizons across all four cases.

| Case | Unique acquisitions | Acquired value | Conflict time | Broadcast time per sensor |
|---|---:|---:|---:|---:|
| Independent | 2 | 49.72 | 1,045 s | 0 s |
| Centralized information | 4 | 76.58 | 193 s | 0 s |
| Perfect intent/status | 3 | 63.46 | 101 s | 0 s |
| LOS intent/status | 2 | 49.72 | 763 s | 90 s |

These deterministic shared-controller runs validate information flow and diagnostics, not
policy performance. No ground delivery completed in the bounded horizon. The revised
observation contains 14 own-spacecraft/environment features and 13 features per target; it
contains no generic peer-resource or peer-action vector. The saved local results include
reward/resource histories, action durations, duplicate counts, message ages/dispositions,
intent conflicts, per-sensor local catalogs and physical products, and local-versus-shared
omission counts.

## Static checks

```bash
$PYTHON -m ruff check \
  src/bsk_rl/__init__.py src/bsk_rl/act/__init__.py \
  src/bsk_rl/comm/__init__.py src/bsk_rl/data/__init__.py \
  src/bsk_rl/obs/__init__.py src/bsk_rl/sats/__init__.py \
  src/bsk_rl/gym.py src/bsk_rl/sats/satellite.py \
  src/bsk_rl/obs/observations.py src/bsk_rl/utils/rllib/discounting.py \
  src/bsk_rl/utils/coordination.py \
  src/bsk_rl/sats/roles.py src/bsk_rl/comm/rso_communication.py \
  src/bsk_rl/comm/typed_messages.py src/bsk_rl/data/multiagent_rso_data.py \
  src/bsk_rl/data/multiagent_rso_reward.py examples/multiagent_imaging \
  tests/unittest/multiagent tests/integration/multiagent

git diff --check
```

Result: **passed**.

No full Monte Carlo campaign or long training run was launched.

## Quick multi-agent visualization check

```bash
$PYTHON examples/multiagent_imaging/run_quick_demo.py \
  --n-sensors 3 --n-targets 12 --n-candidates 4 \
  --duration-s 1200 --seed 0 \
  --output-dir results/multiagent_imaging/quick_demo_3sensors_seed0
```

Result: **passed**. The run produced three per-sensor diagnostic figures and one
multi-agent catalog overview in both vector PDF and PNG formats. A separate one-sensor
check produced only its per-sensor figure, confirming the multi-agent-only plot gate.
