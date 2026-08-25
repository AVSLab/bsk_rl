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

Result: **22 passed**. Six warnings are upstream Ray/Gymnasium deprecation warnings.

This includes role/passive exclusion, independent access and storage, local-knowledge
separation, global-truth non-leakage, message ordering and expiry, deterministic reward
credit, asynchronous `d_ts` condensation, all three information cases, deterministic
two-sensor Basilisk rollout, shared-policy mapping, and one short RLlib PPO update.

## Complete unit regression

```bash
$PYTHON -m pytest -q tests/unittest
```

Result: **508 passed, 1 skipped**.

## Integration regression

```bash
$PYTHON -m pytest -q tests/integration \
  --deselect tests/integration/act/test_int_actions.py::TestDesatAction::test_desat_action_power_draw \
  --deselect tests/integration/scene/test_int_scenarios.py::TestCityTargets::test_city_distribution
```

Result: **57 passed, 1 skipped, 2 deselected**.

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

## Static checks

```bash
$PYTHON -m ruff check \
  src/bsk_rl/__init__.py src/bsk_rl/act/__init__.py \
  src/bsk_rl/comm/__init__.py src/bsk_rl/data/__init__.py \
  src/bsk_rl/obs/__init__.py src/bsk_rl/sats/__init__.py \
  src/bsk_rl/gym.py src/bsk_rl/sats/satellite.py \
  src/bsk_rl/obs/observations.py src/bsk_rl/utils/rllib/discounting.py \
  src/bsk_rl/sats/roles.py src/bsk_rl/comm/rso_communication.py \
  src/bsk_rl/comm/typed_messages.py src/bsk_rl/data/multiagent_rso_data.py \
  src/bsk_rl/data/multiagent_rso_reward.py examples/multiagent_imaging \
  tests/unittest/multiagent tests/integration/multiagent

git diff --check
```

Result: **passed**.

No full Monte Carlo campaign or long training run was launched.
