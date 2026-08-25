# Multi-agent space imaging 2026

This experiment extends the pushed AMOS 2026 environment at commit
`0a05f2bd72872dc8272da673b550b3f1c9daafab` without changing the AMOS worktree.
It is a bounded first implementation: two homogeneous sensing spacecraft, passive
propagated RSOs, parameter-sharing independent PPO, local catalog knowledge, and three
information cases. It does not claim MAPPO, a centralized critic, image relay, or an RF
communications model.

## Quick validation

From the `multi-agent-space-imaging-2026` worktree:

```bash
export PYTHONPATH=src:.
PYTHON=/Users/dahu1128/Repositories/bsk_rl/.venv/bin/python

$PYTHON -m pytest -q tests/unittest/multiagent
$PYTHON -m pytest -q tests/integration/multiagent/test_two_sensor_rollout.py
$PYTHON -m pytest -q tests/integration/multiagent/test_rllib_smoke.py
```

Run and save the bounded deterministic scenario:

```bash
$PYTHON examples/multiagent_imaging/evaluate.py \
  --config examples/multiagent_imaging/configs/smoke.json \
  --output results/multiagent_imaging/smoke_seed0.json \
  --plots-dir results/multiagent_imaging/smoke_seed0_plots
```

Run a quick three-sensor demonstration and generate vector PDFs and PNG previews:

```bash
$PYTHON examples/multiagent_imaging/run_quick_demo.py \
  --n-sensors 3 \
  --n-targets 12 \
  --n-candidates 4 \
  --duration-s 1200 \
  --seed 0 \
  --output-dir results/multiagent_imaging/quick_demo_3sensors_seed0
```

This produces one reward/resource/action figure per sensor. When more than one sensor is
present it also produces `multiagent_overview.pdf` and `.png`, containing cumulative
captures, an RSO capture raster, action-time allocation, per-sensor event counts, and the
team unique-service/conflict summary. The quick runner uses the deterministic shared
controller, not a trained PPO checkpoint.

An existing rollout JSON can also be plotted separately:

```bash
$PYTHON examples/multiagent_imaging/plot_evaluation.py \
  --input results/multiagent_imaging/quick_demo_3sensors_seed0/rollout.json \
  --output-dir results/multiagent_imaging/quick_demo_3sensors_seed0/plots
```

Run all four matched validation cases:

```bash
$PYTHON examples/multiagent_imaging/run_matched_validation.py \
  --output-dir results/multiagent_imaging/matched_validation_phase2
```

Run a single short PPO iteration:

```bash
$PYTHON examples/multiagent_imaging/train.py \
  --config examples/multiagent_imaging/configs/smoke.json \
  --iterations 1 \
  --train-batch-size 64
```

The three `full*.json` files record matched 100-target, 45,000-second configurations
for the independent, centralized-information, and perfect-metadata intent/status cases.
They are launch specifications only; this implementation task intentionally does not
start full training or Monte Carlo.

## Information cases

- `independent`: no teammate information enters a local catalog or observation.
- `centralized_information`: the policy receives an ideal read-only aggregation of the
  newest catalog metadata and intent. This is labeled an information upper bound.
- `intent_status`: sender-scoped typed messages update local knowledge. With
  `perfect_metadata_delivery=true`, metadata is delivered perfectly to validate semantics.
  With it set to `false`, a sender must execute the finite broadcast action and have
  Earth-unoccluded line of sight to the receiver.

The LOS version is only a geometry and opportunity-cost model. It intentionally omits RF
link budget, bandwidth, packet loss, propagation delay, and image relay. The
`smoke_los_broadcast.json` configuration exercises this bounded extension; the ordinary
`smoke.json` configuration exercises perfect metadata semantics first.

## Shared policy and timing

Only explicit sensing roles reach PettingZoo and RLlib. Every sensor maps to the shared
`imager` module; passive targets have no policy. `ContinuePreviousAction` emits
`NO_ACTION` for a sensor still executing an action. `CondenseMultiStepActions` removes
continuation samples, accumulates each sensor's elapsed `d_ts`, and the
`TimeDiscountedGAEPPOTorchLearner` applies the AMOS per-second discount convention.

The actor observation has 14 own-spacecraft/environment features followed by equal
13-feature candidate-target chunks. No generic teammate resource, orbit, or action vector
is appended. Coordination is represented only where it is actionable: each candidate has
known pending and cooldown fields plus a freshness-weighted fraction of known peers
targeting that same RSO. The comparison uses target IDs internally but never exposes raw
IDs or fixed peer slots to the policy. See `OBSERVATION_VALIDATION.md` for the exact
contract, boundary rules, matched-rollout results, and cluster-training proposal.

## Why the adapter remains

The training and asynchronous-policy path otherwise uses the established BSK-RL pattern:
parallel PettingZoo, one shared RLlib module, standard per-satellite data stores, a global
rewarder, and a communication-method subclass. The small role-aware tasking adapter remains
because this space-to-space formulation propagates every RSO as a Basilisk spacecraft.
Treating 100--200 passive RSOs as learned or dummy PettingZoo agents would not match the
AEOS multi-sensor pattern and would scale poorly.
