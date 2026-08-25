# Observation validation and recommendation

## Recommendation

The minimum defensible first coordination observation is the existing own-spacecraft and
target-wise AMOS input plus one relational value on each candidate: the
freshness-weighted fraction of known peers targeting that same RSO. Known pending and
cooldown fields carry locally generated or explicitly shared catalog status.

Do not include generic peer action, peer position/velocity, battery, storage, wheel margin,
or a pooled team vector in the first experiment. Their necessity has not been established,
and the primary coordination problem is whether another sensor is already pursuing or has
recently serviced the same candidate. The compact contract supports any sensor count
without fixed peer slots. A later demonstrated need for richer peer state should be met by
a masked teammate-set attention encoder, not a fixed two-agent vector.

## Before and after

### Own-spacecraft, environment, and teammate features

| Group | Previous implementation | Revised implementation |
|---|---:|---:|
| Own spacecraft resources and attitude | 8 | 8 |
| Eclipse timing | 2 | 2 |
| Two ground-opportunity open/close pairs | 4 | 4 |
| Sensor/known-peer/fresh-intent counts | 3 | 0 |
| Pooled peer geometry, resources, freshness, and link status | 18 | 0 |
| Pooled peer action distribution | 6 | 0 |
| **Global/context total** | **41** | **14** |

### Per-candidate target features

| Feature | Count | Revised meaning |
|---|---:|---|
| Priority | 1 | Candidate value |
| Relative position | 3 | Target geometry relative to this sensor |
| Relative velocity | 3 | Target motion relative to this sensor |
| Pointing angle | 1 | This sensor's pointing requirement |
| Range | 1 | Sensor-to-target distance |
| Illumination | 1 | Target illumination |
| Known cooldown | 1 | Local or explicitly shared knowledge only |
| Known pending state | 1 | Local or explicitly shared knowledge only |
| Known teammate intent | 1 | Freshness-weighted fraction of known peers targeting this RSO |
| **Per-target total** | **13** | |

For `K` candidates, the revised actor input is `14 + 13K`. Candidate ordering remains
equivariant in the target-wise attention network. Teammate ordering is invariant because
only counts of matching, valid intents enter a candidate. Intent/status cases normalize by
received, unexpired senders rather than by a hidden global sensor count. Raw target
identity and team accounting never enter the actor.

## Information-case separation

| Case | Peer dynamics/resources | Peer target/status | Global accounting |
|---|---|---|---|
| `independent` | Never exposed | Never exposed | Never exposed |
| `centralized_information` | Not exposed | Ideal read-only catalog metadata and active matching target | Never exposed |
| perfect `intent_status` | Not exposed | Received, unexpired compact message only | Never exposed |
| LOS `intent_status` | Not exposed | Received compact message after finite LOS broadcast only | Never exposed |

The centralized case is explicitly an ideal metadata upper bound. It is not a centralized
critic. Intent/status messages do not copy a Python datastore and do not carry imagery.

## Bounded paired rollout

The four validation configurations use seed 41, two distinct sensing orbits, identical
six-target states and priorities, three candidates, `alpha=0.1`, a two-orbit cooldown, and
a 1,800-second horizon. The validation runner checks that every paired configuration field
other than information mode and every initial sensor and target state are identical.

The deterministic controller is a shared observation-to-action rule, not a trained policy.
These values validate mechanics and diagnostics, not scientific performance.

| Case | Unique acquisitions | Acquired value | Same-target conflict time | Broadcast time per sensor |
|---|---:|---:|---:|---:|
| Independent | 2 | 49.72 | 1,045 s | 0 s |
| Centralized information | 4 | 76.58 | 193 s | 0 s |
| Perfect intent/status | 3 | 63.46 | 101 s | 0 s |
| LOS intent/status | 2 | 49.72 | 763 s | 90 s |

No ground delivery completed during this short horizon, so unique delivered service and
delivered team value are zero in all four cases. Unit tests separately cover downlink
ownership and deterministic non-double-counting.

## Proposed cluster-training matrix (not submitted)

All cells use one shared target-wise policy, identical catalog distributions, reward,
discounting, and paired evaluation seeds.

| Stage | Sensors | Information case | Training seeds | Purpose |
|---|---:|---|---:|---|
| Reference | 1 | Independent | 3 | Single-sensor scale and regression reference |
| Primary | 2 | Independent | 3 | Parameter-sharing independent-PPO baseline |
| Primary | 2 | Centralized information | 3 | Ideal metadata upper bound |
| Primary | 2 | Perfect intent/status | 3 | Isolate compact-message semantics |
| Primary | 2 | LOS intent/status | 3 | Add finite communication opportunity cost |
| Transfer | 3 and 4 | Frozen two-sensor policies | Evaluation first | Test variable-count transfer before retraining |

The next bounded phase should first run one module instance across one, two, and three
sensors locally and validate the checkpoint observation contract. Cluster training should
wait until those shared-policy evaluations pass.
