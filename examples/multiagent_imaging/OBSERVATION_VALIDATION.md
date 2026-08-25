# Teammate-observation validation and recommendation

## Recommendation

Use the implemented 24-feature permutation-invariant teammate summary as the minimum
defensible first coordination observation. It supplies the state needed to avoid obvious
target conflicts, judge peer resource pressure, estimate whether a finite broadcast is
currently useful, and distinguish fresh from stale information. It does not expose full
peer ephemerides, identity-indexed slots, local datastores, or the team ledger.

The recommendation is bounded. Mean/max pooling cannot retain every correlation between a
specific peer's resources, action, and geometry. The per-target teammate-intent flag retains
the most important target/action association for this study. A later study needing
identity-specific routing, multi-hop coordination, or formation prediction should replace
the pool with a masked teammate-set attention encoder. It should not append `sensor_1`,
`sensor_2`, and similar fixed slots.

## Before and after

### Global, spacecraft, environment, and team features

| Group | Before | After | Details |
|---|---:|---:|---|
| Spacecraft resources and attitude | 8 | 8 | Storage, battery, three wheel fractions, three Sun-heading components |
| Eclipse timing | 2 | 2 | Next eclipse start and end |
| Ground opportunities | 4 | 4 | Open/close timing for two upcoming windows |
| Team counts | 3 | 3 | Sensor count, known-peer count, fresh-intent count |
| Pooled teammate geometry/status/resources | 0 | 18 | Mean and max of nine peer features |
| Teammate action distribution | 0 | 6 | Fractions in six action categories |
| **Global/team total** | **17** | **41** | Fixed size for one or many sensors |

The nine pooled peer features are relative distance, radial rate, battery, storage, maximum
wheel usage, action time remaining, message/status age, sender-catalog age, and current
Earth-unoccluded link availability. Full relative vectors were considered but are not
needed for the first duplicate-avoidance and metadata-sharing study.

### Per-candidate target features

| Feature | Before | After |
|---|---:|---:|
| Priority | 1 | 1 |
| Relative position | 3 | 3 |
| Relative velocity | 3 | 3 |
| Pointing angle | 1 | 1 |
| Range | 1 | 1 |
| Illumination | 1 | 1 |
| Known cooldown | 1 | 1 |
| Known pending state | 1 | 1 |
| Known teammate intent | 1 | 1 |
| **Per-target total** | **13** | **13** |

For `K` candidates, the flat actor input is therefore `41 + 13K`. Candidate ordering is
equivariant in the target-wise attention network. Teammate ordering is invariant in the
pool. Sensor count remains explicit so one shared policy can distinguish constellation
size without learning agent-name conventions.

## Information-boundary matrix

| Case | Peer dynamics/resources | Peer intent/target status | Team ledger |
|---|---|---|---|
| `independent` | Never exposed; teammate pool is zero | Never exposed | Never exposed |
| `centralized_information` | Ideal current compact status | Ideal read-only metadata and active intent | Never exposed |
| perfect `intent_status` | Only received, unexpired typed fields | Only received target messages | Never exposed |
| LOS `intent_status` | Only received, unexpired typed fields | Only after finite LOS broadcast | Never exposed |

Link availability in an intent/status observation is derived from the receiver's current
position and the sender position in its latest valid message. It is not a hidden read of
the current global communication graph. The centralized case uses true current peer state
and remains explicitly labeled as an upper bound.

## Bounded paired rollout

The four validation configurations use seed 41, two distinct sensing orbits, six identical
target initial states and priorities, three candidates, `alpha=0.1`, a two-orbit cooldown,
and a 1,800-second horizon. The runner verifies equality of every paired field and the
complete initial-condition record before accepting the comparison.

The deterministic controller is a shared observation-to-action rule, not a trained policy,
so these values validate mechanics rather than establish scientific performance.

| Case | Unique acquisitions | Acquired value | Same-target conflict time | Broadcast time per sensor |
|---|---:|---:|---:|---:|
| Independent | 2 | 49.72 | 1,045 s | 0 s |
| Centralized information | 4 | 76.58 | 193 s | 0 s |
| Perfect intent/status | 3 | 63.46 | 101 s | 0 s |
| LOS intent/status | 2 | 49.72 | 763 s | 90 s |

No ground delivery completed during the bounded horizon, so unique delivered service and
delivered team value are zero in all four cases. Delivery ownership and deterministic
non-double-counting remain covered by unit tests. No performance conclusion should be
drawn from one deterministic rollout.

## Proposed cluster-training matrix (not submitted)

All cells use one shared target-wise policy, the same reward definition, identical train
catalog distribution, per-second discounting, and paired evaluation seeds.

| Stage | Sensors | Information case | Training seeds | Purpose |
|---|---:|---|---:|---|
| Reference | 1 | Independent | 3 | Single-sensor scale and regression reference |
| Primary | 2 | Independent | 3 | Parameter-sharing independent-PPO baseline |
| Primary | 2 | Centralized information | 3 | Ideal information upper bound |
| Primary | 2 | Perfect intent/status | 3 | Isolate message semantics from link opportunity |
| Primary | 2 | LOS intent/status | 3 | Finite communication opportunity cost |
| Transfer | 3 and 4 | Each frozen two-sensor policy | Evaluation only first | Test variable-count zero-shot behavior before retraining |

Recommended sequence: run three training seeds for the four two-sensor cases, select
checkpoints using the same validation rule, then evaluate every checkpoint on at least 100
paired seeds. Report unique acquisitions, unique deliveries, ground value, conflict and
duplicate rates, resource margins, message age, and broadcast time. Only after frozen
two-sensor evaluation should three- or four-sensor training be authorized. No cluster jobs
were submitted in this validation phase.
