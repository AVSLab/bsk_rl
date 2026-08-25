# Architecture note

## Design alignment

The implementation follows the established BSK-RL multi-spacecraft pattern wherever the
AMOS RSO formulation permits it:

- a native parallel PettingZoo environment;
- one shared RLlib module named `imager` for every sensing spacecraft;
- standard per-spacecraft `DataStore` instances and one `GlobalReward`;
- a `CommunicationMethod` subclass for information exchange;
- `ContinuePreviousAction`, `CondenseMultiStepActions`, and the time-discounted PPO
  learner for asynchronous actions.

The role-aware `SensingAgentConstellationTasking` adapter is retained as one narrow
boundary. Unlike AEOS, the AMOS environment propagates every RSO as a Basilisk spacecraft.
Hundreds of passive targets therefore cannot be exposed as PettingZoo agents or assigned
dummy learned policies. Explicit `SpacecraftRole` values identify sensing agents and
passive targets; the adapter presents only sensing agents to PettingZoo while keeping all
spacecraft in the simulator. It accepts any positive number of sensors and never infers a
role from a list index or name substring.

## State ownership

There are three owners, but no parallel public datastore or ledger hierarchy:

| State | Owner | Purpose | Actor access |
|---|---|---|---|
| Physical data partitions | Each sensor's Basilisk storage unit | Authoritative onboard image volume | Own resources only |
| `MultiSensorRSOTargetImageStore` | Standard BSK-RL `DataStore` on each sensor | Product provenance and `LocalCatalogKnowledge` | Own and explicitly received knowledge |
| `_TeamServiceAccounting` | `MultiSensorRSOTargetImageReward` | Unique team service, duplicates, and deterministic credit | Never |

Every product retains `record_id`, `source_sensor`, `target_id`, `capture_time`,
`delivery_time`, `quality`, and `storage_owner`. Source and storage owner must match because
image relay is disabled. A sensor can remove and downlink only its own onboard product.
Team accounting is private to the global rewarder and is exposed only as read-only
evaluation history and summary metrics. It cannot modify a local catalog, candidate
eligibility, cooldown, or observation.

## Minimal shared-policy observation

The actor receives 14 own-spacecraft/environment features followed by equal 13-feature
candidate chunks:

| Block | Features |
|---|---:|
| Own storage, battery, three wheel fractions, and three Sun-heading components | 8 |
| Eclipse timing | 2 |
| Two upcoming ground-window open/close pairs | 4 |
| Each candidate target | 13 |

Thus, for `K` candidates, the flat input size is `14 + 13K`. The per-target features are
priority, relative position (3), relative velocity (3), pointing angle, range,
illumination, known cooldown, known pending state, and known teammate intent.

The earlier 24-feature pooled teammate vector and three team-count features were removed.
Peer position, velocity, battery, storage, wheel state, generic current action, action time
remaining, and action-category fractions did not have a demonstrated decision role in the
first duplicate-avoidance study. Including them enlarged the policy input and risked making
the coordination claim harder to interpret.

The retained teammate signal is target relational. Raw target IDs are compared internally,
and the candidate receives a freshness-weighted fraction of known sensing peers currently
targeting that same RSO. The policy never receives a raw identifier or fixed peer slot. The
scalar is zero for one sensor and remains the same size for any constellation size.

If a later experiment demonstrates that formation geometry, link scheduling, or peer
resource allocation is necessary, that information should be represented by a masked
teammate-set encoder or attention block. It should not be appended as `sensor_1`,
`sensor_2`, and similar fixed fields.

## Information boundaries

| Case | Candidate pending/cooldown | Same-RSO teammate intent |
|---|---|---|
| `independent` | Local events only | Always zero |
| `centralized_information` | Ideal read-only aggregation of all local catalogs | Current matching peer targets |
| perfect `intent_status` | Only compact, received, unexpired status | Fresh received matching intents |
| LOS `intent_status` | Only status received after a finite broadcast | Fresh matching intent if one was actually messaged |

The centralized view aggregates local catalog metadata, not the global reward accounting.
The intent/status message contains only sender, sequence number, target ID, action/intent,
creation and expiry times, and the newest acquisition, delivery, cooldown, and lifecycle
status for that target. It contains no resource vector, ephemeris, image product, complete
datastore, or global service data.

Receiver handling is deterministic: expired messages are rejected, repeated
`(sender, sequence_number)` values are duplicates, lower sequences are stale, and accepted
catalog updates merge by `(creation_time, sender)`. Perfect delivery validates semantics.
The LOS extension requires a finite `BroadcastIntent` action and Earth-unoccluded
sender-to-receiver geometry. RF link budget, bandwidth, packet loss, propagation delay,
multi-hop relay, and image relay remain outside this bounded phase.

## Centralized information as an upper bound

Direct centralized information is an upper bound for the metadata available to the actor
under the adopted observation contract. LOS sharing does not automatically converge to the
same bound merely because more sensors are present. The time-varying communication graph
must be sufficiently connected, status must propagate before expiry, and the necessary
fields must be transmitted. Disconnected components or stale messages can preserve local
catalog differences.

## Reward and asynchronous timing

The reward preserves the AMOS mixture

\[
r_i=(1-\alpha)r_{i,\mathrm{image}}+\alpha r_{i,\mathrm{ground}}.
\]

Global truth groups simultaneous service deterministically. A priority value is split
among simultaneous qualifying source sensors, so summed agent reward does not multiply
the team total. Capture duplicate attempts and successfully delivered duplicates are
logged separately. Optional communication or duplicate penalties remain separate from the
finite time cost of broadcasting.

Basilisk advances to the earliest enabled terminal event. A sensor still executing an
action receives `NO_ACTION` and is not retasked because a teammate finished first.
Continuation transitions are condensed before learning, and elapsed global intervals are
accumulated into that sensor's `d_ts`. A message received while a sensor is busy is visible
at its next decision epoch.

This is parameter-sharing independent PPO. The policy weights are shared, while physical
state, observations, actions, rewards, and value samples remain per sensor. No centralized
critic or MAPPO implementation is claimed.
