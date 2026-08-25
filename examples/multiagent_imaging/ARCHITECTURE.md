# Architecture note

## Roles and simulation boundary

`SpacecraftRole` is attached explicitly when a spacecraft is constructed. The role-aware
`SensingAgentConstellationTasking` adapter keeps every spacecraft in Basilisk but exposes
only `SENSING_AGENT` spacecraft through PettingZoo. `PASSIVE_TARGET` spacecraft receive a
deterministic drift action when needed. The adapter accepts any positive number of sensing
agents; no role is inferred from a list index or a substring in a spacecraft name.

The AMOS `RandomSatellites` scene now has an explicit-role path that registers every
passive RSO with every sensing spacecraft's `targetLocation` model. Each sensing spacecraft
receives its own target-named storage partitions and its own ground-station/access model.
The shared team cooldown converts the configured orbit count using the order-independent
median sensing-orbit period, while each sensor maintains its own resulting cooldown
knowledge. The legacy single-sensor path and its buffer naming are retained for existing
AMOS runs.

## Three different kinds of state

| State | Owner | Purpose | May affect a local policy? |
|---|---|---|---|
| `SensorProductStore` | One sensor | Physical onboard image products | Yes, for that owner only |
| `LocalCatalogKnowledge` | One sensor | Acquisition, delivery, pending, cooldown knowledge | Yes |
| `TeamServiceLedger` | Environment truth | Unique service, duplicate service, reward and analysis | No |

Every product retains `record_id`, `source_sensor`, `target_id`, `capture_time`,
`delivery_time`, `quality`, and `storage_owner`. Source and owner must match because image
relay is disabled. A downlink removes a record only from the executing sensor's physical
store. The team ledger deliberately has no reference to a local catalog, so it cannot
silently change observations, candidate filters, or cooldown knowledge.

## Shared-policy observation contract

The AMOS target-wise contract is preserved: all global/context features precede equal
13-feature candidate chunks. The original 17 global features are augmented by a
24-feature, permutation-invariant teammate-set summary. The result is 41 global/team
features plus 13 features per candidate, independent of the number or ordering of
sensing agents.

Each available peer contributes nine compact values: relative distance, relative radial
rate, battery fraction, storage fraction, maximum wheel-speed fraction, remaining action
duration, status age, local-catalog age, and Earth-unoccluded link availability. Mean and
maximum pooling produce 18 values. Six additional values give the fractions of known peers
currently charging, downlinking, desaturating, broadcasting, imaging, or doing another
action. Sensor count, known-peer count, and fresh-intent count remain in `TeamKnowledge`.
The teammate-selected target remains a per-target feature, preserving the target/action
association that would be lost in a pooled vector.

The pool is deliberately a minimum coordination interface. Full relative position and
velocity vectors are not included because duplicate avoidance and first-stage metadata
communication require proximity, closing geometry, availability, intent, freshness, and
resource margins rather than formation reconstruction. If later experiments require
identity-specific multi-hop routing or predictive formation coordination, the replacement
should be a masked teammate-set attention encoder, not fixed peer slots.

Information boundaries are enforced before pooling:

| Case | Teammate source | Local catalog effect |
|---|---|---|
| `independent` | Empty set; all 24 values are zero | Local events only |
| `centralized_information` | Ideal current status of every peer | Read-only global metadata; no ledger access |
| perfect `intent_status` | Received, unexpired typed messages | Only explicitly messaged target status merges |
| LOS `intent_status` | Received, unexpired typed messages after a finite broadcast | Same merge rule, subject to LOS and action time |

The target-wise actor previously sliced the global vector but did not use it. Multi-agent
training now opts into a spacecraft-context encoder that adds the encoded global/team
context to every target embedding. The flag defaults to off, preserving existing AMOS
checkpoint behavior. The critic already consumed the global vector.

## Knowledge and communication

An independent sensor updates only its own catalog. The centralized-information case reads
a separate aggregation view and never writes the result back to the local catalogs. The
intent/status case uses `IntentStatusMessage`, which contains sender, sequence number,
target, action/intent, creation and expiry times, and latest acquisition, delivery,
pending, and cooldown status. It also carries the compact sender state used by the
teammate-set pool: position, velocity, resource fractions, action time remaining, and
catalog timestamp. It never carries an image product, team ledger, or complete datastore.

Receiver inboxes make message handling deterministic:

1. expired messages are rejected;
2. an already seen `(sender, sequence_number)` is a duplicate;
3. a lower sequence number from the same sender is stale/out of order;
4. remaining updates merge by `(creation_time, sender)`.

Perfect metadata delivery is used first. The broadcast extension ports the minimal design
idea from commit `494df0f`: a finite action sets a sender-side broadcast flag, communication
directions are explicit, and only broadcasting senders may transmit. Unlike that reference,
the new implementation transmits a typed message rather than copying a complete Python
datastore. The initial non-perfect model uses Earth-unoccluded inter-sensor geometry only,
sampled when the finite broadcast action completes.

## What “centralized information upper bound” means

The direct `centralized_information` case is a valid information upper bound for the
adopted policy inputs because every sensor can read the newest metadata from all sensors at
every decision. LOS-only sharing is not automatically the same upper bound. Even with many
satellites, full synchronization occurs only when the time-varying communication graph is
connected often enough, information propagates with the assumed timing, and every relevant
field is shared. A disconnected component or stale message can preserve local divergence.
Therefore the direct ideal view and the LOS broadcast case remain distinct experiments.

## Reward and credit

The reward preserves the AMOS mixture:

\[
r_i = (1-\alpha) r_{i,\mathrm{image}} + \alpha r_{i,\mathrm{ground}}.
\]

Global truth groups simultaneous service deterministically. A fixed priority value is split
between simultaneous source sensors, so summing agent rewards does not multiply the team
total. Later service inside the team cooldown has zero unique-service value and is logged as
a successful duplicate. Capture duplicate attempts and ground-verified successful duplicates
are separate metrics. Optional duplicate and communication penalties are separate parameters;
the broadcast action already has an inherent finite time cost.

## Asynchronous decision timing

Basilisk advances to the earliest enabled terminal event. A sensor still executing an action
receives `NO_ACTION`; it is not retasked merely because a teammate completed an action.
Continuation transitions are condensed before learning, rewards are associated with the
originating action, and the elapsed global intervals are summed into the corresponding
sensor's `d_ts`. A typed message delivered while a receiver is busy is present in that
receiver's local catalog at its next observation and decision epoch.

This is parameter-sharing independent PPO. The policy weights are shared, but physical
state, observations, action histories, and value samples are per agent. There is no
centralized critic and the implementation should not be described as MAPPO.
