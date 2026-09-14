# Four-sensor Walker completion-v3 learned pilot

This is the reviewed learned-policy contract after the completed three-sensor
baseline campaign. The baseline evidence under
`cluster/evidence/three_sensor_v2_full_campaign` is immutable and is not reused as
four-sensor evidence. The learned pilot creates new, seed-matched independent and
centralized-full-state heuristic references for its own four-sensor initial conditions.

## Physical mission

- Four sensing agents use a seeded Walker Delta **4/2/1** constellation: four
  circular 700 km, 97 degree spacecraft in two planes with Walker phasing `F=1`.
  A reset randomizes only the common true-anomaly and RAAN orientation. Relative
  plane and in-plane spacing remain fixed.
- One hundred passive targets are full Basilisk spacecraft and Vizard spacecraft.
  They never appear in the PettingZoo/RLlib agent list. Every episode contains
  exactly 50 LEO, 30 MEO, and 20 GEO targets. Target-ID regime assignment and
  orbital sampling are deterministic from the worker/episode reset seed and use
  the same distributions as the completed baseline campaign.
- Episodes run for 45,000 physical seconds with complete-episode sampling and ten
  candidate target slots. The two-orbit, capture-anchored cooldown evaluates to
  11,834.835756586714 seconds within floating-point roundoff for the 700 km team.
- The reward discount half-life is 45,000 seconds. The GAE trace half-life is
  6,000 seconds. Both use actual asynchronous elapsed seconds.

## What the policy sees and selects

All four agents share one target-set attention policy and local critic. Each actor
gets 26 own/environment features, ten 17-feature candidate rows, and three
12-feature peer rows: **232 values**. It selects one of **18 actions**: charge,
downlink, desaturate, reserved/masked broadcast, continue, ten target slots, or
three receiver slots. Peer slots are the other configured sensor names in lexical
order. The exact feature and action orders are serialized in the checkpoint.

Candidate and peer padding is excluded from actor attention and actor/critic
pooling. All-empty sets have a finite zero identity, leaving own-resource actions
operational. Tests permute target rows and all three peer rows, verify action
identity, and corrupt schema fields to prove that restore rejects feature-order,
mask, action, target-count, candidate-count, peer-count, and mission-setting
mismatches.

The actor sees its own resources, own physical products, its durable local catalog,
declared target ephemerides/candidates, contact-navigation beacons, and its own
exchange/ACK history. It does not see another sensor's catalog, active task,
intention, resources, product inventory, or uncommunicated navigation state. There
is no centralized critic.

## Directed completion transfer

A transmit action names one peer slot. Basilisk `locationPointing` tracks that
peer's `SimpleNav` message. Earth LOS and the same attitude/rate requirements as
imaging must remain valid continuously. A valid link draws 25 W. Packet progress
resets after loss of lock. The minimum hold is 10 seconds, extended only when the
frozen metadata requires more airtime at 64 kbit/s including its 64-byte transport
header. The attempt deadline is 300 seconds.

At action start, the sender freezes every qualified completion-record revision it
knows that the chosen receiver has not acknowledged, excluding records originally
sourced by that receiver. A record contains exposure identity and provenance,
source sensor, target, request epoch, numeric quality, qualification, capture time,
completion time, and optional full-ground-delivery time. It never contains image
bits, physical ownership, a future action, or a sender-computed eligibility list.

Only the chosen receiver merges a successfully delivered packet at the next
simulation boundary. The receiver keeps distinct first-receipt times for capture
and delivery revisions. Merge is monotone and independent of packet order: an old
or relayed acquisition cannot erase a delivery or replace a newer target summary.
The receiver then recomputes eligibility from its local catalog. Thus a peer capture
that remains physically onboard elsewhere suppresses immediate duplicate work, but
the target becomes useful again after the capture-anchored cooldown. First catalog
coverage, cooldown-qualified repeat services, and physical ground delivery remain
separate metrics.

## Bounded cluster stages

The one-worker gate first audits the source, Python packages, loaded Basilisk native
modules, allocation, and five support-data hashes. It profiles one complete mission,
then runs conflict and continuous retasking serially. Each mode performs one PPO
update, saves the policy and full PPO/Adam/worker seed state, restores in a fresh
process, proves the stored logits/actions exactly on a frozen observation batch,
and performs one resumed update. It requires finite losses, finite nonzero gradients,
nonzero parameter changes, full physical action coverage, acquisitions, complete
ground deliveries, an accepted directed packet, and a useful post-cooldown revisit.

Only a passing one-worker gate permits four workers. The worker topology changes,
so that stage starts fresh rather than claiming exact one-to-four-worker optimizer
continuation. It runs eight updates per mode; together with the two gate updates,
the authorized work remains ten updates per mode. This is a learning-mechanics
pilot, not convergence evidence.

Final restored policies run on held-out mixed seeds 10000–10004. Each is paired by
an exact initial-condition hash with an independent local greedy reference and a
centralized-full-state joint greedy reference. Both references use zero radio and
no checkpoint. The centralized reference has maximum instantaneous team information
but no future-trajectory optimizer, so it is not a guaranteed mathematical upper
bound. Results report policy-minus-reference differences with 95% paired t intervals;
five seeds are too few for a broad performance claim.

The exact Alpine commands and job history are in `cluster/README.md` and
`cluster/EXECUTION.md`.

## Completed cluster result

Jobs `32528119` and `32530057` completed the one-worker and four-worker stages.
Every physics, schema, optimization, restore, resume, and evidence-integrity gate
passed. During sampling the policy selected all three peer slots and completed
directed transfers, so receiver selection and completion transport were exercised
in the exact mission configuration.

The final deterministic policy did not pass the operational performance gate.
For both retasking modes, every held-out action was empty downlink: a mean 17,951
downlink selections and zero charge, image, continue, desaturate, or transmit
selections. All five mixed held-out seeds consequently had 0% qualified-capture
and ground-confirmed coverage, no useful revisit, no final-policy packet, and two
depleted sensors. The independent and centralized references on the same state
hashes established that the scenarios were serviceable. This is therefore a
policy-collapse result after a deliberately small ten-update budget; it does not
invalidate the directed transport implementation and it does not demonstrate that
the architecture cannot learn with a corrected training setup.

Do not expand this checkpoint to more training seeds. First add a resource and
empty-operation action shield, evaluate every checkpoint before choosing the
final one, and use a curriculum or behavior-cloned heuristic warm start so early
updates contain useful imaging behavior. The full result is in
[`cluster/evidence/walker4_completion_v3_cluster_pilot`](cluster/evidence/walker4_completion_v3_cluster_pilot/REPORT.md).
