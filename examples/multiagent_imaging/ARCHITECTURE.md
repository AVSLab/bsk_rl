# Option 2: persistent completion sharing

Implemented on `multi-agent-space-imaging-2026`, based on the existing AMOS-derived
branch. No rebase onto an RSO inspection or tip-and-cue branch is required. The comparison
and upstream commit audit are in [RETASKING_COMPLETION_REVIEW.md](RETASKING_COMPLETION_REVIEW.md).
That review describes the pre-implementation code; this document describes the new code.

## Reused interfaces and new responsibilities

| Component | Responsibility |
|---|---|
| `CompletionConstellationTasking` in `src/bsk_rl/completion_gym.py` | Decision scheduling, physical task lifetime, interruption, receiver-local conflict detection, waste metrics |
| Existing `SensingAgentConstellationTasking` | Parallel PettingZoo boundary, sensor-only agents, deterministic passive spacecraft propagation |
| `CompletionImageStore` / `CompletionImageReward` | Sensor-owned image products, hold-gated capture metadata, complete-product ground verification, existing AMOS reward mixture |
| `CompletionCatalog` | Durable exposure facts and derived local target summaries |
| `CompletionCommunication` | Completed-record snapshots, finite radio operation, delayed/lost/expired packets, acknowledgments and later retries |
| `CompletionContext` / `CompletionTargets` / `CompletionPeers` | Versioned composable BSK-RL observations and one shared candidate snapshot |
| `ImageCompletion` | Decode that snapshot and reuse existing `ImageRSO` FSW and pointing-hold gate |
| `BroadcastCompletions` / `TransmitCompletions` / `ContinueTask` | Finite metadata operation / explicit policy choice to preserve physical progress |
| Existing target-wise `GNNModule` | Shared PPO actor/critic, with optional completion-contract action masking |
| Existing RLlib connectors and time-discounted learner | Forced continuation, action condensation, and physical-time discounting with the corrections below |

Legacy `IntentStatusCommunication`, `BroadcastIntent`, and their observations remain
available to legacy callers. The multi-agent imaging example now imports only completion
components. Reusing the BSK-RL event/FSW infrastructure avoids maintaining a second
simulator. Conflict detection and durable completion metadata are application semantics
implemented in this branch; they are not provided by Basilisk dynamics alone.

## Ownership and what is shared

All spacecraft propagate in Basilisk. A sensor owns its physical storage partitions and
`ImageProductRecord` metadata. `CompletionCatalog` is attached to its ordinary BSK-RL
`DataStore`. Private `_TeamServiceAccounting` computes unique team rewards and evaluation
metrics; the actor and retasking rule never query it.

A shared `CompletionRecord` contains:

- `record_id`, original `source_sensor`, and `target_id`;
- `request_epoch`: the generation of the mission request serviced by this exposure;
- `capture_time`: first recorded image acquisition time;
- `completion_time`: when the pointing-hold/exposure result became available;
- `qualified`: whether the completed exposure passed the configured quality threshold;
- optional `delivery_time`: when the entire owned image product reached ground.

The channel sends **qualified completed exposures only**. Failed exposures can occupy
local storage, but do not announce completion to peers. Payloads contain no intentions,
reservations, planned targets, peer positions/velocities/resources, or image bits. Received
metadata cannot create or remove another sensor's physical products. Completed metadata
may be forwarded in a later transmission, retaining its original source and timestamps.

## Catalog merge and freshness

Do not replace a whole target entry using the timestamp of the latest packet. Merge
facts by exposure identity; retain delivery as a monotone upgrade of that exposure.
Duplicate packets are idempotent. An older capture message cannot undo its delivery.
Conflicting immutable provenance is rejected. Different exposures remain independent,
even when received out of sequence or via different relays. There is no sender-wide
sequence rejection that discards an older packet containing a useful independent fact.

For example, receive capture B at time 40, then a delayed ground-delivery record for
capture A at time 10. The newest acquisition remains 40. The freshest *delivered capture*
is 10 until B is delivered. A packet's creation or reception time never makes old imagery
fresh. The compatibility summary also retains the latest ground delivery event time;
that is distinct from age of information.

`received_at[record_id]` retains the first local receipt, including across a later delivery
upgrade. `version_received_at[record_id][version]` separately retains when the capture
fact and ground-delivery upgrade were first received. It supports the prospectus's distinction between remote completion and local
learning about it. Packet TTL governs undelivered transport only. It does not erase a
completion already learned. The bounded implementation retains exposure history for the
episode; future long-duration runs may compact dominated records after preserving their
metric/audit history.

Eligibility is derived from the receiver's own facts. A qualified capture satisfies its
matching request until `capture_time + cooldown_s`. The cooldown is the configured orbit
multiple times the median initial sensor orbital period. Old records remain after that
interval, but no longer suppress a revisit. Mission code can open a new request by
advancing `target.request_epoch_s` monotonically at an environment boundary. The example
has no scheduled request-generation changes. Existing onboard products still reserve
that sensor's target partition until downlinked, even if a new request arrives.

## Physical capture and ground delivery

The Basilisk instrument may write bits before its requested pointing hold ends. The new
store tracks this unresolved volume across teammate events. Only staged hold metadata
creates a qualifying completed exposure. An interrupted exposure retains its raw storage
as an unqualified product; it earns no acquisition credit and is not shared as completed.
Unresolved partitions cannot be selected as new imaging candidates.

Downlink consumes actual decreases in named storage partitions, oldest product first.
A partial decrease reduces the remaining product volume; the first log boundary detecting complete drainage sets
`delivery_time`, removes that owned product, and permits ground credit. Source and storage
owner remain unchanged. Ground delivery is a separate fact from onboard completion.
The experiment assumes onboard quality is known at exposure completion; adding uncertain
onboard quality or ground rejection would require an explicit revision/retraction model.

## Event order and two retasking modes

At each environment boundary:

1. Decode policy actions from the exact observation candidate snapshot. Continuing the
   current target, selecting the same unfinished operational action, or choosing
   `ContinueTask` preserves FSW progress, hold state, and the original deadline.
2. Reuse BSK-RL's apply → simulate → local data logs → reward → communicate sequence.
   Propagation ends at the earliest enabled terminal event, pending radio reception, or
   configured `max_step_duration_s` heartbeat. Packet delays are rounded up to simulation
   ticks; reception is never earlier than the requested delay.
3. Merge ready packets into receiver-local catalogs and process local action endings.
4. Interrupt an unfinished image if the local catalog now knows its request is satisfied,
   or if the mission opened a newer request epoch. Clear its events, not its stored data.
5. Set the decision set and invalidate observation caches before building the next snapshot.

**Conflict:** retask on own action completion/timeout, or a locally known completion of
the active imaging request/new request epoch. An unrelated peer event or completion
leaves the agent busy; RLlib sends `NO_ACTION`. Failure removes the sensor from active
agents. An idle/initial sensor must receive an action.

**Continuous:** every surviving sensor reevaluates its policy at every environment
boundary. Retasking means reevaluating, not necessarily interrupting. The policy can retain
an unfinished task without restarting its slew, pointing hold, downlink, or broadcast.
Natural completion or conflict masks `ContinueTask`. Charge/downlink/desaturation and
broadcast each have their own finite ending conditions.

A global simulator heartbeat is part of the decision schedule, not a physical action
completion. Keep it fixed across matched experiment cells. Directed transmission samples LOS and pointing every FSW tick in both modes. The
broadcast baseline samples LOS at environment boundaries; its sampling can differ when
policies produce different event schedules.

## Communication cases and limits

| Case | Delivery | Policy radio operation required? |
|---|---|---|
| `independent` | No inter-sensor metadata | No useful exchange |
| `ideal_completion` | Automatic all-to-all completion deltas at boundaries | No; information reference |
| `completion` | Snapshot transmitted for `broadcast_duration_s`, then configurable delay/loss/TTL | Yes |

The finite channel freezes completed facts at radio start. New captures cannot be added
mid-transmission. Interrupting a broadcast sends nothing. With `link_mode="los"`, receivers
must have Earth-unoccluded geometry at the start and at every intervening environment
boundary including the end. `link_mode="ideal"` removes geometry while retaining finite
duration and packet impairment, useful for isolating timing behavior.

Successful receipt acts as an ideal small ACK; only then are delivered record versions
removed from the sender-to-receiver backlog. Loss, expiry, or unavailable receivers leave
them eligible for a later broadcast. The original source is never sent its own fact. Receipt also records that the sender
already knew that version, suppressing unnecessary echoes without inspecting peer catalogs.
No autonomous retransmission occurs without another
broadcast action. Delivery of a record's ground update creates a new version to send.

### Directed transmission and broadcast baseline

Set `communication_mode="directed"` to append receiver-selective actions. A selected peer
must appear in the exact observed peer snapshot. `TransmitCompletions` freezes that
sender/receiver delta and calls `ImagingSCFSWModel.action_point_peer(peer)`. That action
connects `peer.dynamics.simpleNavObject.transOutMsg` to `locPoint.scTargetInMsg`, reusing
Basilisk locationPointing and the existing attitude controller. The instrument and ground
transmitter device remain off. No target-location imaging access index is used for peers.

At each FSW tick, a Basilisk event checks Earth LOS and the same imaging attitude/rate
requirements (MRP norm ≤ 0.0025; body rate error ≤ 0.01 rad/s in the full configuration).
A fresh guidance tick is required after action reset. Valid pointing enables the existing
25 W transmitter power sink; this charges the battery without downlinking image bits.
The receive antenna is omnidirectional. The sender alone slews and points.

A **continuous** valid hold defaults to 10 seconds. With `metadata_bitrate_bps` configured,
required duration is `max(10, 8*payload_bytes/bitrate)`. Wire size is canonical compact UTF-8
JSON of frozen records plus a 64-byte header. The mission configuration uses 64 kbit/s and
a 300-second total attempt deadline including slew. A loss of LOS/pointing resets packet
progress and disables radio power; reacquisition can occur before the deadline. Timeout
or policy interruption discards the whole partial packet. Very large deltas can exceed
the deadline; there is no packet fragmentation. A successful hold creates one packet to
one receiver. Catalog mutation happens in `CompletionCommunication.communicate()` after
simulation/reward, never inside the pointing controller. Delay/loss/TTL then apply.

`communication_mode="broadcast"` retains the original drift-based, omnidirectional
baseline and `broadcast_duration_s=30`. That baseline has no modeled radio electrical
load. Directed mode models sender power, pointing and payload airtime, but neither mode
models RF link budget, interference, receiver contention, ACK power or range-derived
light time. `link_mode="ideal"` removes only the Earth-occultation constraint in finite
mode. The ideal-completion case remains an automatic information reference.

The public contact-navigation beacon and ideal ACK assumptions are specified in
[OBSERVATION_VALIDATION.md](OBSERVATION_VALIDATION.md). The completion payload remains
completion-only; neither private peer resources nor intent is exposed.

## Reward, learner timing, and metrics

Acquisition and ground reward retain the AMOS mixture
`(1-alpha) * image_credit + alpha * ground_credit`. The private ledger splits a fixed
priority value across simultaneous qualifying products, and keys service by target and
request epoch. Duplicate acquisition attempts and successful duplicate ground products
remain separate diagnostics. Communication time cost is additional and explicit.

`CondenseMultiStepActions(gamma=discount_per_s)` moves each primitive endpoint reward to
the start of its selected action using its actual elapsed seconds. Forced `NO_ACTION`
steps disappear; deliberate policy choices to continue remain policy samples. A constant
communication rate is integrated as `c * (1-gamma**dt)/(-log(gamma))` (or `c*dt` at gamma=1),
so changing the number of internal boundaries does not change the physical-time cost.
The learner's now-connected `reward_time="step_start"` avoids discounting these rewards
a second time. Omit connector gamma for legacy undiscounted aggregation.

The new default uses a **45,000-second reward half-life**:
`gamma_s = exp(log(0.5)/45000) = 0.99998459684795`. Rewards at one 6000-second orbit retain
about 0.912 weight and those at the episode horizon retain 0.5. The former `gamma_s=0.999`
has a 693-second half-life and gives horizon rewards approximately 2.8e-20 weight. It is
inappropriate for the stated orbit-scale completion/downlink/revisit objective. The
chosen half-life is an explicit modeling choice, not a claim of an optimized value.

GAE traces also decay in physical time: `lambda_s = exp(log(0.5)/6000)`, with both lambda
and gamma raised to each actual transition duration. This gives one nominal orbit of
trace half-life without tying it to policy-call frequency. The generic learner retains
legacy per-transition lambda unless `lambda_time="second"` is requested. Continuous
retasking still changes available decisions and can change learned behavior.

The missing-agent early return in `ContinuePreviousAction` is corrected so processing
continues for other busy agents. Busy agents receive their real final observation at a
time-limit truncation, allowing a meaningful bootstrap instead of an all-zero one.
Reward preprocessing is idempotent because PPO calls the connector for value-target
construction and again for the optimization batch. The learner explicitly marks its
added bootstrap step, including when complete-episode sampling overshoots the nominal
batch size. Training uses complete episodes so forced continuations are never separated
from their selected action/log probability by a rollout fragment. This can increase the
actual batch size above its requested minimum.
The example's PettingZoo wrapper samples valid masked actions for RLlib's API checker.
When RLlib omits an environment reset seed, a deterministic episode sequence is used;
explicit reset seeds do not advance it. RLlib derives each worker's stream from a
NumPy SeedSequence keyed by experiment seed, worker index and vector index. Complete-
episode checkpoints preserve the next episode index and worker Torch sampling state.
PPO optimizer moments and counters are stored in the full algorithm checkpoint; policy
exports are independently loaded and checked against saved logits. See
[CLUSTER_READINESS.md](CLUSTER_READINESS.md) for measured validation and limits.
These are selective fixes; the branch has not been rebased onto upstream develop.

`coordination_metrics()` uses private event history solely for evaluation. For each image
attempt, duplicate waste is the overlap after another qualifying matching completion and
before local receipt or the attempt's end, whichever occurs first. Nonduplicate policy
interruptions contribute their elapsed attempt duration. These categories are disjoint.
The sum divided by `N_sensors * elapsed_episode_time` is `wasted_time_fraction`. This first
metric counts task-level waste, including slew/hold time; it does not classify charge or
radio as imaging waste. Failures and horizon-ended attempts remain visible in history.

Evaluator JSON includes per-sensor rewards, physical operation times, policy decision
counts, packet/transmission history, durable completion records, unique/duplicate counts,
resource histories, and the waste metric. `concurrent_target_conflicts` is a separate
same-target concurrency diagnostic; concurrent work before either completion is not yet
duplicate waste. Operational timings come from task lifetimes, not policy-call counts.
