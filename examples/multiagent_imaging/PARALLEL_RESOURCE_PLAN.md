# Four spacecraft study: parallel resources and launch boundary

## Why the two resource requests differ

The immutable v4 readiness launcher uses one remote simulation runner and one
local CPU learner. Its8 CPUs / 32 GiB / 4 h request is a bounded correctness diagnostic.
Four learned spacecraft live inside that one simulation; they are not four
separate CPU workers. More CPUs without more simulation runners do not increase
sampling parallelism.

The preserved AMOS mixed-fixed100-target reward-sweep Slurm script requests
32 CPUs / 100 GiB / 48 h. Its trainer defaults to available CPUs minus four runners:
28 under a32 CPU allocation, with minimum batch4,200 and30 PPO epochs. The
selected saved W&B optimizer config confirms4,200/30. The actual scheduler
allocation and peak aggregate RAM of that selected historical run are not yet
verified here; source defaults must not be mislabeled as its measured utilization.

AMOS also declares11 Torch threads. This study retains the reviewed one-thread
execution setting while scaling simulation workers. Optimizer/gamma/lambda remain
the explicitly approved `amos_best` profile. Resources are declared separately.

## Prepared long-run resources

| Item | Per policy | Two-policy pair |
|---|---:|---:|
| Requested/effective CPU ceiling |32|64 across both jobs|
| RAM |100 GiB|200GiB if concurrent|
| Wall limit including evaluation/restore |24h|Two independent24h jobs|
| Allocated CPU-hour reservation |768 |1536 |
| Remote mission runners |28|28 in each arm|
| Frozen evaluation processes |8 after training|Eight held-out missions per arm|
| Learner |Local CPU;one Torch thread|One shared policy per arm|

Allocation gates require exactly32 CPUs/100 GiB/24h and no automatic requeue.
`acpu/cpu-normal` reports a24-hour maximum wall limit. Its memory-per-CPU ceiling
is3840MiB:100 GiB fits32 CPUs, whereas128GiB could require more than32 effective
CPUs and violate this budget. Do not increase RAM silently.

100 GiB is **RAM**, not a disk reservation. Scratch stores checkpoints, raw
trajectories, logs and figures. The retained100-target checkpoint0003 occupies
about7.2MiB, but neither200-target trajectory growth nor aggregate28 runner peak
RAM has been measured. Check scratch quota/free space and projected artifact
growth during readiness; report batch MaxRSS separately from aggregate job RAM.

## Necessary new-source parallel readiness

`study_parallel_readiness.slurm` is prepared, **not approved or submitted**:

- 32 CPUs / 100 GiB / 2 h: hard 64 allocated CPU-hours; no requeue.
- 28 remote runners/local CPU learner;one PPO update per arm, maximum two total.
- At most32 sampled training missions/60000 true decisions per arm.
- 16 frozen full missions total:initial/final×401/402×argmax/stochastic×two arms.
- Eight independent spawned evaluation processes;private seeds and raw outputs.
- Same source/runtime/config/weights/initial-state/horizon/metric/gradient/W&B/
 full-topology restore/checkpoint-byte gates as the serial readiness.

This expands the original six-training-mission cap, so it needs explicit review.
Do not substitute it for the unchanged v4 job33510037 or submit both concurrently
under a presumed36 CPUh total budget. If the task-wide readiness cap remains36,
the new 64 CPUh reservation exceeds that cap by itself. Request an additional
explicit 64 CPUh readiness budget. Worst-case reservation for both readiness
stages is 100 CPUh; including the first 24h pair it is 1,636 CPUh. Actual final
charges are reconciled with sacct.

The initial one-hour local draft is superseded. The first two completed v4
200-target missions took approximately 554 and 709 seconds after startup.
Two arms with a training group and a parallel evaluation group each already
use about 47 minutes at the slower observed mission rate, before preflight,
28-runner initialization, PPO optimization, exact restore and shutdown. A
one-hour limit leaves insufficient headroom for unmeasured parallel contention.
The two-hour request remains a hard cap, not a claim of measured 28-runner speed.

## Common experience budget and continuation

The long scripts require passing **same source 28 runner** readiness. One-runner
v4 timing cannot satisfy that gate. Each arm uses seed 3105 and fresh equal
weights, separate W&B IDs, and the same set of worker seed streams. Both learn
from every real spacecraft decision; forced continuation/padding/bootstrap rows
remain excluded.

From measured update throughput, use65% of the slower arm's rate times the common
safe training cutoff, rounded down to50,000 decisions. Reserve two observed batches
above that soft target for a hard cap. Record all resulting mission/update caps
and per-worker quotas before sampling. Eight evaluation missions, startup,
restore, checkpoint shutdown and update overrun are inside the24-hour limit.
This estimate may be conservative or wrong as behavior changes; stop truthfully
on any resource/budget/gate failure. Do not claim equal wall time means equal
experience. Retain actual milestone overshoot and individual evaluation seeds.

28 complete missions can produce a batch far above the4,200 minimum. Report true
learner rows, actual batch size and repeated30-epoch presentations separately;
worker count is not an innocuous way to claim optimizer equivalence.

Another24-hour tranche needs its exact checkpoint, source, topology, exposure,
critic/action/storage trajectories, both selection modes, actual cost and renewed
budget reviewed. It resumes the same W&B ID with `resume=must` and exact pre/post
zero-update restore checks. It is never an automatic successor.

## Plain labels and constellation notation

Both policies are learned decentralized parameter-sharing PPO. **With
communication** receives physically delivered inventory metadata; **Without
communication** has no transmit action and observes only the declared LOS public
position/velocity beacons. Both use their own private mission information.

Walker4/2/1 means four total spacecraft, two orbital planes, phasing index one:
two spacecraft per plane,180° in-plane spacing,90° relative inter-plane phase
under the declared circular Walker convention. Both fly700 km / 97°. Plot
spacecraft1..4 correspond to internal sensor_0..3. Plasma/solid and
viridis/dashed distinguish arms;consistent marker shapes distinguish spacecraft.

## Current boundary

Only immutable v4 was transferred/deployed and its unchanged serial readiness
submitted. This revised source is local preparation. Its parallel evaluator has
offline failure/matching/byte-reconciliation tests, not a new physical rollout
claim. Gate33505950 and all100-target/two-orbit evidence remain historical and
separate. Fixed-batch fitting is not policy convergence or cooperation evidence.
