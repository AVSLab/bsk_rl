# Next task prompt — bounded three-sensor directed-completion learning pilot

Continue from the current reviewed and pushed HEAD of
`multi-agent-space-imaging-2026`. Read
`examples/multiagent_imaging/cluster/evidence/three_sensor_v2_full_campaign/REPORT.md`,
`examples/multiagent_imaging/BASELINE_MONTE_CARLO.md`,
`examples/multiagent_imaging/CLUSTER_READINESS.md`, and
`examples/multiagent_imaging/cluster/EXECUTION.md` first. Treat the completed
200-episode three-sensor independent/centralized campaign as immutable baseline
evidence; do not rerun or overwrite it.

Use the existing separate cluster checkout at
`/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026`, Python at
`/projects/dahu1128/.venv-completion-v2/bin/python`, and Basilisk runtime at
`/projects/dahu1128/basilisk-completion-v2`. Preserve the AMOS checkout and its
environment unchanged. Audit the runtime and support data again before simulation.

Prepare and execute a bounded learned-policy pilot with **three sensing agents**,
**100 passive Basilisk/Vizard RSO spacecraft**, **ten candidate targets**, and
**45,000-second complete episodes**. Passive RSOs must remain outside the
PettingZoo/RLlib agent list. Use the shared target-set attention policy as the
primary architecture. Keep the 45,000-second reward half-life, 6,000-second GAE
trace half-life, one Torch/BLAS thread per process, CPU learning, and 1,800-second
rollout timeouts.

Before training, add a versioned learned-environment target-population option that
uses exactly the baseline samplers: an all-LEO case and a mixed case with exactly
50 LEO, 30 MEO, and 20 GEO targets. Preserve deterministic regime assignment and
orbital sampling by worker/episode seed. Train the pilot on the mixed population
and evaluate the restored policies on matched held-out all-LEO and mixed seeds.
Document this choice: mixed training tests one policy across the mission's orbital
diversity, while held-out evaluation reveals whether it transfers to LEO-only.

Three agents create two selectable peer slots per sender. Version the changed
observation/action/checkpoint contract and reject checkpoints with any schema,
peer-count, target-count, candidate-count, feature-order, mask, or action-identity
mismatch. Re-run the target-set attention mask and permutation tests, including an
all-empty candidate set and peer permutations. Verify explicitly that selecting
peer slot A rather than B commands and delivers only to the corresponding receiver.
Do not expose centralized-baseline state to the learned policy. Each agent may see
only its own resources and catalog, declared target ephemerides/candidates,
declared contact-discovery information, and sender-local exchange/ACK history. It
must not observe a peer's private catalog, intentions, resources, active task,
onboard products, or uncommunicated navigation state.

Preserve the implemented directed completion link. A transmit action selects one
eligible peer, points to that peer's SimpleNav message using Basilisk
`locationPointing`, enforces Earth LOS and the imaging-equivalent attitude/rate
requirements, and holds valid pointing continuously for at least 10 seconds. Use
64 kbit/s for metadata, include the existing transport header, extend the hold only
when the frozen payload requires it, and retain the 300-second attempt deadline.
Loss of lock resets continuous packet progress. Charge 25 W through the existing
power sink. Do not transfer image bits or physical product ownership.

At transmit start, freeze the sender's completion-only catalog delta for the chosen
receiver: every qualified, durable record that the sender knows and that receiver
has not acknowledged, excluding records originally sourced by that receiver. Keep
source sensor, target and request epoch, exposure provenance, quality, and distinct
capture, completion, ground-delivery, and receipt timestamps. Successful receipt
must merge at the next simulation boundary. The merge must be durable, monotone,
and order independent: an old or relayed packet cannot overwrite a newer capture
or erase known delivery. Record per-receiver ACK state, payload records/bytes,
pointing/LOS duration, attempt outcome, and receipt time.

Preserve `reimage_cooldown_orbits=2.0`, derived as
11,834.835756586714 seconds for the 700/800/700-km sensing team and anchored at
qualified capture time. Shared completion knowledge should suppress immediate
duplicate work but must not permanently remove a target. Once the newest locally
known qualified capture is at least one cooldown old, the target becomes eligible
for useful repeat imaging. Keep first catalog coverage, cooldown-qualified repeat
services, and physical ground delivery as separate metrics.

First run focused local tests and a short real-Basilisk three-sensor scenario that
exercises both possible receivers, ACK/delta behavior, relayed/out-of-order records,
revisit eligibility after cooldown, and physical ownership. Then submit a
**one-rollout-worker mission-scale gate** for directed finite completion in both
conflict and continuous retasking. For each mode, collect a complete 45,000-second
episode, perform one PPO update, save the checkpoint, restore it in a fresh process,
prove exact saved-policy logits/actions on a frozen observation batch, resume the
optimizer/worker seed state for one additional update, and verify finite losses,
finite nonzero gradients, and nonzero parameter changes. Verify actual asynchronous
durations/rewards, all three agents, 100 passive targets, resources, products,
communication, and useful revisit behavior. Use targeted scenarios when random
rollouts do not exercise a required path.

I authorize submission of that one-worker gate. If and only if every gate passes,
I also authorize a bounded four-rollout-worker pilot for directed finite completion,
one matched training seed, conflict and continuous retasking, and at most **ten PPO
updates total per mode**, including the two gate updates if resumed with compatible
topology. If topology prevents exact one-to-four-worker continuation, start fresh,
state that clearly, and still cap each mode at ten updates. Do not start the broad
six-cell learned-policy study or a multi-training-seed study.

Evaluate each final restored policy on held-out seeds 10000–10004 in both all-LEO
and mixed environments. Generate independent and centralized-full-state heuristic
references for those same initial conditions without communication, training, or
checkpoint use. Treat centralized control as a maximum-information greedy reference,
not a guaranteed optimum. Pair every comparison by exact environment/seed hash and
report policy-minus-independent and policy-minus-centralized differences. Do not
mix the held-out runs into the completed baseline campaign's 50-seed estimates.

For every update and evaluation, record actual environment/agent batch sizes,
policy decisions, losses, gradient and parameter-change norms, checkpoint schema,
restore/resume proof, wall time, MaxRSS, AllocCPUS, and simulated seconds per wall
second. Report qualified first capture coverage, full ground-confirmed coverage,
cooldown-qualified acquisition and ground services, 90% acquisition and 10%
ground-delivery reward components, every penalty/adjustment, total reward,
duplicate attempts, successful duplicates, duplicate and interrupted sensor-seconds,
wasted-time fraction, stale and causally avoidable stale deliveries, cross-sensor
onboard overlap products/targets, redundant acquisitions, excess-holder time,
radio occupancy, recipient-selection counts, payload records/bytes, ACKs, packet
outcomes, and useful post-cooldown revisits. Plot learning diagnostics and paired
held-out effects with clear uncertainty limits; do not infer convergence from ten
updates or five evaluation seeds.

Save reproducible configs, source/dependency/runtime records, checkpoints, tables,
and plots under a new results directory. Update `CLUSTER_READINESS.md` and
`cluster/EXECUTION.md`, commit and push the code and compact evidence from the local
checkout, and finish with a candid go/no-go recommendation for a multi-seed learned
study. Include exact Slurm job IDs and commands actually used.
