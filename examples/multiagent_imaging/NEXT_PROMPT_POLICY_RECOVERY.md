Continue from the final Walker-4 completion-v3 cluster-pilot evidence commit on
`multi-agent-space-imaging-2026`. Read
`examples/multiagent_imaging/cluster/evidence/walker4_completion_v3_cluster_pilot/REPORT.md`,
`examples/multiagent_imaging/WALKER4_LEARNED_PILOT.md`,
`examples/multiagent_imaging/CLUSTER_READINESS.md`, and
`examples/multiagent_imaging/cluster/EXECUTION.md` first. Treat jobs 32528119 and
32530057, their checkpoints, raw results, and compact evidence as immutable. Do
not rerun or overwrite them.

Diagnose the final deterministic empty-downlink collapse before doing more
training. For every saved checkpoint in conflict and continuous modes, evaluate
frozen observations and a small set of matched real-Basilisk episodes with both
deterministic and stochastic action selection. Record per-action logits,
probabilities, masks, entropy, value predictions, advantage and return summaries,
empty-downlink selections, candidate availability, resource state, and reward
components. Verify that target actions remain unmasked when valid candidates
exist, the action identities are unchanged, PPO advantage signs and physical-time
discounting are correct, and restored inference applies the same masks and
preprocessing used during training. Explain why stochastic sampling produced
useful training trajectories while the final argmax chose downlink exclusively.

Add a versioned operational action shield using only receiver-local information.
Mask downlink whenever local physical storage is empty, preserve charge as a
valid fallback for every finite observation, and add a reviewed low-battery rule
that prevents a newly selected power-consuming task when the sensor cannot retain
the required reserve through its minimum action duration. Do not expose peer
resources, catalogs, products, intentions, active tasks, or uncommunicated state.
Re-run all action-identity, candidate/peer permutation, all-empty-set, resource,
and checkpoint-mismatch tests. Add focused real-Basilisk tests proving that the
shield prevents empty-downlink loops without blocking valid ground delivery,
imaging, useful post-cooldown revisits, or either directed receiver slot.

Keep four Walker Delta 4/2/1 sensing agents, 100 passive Basilisk/Vizard RSO
spacecraft outside the RL agent list, the exact mixed 50 LEO/30 MEO/20 GEO target
population, ten candidates, three peer slots, completion-only durable time-tagged
catalog deltas, directed SimpleNav/locationPointing communication, Earth LOS,
imaging-equivalent attitude/rate requirements, 10-second minimum hold, 64-kbit/s
metadata rate, 64-byte header, 300-second deadline, 25-W draw, receiver-local
eligibility, 11,834.835756586714-second capture-anchored cooldown, 45,000-second
reward half-life, and 6,000-second GAE trace half-life. Do not transfer image bits
or physical ownership. Keep the independent and centralized-full-state references
as zero-radio evaluation controls; label centralized control as a
maximum-information greedy reference, not a guaranteed optimizer.

Prepare a policy warm start from reviewable heuristic demonstrations. The image,
charge, downlink, desaturation, and completion-transmit labels must use information
available to the learned actor. Include examples for all three receiver slots and
communicate only when a nonempty completion delta and a useful eligible peer exist.
Compare behavior cloning plus PPO with PPO from random initialization on identical
short validation seeds. Keep a separate validation-seed set for checkpoint
selection; do not use held-out seeds 10000–10004 until one final test.

Run local curriculum gates first. Preserve the 100-target mixed population and
the observation/action contract while using 3,000-, 12,000-, and finally
45,000-second complete episodes. Advance only when deterministic policies produce
qualified captures, ground delivery, stable batteries, more than one action type,
and selected-recipient completion packets when deltas exist. Evaluate every
checkpoint and select by a declared validation score with hard operational
constraints; never assume the last checkpoint is best. Report deterministic and
stochastic results separately.

Prepare, but do not submit, a one-worker cluster recovery gate using the existing
separate checkout, Python environment, Basilisk runtime, support-data cache, and
Slurm resource shape. Make the exact command reviewable. Do not launch a
four-worker job, broad six-cell comparison, or multi-training-seed study without
new explicit authorization. Finish with the diagnosed cause, tests, local
evidence, chosen checkpoint rule, remaining risks, estimated cluster cost, and a
candid go/no-go recommendation for that one-worker recovery gate.
