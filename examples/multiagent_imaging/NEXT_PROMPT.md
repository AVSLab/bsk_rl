# Next task prompt — authorize the bounded expansion

Continue from the current remote HEAD of `multi-agent-space-imaging-2026`, which
includes evidence commit `fd3c45278d37281abd0afa94e8cb9bbbd267c0f8`. Read
`examples/multiagent_imaging/cluster/EXECUTION.md`, `cluster/README.md`,
`BASELINE_MONTE_CARLO.md`, the passed one-worker gate at
`results/multiagent_imaging/cluster-one-worker-20260908-006b71c/validation_gate.json`,
and the two completed baseline episodes first. Preserve the separate completion-v2
checkout/environment and do not modify AMOS.

I authorize submission of the remaining baseline array tasks 1-99 and 101-199 from
the existing manifest in
`results/multiagent_imaging/baseline-mc-20260908-fd8ebd3/manifest.json`, with at most
eight concurrent tasks. Do not regenerate the manifest or rerun tasks 0 and 100.
Keep two sensors, 100 passive Basilisk/Vizard RSO spacecraft, ten candidates,
45,000-second complete episodes, and the existing 11,960.807123947805-second
cooldown for both LEO and mixed catalogs. After all 200 episodes exist, validate all
matched initial-condition pairs and aggregate capture and ground-delivery coverage,
duplicates, wasted sensor-seconds, service counts, resource behavior, runtimes and
paired 95% bootstrap intervals. Generate the final baseline coverage and paired-
difference plots.

I also authorize the prepared four-worker directed finite-completion pilot after
revalidating the saved one-worker gate against the current runtime. Run conflict and
continuous retasking only, training seed zero, eight fresh PPO updates per mode,
complete episodes, CPU learner, one Torch/BLAS thread per process, and the existing
eight-CPU/32-GiB allocation with 1800-second rollout timeouts. Preserve shared
target-set attention, 45,000-second reward half-life, 6000-second GAE trace half-life,
directed SimpleNav/LOS peer pointing, completion-only durable time-tagged catalogs,
receiver-local eligibility, 10-second minimum hold, 64-kbit/s metadata and
300-second attempt deadline. Do not add peer intent or private peer state.

Evaluate each final restored checkpoint on held-out seeds 10000-10004 against the
matched closest-angle heuristic. Report actual batch sizes and complete-episode
counts, decisions, finite losses/gradients, parameter changes, resume/restore checks,
rewards, services, duplicates, interruption waste, radio occupancy, packet outcomes,
resource/ownership checks, wall time and peak memory. Save reproducible configs,
source/dependency records, checkpoints, tables and learning curves. Recommend whether
the evidence justifies a larger multi-seed learned-policy study. Do not start the broad
six-cell study or any additional training seeds.
