# Next task prompt — authorize only setup and validation

Continue on `multi-agent-space-imaging-2026`. Read `examples/multiagent_imaging/cluster/README.md`, `BASELINE_MONTE_CARLO.md`, and the prepared release/source records first.

Deploy the reviewed release into `/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026`, preserving the existing AMOS checkout and environment. Use the separate `/projects/dahu1128/.venv-completion-v2` environment and the recorded Basilisk source commit. I authorize submission of the prepared `build_runtime.slurm` job on account `ucb550_asc2`, partition `acpu`, QOS `cpu-normal`. Stop and diagnose if the build or runtime audit fails.

Once the runtime passes, generate the baseline manifest from the deployed sources. I authorize only the first two baseline array tasks, IDs 0 and 100: independent and centralized LEO, seed zero. Keep two sensors, 100 passive RSO spacecraft, ten candidates, 45,000-second episodes, and the existing implemented cooldown unchanged. Verify matched initial states, no radio actions, physical capture/ground delivery, resources, coverage denominators and saved artifacts.

I also authorize the one-worker directed-learning validation stage in `pilot.slurm`: conflict and continuous retasking, one initial and one resumed PPO update per mode, matched seed zero, complete episodes, eight CPUs/32 GiB, CPU learner, one thread/process, and 1800-second rollout timeout. Preserve the completion-v2 formulation, half-lives and communication settings. Verify actual restored weights, Adam/seed resume, finite losses/gradients, parameter changes, physical durations, resource/ownership behavior, and matched held-out evaluation.

Report actual cluster runtime, memory, batch sizes, outcomes, limitations and exact commands for expansion. Stop before submitting the remaining 198 Monte Carlo tasks or four-worker learning. Do not start a broad six-cell or multi-seed learning study.
