# Suggested next prompt

Work only in `/Users/dahu1128/Repositories/bsk_rl-multi-agent-space-imaging-2026` on
branch `multi-agent-space-imaging-2026`. Do not modify, switch, reset, stash, or clean the
original `/Users/dahu1128/Repositories/bsk_rl` AMOS worktree.

First verify that the branch is clean, pushed, and matches origin. Then implement the next
bounded evaluation phase for multiple sensing spacecraft executing one frozen shared
policy. Do not start cluster jobs or long training.

1. Review the existing AMOS 2026 evaluation/checkpoint-loading path and the current
   `examples/multiagent_imaging/evaluate.py` path. Identify every assumption that the
   evaluator has one agent, one `SS1`, one action stream, or one local datastore.
2. Add a multi-agent checkpoint evaluator that loads one shared `GNNModule` checkpoint and
   maps every explicit sensing role to it. Passive `target_*` spacecraft must never reach
   policy mapping or checkpoint inference.
3. Run the same frozen policy with 1, 2, and 3 sensing spacecraft in distinct configurable
   orbits. Keep the target catalog, seed, reward, cooldown, horizon, and information case
   paired. Do not create fixed peer slots; use the current pooled teammate observation.
4. Add configuration support for an explicit list of sensor orbits rather than inferring
   all orbits from agent names or list positions. Preserve the current two-orbit defaults.
5. Confirm that a checkpoint's observation contract records and validates global-feature
   count, per-target-feature count, candidate count, and whether spacecraft/team context
   conditioning is enabled. Fail clearly on an incompatible AMOS single-agent checkpoint;
   do not silently reshape it.
6. Produce one deterministic, untrained shared-controller rollout and one frozen-policy
   rollout for each sensor count. If no compatible trained multi-agent checkpoint exists,
   complete and test the loading/evaluation path with a smoke checkpoint and state that
   limitation explicitly.
7. Save per-sensor actions, elapsed action durations, rewards, battery/storage/wheel
   histories, local catalogs, received messages, and product ownership. Save team-level
   unique acquisitions/deliveries, ground value, duplicates, conflicts, broadcast time,
   and message-age diagnostics without double counting.
8. Add tests for one shared module instance, sensing-role-only mapping, 1/2/3-sensor shape
   compatibility, deterministic paired initial conditions, checkpoint-contract rejection,
   and absence of team-ledger leakage.
9. Update the architecture and validation notes with exact commands and pass/fail results.
   Recommend the smallest subsequent local training comparison, but do not submit jobs.
10. Commit the work in sensible commits and push the same branch. Report commit SHAs and
    provide the exact next prompt for authorizing short multi-agent training.
