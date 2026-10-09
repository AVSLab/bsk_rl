# Canonical multi-agent space-imaging workspace

All current multi-agent space-imaging source belongs to branch
`multi-agent-space-imaging-2026` in
`/Users/dahu1128/Repositories/bsk_rl-multi-agent-space-imaging-2026`.
This includes directed transmit, completion catalogs, completed onboard inventory,
observations, shared-policy training, checkpoint/resume, study scripts, and tests.
GNC is a conference venue; it is not a separate implementation project.

The October 8 consolidation incorporated the latest checkpoint/transmit source
`e087f19fcc121bfbc0b1e35567570c3411106742` into the existing branch, including
the earlier policy-recovery and study revisions. That source was a standalone
snapshot with different Git ancestry, so its scoped files were reconciled into
the existing branch instead of merging an unrelated repository history.
The canonical checkout's existing edits were backed up first and retained in the
consolidated implementation. Existing baseline plotting and analysis tools remain
here too.

Use this checkout for future edits and new local evidence. Do not create another
implementation branch or checkout because a result will appear in a GNC paper.
Before substantial changes, preserve the dirty patch and source fingerprint here.
Historical GNC-named directories are retained source/release archives and cease
to be development workspaces. Their original paths and identifiers in sealed
release/checkpoint manifests remain provenance, not instructions to edit there.
Future prepared job labels and new W&B defaults use multi-agent space-imaging
names. Explicit environment settings and saved W&B resume identities still take
precedence, so historical runs retain their original projects and groups.

## Source and evidence locations

- Current implementation: this branch and checkout.
- Consolidation source manifests, original dirty source backup, and retained Git
  bundles: `output/branch-consolidation-2026-10-08/`.
- Retained recovery/checkpoint results: `results/multiagent_imaging/` in this
  checkout. The original immutable release and verified-download are preserved.
- Original transmit audit and its frozen source:
  `output/transmit-audit-2026-10-08/`. Its audit covers the earlier completion
  transport, and is not evidence that the later inventory mode was tested then.
- Current legacy and inventory-mode regression logs and native inventory canary:
  `output/branch-consolidation-2026-10-08/`.

The active recovery chat has been told to continue here. Existing authorized
cluster releases retain their reviewed source identities; consolidation itself
does not replace a deployed release or submit a job.

## Information contracts

The earlier `completion` case remains available with its own checkpoint schema.
The opt-in `catalog_tx` and `geometry_only` study cases have a distinct schema and
are documented in [STUDY_IMPLEMENTATION.md](STUDY_IMPLEMENTATION.md). Keeping both
in one branch does not make their checkpoints interchangeable.

Directed payload delivery is one-way: the selected sender's frozen metadata
merges into the selected receiver's local catalog. The receiver does not
automatically send its catalog back. Successful-delivery bookkeeping currently
uses an ideal acknowledgment assumption, documented separately from the payload.
Configured channel delay and loss are zero in the default study settings; the
physical pointing and continuous-hold time still occupy the sender.
