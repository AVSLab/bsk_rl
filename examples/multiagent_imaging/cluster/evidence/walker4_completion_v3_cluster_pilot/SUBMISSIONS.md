# Walker-4 cluster submissions

The output root was
`/projects/dahu1128/bsk_rl-multi-agent-space-imaging-2026/results/multiagent_imaging/walker4-completion-v3-20260913`.
The jobs used the separate completion-v2 Python and Basilisk installations; the
AMOS checkout and environment were unchanged.

The one-worker validation gate was submitted as:

```bash
export BSK_STAGE=validate
export BSK_OUTPUT="$BSK_PILOT_ROOT/one-worker"
test ! -e "$BSK_OUTPUT" && test ! -e "${BSK_OUTPUT}.preflight"
sbatch --export=ALL examples/multiagent_imaging/cluster/pilot.slurm \
  | tee "$BSK_PILOT_ROOT/submission-one-worker.txt"
```

Slurm returned job `32528119`.

After `one-worker/validation_gate.json` reported `passed: true`, the bounded
four-worker stage was submitted as:

```bash
export BSK_STAGE=learn
export BSK_VALIDATION_GATE="$BSK_PILOT_ROOT/one-worker/validation_gate.json"
export BSK_OUTPUT="$BSK_PILOT_ROOT/four-worker"
export BSK_UPDATES=8
test ! -e "$BSK_OUTPUT" && test ! -e "${BSK_OUTPUT}.preflight"
sbatch --export=ALL examples/multiagent_imaging/cluster/pilot.slurm \
  | tee "$BSK_PILOT_ROOT/submission-four-worker.txt"
```

Slurm returned job `32530057`. No broad six-cell or multi-training-seed learned
study was submitted.
