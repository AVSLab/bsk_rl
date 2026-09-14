# Walker-4 completion-v3 bounded cluster pilot

This compact record validates the authorized one-worker gate and four-worker pilot. Full checkpoints, event histories, and raw episodes remain in the immutable cluster results directory. Eight four-worker updates plus the two one-worker gate updates were run per mode. This is mechanics and early-learning evidence, not convergence evidence.

Executable source: `9ec3071ca5dd481e098b28cd84b813a035286789`; Basilisk: `8fcb54b2fb28388efb711786630501944fddec28`.

## Recommendation

**NO-GO for a multi-seed learned study.** conflict final policy had zero held-out capture coverage; conflict final policy depleted sensors to zero battery; conflict final policy delivered no held-out completion packets; continuous final policy had zero held-out capture coverage; continuous final policy depleted sensors to zero battery; continuous final policy delivered no held-out completion packets. The implementation and checkpoint mechanics passed, but these final policies are not operational candidates.

## Held-out paired effects

Every value is restored policy minus the named reference on the same exact mixed-population seed and initial-condition hash. Intervals are two-sided 95% paired t intervals over five seeds (10000-10004).

| Mode/reference | Capture coverage | Ground coverage | Cooldown acquisitions | Ground services | Total reward | Duplicate attempts | Wasted sensor-s |
|---|---:|---:|---:|---:|---:|---:|---:|
| conflict vs independent | -100.00 [-100.00, -100.00] pp | -100.00 [-100.00, -100.00] pp | -361.20 [-364.86, -357.54] | -337.40 [-354.03, -320.77] | -18312.92 [-18323.88, -18301.97] | -704.40 [-740.54, -668.26] | -86785.40 [-90600.17, -82970.63] |
| conflict vs centralized full state | -100.00 [-100.00, -100.00] pp | -100.00 [-100.00, -100.00] pp | -396.20 [-401.35, -391.05] | -382.20 [-405.32, -359.08] | -18347.77 [-18356.58, -18338.95] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| continuous vs independent | -100.00 [-100.00, -100.00] pp | -100.00 [-100.00, -100.00] pp | -341.60 [-351.56, -331.64] | -320.00 [-336.31, -303.69] | -18293.60 [-18304.32, -18282.88] | -772.40 [-794.81, -749.99] | -101285.80 [-104969.97, -97601.63] |
| continuous vs centralized full state | -100.00 [-100.00, -100.00] pp | -99.60 [-100.28, -98.92] pp | -337.40 [-376.38, -298.42] | -296.60 [-345.54, -247.66] | -18288.70 [-18336.65, -18240.76] | -10.00 [-14.73, -5.27] | -48424.80 [-58106.77, -38742.83] |

The centralized-full-state controller is a maximum-information greedy coordination reference, not a guaranteed global optimum. The learned actor has receiver-local information only. It selects one of three peer slots and receives completion metadata only after physical directed pointing and a successful packet.
Centralized catalog updates are counted as ideal catalog merges rather than transport packets; both heuristic references have zero physical radio packets.

## Held-out action behavior

- **conflict:** mean downlink selections 17951.0; mean selections of all other actions 0.0.
- **continuous:** mean downlink selections 17951.0; mean selections of all other actions 0.0.

Both restored final policies selected downlink at every held-out decision, despite having no physical products to deliver. That collapse explains the zero capture/ground coverage, zero useful revisits, zero completion packets, and repeated operational penalties; it is a policy-quality failure rather than a checkpoint or simulator-execution failure.

## Communication validation

- **conflict training:** recipient selections `{'sensor_0': 237, 'sensor_1': 222, 'sensor_2': 186, 'sensor_3': 234}`; peer slots `{'0': 237, '1': 408, '2': 234}`; packet outcomes `{'accepted': 879}`; ACK links 8.
- **conflict final held-out policy:** recipient selections `{}`; peer slots `{}`; packet outcomes `{}`; ACK links 0.
- **continuous training:** recipient selections `{'sensor_0': 169, 'sensor_1': 189, 'sensor_2': 172, 'sensor_3': 191}`; peer slots `{'0': 169, '1': 361, '2': 191}`; packet outcomes `{'accepted': 721}`; ACK links 8.
- **continuous final held-out policy:** recipient selections `{}`; peer slots `{}`; packet outcomes `{}`; ACK links 0.

All training and held-out episodes reached 45,000 seconds with four sensing agents, 100 passive Basilisk/Vizard RSO spacecraft outside the RL agent list, and the exact 50 LEO/30 MEO/20 GEO target split. The capture-anchored two-orbit cooldown remained 11,834.835756586714 seconds within floating-point roundoff. Every update had finite nonzero gradients and parameter changes. Checkpoint logits matched exactly and restored actions were identical. The independent and centralized references had zero radio activity.

See `summary.json` for every controller mean and paired interval, `held_out_metrics.csv` for all 30 evaluations, `paired_differences.csv` for the 20 exact-seed effects, and `updates.csv` for all 20 gate/pilot updates. Runtime, package/native-module hashes, support-data audits, launch plans, checkpoint manifests, exact submissions, and Slurm accounting are retained beside them.
