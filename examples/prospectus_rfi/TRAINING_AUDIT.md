# AMOS 2025 architecture-training audit

## Outcome

The currently running 100-second, N=100--200 candidate sweep is operationally
sampling and updating, but its W&B chart does not establish whether it is learning.
The chart uses PPO iteration on the x-axis and raw episode return on the y-axis. Those
quantities are not comparable to the historical AMOS 2025 curve because the current
batch is 4,200 transitions per iteration while the archived August 13 run used 180,
and the current episode target count varies from 100 through 200.

Fourteen current iterations contain approximately 58,800 sampled transitions. The
first 120 historical iterations contained approximately 21,600. A low iteration count
therefore is not evidence that the simulator is stuck. Conversely, a noisy or flat raw
return is not evidence that training is correct: catalog-size variation can move raw
return even when the observation fraction is unchanged.

## Changed factors

Relative to the archived best MLP run, the new sweep changes the catalog distribution,
image and downlink durations, initial battery range, target observation normalization,
validity masking, PPO batch size, PPO epochs, entropy, discount, GAE lambda, and network
class. This prevents the existing sweep from serving as a one-variable architecture
control.

The added historical attention control therefore fixes N=100, K=10, the episode at
45,000 seconds, image/charge/downlink/desaturation durations at 300/300/180/150 seconds,
initial battery at 10--40%, alpha=0 observation reward, and the archived target/global
observation fields and normalization. It appends only the validity mask required by the
target-set attention policy. It also restores the archived PPO cadence: batch 180,
10 epochs, learning rate 1e-6, clip 0.15, gamma 0.9997, lambda 0.95, entropy 0, and
gradient clip 1.0.

This is a checkpoint-physical-regime control, not a bitwise reproduction: the attention
architecture did not exist in the AMOS 2025 campaign, and its mask adds ten observation
values that the frozen 87-input MLP did not consume.

## Required decision metrics

Training monitoring should use environment steps and wall-clock hours. Raw return should
be accompanied by successful-observation fraction, illuminated-observation fraction,
episode target count, survival, and constraint interventions. Architecture claims should
use held-out, fixed-N paired evaluation rather than training return. For the prospectus
comparison, cumulative illuminated unique targets should additionally be evaluated over
simulation time (including the 15,000-, 30,000-, and 45,000-second table landmarks) to
detect target-saturation behavior that final counts conceal.

## Limitations

The screenshots alone cannot distinguish weak learning from reward noise, catalog-size
mixture effects, or an optimizer mismatch. The added diagnostic reports descriptive
slopes but does not turn training telemetry into a performance test. Superiority remains
unclaimed until paired Monte Carlo evaluation supports it.
