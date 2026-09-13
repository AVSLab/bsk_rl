# Walker-4 completion-v3 local validation

This is bounded preflight evidence generated with the real Basilisk dynamics stack. It is not a learning result and it is not statistical evidence. The full 3,000-second rollout JSON remains outside Git because it is 512 KiB; `summary.json` is the compact, reviewable record.

## Contract exercised

- Four sensing agents in a 4/2/1 Delta Walker constellation at 700 km and 97 degrees.
- Exactly 100 passive RSO spacecraft: 50 LEO, 30 MEO, and 20 GEO.
- Ten target candidates and three receiver-selective peer actions per sender.
- Completion-only metadata at 64 kbit/s with a 64-byte transport header, at least 10 seconds of uninterrupted valid pointing, Earth LOS, a 300-second deadline, and the existing 25 W radio load.
- Capture-anchored two-orbit cooldown: 11,834.835756586708 seconds in the live environment (floating-point equivalent to the documented 11,834.835756586714 seconds).

## Result

The scenario passed. It produced 54 qualified first-capture targets, 33 fully ground-delivered targets, nine accepted directed packets, and nine completed pointing holds. Every receiver identity and every peer-slot index appeared in accepted transmissions. The packets carried 99 completion records and 22,756 bytes in total. Physical products remained owned by the capturing sensors; catalog exchange did not transfer image bits or ownership.

The resulting receiver-local ACK links were `sensor_0->sensor_3`, `sensor_1->sensor_2`, `sensor_1->sensor_3`, `sensor_2->sensor_1`, `sensor_3->sensor_0`, and `sensor_3->sensor_1`. Targeted tests separately cover every selected peer slot, exclusion of unselected receivers, frozen deltas, retries, loss of lock, expiry, relayed/out-of-order monotone merge, distinct timestamps, cooldown re-eligibility, and preservation of physical ownership.

## Interpretation

This validates environment construction and the physical communication path at short duration. It does not replace the authorized complete 45,000-second one-worker cluster gate, checkpoint restore/resume proof, or held-out evaluation.
