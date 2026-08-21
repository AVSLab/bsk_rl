# AMOS 2026 200-target Vizard validation — superseded 2 Hz playback

This playback has been replaced by the final 1 Hz recording documented in
`vizard_200target_1hz_validation_20260812.md`. The 2 Hz binary was moved recoverably to
macOS Trash as
`AMOS2026_mixed_a0p1_200targets_seed0_2Hz_superseded_20260812_UnityViz.bin`.

Canonical playback:

`artifacts/amos_2026/vizard/_VizFiles/AMOS2026_mixed_a0p1_200targets_seed0_2Hz_UnityViz.bin`

Configuration:

- mixed-trained alpha-0.1 checkpoint 119
- seed 0
- exact 100/60/40 LEO/MEO/GEO catalog
- 200 targets, priority sum 200
- 45,000 s episode
- 2 Hz Vizard sampling (0.5 s between frames)
- midpoint HIO/SHIO event with five HIOs and three SHIOs

Completed-run checks:

- 90,000 Vizard frames through 45,000 s
- sprite counts remain 192 circles, five HIO stars, and three SHIO triangles
- transceiver label remains empty
- sampled ground-station lines never appear with the transceiver off
- illuminated and non-illuminated onboard storage categories both occur
- cyan eligible, red cooldown, and green onboard status-ring colors all occur
- promotion halos turn on after the midpoint event
- 454 illuminated images acquired; 447 useful images delivered
- 461 imaging actions, 30 downlink actions, and zero Desat actions

The superseded 1 Hz 200-target playback was moved to macOS Trash as
`AMOS2026_mixed_a0p1_200targets_seed0_1Hz_superseded_20260812_UnityViz.bin` and remains
recoverable until Trash is emptied.

Vizard's native generic-storage panel automatically appends the word `Storage` to the
spacecraft display name. The public API does not expose a separate panel-title field.
The custom live dialog and spacecraft display name are both set to `Space Surveillance`.
Because the event-dialog text also has no documented rich-text color field, the priority
key uses three always-on GenericStorage bars with the exact light, medium, and dark blue
fills instead of relying on colored inline text.
