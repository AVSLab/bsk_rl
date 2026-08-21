# AMOS 2026 200-target Vizard validation — 1 Hz final playback

Canonical playback:

`artifacts/amos_2026/vizard/_VizFiles/AMOS2026_mixed_a0p1_200targets_seed0_1Hz_UnityViz.bin`

Configuration:

- mixed-trained alpha-0.1 checkpoint 119
- seed 0
- exact 100/60/40 LEO/MEO/GEO catalog
- 200 targets, priority sum 200
- 45,000 s episode
- 1 Hz Vizard sampling
- midpoint HIO/SHIO event with five HIOs and three SHIOs

Completed-run checks:

- 45,000 Vizard frames through 45,000 s
- one `bskSat` observer silhouette, 200 blue catalog markers, and eight initialized promotion
  proxies are present from the first frame
- all 200 targets are blue priority-tercile circles before promotion
- the priority event is applied at 22,520 s
- the five star and three triangle proxies remain hidden at Earth's center through the
  22,521 s recorded frame
- at 22,522 s, the eight purple proxies move to their promoted targets and the
  corresponding eight ordinary blue proxies move inside Earth, eliminating marker
  overdraw
- after promotion, exactly five dedicated HIO markers are medium-purple stars with RGBA
  `(143, 99, 167, 255)`
- after promotion, exactly three dedicated SHIO markers are deep-purple triangles with RGBA
  `(84, 39, 136, 255)`
- the purple proxy orbit and true-trajectory lines are opaque white
  `(255, 255, 255, 255)`, matching the ordinary catalog presentation
- every promoted marker is co-located with its corresponding target to numerical
  precision in the audited frames
- the native generic-storage panel is enabled, restoring its expandable Vizard panel
- the live dialog title is exactly `SPACE SURVEILLANCE`
- 1,000-km visualization-only lifecycle halos include cyan eligible
  `(0, 180, 216, 230)`, red cooldown `(228, 87, 86, 230)`, and green onboard
  `(42, 157, 91, 230)`
- promoted targets also carry a 1,600-km HIO or 1,888-km SHIO purple halo
- halo gridlines are disabled to avoid dark arcs and retain a clean translucent shell
- 457 illuminated images were acquired and 450 useful images were delivered
- 169 unique targets were imaged

Vizard reads a spacecraft's sprite shape and fill from its first message and ignores
later changes to that field. Eight immutable purple promotion sprites are therefore
initialized at Earth's center in frame 1. At the event, each purple proxy receives its
target's state while the matching blue visualization proxy receives an Earth-centered
state. This guarantees that the initialized star or triangle replaces, rather than
overlaps, the ordinary circle.

Vizard's spacecraft-sprite message supports one primitive shape and one RGBA fill. It
does not expose a separate marker-border or stroke field. The lifecycle visualization
therefore uses a larger, separate ellipsoid status halo: cyan for eligible, red for
cooldown, and green for an image in onboard storage.

Vizard hard-codes the expandable generic-storage panel title as the spacecraft display
name plus `Storage`; the protocol exposes no independent panel-title field. Restoring
the expandable panel therefore restores the displayed `Space Surveillance Storage`
title as well.

The superseded 2 Hz playback was moved recoverably to macOS Trash as
`AMOS2026_mixed_a0p1_200targets_seed0_2Hz_superseded_20260812_UnityViz.bin`.
The prior 1 Hz file with dynamically introduced, overlapping promotion markers was
moved recoverably to Trash as
`AMOS2026_mixed_a0p1_200targets_seed0_1Hz_pre_proxy_halo_fix_20260812_UnityViz.bin`.
