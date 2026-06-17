# Breckenridge / AAS GNC 2026 Onboarding Notes

This file is a handoff note for continuing the Breckenridge/AAS GNC imaging-vs-downlink work as a journal paper. It records the scripts, paper folders, figure style, and policy conventions discovered while preparing the AMOS 2026 GAT analysis.

## Main Paper Folders

- Final AAS/GNC paper folder: `/Users/dahu1128/Library/CloudStorage/OneDrive-UCB-O365/Documents/PhD/Research/journal_papers/AAS_GNC26_DHP_final`
- Journal-conversion working folder: `/Users/dahu1128/Library/CloudStorage/OneDrive-UCB-O365/Documents/PhD/Research/journal_papers/Imaging_vs_Downlink_JAIS`
- Both folders contain a `main.tex`, `references.bib`, AAS class/style files, and a `Figures/` directory with the alpha-sweep outputs.

## Main Analysis and Plotting Scripts

- `examples/mc_overall_from_json.py`
  - Aggregates Monte Carlo policy-evaluation JSON outputs into alpha/environment summary CSV and TeX tables.
  - Produces files like `overall_summary_by_alpha_<timestamp>.csv` and `overall_summary_by_alpha_<timestamp>.tex`.
  - Contains the policy directory mapping used for the AAS/GNC alpha sweep.
- `examples/updated_plot_alpha_sweep.py`
  - Primary plotting script for the paper-style alpha sweep figures.
  - Reads `results/overall_summary_by_alpha_allPolicies_20260116_150922.csv` by default.
  - Uses the plasma colormap, alpha-coded markers, shaded uncertainty bands, black-edged markers, and large serif labels.
- `examples/plot_alpha_sweep.py`
  - Earlier/related plotting script with similar alpha-sweep functionality.

## Key Figures

These are included in both the final GNC folder and the JAIS conversion folder:

- `Figures/reward_images_downlinks_simple_COLORBAR.png`
  - Combined alpha-sweep plot for reward, illuminated images, and delivered/downlinked images.
- `Figures/useful_downlinks_vs_alpha.png`
  - Delivered illuminated images and mean downlink fraction vs alpha.
- `Figures/actions_vs_alpha_combined_oneaxis.pdf`
  - Imaging and downlink action counts vs alpha.
- Supporting figures include `reward_vs_alpha.png`, `images_vs_alpha.png`, `downlink_actions_vs_alpha.png`, and `imaging_actions_vs_alpha.png`.

## Policy and Reward-Mix Convention

- Policy names use the reward mix tag `<downlinkPercent>d<imagingPercent>i`.
- The downlink reward weight is `alpha = downlinkPercent / (downlinkPercent + imagingPercent)`.
- Examples:
  - `0d100i` or `00d100i`: alpha = 0.0, all reward at imaging.
  - `20d80i`: alpha = 0.2.
  - `100d00i`: alpha = 1.0, all reward at downlink/ground delivery.
- The AAS/GNC paper discusses alpha as an operator-facing trade between bulk collection and lower-latency data delivery.

## Current Figure Style to Preserve

- Colormap: `plasma`, normalized over alpha in `[0, 1]`.
- Labels: serif/math-style labels with large font sizes.
- Markers: black edges, different marker shapes for reward/images/downlink quantities.
- Uncertainty: mean +/- standard deviation error bars plus light shaded bands.
- Axes: minor ticks and light grid lines.
- Outputs: use both `.png` for quick review and `.pdf` for paper inclusion.

## Known Paper Narrative

The GNC/Breckenridge work studies how a scalar reward split between onboard imaging value and downlinked image value changes learned policy behavior. The main observed behavior is that alpha near zero produces infrequent bulk downlinks, while alpha greater than zero causes more frequent, smaller downlinks. Intermediate alpha values preserve high image throughput while improving delivered-image timeliness without excessive downlink actions.

## Recommended Next Codex Starting Prompt

"Use `/Users/dahu1128/Library/CloudStorage/OneDrive-UCB-O365/Documents/PhD/Research/journal_papers/Imaging_vs_Downlink_JAIS` as the working paper folder. Read `main.tex`, `references.bib`, `examples/mc_overall_from_json.py`, and `examples/updated_plot_alpha_sweep.py`. Preserve the existing alpha-sweep plotting style and help convert the AAS/GNC conference paper into a fuller journal manuscript with expanded methods, limitations, and discussion."
