# Figure 4(c): Adam intervention responses

This panel compares actual fine-slope motion with acquired endpoint slope scale over the completed 130,000–140,000-update pulse. It contains 48 points: six target families, widths 705 and 1409, seeds 30 and 31, and two interventions per matched native start. This cohort is separate from Figure 4(a,b). No training or optimizer selection is repeated.

| Quantity | Exact source and definition |
| --- | --- |
| Accumulated fine-slope path $P_F$ | `states.csv: fine_slope_path` at `offset=10000`; the counter is zero at the fork. |
| Endpoint RMS slope $s$ | `states.csv: slope_rms`, equal to $\sqrt{A/W}$. |
| Horizontal coordinate | $P_F^{\mathrm{intervention}}/P_F^{\mathrm{native}}$; agrees with `denominator_contrasts.csv: fine_path_ratio`. |
| Vertical coordinate | $100(s^{\mathrm{intervention}}/s^{\mathrm{native}}-1)$; agrees with `denominator_contrasts.csv: slope_effect_percent`. |
| Absolute endpoint bandwidth | `states.csv: lambda_rms`, equal to `h * slope_rms`; matched branches have identical `h`. |
| Policy | `gain_only` means Scalar amplification; `tracking_attenuated_variance` means Tracking-attenuated denominator. |

The original evidence lives under `results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/`. Summary inputs are `adam_variance_final_20260926/{denominator_contrasts.csv,native_denominator_gains.csv,facts.json}`. Absolute path and RMS measurements come from `adam_variance_runs_20260926/states.csv`, with completion checked against `runs.csv`. Full input paths, hashes, coordinate mappings, and numerical checks are in [the provenance record](data/adam_intervention_provenance.json). The [plotted CSV](data/adam_intervention.csv) includes both members of every comparison.

The analysis is implemented in `population_adam_variance_analysis.py`. The supplied `population_adam_motion_modal.py` is the campaign launcher. In `population_adam_motion.py`, `apply_proposal` adds `norm(fine[:width])` at every update; `fine` is the effective fine component including coarse compensation. `population_adam_variance.py` modifies that component after momentum and adaptive scaling, before applying and recording the update. The scalar histories therefore contain sums of per-update Euclidean norms, even though histories are exported every 1,000 updates. No path is reconstructed from checkpoints. Passive denominator gains are checked as cohort records, not used as horizontal coordinates.

## Checks and interpretation

All 72 branches are complete, with 24 unique native starts and 48 unique intervention matches. All 792 scalar records over the pulse have the correct ages, finite positive endpoint RMS values, common matched spacing, and nondecreasing cumulative paths. The pulse starts with zero path counters. Recomputed coordinates agree with all 48 summary rows and the corresponding extrema in `facts.json`.

Path ratios span 5.897–679.019; RMS responses span −1.057% to +15.154%. Scalar amplification increases RMS in 19 of 24 cases, and the altered denominator in 17 of 24. Both positive and negative responses are retained. Negative values mean smaller than the native endpoint, not necessarily smaller than the fork. These are intervention responses, not a fitted scaling law.

For the compact bump at width 705 and seed 30, scalar amplification gives $(x,y)=(15.5937365,0.0475123\%)$ and the tracking-attenuated denominator gives $(73.3119094,-1.0145427\%)$, reproducing the report's rounded checks without using them to set coordinates. The largest altered pulse-end bandwidth is 0.01285943, below 0.013. The construction reference 0.25 is a comparison scale, not a universal necessary threshold.

## Caption for panel (c)

**(c)** Adam intervention responses over 10k updates from a shared 130k checkpoint. Each point compares an intervention with its matched native control: accumulated fine-slope path versus percentage difference in final RMS slope. The 48 comparisons span six target families, widths 705 and 1409, and two seeds. Greater motion can increase acquisition but need not produce a substantially larger endpoint scale. All altered pulse-end RMS bandwidths remain below 0.013, compared with the construction reference 0.25. These interventions are separate from the experiments in panels (a,b).

## Rebuild

The bundled CSV suffices for rendering. To re-extract it from the original evidence:

```sh
python figure_handoff/optimization/prepare_adam_intervention.py --evidence /path/to/precision-mlps/results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence
```

Then run the existing `figure_handoff/optimization/render.py` command described in the [style handoff](README.md).
