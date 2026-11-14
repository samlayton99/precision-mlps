# Joint-training diagnostics: analysis starts at 30k

All 24 runs reached 50,000 GD updates. This revision uses the existing training data; no retraining was performed.

The theorem ceiling is initialized once at the measured parameter norm at step 30,000 and then advanced at every GD update. E0 is fixed at the maximum recorded nonaffine residual RMS over steps 30,000–50,000. Moment and tracking inputs are interpolated between checkpoints. All four force alternatives are evaluated, and the smallest is used at each ceiling update.

The ceiling crosses the display cutoff of 1,000 in 24 of 24 cases, between total steps 30,412 and 49,273. This is a loose bound, not divergence of the measured training trajectory. Independent direct-B calculations match all 24 revised ceiling evaluations; there are no recorded parameter-envelope violations before cutoff.

The original analysis and figure starting at 1k are preserved in `figures_restart_1000/`. The training launch manifest is unchanged; `figures/diagnostics.json` records the current 30k analysis start.

[Figure](figures/joint_training_diagnostics.png) · [PDF](figures/joint_training_diagnostics.pdf)
