# Completed Figure 4 revision

The deferred population-bandwidth revision was superseded by the completed three-panel figure. The current artwork and caption use:

- **(a) Slope scaling:** acquired mean absolute slopes versus total width, with the supplied construction scale for comparison.
- **(b) Output error:** executed joint Adam/GD and frozen-readout references through five million updates. The frozen reference uses common bandwidth $\lambda=1/4$; the target and recipe selection are recorded in the caption and provenance.
- **(c) Motion vs. scale:** 48 matched Adam intervention responses from the separate denominator campaign. The horizontal coordinate is accumulated actual fine-slope path relative to native Adam over updates 130,000–140,000. The vertical coordinate is the percentage change in endpoint RMS slope relative to the matched native endpoint.

Panel (c) does not display a bandwidth trajectory or a construction threshold. Its six-target cohort is separate from the joint-training runs in panels (a,b); positive and negative responses are both retained. See the [completed sweep](figure4_completed_sweep.md) for panels (a,b) and the [Adam population note](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/ADAM_POPULATION.md) for the intervention interpretation.

The self-contained plotting package is published on branch `figure/optimization-style-handoff` under `figure_handoff/optimization/`. It contains the exact compact data, column mappings, validation records, renderer, vector PDFs, and high-resolution PNGs. It reproduces the final artwork without training or raw checkpoints.

Raw local histories and redundant export segments were deleted at the user's request on 2026-09-27 after retaining compact evidence. Historical run paths in execution notes describe the original campaign; they are not promises that the raw files remain available. The earlier pending instructions and their local edits are preserved in the consolidation backup, not as the current figure specification.
