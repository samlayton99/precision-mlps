# Paper figures 3 and 4

Figure 3 is updated against submission draft (27). Figure 4's three-panel revision is blocked on access to the Runpod volume. Figure 5 is unchanged.

## Figure 3: spectrum and readout learning

- [600-DPI PNG](../output/diagnostics/section34_long_horizon/main_5m/paired/spectrum_readout.png), 3300 × 1200 pixels.
- [Vector PDF](../output/diagnostics/section34_long_horizon/main_5m/paired/spectrum_readout.pdf), 5.5 × 2 inches.
- [Inline figure and caption](section34_spectrum_subsection.tex).
- [Compiled Sections 3.4–3.5](../output/pdf/section34_spectrum_note.pdf).
- [Source hashes and plotted values](../output/diagnostics/section34_long_horizon/main_5m/paired/spectrum_readout_provenance.json).

The titles name each panel's quantity: target-relevant rates and readout learning. Outlined markers emphasize measured values; dashed boundaries and light shading identify spectral intervals. Direct labels identify the three ranks. The bandwidth key occupies unused space below the learning curves, while the GD/bound key sits above them. A shared external legend would add height without reducing overlap here. The time axis now explicitly labels the five-million-update endpoint.

The figure retains the existing executed GD data, all twelve spectral intervals, target projections, and marker times. Their values and source hashes match the previous provenance record exactly. The three displayed ranks jointly carry about 59–63% of target energy; the output-error bound still uses every resolved target projection. The style changes do not alter the theorem calculation or select a new target.

The PNG and compiled page were inspected at manuscript width. Section 3.4 still fits on one review page. The note compiles without undefined references, duplicate labels, or overfull boxes; its construction reference now points to draft (27)'s Figure 2. Figure 4 remains an explicitly external reference while its revision is pending.

Regenerate Figure 3 without overwriting the legacy Figure 4:

```sh
python -m experiments.expD36_frozen_gamma_probe.section34_paired_figures \
  --analysis output/diagnostics/section34_long_horizon/main_5m/joint_analysis \
  --bounds output/diagnostics/section34_long_horizon/main_5m/bounds \
  --gd results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/section34_long_horizon_20260924/main/h5m/frozen_gd \
  --base results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/section34_long_horizon_20260924/main/base \
  --frozen-analysis output/diagnostics/section34_long_horizon/main_5m/uniform_access \
  --output output/diagnostics/section34_long_horizon/main_5m/paired \
  --spectrum-only
```

## Figure 4: inputs still needed

The agreed panels remain **(a) Slope scaling**, **(b) Output accuracy**, and **(c) Bandwidth acquisition**. Adam and GD keep consistent colors across panels; solid and dashed curves distinguish joint and frozen training. A shared legend below the three panels will avoid competing with the data. Panel (c) uses mean $\lambda_j=h|\gamma_j|$, with standard deviation across neurons pooled over the five equal-sized seed populations. The supplied $\lambda=1/4$ is a construction reference, not a claimed necessary threshold.

The following remote inputs are required:

1. Completed width-128, width-256, and width-1024 groups under `/workspace/junmiaoh/experiments/precision-mlps/runs/figure4_width_sweep_20260925`, including completion summaries, metadata, final states, and base validation arrays. Select one optimizer recipe per width by median final validation error across all five seeds. Check `completed_updates=5000000` and the final state count; preallocated trace lengths do not establish completion.
2. Width-512 parameter checkpoints and checkpoint steps under `/workspace/junmiaoh/experiments/precision-mlps/runs/section34_long_horizon_20260924/main/h5m/joint_adam_cosine` and `joint_gd`. The selected recipes are Adam index 0 and GD index 2. These parameters are needed for the population mean and neuron spread; saved RMS and quantiles cannot reconstruct them.

The existing local width-512 error traces suffice for panel (b). They do not supply the missing widths in (a) or the population trajectories in (c). The legacy two-panel image has been left unchanged and is not presented as the revised Figure 4.

On 2026-09-25, fresh SSH connections to `wkcyllpk7qm5i4-64411fca@ssh.runpod.io` entered the container as `uid=0(root)`. Both a metadata query on `/workspace` and a direct read of the sweep's completion record exceeded an eight-second timeout. `/workspace` is a FUSE network mount. `sbatch`, `srun`, and `squeue` were not found on the container's PATH. Root access therefore did not restore the data. No new training was launched, no storage was changed, and sweep completion remains unverified.
