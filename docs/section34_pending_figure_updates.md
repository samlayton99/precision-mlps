# Deferred population-bandwidth figure

Deferred at the user's request on 2026-09-24 until Runpod is available again. Continue the paper prose assuming the agreed population view; do not relabel the existing RMS data as a mean. The current Figure 4 artwork and its caption still describe RMS and 99th-percentile bandwidths. Replace them together when the checkpoint statistics are ready. No new training is needed.

## Agreed revision

- Keep Figure 4's left error panel, full five-million-update horizon, selected recipes, and five paired seeds. Its existing shading represents seed variation and recorded within-bin extrema, not a learning-rate sweep.
- Replace the right panel with one Adam line and one GD line showing arithmetic mean neuron bandwidth. Label the axis **Bandwidth $\lambda$**, and define individual bandwidths as $\lambda_{s,j}(n)=h|a_{s,j}(n)|$ using the original spacing $h=2/467$.
- Shade one population standard deviation across neurons, pooling the five equally sized seed populations. This is the spread of neuron bandwidths, not the standard error or the variability of seed means. With $S=5$ seeds and $W=512$ neurons,

$$
\overline\lambda(n)=\frac{1}{SW}\sum_{s,j}\lambda_{s,j}(n),
\qquad
\sigma_\lambda(n)=\sqrt{\frac{1}{SW}\sum_{s,j}
(\lambda_{s,j}(n)-\overline\lambda(n))^2}.
$$

- Use a linear bandwidth axis. Show the band from $\max(0,\overline\lambda-\sigma_\lambda)$ to $\overline\lambda+\sigma_\lambda$ and state that it is truncated at zero. Retain the supplied uniform $\lambda=1/4$ reference as a comparison, not a universal necessary threshold.
- Remove RMS and 99th-percentile curves from the paper figure. Keep panel identification in the caption as **Left / Right**. Retain the compact proportions and 600-DPI PNG exports, plus vector PDFs.

## Records to recover

Run root: `/workspace/junmiaoh/experiments/precision-mlps/runs/section34_long_horizon_20260924/main/h5m/`.

- Adam: `joint_adam_cosine/`, recipe index 0, initial learning rate 0.05.
- GD: `joint_gd/`, recipe index 2, initial learning rate 0.2.
- From each run: `parameter_checkpoints.npy`, `checkpoint_steps.npy`, and `metadata.json`. The selected full-horizon cosine schedules and seed selection are already recorded in `output/diagnostics/section34_long_horizon/main_5m/joint_analysis/summary.json`.
- Locally preserved final parameters are in `output/diagnostics/section34_long_horizon/main_5m/joint_analysis/selected_parameters.npz`. They verify endpoint statistics but cannot reconstruct the trajectory. The local `selected_traces.npz` retains RMS and percentile trajectories, not population means.

The verified pooled endpoint mean and population standard deviation are approximately 0.0168845 and 0.0541204 for Adam, and 0.00135795 and 0.00703476 for GD. These are checks on the extraction, not substitutes for intermediate checkpoints.

The prose's below-reference claim is already supported: the retained per-bin RMS maxima are 0.0621243 for Adam and 0.00861984 for GD, and arithmetic mean bandwidth cannot exceed RMS bandwidth. Both therefore stay below the supplied $1/4$ reference. This inequality does not reconstruct the mean or standard-deviation trajectories needed for the new plot.

## Resume and verify

1. Recover the selected checkpoint records and confirm the target, width, spacing, recipe, seed order, and five-million-update endpoint against the existing metadata and final parameters.
2. Compute the population moments at every saved checkpoint. Verify the initial parameters and final statistics, and check $\operatorname{mean}(\lambda^2)=\overline\lambda^2+\sigma_\lambda^2$ against the average of the squared per-seed RMS checkpoints (not the square of their average).
3. Update `experiments/expD36_frozen_gamma_probe/section34_paired_figures.py`, preserve the compact population statistics and source hashes, and regenerate the paired figure. Do not interpolate missing checkpoints into claimed measurements or substitute the separate two-million-update runs.
4. Replace the Figure 4 caption below, update the appendix's measurement definition and population discussion, and remove the temporary source comment in `docs/section35_feature_acquisition.tex`.
5. Verify unchanged error-panel numerical content and Figure 3 bounds, inspect both panels and their legends, and rebuild the excerpt and full note.

## Caption to use after regeneration

```latex
\caption{\textbf{Partial slope acquisition leaves a precision gap.}
\textbf{Left:} Joint Adam/GD relative output errors and frozen uniform
$\lambda=1/4$ references under the same five-million-update budget.
Shading retains seed variation and recorded within-bin extrema.
\textbf{Right:} Mean neuron bandwidth $\lambda_j=h|a_j|$ for the same
joint runs, pooled across five seeds; shading shows one population
standard deviation, truncated at zero. The supplied uniform reference
is a comparison, not a necessary slope threshold. Cosine schedules
span the full horizon; Appendix~\ref{app:note-evaluation} gives the protocol.}
```
