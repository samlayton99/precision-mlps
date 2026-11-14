# Saved readout spectra

Evaluate all 409 saved checkpoints of the 216 completed tanh GD runs: 24 functions, three learning rates, and three initialization scales. No training or parameter changes.

At each checkpoint, form the actual readout feature matrix on the original 1,024 training points:

\[
A=[\tanh(x\gamma^T+\beta^T),\mathbf 1]/\sqrt{1024},\qquad
\rho_i=\sigma_i(A)^2/\sigma_{\max}(A)^2.
\]

Save every rate and the 0th, 10th, …, 100th percentiles across the 197 readout modes (196 neurons plus output bias). Quantiles use NumPy's linear interpolation. Larger percentiles indicate faster modes. The 100th percentile is always one. We do not append the 827 structural zero modes of the larger sample-space kernel.

For the paper's reference readout GD step size, a residual component contracts by `1 - rho_i/2` each update. These normalized rates describe frozen readout geometry; they are not the joint-training learning rate or a target-weighted convergence-time estimate.

Compute singular values directly, avoiding the loss of precision from forming the Gram matrix. Rates below `(1024 * float64_epsilon)^2`, approximately `5.17e-26`, are marked unresolved in figures. This is a conservative numerical rank convention, not an interval-certified eigenvalue bound. Their raw floating-point estimates remain in the saved data; no claim is made about their individual magnitudes.

## Outputs

`results/checkpoint_D_optimizers/expD61_saved_readout_spectra/`

- `runs/<run>/spectra.npz`: all rates (descending), percentile boundaries, largest eigenvalue, checkpoint steps, and unresolved fraction.
- `runs/<run>/percentiles.csv`: percentile boundaries at every checkpoint.
- `runs/<run>/metadata.json`: run identity and hashes of original checkpoint and initial-data files.
- `figures/spectrum_overview.png`: one panel per initialization and training learning rate. Faint lines show individual functions; solid lines give the median across functions for each spectral percentile.
- `figures/<function>.png`: that function's own percentile trajectories, without pooling functions.
- `figures/all_24_functions.pdf`: all individual function figures.
- `validation.json`: agreement of two SVD drivers, the Frobenius trace identity, and independently computed well-resolved Gram eigenvalues on four saved matrices.

Run with `.venv/bin/python experiments/expD61_saved_readout_spectra/run.py`. Completed runs are reusable when their source hashes match. Use `--plot-only` to redraw finished results.
