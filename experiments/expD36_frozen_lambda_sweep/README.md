# Frozen-lambda readout sweep

The first empirical phase of the frozen-geometry capacity/access campaign. Geometry is fixed; only raw readout coefficients and the output bias train. There is no browser or theorem diagnostic in this experiment.

Update, 21 September: lambda=1 (gamma128) was appended to the baseline and both coordinate ablations through 20k steps. There are now nine lambda values per condition. All 12 figures include the addition; existing trajectories and optimizer state entries were preserved exactly.

`config.yaml` specifies the initial 20,000-step run: N256, 24 halo centers per side, 1,021 training midpoints, 4,093 evaluation midpoints, zero readout, ordinary GD and Adam, and the revised note's sine/quadratic/mixed-sine targets plus the repository's Runge function. Eight lambda values span the PR's Xavier RMS slope scale to .5, with the nearest interior log-grid point replaced by .25.

GD uses a constant rate equal to the reciprocal of the largest eigenvalue of the half-MSE Hessian. Adam uses constant rate .001, betas (.9, .999), epsilon 1e-8, and no weight decay. Both use fresh direct feature/residual products in FP64; no rounded Gram matrix, least-squares update, line search, or geometry update enters training.

## Run and continue

From the repository root:

```sh
.venv/bin/python experiments/expD36_frozen_lambda_sweep/run.py
.venv/bin/python experiments/expD36_frozen_lambda_sweep/run.py --steps 50000
.venv/bin/python experiments/expD36_frozen_lambda_sweep/run.py --plot-only
```

The second command resumes the saved coefficients, Adam moments, and global step. It does not reset the optimizer. Configuration changes other than horizon/plot snapshots/thread count are rejected against the existing checkpoint; use a different `--output` directory for an ablation.

## Outputs

One results directory, `results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/`, contains:

- `figures/early.png`, `middle.png`, `final.png`, `best.png`: requested 3×4 figures, four function columns. The top row compares error against lambda. The next rows contain all GD/Adam trajectories. Early/middle/final use 2k/10k/final-step slices and vertical step markers. Best uses each trajectory's best recorded evaluation and marks its location with a dot.
- `data/trajectory.npz`: all recorded training/evaluation errors, selected physical readout snapshots, latest readouts and Adam state, and best-snapshot indices.
- `data/reference.npz`: geometry, sample grids, targets, numerical least-squares coefficients/errors/singular values/ranks, and GD rates.
- `data/metadata.json`: full configuration, equations, axis conventions, and numerical policies. `data/status.json` records completion.
- `expD36_results.md`: protocol and result record.

Evaluations occur every step through step100, then every20 steps. Coefficients are additionally saved every200 steps, and atomic continuation checkpoints every1000 steps. Best means best among those recorded evaluations. Relative L2 is always a norm ratio, not squared loss. All panels use the independent evaluation grid; least squares minimizes the training-grid objective, with one explicit SVD cutoff shared across lambdas.

## Validation

`test_run.py` checks the batched update against standard PyTorch SGD/Adam, the half-MSE gradient normalization, zero-start relative error, exact checkpoint continuation, and consistency of the least-squares evaluation with the actual feature matrix.

```sh
.venv/bin/python -m pytest -q experiments/expD36_frozen_lambda_sweep/test_run.py
```

## One-factor coordinate ablations

Run one arm per invocation, sequentially:

```sh
.venv/bin/python experiments/expD36_frozen_lambda_sweep/ablations.py neighbor_unscaled
.venv/bin/python experiments/expD36_frozen_lambda_sweep/ablations.py sqrt_allowance
```

The first uses adjacent feature differences, retaining the bias and final tanh anchor. The second uses the PR's diagonal square-root allowance map, including its bias and corrected-halo factors, with this campaign's 24 halos per side. Its allowances stay fixed at reference lambda .25 throughout the actual lambda sweep. The two changes are not combined.

Both load the original baseline configuration, use the same zero physical start and samples, and preserve all optimizer settings. GD's numerical rate is recomputed by the same largest-curvature rule in the new coordinates. Adam keeps its native rate and epsilon; there is no hidden retuning to compensate for the physical scale change.

Each arm writes under `ablations/<arm>/` in the existing results directory. Four 4×4 figures retain the original three rows and add a fourth row comparing ablation (solid) with the reused raw baseline (dashed), with orange GD and blue Adam. The best figure independently selects the best recorded error for both conditions within the same budget. Numerical least-squares references are copied from the baseline, not recomputed with map-dependent truncation.

Saved `coefficient_snapshots` and `physical_c` remain physical readouts. The continuation field `c` and its Adam moments are native optimizer coordinates; the reference file stores M so that physical readouts are M times those coordinates. The baseline has M=I. To extend a comparison, first extend the baseline to the desired horizon, then invoke the selected arm with the same `--steps` horizon. `--plot-only` reuses saved data.

Map tests additionally check physical prediction equivalence, the gradient chain rule, full rank, reference preservation, the reviewed PR allowance values, and exact continuation in native coordinates.

## Add a lambda without retraining existing cases

```sh
.venv/bin/python experiments/expD36_frozen_lambda_sweep/append_lambda.py 1
```

This trains only missing cases, sequentially across the baseline and both maps, then merges the new data and redraws the figures. Existing values are skipped. The merge verifies the original metric histories, coefficient snapshots, and optimizer moments exactly. New lambda additions are recorded in each metadata file. Keep `additional_lambdas` in the main configuration synchronized with the saved baseline for subsequent full-sweep continuation; lambda1 is already included there.
