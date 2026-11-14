# D59 — QUILL accuracy scaling on the smooth target subset

Completed November 12, 2026. This is the fixed-geometry, least-squares QUILL recipe requested by Sam, not gradient training or the direct contour encoder.

## Targets

Include all 16 original D56 functions plus D58's erf transition, shifted Gamma, Airy, and smooth compact bump: 20 functions. Exclude the square-root cusp, x sin(1/x), shifted step, and square wave because they are nonsmooth. The compact bump is C-infinity but nonanalytic and is intentionally retained. Steep but smooth erf/tanh transitions are also retained.

Exact definitions remain in the original source modules; this experiment imports them rather than duplicating formulas. All domains are [-1,1]. Each target is divided by its fixed RMS on D56's 1,024 midpoint grid. This normalization stays constant across widths, and relative error is invariant to the overall target scale.

## Construction

- Float64 tanh, uniform centers, lambda=0.25 at every resolution.
- N interior intervals in [8,16,32,48,64,96,128,192,256,384,512,768,1024,1536,2048].
- R=max(20,ceil(sqrt(N))) halo centers on each side; total width W=N+1+2R, from 49 to 2,141.
- Spacing h=2/N, slope gamma=lambda/h, and centers -1+jh for j=-R,...,N+R.
- Endpoint-inclusive uniform fitting grid with max(2003,4(N+1)) samples. Target values are sampled only inside [-1,1]; Gamma's exterior pole is therefore never sampled.
- Append a constant column; use the audited `docs.qi_practical_guide.reference.fit_qi` implementation, SciPy `gelsd`, and explicit relative cutoff 1e-14.
- Fit the 20 target columns in one shared factorization per width. The implementation's internal RHS centering/scaling is reversed in the saved output bias/readout. There is no target-specific bandwidth/cutoff selection.

## Evaluation

The primary plotted score is relative L2 on 65,536 uniform midpoint samples, evaluated from each saved ordinary scalar MLP readout. Hidden features are cached across output columns, but each scalar readout is evaluated by its own matrix-vector operation, matching `MLP.__call__` for a scalar output. Additional metrics use 8,192 fixed random points and endpoint/feature-focused probes. Two interlaced half-density grids check sampling resolution. None of the evaluation points select fit coefficients or hyperparameters.

The initial shared multi-output evaluation remains in `scaling.csv`. The scalar deployment audit is `scaling_scalar.csv` and is the plot source. At low resolution, the smooth bump can require enormous cancelling coefficients: at N32, the small test fit has readout L1 mass around 2.5e10. Equivalent numerical solves/evaluation orders then differ at an absolute scale around 1e-6. We therefore do not demand tiny absolute agreement between ill-conditioned coarse fits. The regression test checks the scalar evaluator exactly against the same coefficients in a standalone scalar MLP, export/reload, and separate fits for well-resolved controls.

## Commands

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_expD59_quill_smooth_scaling.py
OMP_NUM_THREADS=1 .venv/bin/python experiments/expD59_quill_smooth_scaling/run.py
OMP_NUM_THREADS=1 .venv/bin/python experiments/expD59_quill_smooth_scaling/audit_scalar.py
OMP_NUM_THREADS=1 .venv/bin/python experiments/expD59_quill_smooth_scaling/plot.py
```

Outputs are in `results/checkpoint_D_optimizers/expD59_quill_smooth_scaling/`: saved models and diagnostics for each width, evaluation grids and target values, both evaluation CSVs, source manifests, and the PNG/PDF scaling figure. Width counts include all halo neurons. Y is logarithmic relative L2; x is linear width. Curves show every measured point, without smoothing or fitted convergence lines.
