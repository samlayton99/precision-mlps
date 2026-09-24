# Section 3.4 evidence

This directory contains compact evidence for the proposed [Section 3.4](../../../docs/section34_slope_acquisition.md), its [proof](../../../docs/section34_slope_appendix.md), and the [experimental protocol](../../../docs/section34_empirical_appendix.md). The study uses width 512 including halos, five paired joint-training seeds, full-horizon cosine and constant schedules, and raw relative output error. The theorem concerns frozen GD; Adam and evolving features are evaluated empirically.

## Evidence and provenance

- `main_5m/bounds/` contains the slope-dependent spectral endpoints, target weights, predicted curves, and the two quadrature resolutions. These predictions use the theorem's rate upper bounds. Actual small eigenvalues are retained separately for numerical checks.
- `main_5m/bound_diagnostics/` contains executed-GD comparisons, every-update extrema, and tolerance crossings. A missing crossing means it was not observed within the executed horizon.
- `main_5m/uniform_access/` contains the full frozen-Adam bandwidth comparison, final-validation recipe selection, compact raw-error summaries, and kernel diagnostics.
- `main_5m/harmonic_diagnostic/` projects executed GD residuals onto the three target harmonics and their orthogonal complement. These physical-frequency projections are distinct from kernel eigenmode projections.
- `attainability/` retains direct-fit coefficients and recomputed errors at three SVD cutoffs. These are numerical witnesses of attainable accuracy, not proofs of an approximation floor.
- `protocol/` records the target, geometry, initialization, original optimizer grids, and the shared frozen-Adam grid expansion.
- `main_2m/` retains the earlier complete comparison and its constant-rate Adam metric control. It must not be substituted for a five-million-update result.

The raw campaign is stored locally at `results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/section34_long_horizon_20260924/` and durably on Runpod at `/workspace/junmiaoh/experiments/precision-mlps/runs/section34_long_horizon_20260924/`. Large raw trajectories remain outside Git. The records identify their input hashes; the source scripts and numerical inputs determine the experiment. Curated NumPy arrays preserve plotted curves and diagnostics without requiring the raw multi-gigabyte Adam traces.

## Reproduce the calculations

Run from the repository root in the project's numerical environment. The experiment entry points live in `experiments/expD36_frozen_gamma_probe/`. `section34_prepare.py` builds the fixed geometry and paired initializations; `adam_joint_probe_run.py`, `adam_feature_probe_run.py`, and `section34_frozen_gd.py` execute joint training, frozen Adam, and frozen GD. Their saved configurations determine the horizon and schedule. GPU launches use `section34.sbatch` under Slurm.

The completed primary frozen-GD calculation can be checked with:

```sh
study_root=results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/section34_long_horizon_20260924/main
python -m experiments.expD36_frozen_gamma_probe.section34_bounds \
  --input "$study_root/base/input.npz" \
  --output "$study_root/h5m/bounds_recheck" --steps 5000000 \
  --gd "$study_root/h5m/frozen_gd"
python -m experiments.expD36_frozen_gamma_probe.section34_bound_diagnostics \
  --bounds "$study_root/h5m/bounds_recheck" \
  --gd "$study_root/h5m/frozen_gd" \
  --output "$study_root/h5m/bound_diagnostics_recheck"
```

`section34_analyze.py` selects one recipe per joint optimizer across all five seeds, reconstructs final errors, and exports the learned-feature assays. `section34_feature_access.py` analyzes the executed readout restarts and uniform references. `section34_joint_metric.py` uses the actual joint Adam moments for its adaptive-Jacobian diagnostic. Plotting does not replace executed trajectories with spectral forecasts.

Compile the review document from the repository root with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tmp/pdfs/section34_review docs/section34_review.tex
```

The paper-facing text is authored directly in Markdown and LaTeX. Analysis programs produce numerical results and figures, not report prose.

## Verification

The four focused test modules for analysis, bounds, bound diagnostics, and joint metrics pass all nine tests. Each training launch also runs independent optimizer-update checks; the joint runner compares analytic gradients with automatic differentiation and verifies that splitting execution preserves the schedule clock. Saved parameters independently reconstruct the reported errors. The theorem comparison checks every executed GD update, and all spectral diagnostics check projection-energy closure and their quadratic forms.

The repository-wide run recorded 868 passed, 17 failed, and 9 skipped tests. All 17 failures were reproduced on the pre-study baseline `f0e7dff` in an isolated checkout. They concern missing external dependencies or datasets, existing numerical thresholds and dtype expectations, and existing module-import collisions. No unrelated fixes were mixed into this study.

Floating-point quadrature refinement and reconstruction checks support the numerical evaluation; they are not directed-rounding certificates. Direct-fit cutoff sensitivity is reported separately from exact mathematical representability. A frozen-GD theorem is not asserted as an Adam or joint-training convergence law.
