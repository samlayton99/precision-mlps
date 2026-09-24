# Section 3.4 evidence

The [complete section](../../pdf/section34_spectrum_subsection.pdf) places the spectrum plot beside the rate interpretation and the training figure beside the measured consequences. It combines the spectral bounds and GD consequence into one output-error theorem, with a short proof sketch and the transition to joint feature acquisition. The [full note](../../pdf/section34_spectrum_note.pdf) adds the proof and evaluation appendix. Its [editable subsection](../../../docs/section34_spectrum_subsection.tex), [figure definitions](../../../docs/section34_spectrum_figures.tex), and [LaTeX wrapper](../../../docs/section34_spectrum_note.tex) are separate. The compressed text retains the three-quarter-page budget at unchanged typography; figures and captions are budgeted separately and their final selection remains provisional. Both plots reuse completed computations.

This directory contains compact evidence for the proposed [Section 3.4](../../../docs/section34_slope_acquisition.md), its [proof](../../../docs/section34_slope_appendix.md), and the [experimental protocol](../../../docs/section34_empirical_appendix.md). The study uses width 512 including halos, five paired joint-training seeds, full-horizon cosine and constant schedules, and raw relative output error. The theorem concerns frozen GD; Adam and evolving features are evaluated empirically.

The [compiled review PDF](../../pdf/section34_slope_acquisition_review.pdf) combines the proposed subsection, full proof, and empirical appendices. The [drop-in LaTeX subsection](../../../docs/section34_slope_acquisition.tex) is separate from the review wrapper.

## Evidence and provenance

- [Main three-panel figure](main_5m/section34_three_panel.pdf) compares five million executed updates. Its provenance records the exact analyses and display reduction. `main_5m/joint_analysis/` retains all candidate recipes, the globally selected trajectories, selected parameter arrays, and independently reconstructed endpoint errors.
- `main_5m/bounds/` contains the slope-dependent spectral endpoints, target weights, predicted curves, and the two quadrature resolutions. These predictions use the theorem's rate upper bounds. Actual small eigenvalues are retained separately for numerical checks.
- `main_5m/bound_diagnostics/` contains executed-GD comparisons, every-update extrema, and tolerance crossings. A missing crossing means it was not observed within the executed horizon.
- `main_5m/uniform_access/` contains the full frozen-Adam bandwidth comparison, final-validation recipe selection, compact raw-error summaries, and kernel diagnostics.
- `main_5m/harmonic_diagnostic/` projects executed GD residuals onto the three target harmonics and their orthogonal complement. These physical-frequency projections are distinct from kernel eigenmode projections.
- `main_5m/joint_metrics/` uses the actual joint Adam second moments for readout and full-Jacobian metrics. It reports local target and residual projections, not an Adam convergence prediction.
- `geometry_2m/` contains readout restarts and separate slope/center interventions on two-million-update source models. `main_5m/geometry_diagnostics/` contains the longer-source comparison and direct-fit evidence. Source-training and additional readout budgets remain distinct.
- `quadratic/` contains the separate normalized-$x^2$ control: five-million-update joint curves, two-million-update frozen checks, selected recipes, and the continuation budget.
- `attainability/` retains direct-fit coefficients and recomputed errors at three SVD cutoffs. These are numerical witnesses of attainable accuracy, not proofs of an approximation floor.
- `protocol/` records the target, geometry, initialization, original optimizer grids, and the shared frozen-Adam grid expansion.
- `main_2m/` retains the earlier complete comparison and its constant-rate Adam metric control. It must not be substituted for a five-million-update result.

## What the main figure establishes

Panel A answers whether the gamma-dependent theorem predicts a meaningful output-error obstruction. Its lower curve remains at least 75% of executed GD error at every update across the four bandwidths. At the smallest bandwidth, direct fitting attains $6.24\times10^{-10}$ dense-grid error, but five million GD updates leave $0.21013$ training error against a theorem lower bound of $0.20982$. This is evidence of difficult optimization access to already attainable accuracy.

Panel B answers whether joint training obtains the accuracy available from a supplied geometry within the same update budget. The five-million-update median joint Adam error is $0.001815$, compared with $6.62\times10^{-7}$ for the predeclared uniform frozen-Adam reference; joint GD remains near $0.244$. Both schedules were tested, selection uses a complete recipe across all seeds, and the plot retains instability.

Panel C measures slope acquisition in exactly B's runs. Adam's upper percentile approaches the reference, but only 4–7 of 512 features reach $h|a_j|\ge1/4$. This supports heterogeneous acquisition and a remaining accuracy gap; it does not establish a universal slope threshold for learned dictionaries. Direct fits and interventions show why the joint interpretation must include center placement and the distribution of slopes. Longer training improves Adam, so these curves establish a finite-budget cost, not permanent failure or a proved coarse-error tracking mechanism.

The paper argument therefore connects the construction to an explicit frozen-GD obstruction, then measures the broader geometry-acquisition problem. It motivates supplying accurate primitives directly without claiming that a frozen-feature theorem governs Adam or all language-model training.

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

The final main figure uses the validated compact uniform-Adam traces:

```sh
python -m experiments.expD36_frozen_gamma_probe.section34_figure \
  --analysis "$study_root/h5m/analysis" \
  --bounds "$study_root/h5m/bounds" --gd "$study_root/h5m/frozen_gd" \
  --base "$study_root/base" \
  --frozen-analysis "$study_root/h5m/uniform_access" \
  --output "$study_root/h5m/figure"
```

Compile the review document from the repository root with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tmp/pdfs/section34_review docs/section34_review.tex
```

Compile the compact section with integrated figures, followed by the full proof, using:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=tmp/pdfs/section34_compact docs/section34_spectrum_note.tex
```

The build log reports `SECTION34-APPENDIX-START-PAGE`; preceding pages form the complete section PDF. Its navigation annotations are removed when extracted, since the technical appendix is supplied in the full note. The prose and typography are preserved from the text compression; the figures now appear at their discussion points instead of on a separate evidence page.

The paper-facing text is authored directly in Markdown and LaTeX. Analysis programs produce numerical results and figures, not report prose.

## Verification

The four focused test modules for analysis, bounds, bound diagnostics, and joint metrics pass all nine tests. Each training launch also runs independent optimizer-update checks; the joint runner compares analytic gradients with automatic differentiation and verifies that splitting execution preserves the schedule clock. Saved parameters independently reconstruct the reported errors. The theorem comparison checks every executed GD update, and all spectral diagnostics check projection-energy closure and their quadratic forms.

The repository-wide run recorded 868 passed, 17 failed, and 9 skipped tests. All 17 failures were reproduced on the pre-study baseline `f0e7dff` in an isolated checkout. They concern missing external dependencies or datasets, existing numerical thresholds and dtype expectations, and existing module-import collisions. No unrelated fixes were mixed into this study.

Floating-point quadrature refinement and reconstruction checks support the numerical evaluation; they are not directed-rounding certificates. Direct-fit cutoff sensitivity is reported separately from exact mathematical representability. A frozen-GD theorem is not asserted as an Adam or joint-training convergence law.

All 23 GPU allocations completed successfully, using **5.321 aggregate GPU-hours**, including compilation and I/O. The [Slurm accounting record](protocol/slurm_accounting.json) counts allocation records once and preserves CPU-job outcomes separately. The work completed within the six-hour execution window and six-GPU-hour allowance.
