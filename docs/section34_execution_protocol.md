# Long-horizon slope-acquisition study

This study tests whether the slope-acquisition limitation seen after 200,000 updates survives substantially longer joint Adam/GD training. The primary figure compares the established frozen-GD output-error lower bound with executed GD, then shows joint-training error and the slope distribution of those same joint trajectories. The primary target and total width are fixed before examining the longer runs.

## Matched experiment

The hidden width is **512 including halos**. With $N=467$ interior intervals, $R=22$ halo centers per side, and $h=2/467$, the uniform reference centers are $-1+jh$, $j=-22,\ldots,489$. The primary target is $[\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)]/\sqrt{21/32}$. A normalized $x^2$ control connects the comparison to the arithmetic primitive. It does not replace the primary target in the figure.

Training uses 2,048 midpoint samples on $[-1,1]$, FP64, full-batch half-MSE, and five paired affine-Xavier initializations. Model selection uses final relative output L2 error on 4,096 midpoints. An 8,192-midpoint grid checks resolution; this grid has been inspected before and is not an untouched test set. Frozen readouts and their optimizer states start at zero. Joint readouts use the paired nonzero initialization.

Joint Adam rates are $0.0002,0.002,0.02,0.05,0.1$; joint GD rates are $0.0002,0.002,0.02,0.05,0.1,0.2,0.5,1$. Both use constant and full-horizon cosine schedules for two million updates. Adam uses $(\beta_1,\beta_2,\epsilon)=(0.9,0.999,10^{-8})$. A boundary winner triggers one factor-three outward rate expansion for that optimizer/schedule. Instability remains in the results. Select one recipe globally per optimizer by the median final validation error across all five seeds; never splice recipes or select separately for each seed.

Every raw training error and RMS slope is retained, with parameters every 10,000 updates. Compare 200k, 600k, and final checkpoints. Inspect the best constant-rate runs separately: over the final fifth, a greater-than-10% median error reduction or 99th-percentile slope increase triggers a five-million-update confirmation, advancing the best two rates per optimizer/schedule. The error reduction uses trailing windows of length 1% of the horizon at 80% and 100% of the horizon. This is a computation-allocation rule, not a convergence certificate. Each new cosine horizon starts from the original initialization.

## Theorem and feature-access checks

For $\lambda=\gamma h\in\{0.03125,0.0625,0.125,0.25\}$, recompute the theorem's finite eigenvalue-ratio upper bounds and target-weighted output-error lower curves on this geometry. Execute actual frozen GD with $\eta=1/(2\mu_{\max})$. Actual eigenvectors supply target projections; actual small eigenvalues supply only a separate exact-spectrum diagnostic. No width-559 prediction or historical tightness number is transferred to this study.

Frozen Adam comparisons use $\lambda\in\{0.03125,0.0625,0.09375,0.125,0.25,0.5,1\}$, rates $10^{-5},10^{-4},10^{-3},0.002,0.01,0.1$, and both schedules. Expand boundary rates once consistently across compared dictionaries. The predeclared uniform reference is $\lambda=0.25$. Freeze selected learned features at 200k, 600k, and the final checkpoint and restart their readouts. At the final checkpoint, also scale both slopes and intercepts by 4 and 16, preserving centers, to test the contribution of slope at fixed learned center placement.

Panel A compares theorem lower bounds and actual GD errors. Panel B shows joint Adam/GD errors and explicitly labeled frozen references. Panel C uses exactly B's joint runs to show RMS and 99th-percentile $h|a_j|$, with seed variability; the uniform slope is a reference, not a universal sufficient threshold. Display reduction preserves extrema and the full horizon. Raw relative output error is primary; tolerance crossings and spectral forecasts are supporting diagnostics.

Correctness checks cover analytic gradients, independent GD/Adam updates, schedule clocks, exact width accounting, target hashes, loss reconstruction, quadrature refinement, resolved spectral enclosure, and dense-grid evaluation. A successful experiment may show delayed acquisition or successful acquisition, a loose bound, or a limitation due to center geometry. The final interpretation follows those observations.

## Execution and writing

Use the existing feature branch and the Runpod Slurm environment, at most two GPUs concurrently and at most six aggregate GPU-hours within a six-hour work window. Preserve raw evidence under the experiment-specific results directory and produce compact review artifacts separately. The window began on 2026-09-24 at 08:23 UTC; the planned end is 14:23 UTC.

The paper draft replaces the obsolete Section 4 argument with Section 3.4 and a single frozen-GD output-error theorem. Its three substantive reviews examine the core argument, reader understanding, and information density in the context of the surrounding Section 3. There are no word or caption caps. The review must change the writing where necessary, including adding explanation when the reasoning needs it. The panels alone do not establish the proposed coarse-error equilibrium mechanism or a general impossibility result for LLM training.

## Decisions after the two-million-update comparison

Neither optimizer met the preset constant-run continuation trigger. Adam's best constant recipe reduced its trailing-window median error by 8.17% over the final fifth, with 1.41% growth in its 99th-percentile slope. GD's corresponding changes were 0.0258% and 3.23%. Neither winning rate lay at a search boundary.

An additional five-million-update primary comparison was nevertheless launched to address the user's concern about training duration. The selected Adam cosine recipe was still improving, with a 20.60% trailing-window error reduction over the final fifth; this does not imply continued slope growth, since its slope statistics decreased. This extra confirmation is a deliberate extension beyond the trigger rule, not a retrospective change to that rule. It advances Adam constant rates $0.02,0.002$, Adam cosine rates $0.05,0.02$, and GD rates $0.2,0.5$ under both schedules. Every run restarts from the paired initialization, and cosine decay spans all five million updates. The two-million-update evidence is retained separately.

The initial frozen-Adam sweep found boundary winners at both ends of its rate grid. Its single outward expansion therefore adds $10^{-5}/3$ and $0.3$ consistently for all seven uniform dictionaries and both schedules. Readout-restart assays use this same expanded grid and a common two-million-update readout budget. They are diagnostic restarts after feature acquisition, not claims about total end-to-end training cost. The five-million-update uniform reference tests the complete expanded grid.
