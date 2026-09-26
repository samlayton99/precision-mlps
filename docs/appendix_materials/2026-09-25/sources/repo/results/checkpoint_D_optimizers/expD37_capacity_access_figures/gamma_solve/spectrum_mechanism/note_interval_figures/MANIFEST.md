# Figures for the two-sided gamma ratio note

These are spectral calculations, not training runs. Geometry: 263 equally spaced samples on [-1,1], 153 tanh centers including the existing halo, spacing h=1/64, and one bias feature. The target is exactly core.target(x, 'mixed'): sin(2πx) + 0.5 sin(6πx) + 0.25 sin(10πx).

- `ratio_intervals.png`: replot of the saved direct_ratio_interval/data.json at ranks 13, 20, 26, and 33, over all 33 saved slopes. Actual is solid, lower dashed, upper dotted. This plot retains the original unpadded FP64 evaluations of the analytic intervals.
- `full_residual_and_steps.png`: complete target-weighted relative residual and necessary/actual/sufficient count to 1%. Uses newly computed full corrected S spectra at gamma 4, 8, 16, 32, 64; it includes a numerical allowance explained below.
- `data.json`: all corrected eigenvalues, finite-feature SVD eigenvalues, ratios, target weights, computed null mass, unresolved mass, residual arrays, count intervals, and validation statistics. Only PNG figures are generated.

The finite-feature SVD supplies actual target weights p_i = |u_i^T y|² / ||y||² and actual ratios. The Fourier bounds do not replace these weights. The lower error uses upper ratio bounds, and the upper error uses lower ratio bounds. All curves use exp(2 n log1p(-rho/2)); no iteration is simulated. Integer crossing counts use monotone bisection, with a search limit of 10^18 steps.

## Analytic bounds and numerical guard

The existing construction gives beta_i, the descending eigenvalues of corrected S. Before the numerical allowance, the ratio lower numerator is [beta_i-delta-tau]_+ and the upper numerator is beta_(i-1)+delta+integral_tail, with the existing scalar normalizations L and ell. rho_1=1 is exact. Interlacing is applied only to valid indices. There are at most 154 nonzero kernel eigenvalues; the remaining 109 structural sample-space zero modes are represented by the appended computed null component. The final eigenvalue lower bound is zero.

For the residual calculation, nu=64*eps64*||S||_2 is subtracted in every lower numerator and added in every upper numerator and in the centered top-eigenvalue allowance used for L. This is a conservative numerical guard checked against independent quadrature, not a rigorous interval certificate. It prevents tiny spurious positive eigenvalues of the corrected matrix from being treated as trustworthy lower rates. The analytic tails alone do not bound FP64 rounding.

Actual ratios greater than 10^-18 define numerically resolved SVD modes. Unresolved positive-mode mass is never silently treated as exact nullspace. The lower residual drops its nonnegative contribution and also drops the computed null component, whose numerical allocation can mix with unresolved tiny singular directions. The upper residual keeps their full energy at zero assigned lower rate. It also keeps every other mode whose guarded lower ratio is zero. The actual spectral calculation retains the computed null component at zero rate; this mass is reported separately from unresolved positive-mode energy. Because near-zero singular vectors are not individually stable, the aggregate unresolved-plus-null mass is the more meaningful numerical diagnostic; these plots do not certify the true asymptotic null floor.

## Count table

| Gamma | Necessary | Actual spectral | Sufficient | Upper-error floor |
|---:|---:|---:|---:|---:|
| 4 | 920,663,607,184 | 1,007,616,497,993 | 3,303,588,843,394 | 0.000585001 |
| 8 | 18,111,765 | 19,697,563 | 37,941,446 | 1.7317e-06 |
| 16 | 60,905 | 66,209 | 97,172 | 1.55713e-07 |
| 32 | 27,547 | 29,938 | 38,036 | 1.38385e-08 |
| 64 | 16,814 | 18,272 | 21,611 | 5.9712e-05 |

All five sufficient curves cross 1%; no unavailable count was replaced by a finite estimate. The lower-error initial value can be slightly below one because unresolved positive mass and computed null mass are dropped conservatively.

## Validation

At every plotted slope, including gamma 4 and 8, refine the existing center quadrature from order 10/padding 20 to order 16/padding 24. Compare the full corrected matrices after orthogonal change of basis. The observed operator discrepancy is below nu. Recompute finite-feature SVD using both LAPACK gesvd and gesdd, compare resolved ratios and weights, and recompute crossings and bands. The guarded lower <= actual <= upper ordering passes strict floating-point comparisons on every resolved mode, also using the alternate SVD driver. Full retained-mode ordering passes with the primary SVD. The full error curves and count intervals also pass ordering checks. The exact count digits describe the current FP64 calculation; use about three significant figures in explanatory prose.

Run from the repository root with `.venv/bin/python experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/note_interval_figures.py`. Existing saved figures and reports are left intact.
