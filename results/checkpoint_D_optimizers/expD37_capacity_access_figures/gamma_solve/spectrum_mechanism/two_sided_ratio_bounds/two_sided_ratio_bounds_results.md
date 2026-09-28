# Two-sided finite-gamma eigenvalue ratios — draft-pending-Sam

## TL;DR

- Section 3's lower bound is mathematically correct: it proves improvement when the bound exceeds the old ratio. Small-gamma delay needs an **upper** bound on the relevant ratios instead.
- The same kernel decomposition supplies a valid upper bound using the full old trailing sample space. For reference 8 and new gamma 16, rank 32 lies between `3.40e-6` and `7.83e-6`; its actual ratio is `6.30e-6`.
- Backward comparisons from gamma 8 or 64 are loose at gamma 4. A forward comparison from 2 to 4 is useful: the rank-26 upper bound is `8.34e-12`, about four times the actual `2.09e-12`.

## Question

Can the note's decomposition bound the normalized eigenvalues from both sides, with the correct min–max and denominator directions? Does that upper bound resolve small ratios at gamma 4?

## Experiment design

Use the note's uniform geometry: 128 interior intervals, 153 tanh neurons plus a bias, and 263 samples. The feature matrix is

\[
B_{aj}=m^{-1/2}\tanh(\gamma(x_a-c_j)),\qquad B_{a0}=m^{-1/2},
\qquad K_\gamma=B_\gamma B_\gamma^\top.
\]

Keep precisely the decomposition from the note,

\[
K_\gamma=K_\gamma^{(0)}+E_\gamma,\qquad
(K_\gamma^{(0)})_{ab}=(W+1)/m-2d_{ab}\coth(\gamma d_{ab})/(hm).
\]

At zero separation, use \(d\coth(\gamma d)=1/\gamma\). For a reference \(\gamma_a\), let \(\Delta_0=K_\gamma^{(0)}-K_{\gamma_a}^{(0)}\) and \(\Delta E=E_\gamma-E_{\gamma_a}\). Its explicit gamma dependence remains

\[
q^\top\Delta_0q=\frac{2}{\pi hm}\int_{\mathbb R}
\frac{M_\gamma(\omega)^2-M_{\gamma_a}(\omega)^2}{\omega^2}
\left|\sum_bq_be^{-i\omega x_b}\right|^2d\omega,
\quad M_\gamma(\omega)=\frac{z}{\sinh z},\quad z=\frac{\pi|\omega|}{2\gamma}.
\]

The identity holds in either comparison direction. The change is negative semidefinite when gamma decreases.

Let \(U_i\) be the computed old leading \(i\)-dimensional trial basis, and let \(V_i\) span the orthogonal complement of the first \(i-1\) old vectors. Thus \(V_i\) has **\(m-i+1\) columns**, including the 109 sample-space directions omitted by the thin feature SVD. Define

\[
C_i=U_i^\top(K_{\gamma_a}+\Delta_0)U_i,\quad R_i=U_i^\top\Delta E U_i,
\qquad D_i=V_i^\top(K_{\gamma_a}+\Delta_0)V_i,\quad S_i=V_i^\top\Delta E V_i.
\]

Project the old kernel explicitly rather than assuming that numerically computed vectors diagonalize it exactly. These formulas therefore apply to any orthonormal trial bases, including uncertain reference singular directions.

For the lower bound, retain the note's \(\varepsilon_i=\|C_i^{-1/2}R_iC_i^{-1/2}\|_2\), whenever \(C_i\succ0\). Use zero if this positive-definiteness condition fails or cannot be resolved in FP64. Let

\[
L_\gamma=\max_a\sum_b|(K_\gamma)_{ab}|\ge\lambda_1(\gamma),\qquad
d_\gamma=\|B_\gamma^\top u_{1,a}\|^2\le\lambda_1(\gamma),
\]

where \(u_{1,a}\) is a unit old leading vector and \(d_\gamma>0\). The two bounds are

\[
\boxed{\quad
\frac{[1-\varepsilon_i]_+\lambda_{\min}(C_i)}{L_\gamma}
\ \le\ \frac{\lambda_i(\gamma)}{\lambda_1(\gamma)}
\ \le\ \min\left\{1,\frac{\lambda_{\max}(D_i)+\lambda_{\max}(S_i)}{d_\gamma}\right\}.
\quad}
\]

The upper inequality follows from

\[
\lambda_i(K_\gamma)\le\lambda_{\max}(V_i^\top K_\gamma V_i)
=\lambda_{\max}(D_i+S_i)
\le\lambda_{\max}(D_i)+\lambda_{\max}(S_i).
\]

Neither \(D_i\) nor \(S_i\) must be positive definite. Dividing by a **lower** bound on the largest eigenvalue preserves the desired upper ratio bound. The cap at 1 is the trivial ratio bound; it is inactive in this sweep.

The main sweep uses fixed references 8 and 64, gamma 4–64, and ranks 12, 20, 32. Two additional forward comparisons use 2→4 and 4→8, also checking rank 26. No training or polynomial approximation is used. Actual spectra come from rectangular SVD and are used only for comparison and validation. The corrections are evaluated from the new finite feature kernel; this is an evaluated theorem bound, not a prediction from gamma alone.

**Code & data**

- Code: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/two_sided_ratio_bounds.py`
- Output directory: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/two_sided_ratio_bounds/`
- Tidy data: `two_sided_ratio_bounds.csv`, `forward_pair_checks.csv`; full metadata: `two_sided_ratio_bounds.json`; actual spectra: `actual_spectra.npz`.
- Figures: `two_sided_ratio_bounds.png`, `bound_tightness.png`; driver sensitivity: `forward_pair_svd_driver_check.json`.

## Results

For reference 8, the upper bound is within 3–25% of the actual ratios at gamma 16 across the three plotted ranks. At gamma 64 it is within 3–13%. The original lower-bound value for 8→16 at rank 32 is reproduced.

At gamma 4, fixed-reference bounds are much looser. At rank 32, reference 8 gives upper `1.36e-9` against actual `3.37e-15`; reference 64 gives upper `1.07e-5`. The lower bounds are zero because the leading main compression is indefinite. This does not invalidate the upper theorem.

The short forward comparisons show that the backward-reference failure is not an unavoidable limitation of the decomposition:

| Comparison | Rank | Lower bound | Actual ratio | Upper bound | Upper / actual |
|---|---:|---:|---:|---:|---:|
| 2→4 | 12 | 6.38e-7 | 6.55e-6 | 1.05e-5 | 1.60 |
| 2→4 | 20 | 5.29e-11 | 1.28e-9 | 3.41e-9 | 2.66 |
| 2→4 | 26 | 7.85e-14 | 2.09e-12 | 8.34e-12 | 3.99 |
| 2→4 | 32 | 0, unresolved | 3.37e-15 | 2.05e-14 | 6.08 |
| 4→8 | 12 | 2.14e-4 | 4.25e-4 | 5.00e-4 | 1.18 |
| 4→8 | 20 | 1.27e-6 | 4.48e-6 | 6.60e-6 | 1.47 |
| 4→8 | 26 | 2.66e-8 | 1.49e-7 | 2.59e-7 | 1.74 |
| 4→8 | 32 | 5.53e-10 | 4.93e-9 | 1.01e-8 | 2.06 |

The rank-26 upper bound also bounds every ratio at rank 26 and above by eigenvalue ordering. Turning that cutoff into a necessary training time still requires target energy in those positive modes; this experiment does not replace that target-dependent calculation.

### Figures

- **Two-sided ratios:** columns are ranks; rows use fixed references 8 and 64. Gamma is horizontal; normalized eigenvalues are on a log vertical scale. Actual values are solid blue, lower bounds dashed green, and upper bounds dotted orange. Downward green markers explicitly represent zero lower bounds at the plotting floor. The vertical gray line marks the reference.
- **Bound tightness:** the same panels divide bounds by actual ratios. Gray curves additionally show the exact trailing compression, solely to diagnose subspace loss. For 8→4 at rank 32, the reference-space choice alone loses a factor of about 72,400; splitting main and correction loses another factor of 5.56. The denominator loses only 0.012%.

## Additional details

All sampled lower/actual/upper comparisons and intermediate min–max inequalities passed. Full-basis orthogonality error was below `8e-15`. Independent Fourier quadrature agreed with the spatial change within `6e-12` on nonzero-mean test vectors in both comparison directions. Relative-correction norms were cross-checked with generalized eigenvalue solves.

For 2→4, changing the reference SVD driver from `gesvd` to `gesdd` changes the rank-26 upper bound by approximately `1.8e-7` relative and the rank-32 upper bound by `7.6e-5` relative. Rank 32 remains near the FP64 resolution scale for the compressed main matrix and its lower bound is explicitly unresolved. Every numerical value here is an FP64 evaluation, not an interval certificate.

## Conclusions

Section 3 proves the improvement direction correctly; it does not itself establish the small-gamma obstruction used in Section 4. A complementary upper theorem from the same Fourier decomposition does resolve small ratios in the 2→4 comparison, while distant backward references lose many orders through trial-space mismatch.

## Open questions

How much necessary delay does the rank-26 upper cutoff establish when combined with the target's positive-tail energy? Can corrections and reference spaces be controlled without measuring the new finite kernel?
