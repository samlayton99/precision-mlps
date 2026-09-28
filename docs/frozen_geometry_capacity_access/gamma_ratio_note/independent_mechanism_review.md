# Independent review of the direct Fourier obstruction

23 September 2026. Scope: the direct upper-bound report, the corrected interval report, the older four-part note, and their implementation. No training or new parameter sweep was run. I checked the inequalities independently and read the saved numerical results; the reported numerical enclosures remain floating-point evaluations.

## Verdict

**The direct positive Fourier integral, finite-lattice upper comparison, and target-mass inequality already form a coherent explanation and a useful necessary-time result at small gamma.** An additional closed scalar formula for every eigenvalue rank is not necessary for that claim. Rejecting the argument merely because the integral spectrum is evaluated numerically would be an overstatement.

There is a narrower limitation that must remain visible: positivity of the continuum change proves that its ordered, unnormalized eigenvalues increase with gamma. It does not prove that all normalized finite-network ratios increase continuously, or that every target trains faster. The finite corrections, leading eigenvalue, and target projections all vary. The current material does not establish that stronger universal claim, and the small-gamma obstruction does not need it.

For the studied geometry, the uncorrected upper comparison is already quantitatively useful at gamma 4, 8, and 16. The elaborate corrected construction is useful for tighter two-sided intervals and for gamma 32–64. It should not become a prerequisite for the small-gamma theorem.

## What the four moves establish

The setup fixes samples, lattice centers, and a common positive slope. With the sample-normalized feature matrix \(B_\gamma\), kernel \(K_\gamma=B_\gamma B_\gamma^\top\), zero readout initialization, and learning rate \(1/(2\lambda_1(K_\gamma))\), the exact relative residual is

\[
E_\gamma(n)^2=\sum_j p_j(\gamma)
\left(1-\frac{\lambda_j(K_\gamma)}{2\lambda_1(K_\gamma)}\right)^{2n},
\qquad
p_j(\gamma)=\frac{|u_j(\gamma)^\top y|^2}{\|y\|^2}.
\]

This first move is complete. It identifies both the relevant ratios and the target dependence.

For a sample vector \(u\) whose entries sum to zero, the whole-line center integral has quadratic form

\[
\frac{2}{\pi hm}\int_{\mathbb R}
\frac{M_\gamma(\omega)^2}{\omega^2}
\left|\sum_a u_a e^{-i\omega x_a}\right|^2d\omega,
\qquad
M_\gamma(\omega)=\frac{z}{\sinh z},\quad
z=\frac{\pi|\omega|}{2\gamma}.
\]

The sample factor and geometry do not change with gamma. For \(|\omega|/\gamma\) large,

\[
M_\gamma(\omega)^2\sim
\left(\frac{\pi|\omega|}{\gamma}\right)^2
e^{-\pi|\omega|/\gamma}.
\]

The exact multiplier therefore gives an explicit suppression law. Increasing gamma increases the integral for every fixed zero-mean vector. This simultaneous quadratic-form inequality, followed by min–max, proves that every ordered eigenvalue \(\alpha_j(\gamma)\) of the integral compression \(H_\gamma\) is nondecreasing. Eigenvector rotation creates no gap in this argument. No identification of rank with a Fourier frequency is needed or justified. This second move is complete.

The essential third move uses positivity in the center coordinate: extending the finite center sum to the entire lattice adds nonnegative squares. Analytic Poisson-error control then yields

\[
Q^\top K_\gamma Q\preceq H_\gamma+\delta_\gamma I,
\]

where \(Q\) is an orthonormal basis for the zero-mean sample subspace and \(\delta_\gamma\) is the displayed analytic strip bound. Compression interlacing and a scalar lower bound \(\ell_\gamma\le\lambda_1(K_\gamma)\) give

\[
\frac{\lambda_i(K_\gamma)}{\lambda_1(K_\gamma)}
\le b_i(\gamma):=
\min\left\{1,\frac{\alpha_{i-1}(\gamma)+\delta_\gamma}{\ell_\gamma}\right\},
\qquad i\ge2.
\]

This is an absolute upper bound at one gamma. It uses no reference gamma and no measured difference from the actual finite kernel. The one-rank shift is required. Dividing by a lower bound on the largest eigenvalue is the correct direction. This third move is complete as a computable spectral inequality, with useful small-gamma evaluations.

For the fourth move, define the positive-spectrum tail mass

\[
P_i(\gamma)=\sum_{j\ge i:\lambda_j(K_\gamma)>0}p_j(\gamma).
\]

The preceding upper ratio bound gives

\[
E_\gamma(n)^2\ge P_i(\gamma)(1-b_i(\gamma)/2)^{2n}.
\]

Consequently, when \(P_i(\gamma)>\epsilon^2\), reaching error at most \(\epsilon\) requires

\[
n\ge
\frac{\log(\sqrt{P_i(\gamma)}/\epsilon)}
{-\log(1-b_i(\gamma)/2)}.
\]

Equivalently, a budget \(N\) is insufficient if
\(P_i(\gamma)(1-b_i(\gamma)/2)^{2N}>\epsilon^2\).
An integer step count uses the appropriate ceiling. Maximizing over cutoffs is valid. This fourth move is complete conditional on the stated target mass; the current application obtains that mass from the finite-feature SVD.

The saved results demonstrate that the direct upper bound has practical strength:

| Gamma | Necessary steps from upper bound | Finite spectral crossing | Bound / crossing |
|---:|---:|---:|---:|
| 4 | \(4.52997\times10^{11}\) | \(1.00762\times10^{12}\) | 0.450 |
| 8 | \(1.14978\times10^7\) | 19,697,563 | 0.584 |
| 16 | 46,249.3 | 66,209 | 0.699 |

These are substantial necessary delays with useful tightness. They are not merely qualitative vanishing-eigenvalue statements. The spectral crossing is calculated, not an executed training trajectory.

The saved rank-26 calculation also locates the change quantitatively. At gamma 4, 8, and 16, the integral numerator \(\alpha_{25}\) is approximately \(1.95\times10^{-10}\), \(1.08\times10^{-5}\), and \(2.74\times10^{-3}\), while the scalar normalization lower bound changes only from 63.20 to 66.82 to 67.94. The respective lattice allowances are \(1.31\times10^{-59}\), \(3.21\times10^{-26}\), and \(5.68\times10^{-10}\). In these evaluations, the many-order change in the useful bound comes from the positive integral spectrum; the normalization and analytic sampling allowance do not account for it.

## What numerical diagonalization does and does not weaken

Numerically evaluating \(\alpha_j(\gamma)\) does not undo the explanation. The Fourier representation supplies information that an unexplained finite-kernel SVD does not: the exact dependence on gamma, simultaneous increase of every integral quadratic form, and a directional comparison to the actual finite kernel with an analytic sampling allowance. Numerical eigenvalues quantify this analytically identified mechanism on the chosen geometry.

The corrected matrix

\[
S_\gamma=H_\gamma+\sum_{k=1}^{p}A_{\gamma,k}-T_F
\]

is indeed an accurately reconstructed version of the centered finite kernel when many corrections are retained. Its eigendecomposition alone would provide little mechanistic explanation. Its legitimate role is to transfer the already established integral mechanism to finite geometry and obtain useful enclosures where a crude global error bound fails. The alias terms and exterior-center subtraction are derived from the geometry; their remainders are bounded analytically. They are not fitted from a difference against the actual finite matrix.

Thus both of the following statements are true: the corrected calculation closely reconstructs the finite problem, and the complete Fourier-plus-comparison argument has explanatory content. They conflict only if the reconstruction itself is presented as a proof of monotone finite normalized rates.

The paper should be explicit that the theorem is computationally evaluable rather than a closed rank/gamma law. It should also acknowledge that obtaining target mass from the actual SVD does not provide a spectrum-free training-time predictor. Neither fact invalidates the claimed mechanism or the resulting conditional bound.

## Gamma comparisons: sufficient evidence and stronger claims

The absolute two-sided intervals already give rigorous *formulas* for comparing any two evaluated gamma values. If \(l_i(\gamma)\le r_i(\gamma)\le u_i(\gamma)\), where \(r_i=\lambda_i/\lambda_1\), then

\[
\frac{r_i(\gamma_b)}{r_i(\gamma_a)}
\ge\frac{l_i(\gamma_b)}{u_i(\gamma_a)}.
\]

The two intervals are constructed independently; this corollary does not insert a reference gamma into either construction. In the saved rank-26 results, its lower factors are

| Gamma values | Lower factor for ratio increase |
|---|---:|
| 4 to 8 | 34,121.2 |
| 8 to 16 | 156.616 |
| 16 to 32 | 7.41571 |
| 32 to 64 | 1.66419 |

These calculations establish large finite-point separation through the proved inequalities, subject to their stated floating-point status. They do not prove monotonicity between the evaluated points. They are useful supporting evidence; the absolute \(b_i(\gamma)\) bound should remain the main theorem.

There is also a simple monotone *budget envelope*, if that particular statement is wanted. Fix a strip angle, a rank cutoff, a uniform leading-eigenvalue lower bound \(\ell_*>0\), and a target-mass lower bound \(P_i(\gamma)\ge P_*>\epsilon^2\) over the gamma range of interest. The bias always permits \(\ell_*=1\). For a slope cap \(\Gamma\), define

\[
b_i^*(\Gamma)=\min\left\{1,
\frac{\alpha_{i-1}(\Gamma)+\delta_\Gamma}{\ell_*}\right\}.
\]

Both terms in its numerator are nondecreasing for fixed strip angle. Every frozen slope \(0<\gamma\le\Gamma\) in the stated range therefore obeys the necessary-time lower bound

\[
n\ge
\frac{\log(\sqrt{P_*}/\epsilon)}
{-\log(1-b_i^*(\Gamma)/2)}.
\]

This bound is nonincreasing in the allowed cap \(\Gamma\). Its common target-mass hypothesis is additional information; the current pointwise target calculations do not prove it over a continuous interval. Replacing the sharp normalization by one can also weaken the numerical bound substantially. This optional corollary should not be used to claim that the sharp pointwise bound, or actual training time, is universally monotone.

A smaller necessary-time lower bound means that this obstruction has weakened. It does not by itself prove successful training within the smaller budget. The observed reduction in actual spectral crossing time is separate numerical evidence; a theorem asserting fast attainment would need an upper error bound covering the entire target, including any nullspace component.

## Smallest defensible paper revision

Use the direct upper comparison as the main third move, followed immediately by the necessary-time inequality. Keep the positive integral as the mechanism and identify the target mass as measured input to the example. Put the detailed corrected interval construction after this central argument or in an appendix; use its absolute intervals to document finite-point ratio increases when helpful.

No new scalar formula is needed to repair a logical hole in that claim. The existing Fourier-cutoff scalar route was already tested and found too loose: the saved review reports about a factor 1,200 at gamma 8, rank 32, and about four orders of magnitude at rank 48. Requiring that route now would trade away the quantitative result without resolving a necessary objection.

There is presently no validated, tight closed scalar formula in gamma and rank. The current successful bound retains a numerical integral-spectrum calculation.

The remaining distinction is numerical certification. The analytic inequalities are exact-arithmetic results. Quadrature refinement and independent SVD drivers support the displayed values, but analytic alias and exterior-tail allowances do not enclose floating-point and quadrature errors. A paper can report the values as checked numerical evaluations of a theorem. If it claims an interval-certified numerical impossibility at a stated step count, it needs enclosures for the relevant integral eigenvalue, normalization, and target mass. That is a certification task, not a missing explanation of gamma.

I found no substantive algebraic defect in the reviewed direct comparison or corrected-interval formulas. The old polished note's ratio lower bound was a different statement and could not itself supply a necessary delay; the direct upper result fixes that direction. The unsupported statement would be universal monotonicity of actual finite normalized ratios or full target hitting times. Removing that statement leaves a complete small-gamma obstruction with an explicit Fourier mechanism and useful numerical strength.

## Sources checked

- [Direct upper-bound report](../../../results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/direct_ratio_upper_bound/REPORT.md), including the lattice proof, positive target-tail formula, and saved `data.json`.
- [Corrected interval report](../../../results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/direct_ratio_interval/direct_ratio_interval_results.md), especially the matrix inequalities, Poisson coefficients, scalar normalization, and numerical limitations; endpoint factors above were recomputed from its saved `data.json`.
- [Older four-part note](gamma_ratio_note.md), especially the residual formula, anchored lower-ratio theorem, and Appendix D's inequality-direction clarification.
- [Direct upper implementation](../../../experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/direct_ratio_upper_bound.py) and [corrected interval implementation](../../../experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/direct_ratio_interval.py).
- [Earlier Fourier-cutoff review](../../../results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/structural_barrier_review.md), including its recorded numerical failure of tightness.
