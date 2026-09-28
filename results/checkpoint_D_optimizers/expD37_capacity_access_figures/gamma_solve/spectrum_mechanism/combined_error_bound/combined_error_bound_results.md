# Combined eigenvalue-ratio and residual bound — empirical check

Status: data-obvious numerical results; paper selection pending Sam. 23 September 2026.

## TL;DR

- The full residual bound reaches 1% in all six tested reference/final-gamma pairs. Every target component is retained, including components without a usable eigenvalue bound.
- For gamma doubling, the sufficient step count is 26.63, 5.99, 1.56, 1.36, and 1.25 times the actual finite-kernel count as the final gamma increases from 4 to 64.
- A distant reference loses tightness: the bound for gamma 64 is 106,702 steps from reference gamma 8, versus 22,873 from reference gamma 32. The actual gamma-64 count is 18,272 in both cases.

## Question

Does substituting the finite-network eigenvalue-ratio lower bounds into the exact residual formula produce a useful upper bound on the full target error, rather than only informative bounds on a few eigenvalues?

## Experiment design

Use the note's unchanged geometry: 128 intervals in $[-1,1]$, spacing $h=1/64$, 129 interior centers plus 12 halo centers per side (153 tanh neurons and a bias), and 263 uniformly spaced training samples. The target is $f(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x)$. Normalize the feature matrix and target by $\sqrt{263}$, start the readout at zero, and use $\eta_\gamma=0.5/\lambda_1(K_\gamma)$.

The reference/final gamma pairs are $(2,4),(4,8),(8,16),(16,32),(32,64),(8,64)$. These are comparisons between frozen models, not training trajectories in which gamma changes. Reference gamma is used only to build the old trial eigenspaces for the bound. All curves in a given panel train at its final gamma.

For every numerically resolved rank, evaluate the existing theorem with $C_i=\operatorname{diag}(\lambda_1(\gamma_0),\ldots,\lambda_i(\gamma_0))+U_i^\top\Delta K^{(0)}U_i$, correction $R_i=U_i^\top K_\gamma U_i-C_i$, relative correction $\varepsilon_i=\|C_i^{-1/2}R_iC_i^{-1/2}\|_2$, and denominator $L_\gamma=\max_a\sum_b |(K_\gamma)_{ab}|$. The raw ratio bound is $b_i^{\rm raw}=[1-\varepsilon_i]_+\lambda_{\min}(C_i)/L_\gamma$. Negative, unresolved, or unusable bounds are set to zero. Eigenvalue ordering permits $b_i=\max_{j\ge i}b_j^{\rm raw}$; this strengthening did not change any first-crossing count in this check.

Use the actual new target weights $p_i(\gamma)=|u_i(\gamma)^\top y|^2/\|y\|^2$, including the component orthogonal to the computed rectangular left singular vectors. Compare

$$E_\gamma(n)^2=\sum_i p_i(\gamma)(1-\tfrac12\lambda_i(\gamma)/\lambda_1(\gamma))^{2n}$$

with

$$U_{\gamma;\gamma_0}(n)^2=\sum_i p_i(\gamma)(1-\tfrac12 b_i)^{2n}\ge E_\gamma(n)^2.$$

Compute first crossings of 0.01 by integer bisection with stable `log1p` powers. The search cap is $10^{18}$ steps. No optimization trajectory is executed. This is not the earlier center-integral spectrum forecast: the dashed curves here are evaluated directly from the ratio theorem.

The initial singular value and cutoff gap must exceed $10u\max(m,W+1)s_1$, matching the previous transition checks. The compressed-matrix resolution floor is $64u(\lambda_1(\gamma_0)+\|\Delta K^{(0)}\|_F)$, where $u$ is FP64 machine epsilon. These are numerical resolution rules, not interval-arithmetic error certificates. An unresolved direction retains its full target energy as a nondecaying term. Floating-point inequality discrepancies at two extremely small gamma-64 ranks are also treated as zero bounds; they do not affect the 1% crossing.

**Code & data**

- Script: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/combined_error_bound.py`.
- Results: `data/results.json` (parameters, every rank, counts), `data/results.npz` (ratios, target weights, full curves), `data/validation.json` (independent-driver checks and source hash), relative to this report.
- Figures: `combined_error_bound.png`, `bound_rank_coverage.png`, relative to this report.
- Reproduce: `.venv/bin/python experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/combined_error_bound.py`.
- Independent decomposition check: use `--svd-driver gesdd --output /tmp/combined-error-bound-gesdd`. Default output uses `gesvd`.

## Results

All counts below concern the final gamma and reaching 1% relative training error.

| Reference gamma | Final gamma | Actual finite-kernel count | Sufficient count from bound | Bound / actual |
|---:|---:|---:|---:|---:|
| 2 | 4 | 1,007,616,497,993 | 26,834,664,181,449 | 26.63 |
| 4 | 8 | 19,697,563 | 118,019,960 | 5.99 |
| 8 | 16 | 66,209 | 103,255 | 1.56 |
| 16 | 32 | 29,938 | 40,628 | 1.36 |
| 32 | 64 | 18,272 | 22,873 | 1.25 |
| 8 | 64 | 18,272 | 106,702 | 5.84 |

All retained ratio bounds lie below their corresponding finite-kernel ratios, and all evaluated upper residual curves lie above the actual spectral residual. The direct counts reproduce the prior note's saved counts. The previously saved rank-12, rank-20, and rank-32 bounds reproduce exactly for references/final gammas available in that file.

The independent `gesdd` decomposition gives identical first-crossing bounds in five cases. In the most ill-conditioned case, $2\to4$, the relative difference is $4.59\times10^{-8}$. This supports numerical stability at the requested 1% threshold; it is not interval certification.

### Figures

- **Combined residual curves:** six panels, one reference/final-gamma pair per panel. Horizontal axis is GD step on a common logarithmic scale; vertical axis is relative L2 error on a common logarithmic scale. Blue is the exact-arithmetic spectral formula evaluated numerically for the finite feature matrix. Orange dashed is the upper residual bound. The dotted horizontal line and crossing markers identify 1%.
- **Rank coverage:** the same six comparisons. Horizontal axis is descending eigenvalue rank. The left logarithmic axis compares actual ratios and their lower bounds. The right logarithmic axis shows target energy at that rank and higher. The green $10^{-4}$ line is the squared 1% tolerance. This explains why retaining unresolved directions does not prevent the tested bounds from reaching 1%.

## Additional details

The losses in tightness have identifiable sources. As a diagnostic, replace the relative-correction estimate by the exact minimum over the same old trial subspace, keeping the row-sum denominator. This would give counts of $1.94\times10^{12}$ instead of $2.68\times10^{13}$ for $2\to4$, and 37.7 million instead of 118 million for $4\to8$. Thus the correction inequality is a substantial source of slack in these low-gamma comparisons. This diagnostic is not substituted into the reported theorem bound.

For $8\to64$, the same diagnostic gives 102,233 instead of 106,702 steps. Most slack therefore persists even with the exact restricted quadratic form: the old trial subspaces and denominator, rather than primarily the correction estimate, limit this comparison. Using the exact top eigenvalue with those same trial subspaces still gives 84,584 steps versus 18,272 actual. Resetting the reference to gamma 32 substantially improves the trial spaces.

The upper bound can level off above the actual attainable residual because unresolved modes have zero lower bound. For $2\to4$ that limiting error is 0.00255; for $8\to64$ it is 0.00182. Both are below 1%, but neither supports a machine-precision claim. Tests at 1% therefore do not establish tightness at arbitrarily small tolerances. The full target weights and measured finite-kernel corrections are inputs to this bound; it is not a prediction from gamma alone.

The note's separate necessary-time result still supplies the small-gamma obstruction. For example, its saved gamma-8 band requires at least 3,037,926 steps, while the combined gamma-16 upper bound gives 103,255 sufficient steps. Under the stated normalized learning rates, these evaluated bounds already separate the two acquisition times by more than 29 times. The much tighter actual counts are comparisons, not substituted into that inequality.

## Conclusions

For this target and geometry, the combined ratio/residual bound is nonvacuous at 1% for every tested pair. It is close for doubling at moderate and large gamma, substantially looser at small gamma, and sensitive to how distant the reference gamma is. This supports presenting a combined theorem, with its numerical scope stated explicitly.

## Open questions

- Paper selection: which reference/final-gamma pairs communicate the mechanism and the observed slack most clearly?
- Generality beyond the note's mixed-sine geometry and the 1% tolerance remains untested by this diagnostic.
