# Actual joint regularity of dominated parameter integrals

`QIRadon/Radon/ParameterIntegral.lean` supplies the direct differentiation route requested after identifying that the pinned mainline mathlib only has holomorphic-to-analytic conversion for one complex parameter. It proves actual joint complex C∞ regularity of the integral; it does not assume analyticity or differentiability of the integral itself.

Namespace: `QIRadon.Radon.ParameterIntegral`.

## Primary hypotheses

- A finite-dimensional complex normed parameter space E, including `Fin l → ℂ`.
- An arbitrary measure μ on an index space A.
- A scalar complex integrand `F : E → A → ℂ` and an open parameter set U.
- Every parameter slice `F z` is ae strongly measurable for z in U.
- Almost every fiber `z ↦ F z a` is jointly complex C∞ on U. The analytic-fiber corollary derives this from actual `AnalyticOnNhd` fibers.
- A single integrable real envelope M with `norm(F z a)≤M a` for ae a and every z in U.

Under those hypotheses `smoothOn_integral_of_dominated` proves

`ContDiffOn ℂ ∞ (fun z => integral_a F z a) U`.

`smoothOn_integral_of_analytic_fibers` provides the direct analytic-fiber input interface.

The same result applies locally on any open neighborhood with its own integrable envelope. The proof itself uses progressively smaller balls, not a global bound on an unbounded parameter space.

## What is derived

`fderiv_apply_aestronglyMeasurable` derives measurability of every fixed-vector derivative evaluation from the scalar line difference-quotient theorem. The line stays in the supplied parameter neighborhood; no evaluations outside that neighborhood are used.

`fderiv_aestronglyMeasurable` reconstructs the actual bounded linear derivative from its evaluations on a finite basis. The finite coordinate sum is measurable. The complete derivative operator is therefore measurable, rather than assumed measurable.

`local_fderiv_bound` derives an operator bound `norm(DF(z))≤2M/r` on an inner ball from the actual function-value bound on a ball twice as large. It uses the complex Schwarz estimate after centering the image at F(z). The factor two is confined to this qualitative domination argument.

`hasFDerivAt_integral_of_dominated` proves the actual Fréchet derivative formula on arbitrary open U:

`D(integral_a F(z,a)) = integral_a D_z F(z,a)`.

`contDiffOn_integral_of_dominated n` proves every finite order by induction. In finite dimension, smoothness of a bounded-linear-operator-valued map reduces to smoothness of all its fixed-vector evaluations. Each such evaluation is another scalar parameter integral. The same local derivative-envelope argument supplies its integrable majorant. Thus no all-orders derivative bounds are placed among the primary hypotheses.

## Connection to the sharp angular bound

`QIRadon/Quadrature/MixedCauchy.lean` now additionally contains

- `mixedPartial_smooth`,
- `mixedPartial_bound_of_smooth`,
- `mixedPartial_zero_bound_of_smooth`,
- `allCoordinatePartial_zero_bound_of_smooth`.

These consume the actual joint complex C∞ result. Each successive coordinate partial is differentiable, so its restriction to one complex line satisfies the existing one-variable Cauchy theorem. The sharp bound remains

`norm(partial_1 ... partial_l F(0)) ≤ M (2l/δ)^l`

on the open domain `sum_i norm(z_i)<δ`. There is no extra factorial or dimension factor in this quantitative estimate. The qualitative `2M/r` envelope used to justify differentiation under an integral does not replace this sharp estimate.

This route establishes the needed mixed-derivative conclusion directly. It does not claim a new general multivariate `AnalyticOnNhd` theorem for the integral; the coordinator is separately auditing the external SCV library for that stronger library predicate.

## Verification

`lake build QIRadon.Radon.ParameterIntegral QIRadon.Quadrature.MixedCauchy` verifies both modules. The primary measurability, actual derivative, all-orders smoothness, and sharp smooth mixed-Cauchy theorems are inspected with `#print axioms`; only standard foundational axioms are permitted. No `sorry`, custom axiom, or assumed integral-regularity conclusion is introduced.
