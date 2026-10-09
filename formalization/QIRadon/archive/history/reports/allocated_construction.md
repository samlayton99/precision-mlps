# The allocation solver now controls actual finite rows

Completed 2026-10-09 in `QIRadon/AllocatedConstruction.lean`.

## Construction data

`AnalyticRows` records the actual finite profile family, positive angular weights alpha_i, independently chosen positive Bernstein-ellipse widths delta_i, positive magnitude bounds M_i, and the actual analytic/bounded-profile hypotheses. It assumes no network approximation result. A positive magnitude bound can always be chosen even for a zero profile.

The current theorem permits heterogeneous delta_i and M_i with a common kernel, window scale T0>1, oversampling H, and halo constant Z. The actual preparation order and resulting gamma/spacing/halo grid may differ on every row. The finite weights need not sum to one for this scalar error calculation; positive normalized Gauss–Jacobi weights are one direct application.

`rowSpokeData` constructs the solver's numbers from the proved local-row certificate:

- `kappa=max(1,1/mu+1/nu)`.
- `a_i=localRate(delta_i,T0)`.
- `D_i=2 alpha_i localErrorConstant_i`; the factor two pays for exact centering.
- `c_i=(2H/pi+2HZ+1)(preparationGrowth(T0)+3)^kappa`, the actual halo-grid width prefactor.
- The minimum continuous preparation order is one.

Strict positivity is proved from the supplied positive profile bounds, not assumed for a desired approximation inequality.

## What is proved about the actual rows

`actual_rows_error` invokes the actual local sample/Fourier/kernel construction on every row, centers it at zero, and proves

\[
\left|\sum_i\alpha_i[Q_i(s_i)-Q_i(0)]
-\sum_i\alpha_i[g_i(s_i)-g_i(0)]\right|
\le \sum_i D_i e^{-a_iN_i}
\]

for every tuple with |s_i|≤1. No assumption of this network error appears in the theorem.

`actual_rows_width` counts the actual dependent finite neuron type

`Sigma i, localGrid(N_i)`

and proves its cardinality is at most sum_i c_i N_i^kappa. Every halo center is included. Generic higher-order integration-anchor units remain separately charged; the theorem does not silently count an external polynomial as an ordinary neuron.

## Solver and rounding connection

`optimalResolution` selects the already proved general-power root solver when the requested error is below the baseline error. Otherwise it uses the baseline orders directly. The common multiplier is constructed by `AllocationPowerSolver`; no KKT or optimality condition is supplied by the caller.

`allocated_rows_certificate` sets N_i=ceil(n_i), where n_i are these selected continuous resolutions. It proves together:

1. The **actual centered weighted row sum** meets the requested scalar error budget, uniformly over all allowed evaluation coordinates.
2. The **actual total finite grid cardinality** is at most
   \[
   2^\kappa\sum_i c_i n_i^\kappa.
   \]
3. The continuous sum on the right is no larger than the same cost certificate for any feasible continuous competitor satisfying the same minimum-order constraints and numeric error budget.

`allocated_rounding_error_nonincrease` explicitly proves that rounding never increases the summed error certificates. `allocated_projection_certificate` substitutes fixed unit directions and proves the scalar error uniformly on the input unit ball, with all row choices fixed before the input is quantified. `rowNetwork_zero` proves the centered sum vanishes at the origin; adding the reference value in the output bias gives exact anchoring without more hidden units.

This closes the earlier gap between an abstract optimal allocation formula and the actual constructed row errors/counts. It also applies to kappa>1; it does not reuse the incorrect linear-cost water-filling formula for a nonlinear row cost.

## Exact limits of the claim

The optimized objective is the **proved continuous sufficient cost certificate**, not the exact integer count itself. The factor 2^kappa is a rounding/majorization guarantee, not a claim of exact integer-global optimality or optimality over all neural networks. Angular quadrature approximation remains its separately budgeted term; this module proves the allocated scalar-row portion for fixed finite weights/directions.

The root selection uses the existing exact-real solver definitions and proved finite brackets. Certified finite-precision root acquisition remains part of T8. A common window scale/kernel is used here; heterogeneous profile regularity is already supported, while different per-row window scales would require an additional straightforward parameter-generalization theorem rather than being implicitly claimed.

Verification: `lake build QIRadon.AllocatedConstruction`; principal axiom audit in `scratch/AuditAllocatedConstruction.lean`. No existing shared source file was edited.

Build succeeded (8777-job dependency closure), and all eight audited principal declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`. No proof holes or project axioms were introduced.
