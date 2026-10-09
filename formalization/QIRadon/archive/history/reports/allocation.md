# Resource-allocation formalization

## Status

`QIRadon/Allocation.lean` compiles with Lean 4.33.0 and the pinned mathlib revision. The allocation arguments below are proved, with no `sorry`, `admit`, custom axioms, or assumed stationary equations replacing the minimization argument. The module is independent of the uncompleted Radon/kernel approximation theorems: it optimizes explicit mathematical certificates and does not prove those certificates for neural networks.

The actual headline optimization claims are proved globally, not merely by differentiating and rearranging. The file uses targeted analysis and tactic imports and currently compiles without warnings.

## Main theorem names and assumptions

All names below have namespace `QIRadon.Allocation`.

### Algebraic optimum

- `algebraicCost_eq_rpow` identifies the objective exactly as `x^(-A) * (epsilon-x)^(-B)`.
- `algebraicSplit_strict_min`: if `A>0`, `B>0`, `epsilon>0`, the point `A*epsilon/(A+B)` lies in `(0,epsilon)` and is a strict global minimizer of that objective there. Proof: actual derivative, strict decrease on the left and strict increase on the right, mean-value theorem.
- `algebraicSplit_ratio`: the optimal angular/scalar budget ratio is `A/B`.
- `algebraic_geometry_ratio`: substitute `A=p/t`, `B=1/s`, with `p,s,t>0`, to obtain `p*s/t`. For geometry take `p=d-1`.

Positive target amplitude factors and resolution prefactors do not alter the minimizer. `mul_cost_preserves_strict_min` verifies this.

### Analytic optimum, including scalar logarithmic exponent kappa

The objective is

`[log(U/x)]^p * [log(V/(epsilon-x))]^kappa`.

`analyticCost_eq_rpow` verifies the identity between this expression and the positive objective actually minimized in Lean.

- `existsUnique_analytic_optimizer`: under `p>0`, `kappa>0`, `epsilon>0`, `log U - log epsilon > 1`, and `log V - log epsilon > 1`, a unique strict global minimizer exists in `(0,epsilon)`. It satisfies

  `kappa*x*log(U/x) = p*(epsilon-x)*log(V/(epsilon-x))`.

  The coefficient inputs in the application are positive. The formal statement uses `logRatio U x := log U - log x`; `logRatio_eq` proves equivalence to the quotient notation when `U,x>0`.

- `existsUnique_analyticBalance_zero`: the stationarity root is constructed by the intermediate value theorem and its uniqueness follows from a positive derivative. Thus root existence is proved, not assumed.
- `analytic_root_strict_min`: derivative signs around that root prove it is the strict global minimum.
- `analytic_stationarity_of_localMin`: separately derives the stationary equation from Fermat's theorem for a genuine local minimum.

The high-accuracy domain is already the regime stated in the approved angular certificate. It ensures both logarithmic resolutions exceed one and makes the balance strictly increasing. No new smoothness or neural-network approximation premise was added to make the optimization proof work.

### Quantitative and asymptotic split

- `analytic_optimizer_fraction_bound`: for any interior balance root and `0<epsilon<1`,

  `|x/epsilon - p/(p+kappa)| <= C / ((p+kappa)*|log epsilon|)`,

  where `C = kappa*|log U| + p*|log V| + kappa + p`.

  The proof controls `theta*log theta` uniformly for `theta` in `(0,1)`; it does not assume the two logarithms have equal leading terms.

- `analytic_optimizer_fraction_tendsto`: any selection of the proven optimizer has angular fraction tending to `p/(p+kappa)` as epsilon tends to zero through positive values.
- `analytic_optimizer_ratio_tendsto`: the angular/scalar budget ratio tends to `p/kappa`.

The selection hypotheses in the last two lemmas specify an interior balance root for sufficiently small epsilon; the separate optimizer-existence theorem supplies those roots. They do not assume the convergence conclusion.

### Admissible intervals and rounding

- `clipped_min_of_deriv_sign`: a decreasing-then-increasing objective is minimized on a closed subinterval by clipping its free optimizer.
- `analytic_clipped_min`: applies this to the analytic objective for any closed budget interval strictly inside `(0,epsilon)`. The theorem includes all boundary cases of clipping within such an interval.
- `exponential_error_rounding` and `stretched_exponential_error_rounding`: upward natural-integer rounding decreases the respective error certificates.
- `linear_cost_rounding`: rounding nonnegative per-spoke resolutions adds less than the sum of positive linear cost multipliers. These multipliers may include linear halo overhead.
- `rounded_product_width`: the rounded tensor width is at most `(q+1)^p*(H+1)`, with `H` the already-counted total spoke width.

These are feasibility and overhead guarantees. Componentwise rounding is NOT claimed to find the exact discrete optimizer; this matches the original theory's qualification.

### Unequal spokes: closed form and active minimum resolutions

For positive amplitude `D_i`, exponential rate `a_i`, linear width multiplier `c_i`, and total budget epsilon:

- `spokeSplit_error` and `spokeSplit_error_sum` prove that the explicit logarithmic formula produces exactly the predicted individual error shares and total budget.
- `spokeSplit_global_min` proves that formula globally minimizes `sum_i c_i*n_i` among all real allocations satisfying the error budget. The proof uses supporting lines of exponentials. It does not presume a KKT condition.

For active baseline constraints, the error-coordinate formula is

`e_i = min(cap_i, t*weight_i)`,

where `weight_i=c_i/a_i` and `cap_i=D_i*exp(-a_i*n0_i)`.

- `exists_waterfill_level`: for positive caps and weights and `0<epsilon<sum caps`, a positive finite level `t` solving `sum e_i=epsilon` exists. The proof gives a finite upper bracket and uses the intermediate value theorem.
- `waterfill_global_min`: this clipped allocation globally minimizes `sum weight_i*log(D_i/e_i)` among all positive error allocations with the same upper caps and total error at most epsilon. This is exactly the linear resolution cost under the error-coordinate substitution.
- `waterfill_resolution_ge_baseline`: the formula translates exactly back to

  `n_i = max(n0_i, log(mu*a_i*D_i/c_i)/a_i)`.

Thus active baselines are not silently dropped. The scalar root is constructed, and the capped objective is globally optimized.

## Connection and scope still required in the complete project

1. The Radon/kernel modules must provide the actual positive constants and rate certificate to instantiate these results. None of this file establishes those approximation rates.
2. The full construction must map actual integer direction order, spoke widths, halo counts, and anchor neurons to the cost used here. The rounding lemmas are available but that end-to-end accounting is not declared complete by this module.
3. `p=d-1>0` means the nontrivial angular allocation results concern `d>=2`; `d=1` has no angular-versus-scalar allocation problem.
4. The explicit unequal-spoke formula is exact for linear per-spoke costs including linear halo multipliers. A general nonlinear halo cost needs its own optimization statement. No such more general formula is claimed here.
5. Exact global optimization over discontinuous integer ceiling costs is not claimed. The proved continuous optimum and rounding overhead are the guarantees.
6. The lambda-rule work is excluded, as requested.

## Verification

Command: `lake env lean QIRadon/Allocation.lean`.

The selected headline theorems were also inspected using Lean's `#print axioms`. They use only Lean/mathlib foundational axioms (`propext`, `Classical.choice`, `Quot.sound`); no `sorryAx` or additional axioms occur.
