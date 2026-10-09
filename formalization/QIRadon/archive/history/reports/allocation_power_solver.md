# Constructed general-power per-spoke allocation

`QIRadon/AllocationPowerSolver.lean` completes the solver for the continuous
certificate

  minimize Σ_i c_i n_i^κ
  subject to n_i≥base_i and Σ_i D_i exp(-a_i n_i)≤ε,

with κ≥1, positive a_i,c_i,D_i, and base_i≥1. The positive angular weight is
included in D_i. This is certificate optimization, not a global lower bound
on all possible network representations.

For t=log τ and s_i=log(a_i D_i/(κ c_i)), each spoke solves

  (κ-1)log n_i+a_i n_i
    = max((κ-1)log base_i+a_i base_i, t+s_i).

The selected inverse is derived from the earlier unique-root theorem. It is
monotone and Lipschitz with constant 1/a_i. Below its threshold the resolution
is exactly the baseline; above it the inner root lies in the explicit interval
[base_i,max(base_i,(t+s_i)/a_i)]. The offset also has a proved separated-log
formula, avoiding products/divisions of extreme magnitudes in numerical use.

The total error E(t) is continuous and nonincreasing. It decreases strictly
once it drops below the baseline error. For every 0<ε<baselineError, there is
exactly one t with E(t)=ε. Existence is proved using finite explicit endpoints:

- t_low=-1-Σ_i |((κ-1)log base_i+a_i base_i)-s_i|.
- target_i=max(base_i,log(2 M D_i/ε)/a_i), where M is the number of spokes.
- t_high=1+Σ_i |((κ-1)log target_i+a_i target_i)-s_i|.

At t_low all spokes remain at baseline; at t_high total error is at most ε/2.
Thus these endpoints provide a guaranteed outer bisection bracket. Search in
t, not in exp(t), to avoid unnecessary exponential ranges.

The selected allocation's nonnegative slack, complementarity, and stationarity
are proved from these roots. `solver_global_min` then applies the supporting-line
certificate to prove global optimality against every feasible continuous
allocation. No KKT candidate is assumed. If baselines already meet the budget,
`baseline_global_min` proves that no solve is needed.

`rounded_solver_certificate` proves upward-rounded resolutions still meet the
error budget and cost at most 2^κ times the continuous optimum. No exact discrete
optimality is claimed.

`upper_bracket_certificate` shows any outer enclosure [l,u] of the optimum gives
a feasible upper-end allocation, with excess resolution at most (u-l)/a_i per
spoke. Inner roots likewise have explicit monotone brackets. These are real
arithmetic certificates for nested scalar bisection. The formal definitions
select exact real roots; this file does not claim a floating-point implementation
or an audit of numerical evaluation errors.

All headline theorems build warning-free and audit to only propext,
Classical.choice, and Quot.sound. No sorry, admit, or extra axioms are used.
