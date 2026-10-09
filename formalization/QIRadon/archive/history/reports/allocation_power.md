# General-kappa per-spoke allocation

The older water-filling formula minimizes linear resolution cost and covers the matched kappa=1 class. `AllocationPower.lean` adds the correct global optimality certificates for the general sufficient cost `sum c_i n_i^kappa`, kappa>=1, under weighted error `sum D_i exp(-a_i n_i)` (angular weights are included in D_i).

An interior candidate satisfies

`kappa c_i n_i^(kappa-1) = tau a_i D_i exp(-a_i n_i)`.

`per_spoke_global_min` proves that this candidate globally minimizes the continuous certificate among all feasible allocations. `constrained_per_spoke_global_min` adds baseline resolutions, nonnegative slack multipliers, and exact complementary-slackness conditions. The proof combines derived power and exponential supporting lines; it assumes no optimality conclusion.

`power_cost_rounding` proves upward integer rounding increases this cost by at most 2^kappa when n_i>=1. Existing exponential rounding inequalities show that it improves the error certificate.

The scalar stationarity equation can be written stably as

`(kappa-1) log n + a n = b`.

`existsUnique_spoke_root` proves a unique root on the explicit bracket `[base,max(base,b/a)]` whenever base>=1 and b is at least the left endpoint value. The left side is proved strictly increasing. Thus individual active spoke equations have a bisection-compatible characterization.

Remaining: derive existence/selection of the common multiplier from the total error budget and compose this general nonlinear-cost allocation with the exact finite row-grid cardinalities. The implemented statements must not yet be reported as a complete optimal integer allocation algorithm. The original two-factor M-versus-H optimizer, with its full continuous existence/uniqueness and asymptotics, remains in Allocation.lean.
