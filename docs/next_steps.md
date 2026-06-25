# Where this stands and what to do next

A strategy memo (Claude's read after the full reorganization). For findings, see `results/results.md`; for the roadmap, `docs/future_experiments.md`.

## The story is almost complete

The diagnostic work converted the paper's vague "optimization is the bottleneck" into a precise mechanism:

- **The readout is solved.** Convex, lstsq hits the floor, and the "weight blowup" violation is just which null-space representative you land on (expA03). One of the paper's three violations is essentially dissolved.
- **Geometry is the whole game, and geometry is mostly $\gamma$.** $\lambda^*\approx 0.25$ i.e. $\gamma=O(N)$ is the law (expC03); right-$\gamma$ + even random centers + lstsq already hits the floor on smooth targets (the $\gamma$-init finding). Center uniformity is second-order.
- **The one real barrier is $\gamma$-scale, and it is exactly what gradient descent cannot reach** (vanishing $\mathrm{sech}^2$ envelope). Adam stalls even on the convex readout (expD01) and cannot grow $\gamma$ from a standard start.
- **The gap is already closeable:** right-$\gamma$ init -> train -> final lstsq refit recovers the construction floor (expD02 wins #2/#3).

Missing piece: reach machine precision **from a generic random init** (no construction handed in), on the full success-criterion protocol.

## The experiment that closes the paper

Not more diagnostics -- the decisive method test. Two routes, same hypothesis ("raw coordinates are the wrong variables"); run both:

1. **VarPro (expD04) -- cleanest shot.** Eliminate the readout exactly (lstsq each step), optimize only the nonlinear geometry with Gauss-Newton/LM. The nonlinear problem is tiny -- essentially one bandwidth $\lambda$ plus small center deltas. Sidesteps the readout barrier and attacks $\gamma$ with a second-order method immune to the vanishing gradient.
2. **log-$\gamma$ reparameterization (expD03).** Does $\gamma=\exp(\eta)$ let first-order descent grow $\gamma$ where raw coordinates cannot? Pair with the final lstsq refit that already works.

If either reaches the floor from random init across the width ladder, the paper's thesis becomes one sentence: *QI theory reveals the right coordinate system; in those coordinates, train + final solve reaches machine precision.*

## Resolve this first (cheap, decides the method's design)

A genuine tension: expC03 says $\gamma=O(N)$ is required; the $\gamma$-init finding says right-$\gamma$ + random centers + lstsq hits the floor; but expD02/expC04 show random/scaled centers **decay with $N$ while uniform holds**. So likely: **coverage + right $\gamma$ gets you into the basin; uniformity holds the floor as $N$ grows.** Pin this down -- it decides whether the optimizer must learn *uniform centers* or *just $\gamma$*. If it's just $\gamma$, the problem collapses to a one-parameter barrier and the story is beautiful.

## For paper completeness (low cost, high value)

- **The linear-then-floor scaling law (expB02)** could be the quantitative backbone: if the decades-to-floor grow like $\log(1/\varepsilon)$, that *is* the success-criterion form, and it may be provable. The fixed-$\lambda=0.25$ rerun already confirms the shape isn't a bandwidth-selection artifact.
- **Coefficient-magnitude study (expA03 follow-up)** -- cheap, formally kills the "weight blowup" violation.
- **2D (expE01)** already gives the "generalizes beyond 1D" claim. Don't over-invest in the runge/curvature-clustering thread yet.

## Caution

The expD02 wins are **single-seed and fp32** -- enough to motivate, not to claim. Before "we found the method" goes in writing, it must clear the full protocol (3-5 seeds, width ladder, 6 targets, fp64 eval, no construction init). And resist the pull to keep mapping the landscape; the diagnostic phase has done its job. Point everything at the kill shot.

## Order of operations

1. Resolve $\gamma$-only vs $\gamma$+uniformity (cheap, design-critical).
2. VarPro from random init + final lstsq -- the decisive test.
3. log-$\gamma$ in parallel.
4. Scaling-law characterization for the writeup.
