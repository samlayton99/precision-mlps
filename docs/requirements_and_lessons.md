# Requirements and lessons (the evidence behind the gate)

**`docs/REQUIREMENTS.md` is the gate. Read that first.** This file is the primary source it was built from: the five requirements as originally written, the three litmus tests, and the eight measured lessons with their measurements attached. It is kept because the gate compresses these and a compression loses the receipts. If the two ever disagree, this file has the measurement and the gate has the ruling.

The guiding spec for the optimizer hardening phase. Every iteration and every proposed change gets judged against the five requirements. The lessons list is deliberately protected: an entry must be measured, load-bearing, and reusable, or it does not go in.

## Requirements for a successful optimizer

1. **First-order, architecture-blind.** Computation cost in Adam's class (gradients and gradient-cost autodiff products only). Zero assumptions about the architecture or model.
2. **Loss functions: MSE is acceptable for now.** The current hardening phase targets mean-squared loss only. Long term we want arbitrary differentiable losses (including regularization terms), but that generalization is deferred, not a gate on current iterations.
3. **Solves least squares to the floor.** On least-squares problems it must reach the direct-solve (SVD) precision floor, ~$10^{-14}$ relative $L_2$.
4. **Generalizable, not finicky.** Works across many problems and setups with one hyperparameter story, like Adam. No special regimes, mode switches, or per-problem tuning, works with batching. Survives Occam's razor.
5. **Scales.** No exploding memory or per-iteration cost as the problem grows.

## Litmus tests

Every candidate iteration must pass all three before it is promoted:

1. **dl_test.** Works well enough on the expD07 dl_test suite (real-data multilayer regression, standard init): stable, sustained descent, competitive order of magnitude -- not required to win.
2. **Batch sizes.** Works well enough across batch sizes on the expD08 batching grid: no blowups, and sustained improvement rather than an early plateau, at every batch level.
3. **Full-batch floor.** Recovers the machine-epsilon lstsq floor at full batch on the toy targets -- the founding result is never given back.

## Lessons learned

1. **No control decision may compare loss values.** The loss is quadratic in the residual, so loss differences become unresolvable in fp64 around relative error $10^{-12}$; the gradient is linear in the residual and stays informative to $10^{-16}$. Any line search, loss-based trust region, or loss-based stopping caps the method around $10^{-10}$. Safeguards must watch gradient norms. (Measured: Armijo froze the method; loss-blind variant reached the floor.)
2. **Recomputing the residual fresh every iteration caps convergence at ~$10^{-10}$.** The subtraction $f(x;\theta) - y$ injects ~$10^{-16}$ rounding noise that is re-rolled each iteration. Carrying the residual as state ($\hat r \leftarrow \hat r + \alpha Jd$) freezes the noise into a one-time harmless offset. (Measured head-to-head on the same frozen system: $2.2\times10^{-10}$ fresh vs $1.2\times10^{-15}$ carried.)
3. **Near-exact step lengths are non-negotiable for conjugacy, and they are available at first-order cost.** The Gauss-Newton curvature along a direction is one JVP ($c = \frac{2}{n}\|Jd\|^2$), giving the exact quadratic step with nothing to tune.
4. **Gradient-history memory below the effective rank is worse than no memory at all.** A 64-vector window on a rank-900 problem underperformed plain CG; memory at or above the rank reached the floor within rank iterations. Any memory scheme is all-or-nothing with respect to the rank.
5. **Deep in a run, the true signal is a ~$10^{-12}$ fraction of the raw gradient.** Thresholds that distrust tiny projected gradients (e.g. bail out when projection removes 99.9999%) silently recreate the $10^{-10}$ stall. Trust gradient components down to $10^{-24}$ relative energy.
6. **A gauge must never act on readings below its own noise floor.** Hit three independent times: loss comparisons (blind below rel $10^{-12}$), the carried-residual drift gauge (frozen-vs-fresh rounding reads as O(1) relative drift near the floor), and the gradient-space gain ratio (predicted changes below gradient rounding read as "bad" and ratcheted damping to $10^{23}$). Every measured control signal needs an explicit noise-floor guard.
7. **Under clean resampling the target is pinned in function space; parameter space slides O(1) along the null space.** Half-data solutions agree functionally at the floor ($\sim10^{-14}$) and are connected by a flat valley, while their readout parameters differ at relative O(1) (basin_study/, 8 splits x 2 targets). Optimizer state that lives in parameter space (memory basis, conjugacy) chases these functionally-meaningless displacements under batching; function-space quantities are resample-stable. With y-noise the target genuinely moves at the noise scale and more data helps by $1/\sqrt{n}$. (Halo caveat -- the null space here is partly a construction artifact of not sampling the halo; see working_notes.md.)
8. **Deep-spectrum precision is a property of the exact operator.** A batch agrees with the full quadratic only to ~$1/\sqrt{b}$, and the tail eigenspace (relative curvature $10^{-13}$) is completely scrambled between batches, so no batch-local state (conjugacy, memory, exact steps) can make progress below the $1/\sqrt{b}$ agreement scale -- measured: minibatch plateaus sit at that scale for slim AND Adam. Below it, only exact aggregation across batches (deterministic sums, not $\sqrt{}$-averaging) extracts the information, and safeguards/variance tricks cannot substitute (SVRG alone: null; guards: stability only). Corollary, also measured: safeguard value is regime-dependent -- the trust window was harmful full-batch and mildly useful under noise; ablations must be run in the target regime.
