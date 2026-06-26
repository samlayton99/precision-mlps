# Checkpoint C -- The admissible geometry, consolidated

**Status:** consolidation of expC01-C06. C01/C02/C03/C06 approved by Sam; C04 data-obvious; C05 results data-obvious, its conclusions/interpretation proposed.

## What this checkpoint established

The theory proves a machine-precision parameter point exists. The inner layer that controls it is $2N$-dimensional ($N$ weights $w_i$, $N$ biases $b_i$; the readout is solved exactly and is not the issue). Checkpoint C maps the neighborhood of that point: **which of the $2N$ coordinates are actually free, how precision decays as you leave the point, and why other configurations fail.** Two headline results:

1. **The $2N$ degrees of freedom collapse to essentially one.** A precision-reaching geometry is fully specified by a single shared bandwidth; centers are a deterministic grid and biases are derived. The admissible set is a measure-zero structured point, not a basin in $\mathbb{R}^{2N}$.
2. **The failures are approximation-theoretic, not numerical.** Conditioning and curvature are explicitly ruled out as the discriminator. Off-geometry solutions fail because they fail to *cover* the interval, not because the linear algebra is ill-posed.

## 1. What collapses the degrees of freedom (most to least reduction)

- **All $N$ weights share one bandwidth.** The precision optimum is a single $\gamma$ for every neuron, at dimensionless $\lambda^*=\gamma h\approx0.25$ ($h=2/N$), i.e. $\gamma^*=\lambda^* N/2$, so $\gamma^*/N\approx0.10$ (C03). This is the largest collapse: $N$ weight magnitudes $\to$ one number. It is also the answer to "what magnitude should the inner weights be," and it requires $\gamma=O(N)$ -- not $O(1)$.
- **Centers must be the uniform grid.** Among uniform/random/clustered/trained/reg-clustered placements at equal $W$, span, and $\gamma$, only uniform reaches the floor; all others plateau 2-3 decades above and never descend (C04). Given $N$ and the span, the grid is determined: $N$ center positions $\to$ 0 free.
- **Biases are derived, not free.** The center is $c=-b/w$; at the optimum $b=-\gamma c$. Once $\gamma$ and the grid are fixed, the $N$ biases are fixed (C05).
- **Signs carry no information.** For odd tanh a weight sign-flip is a feature-column flip the readout absorbs; gelu and gelu-positive-init are identical to $|\Delta\log_{10}\text{err}|\le0.05$ (C05). The loss is sign-symmetric.

Net: the $2N$-dim inner layer collapses to **one continuous knob (the shared $\lambda$, living in a basin around 0.25)** plus a deterministic grid. The bandwidth basin is wide (forgiving); the grid and shared-magnitude requirements are sharp.

## 2. Perturbation theory: how precision decays as you leave the point

- **Bandwidth $\lambda$ -- the dominant axis.** Error vs $\lambda$ is U-shaped for both the QI construction and a plain lstsq readout (so it is not a formula artifact): ill-conditioning on the low side, aliasing/cancellation on the high side, a viable minimum between (C01). The minimum is at $\lambda^*\approx0.25$, constant across target frequency and width (C02, C03); only the error magnitude grows with frequency. Moving off the band costs 10+ decades. The basin is wide -- lstsq tolerates a broad $\lambda$ range; QI needs a narrow one.
- **Center uniformity -- monotone, never recovers off-grid.** Moving centers from a random/Xavier layout to the grid improves error monotonically, with the steepest gain in the final approach to perfect uniformity (C05). Best-over-$\lambda$ the penalty is ~1-4 decades (target-dependent; exp ~0.3, sine_8pi/runge up to ~4), but the key fact is qualitative: **non-uniform placement plateaus and never reaches machine precision** (C04).
- **Weight uniformity -- conditional and non-monotone.** At the viable $\lambda$, equal weights beat a spread by 2-5 decades; at too-small $\lambda$ the sign flips and a spread is better (some neurons reach a usable scale). Along the path to uniformity, error rises before recovering -- the "hump" (C06).

## 3. Mechanisms: why other solutions fail

- **The two geometry ingredients are coupled one-way.** Center uniformity helps on its own; weight uniformity does not -- uniform weights with random centers is the *worst* corner of the entire surface, several decades worse than the random start (C05). Identical sharp kernels tile perfectly on a regular grid and leave matching gaps on an irregular one. So you cannot install uniform weights before uniform centers.
- **The hump is the loss of the coarse basis.** The small (soft, low-bandwidth) neurons span a low-degree polynomial basis that cheaply fits smooth/convex targets; pulling them to the shared $\gamma$ destroys it before the sharp uniform basis is in place. Protecting the soft neurons (but not a random same-size set) flattens the hump, and the effect tracks target convexity -- a causal result (C06). A few soft neurons can even lower the floor, suggesting a deliberate multi-band ("cascaded") geometry.
- **It is not conditioning, and not curvature.** Uniform, random, and reg-clustered geometries all sit at $\mathrm{cond}(\Phi)\sim10^{19}$-$10^{20}$, yet random is 2-3 decades worse (C04) -- conditioning does not order accuracy. This extends the paper's "curvature does not explain the gap" to the feature matrix from an independent direction. The discriminator is coverage/approximation, not numerics.
- **A bimodal weight vector (mass crossing zero) collapses rank.** Interpolating weights through zero bandwidth dips the effective rank ~15% and spikes error mid-path (C05). A usable geometry wants every neuron at a definite, nonzero bandwidth.

## 4. Why an optimizer (Adam) would miss this geometry

The checkpoint's answer to the research question. The target is a structured, measure-zero point that gradient descent on raw $(w,b)$ has no inductive bias toward:

- **It must scale all weights together to $\gamma=O(N)$.** Standard init gives $\gamma=O(1)$; the viable basin is at $\gamma=O(N)$ ($\gamma^*/N\approx0.10$). Per-weight gradients give no pressure to inflate the whole layer in lockstep (C03). This is the paper's gamma-scaling violation, now with a magnitude target.
- **It must make all weights equal, against a non-monotone path.** Uniformity only pays off once $\lambda$ and the centers are already right, and getting there crosses the hump -- an uphill barrier a local-improvement step sees as worse (C05, C06).
- **It must place centers on the grid, and partial credit backfires.** Off-grid placements plateau and never recover (C04); installing uniform weights before fixing centers is worse than doing nothing (one-way coupling, C05). Greedy/coordinate-wise progress is actively penalized.
- **The landscape is not the obstacle.** Curvature and conditioning are ruled out; the obstacle is that the admissible region is this thin structured set, reached only by moving the geometry *jointly*. This is the direct argument for the $\gamma(x-c)$ / log-$\gamma$ reparameterization (expD03) and a joint / variable-projection objective (expD04), and for seeding a cascaded soft-neuron basis (C06) rather than coordinate descent on $(w,b)$.

## 5. The covering picture (interpretation, implicated not proven)

One mental model unifies the above, consistent with the data though not proven by it: a precision geometry must **cover $[-1,1]$ plus halo with no gaps**, using kernels **steep enough** ($\gamma$ large) to be placed precisely and resolve fine structure but **not so steep** that QI aliasing sets in (the upper $\lambda$ wall), with **uniform steepness** so no region is under-resolved. Uniform centers give gapless coverage; a shared large $\gamma$ gives precise-but-safe kernels; uniform weights ensure no region is shallow. A Xavier draw does passably once $\lambda$ is held right because its centers are roughly uniform over $[-1,1]$ with fat tails that cover the halo; increasing uniformity then mainly lets the geometry build smaller, more precisely placed kernels -- the last decades of precision.

## Open questions (most to least actionable)

- **Does reparameterization remove the joint-movement barrier?** Optimize in $\gamma(x-c)$ / log-$\gamma$ coordinates (expD03) and/or variable projection (expD04) -- the direct test of the one-way-coupling finding.
- **Cascaded multi-band geometry** (from C06): a uniform grid at the ideal $\gamma$ plus a small, evenly spaced sub-grid of soft (low-bandwidth) neurons -- deterministic "protect-the-soft-weights." Does it beat both pure-uniform and Xavier-protected, and how should band count/spacing scale with $N$?
- **The second bandwidth mode near $\lambda\approx0.05$** (C03): a faint second near-floor region appears at large $N$ (for runge it edges out 0.25). Aliasing, or does width make the small-bandwidth regime attainable?
- **Curvature-clustering** (on hold, C05/E): a deterministic test of whether clustering centers at high target curvature beats the uniform grid (the runge small-$N$ lead).
- **Is the few-soft-neuron floor improvement real?** Protecting ~5-10 smallest neurons beating the cardinal floor by up to ~10x needs many more seeds and an explanation for why it fades by $N=512$ (C06).
- **Small-$N$ runge bandwidth starvation:** does a target-aware halo close the residual gap at the exact geometry, or is it intrinsic to equal kernels on a peaked target?
