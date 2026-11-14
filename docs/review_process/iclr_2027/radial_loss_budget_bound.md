# A loss-budget bound for slope acquisition under joint GD

Research result, November 13, 2026. This is a conditional replacement for the exploding parameter-norm recurrence. It uses exact native GD and does not expand tanh into per-neuron polynomial remainders. Its new assumption concerns the direction of the gradient, and must be tested or separately proved. It is not an unconditional trapping theorem for standard initialization.

**Status update:** This is not a repaired long-time theorem. The bounded radial-gain premise is unproved and can diverge even when a slope converges to a finite limit with positive residual floor. The held-out checkpoint results below do not establish indefinite persistence. See [the accumulated-motion analysis](accumulated_outward_motion.md) for the structural obstruction and subsequent attempts.

## Statement and meaning

Let theta contain every trainable parameter. Let gamma be its slope block and g=gradient L(theta) the full gradient, with slope block g_gamma. The objective is half training MSE, L=E²/2, where E is training residual RMS. Write A=||gamma||² and gamma_RMS=sqrt(A/W). Let all parameters undergo simultaneous GD with a constant rate eta.

Assume throughout the analyzed continuation:

1. **Sufficient loss decrease:** L_next <= L - eta ||g||²/2.
2. **Bounded outward radial gain:** for one constant K>=0,

   E [-<gamma,g_gamma>]_+ <= K ||g||².

The second premise is explicitly an assumption about training dynamics. It does not follow merely from tanh, small initialization, or decreasing loss. It says the part of the gradient that increases slope energy cannot be arbitrarily large relative to the gradient's capacity to reduce loss. It allows slopes to grow; it limits the cost of that growth in residual reduction.

Then for every step n in that continuation,

\[
\boxed{
\|\gamma_n\|^2
\le \|\gamma_0\|^2
+4K(E_0-E_n)+2\eta(L_0-L_n).
}
\tag{1}
\]

In particular,

\[
\boxed{
\gamma_{\mathrm{RMS},n}
\le\sqrt{\gamma_{\mathrm{RMS},0}^{2}
+\frac{4KE_0+\eta E_0^2}{W}}.
}
\tag{2}
\]

Here E0 is the **full residual RMS at the restart**, not Appendix F's fixed upper bound on the nonaffine residual. The right side of (2) contains no step count or self-fed norm ceiling. It is a remaining-loss budget. Both inequalities are conditional for as long as the two premises hold, not an empirical guarantee that they hold indefinitely.

## Exact proof

The slope update gives

\[
A_{n+1}-A_n
=-2\eta\langle\gamma_n,g_{\gamma,n}\rangle
+\eta^2\|g_{\gamma,n}\|^2
\le \frac{2\eta K}{E_n}\|g_n\|^2+\eta^2\|g_n\|^2.
\]

For E_n>0, sufficient decrease implies E_next<=E_n and

\[
E_n-E_{n+1}
=\frac{2(L_n-L_{n+1})}{E_n+E_{n+1}}
\ge\frac{L_n-L_{n+1}}{E_n}
\ge\frac{\eta\|g_n\|^2}{2E_n}.
\]

Also eta²||g_n||² <= 2 eta(L_n-L_next). Consequently

\[
A_{n+1}-A_n
\le4K(E_n-E_{n+1})+2\eta(L_n-L_{n+1}).
\]

Summing telescopes and proves (1); nonnegativity of E_n and L_n proves (2). If E_n=0, squared loss has g_n=0 and GD makes no update. If g_n=0 at positive residual, the outward condition is satisfied and that update is also stationary.

Equivalently the potential

\[
\Phi_n=A_n+4KE_n+2\eta L_n
\]

is nonincreasing under the premises. Variable rates 0<=eta_n<=eta_max admit the same proof with eta_max replacing eta in the potential and final bound, provided sufficient decrease uses each actual eta_n.

## Why this preserves the needed cancellation

The original norm recurrence charges every gradient component as outward growth. The first signed-energy attempt retained direction initially but then bounded a sum of nonlinear remainders by absolute per-neuron moments. Those estimates were enormous for default initialization.

This proof retains the signed slope-gradient inner product and never introduces those moment estimates. It also bounds the desired slope block directly, rather than charging hidden bias and readout magnitudes to the slope ceiling. It does not use future observed slopes to construct (2).

The price is the new radial-gain condition. Its empirically measurable value is

\[
k_n=\frac{E_n[-\langle\gamma_n,g_{\gamma,n}\rangle]_+}{\|g_n\|^2}.
\tag{3}
\]

When g_n=0 define k_n=0. The numerator retains the sign of the aggregate inner product; one must not replace it by the sum of absolute per-neuron contributions. A positive global K bounds k_n. The bound does not require each individual slope to grow slowly or monotonically.

For an empirical output Jacobian J with its adjoint defined using the training RMS inner product, let r=e/E be the unit residual direction and s=J_gamma gamma be the output change under uniform slope dilation. Then (3) can equally be written

\[
k_n=\frac{[-\langle r_n,s_n\rangle]_+}{\|J_n^*r_n\|^2}.
\]

Thus residual amplitude cancels out of k_n: it measures a directional relation between slope dilation, the residual direction, and tangent sensitivity. This is a concrete assumption to investigate, not a general fact about neural networks.

## What can be claimed about a scale threshold

Given a reference Gamma_star, condition (2) rules out reaching gamma_RMS >= Gamma_star while both premises hold if

\[
K < K_{\rm crit}
=\frac{W\Gamma_\star^2-A_0-\eta E_0^2}{4E_0}
\]

with a positive numerator. The premises need only hold before first crossing for the usual first-exit contradiction. They do not become automatic from that formulation.

Gamma_star can be the QUILL construction reference for comparison. That reference is not a necessary threshold for accurate approximation. This proof supplies an RMS scale ceiling, not a maximum-neuron ceiling, an error lower bound, or a universal training-failure theorem. Applying the frozen-geometry theorem to moving geometry requires its own assumptions.

## Empirical design

The primary check uses the existing 24-function tanh runs at width 196, PyTorch-default initialization, and learning rates 0.001/0.0001 (48 runs). Nothing was retrained.

- Calculate k_n on the saved 20k–30k prefix.
- Set one shared K to twice the maximum over those prefixes, targets, and learning rates.
- Freeze K and check it on all saved 30.5k–200k states.
- Calculate the constant ceiling using only the 30k slopes and residual plus the frozen K.
- Separately test the frozen K on 2x and 4x initialization without retuning.

This is an exploratory temporal split. The form of the condition was developed after looking at a few saved states, including later states. It is not preregistered or independent replication. Checkpoints are every 500 steps, so sampled persistence does not establish the gradient premise at every intervening update or beyond 200k.

A sufficient loss-descent condition is available separately from the starting state and finite horizon, as in loss_descent_bound.md. Pointwise Hessian checks along proposed one-step segments are additional diagnostics, not a substitute for checking intervening states. No radial premise is justified solely by a plotted ceiling enclosing measured slopes.

Results and reproducible diagnostics: [D65 results](../../../results/checkpoint_D_optimizers/expD65_radial_loss_budget/results.md).

## Observed result

The frozen common threshold is K=1000.272842. It passes at all held-out saved states in all 48 default-initialization cases. The worst prospective constant RMS-slope ceiling is 4.5544, compared with construction reference 18.375. The largest observed held-out radial gain is 924.7638.

The same threshold passes 47/48 twofold-initialization cases and 48/48 fourfold-initialization cases. Airy with 2x initialization and LR 0.001 fails (radial gain 1389.4143); it is not relabeled successful merely because observed slopes remain below the candidate curve. The largest conditional RMS ceilings among passing 2x and 4x cases are 4.6610 and 5.0650.

All 144 cases pass the separate finite-horizon starting-state sufficient descent check. The new radial assumption is only sampled. In particular the hardest primary radial gains are still increasing near 200k. This finding supports an informative conditional theorem on these trajectories; it is not a proof of indefinite persistence or a replacement for structural analysis of the gradient-direction premise.
