# Audit of the readout/geometry mechanism: note sections 3–4 and D28–D29

Read-only evidence review, 14 September 2026. Scope: the note, expD28, and expD29 including its completed six-weight sweep. No new training or intervention was run. Existing saved arrays were read for a few endpoint calculations. This report does not replace the coordinator's synthesis.

## Main finding

The new experiments support a separation between improving the current fit and improving the best numerically accessible fit on the current geometry. In many measured states, ordinary geometry GD is dominated by the first task. A deliberately amplified numerical approximation gradient can improve the second task substantially. This is evidence of a useful direction receiving little influence under ordinary training; it is not evidence that useful geometry information has disappeared, nor proof that readout adaptation caused its disappearance.

The completed D29 sweep must supersede the weaker mu=1000 pilot when summarizing the project. At mu=100000 all four Xavier targets acquire better refitted geometry, including on independent evaluation points and at every audited cutoff. These are meaningful counterexamples to claims that geometry cannot improve or that amplification never helps. They remain far from a stable, bounded-coefficient, architecture-independent recovery method.

## The mathematical statements and their limits

Section 3.1's decomposition is exact for finite-sample squared loss and the exact feature projector at a given state:

\[
\nabla_\theta L=J(v)^TPr+J(v)^Tr_\perp
=C^T\nabla_vL+Z^Tr_\perp.
\]

It identifies which parts of the *current geometry gradient* originate in residual inside and outside the current readout span. Neither contribution is guaranteed to increase gamma or improve approximation. The exact frozen-geometry readout recurrence explains removal of expressible residual at squared singular-value rates, but using it as a joint-training explanation requires an additional timescale argument while geometry and coefficients change.

Section 3.2's exact profile gradient requires locally constant rank and a smooth minimum-norm readout. The Jacobian is evaluated at the *solved* coefficients. Small coefficients, large coefficients, and readout replacement can change this map independently of residual depletion. The note explicitly warns that truncated/regularized solvers are different objectives.

Section 3.3 is a factorization, not an attribution result. For a proposed geometry direction, the loss derivative factors into residual size, raw tangent size, the non-compensable fraction, and signed alignment. To claim a compensation barrier, one must show that the non-compensable fraction is small in directions relevant to the desired approximation, beyond weakness already present in the raw tangent. D28 does not separately measure these factors.

Section 4.1's Hessian/sensitivity-squaring result is local near an exactly fitted, constant-rank profiled state. A weak singular direction matters only if the error loads onto it. It cannot directly establish that far-from-fit joint GD stays near Xavier or never finds the construction's scales. Equation (4.4) is especially useful: small *absolute* gradient is insufficient; slow relative profiled convergence requires small residual-normalized sensitivity. For exact profiled flow,

\[
-\frac{d}{dt}\log\|r_*\|=\frac{\|\nabla\Phi\|^2}{\|r_*\|^2}.
\]

A uniform upper bound on this rate over a region gives a time lower bound. An endpoint measurement does not establish a trajectory-wide bound or that the trajectory stays in that region.

Section 4.3 gives an actual small-scale mechanism: the leading centered tanh scale derivative is proportional to the current feature, hence reproducible by readout changes. Its projected remainder begins at higher Taylor order. This is a plausible analytic starting point for a restricted theorem, but the required small-preactivation regime, readout bounds, target loading, and persistence remain to be proved for the intended training model. The numerical projector used in D28 is not automatically the exact projector required by this argument.

## The loss split in D28 is not the span split in section 3

For exact least squares, let

\[
F(\theta)=\min_vL(\theta,v),\qquad
G(\theta,v)=L-F=\tfrac12\|A(v-v_*)\|^2\geq0.
\]

Then \(\nabla F=J(v_*)^Tr_\perp\), whereas the note's outside-span term is \(J(v)^Tr_\perp\). Define

\[
K=J(v-v_*)^Tr_\perp.
\]

Because the geometry Jacobian is linear in readout coefficients,

\[
\nabla F=g_{\mathrm{outside}}-K,
\qquad
\nabla G=g_{\mathrm{shared}}+K.
\]

Thus observing \(\|\nabla F\|\ll\|\nabla G\|\) does not directly measure how much projection suppresses the current tangent. In particular, D29 weighting of \(\nabla F\) is not simply amplification of the note's current outside-span term.

For the implemented cutoff-based reference, \(F_\tau\) is the target residual outside the *retained singular subspace*. It is not the unrestricted exact approximation floor, and \(G_\tau=L-F_\tau\) need not be nonnegative for arbitrary current coefficients that use discarded directions. It was positive at every saved D28 state. Its derivative includes motion of the retained subspace; blindly using the exact envelope formula with truncated coefficients would be wrong. D28 implements and independently checks the additional derivative terms.

## What D28 establishes

The experiment runs ordinary GD uninterrupted for 2000 steps on four targets and three initializations. All profile calculations are offline. Geometry means all raw slopes and biases, not center-preserving scales alone.

- Scaled Xavier gives the cleanest resolved evidence of a magnitude imbalance: terminal \(\|DF_\tau\|/\|DG_\tau\|\) lies between approximately \(4\times10^{-8}\) and \(2\times10^{-4}\). The actual gradient is almost the readout-gap gradient.
- Xavier mixed sine has opposition, but not near-total cancellation: terminal norms are approximately 0.00102 and 0.00295, cosine −0.377, actual norm 0.00274. Xavier Runge has positive cosine +0.398. Universal equal-strength cancellation is contradicted by these examples.
- Approximation can stagnate or worsen while actual loss improves. Along smooth exact or numerical-profile regions, \(\dot F=-\|DF\|^2-DF^TDG\). A much larger mildly opposing \(DG\) can prevent progress on \(F\) without nearly canceling the total gradient.
- The QI numerical floor is already at roughly 1e-30–1e-27 in squared loss; its profile-gradient directions are unresolved. This does not show exact zero gradient, orthogonality, or that a nonzero target approximation barrier has been reached.

The unresolved directions are material: no Xavier sine/Gaussian cosine passes the stated screen; mixed sine passes at 121/150 snapshots, Runge at all 150. The screen compares SVD algorithms and tanh rounding and is empirical, not a rigorous error bound. Rank stayed constant at the *saved* D28 states (10/99/140 for Xavier/scaled Xavier/QI); this neither proves exact rank nor excludes intervening cutoff crossings.

## What the completed D29 sweep adds

D29 changes the geometry objective to

\[
H=L+(\mu-1)F_\tau,
\qquad
\nabla_\theta H=\mu DF_\tau+DG_\tau.
\]

Readout continues ordinary GD; solved coefficients are used only to calculate the added gradient, never installed in the model. This is a controlled intervention, but it is a different objective and performs a dense SVD each step.

After 500 steps, the largest weight gives these independent-grid refit relative errors:

| Xavier target | Ordinary GD | mu=100000 | Current training loss change |
|---|---:|---:|---:|
| Sine | 0.00519 | 0.000207 | −1.02% |
| Mixed sine | 0.448 | 0.213 | +15.57% |
| Runge | 0.147 | 0.0128 | +14.24% |
| Gaussian envelope | 0.492 | 0.197 | −27.18% |

Negative loss change means improvement. Thus useful geometry improvement and actual fitting can oppose each other. All four refit improvements survive cutoffs 1e-12, 1e-13, and 1e-14 relative to the largest singular value. Runge's strong gain at mu=100000 survives the stricter cutoff and uses smaller solved-coefficient norm than its baseline; this improves on the cutoff-sensitive mu=1000 pilot.

The endpoint geometry remains heterogeneous and far from uniform QI: mean gamma is about 0.091, 1.76, 0.139, 0.757 across these targets, versus the construction reference 16. Mixed sine has maximum gamma 34 despite mean 1.76. Biases and centers can also move, so these gains cannot be attributed solely to mean gamma.

The endpoint refitted functions are observable and robust across the tested cutoffs; the path still has significant numerical limitations. Primary-cutoff solved-coefficient norms remain approximately 3e8–1.4e10. Five of seven audited high-weight sine states are unresolved, and one of seven for each other target. Saved maximum relative update variation across alternative numerical calculations reaches 0.81/0.30/0.13/0.34 for sine/mixed/Runge/Gaussian respectively. Hard-cutoff rank changes introduce jumps. All runs being finite does not imply that the exact unrestricted-profile trajectory has been accurately integrated.

The evidence supports the existence of useful geometry changes accessible when the numerical approximation gradient receives much more influence. It does not show that projection/readout adaptation originally destroyed that direction, nor establish an effective production optimizer.

In particular, these results do **not** establish that reaching useful gamma requires a provably unstable step size. The D29 intervention changes the objective, and its actual-loss spikes, cutoff jumps, and numerical sensitivity cannot prove necessary instability of ordinary GD or every possible method. Its successful refitted endpoints are evidence against an absolute absence of useful geometry information; they are not yet a stable, scalable workaround satisfying the project's accuracy and coefficient requirements.

## What is still needed for the proposed stall theorem

1. **Specify the object that is failing.** Distinguish sampled training error, independent-grid approximation with current coefficients, exact unrestricted readout approximation, and a stable bounded-coefficient approximation class. A cutoff-defined floor alone cannot certify an exact-arithmetic impossibility.
2. **Specify dynamics and budget.** Ordinary raw slope/bias GD, its initialization, coefficient normalization, learning-rate bound, target family, sampling, and width scaling must be fixed. Center-preserving laws cannot silently govern raw-coordinate updates.
3. **Prove the relevant useful direction stays weak.** Small gradient norms, loss decomposition, and compensation identities are not themselves a persistent rate bound. Establish residual-normalized sensitivity or accumulated directional drift on the region the trajectory actually visits.
4. **Connect weakness to missing approximation.** Show that leaving this region is necessary for the desired error under the admissible coefficient/architecture class. The construction's sufficient gamma scaling is not automatically a necessary condition for every tanh representation.
5. **Prove the causal depletion claim if retained.** Demonstrate that a geometry signal initially improves refitted approximation, that readout adaptation removes it before enough useful motion occurs, and that the remaining signal cannot compensate on the claimed budget. D28/D29 do not establish this sequence.
6. **Separate finite-time impracticality from nonconvergence.** A polynomial lower bound with explicit constants can be valuable without saying GD permanently fails. The present interventions also show that useful approximation changes are possible; an impossibility statement must be scoped to a particular algorithm and assumptions.

## Sources reviewed

- Note text: `/tmp/gamma_note_current.txt`, sections 3–4, with section 7's explicit statement that general joint-GD convergence/nonconvergence is not proved.
- [D28 results](../../results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition/expD28_results.md), [status](../expD28_status.md), saved endpoint arrays, and visually reviewed Xavier/scaled-Xavier figures.
- [D29 results including complete sweep](../../results/checkpoint_D_optimizers/expD29_weighted_profile/expD29_results.md), [status](../expD29_status.md), sweep `summary.json` and `terminal_cutoff_audit.json`, and visually reviewed Xavier sweep figure.

No existing experimental files were edited. This audit owns only this report.
