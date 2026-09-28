# Why a weak effective force persists

The experiments support **slow reinforcement as a mechanism at wide,
nearly affine states**. They do not support universal contraction or a
single scalar growth rate as a complete motion model. The useful theorem
target is weaker: an upper envelope for accumulated slope travel can remain
informative even when detailed forecasts miss.

The [theory and proofs](../../../../docs/d34_state_dependent_persistence.md)
separate residual relaxation from loaded geometry feedback. The
[protocol](PROTOCOL.md) records the prospective tests. The main experiment
contains 40 forks and 240 feedback branches, plus 16 physical-kick families
and 112 released ordinary-GD branches. Every branch ran 20,000 additional
updates at $\eta=0.002$. No branch failed numerically. This round studies GD;
it supplies no new Adam verification.

## 1. The mechanism is inside the effective fine force

**Example.** At physical width 1409, removing residual relaxation changes
positive normalized slope travel by only 0.00505% at the median of the
12 width-panel states. Doubling geometry feedback changes it by 1.51%.
At width 705, the corresponding changes are 0.0206% and 3.35%.
These are paired intervention effects, not different initial gradients.

**Explanation.** Write the full empirical gradient as $F+R$, with $F$ the
effective fine gradient and $R$ its coarse-disequilibrium correction. Then

$$
\frac{d}{dt}\frac{\|F\|^2}{2}
=-\mathcal D-\mathcal C-F^TDF[R].
$$

$\mathcal D\ge0$ measures depletion of the force by relaxing its driving
residual. The signed $\mathcal C$ measures changing features and their
coarse compensation under the current residual load. It can strengthen or
weaken the force. This compensation remains present at exact coarse balance;
it is part of $F$, not the small disequilibrium correction.

The six interventions independently remove or double these initial feedback
contributions while matching the entire gradient at the fork. Natural GD
is one arm. The other arms retain the current $R$ at their own states.
Their first-step agreement and exact initial derivative identities passed
independent numerical checks.

**Prediction and outcome.** Geometry feedback should matter more than
residual relaxation in the wide states if their force is weak but changes
mainly through geometry. The observed intervention effects support that
ordering. They also get smaller with width. This favors weak reinforcement
over a strong restoring mechanism for this panel. At later archived states,
net force decay is common: 37 of the 46 checkpoint audits have negative
initial logarithmic force growth. These are different regimes of the same
identity, not a universal sign rule.

The width panel contains degree five, mixed sine, an off-center Gaussian,
a smooth bump, a tanh step and a kink, with two seeds at each of three
independently initialized widths. Late pure sine and degree nine add four
forks. The checkpoint audit covers 23 target instances in each of two
cohorts. This tests breadth across these functions; it is not a theorem
about a random population of all targets.

The [per-case natural results](evidence/feedback_summary/natural_20k.csv),
[feedback contrasts](evidence/feedback_summary/contrasts_20k.csv) and
[compact aggregates](evidence/overview.json) retain both cohorts.

## 2. A scalar rate is useful, but does not close the dynamics

**Example.** The force norm at width 1409 is predicted much more accurately
than the slope-motion vector. Across its 12 natural-GD cases, the largest
force-norm error is 0.342%, while the largest slope-vector error is 2.10%.
The distinction matters because a changing force can redistribute motion
among slopes, readouts and biases.

**Theory tested.** Before continuation we saved the checkpoint-only forecast
$F_n=\exp(\eta k_0n)F_0$, together with the frozen direct correction $R_0$.
This assumes both a fixed logarithmic growth rate and a fixed direction.
The baseline holds the entire initial gradient constant. Neither model
uses future fitted coefficients.

| Physical width $W$ | Median force-norm error | Median slope-vector error, scalar rate | Median slope-vector error, constant gradient |
|---:|---:|---:|---:|
| 177 | 15.0% | 16.7% | 21.6% |
| 705 | 0.917% | 3.41% | 4.75% |
| 1409 | 0.216% | 1.67% | 2.19% |

Each row contains six targets and two seeds, excluding the separate late
panel. The scalar correction improves slope forecasts in 33 of these
36 cases. It is a useful compression, not a replacement for the previously
more accurate coupled quintic surrogate.

For the 200 paired feedback contrasts, 190 mean-scale contrasts exceed the
predeclared reporting floor; 170 of those signs are predicted correctly.
At the largest width, all 50 resolved mean signs are correct, with ten
unresolved means. At width 177, only 47 of 60 are correct. Vector predictions
beat a zero-contrast forecast in 195 of 200 cases, but often leave a large
fraction of the contrast unexplained. The floor is a reporting convention,
not a rigorous roundoff certificate.

**A decisive miss: late sine.** For seed 0, $k_0=0.0290$ predicts a force
norm of 0.0997 after the window; the actual norm is 0.0389. The 156% relative
error is an overprediction. Slope error also worsens, from 25.6% under
constant gradient to 41.2% under the scalar correction. The retrospective
path audit explains why: the relaxation rate $\mathcal D/\|F\|^2$ grows
from 0.00920 to 0.0391, while the net geometry reinforcement
$-\mathcal C/\|F\|^2$ falls from 0.0382 to 0.0172. The actual logarithmic
rate changes from positive to negative. The tracking contribution stays
small relative to those terms.

This failure is about changing feedback, not ninth-degree orthogonality.
The other sine seed starts with a negative rate and has a much smaller
force forecast error. Target name alone does not determine the mechanism.
At the wider generic states, total fine-residual change is much smaller:
at most 0.0695% for width 705 and 0.0152% for width 1409. Their remaining
vector errors accompany measurable changes in force direction even under
an almost fixed error load.

The [path audit](evidence/path_audit/path_diagnostics.csv) records the
retrospective rates, force directions and residual changes separately from
the issued forecasts.

## 3. Physical perturbations expose what scalar matching omits

**Example.** A 1% physical kick can preserve force magnitude to high accuracy
yet create a tracking force 25.7 times the original effective-force scale.
The absolute disturbance is small; the original force can be smaller still.
This is especially relevant to the weak polynomial-target states.

**Theory tested.** The kicks leave every slope exactly fixed and match coarse
outputs, disequilibrium, the initial slope gradient and force norm to first
order. They increase or decrease intrinsic amplification in the feasible
tangent space. The full force direction is not matched. Amplitude halving
should divide the matching errors by four and leave the normalized
antisymmetric response approximately unchanged.

**Outcome.** All 16 tangent constructions resolved. Slopes remain exactly
unchanged for all 96 nonzero kicks. The halving ratios of the matched errors
lie between 3.92 and 4.08, supporting their predicted quadratic order.
Nevertheless, at amplitude 0.01, eight of 32 signed kicks have tracking
larger than their effective force. Even at 0.0025, two of 32 do. These are
confounded tests of a low-disequilibrium mechanism and remain in the results.

Antisymmetric responses remove the leading common quadratic disturbance,
but the scalar model still performs poorly. At the smallest amplitude,
its median slope-response error is 90.1%. The median discrepancy between
the 0.005 and 0.0025 normalized slope responses is only 0.0657%, although
the worst is 30.2%. Thus many responses have converged much more closely
with amplitude than the scalar forecast predicts them. Small initial
force-norm and slope-gradient changes do not close their subsequent coupled
evolution. These experiments motivate testing the evolving vector response,
and a nonlinear coarse-balance correction for the contaminated kicks.

**Follow-up: retaining the vector is insufficient if its operator is frozen.**
We continued this thread after seeing the scalar misses. A frozen full loss
Hessian and a frozen effective-force Jacobian predict the infinitesimal
response to the same initial kick. Two existing quintic constructions instead
evolve the parameters and their sensitivities: one uses the anchored effective
fine field, the other the anchored full polynomial gradient. All corrections
are computed at each fork; no future coefficients are fitted. These are
retrospective model comparisons, not fresh prospective confirmations.

The table gives median slope-response error at amplitude 0.0025 and 20k
updates for the six predeclared width-panel targets at $W=177$.

| Model | Development seed 30 | Confirmation seed 32 |
|---|---:|---:|
| Scalar amplification | 89.1% | 87.8% |
| Frozen full Hessian | 92.7% | 93.4% |
| Frozen effective Jacobian | 91.7% | 94.8% |
| Evolving anchored fine quintic | 14.0% | 14.2% |
| Evolving anchored full quintic | 14.3% | 11.8% |

All twelve cases are finite for all models, and the evolving fine quintic
improves over the scalar forecast in every case. Its worst errors remain
about 50%; this is a substantial refinement rather than precise closure.
Adding the coarse channel to the evolving quintic gives a much smaller
change than allowing geometry to evolve. This supports continued work on
the effective fine dynamics.

The four late cases remain visible separately. Degree nine has fine-quintic
response errors of 5.23% and 1.43%. Pure sine is unresolved: the seed-0 errors
are 301% for fine and 551% for full quintic, while both models fail numerically
for seed 20. Across all sixteen families, each quintic model therefore has
fifteen finite predictions and one failed family. This surrogate is not a
general solution for late sine.

There is a concrete reason freezing the operator can fail despite the
initial matching. If $H=Dg$ and $v$ is the physical perturbation, then
$v_a=(H_0v)_a=0$, but its next nonzero raw-slope response contains

$$
\bigl[H_0^2v+DH_0[g_0]v\bigr]_a.
$$

The first term describes coupling through other parameters; the second
describes changing curvature along the base motion. A frozen Hessian drops
the second term at the same leading order. The note derives this for both
continuous flow and discrete GD. It explains why matching a scalar force
rate and the initial slope update leaves substantial dynamics unconstrained.

The fork-only acceleration audit finds the two terms opposed in all sixteen
constructed directions: their slope-block cosines range from $-0.765$ to
$-0.999997$. For the confirmation degree-five state, each term has norm
about $4.21\times10^{-6}$, but their sum has norm $1.10\times10^{-8}$.
Dropping curvature drift destroys this cancellation. This identifies a
concrete quantity that a useful response bound should preserve. It concerns
these deliberately matched perturbations, not every direction or a universal
cancellation of the original slope force. An independent analytic fixture
checks the two-step formula and its step-size order; archival step-halving
records retain roundoff-limited cases rather than claiming uniform observed
convergence order.

The [response comparison](evidence/response_comparison/primary_slope_response.csv)
retains every target, including both sine limitations. The
[acceleration decomposition](evidence/acceleration/acceleration.csv) and
[step-halving records](evidence/acceleration/step_halving.csv) expose the
signed terms and numerical resolution of the initial-response calculation.

## 4. An inaccurate forecast can still yield a useful slope bound

**Example.** Late sine's large overprediction is harmless for an upper
bound on speed. Across all 40 natural-GD cases, the largest observed
speed/reference-speed ratio at saved times is only 1.477. A factor-two
reference travel budget also exceeds the actual accumulated effective-force
travel in all 40 cases. This is a retrospective diagnostic, not a
prospective or all-step speed certificate.

**Theory.** The new constant-factor corollary asks for

$$
\|F_n\|\le M\widehat q_n+b_n,
$$

where $\widehat q_n$ is a preissued speed reference and $b_n$ charges the
effect of tracking on force evolution. It derives this from a bound on
accumulated excess amplification. A separate allowance charges direct slope
motion from tracking. Summing the resulting speeds bounds total travel and
the fraction of distinct neurons that can ever reach $\lambda_*$.
The factor $M$ costs linearly in travel and quadratically in the population
allowance. Neither an accurate force direction nor a close two-sided
trajectory forecast is required for this conservative bound.

**What the numerical margin says.** Using the observed tracking travel only
as a diagnostic, the smallest multiplier that would consume the entire
single-neuron distance to $\lambda=0.25$ is 6.19; in the width-1409 groups
it exceeds 4,000. There is substantial room for a conservative bound.
The factor two was chosen after viewing outcomes. Proving regional
amplification, tracking and geometric closure remains necessary; these
numbers do not certify that factor.
The [one-sided audit](evidence/envelope_audit/summary.json) records its
post-outcome factor choice and use of observed tracking explicitly.

The width corollary proves a structural reason to expect a bounded multiplier
on a controlled interval. When $\sqrt W(a,b,c)$ is bounded and the coarse
Gram matrix stays conditioned,

$$
\|F\|=O(W^{-1}),\qquad
\mathcal D\le O(W^{-2})\|F\|^2,\qquad
|\mathcal C|\le O(W^{-1})\|F\|^2.
$$

Thus intrinsic force reinforcement follows the $\eta n/W$ clock within that regime.
It need not contract. The discrete theorem includes tracking and an explicit
first-exit test. With $h$ proportional to $W^{-1}$, its per-neuron normalized
rate is $O(\eta W^{-5/2})$. The target slope $\lambda_*/h=O(W)$ and the
feedback time $\eta n=O(W)$ are different statements; neither justifies
extrapolating the small-parameter approximation to arbitrarily large slopes.

**The current proof bottleneck is quantitative.** We evaluated the generic
width constants on all 36 width-panel forks with 10%, 25% and 50% outer
parameter margins, setting tracking to zero as a best-case diagnostic.
All 108 pointwise force/curvature/relaxation checks passed. But the generic
conditioning estimate restricts every case, and the largest supported
integer horizon is only 18 updates. These constants cannot explain the
20,000-update observations quantitatively. The useful next bound must retain
the directional loaded curvature and actual coarse geometry rather than
replace them by global parameter maxima. The theorem's scaling is informative;
these evaluated constants are not a new useful numerical certificate.
The [constant audit](evidence/width_audit/summary.json) gives every fixed
margin and its limiting condition.

## 5. What is proved, and what remains to establish

The force-growth identity, conditional discrete persistence theorem,
normalized distinct-ever acquisition bound, width corollary and
constant-factor comparison are proved in the linked note. Independent
mathematical reviews checked the proofs. The numerical experiments verify
the implementation and test forecasts; they do not prove uniform regional
assumptions.

The new empirical conclusion is that wide-state persistence often reflects
weak force with slow geometry reinforcement. Later states can transition
from reinforcement to depletion. The unresolved closure concerns the
changing loaded geometry, residual relaxation and distribution of force
across parameters. The previous force reduction remains the starting point;
these results do not restore coarse disequilibrium as the principal driver
of ordinary-GD scale acquisition.

The primary suite and follow-up models passed 27 focused tests under CPU-only
Slurm: 17 primary checks, six frozen-propagator checks and three coupled-model
checks, plus the independent acceleration fixture.
The two GPU allocations consumed 252 and 65 seconds, including compilation:
317 GPU-seconds total. All numerical work ran under Slurm. Sparse snapshots
and cumulative motion budgets replace dense optimization traces. Source,
input and issued-forecast hashes are retained with the evidence; the
[budget ledger](budget.json) records allocation accounting.
The [scheduler record](evidence/slurm_accounting.txt), test logs in
`evidence/logs/`, and exact launch/aggregation scripts in `execution/`
preserve the computation. Full input arrays and sparse parameter snapshots
remain in the existing Runpod campaign under
`evidence/persistence_1bf7138/`; the repository retains the compact evidence.
