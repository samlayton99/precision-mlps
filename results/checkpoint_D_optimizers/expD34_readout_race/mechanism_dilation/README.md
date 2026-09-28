# Larger slopes do not automatically produce rapid further acquisition

**The finite perturbation test is informative.** Doubling slopes and biases,
then balancing the coarse tracking correction, increases the initial effective
fine slope force substantially. Nevertheless, the wide networks acquire very
little additional scale over 20,000 ordinary GD updates. The later, narrower
networks often show a transient corrective response, but retain most of the
injected scale. These are different responses within a slowly evolving region;
they do not establish a common attracting equilibrium.

All 420 planned branches ran successfully. The clean wide result covers six
functions at two widths and two seeds; the separate late panel covers 23
functions and two seeds. We do not transfer the six-function conclusion to all
23 functions or interpret the different checkpoint ages as a width comparison.

Here $\gamma_i=|a_i|$ is physical slope scale and $\lambda_i=h|a_i|$ is
normalized scale, with $h=2/N_{\rm ref}$. The **effective fine gradient** is
$F$, the **coarse tracking correction** is $R$, and GD uses $-(F+R)$.
The plots measure changes after the intervention, excluding the scale injected
by hand. “Primary” and “inverse readout” identify two repair references,
explained below; neither changes the subsequent GD rule.

## First read the trajectories: most of the extra scale stays

As one concrete example, the left Gaussian target at width 1409, seed 30,
gains only **0.588%** additional mean scale after a twofold primary dilation.
This is the largest increase among the twelve primary twofold cases at that
width. The mixed-sine cases contract slightly, whereas Gaussian, bump, and step
targets expand. Slow acquisition therefore occurs with either sign of motion.

![Post-intervention mean-scale changes, excluding the injected increase. Solid curves are case medians; shading is the interquartile range. Dashed curves are medians of each state's own initial-effective-force forecast. Left: 23 targets and two seeds, width 177, restarted after 600k updates. Middle and right: six targets and two seeds per width, restarted after 20k updates. All branches continue for 20k updates at learning rate 0.002, so final flow time is 40. Symmetric logarithmic vertical axes accommodate either sign. The different ages prevent reading all three panels as a controlled width experiment.](evidence/analysis/scale_trajectories.png)

For twofold primary dilation, median additional mean-scale growth is **0.392%**
at width 705 and **0.181%** at width 1409. The respective ranges are
$[-0.410\%,1.394\%]$ and $[-0.336\%,0.588\%]$. Four of twelve cases
contract at each width. The corresponding repaired baselines have medians
$0.245\%$ and $0.0968\%$. Dilation does increase subsequent movement on
many targets; the resulting rates remain small.

At width 177, the twofold primary dilation instead produces contraction in
33 of 46 cases, with median change **$-0.150\%$** and range
$[-4.365\%,6.054\%]$. This does not undo a 100% injected increase. The
median gap to the repaired baseline retains **99.17%** of its initial value.
The minimum retained gap is 86.63%. Some late baselines themselves grow
substantially: the largest ordinary-GD mean increase is 25.48% over this window.
The late panel is not uniformly stalled.

![Remaining gap to the repaired baseline after 20k updates, divided by the gap immediately after dilation. Each point is a checkpoint/intervention pair: 46 cases per arm in the late panel and 12 per arm in each wide panel. A value of 1 means the initial gap persists. Gap closure is distinct from actual contraction: the baseline can also move. Nearly unchanged gaps in wide states and mostly retained gaps in late states argue against a strong return over this horizon.](evidence/analysis/gap_retention.png)

The alternate repair, referenced to $c_0/s$, retains the broad conclusion but
changes the details. At width 177 its twofold median change is $-0.0151\%$,
and only 24 of 46 cases contract. At widths 705 and 1409 the twofold medians
are $0.299\%$ and $0.147\%$. Thus readout choice matters; contraction is
not an invariant response to slope dilation alone.

## A force jump is different from force reinforcement

The twofold primary kick multiplies the initial effective slope-gradient norm
by medians **3.44** and **3.53** in the two wide panels. During the subsequent
continuations, those norms change by median factors **1.139** and **1.034**
relative to their own post-kick values. Some outward rates increase as well.
The finding is limited amplification over this interval, not absence of
reinforcement or a universally decreasing force.

The late twofold primary kicks behave differently: the initial norm increases
by a median factor **13.8**, then falls **to 17.0%** of its post-kick value.
A much larger instantaneous force can therefore be a transient response to
the changed state. It does not by itself predict rapid further scale growth.

![Top: initial effective slope-force amplification relative to the paired repaired baseline versus subsequent amplification relative to the kick's own starting force. The horizontal line marks unchanged force. Bottom: final versus initial signed effective mean-scale rate, with equality diagonals. Positive rates mean outward mean-scale motion; force norms alone do not give that sign. Every case is shown, with the same panel coverage and 20k-update horizon as the trajectory plot. Late force bursts often decay; the wide forces undergo much smaller relative changes.](evidence/analysis/force_norm_and_signed_rates.png)

The forecast keeps $F_a$ fixed at the actual repaired starting state:
$a_{\rm lin}(n)=a(0)-n\eta F_a(0)$. In the wide twofold primary cases,
its endpoint slope-vector error divided by actual slope-vector displacement
has medians 13.7% and 6.41%, and maxima 16.7% and 7.86%, respectively.
The same approximation often fails badly in the late kicked states because
the force changes substantially. These are numerical forecast errors, not
certified error envelopes or bounds beyond this interval.

![Actual mean-scale change versus the state's own initial-effective-force prediction. Equality diagonals provide the reference. All six arms are shown, including ordinary GD and the repaired baseline. The wide panel follows the local forecast much more closely than many of the late kicked states. Both coordinates exclude the imposed scale increase. The signed logarithmic axes preserve inward and outward responses.](evidence/analysis/motion_vs_initial_force.png)

## What the decomposition explains, and what it does not

For $f(x)=\sum_i c_i\tanh(a_ix+b_i)+d$, let $J_C$ contain the empirical
constant and linear output Jacobian rows. Let $e_C$ be the corresponding
residual coefficients and $g_H$ the gradient from the complete orthogonal
complement. The exact decomposition is

$$
\ell=(J_CJ_C^T)^{-1}J_Cg_H,\qquad
F=g_H-J_C^T\ell,\qquad
R=J_C^T(e_C+\ell),\qquad g=F+R.
$$

The compensation $-J_C^T\ell$ remains inside $F$; balancing tracking does
not remove compensation. Repairs enforce $e_C+\ell=0$ at the start.
Subsequent steps are ordinary full-batch GD, and tracking is allowed to recur.

In all wide branches, the accumulated slope tracking-norm budget is below
0.1% of the corresponding effective-fine budget. This ratio is
$\sum_n\eta\|R_{a,n}\|/\sum_n\eta\|F_{a,n}\|$; it does not assume
that signed mean contributions cannot cancel. Late interventions require
more caution: the largest such ratio is 7.21%, and the largest sampled
instantaneous ratio is 15.4%. In the twofold primary wide branches, the signed
tracking contribution is also small: at most 0.072% and 0.045% of net mean-scale
change at widths 705 and 1409. In contrast, a late inverse-reference branch
has signed tracking equal to 144% of its net mean-scale change because the
signed channels cancel. The late response is a full-GD result with measured
tracking contamination, rather than an exact effective-flow experiment.

**An important counterexample prevents an overly simple correction story.**
At width 1409, mixed sine, seed 30, the primary twofold kick starts with signed
effective mean-scale rate $-6.22\times10^{-9}$. The target-side cubic
contribution is $-6.18\times10^{-9}$, whereas the sum of generated quadratic
and cubic contributions is only about $-1.01\times10^{-11}$. Generated
lower-mode error points inward here, but does not explain most of the inward
motion. The target's correlation with the current compensated sensitivities
can itself favor decreasing mean slope.

These contributions are obtained from empirical orthogonal polynomial
coefficients, not a Taylor fit. If $f_k,y_k$ are generated and target
coefficients and $T_{a,k}$ is the slope block of the compensated sensitivity
$\Pi J_H^T$ for mode $k$, their signed mean-scale rates are

$$
v_k^{\rm generated}=-\frac hW\operatorname{sign}(a)^TT_{a,k}f_k,
\qquad
v_k^{\rm target}=\frac hW\operatorname{sign}(a)^TT_{a,k}y_k.
$$

They sum to the mode's contribution to the signed mean-scale rate induced by
$-F_a$. They are an attribution at the
observed state, not separate training interventions. Moreover, observing force
decay does not identify how much comes from residual relaxation versus changing
geometry and compensation; that energy-identity attribution was not measured
in this experiment.

## The refined theoretical prediction

The wide result is consistent with the expected local sensitivity powers.
In the nearly affine regime $|ax+b|\ll1$, removing constant and linear output
leaves a leading cubic activation term. Under $(a,b)\mapsto s(a,b)$ and
$c\mapsto tc$, generated quadratic and cubic coefficients scale like
$ts^3$, while direct fine slope sensitivities scale like $ts^2$.
Consequently, the direct target-loading contribution scales like $ts^2$
and the generated-cubic-error contribution like $t^2s^5$.

For $t=1/s$, these become $s$ and $s^3$. The observed wide inverse-reference
twofold initial-force medians, 1.94 and 1.99, are consistent with the first
scaling; the primary medians 3.44 and 3.53 are near $s^2=4$. This is an
illustrative consistency check, not a fitted law for the effective force.
The repair does not scale all readouts uniformly, compensation changes its
projection, leading terms can cancel, and target orthogonality can make a
higher Taylor term dominant.

The useful theorem target remains **persistence of a broad population with
limited reinforcement**, allowing either sign of individual and mean motion.
A fixed finite dilation can change force prefactors without leaving the width
regime $a,b,c=O(W^{-1/2})$. Under the existing shape condition controlling
the sixth moment by the cube of the second moment, the effective-force bound
has the form $\|F\|\lesssim Y M^{3/2}/W$, and the coupled moment evolution
satisfies $\dot M\lesssim Y M^2/W$. Here $M=\sum_i(a_i^2+b_i^2+c_i^2)$
and $Y$ bounds the fine residual norm; constants depend on the shape bound.
This controls a rate through population structure rather than assuming the
future force is small. Proving that the relevant shape condition persists
remains the substantive gap.

The new experiment supports robustness under the tested finite dilations,
and rejects strong restoration as a necessary explanation. It does not prove
shape invariance, a universal contraction sign, or infeasible acquisition time
for arbitrary initializations. It also does not reach the neighborhood of
$\lambda=0.25$ in the wide panels: their largest final slopes are only
0.00114 and 0.000384. The value 0.25 is a construction benchmark, not a proved
necessary approximation threshold. At width 177, twofold dilation injects six labels above
0.25 in each readout-reference arm; these are not learned crossings. There
are no new GD crossings of 0.25 in the 420 main branches.

## Reproduction and numerical checks

The [pre-recorded protocol](PROTOCOL.md) specifies the comparisons and
acceptance rules. Each of 70 checkpoints has six branches: original, balanced
baseline, and scales 1.25 and 2 with each of two readout references. Repair
locks geometry and locally minimizes physical squared displacement of $(c,d)$
from either $(c_0,d_0)$ or $(c_0/s,d_0)$ subject to exact coarse balance.
It verifies constraint and stationarity residuals, not global optimality.
All 350 repairs passed; the remaining 70 branches are original states.

All runs use 2,048 training points, FP64, and ordinary GD at learning rate
0.002 for 20,000 updates. The late panel restarts at update 600,000; the wide
panel at 20,000. Late development/confirmation use seeds 0/20 for the original
13 functions and 22/23 for the ten added functions. Wide cases use seeds 30
and 31 for degree five, mixed sine, left Gaussian, right bump, right step,
and absolute-value kink. These are archived training states; “confirmation”
is a cohort label and does not make this new held-out generalization evidence.

Seventeen focused tests verify independent gradients, exact force splitting,
repair derivatives and stationarity, failure handling, and crossing-aware
scale accounting. A zero-fine-force fixture initially exposed an inappropriate
relative-ratio expectation in the test; the acceptance gate was retained and
the fixture corrected. The maximum production accounting defect is
$3.23\times10^{-16}$ in normalized individual scale.

The six prespecified development targets were repeated in all six arms at
learning rate 0.001 for 40,000 updates, matching flow time. The complete initial
parameters and prepared input hashes match. Across these 36 controls, the
largest endpoint slope-vector difference is $2.34\times10^{-5}$ times
the original-step displacement, and the largest absolute difference in mean
percentage scale change is $5.07\times10^{-5}$ percentage points. This checks
step sensitivity on that subset; it is not interval certification.

The verified training implementation is commit `8e7c8c5`. Slurm jobs 1370,
1371, and 1372 performed tests, repair preparation, and coverage checks. GPU job
1377 completed in **228 seconds, or 0.0633 GPU-hours**, including all controls,
well below the one-hour follow-up cap. Analysis uses the existing plotting
environment after the training environment lacked Matplotlib. Sparse raw
snapshots are archived locally under `evidence/` with manifests, source hashes,
and scheduler logs. The [analysis summary](evidence/analysis/summary.json),
[per-branch endpoints](evidence/analysis/endpoints.csv), and
[curated numerical facts](evidence/analysis/interpretation_facts.json) support
the reported comparisons. The helpers generate evidence only; this report
was written after inspecting the numerical outputs and plots.
