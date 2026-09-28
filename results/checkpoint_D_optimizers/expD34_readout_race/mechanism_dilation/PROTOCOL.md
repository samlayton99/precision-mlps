# Do larger slopes restore, remain slow, or reinforce their growth?

This experiment tests whether the observed slow motion survives a finite
increase in feature scale. Earlier perturbations changed mean scale by only
about 0.7–3% at the median and deliberately preserved the initial slope
gradient. Their persistence therefore did not test a generic restoring force.
Here we increase every slope and bias by 25% or 100%, repair the coarse
tracking correction, and resume ordinary gradient descent. Success means an
interpretable comparison, including failed repairs or renewed tracking; it
does not require the proposed slow-growth mechanism to survive.

This protocol is recorded before the new runs. The late and wide panels
answer separate questions because their checkpoints have different ages.

## What changes, and what stays controlled

The network is $f(x)=\sum_i c_i\tanh(a_i x+b_i)+d$. Physical feature scale is
$\gamma_i=|a_i|$; normalized scale is $\lambda_i=h|a_i|$, with the archived
reference spacing $h=2/N_{\rm ref}$. Width $W$ counts hidden units and is not
$N_{\rm ref}$.

Each checkpoint has six branches: the original state, a repaired state with
unchanged geometry, 1.25-fold and twofold geometry dilations with the original
readout as repair reference, and the same two dilations with $c_0/s$ as the
readout reference. The output-intercept reference is always $d_0$. Dilation
means $(a,b)=(sa_0,sb_0)$, preserving centers wherever $a_i\ne0$. It does not
preserve the initial slope gradient. The alternative readout reference tests
whether the answer depends on how scale is shared between slopes and readouts.

Repair changes only $(c,d)$, keeping the dilated geometry fixed. Write $J_C$
for the two coarse rows of the output Jacobian, $e_C$ for the residual's
constant and linear coefficients, and $g_H$ for the gradient from the entire
orthogonal complement. Define

$$
K=J_CJ_C^T,\qquad \ell=K^{-1}J_Cg_H,\qquad
z=e_C+\ell,\qquad F=g_H-J_C^T\ell,\qquad R=J_C^Tz.
$$

Thus the full gradient is $g=F+R$: $F$ is the effective fine gradient and $R$
the coarse tracking correction. The compensating coarse gradient is
$-J_C^T\ell$. The repair enforces $z=0$, rather than $e_C=0$.

For each requested scale and reference $v_{\rm ref}=(c_{\rm ref},d_0)$, solve
locally

$$
\min_{v=(c,d)}\frac12\|v-v_{\rm ref}\|_2^2
\quad\text{subject to}\quad z(v)=0.
$$

An equality-constrained SQP step uses the exact autodifferentiated constraint
Jacobian and identity objective Hessian, with merit backtracking. Continuation
uses scale increments at most $1/8$, halving failed increments down to
$1/1024$, and at most 100 iterations per continuation point. The reference
remains fixed throughout each requested solve. This establishes a locally
stationary repair, not a globally nearest repair.

Acceptance requires normalized $\|z\|_2\le10^{-12}$, normalization by target
RMS floored at $10^{-12}$; numerically resolved $K$; normalized stationarity
residual at most $10^{-8}$; and initial slope tracking norm at most $10^{-3}$
times the larger of the original and repaired effective slope-gradient norms.
Stationarity is normalized by the larger of reference norm, repair displacement
norm, and $10^{-12}$. Every attempted repair and its residuals are retained.
Failed repairs are not silently replaced by smaller dilations.

## Fixed coverage and measurements

The late panel uses all 23 target functions in each of the two existing
cohorts: 46 checkpoints at $W=177$, after 600,000 updates. The wide panel uses
the existing six targets and two seeds at each of $W=705$ and $W=1409$, after
20,000 updates: 24 checkpoints. There are 420 attempted branches in total.

Continue each accepted branch with full-batch, FP64 GD, learning rate 0.002,
for 20,000 updates. Record snapshots at 0, 1, 10, 100, 1,000, 2,000, 5,000,
10,000, and 20,000 updates. Accumulate signed direct-fine, compensating, and
tracking contributions to mean normalized scale at every update, together
with positive and negative travel and the exact correction for crossings of
$a_i=0$. Integrate absolute channel motion as well, so small net displacement
cannot be mistaken for weak forces.

Snapshots record scale distributions, effective and tracking force norms,
outward mean and RMS rates, loss, and the quadratic/cubic generated and target
coefficients and their slope-gradient contributions. Compare actual motion
with the state's own initial-effective-force forecast

$$
a_{\rm lin}(n)=a(0)-n\eta F_a(0).
$$

The injected increase is not learned motion. Plot subsequent displacement
from each branch's own start and the evolving gap to the repaired baseline.
Report both absolute motion and ratios, avoiding ratios with negligible
denominators. Any first-hit counts include a separately identified injected
population and cannot be presented as acquired during GD.

## Predictions that separate the explanations

Actual contraction after the dilation supports restoration over the tested
range. A closing gap to baseline is weaker evidence: both states might still
be expanding. A persistent offset with little subsequent travel supports
slow evolution without a strong restoring mechanism. An immediate larger
force with motion close to its own initial-force forecast shows static scale
sensitivity; a rising outward effective force and accelerating displacement
show reinforcement developing during training. Different answers under the
two readout references show that geometry alone does not determine the
response. Renewed tracking limits an effective-fine-only interpretation and
is reported as a full-GD response.

These comparisons concern finite perturbations of archived states. Even
twofold dilation can remain far below the scales required for accurate
approximation. Persistence here is not proof of an equilibrium, a permanent
trap, or a universal acquisition barrier.

## Numerical checks and resource limits

Before the full panel, verify exact dilation, center preservation, repair
Jacobian directional finite differences, repair residuals, the decomposition
identity, an independent ordinary-GD step, and closure of the accumulated
scale-change accounting. On the fixed development subset
`moment5`, `mixed_sine`, `gauss_left`, `bump_right`, `step_right`, `kink_abs`,
repeat the late-panel branches at learning rate 0.001 for 40,000 updates,
matching the original flow time. This controls step-size sensitivity of the
observed response; it does not certify an infinite-horizon flow statement.

Preparation, numerical tests, training, and analysis run through Runpod Slurm.
The first GPU allocation is capped at 30 minutes and all GPU allocations for
this follow-up, including retries, at one GPU-hour. Keep sparse snapshots and
compact counters; preserve source archives and unique evidence. Save code and
input hashes, branch metadata, scheduler accounting, and numerical-check
results with the evidence. Write the interpretation after inspecting those
artifacts, then add a concise plot-led update to the self-contained PI note.
