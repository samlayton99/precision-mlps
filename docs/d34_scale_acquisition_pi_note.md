---
title: "Why gradient descent acquires slope scale slowly"
subtitle: "Mechanism, empirical evidence, and a population theorem"
date: "23 September 2026"
fontsize: 10pt
geometry: margin=0.8in
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - '\usepackage{microtype}'
  - '\usepackage{float}'
  - '\usepackage{needspace}'
  - '\floatplacement{figure}{H}'
---

## The question and the present answer

Our numerical construction obtains accurate approximation using narrow tanh
transitions. Training the slopes from small initialization often produces much
broader features. **We want to explain why deterministic gradient descent can
continue fitting while failing to acquire the scales used by that construction.**
The useful conclusion is a rate or a finite-time population bound; it need not
assert that every slope shrinks or that training reaches a trapping equilibrium.

The strongest empirical finding is that, after the initial coarse fit, slope
motion is driven almost entirely by the **effective fine force**: the force
from the remaining errors, including the compensation needed to maintain the
constant and linear fit. The departure from this balance is small in the
audited GD regime. The remaining question is why the effective force does not
rapidly reinforce itself. Wide populations have weak nonlinear sensitivity,
and experiments indicate slow geometry feedback even when substantial error
remains. Correcting generated lower-order error can further suppress motion,
but it is not the sole explanation across targets.

This note derives that reduction, tests its interpretation against the
experiments, and proves a conditional population bound. Its substantive open
assumption concerns preservation of the population's shape, not an assumed
small future force. Existing proofs that derive persistence from initial data
are presently too conservative to explain the observed duration. Thus we have
a supported mechanism and a precise proof target, not yet a general theorem
that GD cannot learn useful slopes on practical horizons.

**Notation used throughout.** Norms on parameter vectors are Euclidean;
function norms use the empirical mean over training inputs.

| Symbol | Meaning |
|---|---|
| $a_j,b_j,c_j,d$ | Signed slope, hidden bias, readout weight, and output bias. |
| $W$, $h=2/N_{\rm ref}$ | Actual neuron count and construction spacing; $W$ includes halo neurons. |
| $\gamma_j=\lvert a_j\rvert$, $\lambda_j=h\lvert a_j\rvert$ | Physical and normalized slope scales. |
| $g=\nabla L$, $F$, $R$ | Full gradient, effective fine gradient, and coarse tracking correction. The velocity has the negative sign. |
| $e_C$, $e_H$ | Constant/linear residual coefficients and their full orthogonal complement. |
| $M$, $C_6$ | Total particle second moment and normalized sixth moment, defined in Section 6. |

\newpage

**The empirical starting point.** Across thirteen targets, the orange tracking
force drops below the blue effective fine force after the initial transient.
Black then follows blue. The derivation below explains this separation; the
later question is why the surviving force often produces so little scale travel.

![Full slope-gradient norm (black), effective fine-force norm (blue), and coarse-tracking norm (orange) during ordinary GD. Each panel shows five seeds; lines are medians and shading is the full seed range, not a confidence interval. Width 177, 2,048 training points, learning rate 0.002, through 600k updates. The norm axis is logarithmic; the update axis is logarithmic above update 10. Blue includes balanced coarse compensation. The vectors add, but their norms need not. These are sampled force magnitudes, not outward-motion or acquisition measurements. Target labels are defined in Appendix B.1.](figures/d34_pi_force_decomposition.png)

\newpage

## 1. Start with the coupled training dynamics

**Example and objective.** At reference resolution $N_{\rm ref}=512$,
$\lambda=0.25$ means $|a|=64$. At $N_{\rm ref}=1024$ it means $|a|=128$.
The question therefore concerns scale relative to resolution, rather than a
fixed physical slope. The numerical construction places transitions on a grid
of spacing $h$ and uses $|a|$ of order $1/h$ to balance resolution and
conditioning. The particular
threshold $0.25$ is an informative construction benchmark, not a proved
necessary or sufficient condition for every possible approximation.

For inputs $x_i\in[-1,1]$, let

$$
f_\theta(x)=d+\sum_{j=1}^Wc_j\tanh(a_jx+b_j),\qquad
r=f_\theta-y,\qquad L(\theta)=\tfrac12\langle r,r\rangle_m,
$$

where $\langle u,v\rangle_m=m^{-1}\sum_i u(x_i)v(x_i)$.
Writing $u_j=a_jx+b_j$, differentiation gives

$$
\begin{aligned}
g_{a,j}&=c_j\langle r,x\operatorname{sech}^2u_j\rangle_m,
&g_{b,j}&=c_j\langle r,\operatorname{sech}^2u_j\rangle_m,\\
g_{c,j}&=\langle r,\tanh u_j\rangle_m,
&g_d&=\langle r,1\rangle_m.
\end{aligned}
\tag{1}
$$

The gradient-flow ODE is $\dot\theta=-g(\theta)$. Ordinary full-batch GD is
$\theta_{n+1}=\theta_n-\eta g(\theta_n)$, with every component evaluated at
the same old state. In particular, **$g_c$ always means the readout gradient**;
it does not denote a coarse gradient. These are noiseless coupled dynamics:
readouts influence slope sensitivity, while moving slopes and biases changes
which errors the readouts can fit.

The persistence statements below restart the clock at an observed
post-transient checkpoint. They do not attempt to explain the initial
coarse-fitting phase or prove entry into the assumed regime.

Equation (1) already explains why a large residual need not produce large
slope movement. Its contribution depends on its correlation with the current
tangent feature $c_jx\operatorname{sech}^2u_j$. Better frozen-geometry readout
fitting can reduce error without acquiring the intended slope regime. Such
improvement is not evidence of precision approximation or of successful
geometry learning.

**Prediction to test.** Identify the error components that actually produce
slope motion, then explain how their sensitivities evolve. A loss curve alone
cannot distinguish small driving error from poor coupling to a large error.

## 2. Separate coarse tracking from the force that survives balance

**Motivating observation.** Figure 1 makes the reduction visible: after the
transient, the full slope force follows the effective fine force, including
its later recovery on some targets. This is not a claim of universal force
decay. Across 60 starts on ten additional target functions,
the slope-gradient correction beyond the effective fine force is small after
coarse fitting. We need a decomposition that retains what coarse fitting still
does to the effective dynamics; simply deleting the constant and linear
residual from (1) would miss that effect.

Choose a fixed empirical orthonormal basis with coarse space
$C=\operatorname{span}\{1,x\}$. Let $e_C$ be its two residual coefficients,
and $e_H$ the full orthogonal complement. We write $\|e_H\|_m$ for the norm
of the reconstructed fine residual, equal to its coefficient-vector norm.
Write $J_C=D_\theta e_C$ and $J_H=D_\theta e_H$. Transposes below are
adjoints in these orthonormal coordinates. Then

$$
g=J_C^Te_C+J_H^Te_H.
$$

If polynomial basis functions are used, a *quadratic error* is the coefficient
along the degree-two orthogonal polynomial, and a *cubic error* is the
degree-three coefficient. This is an exact decomposition of residual shapes,
not a Taylor approximation. A Taylor approximation of the activation enters
separately in Section 5.

\Needspace{8\baselineskip}

Assume $K=J_CJ_C^T$ is invertible. Define

$$
\begin{aligned}
g_H&=J_H^Te_H,& \ell&=K^{-1}J_Cg_H,\\
\Pi&=I-J_C^TK^{-1}J_C,& F&=\Pi g_H,\\
z&=e_C+\ell,& R&=J_C^Tz.
\end{aligned}
\tag{2}
$$

These definitions give the **exact identity** $g=F+R$ and $J_CF=0$.
The effective force $F=g_H-J_C^T\ell$ includes the compensating coarse
response. The variable $z$ measures departure from that instantaneous balance:
$\dot e_C=-Kz$. Thus small $z$ is a tracking statement. It is neither
$e_C=0$ nor optimal least-squares fitting of every readout.

At exact balance, $R=0$ but the compensation $-J_C^T\ell$ usually remains.
This distinction is essential: **small coarse disequilibrium does not mean
coarse compensation is irrelevant**. Also, $z=0$ is not automatically an
invariant manifold of full GD. The reduced flow $\dot\theta=-F$ is a model
whose discrepancy must be controlled.

\Needspace{6\baselineskip}

The remaining errors evolve according to

$$
\dot e_H=-S e_H-J_HJ_C^Tz,\qquad
S=J_H\Pi J_H^T\succeq0.
\tag{3}
$$

Equations (2)-(3) expose the two corrections to audit: direct tracking force
on slopes and tracking's effect on the errors driving later motion. They also
show why the model remains coupled when tracking is small: $S$, $\Pi$, and
$J_H$ all change with the parameters.

Some experiments retain polynomial degrees only through 65. Their identity
instead includes an omitted-residual term $g_\perp$; their reported remainder
combines tracking and this term. The full-complement formulas used in the
theorem and the later population audit have no omitted-residual term.

**Prediction and evidence.** The ten new functions comprise two each of
exponentials, off-center Gaussians, compact bumps, tanh steps, and continuous
kinks. Two seeds and forks at updates 100k, 400k, and 600k give 60 starts at
$W=177$, with $m=2048$ and $\eta=0.002$. The functions were fixed before
their outcomes were inspected.

**Table 1. Effective-force dominance on the new-function panel.** Each row
summarizes the same 60 ordinary-GD continuations. Ratios compare Euclidean
slope-gradient norms, except the final row, which compares signed sums.

| Diagnostic | Median | Maximum |
|---|---:|---:|
| Remainder / effective slope force at the fork | 0.0206% | 1.03% |
| Same ratio after 200k additional updates | 0.0103% | 0.0958% |
| Accumulated signed remainder-travel / effective-travel vector norms | -- | 0.389% |

These measurements support force attribution, not uniform bounds between
saved states. Signed
cancellation also makes the last row different from an absolute travel budget.
A broader cross-function audit of 1,665 saved states from 333 starts found
direct tracking slope-force ratios at most 1.24%, and tracking-induced
fine-residual forcing ratios at most 0.70%. The latter compares the norms of
the two right-hand-side terms in (3). Neither study turns 200k into a universal
barrier time.

## 3. What keeps the effective force weak?

**Example.** At width 1409, removing the model's residual-relaxation feedback
changes positive normalized slope travel over 20k additional updates by a
median 0.00505%. Doubling geometry feedback changes it by 1.51%. The arms
match the entire gradient at the initial fork and retain tracking evaluated
at their own states. This comparison concerns how the force changes, rather
than giving one arm a larger initial push.

\Needspace{8\baselineskip}

For the pure effective flow, put $T=\Pi J_H^T$, so $F=Te_H$.
Differentiating along $\dot\theta=-F$ gives

$$
\frac{d}{dt}\frac{\|F\|^2}{2}
=-\underbrace{\|J_HF\|^2}_{\text{residual relaxation}}
-\underbrace{F^T\bigl(DT[F]\bigr)e_H}_{\text{geometry and compensation feedback}}.
\tag{4}
$$

The first term always depletes force. The second is signed: changing features,
readouts, biases, and coarse compensation can reinforce or suppress it. Full
flow adds $-F^TDF[R]$. Equation (4) identifies a mechanism to investigate,
rather than imposing a favorable sign on all motion.

There are two compatible routes to slow acquisition. **Generated-error
correction** can oppose expansion: a broad feature creates unwanted quadratic
or cubic output, and reducing that error can be easier than resolving the
target's finer structure. **Weak reinforcement** can also keep motion small
when those errors barely relax: the population's sensitivity itself changes
too slowly to produce a much stronger force. The latter matters for sine and
other targets without degree-nine orthogonality.

For a concrete local Taylor illustration of generated-error correction, take
one zero-bias feature, hold its readout $c$ fixed, and retain its cubic coefficient
$A(a)=-\chi ca^3$, with $\chi>0$ set by the polynomial normalization.
If the target has no cubic component, the cubic-error loss $A(a)^2/2$
alone gives $\dot a=-3\chi^2c^2a^5$, which shrinks either sign of slope.
A nonzero target cubic component changes this force, and joint training
adds the coupling in (2). This scalar illustration explains a possible
contribution, not the net direction of the full population.

\Needspace{20\baselineskip}

**Table 2. Experiments distinguish slow motion from a universal restoring
force.** These are finite-window, matched comparisons; they do not establish
a sign rule for every neuron or target.

| Test | Observation | Consequence for the explanation |
|---|---|---|
| Feedback interventions, 12 cases at $W=1409$, 20k updates | Residual-relaxation removal: 0.00505% median travel change; doubled geometry feedback: 1.51% | Geometry feedback matters more here, but neither perturbation produces rapid reinforcement. |
| Small outward pulses, 23 functions and two seeds | At the smallest tested amplitude, 99.0-100.3% of the directional offset remains after 20k updates | Strong restoration is absent in the tested directions. |
| Exact fourfold neuron cloning, development cohort | Ordinary cloned motion is about 0.275 of original; compensating geometry mobility gives about 1.05; compensating only readout mobility changes little | A universal readout-dissipation explanation fails this controlled comparison. |

The cloning test preserves the represented function by splitting each readout
across identical copies. It divides each copy's geometry mobility by four
while multiplying aggregate readout mobility by four. Compensating both
recovers the original trajectory to the reported numerical precision. This
separates two effects that width changes alone would confound. Readouts still
matter through the coupled force; the test rejects their accelerated fitting
as a sufficient explanation of the clone slowdown.

The feedback study contains 40 forks and 240 branches. The outward pulses
match coarse outputs, coarse disequilibrium, and the slope gradient to first
order; symmetric pulses at halved amplitudes check the finite perturbation.
Separate physical parameter kicks provide a caution:
first-order matching can still create substantial tracking at finite amplitude.
Those contaminated kicks are not clean tests of a low-tracking mechanism.

**The refined prediction.** Wide, diffuse populations should exhibit weak
reinforcement even when some slopes expand and the fine residual remains
almost fixed. A useful theory should bound collective travel without requiring
universal contraction, common signs, or a stable equilibrium.

## 4. The population is heterogeneous, but its motion is small

**Example.** In the width-1409 panel, particles violating an earlier theorem's
alignment/readout-dominance condition constitute a median 71.5% of the
population and account for 70.2% of outward travel. Yet the largest endpoint
increase in the population maximum of $\lambda$ is only $9.43\times10^{-6}$
over 20k updates. These are ordinary, slowly moving populations, not an aligned
population with a negligible exceptional set.

The population audit contains 223 distinct checkpoints from 23 target
instances. None satisfies
the old aligned, zero-hidden-bias sector: after choosing neuron orientations,
that sufficient regime requires $c_j\ge a_j\ge0$ and $b_j=0$ for every neuron.
The median negative slope-readout
product fraction is 44.6%; hidden biases remain substantial in rescaled
coordinates. This falsifies that sector as an explanation of these states,
while leaving its conditional mathematical result intact.

There is nevertheless a coherent width pattern. Set
$X_j=(\alpha_j,\beta_j,\zeta_j)=\sqrt W(a_j,b_j,c_j)$.
At three independently initialized widths, typical rescaled slopes stay
comparable, whereas physical slope forces shrink strongly with width.

\Needspace{11\baselineskip}

**Table 3. Width scaling at the 20k-update forks.** Medians over six targets
and two seeds at each width; force values use the retained modal audit.

| $N_{\rm ref}$ | Actual $W$ | $\operatorname{RMS}(\sqrt W a)$ | $W^{3/2}\operatorname{RMS}(F_a)$ |
|---:|---:|---:|---:|
| 128 | 177 | 1.456 | 0.628 |
| 512 | 705 | 1.471 | 0.695 |
| 1024 | 1409 | 1.466 | 0.661 |

The six functions are degree five, mixed sine, an off-center Gaussian, a
compact bump, a tanh step, and a kink. No case reaches $\lambda=0.25$ through
40k total updates. Three widths support this regime; they do not establish
an asymptotic law. The latest audit also finds that, at $W=1409$, the endpoint
full-force norm after 20k further updates is only 0.984-1.045 times its initial
value across the twelve paths. Endpoint comparisons do not bound intervening
peaks. They identify the persistence that a theorem must explain.

**Mechanistic interpretation.** Fine motion depends on mixed population
moments, not just slope size or a single readout norm. For example,
$\mathbb E_W[\alpha\zeta]$ controls a leading linear coefficient, whereas
$\mathbb E_W[\alpha^3\zeta]$ helps control a cubic coefficient. Positive and
negative groups can affect these moments differently. Hidden biases add more
mixed moments. Shared coarse compensation couples all groups, so a group
carrying outward motion may simultaneously supply compensation opposing motion
elsewhere. Discarding all misaligned neurons loses much of the mechanism.

**Prediction.** Control a few population moments and their correlations.
The conclusion should concern the fraction of neuron labels that ever travel
far enough, allowing individuals to move in different directions.

## 5. A useful coupled surrogate, with a visible domain of validity

**Example.** At supplied wide-network states, a quintic approximation to tanh
reconstructs the exact effective slope force with median relative errors of
0.0396% at width 705 and 0.00863% at width 1409. This supports a simpler local
description, provided we retain the actual heterogeneous slopes, biases, and
readouts.

The approximation is now a Taylor expansion of the activation,
$\tanh u\approx u-u^3/3+2u^5/15$. With
$M_{ijk}=\mathbb E_W[\alpha^i\beta^j\zeta^k]$, its output is

$$
\begin{aligned}
f_5(x)={}&d+M_{011}+xM_{101}\\
&-\frac{1}{3W}\sum_{r=0}^3\binom3r M_{r,3-r,1}x^r
+\frac{2}{15W^2}\sum_{r=0}^5\binom5r M_{r,5-r,1}x^r.
\end{aligned}
\tag{5}
$$

After removing affine output, the leading nonlinear output carries a factor
$1/W$. This is the origin of weak fine sensitivity in the diffuse regime.
The surrogate evolves its parameters and recomputes its compensation, rather
than freezing an initial Jacobian. It is still a coupled dynamical system;
a closed system for just a few moments would require additional justification.

![Quintic force accuracy is strong in the wide panel but fails in late sine. Each point is a median and each segment a full range. Left: 96 sampled states per width, comprising twelve cases and eight horizons. Right: four late cases kept separate; source indices in the labels are archive indices, not seeds. Blue is cubic, orange quintic; triangles restrict to the largest 10% of slopes. These are force reconstructions at actual states, not forecasts.](../results/checkpoint_D_optimizers/expD34_readout_race/population_coverage/evidence/summary/polynomial_error.png)

**Prediction and separate forecast test.** If the evolving surrogate captures
the relevant coupling, it should predict motion from its own state without
receiving future GD checkpoints. On a separate seed-confirmation panel,
the initially anchored quintic field has median relative slope-displacement
errors of 0.876%, 0.0547%, and 0.0260% at widths 177, 705, and 1409 over 20k
updates. The corresponding constant-force forecast errors are 20.0%, 4.71%,
and 2.26%. Anchoring adds a fixed correction computed at the initial state;
no future coefficients are fitted. The unanchored quintic errors are 2.86%,
0.0834%, and 0.0324%. These forecasts are stronger evidence than reconstructing
force at supplied states, but remain
finite-window tests on those functions and seeds.

Late pure sine is an important failure case. Its polynomial force can be wrong
by orders of magnitude. Even the exact-force scalar growth forecast can miss
because geometry reinforcement declines while residual relaxation strengthens.
Thus neither ninth-degree orthogonality nor one frozen growth rate explains
all targets. The exact decomposition (2) survives these failures; the polynomial
approximation and frozen-feedback models do not have universal validity.

## 6. A population theorem that exposes the missing assumption

**Motivation.** We do not need to predict every slope accurately. We need to
show that enough slopes cannot travel from their small initial values to
$\lambda_*$ quickly. The width evidence suggests preserving a diffuse
population, rather than a common sign pattern. The following proposition
makes that idea precise without assuming that the future force is small.

Define the empirical particle moments

$$
M=\mathbb E_W|X|^2=\sum_j(a_j^2+b_j^2+c_j^2),\quad
M_6=\mathbb E_W|X|^6,\quad C_6=M_6/M^3.
\tag{6}
$$

$C_6$ measures concentration: it grows when a small group carries a
disproportionate share of the particle magnitude. Uniformly scaling every
particle leaves $C_6$ unchanged. A bound on $C_6$ therefore does not assume
small slopes, a small second moment, or little motion.

**Proposition 1: conditional delay under effective flow.** Consider
$\dot\theta=-F$ with the full complement in (2), $|x_i|\le1$, and a
well-defined coarse projector throughout $[0,T]$. Suppose $M_0>0$ and
$C_6(t)\le\overline C_6$ on that interval. Let

$$
Y_0=\|e_H(0)\|_m,\qquad
k=\frac{6Y_0\sqrt{\overline C_6}M_0}{W}.
$$

For $0\le t\le T$ with $kt<1$, the total physical parameter path and second moment obey

$$
\int_0^t\|F(s)\|\,ds\le A(t):=
\sqrt{M_0}\bigl[(1-kt)^{-1/2}-1\bigr],\qquad
M(t)\le\frac{M_0}{1-kt}.
\tag{7}
$$

For $0\le\lambda_0<\lambda_*$, let $p_0$ be the fraction initially above
$\lambda_0$. The fraction of labels ever reaching $\lambda_*$ by time $t$
satisfies

$$
p_{\rm ever}(t)\le
\min\left\{1,\ p_0+
\frac{h^2 A(t)^2}{W(\lambda_*-\lambda_0)^2}\right\}.
\tag{8}
$$

Appendix A proves this for exact tanh, not a polynomial substitute. The central
step is a structural sensitivity bound,

$$
\|F\|\le\frac{3}{W}\|e_H\|_m\sqrt{M_6}.
\tag{9}
$$

Affine projection removes the leading Jacobian, leaving a cubic-size
remainder; the coarse compensation is an orthogonal projection and cannot
increase its full norm. Under effective flow the fine loss decreases. These
facts close a differential inequality for $M$ under the shape assumption.
Small future force is a consequence, not a premise.

For a desired additional crossing fraction $\varepsilon>0$, (8) remains at
most $p_0+\varepsilon$ whenever

$$
t\le \frac{1}{k}\left[
1-\left(1+
\frac{\sqrt{\varepsilon W}(\lambda_*-\lambda_0)}{h\sqrt{M_0}}
\right)^{-2}\right],
\tag{10}
$$

within the assumed interval and with $k>0$. At fixed $kt<1$, bounded
$M_0,Y_0,\overline C_6$ give a time scale proportional to $W$ and a small
population crossing allowance. If also $h$ is proportional to $1/W$, that
allowance in (8) is $O(W^{-3})$ for a fixed normalized gap. This is a conditional
width prediction, not an evaluated practical horizon. It is an absolute speed
bound; it need not preserve an unusually tiny measured initial force.

**Ordinary GD requires its own statement.** Suppose the projector (2) exists
at each pre-update state of actual GD, and $L_n\le L_*$,
$C_{6,n}\le\overline C_6$, and
$\|R_{abc,n}\|\le\delta_n/W$, with $\delta_n\ge0$; the subscript excludes
the output bias. Set

$$
\begin{aligned}
\overline m_0&=\sqrt{M_0},\\
\overline m_{n+1}&=\overline m_n+
\frac{\eta}{W}\left(3\sqrt{2L_*\overline C_6}\,
\overline m_n^3+\delta_n\right).
\end{aligned}
\tag{11}
$$

Then $\sqrt{M_n}\le\overline m_n$, and (8) holds through update $N$ with
$A(t)$ replaced by $\overline m_N-\overline m_0$. Appendix A proves this
directly from simultaneous GD. Loss descent needs a justified step size, and
the tracking assumption concerns all slope, bias, and readout coordinates.
Small measured slope tracking alone does not supply that premise. This
corollary shows exactly where tracking control enters a population argument;
it is not yet a certified bound for the experimental continuations.

**Assumptions to earn.** Small initial normalized scales and broad particle
distributions are observed. Effective-force dominance and slow moment changes
are observed in specified windows. But a useful interval-wide bound on
$C_6$, the required full tracking budget, and their preservation from initial
data have not been established by these observations. The proposition names
this gap rather than hiding it inside a future-force assumption.

## 7. What is proved, and what would complete the mechanism?

**A concrete comparison.** At width 1409 the observed second moment changes
by only a factor 0.99790-1.00433 between the endpoints of a 20k-update
continuation. Nevertheless, FP64 evaluation of our latest initial-data
persistence bounds reaches at most 38 GD updates on the archived panel. That mismatch is
a limitation of the sufficient estimates, not a prediction that the dynamics
escape after 38 steps.

**Table 4. Current proof and evidence status.** The different rows answer
different questions and should not be combined into one claimed horizon.

| Result | What is established | What it does not establish |
|---|---|---|
| Exact decomposition and Proposition 1 | Effective-force identity; conditional population travel bound with a complete proof | Preservation of the proposed population shape on an informative interval |
| Generic GD energy baseline | Outward-rounded initial checks for 18 width-panel states exclude $\lambda=0.25$ for 50k further updates | Why the effective force remains weak; a universal 50k barrier |
| Existing exact-tanh persistence proof | Derives conditioning, force, and tracking bounds from initial data, allowing biases and mixed signs | Useful constants: only 22/223 states admitted, all at width 1409; best refined interval 38 updates |
| Coupled quintic forecasts | Accurate 20k motion forecasts on the specified wide confirmation panel | A regional error certificate or a valid late-sine model |

The energy baseline is ordinary-GD mathematics,
with verified assumptions at the archived 20k states of six targets and three
widths. It reaches nominal update 70k and already rules out a distant threshold
without any effective-force analysis. Mechanistic theorems must add an
explanation of rates, width dependence, or much longer persistence, rather
than claiming that threshold exclusion alone validates the mechanism.
The latest persistence audit evaluates its bounds in FP64, without interval
certification; its short valid
windows should be reported alongside the successful observations.

**The next theoretical question is preferential growth.** For a nonzero
particle population, exact differentiation gives

$$
\frac{d}{dt}\log C_6
=6\left[
\frac{\mathbb E_W(|X|^4 X\cdot\dot X)}{M_6}
-\frac{\mathbb E_W(X\cdot\dot X)}{M}
\right].
\tag{12}
$$

Concentration increases when large particles receive disproportionately
positive radial motion. Thus the missing shape condition has a concrete
mechanistic test: attribute the two weighted radial-growth terms to effective
fine correction and shared compensation, retaining mixed signs and biases.
An upper bound on their difference, derived from mixed moments of the ODE,
would propagate $C_6$ and close the population argument. This is more useful
than controlling every neuron's position separately. It remains a research
target, not a proved invariant or an already measured empirical success.

The discriminating follow-up is to measure (12) on existing trajectories,
then perturb concentration or the relevant mixed moments while matching
initial coarse balance and force as closely as possible. If stronger
preferential growth accelerates acquisition, the shape mechanism gains a
causal test. If concentration grows but scale travel remains small, a bound
using signed outward coupling or the sensitivity actually reached by the
residual should replace the isotropic sixth-moment bound. Either result changes
the theorem's assumptions; it is not a search for smaller constants alone.

**Scope across targets and optimizers.** The evidence extends beyond degree
nine and sine: 23 target instances in the broad audit, ten separately chosen
new functions in the force test, and six families in the width/intervention
panel. Repeated checkpoints are not independent target samples, and the
wide-panel conclusions have not been established for all 23 functions.
The horizon is part of each statement. Eventual movement after millions of
updates does not refute a finite-time slowdown, and long-run forecast failures
do not define its practically relevant duration.

Adam needs a different dynamical theorem. Its tracking activity can be large
in raw gradients and component step lengths while largely cancelling in net
outward motion. The Adam attribution experiments measure these separately
and also contain escapes in some regimes.
Momentum and adaptive scaling change both the metric and the state. The
Euclidean GD bound above does not transfer to Adam merely because some of its
trajectories also show scale saturation.

\newpage

## Appendix A. Proofs and the transport interpretation

### A.1. Exact-tanh sensitivity after removing affine output

Let $P_H$ be the orthogonal projection, in the empirical function norm, onto
the complement of $\operatorname{span}\{1,x\}$. The output Jacobian columns
for one neuron are $cx\operatorname{sech}^2u$, $c\operatorname{sech}^2u$,
and $\tanh u$. The corresponding affine columns are $cx$, $c$, and $u$;
$P_H$ annihilates them, as it does the output-bias column $1$.

For every real $u$,

$$
|1-\operatorname{sech}^2u|=\tanh^2u\le u^2,
\qquad |\tanh u-u|\le |u|^3/3.
$$

The second inequality follows by integrating $\tanh^2s\le s^2$ from
zero to $u$. With $s_j^2=a_j^2+b_j^2+c_j^2$ and $|x|\le1$,
$|u_j|^2\le2s_j^2$. Summing squared column errors gives

$$
c_j^2(1+x^2)u_j^4+u_j^6/9
\le 8s_j^6+\tfrac89s_j^6
=\tfrac{80}{9}s_j^6.
$$

Projection is contractive, so the Hilbert-Schmidt bound, hence also the
operator bound, is

$$
\|J_H\|^2\le\frac{80}{9}\sum_j s_j^6
=\frac{80}{9W^2}M_6 \le\frac9{W^2}M_6.
\tag{A1}
$$

Because $\Pi$ is an orthogonal parameter-space projector,
$\|F\|=\|\Pi J_H^Te_H\|\le\|J_H\|\|e_H\|_m$, proving (9).
No parity, sign alignment, or individual support cap is required. The
inequality is global for tanh, although useful scaling requires controlled
moments.

### A.2. Closing the effective-flow moment bound

Under $\dot\theta=-F$,

$$
\frac{d}{dt}\tfrac12\|e_H\|_m^2
=-e_H^TJ_H\Pi J_H^Te_H=-\|F\|^2\le0.
\tag{A2}
$$

Let $p=(a,b,c)$, so $M=\|p\|^2$. Using (9) and the shape assumption,

$$
\dot M=-2p^TF_{abc}
\le2\sqrt M\|F\|
\le\frac{6Y_0\sqrt{\overline C_6}}{W}M^2.
\tag{A3}
$$

Comparison with the scalar equality gives $M(t)\le M_0/(1-kt)$.
Substituting this upper bound in
$\|F\|\le3Y_0\sqrt{\overline C_6}M^{3/2}/W$ and integrating gives
(7). If $Y_0=0$, then $F=0$ and the trajectory is stationary; (7) is read
with $A=0$. The proposition assumes the coarse solve remains defined and
does not derive that premise from (A3).

### A.3. From collective travel to ever-acquired labels

For each neuron set $v_j(t)=\int_0^t|\dot a_j(s)|\,ds$. Minkowski's
inequality gives

$$
\left(\sum_jv_j(t)^2\right)^{1/2}
\le\int_0^t\|\dot a(s)\|\,ds\le A(t).
$$

A label initially satisfying $h|a_j(0)|\le\lambda_0$ can reach
$\lambda_*$ only if $v_j(t)\ge(\lambda_*-\lambda_0)/h$.
Counting these labels by their squared travel gives (8), including crossings
followed by a return below threshold. It is therefore stronger than an
endpoint count. Solving $h^2A^2/[W(\lambda_*-\lambda_0)^2]\le\varepsilon$
gives (10). No assumption on the direction of individual velocities was used.

### A.4. Direct discrete-GD proof

At each iterate the complete empirical residual obeys
$\|e_{H,n}\|_m\le\sqrt{2L_*}$. The exact identity and (A1) imply

$$
\begin{aligned}
\|p_{n+1}-p_n\|
&\le\eta\bigl(\|F_{abc,n}\|+\|R_{abc,n}\|\bigr)\\
&\le\frac{\eta}{W}
\left(3\sqrt{2L_*\overline C_6}\,\|p_n\|^3+\delta_n\right).
\end{aligned}
$$

The right side is monotone in $\|p_n\|$. The triangle inequality and
induction give $\|p_n\|\le\overline m_n$. Summing the increment bounds
then gives $\sum_{n<N}\|p_{n+1}-p_n\|\le\overline m_N-\overline m_0$.
The discrete version of A.3 proves the asserted population count.
This proof uses the actual GD step; it does not replace $t$ by $\eta N$
in a continuous-time theorem. Loss descent, full tracking control, and shape
persistence remain explicit premises to verify or derive separately.

### A.5. What the transport PDE contributes

Associate each neuron with $\xi_j=(a_j,b_j,c_j)$ and let
$\rho_t=W^{-1}\sum_j\delta_{\xi_j(t)}$. For gradient flow, its weak
transport equation is

$$
\partial_t\rho+\nabla_\xi\cdot(\rho v[\rho,d])=0,\qquad
f(x)=d+W\int c\tanh(ax+b)\,d\rho,
\quad \dot d=-\langle r,1\rangle_m.
\tag{A4}
$$

Here $v$ consists of the three negative gradients in (1). It depends on
$\rho$ through the shared residual. Effective flow replaces both particle
velocities and the output-bias equation by the corresponding components of
$-F$, incorporating the shared coarse projector. The PDE is nonlinear, with no
diffusion because training is deterministic. For the finite empirical measure
it is an exact rewriting of the particle ODEs, not an additional limiting
assumption.

Its useful contribution is to organize the theorem around transported mass,
moments, and accumulated action. Equation (A3) bounds a population moment;
(8) bounds mass whose characteristics ever cross a scale threshold.
Transport alone does not prevent concentration or reinforcement. The missing
step is a property of this velocity field, captured by (12), rather than a
generic consequence of noiseless transport. Discrete GD transports the
empirical measure by a map at each update; A.4 supplies the corresponding
population bound without introducing artificial diffusion.

## Appendix B. Experimental methods and coverage

### B.1. Training setup and target functions

The width studies use 2,048 equally weighted midpoint inputs
$x_i=-1+2(i+1/2)/2048$, $i=0,\ldots,2047$, and binary64 arithmetic.
All parameters are trained by simultaneous, deterministic full-batch GD with
$\eta=0.002$. Initially, $a,b,c$ are independent uniform draws from
$[-\sqrt{6/(W+1)},\sqrt{6/(W+1)}]$, with $d=0$. Initializations are paired
across targets at a fixed width and seed, and generated independently across
widths. We use $(N_{\rm ref},W)=(128,177),(512,705),(1024,1409)$.
The difference includes the extra neurons used beyond the reference grid.

**Table B1. Six targets in the width and feedback panels.** Except for the
already normalized polynomial target, each listed function is divided by its
RMS on the original training grid. That normalization stays fixed during
training and evaluation; target means are not subtracted for training.

| Target | Function before RMS normalization |
|---|---|
| Degree five | $0.3q_0+0.4q_1+\sqrt{0.75}\,q_5$; $q_k$ are empirical orthonormal polynomials with positive leading coefficient |
| Mixed sine | $\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$ |
| Off-center Gaussian | $\exp(-((x+0.35)/0.22)^2)$ |
| Right compact bump | $\exp(1-1/(1-u^2))$ for $\lvert u\rvert<1$, otherwise zero; $u=(x-0.35)/0.22$ |
| Right step | $\tanh(14(x-0.31))$ |
| Kink | $\lvert x+0.23\rvert$ |

The degree-five label thus denotes a prescribed orthogonal residual shape,
not the monomial $x^5$. The same distinction applies to the degree-nine
control. The development width panel has two seeds and starts its continuation
after 20k updates; its 20k further updates end at 40k total. The quintic
confirmation repeats the six targets and three widths with two new seeds.
It checks seed variation, not generalization to unseen functions. Where
independent-grid fitting errors are measured, evaluation uses 8,192 midpoint
inputs; the force and trajectory quantities here concern the training grid.

**The thirteen targets in Figure 1.** Sine is $\sin(2\pi x)$, and Runge is
$1/(1+25x^2)$; these two functions are not RMS-rescaled. Degree $d$, for
$d\in\{3,4,5,9\}$, denotes
$0.3q_0+0.4q_1+\sqrt{0.75}\,q_d$, using the same empirical orthonormal
polynomials as Table B1. Each of these polynomial targets has unit empirical
RMS.

Let $S(x)=\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$.
Mixed sine is $S(x)$ divided by its training-grid RMS. Localized sine is
$S(x)\exp[-(x/0.4)^2/2]$, divided by its own RMS. Chirp is
$\sin[2\pi(u+4u^2)]$, with $u=(x+1)/2$, likewise divided by its own RMS.
All these normalizers are fixed from the original 2,048 training points.
Finally, the four panels labeled Mix $s$ use

$$
y_s(x)=0.3q_0(x)+0.4q_1(x)
+\sqrt{0.75}\left[sq_3(x)+\sqrt{1-s^2}\,q_9(x)\right],
\qquad s\in\{-0.1,0.01,0.1,0.3\}.
$$

These blends already have unit empirical RMS. Figure 1 uses five seeds
numbered 0 through 4, with the same width-177 initialization law and training
grid described above. Its full-complement decomposition has no omitted
modal-residual term. At each saved state, the median and range are taken
across the five seeds separately for each force norm. The shading is neither
a confidence interval nor a bound on the unsampled times between states.

### B.2. What the interventions change

The feedback experiment separates two evolving contributions while preserving
the initial force. At a fork $\theta_0$, save $e_{H,0}$ and $F_0$. At the
current state define

$$
A(\theta)=\Pi(\theta)J_H(\theta)^Te_{H,0},\quad
B(\theta)=F(\theta)-A(\theta),\quad
C(\theta)=\Pi(\theta)F_0.
$$

Use the modified gradient

$$
g_{\kappa,\nu}=g+(\kappa-1)(A-C)+(\nu-1)B.
\tag{B1}
$$

At the fork, $A=C=F_0$ and $B=0$, so all arms have the same gradient.
The six choices $(\kappa,\nu)=(1,1),(0,0),(0,1),(2,1),(1,0),(1,2)$
respectively give ordinary GD, a projected constant field, removed or doubled
geometry feedback, and removed or doubled residual-relaxation feedback.
Each arm retains its own current tracking correction. These are controlled
vector-field interventions, not all gradients of a common modified loss.

For the pulse experiment, an outward parameter direction lies in the common
nullspace of the derivatives of coarse output, coarse disequilibrium, and the
actual slope gradient. Symmetric amplitudes $\pm0.01,\pm0.005,\pm0.0025$
are measured in the experiment's relative block-RMS units. The offset-retention
numbers in Table 2 use the smallest amplitude. Halving checks distinguish the
predicted first-order response from finite-amplitude matching errors.

For cloning, $k$ identical copies each receive readout $c_j/k$. The function
and residual are initially unchanged. Replica symmetry makes the geometry
rate $1/k$ of the original and the aggregate readout rate $k$ times the
original. Multiplying geometry learning rates by $k$, or dividing readout
learning rates by $k$, isolates these effects. The construction spacing $h$
is held fixed because this test changes parameterization, not resolution.

\Needspace{28\baselineskip}

### B.3. Coverage and numerical meaning

**Table B2. Evidence roles.** These studies overlap. Their counts must not be
added and interpreted as independent samples of target functions.

| Study | Design and role |
|---|---|
| Opening force figure | Thirteen targets and five seeds; 65 ordinary-GD trajectories through 600k updates at width 177. Full-complement decomposition at saved states. |
| New-function force test | Ten additional functions in five families, two seeds, three forks; 60 starts. Functions fixed before inspection of their outcomes. |
| Width development / confirmation | Six targets, two seeds, three widths per panel; 36 cases each. Starts at 20k, followed for 20k further updates. |
| Feedback interventions | 36 width cases plus four late sine/degree-nine starts; six arms each, 240 branches. |
| Pulse and cloning tests | 23 target instances in separate development and confirmation cohorts. Matched-fork comparisons. |
| Population audit | 223 deduplicated static states; 40 natural continuations with eight snapshots each. Retrospective structural and approximation tests. |
| Persistence-bound audit | The same 223 states; seven fixed region choices per state. Duration selected from initial information. |
| Energy verification | Eighteen initial states: six targets, three widths, one seed. Outward rounding checks sufficient GD assumptions. |

The ordinary floating-point diagnostics support the reported empirical
comparisons, not interval-wide rigorous bounds. In particular, sampled force
ratios do not control their maxima between snapshots. The energy verification
is different: outward-rounded checks establish its sufficient initial
inequalities for exact-real GD starting from the stored binary64 data. It does
not certify all earlier training or every rounding error in later machine
updates.

For completeness, the generic energy argument underlying Table 4 uses a
descent factor $\mu>0$ satisfying
$L_{n+1}\le L_n-\eta\mu\|g_n\|^2$. Summation and Cauchy-Schwarz give

$$
\sum_{n<N}\|\theta_{n+1}-\theta_n\|
\le\sqrt{\frac{\eta N L_0}{\mu}}.
\tag{B2}
$$

Bounding the loss Hessian along outgoing steps closes a bootstrap ensuring
descent. For the eighteen stated checkpoints, outward-rounded initial bounds
give $\|(a,b,c)\|\le3$, $L_0<0.417$, and $\eta=0.002<1/490$.
Here is the needed curvature control. On a radius-eight travel region,
$\|(a,b,c)\|\le11$ and the output Jacobian has norm below 16. The next
step has length below $16\sqrt2/490<0.05$, so it lies in a padded
radius-8.05 region. There the squared Jacobian norm is below 246, the output
Hessian norm below 19, and the residual norm along the outgoing step below
2.18. These estimates follow from $|\tanh u|\le|u|$,
$|\tanh' u|\le1$, $\sup|\tanh''|=4/(3\sqrt3)$, and
$\|Df\|^2\le1+2\|(a,b,c)\|^2$.
Consequently $\|D^2L\|<246+19(2.18)<288$, giving
$\mu\ge1-144/490>0.7$. The resulting travel bound is
$\sqrt{\eta(50{,}000)(0.417)/0.7}<8$; all slopes therefore remain below
$3+8=11<16$. Since $N_{\rm ref}\ge128$, this excludes $\lambda=0.25$.
It illustrates how a distant-threshold statement can hold without identifying
the fine-force mechanism.

The scientific question left for review is whether a condition such as (12)
can be derived and shown to persist on the observed mixed-sign populations.
Equations (7)-(11) state what that would buy: a rate and an ever-acquired
population bound, without an accurate forecast of every neuron.
