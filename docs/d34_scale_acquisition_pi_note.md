---
title: "Why gradient descent acquires slope scale slowly"
subtitle: "Mechanism, empirical evidence, and a population theorem"
date: "24 September 2026"
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
broader features and substantial output error. **We want to explain why
deterministic gradient descent can continue fitting without producing accurate
output within a useful training budget.** Population scale acquisition is the
mechanism we study, rather than a substitute for measuring that error.

For a budget of $N$ updates and tolerance $\varepsilon$, success means that
the network's actual readouts and geometry attain raw relative $L^2$ error
$\|f_{\theta_n}-y\|/\|y\|\le\varepsilon$ at some $n\le N$.
We report several tolerances, distinguish training from independent-grid
error, and use the same checkpoint-selection rule for GD and Adam. A separate
readout refit answers a different question about the supplied geometry. A
failed endpoint does not establish failure at every earlier update; sparse
archived samples cannot exclude an intervening crossing.

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
experiments, and connects population structure directly to output-error lower
bounds. The new persistence proof starts from population moments and coarse
tracking at a post-transient state, then derives a window of slow evolution.
It allows the features and their Jacobian to change throughout that window.
Its mathematical validity and the usefulness of its constants at experimental
widths are separate questions. Neither the mechanism nor the theorem requires
every slope to shrink or a trapping equilibrium to exist.

**How to read the plots.** A neuron's physical slope scale is
$\gamma_j=|a_j|$. Its normalized scale is $\lambda_j=h|a_j|$, where
$h=2/N_{\rm ref}$ is the construction spacing. The actual neuron count $W$
includes halo neurons and differs from $N_{\rm ref}$. The opening plots pair
the gradient signals with the scales they produce; the later plots test why
that motion stays small. Every comparison identifies its baseline and the
cases represented by its points or shading.

We keep three gradient names throughout: the full gradient $g$, the effective
fine gradient $F$, and the coarse tracking correction $R$, with $g=F+R$.
Training velocity has the opposite sign. Parameter-vector norms are Euclidean;
function norms use the empirical mean over training inputs. Other symbols
are introduced where they enter the argument.

\newpage

**The empirical starting point.** Read each row from force to scale. The orange
tracking force falls below the blue effective fine force early; black then
follows blue. Degree five and degree nine barely change scale afterward, while
other targets develop later growth. The question is what controls that delay.

![Forces and scales for seven targets under ordinary GD: width 177, 2,048 inputs, learning rate 0.002, five seeds, through 600k updates. Lines show seed medians; shading shows the full seed range. Right: mean and maximum normalized slopes, with $h=2/128=1/64$. The horizontal dashed line is the construction benchmark $\lambda=0.25$. Each vertical dotted line is the median first sampled tracking/effective-force crossover across seeds. All axes are logarithmic. Blue includes balanced coarse compensation; force norms need not add. A maximum measures one extreme, not population acquisition. Targets are defined in Appendix B.1.](figures/d34_pi_forces_scales_1.png)

\newpage

**The same comparison across the remaining targets.** The force crossover
appears throughout this panel, too, but the later scale trajectories differ.
An explanation of persistence must account for this target dependence after
tracking becomes small.

![The remaining six targets, using the same settings, axes, five-seed aggregation, and crossover definition as Figure 1. Scale traces use the same saved states as the forces. The horizontal dashed line is $\lambda=0.25$; no saved maximum reaches it in any of the 65 trajectories across both figures. Some targets show later effective-force reinforcement and scale growth while tracking remains subdominant. These observations concern sampled states, not a proof between samples. A curve's visual slope on logarithmic axes is not its movement per optimizer update.](figures/d34_pi_forces_scales_2.png)

\newpage

**Does the trained network produce accurate output?** On the same thirteen
targets and five seeds, Adam makes substantially more progress than GD.
At 600k updates its median relative error is 1.39%, versus 86.6% for GD.
Nevertheless, 37 of 65 Adam endpoints exceed 1% error, 60 exceed 0.1%,
and all 65 exceed $10^{-4}$. These numbers concern the actual trained
readouts, with no refitting or smoothing. They make the accuracy requirement
explicit: a circuit can improve substantially and still miss the required
precision. The endpoints alone do not establish failure throughout training.

![Output error and mean slope measured at the same three update counts for GD and Adam. Each panel contains thirteen targets and five seeds per optimizer at width 177 and learning rate 0.002. The horizontal line marks 1% relative error on an independent input grid. Larger slopes accompany much better Adam fits, but slope size alone does not determine accuracy. There is no separate checkpoint selection for the two methods.](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/adam_plots/adam_output_vs_scale.png)

The rest of the note asks two different questions in order: why does the
population often remain broad under GD, and why does that population structure
leave substantial output error? Section 7 returns to Adam, where the adaptive
metric and momentum require their own measurements.

## 1. Start with the coupled training dynamics

**Example and objective.** At reference resolution $N_{\rm ref}=512$,
$\lambda=0.25$ means $|a|=64$. At $N_{\rm ref}=1024$ it means $|a|=128$.
The question therefore concerns scale relative to resolution, rather than a
fixed physical slope. The numerical construction places transitions on a grid
of spacing $h$ and uses $|a|$ of order $1/h$ to balance resolution and
conditioning. The particular
threshold $0.25$ is an informative construction benchmark, not a proved
necessary or sufficient condition for every possible approximation.

For inputs $x_i\in[-1,1]$, let $a_j$ denote signed slopes, $b_j$ hidden
biases, $c_j$ readout weights, and $d$ the output bias. The model and loss are

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

**Motivating observation.** Figures 1 and 2 make the reduction visible: after the
transient, the full slope force follows the effective fine force, including
its later recovery on some targets. This is not a claim of universal force
decay. Across 60 starts on ten additional target functions,
the slope-gradient correction beyond the effective fine force is small after
coarse fitting. We need a decomposition that retains what coarse fitting still
does to the effective dynamics; simply deleting the constant and linear
residual from (1) would miss that effect.

The paired scale traces show **an early slowdown accompanying tracking
decay**. In the sine, degree-three, and degree-nine examples, the sampled
mean-slope increment changes from positive before the force crossover to small
or negative near it. The scale levels already change little before the
crossing, so this is not a sharply identified onset time. **Later behavior
differs across targets:** between each seed's crossover and the final saved
state, the median mean-slope ratio is 0.999 for degree nine and 1.002 for degree
five, but 1.72 for sine and 2.49 for degree three. Maximum slopes can grow much
more. Across all 65 runs, tracking first becomes smaller at saved updates
799--4,999 and stays smaller at every later saved state. Thus the early
crossover identifies the regime whose dynamics we should study; it does not
by itself explain its duration. That is the role of the effective-force
feedback studied below.

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

**Read the next plot relative to equal force.** A ratio of 100% would mean
the remainder is as large as the effective fine gradient. The measured ratios
sit far below that reference at both ends of the continuations. The median
falls from 0.0206% to 0.0103%; the largest final ratio is 0.0958%.

![Effective fine gradients dominate on ten additional functions. Each paired case is one of 60 ordinary-GD starts: ten functions, two seeds, and three fork times at width 177. The vertical coordinate is the remainder slope-gradient norm divided by the effective fine slope-gradient norm, expressed as a percentage. The remainder includes coarse tracking and residual modes omitted beyond degree 65. The comparison is between the fork and 200k additional updates; 100% denotes equal force. These are endpoint observations, not bounds between samples.](figures/d34_pi_force_dominance.png)

A separate accumulated measurement reaches the same attribution conclusion:
the norm of the signed remainder-travel vector is at most 0.389% of the
corresponding effective-travel norm. This is not plotted on the force axis
because it measures accumulated motion, with cancellation, rather than an
instantaneous force. Neither measurement supplies an absolute travel budget.
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

**Three visual tests distinguish slow motion from restoration.** In the first
panel, changing geometry feedback has the larger effect, but neither
intervention produces rapid reinforcement in this wide regime. In the second,
100% means a small outward perturbation survives unchanged: the points stay
near that line, rather than returning toward zero. In the third, 1 means the
cloned model moves as much as the original. Correcting the geometry learning
rate largely restores that motion; correcting only the readout rate does not.

![Matched perturbations test three explanations of slow GD motion. The feedback panel measures the absolute percentage change in 20k-update positive normalized slope travel relative to ordinary GD at width 1409: six functions in each of two cohorts. The pulse panel measures the directional offset remaining after 20k updates at the smallest tested symmetric pulse amplitude, across 23 functions in each cohort; 100% means no restoration. The cloning panel compares Euclidean slope-displacement norms with the original network on the two 23-function cohorts; 1 means equal motion. Geometry-rate and readout-rate corrections are distinct interventions. Points retain case variation; the three panels have different units and must be read against their own reference.](figures/d34_pi_interventions.png)

The pulse retention range is 99.0-100.3%. This argues against strong
restoration in the tested directions, rather than proving the population is
at an equilibrium. The clone slowdown likewise has a specific interpretation.

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

**A larger test: double the slopes, then let GD respond.** The earlier pulses
changed mean scale by only about 0.7--3% at the median and deliberately
preserved the initial slope gradient. To test a larger neighborhood, we now
multiply both slopes and hidden biases by 1.25 or 2, preserving feature
centers. We adjust only readouts and output bias to restore $z=0$, then resume
ordinary GD. The repair is chosen near either the original readout $c_0$
(“primary”) or $c_0/s$ (“inverse readout”), where $s$ is the dilation factor.
Neither repair preserves the initial slope gradient or the fine residual.

For a concrete example, doubling the geometry of the left-Gaussian run at
width 1409, seed 30, produces only 0.588% additional mean-scale growth over
20k updates. This is the largest growth among the twelve primary twofold
cases at that width. Some other targets contract. The issue is the rate of
subsequent movement, not a common direction of movement.

![Finite geometry dilations followed by ordinary GD. Each curve starts after the imposed increase, so injected scale is excluded from learned motion. Solid curves are medians and shading is the interquartile range; dashed curves freeze each state's own initial effective slope gradient. Left: 23 targets and two seeds, width 177, after 600k prior updates. Middle and right: six targets and two seeds per width, after 20k prior updates. Every branch continues for 20k updates at learning rate 0.002; flow time 40 means 20k updates. All six branches per checkpoint passed preparation and completed training. Primary and inverse identify readout repair references, not different GD rules. Different checkpoint ages prevent treating the three panels as a controlled width comparison.](figures/d34_pi_finite_dilations.png)

**What the wide states show.** Twofold primary dilation raises the initial
effective slope-force norm by median factors 3.44 and 3.53 at widths 705 and
1409. Yet subsequent median mean-scale growth is only 0.392% and 0.181%; the
largest increases are 1.394% and 0.588%. The force itself grows by median
factors 1.139 and 1.034 during the continuation. Thus there is reinforcement,
but it is limited over this window. Tracking's accumulated slope-norm budget
is below 0.1% of the effective-fine budget in every wide branch. These are
cleaner evidence for persistence of slow effective dynamics than the earlier
gradient-matched pulses.

**What the late states add.** After twofold primary dilation at width 177,
33 of 46 cases contract, but the median contraction is only 0.150% of the
enlarged starting scale. The median gap to the repaired baseline retains
99.17% of its initial size. The initial force burst typically subsides:
its norm rises 13.8-fold at the median, then falls to 17.0% of its post-kick
value. This supports transient correction without a strong return to the
old scale. It does not describe every late state: some ordinary baselines
grow substantially, and the inverse-readout repair reduces contraction to
24 of 46 cases. Late interventions also renew tracking more strongly, with
norm-budget ratios up to 7.21%; their attribution needs that qualification.
Signed contributions can cancel: in one late branch, tracking's contribution
exceeds the net mean-scale change. These late responses are full-GD findings,
not exact effective-flow experiments.

**The mechanism must retain target-side forces.** In the width-1409
mixed-sine example, seed 30, the doubled state has signed effective
mean-scale rate $-6.22\times10^{-9}$. Its target-side cubic contribution is
$-6.18\times10^{-9}$, while generated quadratic and cubic contributions sum
to only about $-1.01\times10^{-11}$. Here initial inward motion is mainly the target's
coupling to the current compensated sensitivities, not correction of generated
lower-mode error. The contributions split the cubic residual coefficient
$f_3-y_3$ while retaining the same compensated sensitivity. That distinction
prevents a universal “unwanted cubic makes
slopes shrink” explanation. A theory must allow target-dependent signs while
explaining why the coupled sensitivity changes slowly.

\Needspace{5\baselineskip}

**The refined prediction.** Wide, diffuse populations should exhibit weak
reinforcement even when some slopes expand and the fine residual remains
almost fixed. The finite dilations support that prediction within the tested
broad-feature neighborhood; they do not reach the desired construction scale
in the wide panels or prove persistence for longer horizons. A useful theory
should bound collective travel without requiring universal contraction,
common signs, or a stable equilibrium.

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

**Read the width comparison in two steps.** First look at the physical slope
force: it becomes much smaller as width increases. Then look at that same
force multiplied by $W^{3/2}$: its median is nearly unchanged. The rescaled
slopes also remain comparable. This separates a weak physical force from a
population that has already changed into a different scale regime.

![Wider populations have much weaker physical slope force at the 20k-update forks. Each width contains six targets and two seeds; points retain individual cases and larger markers summarize medians. The physical-force panel shows the decline in RMS effective fine slope gradient. Multiplying by $W^{3/2}$ gives medians 0.628, 0.695, and 0.661; rescaled slope RMS values are 1.456, 1.471, and 1.466. Actual widths 177, 705, and 1409 correspond to reference resolutions 128, 512, and 1024. Force values use the retained modal audit. A power-law guide is a comparison, not a fitted asymptotic theorem.](figures/d34_pi_width_scaling.png)

Degree five is visibly different: its force decreases faster than the
$W^{-3/2}$ guide over these widths, so its rescaled force also falls.
The median describes typical behavior; individual targets have different rates.

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

### From population structure to output error

**A direct test of what the population cannot represent.** At the width-1409,
20k-update checkpoints, measured relative error ranges from 42.8% to 91.3%.
A simple population quantity already explains almost all of it: the lower
bound below is 99.5% of the actual error at the median across six targets
and four seeds. This is stronger than observing that slopes miss a preferred
scale. It bounds the output error of the network with its current readouts.

\Needspace{6\baselineskip}

Define the readout-weighted nonlinear capacity

$$
Q(\theta)=\sum_j |c_j|a_j^2\left(|b_j|+\frac{|a_j|}{3}\right).
$$

For $|x|\le1$, subtract the affine function obtained by expanding each
feature to first order at $x=0$. Since
$|\tanh''u|\le2|u|$, the integral remainder gives

$$
\left|\tanh(b+ax)-\tanh b-ax\tanh'b\right|
\le a^2|b|+\frac{|a|^3}{3}.
$$

Projection onto the fine space removes that affine function. The triangle
inequality then yields the exact bound

$$
\boxed{\quad
\|f_\theta-y\|_m\ge
\left[\|P_Hy\|_m-Q(\theta)\right]_+.
\quad}
$$

This is a global tanh inequality, not a truncated polynomial training model.
It also holds on an independent input measure in $[-1,1]$, using that
measure's own affine projection and target norm. When $Q$ is small, the
population cannot supply enough nonlinear output to cancel the target's fine
component. Preserving small $Q$ for a duration therefore preserves large
output error for that duration.

![Population inequalities explain the output error of broad states, but become uninformative for many concentrated late states. Each point is one of 223 archived checkpoints. Widths 705 and 1409 each contain six targets and four seeds at 20k updates; width 177 includes 175 checkpoints with different restart ages and broader target coverage. The vertical coordinate is the largest of the $Q$ bound and four valid polynomial-tail bounds. The dotted diagonal is equality with actual raw error. Training and independent-grid calculations use their own projections. A zero lower bound means this argument is uninformative, not that fitting succeeded.](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/archive_summary_refined/static_strongest_output_floors.png)

The distinction between diffuse and concentrated populations matters. The
$Q$ bound is positive at all 48 wide checkpoints. Among 59 width-177
checkpoints at 600k updates, it is positive at only 20; a small group often
dominates $Q$, making this absolute-value bound too large. The late network
can still have large error. Thus the theorem should identify a population
regime and its duration, rather than claim a universal error floor from
width alone.

**A sharper variant when target structure matters.** Project onto the
orthogonal complement of polynomials of degree below $k$. Taylor's theorem
in the input gives

$$
\|f-y\|_m\ge
\left[\|P_{\ge k}y\|_m-
\frac{C_k}{k!}\sum_j|c_j||a_j|^k\right]_+,
\qquad C_k=\sup_{u\in\mathbb R}|\tanh^{(k)}u|.
$$

The plot uses $k=2,3,5,9$ in addition to $Q$. These inequalities concern
the exact network; their empirical usefulness depends on both the target
tail and the current weighted slope moments. They are not restricted to a
degree-nine target.

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

\Needspace{4\baselineskip}

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

## 6. A population theorem: slow reinforcement preserves output error

**What this section shows.** We can now prove a persistence statement from
the initial population, rather than assume that future fine forces or
concentration remain small. The mechanism is a feedback limitation: broad
features have weak nonlinear sensitivity, and their coupled motion cannot
rapidly build much more of it. The conclusion is an output-error lower bound;
limited population scale acquisition follows from the same argument.

With $X_j=\sqrt W(a_j,b_j,c_j)$, define

$$
\begin{aligned}
M&=\mathbb E_W|X|^2=\sum_j(a_j^2+b_j^2+c_j^2),&
M_6&=\mathbb E_W|X|^6,\\
B&=\frac{M_6^{1/6}}{W^{1/3}},&C_6&=M_6/M^3.
\end{aligned}
\tag{6}
$$

$B=[\sum_j(a_j^2+b_j^2+c_j^2)^3]^{1/6}$ is a norm of the whole
population; it allows heterogeneous particles.
$C_6$ records concentration, but the theorem below does not assume a bound
on its future values. Write $Y=\|e_H\|_m$ and
$E_s=\sum_j(a_j^2+c_j^2)$.

**Proposition 1: initial-data persistence under effective flow.** Consider
the exact flow $\dot\theta=-F$ on a centered symmetric grid in $[-1,1]$,
with $v=\langle x^2\rangle_m>0$. Suppose $B_0,Y_0>0$. Choose $q>1$,
put $A_q=(q-1)B_0$, and require the initial-data rank test

$$
\delta_q=
\sqrt{\frac{v(\sqrt{E_{s,0}}-A_q)_+^2}
{1+2(\sqrt{M_0}+A_q)^2}}-3q^3B_0^3>0.
$$

Appendix A.2 gives an alternative test using the measured initial coarse
singular value. Set $k=6Y_0B_0^2$ and $T_q=(1-q^{-2})/k$.
For $0\le t\le T_q$, the coarse projector remains defined and

$$
\begin{aligned}
B(t)&\le\frac{B_0}{\sqrt{1-kt}},\\
\int_0^t\|F(s)\|\,ds&\le A(t):=
B_0\bigl[(1-kt)^{-1/2}-1\bigr],\\
Y(t)&\ge Y_0\exp\left\{-\frac{9B_0^6}{2k}
\bigl[(1-kt)^{-2}-1\bigr]\right\}.
\end{aligned}
\tag{7}
$$

There is also a direct output interpretation. Throughout this interval,

$$
\|f(t)-y\|_m\ge
\left[\|P_Hy\|_m-
\frac{2\sqrt2q^4}{3W}M_6(0)^{2/3}\right]_+.
$$

The population still cannot produce enough non-affine output. This avoids
assuming that every accurate network must attain a prescribed slope scale.
For a nonzero target, dividing by $\|y\|_m$ gives a relative-error floor. The same capacity bound
holds on an independent evaluation measure in $[-1,1]$, with that measure's
own projection and target norm.

Population movement remains part of the conclusion. For
$0\le\lambda_0<\lambda_*$, let $p_0$ be the fraction initially above
$\lambda_0$. The fraction of labels ever reaching $\lambda_*$ satisfies

$$
p_{\rm ever}(t)\le
\min\left\{1,\ p_0+
\frac{h^2 A(t)^2}{W(\lambda_*-\lambda_0)^2}\right\}.
\tag{8}
$$

**Why the result closes.** Exact tanh derivatives, after removing affine
output, give the structural bound

$$
\|F\|\le\frac{3}{W}\|e_H\|_m\sqrt{M_6}=3YB^3.
\tag{9}
$$

Effective training decreases $Y$. A population norm cannot grow faster
than total parameter speed, so $\dot B\le\|F\|\le3Y_0B^3$.
This scalar differential inequality bounds future $B$ from $B_0$ alone;
its travel allowance also preserves coarse conditioning. The proof concerns
the evolving ODE, not the validity duration of a frozen Jacobian or a
polynomial training model. It does not require slopes to shrink.

For a desired additional crossing fraction $\varepsilon>0$, the same
travel bound gives $p_{\rm ever}\le p_0+\varepsilon$ whenever

$$
t\le\min\left\{T_q,\ \frac1k\left[
1-\left(1+
\frac{\sqrt{\varepsilon W}(\lambda_*-\lambda_0)}{hB_0}
\right)^{-2}\right]\right\}.
\tag{10}
$$

For bounded initial rescaled moments, nondegenerate coarse response, and
$Y_0$ bounded above and away from zero, fixed $q$ gives $T_q$ proportional to $W^{2/3}$ and travel
$O(W^{-1/3})$. These are sufficient asymptotic rates; the constants must be
evaluated before claiming a useful experimental training budget.

**Ordinary GD: tracking can also be controlled from initial data.**
Consider a sequence of widths with initial $M_6$ and total loss bounded
above, $v$ and initial $E_s$ bounded below by positive constants, and
$\|z_0\|=o(W^{-1/3})$. These are post-transient starting assumptions,
not claims about how early training reaches such a state. There are
width-independent $c>0$ and $\eta_*>0$ such that ordinary GD with
$0<\eta\le\eta_*$ obeys, for all sufficiently large widths,

$$
n\eta\le cW^{2/3}\quad\Longrightarrow\quad
\begin{cases}
M_{6,n}=O(1),\quad \displaystyle\sum_{k<n}\|\theta_{k+1}-\theta_k\|
=O(W^{-1/3}),\\[2pt]
\|f_n-y\|_m\ge[\|P_Hy\|_m-O(W^{-1})]_+.
\end{cases}
\tag{11}
$$

Appendix A.4 proves this using an explicit initial-data recurrence, including
tracking and finite-step error. It does not identify GD with gradient flow
by substituting $t=n\eta$. Future small tracking is a conclusion of that
comparison, not an additional premise. The remaining practical question is
whether the sufficient bounds retain the slow behavior for a useful budget
at the measured widths; a sharper signed population inequality may extend
that budget. Adam requires a different dynamical argument.

## 7. Adam: measure access to the remaining error in its actual metric

**The motivating example.** Adam reaches much lower error than GD in the
opening comparison, but many runs remain far from the desired accuracy.
The Euclidean GD theorem cannot explain Adam merely by changing a learning
rate. Adam rescales parameter directions and carries momentum, so both
effects must enter the diagnosis.

First consider a simpler question whose answer is exact. If geometry is
frozen, let $A$ be the readout feature matrix, including output bias and
divided by $\sqrt m$ to match the loss normalization. Ordinary readout GD
obeys $r_{n+1}=(I-\eta AA^T)r_n$. If $AA^T$ has eigenvalues
$\kappa_i$ and orthonormal eigenvectors $u_i$, then

$$
\|r_n\|^2=\sum_i(1-\eta\kappa_i)^{2n}|u_i^Tr_0|^2.
$$

For $\eta\kappa_{\max}\le1$, energy initially in eigenvalues at most
$s$ cannot decay faster than $(1-\eta s)^{2n}$. Thus a slow subspace
containing substantial residual energy yields an **output-error lower bound
through a specified budget**. A large condition number alone is insufficient:
the remaining error must actually occupy the difficult directions.
Appendix A.6 states the bound precisely.

For an evolving Adam state we can ask the corresponding instantaneous
question. Let $J$ be the full output Jacobian in the same RMS coordinates,
$D=\operatorname{diag}[(\sqrt{\widehat v}+\epsilon)^{-1}]$ the next-update
adaptive scaling, and $K_D=JDJ^T$. The spectrum of $K_D$, together with
the residual energy in its eigenvectors, measures the current gradient's
access to that error. We also compute the readout-only version. These are
current-state measurements; evolving $D$, $J$, and momentum prevent treating
the eigenvalues as Adam convergence rates.

![Most remaining Adam error lies in directions with weak instantaneous sensitivity, even after adaptive scaling. Each paired line joins the raw and adaptive calculation for one target/seed state; there are 65 states per panel. The displayed fraction includes eigenvalues with $\eta\kappa<10^{-5}$ and numerically unresolved directions. The cutoff is a sensitivity diagnostic, not a predicted 100k-update hitting time. At 600k updates the median joint fraction is 98.97% before scaling and 97.83% after scaling. These panels include successful and failing 1% endpoints alike.](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/adam_plots/adam_bulk_slow_residual_energy.png)

At the final checkpoint, the median unresolved fraction in the adaptive
joint calculation is only $2.94\times10^{-12}$, with maximum 0.611%.
Thus the large slow-energy fraction is mostly in resolved weak directions.
The result is compatible with a few strongly accessible directions
contributing a substantial gradient while most output error remains hard to
correct. An average sensitivity can hide this distinction.

**Does the actual Adam step exploit the accessible directions?** Let
$\widehat m$ be the next bias-corrected first moment. With
$g=J^Tr$, write the update as

$$
\Delta\theta=-\eta D\widehat m
=-\eta Dg-\eta D(\widehat m-g).
$$

The first term is the current-gradient step; the second is the momentum
lag relative to that step. Set $u=J\Delta\theta$ and
$w=f_{\theta+\Delta\theta}-f_\theta-u$, all in RMS coordinates.
The exact output-loss change is

$$
\begin{aligned}
L(\theta+\Delta\theta)-L(\theta)
={}&\underbrace{-\eta g^TDg}_{\text{current-gradient reduction}}
+\underbrace{-\eta g^TD(\widehat m-g)}_{\text{momentum-lag contribution}}\\
&+\underbrace{\tfrac12\|u\|^2}_{\text{quadratic step cost}}
+\underbrace{\langle r+u,w\rangle+\tfrac12\|w\|^2}_{\text{nonlinear remainder}}.
\end{aligned}
$$

We reconstruct this next step from the archived parameters and optimizer
moments. Among the 37 primary Adam endpoints still above 1% error at 600k,
momentum lag suppresses 94.4% of the current gradient's predicted reduction
at the median. The virtual next step increases loss in 22 of 37 cases.
Only one has lag large enough to cancel the entire linear gradient reduction:
the positive quadratic step cost also matters. This is a more precise
diagnosis than saying that tracking simply cancels the fine gradient.

![Current-gradient descent predictions can greatly overstate actual next-step progress. Left: positive vertical values mean loss decreases; negative values mean it increases. Right: a lag ratio of one cancels the current gradient's entire linear reduction before the quadratic step cost. Orange includes thirteen targets and five seeds at learning rate 0.002; the other rates use the same thirteen targets at seed zero. All three checkpoint ages are shown. These are virtual next updates reconstructed from saved optimizer states, not cumulative attributions over training.](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/adam_plots/adam_actual_output_progress.png)

Rate controls prevent interpreting this as an unavoidable property of every
Adam configuration. On the matched thirteen-target, seed-zero subset at 600k,
median errors are 1.22%, 0.991%, and 0.667% for rates 0.002, 0.001, and
0.0002. All thirteen endpoints at each rate still exceed $10^{-4}$.
Smaller steps improve some cases and reduce some update inefficiency; they
do not remove the remaining accuracy gap on this panel.

**What this adds, and what it does not.** Weak access to most remaining
error coexists with inefficient updates in accessible directions. This
supports an Adam theory based on residual-weighted adaptive sensitivity
and an accumulated output-progress budget. Three checkpoint diagnostics
do not prove sustained oscillation, a uniform lower bound, or persistent
tracking cancellation. Establishing duration requires controlling the
evolving metric and optimizer moments together.

## 8. What is proved, and what would complete the mechanism?

**The main advance is a structural ODE result.** Starting from a diffuse
population and sufficiently small coarse tracking, the theorem propagates
bounded moments, limited travel, and substantial output error. It no longer
assumes that the future fine force stays small. The proof applies to changing
features and changing coarse compensation. Its ordinary-GD version closes
tracking and finite-step error from initial data as well.

**The practical duration remains the gap.** On the 24 width-1409 checkpoints,
the current ordinary-GD recurrence gives 86--172 additional updates, with
median 95. The actual continuations remain slow for 20k updates. The bound
does not predict escape after 172 updates; it stops guaranteeing persistence.
This is an FP64 evaluation of sufficient inequalities, not an outward-rounded
certificate. Increasing the moment order improves an asymptotic exponent but
worsens the tested finite-width constants, so it is not the useful refinement
on this panel.

We also tested two initial-data refinements for effective flow: start from
the measured fine Jacobian norm, or from the smaller actual effective-force
norm. Both derive subsequent growth bounds along the evolving ODE. At
width 1409 their median guaranteed physical flow times are 7.62 and 4.54,
respectively, versus 2.07 for the sixth-moment bound. The observed
20k-update continuation has physical time 40. These effective-flow intervals
are not certified GD update counts. The actual-force proof retains more
residual energy but is limited by absolute curvature estimates.

![Initial-data bounds distinguish mathematical persistence from the duration supported by conservative constants. The left panel shows available effective-flow time; the right shows the fraction of the initial fine-error norm retained by the bound. Points include all 223 static states, so the width-177 group contains different restart ages. Each method allows its sensitivity or force to evolve. A zero energy floor is uninformative. These FP64 calculations do not certify machine-rounding errors, unsampled trajectories, or ordinary GD.](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/archive_summary_refined/effective_flow_enclosures.png)

A separate generic GD energy argument, with outward-rounded initial checks
at eighteen wide-panel states, already excludes $\lambda=0.25$ for 50k
additional updates. Appendix B gives that proof. It is useful as a baseline,
but a distant-threshold exclusion alone does not explain the effective-force
mechanism. The population theorem adds output error, width dependence, and
a reason for suppressed reinforcement. The remaining task is to preserve
more of the signed collective structure in its duration estimate.

**Which structure matters?** For a nonzero population, exact differentiation
gives

$$
\frac{d}{dt}\log C_6
=6\left[
\frac{\mathbb E_W(|X|^4 X\cdot\dot X)}{M_6}
-\frac{\mathbb E_W(X\cdot\dot X)}{M}
\right].
\tag{12}
$$

Concentration increases when larger particles receive disproportionately
positive radial motion. This is a collective correlation, not a requirement
to predict each neuron. We now measure its effective, tracking, generated,
and target contributions, as well as the corresponding derivatives of $Q$.
For the last two, split $F=\Pi J_H^TP_Hf-\Pi J_H^TP_Hy$ at the same
state. Both terms retain the same coarse compensation.

The measurements distinguish two regimes. At width 1409 after 20k updates,
generated-output correction reduces $Q$ in all 24 states. The target
contribution increases it in sixteen and decreases it in eight. Nevertheless,
the median ratio $|\dot Q|/(|\dot Q_{\rm generated}|+
|\dot Q_{\rm target}|)$ is 0.999: there is little cancellation in the
typical wide state. At width 177 after 600k, the same ratio has median
0.0153 across 59 states. Strong cancellation is common there, but its signs
are not universal. **Correction and weak sensitivity are complementary
mechanisms whose relative importance changes with the population.**

![The same population equation has different balances. Left: six targets and four seeds at width 1409 after 20k updates. Right: 59 late states across 23 targets at width 177 after 600k; thirteen targets have an additional seed-zero state, so this is not a balanced target average. Both axes are signed contributions to $\dot Q$ along the current effective flow. Points near the dashed line cancel; the color gives the fraction of absolute contributions retained in the net rate. Yellow means little cancellation. Signed logarithmic axes have a linear neighborhood around zero. These are instantaneous derivatives at supplied states, not a proof of future persistence.](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/archive_summary_refined/population_correction_regimes.png)

An improved theorem should bound the growth-producing collective terms while
retaining these correlations. A global absolute-value curvature bound loses
both cancellation and weak target alignment. The proposed assumptions should
be statements about those population correlations and their own evolution,
verified against interventions; assuming a small future force would simply
restate the conclusion. Neither a fixed Jacobian nor universal contraction
is the appropriate target.

**Scope across targets and optimizers.** The evidence extends beyond degree
nine and sine: 23 target instances in the broad audit, ten separately chosen
new functions in the force test, and six families in the width/intervention
panel. Repeated checkpoints are not independent target samples, and the
wide-panel conclusions have not been established for all 23 functions.
The horizon is part of each statement. Eventual movement after millions of
updates does not refute a finite-time slowdown, and long-run forecast failures
do not define its practically relevant duration.

The Adam audit strengthens the output-based framing but remains a separate
dynamical problem. The evidence supports studying residual energy, adaptive
sensitivity, and actual accumulated progress together. It does not support
transferring the Euclidean GD rates or imposing one sign on Adam's tracking
contribution across targets.

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

### A.2. Closing population persistence from initial data

Under $\dot\theta=-F$,

$$
\frac{d}{dt}\tfrac12\|e_H\|_m^2
=-e_H^TJ_H\Pi J_H^Te_H=-\|F\|^2\le0.
\tag{A2}
$$

Let $p_j=(a_j,b_j,c_j)$. The upper derivative of a norm is at most
the norm of its velocity. Therefore

$$
D^+B\le\left(\sum_j|\dot p_j|^6\right)^{1/6}
\le\left(\sum_j|\dot p_j|^2\right)^{1/2}
\le\|F\|\le3Y_0B^3.
\tag{A3}
$$

Comparison gives $B\le b(t)=B_0(1-kt)^{-1/2}$, with
$k=6Y_0B_0^2$. Since $\|F\|\le3Y_0b^3=b'$, total travel is at
most $b-B_0=A(t)$. Also $\dot Y=-\|F\|^2/Y\ge-9B^6Y$ when
$Y>0$. Integrating this inequality with the bound on $b$ proves all
three estimates in (7). If $Y_0=0$, effective flow is stationary wherever
the coarse projector is defined; no division by $k$ is needed.

**Why coarse rank survives.** The affine network
$f_0=d+\sum_jc_j(a_jx+b_j)$ has coarse Gram matrix

$$
K_0=\begin{pmatrix}
1+\sum_j(b_j^2+c_j^2)&\sqrt v\sum_ja_jb_j\\
\sqrt v\sum_ja_jb_j&v\sum_j(a_j^2+c_j^2)
\end{pmatrix}.
$$

\Needspace{9\baselineskip}

Cauchy–Schwarz gives $\det K_0\ge vE_s$ and
$\operatorname{tr}K_0\le1+2M$. Thus the affine coarse Jacobian has
least singular value at least $\sqrt{vE_s/(1+2M)}$. The column-remainder
calculation in A.1 bounds the difference between the exact and affine
coarse Jacobians by $3B^3$. Consequently

$$
\sigma_{\min}(J_C)\ge\sqrt{\frac{vE_s}{1+2M}}-3B^3.
$$

Travel at most $A_q$ gives
$\sqrt M\le\sqrt{M_0}+A_q$ and
$\sqrt{E_s}\ge(\sqrt{E_{s,0}}-A_q)_+$.
Until $B$ reaches $qB_0$, the rank bound is therefore at least
$\delta_q>0$. A first-exit argument closes the estimates: neither rank
loss nor escape from the moment envelope occurs before $T_q$.
Bounded total travel also permits continuation of the smooth ODE through
that interval. This proves preservation, rather than assuming it.

An alternative initial-data rank margin, valid for the same travel, is

$$
\sigma_{\min}(J_C(0))-
\left[\sqrt2+4(\sqrt{M_0}+A_q)\right]A_q.
$$

Indeed $\|DJ_C\|\le\sqrt2+4\sqrt M$: the full neuron Hessian has
geometry–geometry norm at most $4|c_j|$ and cross-block norm at most
$\sqrt2$. Integrating along the path gives this alternative. The maximum
of the two rank margins may be used in the theorem.

**Why this implies output error.** Subtracting the affine network and
using $|\tanh u-u|\le|u|^3/3$ yields

$$
\|P_Hf\|_m\le\frac{2\sqrt2}{3}\sum_j|p_j|^4
\le\frac{2\sqrt2}{3}W^{1/3}B^4.
$$

With $B\le qB_0$, the reverse triangle inequality proves the stated
output floor. The remainder estimate is pointwise on $[-1,1]$, so the
capacity floor also holds for another evaluation measure on that interval.
The dissipation identity (A2), however, concerns the training measure only.

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

### A.4. Ordinary GD: close tracking and finite steps together

This proof constructs an initial-data certificate and then derives the
width scaling in (11). It does not assume a future tracking budget.
For the full gradient flow, the exact identities are

$$
\dot z=-Kz+\dot\ell,\qquad
\dot e_H=-J_H\Pi J_H^Te_H-J_HR.
$$

The first displays the stability balance: coarse relaxation opposes changes
in the compensating response. The relevant output disturbance is $J_HR$,
not merely the slope coordinates of $R$.

Fix $q>1$, let $B_*=qB_0$, $A_*=(q-1)B_0$, and choose a positive
initial-data coarse singular-value margin $\sigma$ from A.2. Define

$$
\begin{aligned}
\overline Y&=\sqrt{2L_0},& J_*&=\sqrt{1+2(\sqrt{M_0}+A_*)^2},\\
H_C&=\sqrt2+4(\sqrt{M_0}+A_*),&u_*&=3\overline YB_*^3,\\
Y_{\rm seg}&=\overline Y+J_*A_*,&
D_\ell&=\frac{6H_CY_{\rm seg}B_*^3}{\sigma^2}
+\frac{9B_*^6+6\sqrt2Y_{\rm seg}B_*^2}{\sigma}.
\end{aligned}
$$

These constants bound the region of total travel at most $A_*$.
Require $\eta[J_*^2+Y_{\rm seg}H_C]\le1$. Starting from
$Z_0=\|z_0\|$, $A_0=0$, compute

$$
\begin{aligned}
v_n&=u_*+J_*Z_n,\\
A_{n+1}&=A_n+\eta v_n,\\
Z_{n+1}&=(1-\eta\sigma^2)Z_n+\eta D_\ell v_n
+\tfrac12H_C\eta^2v_n^2.
\end{aligned}
$$

Every step with $A_{n+1}<A_*$ is certified: total loss decreases,
parameter travel is at most $A_n$, $B_n\le B_*$, coarse rank is
preserved, and $\|z_n\|\le Z_n$.

To verify this claim, first note that the full output Jacobian and Hessian
are bounded by $J_*$ and $H_C$. On the entire proposed GD segment,
the residual norm is at most $Y_{\rm seg}$, so the loss Hessian is at
most $J_*^2+Y_{\rm seg}H_C$. The step restriction gives loss descent
and preserves the nodal residual bound $\overline Y$.

For the fine output, subtract the affine network before differentiating
twice. The remaining Hessian blocks have norms at most
$4\sqrt2|p_j|^2$ and $2\sqrt2|p_j|^2$; thus
$\|D^2e_H\|\le6\sqrt2B_*^2$ and
$\|Dg_H\|\le9B_*^6+6\sqrt2Y_{\rm seg}B_*^2$.
Differentiating $\ell=K^{-1}J_Cg_H$ in a parameter direction $w$ gives

$$
D\ell[w]=K^{-1}(DJ_C[w])F+K^{-1}J_CDg_H[w]
-K^{-1}J_C(DJ_C[w])^T\ell.
$$

Using $\|\ell\|\le3Y_{\rm seg}B_*^3/\sigma$ bounds this derivative
by $D_\ell$. Finally, the exact coarse update with its Taylor remainder is

$$
z_{n+1}=(I-\eta K_n)z_n+(\ell_{n+1}-\ell_n)+r_{C,n},\qquad
\|r_{C,n}\|\le\tfrac12H_C\eta^2\|g_n\|^2.
$$

The step restriction ensures
$\|I-\eta K_n\|\le1-\eta\sigma^2$. Since
$\|g_n\|\le u_*+J_*Z_n$, induction proves the recurrence and its
travel premise. The output-capacity proof in A.2 immediately supplies an
error floor on every accepted step.

For a separate fine-error progress bound, if $9\eta B_*^6\le1$, the
fine update gives

$$
Y_{n+1}\ge(1-9\eta B_*^6)Y_n
-3\eta B_*^3J_*Z_n-3\sqrt2\eta^2B_*^2v_n^2.
$$

The last term is the actual nonlinear GD remainder allowance; it is not
omitted by a flow approximation.

**Derive the asymptotic population window.** Under the initial conditions
of (11), $B_0=\Theta(W^{-1/3})$, $A_*=\Theta(W^{-1/3})$,
$\sigma$ is bounded below, $u_*=O(W^{-1})$, and
$D_\ell=O(W^{-2/3})$. Choose a width-independent sufficiently small
$\eta_*$. In the region $Z\le\varepsilon W^{-1/3}$, the recurrence
then implies, for positive width-independent constants $c_0,C$,

$$
Z_{n+1}\le(1-c_0\eta)Z_n+C\eta W^{-5/3}+C\eta^2W^{-2}.
$$

For large widths this region is preserved from $Z_0=o(W^{-1/3})$.
Summing through physical time $T=n\eta$ gives
$\eta\sum_{k<n}Z_k\le CZ_0+CTW^{-5/3}+C\eta TW^{-2}$.
Hence the travel recurrence is at most
$CTW^{-1}+o(W^{-1/3})$ for $T=O(W^{2/3})$.
Choosing the constant in $T\le cW^{2/3}$ sufficiently small keeps
travel strictly below $A_*$, closing the induction. It follows that
$M_{6,n}\le q^6M_{6,0}$, and the capacity bound proves (11).
The discrete counting argument in A.3 also applies to this travel bound.

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
Transport alone does not prevent concentration or reinforcement. The initial-data
moment inequality uses the specific tanh velocity field to prove persistence.
Retaining the signed effects in (12) may explain a longer duration than this
sufficient bound, but such an extension needs its own argument. Discrete GD transports the
empirical measure by a map at each update; A.4 supplies the corresponding
population bound without introducing artificial diffusion.

### A.6. Frozen features: an output-error bound through a budget

Let $\widetilde y=y/\sqrt m$, $r_n=Aw_n-\widetilde y$, and
$K_{\rm fr}=AA^T$. This matrix is distinct from the two-dimensional coarse
Gram matrix $K$ in (2). For $0<\eta\le1/\kappa_{\max}$, choose a cutoff
$0<s\le\kappa_{\max}$ and define

$$
E_0=\|P_{\ker A^T}r_0\|^2,
\qquad
E_{\rm slow}(s)=\sum_{0<\kappa_i\le s}|u_i^Tr_0|^2.
$$

Every readout-GD update $0\le n\le N$ satisfies

$$
\frac{\|r_n\|}{\|\widetilde y\|}
\ge
\frac{\sqrt{E_0+(1-\eta s)^{2N}E_{\rm slow}(s)}}{\|\widetilde y\|}.
$$

Indeed, $r_n=(I-\eta K_{\rm fr})^nr_0$. In the slow eigenspace every
multiplier is at least $1-\eta s\ge0$, and its $n$th power is at least
its $N$th power. The nullspace component never changes. Retaining those
terms in the exact squared-norm formula proves the bound. If the right-hand
side exceeds an accuracy requirement, no checkpoint through $N$ meets it.
The relevant weights are residual energies at the restart; they equal target
energies only for zero initial output. Evolving-feature GD and Adam require
separate arguments.

### A.7. A refinement using the actual initial effective force

The measured force can be much smaller than the isotropic bound $3YB^3$.
To exploit that observation without freezing the force, differentiate the
projector in (2). With $f=\|F\|$, the exact identity is

$$
\frac12\frac{d}{dt}f^2
=-\|J_HF\|^2
-\langle e_H,D^2e_H[F,F]\rangle
+\ell\cdot D^2e_C[F,F].
$$

For example, $(D\Pi[F])J_C^T=-\Pi(DJ_C[F])^T$ and
$\Pi(D\Pi[F])\Pi=0$ follow by differentiating the projector identities.
Substituting $g_H=F+J_C^T\ell$ gives the displayed compensation term
with its positive sign. Thus this identity includes changes in geometry,
readouts, and the coarse projector.

Choose $q>1$, $A_q=(q-1)B_0$, a positive rank margin $\sigma$ from A.2,
and $H_C=\sqrt2+4(\sqrt{M_0}+A_q)$. The fine Hessian estimate in A.4
and $|\ell|\le3Y_0B^3/\sigma$ imply

$$
D^+B\le f,\qquad
\dot f\le Y_0\left(6\sqrt2B^2+\frac{3H_C}{\sigma}B^3\right)f.
$$

For $f_0>0$, a comparison system is $\dot b=U(b)$, where

$$
U(b)=f_0+2\sqrt2Y_0(b^3-B_0^3)
+\frac{3Y_0H_C}{4\sigma}(b^4-B_0^4),\qquad b(0)=B_0.
$$

To verify it, introduce $u'=Y_0(6\sqrt2b^2+3H_Cb^3/\sigma)u$ and
$b'=u$. These right-hand sides are nondecreasing in the nonnegative
comparison variables. Integrating $du/db$ gives $u=U(b)$, so
$B\le b$, $f\le U(b)$, and total travel is at most $b-B_0$.
\Needspace{6\baselineskip}

The rank-margin first-exit argument closes these estimates through

$$
T_q^F=\int_{B_0}^{qB_0}\frac{db}{U(b)}.
$$

The exact energy identity (A2) also yields

$$
Y(t)^2\ge\left[Y_0^2-2\int_{B_0}^{b(t)}U(u)\,du\right]_+.
$$

This proof propagates a small initial force from the changing ODE. It still
drops residual relaxation and all favorable curvature signs, which explains
why a very small initial force need not produce a proportionally long
guarantee. If $f_0=0$, the exact effective flow is stationary wherever the
projector is defined; the observed slow states need not satisfy that special
case. No Adam or discrete-GD conclusion follows from this refinement alone.

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

**The six targets in the width and feedback panels.** These formulas specify
the functions before RMS normalization:

- **Degree five:** $0.3q_0+0.4q_1+\sqrt{0.75}\,q_5$, where $q_k$ are empirical
  orthonormal polynomials with positive leading coefficient.
- **Mixed sine:** $\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$.
- **Off-center Gaussian:** $\exp(-((x+0.35)/0.22)^2)$.
- **Right compact bump:** $\exp(1-1/(1-u^2))$ for $|u|<1$, otherwise zero,
  with $u=(x-0.35)/0.22$.
- **Right step:** $\tanh(14(x-0.31))$.
- **Kink:** $|x+0.23|$.

Except for the already normalized polynomial target, each function is divided
by its RMS on the original training grid. That normalization stays fixed
during training and evaluation; target means are not subtracted for training.

The degree-five label thus denotes a prescribed orthogonal residual shape,
not the monomial $x^5$. The same distinction applies to the degree-nine
control. The development width panel has two seeds and starts its continuation
after 20k updates; its 20k further updates end at 40k total. The quintic
confirmation repeats the six targets and three widths with two new seeds.
It checks seed variation, not generalization to unseen functions. Where
independent-grid fitting errors are measured, evaluation uses 8,192 midpoint
inputs; the force and trajectory quantities here concern the training grid.

**The thirteen targets in Figures 1 and 2.** Sine is $\sin(2\pi x)$, and Runge is
$1/(1+25x^2)$; these two functions are not RMS-rescaled. Degree $d$, for
$d\in\{3,4,5,9\}$, denotes
$0.3q_0+0.4q_1+\sqrt{0.75}\,q_d$, using the same empirical orthonormal
polynomials defined above. Each of these polynomial targets has unit empirical
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

These blends already have unit empirical RMS. Figures 1 and 2 use five seeds
numbered 0 through 4, with the same width-177 initialization law and training
grid described above. Their full-complement decomposition has no omitted
modal-residual term. At each saved state, the median and range are taken
across the five seeds separately for each force norm and each scale statistic.
The shading is neither a confidence interval nor a bound on unsampled times.
There are 172 saved pre-update states per trajectory, from update 9 to 599,999.
The plotted values are instantaneous measurements at those states, not
averages over the intervals between them. The normalized scale uses
$N_{\rm ref}=128$, not the actual width 177, so $h=1/64$.

For each seed, the crossover is the first saved state satisfying
$\|R_a\|\le\|F_a\|$. The plotted marker is the median of those five times,
not the crossing of the median curves. The final/crossover ratios in Section 2
are computed within each seed before taking the median. The early-motion
comparison uses the nearest saved states to half, once, and twice each seed's
crossover time. Its signed mean-slope increment is
$W^{-1}\sum_j\operatorname{sgn}(a_j)\Delta a_j$; the correction from slopes
crossing zero is negligible in those comparisons. This is a retrospective
timing check, not an intervention establishing that norm equality causes the
slowdown.

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
numbers in the pulse panel use the smallest amplitude. Halving
checks distinguish the predicted first-order response from finite-amplitude
matching errors.

The finite-dilation experiment instead fixes $(a,b)=s(a_0,b_0)$ with
$s\in\{1,1.25,2\}$. With $v=(c,d)$, the repair finds a locally stationary
solution of

$$
\min_v\tfrac12\|v-v_{\rm ref}\|^2\quad\text{subject to}\quad z(v)=0,
\qquad v_{\rm ref}=(c_0,d_0)\ \text{or}\ (c_0/s,d_0).
$$

The distance reference is fixed throughout continuation in $s$; this is not
full readout least squares or a claim of a globally nearest repair. The six
branches are ordinary GD, the repaired $s=1$ baseline, and both references
at each larger scale. All 350 repairs passed normalized balance tolerance
$10^{-12}$ and normalized stationarity tolerance $10^{-8}$; the maximum
initial tracking norm divided by the larger original/repaired effective
slope-force norm is $8.01\times10^{-7}$. All 420 branches then completed 20k ordinary-GD
updates. The late panel uses all 23 functions with two seeds each; each wide
panel uses degree five, mixed sine, left Gaussian, right bump, right step,
and the absolute-value kink, with two seeds per function.

The construction threshold is not injected into the wide populations: their
largest final $\lambda$ values are 0.00114 and 0.000384. Twofold dilation of
the late states does inject six neuron labels above $\lambda=0.25$ in each
readout-reference arm. No new GD crossings occur in any main branch.
Injected labels are kept separate from subsequent acquisition.

For cloning, $k$ identical copies each receive readout $c_j/k$. The function
and residual are initially unchanged. Replica symmetry makes the geometry
rate $1/k$ of the original and the aggregate readout rate $k$ times the
original. Multiplying geometry learning rates by $k$, or dividing readout
learning rates by $k$, isolates these effects. The construction spacing $h$
is held fixed because this test changes parameterization, not resolution.
The plotted motion is the Euclidean norm of the signed slope displacement,
with one slope per group of identical copies, divided by that norm in the
original branch. Counting all replicas separately would introduce an
irrelevant factor $\sqrt{k}$. This norm measures motion in either direction,
not just outward scale acquisition.

### B.3. Coverage and numerical meaning

**Which evidence answers which question?** The opening plots establish the
phenomenon over 65 ordinary-GD trajectories: thirteen targets and five seeds,
through 600k updates at width 177. The new-function force test checks the
reduction on ten additional functions in five families, using two seeds and
three forks for 60 starts. Those functions were fixed before their outcomes
were inspected.

The width development and confirmation panels each contain 36 cases: six
targets, two seeds, and three widths, continued from 20k to 40k total updates.
The feedback interventions use the 36 width cases and four later
sine/degree-nine starts, with six arms each for 240 branches. Pulse and cloning
tests use 23 target instances in separate development and confirmation
cohorts. These matched comparisons test proposed mechanisms.

The population audit instead tests structural assumptions retrospectively:
223 deduplicated static states and 40 natural continuations with eight
snapshots each. The persistence-bound audit uses the same 223 static states.
It evaluates the sixth-moment, higher-moment, evolving-Jacobian, and
evolving-force bounds using only information at each proposed restart.
The force refinement selects the longest of six prespecified region fractions
that retains at least half the initial fine-error norm. Its numerical
quadrature error is recorded separately from the theoretical inequalities;
none of these FP64 evaluations is an interval certificate. Ordinary GD uses
the separate tracking recurrence in A.4.
The energy verification uses eighteen initial states: six targets, three
widths, and one seed. These studies overlap; their counts must not be added
and interpreted as independent samples of target functions.

For finite dilations, every-update channel accounting reproduces individual
normalized scale changes with maximum absolute defect $3.23\times10^{-16}$.
The six prespecified late development targets were also run in all six arms
with half the learning rate and twice the updates. Across these 36 controls
at matched flow time, the largest endpoint slope-vector difference is
$2.34\times10^{-5}$ times the original-step displacement. This controls
step-size sensitivity on the selected subset, not every wide trajectory.

The ordinary floating-point diagnostics support the reported empirical
comparisons, not interval-wide rigorous bounds. In particular, sampled force
ratios do not control their maxima between snapshots. The energy verification
is different: outward-rounded checks establish its sufficient initial
inequalities for exact-real GD starting from the stored binary64 data. It does
not certify all earlier training or every rounding error in later machine
updates.

The Adam audit uses thirteen targets, five seeds, width 177, and the archived
states at updates 20k, 100k, and 600k. GD is evaluated at those same update
counts with the same error definitions. Training inputs are the original
2,048 points; independent evaluation uses 8,192 midpoints in $[-1,1]$ and
the original training-target normalization. No smoothed error or best-readout
refit enters the comparison. The auxiliary rate controls retain seed zero,
all thirteen targets, $\beta_1=0.9$, $\beta_2=0.999$, and
$\epsilon=10^{-8}$. Their reported differences are paired by target and
seed, not compared with the five-seed aggregate as if sampling were identical.

The spectral calculation uses singular vectors of $J$ and $JD^{1/2}$,
for either readout-only or joint parameters. Residual energy orthogonal to
the computed range and energy below the numerical singular-value resolution
floor are reported as unresolved. The full residual-energy and virtual-step
identities close to approximately $2\times10^{-15}$ absolute error in the
audit. Such numerical agreement verifies the diagnostic calculation, not
its extrapolation to future training.

For completeness, the generic energy argument discussed in Section 8 uses a
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
