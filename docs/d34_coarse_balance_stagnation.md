# Why scale acquisition can stall during GD

For a self-contained introduction for external readers, see the
[PI technical note](d34_scale_acquisition_pi_note.md). It develops the ODEs,
empirical mechanism, population bound, and remaining proof obligations without
requiring access to other repository documents.

A network can keep reducing its fitting error while making little progress
toward the slope scales required by the representation we seek. The question
is why the remaining errors do not produce enough sustained outward slope
motion. We study this after the initial constant and linear fit has settled,
starting from observed training states. Explaining how training enters those
states is a separate problem.

Our strongest finding is that **the effective fine force drives the measured
ordinary-GD slope motion**. The corrections due to coarse disequilibrium and
the omitted residual are small in the audited regime. The remaining problem
is therefore inside that force: learning changes both its driving errors and
the sensitivities that convert those errors into slope movement.

The refined hypothesis is that **acquisition is slow when the remaining errors
couple weakly to outward geometry motion, and the subsequent feedback does not
increase that coupling fast enough**. Correcting generated error can contract
slopes, while other errors can drive expansion. At later checkpoints, evolving
errors improve motion forecasts. At wider, nearly affine states, explicitly
evolving the geometry substantially improves forecasts even while the total
fine residual changes little. We have conditional
theorems for these regimes and numerically certified finite windows;
we do not have a universal stagnation theorem.

The newer [coupled-ODE results](d34_coupled_ode_mechanisms.md) derive examples
of persistent rate suppression without assuming an equilibrium or a small
future force. They identify structural conditions to test on the trained
population; they do not establish those conditions across the target panel.
The [population persistence theorem](d34_population_persistence_theorem.md)
now proves preservation of one such structure in a heterogeneous ODE and
states an explicit transfer to ordinary GD. The subsequent
[population audit](../results/checkpoint_D_optimizers/expD34_readout_race/population_coverage/README.md)
found that none of 223 archived states satisfies its aligned sector: mixed
signs and hidden biases are substantial. This sufficient example does not
explain the archived populations. The full heterogeneous polynomial force
is accurate at the tested wide states, so the next theorem should retain
those variables rather than perturb away the violations.

This is the main reading note. Sections 1–2 explain the force and its direction;
Sections 3–4 test its coupled evolution and develop a surrogate; Section 5
connects that surrogate to acquisition bounds. Section 6 states the remaining
scientific question. Examples precede the equations they motivate. Detailed
proofs and complete experiment tables are linked where needed.

**The notation needed to follow the argument.** Other symbols are introduced
when they become useful; lowercase $c$ always denotes readout weights.

| Symbol | Meaning |
|---|---|
| $a_j$, $\gamma_j=\lvert a_j\rvert$ | Signed slope of neuron $j$, and its magnitude or scale. |
| $\lambda_j=h|a_j|$, $h=2/N_{\rm ref}$ | Scale relative to the construction resolution. We study $\lambda_*=0.25$; actual neuron count $W$ also includes halos. |
| $c_j$, $\theta$ | A readout weight, and the collection of all network parameters. |
| $e_H$ | Coefficients describing the remaining error beyond the constant and linear components. |
| $T_a$, $F_a=T_ae_H$ | Effective sensitivity map for slopes, and the gradient contribution it produces. |
| $R_a$ | Remaining slope-gradient corrections; ordinary GD uses $F_a+R_a$. |

## 1. Which part of the gradient needs an explanation?

**Example: different targets, the same dominant force.** On the new-function
panel, a narrow Gaussian makes appreciable fitting progress while an
exponential changes much less. Across their six starting states, median
relative evaluation MSE changes from 0.198 to 0.0396 for the Gaussian, and
from 0.000229 to 0.000154 for the exponential over 200k additional GD updates.
The force reduction holds in both cases. It identifies what drives movement;
it does not label every trajectory as stalled or every improvement as acquisition.

The evidence includes 13 original function instances and ten new ones: two
each of exponentials, off-center Gaussians, compact smooth bumps, tanh steps,
and continuous kinks. The new functions were fixed before inspecting their
training outcomes. Two seeds and three starting checkpoints give 60 new-function
states. Repeated seeds and checkpoints test variation; they are not additional
independent target functions.

**Explanation: account for the coarse response, then identify what remains.**
Our network is a sum of tanh features,

$$
f_\theta(x)=d+\sum_{j=1}^W c_j\tanh(a_jx+b_j).
$$

Here $b_j$ is a hidden bias, $d$ the output bias, and $W$ the number of
neurons. The residual is $r=f_\theta-y$. We resolve it into fixed orthonormal
polynomial shapes $q_k$, with coefficient $e_k=\langle q_k,r\rangle_m$;
the inner product averages over the training points. Thus a *quadratic error*
means the component along $q_2$, not an error squared. This is an exact
projection onto chosen shapes, not a Taylor approximation to the tanh network.
Here $e_H$ retains degrees 2–65; any residual outside the retained basis is
accounted for separately.

Fitting these errors also disturbs the constant and linear outputs. The
effective map $T_a$ includes the compensating response that maintains their
instantaneous balance. Eliminating that response gives the decomposition

$$
g_a=\underbrace{T_ae_H}_{F_a}+R_a,\qquad
a_{n+1}=a_n-\eta(F_{a,n}+R_{a,n}),
$$

where $g_a$ is the slope gradient of the empirical half-MSE
$L=\|f_\theta-y\|_m^2/2$, and $\eta$ is the GD learning rate. We call a
gradient contribution a force; the update has its negative sign. The remainder
$R_a$ contains departure from coarse balance and the omitted-residual force.
Coarse balance is not a least-squares refit of all readout weights.

Across all 60 new-function ordinary-GD endpoints after 200k additional updates,
the largest ratio $\|R_a\|/\|F_a\|$ is 0.0958%. The largest ratio of
accumulated signed remainder-travel vectors to effective-travel vectors is
0.389%. These are endpoint and signed-vector diagnostics, not bounds at every
intervening step. They support studying $T_ae_H$ as the dominant driver.
The [full report](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/analysis/heldout/README.md)
retains all functions and the measurements; the
[derivation](d34_coarse_balance_stagnation_details.md#1-what-small-readout-disequilibrium-actually-controls)
constructs the map explicitly.

The cross-function audit also checks how coarse tracking changes the fine
errors that drive later motion. Across 1,665 saved states from 333 starts,
its forcing is below 0.70% of the effective residual forcing; its direct
slope-force ratio is below 1.24%. These are two distinct small corrections.
The [theorem note](d34_effective_force_acquisition_theorems.md#1-which-empirical-findings-should-the-theorem-use)
records the evidence and explains why both need interval-wide bounds.

**What this lets us ask next.** A small correction does not imply a small
effective force. To explain stagnation, we must determine the direction,
distribution across neurons, and persistence of the force that remains.

## 2. Why correcting error need not produce larger slopes

**Example: flattening a feature can improve the fit.** Our ninth-degree target
has constant, linear, and ninth-degree orthogonal-polynomial components.
The network produces unwanted quadratic and cubic components while fitting
its constant and linear parts. Reducing these unwanted outputs
can favor smaller slopes. Readouts can grow at the same time to preserve the
linear fit. In the observed stalled states, these generated errors drive most
of the small contraction; the large ninth-degree error supplies little opposing
force because the current features respond weakly to it.

The [reduced transport model](d34_rescaled_transport_model.md#5-the-next-clock-generated-error-competes-with-fourth--and-fifth-degree-target-load)
now makes this mechanism explicit. In a simple example, the slope decreases
and its readout increases while their product, which supplies the linear
fit, stays fixed. More generally, correcting generated lower-mode error
decreases geometry energy relative to readout energy. Fourth- and
fifth-degree target components add a competing term. These are proved
identities in the reduced model; transferring them to tanh GD requires
controlling the stated approximation and tracking errors. They do not imply
that every neuron contracts.

This example explains how learning can actively favor contraction. It is not
the template that every other target must follow. In sine, the cubic target
component is nonzero and more accessible. Yet its contribution can still point
inward. Across the five original seeds, the cubic error has the same sign at
100k and 400k updates, while its contribution to mean-scale motion changes
from inward to outward. The effective sensitivity changed. Later, the error
itself reverses in some seeds and opposes an outward fifth-mode contribution.
The [signed-mode audit](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/analysis/cubic_sign_audit/README.md)
separates these effects. Those sine trajectories still do not reach the
intended geometry and precision.

**Explanation: an error's sign alone does not determine scale motion.** Each
error coefficient is multiplied by a sensitivity vector. That vector depends
on readouts, slopes, and biases, and its entries need not push all neurons
outward. A change in error, a change in sensitivity, or cancellation among
different errors can therefore change the net scale motion.

This is why neither total loss nor readout size alone explains acquisition.
Two networks with the same coarse and cubic errors can even have opposite
mean-scale responses because their readout–slope correlations differ. The
[heterogeneous example](d34_coarse_balance_stagnation_details.md#13-a-heterogeneous-cubic-example-explains-what-must-evolve)
makes that statement explicit. It explains a direction mechanism; its cubic
approximation is not a quantitatively valid model of every later tanh state.

**Prediction.** A mechanism should predict how the signed force changes when
we alter its feedback. Seeing a large error remain, or observing some slopes
grow, is insufficient. We now test the two evolving factors separately.

## 3. Test feedback without changing the initial force

**Example: the same starting sine force develops different motion.** At the
600k sine checkpoints, holding the sensitivity map fixed reduces the median
mean-scale growth over the next 10k updates. Holding the errors supplied to
the slope force fixed increases it. Both modified branches start with the
same update as ordinary GD. Their subsequent separation tests how learning
changes its own drive, rather than simply giving one branch a stronger start.

**Experiment: let one factor respond while fixing the other.** At a checkpoint
$\theta_s$, save its map $T_{a,s}$ and error $e_s$. Apply one of these
effective slope forces:

| Branch | Effective slope force | Feedback allowed inside that force |
|---|---|---|
| Ordinary GD | $T_a(\theta)e_H(\theta)$ | Both sensitivities and errors change. |
| Freeze the map | $T_{a,s}e_H(\theta)$ | Errors change. |
| Clamp the supplied error | $T_a(\theta)e_s$ | Sensitivities change. |

Every branch also applies its own current $R_a$. Readouts and both bias
blocks retain their ordinary gradients. Clamping the supplied error does
**not** freeze the actual network residual: the network continues learning.
The first update agrees in real arithmetic; the next differences expose map
and error feedback. Over longer intervals, the entire coupled trajectory must
be predicted. The
[response identities](d34_coarse_balance_stagnation_details.md#121-match-the-initial-signal-then-alter-its-feedback)
give the exact first comparisons.

**Prediction and result.** Forecasts issued from the new-function checkpoints
correctly predict all 120 mean-scale intervention contrast signs at 10k
additional updates. Every contrast-vector forecast also improves on predicting
no intervention effect; median relative vector errors are 3.79% for freezing
the map and 2.67% for clamping the supplied error. The result supports a local
coupled description beyond the motivating sine example.

It does not imply that either intervention has one universal direction.
Later predictions sometimes fail, and modified trajectories can amplify
$R_a$ even when it remains small under ordinary GD. Those later contrasts
must be read with the corrections included. They do not turn coarse
disequilibrium into the explanation of the ordinary baseline.

The experiment narrows the question: **how much of the coupled evolution must
we retain to predict useful motion from a checkpoint?**

## 4. A simpler model predicts motion across new functions

**Example: evolving errors can already give a useful forecast.** On all 60
new-function states, a model that freezes effective sensitivities but lets
the fine errors evolve predicts slope motion better than assuming no movement,
through every measured horizon. At 10k additional updates its median error
is 1.55%. Including the local change of sensitivities improves that median
to 0.123%, but does not guarantee better extrapolation farther ahead.

**Model: preserve coupled error relaxation.** Let $T_s$ be the effective map
for all parameters at the checkpoint, and let $S_s=T_s^TT_s$. Hats denote
predicted quantities, and $n$ counts additional updates. The surrogate is

$$
\widehat e_{n+1}=(I-\eta S_s)\widehat e_n,\qquad
\widehat\theta_{n+1}=\widehat\theta_n-\eta T_s\widehat e_n,
\qquad \widehat e_0=e_s,\quad\widehat\theta_0=\theta_s.
$$

Read the first equation as: the current errors drive fitting, which changes
the next errors through their coupling. Read the second as: the same errors
produce parameter movement through the effective sensitivities. The matrix
mixes error components, so they need not relax independently. The model runs
from the checkpoint alone, without receiving future trajectory measurements.

This is a **64-error-state surrogate**. It is different from the slope-only
freeze-map intervention: it uses the fixed effective map for every parameter
block and omits the remainder. For comparison, we also linearize the complete
applied field at the checkpoint. That *local affine model* retains all 532
parameters and the first-order feedback of the map, errors, and remainder.
It is a more detailed local approximation, not a two-variable explanation.

**Prediction and result across all new functions.** Relative error below
100% improves on predicting no slope movement. These medians describe the
full slope-displacement vector, rather than just mean gamma.

| Additional updates | Local affine: median error | Fixed map: median error | Cases beating no motion, local / fixed map |
|---|---:|---:|---:|
| 10k | 0.123% | 1.55% | 60/60 / 60/60 |
| 50k | 1.12% | 7.63% | 57/60 / 60/60 |
| 200k | 8.22% | 18.1% | 48/60 / 60/60 |

Medians conceal some substantial misses. At 10k the worst local error is
16.2%. Its three failures against no motion at 50k all start from the 100k
checkpoint, ending at 150k total updates. All twelve such failures at 200k
also start at 100k. The fixed-map model avoids these large extrapolation
failures, but its worst 200k error is still 79.8%. Numerical controls support
the measured comparisons; a separately checked Gaussian intervention sign
miss survives both degree and step refinement. The
[function-by-function report](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/analysis/heldout/README.md)
retains all outliers and controls.

The gain is a useful, target-general description with measurable approximation
error. Degree nine admits a particularly accurate near-fixed map; other
states require more of its evolution. The usable approximation depends on
the state and interval. It need not remain accurate for millions of updates.

The horizons above are experiment checkpoints, not prescribed theorem targets.
Runs start at 100k, 400k, or 600k total updates; all use noiseless full-batch GD
with $\eta=0.002$. The broader campaign contains 23 function instances and
999 primary branches. Its
[synthesis](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/README.md)
records the original-function comparisons and execution details. This remains
evidence in one architecture and optimization setup, not a function-class
guarantee.

### 4.1 Is this restoration, or a race between readouts and geometry?

**Example: an outward perturbation can survive without continuing to grow.**
We perturb each checkpoint in a direction that increases scale while leaving
its coarse output, coarse disequilibrium, and initial slope force unchanged
to first order. If delayed feedback restores that geometry, the difference
from an unperturbed run should shrink. Across 23 functions and two seeds,
the smallest pulses retain between 99.0% and 100.3% of their directional
offset after 20k additional updates. They show little restoring motion over
this window. This does not exclude restoration in other directions or later.

**Explanation and second test.** Slow motion can persist without an attracting
low-scale state. Another candidate explanation is that readouts remove useful
error before geometry has time to respond. Exact neuron copying lets us alter
these two learning rates without changing the starting function. With four
copies, uncompensated GD slows geometry by four and speeds aggregate readout
learning by four. Compensating both rates recovers the original trajectory
exactly; compensating them separately identifies their effects.

**Prediction and result.** Most of the immediate slowdown follows the geometry
rate. Readout feedback changes individual responses but is not a universal
dominant sink for the driving error. The fixed-map model, which evolves that
error, predicts 109 of 138 confirmation contrasts more accurately than
holding the initial effective force constant. It therefore captures useful
feedback beyond the initial rate change. A much smaller two-observable
restoration model fails for some targets. The
[complete intervention report](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/README.md)
preserves those failures and the target-level comparisons.

The conclusion is specific: at these later states, coupled residual evolution
helps explain motion; rapid restoration and universally readout-dominated
depletion are not supported as general explanations.

### 4.2 Why increasing width can make acquisition slow before errors relax

**Example: the same error load can produce much less motion.** In a separate
experiment with fresh initializations, six targets and two seeds are trained
at actual widths 177, 705 and 1409. At the 20k checkpoints, slopes, hidden
biases and readouts remain comparable to $W^{-1/2}$. Coarse tracking is small.
Over the next 20k updates, evolving the errors with a frozen map predicts
almost the same motion as holding the effective force constant. The later
checkpoint explanation cannot simply be transplanted to these states.

**Theory: almost affine features have weak fine sensitivity.** For small
$u=ax+b$, tanh differs from $u$ by at most $|u|^3/3$. The affine part vanishes
when projected onto fine error shapes. The surviving slope sensitivity
contains $c\,[\operatorname{sech}^2(u)-1]x$, whose magnitude is at most
$|c|u^2$. Thus $a,b,c=O(W^{-1/2})$ gives a force of order $W^{-3/2}$ per
neuron, provided the target load and the coarse-balance conditioning stay
bounded. The compensating coarse response has the same order; we include it.

Consequently, when tracking satisfies the corresponding small absolute bound,

$$
|\lambda_{j,n+1}-\lambda_{j,n}|=O(\eta W^{-5/2}),
\qquad h=O(W^{-1}).
$$

This is a conditional rate bound for the exact network. It permits changing
sensitivities and arbitrary empirical target values. The
[theorem and first-exit proof](d34_mechanism_rate_theorems.md#8-a-width-dependent-rate-without-freezing-the-effective-map)
give explicit constants and sufficient conditions for an interval on which
the small-parameter regime closes. They also show that rescaled parameters
can change appreciably while
the fine error changes only a little. We cannot extrapolate the rate beyond
that regime to claim a much longer acquisition time.

**Prediction and evidence.** Multiplying the measured per-neuron force by
$W^{3/2}$ should remove much of its width dependence. Across these three
widths, the median rescaled slope-force RMS is 0.628, 0.695 and 0.661.
The corresponding rescaled slope RMS is 1.456, 1.471 and 1.466.
These checkpoint measurements support the regime, although they do not prove
the interval-wide assumptions. None of the 36 runs reaches $\lambda=0.25$
through 40k updates. Their targets include mixed sine, degree five, a Gaussian,
a compact bump, a smooth step and a kink; the result does not rely on degree
nine's missing lower target modes.

### 4.3 A simpler evolving model predicts the missing response

**Example: match the initial force, then predict how it changes.** At the
widest fresh-seed checkpoints, holding the exact initial effective force
constant predicts the following slope motion with about 2.26% median error.
A model that evolves the polynomial feature geometry reduces that error to
0.026%. Both start from the same parameters and effective force. Their
different predictions test the feedback after the starting state.

**Model: keep the particle coupling, approximate the activation.** Replace
tanh by its cubic or quintic Taylor polynomial. At every predicted state,
recompute the fine errors, their sensitivities, and the coarse compensation.
Slopes, biases and readouts all evolve. In the cubic model, the generated
quadratic and cubic outputs depend on $\sum_jc_ja_j^2b_j$ and
$\sum_jc_ja_j^3$. These changing correlations determine how the target error
moves each neuron. This retains a concrete nonlinear mechanism; it is not
a two-variable closure or a Jacobian fixed at the checkpoint.

The pure quintic model works well for five of the six targets, but its
degree-five error already appears in its approximation of the initial force.
We therefore also test a fixed correction that makes the initial effective
force exact. If $F_5$ is the quintic field, this model uses

$$
\widetilde F_5(\theta)
=F_5(\theta)+\bigl[F(\theta_s)-F_5(\theta_s)\bigr].
$$

Only the bracket is fixed. The polynomial field continues to evolve with
its own parameters. No future trajectory is used to choose that correction.
It does not enforce exact coarse balance away from the fork; that mismatch
belongs in its approximation error.
Here the checkpoint force uses tanh with retained degrees 2–65. An audit with
degrees 2–129 agrees to numerical precision; “exact initial force” does not
claim an interval certificate for the full orthogonal complement.

**Prediction and confirmation.** After developing the model on seeds 30 and
31, we issue all forecasts for fresh seeds 32 and 33 before continuing their
ordinary GD runs from 20k to 40k updates. Each width has the same six targets
and two seeds. Median slope-displacement errors are:

| Actual width | Constant exact effective force | Pure quintic | Quintic with exact initial force |
|---:|---:|---:|---:|
| 177 | 20.0% | 2.86% | 0.876% |
| 705 | 4.71% | 0.0834% | 0.0547% |
| 1409 | 2.26% | 0.0324% | 0.0260% |

The corrected quintic model improves on constant force in all 36 confirmation
cases. The pure quintic improves in 30; degree five remains its exception.
At the smallest width, the corrected model's worst error is still 12.6%;
the median does not describe every target equally well.
This is confirmation across seeds of six fixed functions, not a claim over
all targets. The [approximation theorem](d34_polynomial_surrogate_theorem.md)
explains why such a model can be accurate in the small-parameter regime and
why small absolute truncation error can still be a large fraction of a weak
degree-five force. Its uniform error bounds remain conditional. Observed
forecast precision is evidence for the mechanism, not a numerical proof of
those conditions.

A secondary test asks how much error evolution is needed inside this model.
At widths 705 and 1409, the actual total fine-residual change is below 0.075%
over the development window. Holding the supplied polynomial error fixed
while evolving its sensitivities still predicts the generic-target motion
well. Degree five is more delicate: tiny changes in generated lower-mode
errors materially improve an already small forecast error. A nearly unchanged
total residual norm therefore does not justify discarding every error's
evolution. This secondary comparison was analyzed retrospectively; it was
not part of the fresh-seed confirmation.

## 5. From a motion mechanism to an acquisition bound

**Example: the relevant quantity is the distance still to travel.** A neuron
starting at gamma 0.2 cannot reach gamma 1 during an interval in which its
total outward travel is at most 0.1. It may move in both directions during
that interval. Its final displacement alone does not reveal whether it
crossed a threshold and came back.

**Theory: bound cumulative outward travel.** For neuron $j$, define

$$
P_{+,j}(N)=\sum_{n=0}^{N-1}
[\gamma_{j,n+1}-\gamma_{j,n}]_+,
\qquad [u]_+=\max(u,0).
$$

Suppose $B_j(N)$ is a bound established from the starting state and controlled
future dynamics, satisfying $P_{+,j}(N)\le B_j(N)$. Then

$$
\gamma_{j,0}+B_j(N)<\Gamma
\quad\Longrightarrow\quad
\gamma_{j,n}<\Gamma\ \text{for every }n\le N.
$$

Here $\Gamma$ is the scale relevant to the acquisition event under study.
Applying the argument neuron by neuron also bounds the fraction that can
newly reach it, while accounting for neurons already large at the checkpoint.
The diagnostic thresholds in the experiments are not universal necessary or
sufficient conditions for precision fitting.

The fixed-map model makes a candidate budget explicit. Each coupled error
direction has an initial amplitude, a sensitivity, and a relaxation rate.
Under the nonoscillating-step condition in the spectral calculation,
summing the motion supplied by those relaxing directions bounds its outward
travel. Weak sensitivity restricts travel over a finite interval, even when
the remaining error is large. This does not imply small eventual displacement.
The [spectral calculation](d34_coarse_balance_stagnation_details.md#122-the-fixed-effective-map-gives-a-finite-movement-budget)
derives that budget.

**The ordinary-GD theorem uses the measured mechanism.** Small coarse tracking
reduces two explicit allowances: its direct slope displacement and its later
effect through the fine errors. Omitted modes receive their own allowances.
Sensitivity evolution remains in the model: its direct effect and the residual
response it induces are combined with their signs before bounding uncertainty.
This permits appreciable map drift while retaining signed cancellation.

The [conditional acquisition theorem and proof](d34_effective_force_acquisition_theorems.md#4-theorem-2-ordinary-gd-acquisition-with-explicit-mechanism-corrections)
give a corrected forecast $\widehat a_{j,n}+\overline D_{a,j}(n)$ and
uncertainty $E_{a,j}(n)$. If

$$
\max_{n\le N}
\left(|\widehat a_{j,n}+\overline D_{a,j}(n)|+E_{a,j}(n)\right)<\Gamma,
$$

that neuron stays below the threshold at every update through $N$. Counting
the remaining neurons bounds how many can ever acquire the scale, even at
different times. This tests the whole enclosed path directly. The note also
proves a sufficient neighborhood condition for establishing the allowances
without knowing the future trajectory, and a separate lemma for persistence
of small coarse tracking. A small observed endpoint error alone proves none
of those conditions.

This is also the connection to the transport PDE. The parameter distribution
moves with a velocity determined by the same coupled gradient. Acquiring scale
requires enough mass to travel outward; bounding that travel limits acquisition.
The discrete argument above applies directly to GD, without adding diffusion
or assuming stochastic updates. The
[transport walkthrough](d34_barrier_theorem_walkthrough.md#how-the-transport-pde-fits-into-this-description)
gives the continuous formulation.

The [rescaled transport model](d34_rescaled_transport_model.md) makes that
velocity more explicit: each particle
responds through products of its slope, bias and readout, with coefficients
set by the shared residual moments. The transport remains self-consistent
and deterministic. Weak velocity, changing correlations, and cancellation
can limit outward movement without diffusion or a stationary trapping state.

For a fine target orthogonal through degree five, the higher-order reduced
model has an additional [transport energy bound](d34_transport_action_bound.md).
Its generated-error energy pays for squared particle travel, so it limits
the fraction that can ever acquire a specified scale during a finite window.
The large unresolved target error contributes no energy at this order.
This is a special-case result; sine and other targets with lower-mode loading
still use the general coupled model. Transfer to actual GD requires the
stated trajectory-error allowance, which has not been numerically evaluated
for this new bound.

**What is established, and what remains conditional.** The theorems prove
acquisition bounds under explicit conditions on effective dynamics and their
corrections. A moving neighborhood around the predicted trajectory lets us
check containment by induction, rather than assume that an observed future
path stays nearby. Numerical evaluation of this bound remains conservative
for several targets; accurate forecasts alone do not close it.

At the later checkpoints, the current formula evaluated in ordinary FP64
excludes every neuron through 20k further updates for 13 of 23 functions,
and through 200k for both tested degree-nine seeds. The other ten 20k bounds
become uninformative; this is not observed acquisition. The
[full enclosure report](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/moving_tube/README.md)
retains those failures. These longer computations do not control numerical
rounding and are distinct from the certificate below.

At independently initialized wider states, the same FP64 formula encloses
all six tested functions through 20k additional updates at actual widths 705
and 1409. Its correction radius decreases as width increases, in addition to
the smaller normalization factor $h$. This supports an informative
small-parameter regime, but does not establish an asymptotic law from three
widths or certify the arithmetic of that panel.
Selected longer checks show the remaining target dependence: mixed sine
retains exclusion through 50k additional updates at width 705 and 100k at
width 1409. The smooth-step bounds instead become uninformative at updates
25,095 and 39,096. These are limits of this FP64 enclosure, not measured
acquisition times. The full report retains all four attempted extensions.

There are now two rounding-controlled moving-reference instances. Starting from the
archived degree-nine checkpoint, all 177 neurons satisfy
$\lambda_j<0.002973994$ throughout the next 20,000 updates, well below 0.25.
For mixed sine at width 705, all neurons satisfy $\lambda_j<0.002$ through
13,000 additional updates from its 20k checkpoint. Its attempted 20k
enclosure becomes uninformative after that prefix; the failure is retained.
The [certificate and proof](d34_certified_instance.md) bound every intervening
state using interval arithmetic. They apply to exact GD on the respective
archived empirical dataset, initialized at its checkpoint. They do not
certify earlier training history, floating-point training roundoff, the other
moving tubes, or a population loss.

A useful baseline prevents us from overstating this gain. A
[generic GD energy argument](d34_energy_baseline.md) already excludes
$\lambda=0.25$ for 20k updates when the initial combined $(a,b,c)$ norm is
at most 3 and the half-MSE is at most 1, at our step size. It says little
about the small motion inside that allowance. Those initial conditions are
now verified with rounding control for all 18 seed-30 width-panel states:
six functions at each of three widths. Their tighter verified initial loss
bound extends this generic exclusion to 50k additional updates, by the same
proof and without running a longer trajectory. The effective-force theory
adds direction, target dependence, and a much tighter trajectory prediction.
Its value must be judged on those quantities as well as the distant threshold.

There is no privileged 200k barrier to prove. A useful exclusion time should
follow from the starting state, the evolving force, the learning rate, and
the distance to acquisition. It can differ across targets and states. An
acquisition bound can also remain informative without predicting every small
movement accurately: its travel allowance only needs to stay below that distance.

### A stronger degree-9 result is available as a special case

The degree-nine analysis separately bounds the small amount of loss that GD
can remove inside a neighborhood. Descent then limits total parameter travel
and excludes acquisition from specified stalled checkpoints. This gives a
stronger persistence result for that regime; it is not a required time scale
or mechanism for other targets. Its
[proof and numerical evaluation](d34_coarse_balance_stagnation_details.md#9-from-a-force-decomposition-to-a-persistence-prediction)
are retained in the companion.

## 6. What we currently understand

**Example: fitting can improve without resolving the acquisition question.**
The Gaussian example in Section 1 improves its fit and grows some slopes;
the degree-nine example in Section 2 contracts slightly while retaining a
large error. A theory must allow both. Their common description is the
effective force, not a universal inward direction.

**The refined hypothesis.** After the coarse transient, the effective fine
force determines motion. Nearly affine states suppress that force even when
substantial target error remains. Geometry and readout correlations can evolve
before that error appreciably relaxes. At later states, residual relaxation
also improves predictions. Correcting generated errors and competing target
errors can point in different scale directions. The shared explanation is
weak or insufficiently sustained outward coupling, not one universal sign,
one dominant parameter block, or rapid return to a fixed small-scale geometry.

**The new persistence test.** At width 1409, removing residual relaxation
changes positive slope travel by only 0.00505% at the median of twelve
states; doubling geometry feedback changes it by 1.51%. These interventions
start with the same gradient. Thus the wide-state explanation is often a
weak force that reinforces slowly, rather than a strong force continuously
cancelled by rapid error correction. Later states can behave differently.
For one late sine state, reinforcement initially wins, but error relaxation
strengthens enough to reverse the force-growth rate during the next 20k
updates. This is a state-dependent distinction within the effective fine
force, not a return to coarse disequilibrium as the primary explanation.

The [persistence note](d34_state_dependent_persistence.md) gives an exact
identity for this mechanism. Here $F$ denotes all coordinates of the same
effective fine gradient whose slope block is $F_a$ above. Its squared norm
changes according to

$$
\frac{d}{dt}\frac{\|F\|^2}{2}
=-\mathcal D-\mathcal C-F^TDF[R].
$$

$\mathcal D\ge0$ is residual relaxation; the signed $\mathcal C$ is the
effect of changing geometry under the remaining error load, including
coarse compensation. The last term explicitly charges disequilibrium's
effect on force evolution. At bounded rescaled parameters, the new theorem
bounds force magnitude by $O(W^{-1})$, intrinsic amplification rate by
$O(W^{-1})$, and the relative relaxation contribution by $O(W^{-2})$.
It permits growing forces and assumes persistent small tracking. It also
states the geometric and conditioning conditions needed to keep these
orders valid over a finite interval.

The physical-kick thread sharpens the coupling requirement. Matching slopes
exactly, and their first update and force magnitude to first order, still
leaves later responses different. On the six-target width-177 panel, the scalar model has median
response error near 88–89%, and a frozen full Hessian has about 93% error.
Evolving the anchored quintic effective-force model reduces this to about
14% in both cohorts. These follow-up comparisons are retrospective and
their errors remain larger than those for the unperturbed paths. Late pure
sine remains unresolved: one quintic case fails numerically and the other
predicts the response poorly. The mechanism needs evolving coupled geometry;
its small-parameter approximation is not uniformly successful at late states.

**We need an upper bound, not an exact forecast.** A frozen scalar growth
rate still misses some coupled motion. That can matter for mechanism
identification without defeating an acquisition bound. The late sine
forecast misses force magnitude by 156%, but it overpredicts it. Across
the forty natural-GD forks, the largest saved-time ratio of actual to
predicted speed is 1.477. A factor-two reference budget exceeds the observed
accumulated effective-force travel in every case. These are retrospective
diagnostics, not a proof that the same factor controls every intermediate
state or a longer window.

The [constant-factor theorem](d34_state_dependent_persistence.md#8-a-forecast-can-miss-the-trajectory-and-still-give-a-useful-upper-bound)
formalizes the weaker objective: prove $\|F_n\|\le M\widehat q_n+b_n$,
where $\widehat q_n$ is a preissued speed forecast and $b_n$ bounds tracking's
propagated effect. Add direct slope tracking, sum the speeds, and apply the
travel bound from Section 5. The multiplier $M$ costs linearly in travel and
quadratically in the allowed acquisition fraction. It need not be close to
one when the remaining scale distance is large. Directional prediction is
valuable for explaining motion; this conservative upper bound does not
require it to be accurate.

**The theoretical objective is persistent rate suppression.** In one exact
reduced correction ODE, rescaled slopes decrease like $\tau_2^{-1/6}$ while
readouts grow like $\tau_2^{1/6}$. The slope speed falls like
$\tau_2^{-7/6}$. The system keeps changing and has no finite parameter
equilibrium. Correction weakens its own sensitivity, so the force becomes
small as an outcome of the dynamics. These are derived rates for that
restricted ODE, not fitted exponents or a theorem for every training target.

The [mechanism note](d34_coupled_ode_mechanisms.md) develops three connected
arguments with proofs. First, maintaining a coarse contribution can require
large compensating readout motion, producing long escape times even when the
target pushes outward. This compensation is already part of the effective
fine flow at zero disequilibrium. Second, explicit target and generated-error
coefficients determine the time needed to pass through a scale interval.
Third, a population-level correction equation gives either a finite correction
budget or algebraically slow continuing motion, according to how sensitivity
changes with the remaining error. For the latter mechanism, an independently
derived scalar example has squared sensitivity proportional, up to constants,
to the cube of the generated-error norm. For zero competing drive, the
population theorem shows that where this relation holds, the full-parameter
path allowance grows only as $T^{1/6}$ in reduced time on the established interval. An attracting
equilibrium is treated only as a separate contrast case.

**One heterogeneous persistence mechanism now has a proof.** The
[population theorem](d34_population_persistence_theorem.md) starts with an
aligned readout–slope sector and retains the full coupled moments. The common
compensation preserving the coarse fit is a weighted average of local fine
penalties. Under a specified ordering of the target and generated-error
loads, this average cannot push the largest slope outward. Smaller slopes
can still grow. The equations preserve the alignment, bound readout growth,
and keep the coarse projection conditioned.

When a fifth-order target load pushes outward, the same proof gives an
explicit finite persistence horizon from initial moments. It derives how
long correction retains its advantage, rather than assuming that a future
fine-force bound holds. A second result derives tracking control and converts
the reduced persistence into a bound on the fraction of neurons that ever
acquire a chosen normalized scale. The note includes the discretization and
containment conditions for ordinary GD.

The [coverage audit](../results/checkpoint_D_optimizers/expD34_readout_race/population_coverage/README.md)
rules out direct application of this sector theorem to the archived states:
zero of 223 satisfy the cone, and the median negative-product fraction is
44.6%. Biases are also non-negligible in rescaled coordinates. These are not
rare exceptional neurons. The theorem remains a sufficient illustration;
the next structural argument must keep mixed signs and bias moments in the
shared compensation. Generated-error correction can also lose sensitivity,
so the separate algebraic-slowing mechanism remains relevant. Neither result
justifies assuming strong restoration in the wide panel.

Generic parameter-maxima bounds currently support at most eighteen updates
in the recent first-exit audit, even with tracking set to zero. That is a
limitation of those constants, not observed escape after eighteen updates.
The immediate objective is a mechanism-based lower bound on acquisition time
with a justified duration. Accurate prediction of every small movement is
not required. The [experiment report](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_persistence/README.md)
retains the numerical limitations alongside the successful comparisons.

The force reduction and coupled forecasts have evidence across functions;
the width-dependent rate has a conditional proof; selected empirical instances
have finite rounding-controlled certificates. A broadly useful theorem still
needs a preserved structural condition shown to cover the observed
heterogeneous states, with its transfer errors controlled. The new theorem
proves preservation in an explicit sector whose direct empirical coverage
has now failed. Accurate wide-state polynomial force reconstruction supports
the fuller coupled model, but does not establish a future feedback bound or
an autonomous trajectory forecast. None of these claims requires
permanent trapping or establishes entry into the post-transient regime from
initialization.

Two related observations are useful but are not prerequisites for this argument.
The [frozen-geometry readout study](../results/checkpoint_D_optimizers/expD34_readout_race/readout_scale/README.md)
finds small, approximately $O(h)$ individual readouts at large fixed slopes,
where $h$ is center spacing. That supports an available representation; it
does not make readout size a scalar cause of scale failure. The
[Adam moment interventions](d34_adam_moment_results.md) show that changing
tracking's second-moment input can affect scale despite its cancelling signed
motion. The effect is small and target-dependent. Repeating the initial force
phases fails to predict it, so an Adam explanation must also evolve the force
and moment history. The GD surrogate and theorem above do not automatically
apply to Adam.

For proofs, use the [technical companion](d34_coarse_balance_stagnation_details.md).
For complete measurements, use the
[mechanism campaign report](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/README.md)
and the earlier [force audit](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/README.md).
They support this argument; following it does not require reading either first.
