# Why GD acquires scales slowly: a tutorial on the effective gradient

We know that these networks can represent much better approximations than
ordinary GD sometimes finds. The problem we want to explain is the **rate
at which training acquires useful slopes**. A representation can exist while
the gradient takes a very long time to move the network toward it.

Our current explanation begins with one empirical finding: after the early
coarse-fitting transient, departure from coarse balance contributes very
little to GD's slope motion. Most motion is explained by the **effective
fine-gradient**. Understanding what that phrase includes is the first task
of this tutorial. It includes a substantial coarse contribution, even when
tracking error is tiny.

**Five terms to keep in mind. The symbols will be introduced when we use them.**

| Term | Meaning here |
|---|---|
| Scale | The magnitude of a neuron's slope; a larger magnitude gives a steeper tanh feature. |
| Coarse error | The constant and linear parts of the output residual. |
| Fine error | Everything remaining after those two parts are removed. |
| Gain | The current map that converts residual error into a parameter gradient. |
| Tracking error | The difference between the actual coarse error and its moving balance. |

Sections 1–6 build one argument: identify the slope gradient, explain its
two gain terms, identify the errors driving it, and turn their slow motion
into a bound. Section 7 states what we can now claim. The technical details
afterward are a second reading path; the main argument does not require
working through them first.

## 1. Begin with the quantity that moves a slope

We first need to distinguish a large remaining error from a large slope
update. This will explain why the loss alone cannot tell us the acquisition
rate.

Our network is

$$
f(x)=d+\sum_{j=1}^{W}c_j\tanh(a_jx+b_j).
$$

Here $W$ is the number of neurons. Each $a_j$ is a slope, $b_j$ shifts
the feature, and $c_j$ determines how much that feature contributes to the
output; $d$ is the output bias. All of these parameters train. For target
$y$, the residual is $r=f-y$, and the loss is half its mean square.

With step size $\eta$ and update count $n$, GD changes a slope according to

$$
a_{j,n+1}=a_{j,n}-\eta g_{a,j,n}.
$$

The gradient $g_{a,j,n}$ is therefore the quantity to explain. Notice the
minus sign: at a positive slope, a positive gradient causes contraction.
A gradient norm tells us how much motion is available, but not whether
that motion increases slope magnitudes.

There is also a distinction between error and sensitivity to error. A
residual can be large while barely changing when a slope moves. That error
then produces little slope gradient. This is the situation we will find
for the hard degree-9 component.

For the rest of the note, keep the following chain in mind:

**remaining error → current gain → slope gradient → accumulated movement.**

The theorem must control the last quantity. Our experiments explain the
middle two links.

## 2. Why a well-tracked coarse error need not be zero

Before deriving the effective gain, we need to understand the balance it
uses. The key point is that fitting fine error can disturb the coarse
output: the same parameters affect both.

Let $e_C$ collect constant and linear error, and let $e_H$ collect fine
error. The coarse error has two sources of change. It relaxes under its own
gradient, and it is affected by parameter changes driven by fine error.

A scalar example makes this concrete. Suppose its current equation is

$$
\dot e_C=-9e_C-3e_H.
$$

If $e_H=1$, setting $e_C=0$ does not stop coarse motion: the derivative is
then $-3$. Instead, the two contributions balance at $e_C=-1/3$.
If the fine error later becomes $0.6$, the balance moves to $-0.2$.
Accurate tracking means staying near this changing value.

The matrix version has the same meaning:

$$
\dot e_C=-K_{CC}e_C-K_{CH}e_H.
$$

$K_{CC}$ describes the response to coarse error; $K_{CH}$ describes how
fine fitting affects coarse error. These matrices include the effects of
all trained parameter blocks. With the current matrices and fine error
held fixed, solving for zero coarse velocity gives

$$
e_C^{\rm bal}=-K_{CC}^{-1}K_{CH}e_H.
$$

This calculation assumes that $K_{CC}$ is invertible. We call the result
the instantaneous coarse balance. It is a relation at the current state,
not an instruction to freeze parameters or replace training by a solve.

Now define the tracking error as actual minus balanced coarse error:

$$
z_C=e_C-e_C^{\rm bal},\qquad e_C=e_C^{\rm bal}+z_C.
$$

We can now interpret the small-tracking observation correctly. It says that
$z_C$ is small in the force-relevant sense. It does not say that the
balanced coarse error, or the gradient it produces, is small.

The dots above describe the gradient-flow vector field. We use it to expose
the balance at a GD state. The gradient decomposition next is algebraic
and holds at that state without taking a small-step limit.

## 3. Two contributions remain when tracking is accurate

We now substitute that balance into the slope gradient. The purpose is to
identify exactly what remains after the small tracking correction is set
aside.

Write $J_{a,H}$ for the sensitivity of fine error to slopes, and $J_{a,C}$
for the sensitivity of coarse error to slopes. Their transposes convert
the corresponding errors into slope gradients:

$$
g_a=J_{a,H}^Te_H+J_{a,C}^Te_C.
$$

Substituting $e_C=e_C^{\rm bal}+z_C$ gives three contributions:

$$
\begin{aligned}
g_a={}&\underbrace{J_{a,H}^Te_H}_{F_{\rm direct}}\\
&+\underbrace{\left(-J_{a,C}^TK_{CC}^{-1}K_{CH}e_H\right)}_{F_{\rm balanced}}\\
&+\underbrace{J_{a,C}^Tz_C}_{F_{\rm tracking}}.
\end{aligned}
$$

The first is the direct fine-residual gradient. The second is the gradient
of the coarse residual required to balance fine fitting. The third is the
additional gradient caused by departure from that balance.

Grouping the first two gives the effective fine-gradient:

$$
\boxed{
F_a=T_ae_H,\qquad
T_a=J_{a,H}^T-J_{a,C}^TK_{CC}^{-1}K_{CH}.
}
$$

It is called “fine” because the remaining input is $e_H$. Its gain includes
the induced coarse response. **Small tracking permits
$g_a\approx T_ae_H$; it does not permit dropping the second term in $T_a$.**

### Follow the scalar example through the gradient

For the scalar balance above, take the direct slope gradient to be $e_H$
and the coarse slope gradient to be $3e_C$. At $e_H=1$ and its balance
$e_C=-1/3$,

$$
g_a=\underbrace{1}_{\text{direct fine}}
+\underbrace{3(-1/3)}_{\text{balanced coarse}}=0.
$$

Ignoring the balanced term would predict a nonzero slope update. In this
example, the readout can still change and reduce fine error while the
instantaneous slope velocity is zero. The compatible Jacobians are given
in [Technical detail A](#technical-detail-a-exact-definitions-and-fine-fitting).
This is an example at one balanced state; subsequent motion can create
tracking error as the balance moves.

### Check the distinction against the actual GD measurements

The real data do not require exact cancellation to make the second term
important. Compare the following separately recorded gradient norms.

**GD seed 0 near 600k updates. These are two individual states; the norms
measure gradient magnitude, not outward movement.**

| Target | Direct fine | Balanced coarse | Their effective sum | Tracking correction |
|---|---:|---:|---:|---:|
| Degree 3 | $8.77\times10^{-4}$ | $8.98\times10^{-4}$ | $3.13\times10^{-4}$ | $2.85\times10^{-7}$ |
| Degree 9 | $5.99\times10^{-6}$ | $1.55\times10^{-6}$ | $5.67\times10^{-6}$ | $2.24\times10^{-11}$ |

For degree 3, the first two contributions are comparable and substantially
oppose each other as vectors. For degree 9, the direct term is larger,
but the balanced term remains appreciable. This degree-9 example does not
show near-complete cancellation as the explanation for its small force.
In both rows, tracking is a different, much smaller contribution.

The direction of movement adds another insight. In degree-3 seed 0, over
20k–600k updates, the direct fine contribution to mean absolute slope is
$-0.0893$, while the actual coarse contribution is $+0.2429$. Their sum,
including a small zero-crossing correction, gives net growth of $0.1537$.
The coarse contribution contains balanced plus tracking forces; the
separate small-tracking audit supports attributing most of it to balance.

Thus the balanced coarse contribution helps produce outward motion in this
trajectory. The direct fine contribution alone would predict the wrong
sign. For degree 9, coarse motion also offsets direct contraction, but
does not overcome it. A minus sign in the gain formula is therefore not a
universal statement that coarse balancing suppresses outward motion.

**What we have established:** both gain terms matter. The empirically
small correction is departure from balance. We now have the correct
gradient to analyze, rather than a reason to remove all coarse effects.

## 4. Identify which fine errors drive the combined gain

We have answered what is inside $T_a$. The next question concerns its
input: which parts of $e_H$ actually move the slopes? This is a different
decomposition, and it must preserve both terms we just derived.

A polynomial mode is a fixed normalized polynomial of a given degree;
its residual coefficient measures how much error lies in that output
direction. Let $e_k$ be this coefficient for mode $k$. A column $T_{a,k}$
maps that coefficient into a slope gradient. Then

$$
F_a=T_{a,2}e_2+T_{a,3}e_3+\cdots+T_{a,9}e_9+\cdots.
$$

Every $T_{a,k}$ contains its direct gain **and** its balanced coarse
correction. Identifying an important mode does not identify one of those
gain terms as negligible.

The degree-9 target contains constant, linear, and ninth-degree components.
It has no target quadratic or cubic component. Nevertheless, the network
generally produces quadratic and cubic output. Those unwanted components
are residual errors too, and GD tries to remove them.

The measurements show a large disparity. The hard residual coefficient is
about $0.866$, but its effective force is of order $10^{-10}$. Generated
lower-mode errors produce forces of order $10^{-6}$–$10^{-5}$. In the
audited states, their combined inward contribution to mean-scale velocity
is roughly 8,000–30,000 times the hard mode's outward contribution.

This resolves an apparent puzzle: a small output error can dominate
parameter motion when the current network is much more sensitive to that
error than to the large hard residual. GD can spend its motion correcting
its own generated curvature while making little progress on the target.

The quadratic contribution is essential: it supplies 14–63% of generated
inward mean motion in the earlier audit. A cubic-only explanation misses
the effect of hidden biases and heterogeneous neurons.

### A prediction, followed by its intervention test

If penalizing generated lower-mode output causes contraction, reducing
those penalties should reduce contraction. It need not create fast
acquisition, because the hard-target gain may remain weak.

**Degree-9 mean-scale changes over 600k–1.1m, medians across independent
seeds 20–24. Weight 1 is ordinary GD. Other weights change the loss penalties
on residual modes 2–8, holding the target and starting parameters fixed.**

| Lower-mode weight | Change in mean absolute slope |
|---|---:|
| 0 | $+3.1\times10^{-8}$ |
| 0.1 | $-7.9\times10^{-6}$ |
| 1 | $-7.5\times10^{-5}$ |
| 10 | $-6.4\times10^{-4}$ |

The response follows the proposed mechanism: contraction decreases when
the penalties decrease. But removing them leaves negligible motion, and
all arms retain relative evaluation MSE about $0.750006$. This supports
that the source of contraction and the weakness of acquisition are
separate parts of the explanation.

The tiny positive number at weight zero is not evidence of useful
acquisition. Numerical controls also do not establish its sign as an
invariant feature under grid refinement.

**What this changes in the theory:** a small hard-target gain is not enough
to describe the full gradient. We must retain the generated-error terms,
their signs, and the motion they can supply. These conclusions combine
the [earlier mode audit](d34_coarse_balance_stagnation.md#7-what-the-audit-and-continuations-establish)
with the [longer independent interventions](../results/checkpoint_D_optimizers/expD34_readout_race/force_plateaus/README.md).

## 5. Explain why the effective gradient changes so slowly

An instantaneous small gradient does not explain a long acquisition time.
It might grow sharply later. We therefore need to understand the evolution
of the product $F_a=T_ae_H$.

The first useful calculation is only the product rule:

$$
\dot F_a=
\underbrace{\dot T_ae_H}_{\text{the gain changes}}
+\underbrace{T_a\dot e_H}_{\text{the driving error changes}}.
$$

These terms distinguish two possible explanations for a changing force.
Parameters can alter how strongly errors couple to slopes. Alternatively,
errors can relax through a nearly unchanged gain. The coarse-balance
equations further separate tracking influence from fine-error relaxation;
the full expression appears in [Technical detail B](#technical-detail-b-how-the-gain-and-residual-evolve).

To compare these mechanisms, we measure how each contribution increases
or decreases the force magnitude. The gain-change term separates into
effects of hidden-parameter motion and readout motion; the residual-change
term separates into fine fitting and tracking influence.

In the original degree-9 diagnostic states, fine-residual evolution
accounts for about 89–95% of the decline in the force norm. Hidden-parameter
changes account for about 5–9%, readout changes for less than 2%, and
tracking is negligible. These are shares of the decline rate, not shares
of scale growth or of the gradient itself.
The contributions do not conceal large opposing rates. Replenishment of
the generated errors by the hard mode is also very small.

The explanation is therefore slow relaxation of generated errors, rather
than substantial replenishment maintaining them. The two force-carrying
directions closely associated with quadratic and cubic error have local
relaxation times of approximately **3 million** and **50–80 million
updates** per e-fold. An e-fold is the time to fall to about 37% of the
starting amplitude. A few hundred thousand updates is short on these scales.

The long continuation supports this interpretation. From 600k to 6m,
quadratic error falls to 15–25% of its starting magnitude, while cubic
error retains 90–94%. The hard residual barely changes. These are measured
changes, not an extrapolation of constant rates indefinitely.

We also tested a prediction: start at a checkpoint, hold the full model
Jacobian fixed, and evolve its linearized GD dynamics without consulting
future true states. Over 600k–6m, its slope-displacement error is 4.1–5.5%
of the actual displacement. Holding the gradient itself constant gives
71–114% error. Accounting for relaxation therefore explains substantially
more of the motion than a constant-force picture.

The [forecast comparison](d34_coarse_balance_stagnation.md#how-much-coupling-must-evolve-to-predict-the-motion)
does not make the gain exactly fixed. Later geometry changes become a
material correction, and degree-5 recovering trajectories show why this
approximation cannot be applied to every regime. What it establishes is a
testable local mechanism for the degree-9 rates.

## 6. Turn slow motion into a theorem about acquisition time

We can now connect the mechanism to the desired theorem. The essential
comparison is between **how far slopes can move** and **how far they must
move to acquire the specified population**.

### First understand the distance that acquisition requires

Imagine three absolute slopes, $0.1$, $0.2$, and $0.8$. To make two of them
reach 1, the cheapest choice is to move the last two by $0.8$ and $0.2$.
Their combined Euclidean displacement is

$$
\sqrt{0.8^2+0.2^2}\approx0.825.
$$

A path of total Euclidean length at most $0.5$ cannot accomplish this,
however it distributes motion among the neurons. This remains true if
some neurons move inward or reverse direction along the way.

For a general starting slope vector, the same construction defines the
distance $\mathcal D_{p,\Gamma}$ to acquiring fraction $p$ at scale
$\Gamma$. GD's accumulated slope path is
$\eta\sum_n\|g_{a,n}\|$. Thus a bound

$$
\underbrace{\eta\sum_n\|g_{a,n}\|}_{\text{available movement}}
<
\underbrace{\mathcal D_{p,\Gamma}}_{\text{required movement}}
$$

excludes that acquisition event throughout the bounded interval. The
exact formula and argument are in [Technical detail C](#technical-detail-c-the-population-distance-and-signed-motion).

This is why mean contraction is not our theorem premise. About half the
neurons can move outward in the audited degree-9 states. What matters is
whether their available travel can deliver the specified scales.

### Two bounds serve different purposes

We currently have one route that follows the measured force mechanism
closely, and another that already gives a longer bound for full GD. Their
roles should be kept distinct.

**The mechanism route sums the predicted force.** To see the idea, suppose
one gradient component has magnitude at most $A\rho^n$, with $0<\rho<1$.
Its accumulated movement is bounded by the geometric series

$$
\eta\sum_{n=0}^{N-1}A\rho^n
=\eta A\frac{1-\rho^N}{1-\rho}.
$$

Slow relaxation can sustain motion for many updates, but this transient
has a finite total budget. A weak component that stays approximately
constant instead contributes a term proportional to $N$. The two-mode
model consequently produces: generated-error transient movement, plus
slow accumulation from the hard drive, plus any error needed to transfer
the prediction to real GD.

The last part is essential. The reduced-model budget is derived, but its
long transfer to the nonlinear trajectory needs control of changing gains,
other modes, tracking, and hard-residual drift. For the richer frozen
full-tangent forecast, our conservative prediction-error bound currently
closes for about 278k–438k further updates, even though forecasts agree
well over longer measured intervals. Transferring the two-mode reduction
also requires a bound on the terms it omits. Accuracy observed afterward
cannot substitute for a forward bound.

**The full-GD route bounds how much loss can fund movement.** The hard
residual is large, but within a suitable neighborhood the network cannot
remove much of it. An analytic lower bound on the loss in that neighborhood
lets us count only the loss that can actually decrease there.

Call this available loss $\Delta$, and write $g_n$ for the gradient over
all parameters at update $n$. GD descent controls the sum of squared
gradient norms. Cauchy–Schwarz converts it to a bound on total parameter
travel, which also bounds slope travel:

$$
\eta\sum_{n=0}^{N-1}\|g_n\|
\le\sqrt{\frac{\eta N\Delta}{q}}.
$$

Here $q>0$ is the factor in the descent guarantee. For fixed step size and
distance, a smaller $\Delta$ requires a longer time to travel that distance.
This is the rate restriction we need. The proof also checks that the
trajectory stays in the neighborhood where its loss floor is valid; it
does not assume that conclusion.

At the retained degree-9 checkpoints, the remaining half-MSE is about
$0.375$, but the loss available above the local floor is only of order
$10^{-6}$. Using the full $0.375$ in a generic bound would miss this
large difference.

The existing theorem conditions, evaluated from ten starting checkpoints,
give a common bound of **ten million additional updates**, with every
absolute slope below **0.388**. This is a prediction from the starting
state and analytic neighborhood bounds, not a report that all those
updates were simulated. The numerical constants were evaluated in FP64;
they have not been certified with directed-rounding interval arithmetic.

The full-GD result retains changing gains and tracking. It establishes
slow acquisition without proving the detailed modal description at every
future step. The modal experiments explain the observed motion; the
loss-floor theorem independently limits how much motion GD can accumulate.
[Technical detail E](#technical-detail-e-the-full-gd-bound-and-its-closure)
spells out this proof.

## 7. The understanding we should carry forward

The evidence supports a specific chain, with a different job for each link.

**First, accurate coarse tracking simplifies the slope gradient.** The
remaining quantity is the two-term effective gain acting on fine error.
Balanced coarse forcing is retained. Across the expanded GD study,
tracking accounts for at most 0.469% of the summed component-norm budget
over 20k–600k. Removing slope tracking in later continuations has little
effect on the degree-9 outcome. These observations support a small
tracking allowance in a conditional theorem.

**Second, the degree-9 effective force is dominated by generated errors.**
Their relaxation explains most observed motion. The hard target can remain
largely unfitted because its gain is much weaker. Lower-mode penalty tests
support this separation between contraction and acquisition.

**Third, the rate mechanism supplies a candidate movement budget.** Two
slow generated-error directions and a weak hard drive give a quantitative
reduced model. We have a partial forward error bound for that route, and
a separate full-GD theorem with a much longer finite-time horizon.

The remaining obligation is precise: bound the changes and errors strongly
enough for the acquisition claim we want. Small terms relative to total
contraction may still matter relative to the tiny outward hard-mode drive.
In particular, modes 4–8 contribute 18–84% of that outward contribution
in the earlier audit. Discarding them may be adequate for bulk motion but
inadequate for a sharp acquisition-rate prediction.

Small slope tracking alone also does not guarantee small influence on
future residual evolution. The theory has explicit tracking equations for
that purpose. Their conditioning, drift, and forcing terms need bounds
over the interval being claimed.

This is the appropriate purpose of further analysis: test or bound a named
term in this chain. Each additional experiment should state its framework
prediction, the observation that would contradict it, and how either
outcome changes the theorem. The latest long runs mainly strengthen and
delimit this mechanism; they do not replace it with a new explanation.
Adam remains a separately scoped comparison because its update histories
and adaptive scaling do not satisfy the same GD reduction.

---

The main argument ends here. The following details make its definitions,
proofs, and numerical examples checkable without making them prerequisites
for the first reading.

## Technical detail A: exact definitions and fine fitting

Here we supply the definitions behind the gain and show why fine-error
reduction need not imply slope movement.

Use the empirical mean inner product
$\langle u,v\rangle_m=m^{-1}\sum_i u(x_i)v(x_i)$ and orthonormal residual
modes $q_k$. Their coefficients are $e_k=\langle q_k,f-y\rangle_m$.
Take $H$ to be the complete empirical complement of constant and linear
functions; a truncated basis requires additional omitted-residual terms.

For a parameter block $I$, define $J_I=\partial e/\partial\theta_I$.
In particular,

$$
(J_a)_{kj}=c_j\langle q_k,x\operatorname{sech}^2(a_jx+b_j)\rangle_m.
$$

With $J=[J_a\ J_b\ J_c\ J_d]$, orthonormality gives
$g=J^Te$, $K=JJ^T$, and the flow equation $\dot e=-Ke$.
Equal learning rates for all blocks are assumed. Write
$B=K_{CC}^{-1}K_{CH}$ and $T_I=J_{I,H}^T-J_{I,C}^TB$.
Substitution into the fine equation gives

$$
\dot e_H=-Se_H-K_{HC}z_C,\qquad
S=K_{HH}-K_{HC}B.
$$

The effective gains satisfy a useful identity. Expanding their squared
sum and using $K_{CC}B=K_{CH}$ gives

$$
\begin{aligned}
\sum_I T_I^TT_I
&=K_{HH}-K_{HC}B-B^TK_{CH}+B^TK_{CC}B\\
&=S.
\end{aligned}
$$

Consequently, at exact balance, $e_H^TSe_H$ is the fine-error energy
dissipation through all blocks, while $\|T_ae_H\|^2$ is only its slope
share. This proves the distinction used in the tutorial.

The scalar example uses $J_C=(3,0)$ and $J_H=(1,1)$, with coordinates
named slope and readout. These yield $K_{CC}=9$, $K_{CH}=3$, $T_a=0$,
$T_c=1$, and $S=1$. At the stated balance, fine-error fitting proceeds
through the readout while the instantaneous slope velocity vanishes.
It is an algebraic example, not a calibrated model of a D34 trajectory.

## Technical detail B: how the gain and residual evolve

Here we expand the product rule used in Section 5. This identifies the
terms a future bound must control separately.

Using the fine-residual equation from detail A,

$$
\dot F_a=\dot T_ae_H-T_aSe_H-T_aK_{HC}z_C.
$$

The terms are gain change, fine-residual relaxation, and the tracking
influence on residual evolution. Differentiating the gain explicitly gives

$$
\dot T_a=\dot J_{a,H}^T-\dot J_{a,C}^TB-J_{a,C}^T\dot B,
\qquad
\dot B=K_{CC}^{-1}(\dot K_{CH}-\dot K_{CC}B).
$$

These expressions retain changes in both gain terms. A contribution $D$
to $\dot F_a$ contributes $F_a^TD/\|F_a\|^2$ to the logarithmic rate of
the force norm, when that norm is nonzero. The derivative measurements
use this flow vector field evaluated at GD states; they are not exact
finite-step differences.

The reported 89–95% residual-relaxation share concerns 25 early degree-9
diagnostic states. Later geometry change can account for about 20% of the
decline in some states. The measured relaxation times belong to the local
reduced model, not fixed constants of nonlinear training.

## Technical detail C: the population distance and signed motion

Here we justify the distance comparison in Section 6 and account for the
absolute value when interpreting signed movement.

At starting update $s$, sort $d_j=(\Gamma-|a_{j,s}|)_+$ increasingly.
Then the distance to having at least fraction $p$ of slope magnitudes at
least $\Gamma$ is

$$
\mathcal D_{p,\Gamma}(a_s)
=\left[\sum_{j=1}^{\lceil pW\rceil}d_{(j)}^2\right]^{1/2}.
$$

Each chosen coordinate costs at least its deficit; choosing the cheapest
required coordinates achieves the bound. For every $k\le N$, GD obeys

$$
\|a_{s+k}-a_s\|
\le\eta\sum_{n=s}^{s+k-1}\|g_{a,n}\|
\le\eta\sum_{n=s}^{s+N-1}
\big(\|F_{a,n}\|+\|F_{{\rm tracking},n}\|\big).
$$

Bounding the last expression below $\mathcal D_{p,\Gamma}$ excludes
acquisition at every prefix. This is an exact discrete-GD statement.
A sum measured after training is retrospective; a predictive application
needs an independently obtained envelope.

For $\bar\gamma=W^{-1}\sum_j|a_j|$ and $\delta a_n=-\eta g_{a,n}$,

$$
\bar\gamma_{n+1}-\bar\gamma_n
=-\frac\eta W\operatorname{sign}(a_n)^Tg_{a,n}+X_n,
$$

$$
X_n=\frac1W\sum_j\left[
|a_{j,n}+\delta a_{j,n}|-|a_{j,n}|
-\operatorname{sign}(a_{j,n})\delta a_{j,n}
\right]\ge0,
$$

with $\operatorname{sign}(0)=0$. The nonnegative remainder handles
crossings through zero. It must accompany signed gradient attribution;
it is not an additional force.

## Technical detail D: the two-mode movement budget

Here we give the matrix version of the geometric-series argument. Its
assumptions are explicit so that its theorem status is not confused with
the full-GD result.

Freeze the full-parameter effective tangents of modes 2 and 3 as columns
of $T_G$, and let $u=(e_2,e_3)^T$. Hold the hard residual and its tangent
fixed, giving $h=T_{\cdot,9}e_9$. Set tracking to zero and omit fine modes
other than 2, 3, and 9. Under these approximations,

$$
u_{n+1}=u_n-\eta(A_Gu_n+T_G^Th),\qquad
\widehat\theta_{n+1}=\widehat\theta_n-\eta(T_Gu_n+h),
\qquad A_G=T_G^TT_G.
$$

Assume $A_G$ is positive definite. Its forced equilibrium is
$u_*=-A_G^{-1}T_G^Th$, leaving force $f_\infty=T_Gu_*+h$.
For orthonormal eigenvectors $v_i$ with eigenvalues $\nu_i>0$, set
$b_i=v_i^T(u_0-u_*)$. Solving each scalar recurrence gives

$$
\widehat g_n=f_\infty+
\sum_{i=1}^{2}b_i(1-\eta\nu_i)^nT_Gv_i.
$$

For $0<\eta\nu_i\le1$, the triangle inequality and the geometric series yield

$$
\sum_{n<N}\|\widehat a_{n+1}-\widehat a_n\|
\le\sum_{i=1}^{2}|b_i|\|(T_Gv_i)_a\|
\frac{1-(1-\eta\nu_i)^N}{\nu_i}
+\eta N\|(f_\infty)_a\|.
$$

This proves a transient movement budget plus a persistent weak-drive
term for the reduced dynamics. A transfer to actual GD needs an additional
bound on model discrepancy. The forecast accuracy and the separate
278k–438k forward enclosure reported above concern the frozen full-tangent
predictor. They do not certify this two-mode reduction: its omitted terms
require an additional discrepancy bound.

## Technical detail E: the full-GD bound and its closure

Here we fill in the two steps omitted from the main explanation: construct
a loss floor throughout a parameter neighborhood, then prove that GD
cannot leave the neighborhood too quickly.

### Bound the hard output throughout the neighborhood

Consider $\|\theta-\theta_s\|\le R$. An analytic polynomial approximation
of degree 8 to each tanh feature has uniform error at most $\delta_j(R)$
over this ball. Because the empirical unit mode $q_9$ is orthogonal to
such polynomials,

$$
|\langle q_9,\tanh(a_jx+b_j)\rangle_m|\le\delta_j(R).
$$

Write each readout as its starting value plus its increment. The output
bias has no degree-9 component, and Cauchy–Schwarz bounds the increments:

$$
|\langle q_9,f_\theta\rangle_m|
\le H_R:=\sum_j|c_{j,s}|\delta_j(R)+R\|\delta(R)\|_2.
$$

For target coefficient $Y_9=\langle q_9,y\rangle_m$, this supplies

$$
L(\theta)\ge\ell_R:=\tfrac12[|Y_9|-H_R]_+^2.
$$

The [analytic construction](d34_coarse_balance_stagnation.md#a-stronger-bound-from-the-loss-the-network-cannot-yet-remove)
bounds $\delta_j(R)$ using tanh's analyticity and a polynomial-tail bound.
The important point for the theorem is uniformity over the ball: future
measured parameters do not enter the floor.

### Close the neighborhood condition using descent

Let $U_R$ bound the Hessian operator norm throughout the ball, and define

$$
q_R=1-\eta U_R/2>0,\qquad
\Delta_R=L(\theta_s)-\ell_R,\qquad
r_R=\frac{R-\eta\|g_s\|}{1+\eta U_R}>0.
$$

Suppose the previous cumulative path is below $r_R$. The gradient bound
$\|g_n\|\le\|g_s\|+U_Rr_R$ puts the whole next step inside radius $R$,
since $r_R+\eta(\|g_s\|+U_Rr_R)=R$. The descent lemma therefore applies.
Summing it for $k$ steps and using the floor gives

$$
\eta q_R\sum_{n=s}^{s+k-1}\|g_n\|^2\le\Delta_R.
$$

Cauchy–Schwarz now bounds the path by

$$
\eta\sum_{n=s}^{s+k-1}\|g_n\|
\le\sqrt{\frac{\eta k\Delta_R}{q_R}}=:Q_k.
$$

If $Q_N<r_R$, every prefix has $Q_k\le Q_N<r_R$. This closes the
induction: the condition used for the next step is reproduced by the
bound. Requiring also $Q_N<\mathcal D_{p,\Gamma}(a_s)$ excludes the
specified scale population through those $N$ updates.

For $\Delta_R>0$, a sufficient horizon is therefore

$$
N<\frac{q_R}{\eta\Delta_R}
\min\{r_R^2,\mathcal D_{p,\Gamma}(a_s)^2\}.
$$

For $\Delta_R=0$, use the $Q_N$ criterion directly. This proof concerns
full simultaneous GD: no frozen gain, zero-tracking assumption, or
particlewise contraction is required.

The [numerical evaluation](d34_coarse_balance_stagnation.md#what-can-be-bounded-without-the-future-trajectory)
uses five seeds at each of 100k and 600k. Available loss is
$7.74\times10^{-7}$–$1.52\times10^{-6}$, and the computed horizons range
from 10.237 to 17.094 million additional updates. The common ten-million
statement leaves margin. Along those bounded intervals, every absolute
slope is below 0.388 and relative **empirical training** MSE exceeds
0.749998. This is not a bound on the independent evaluation-grid error.
The constants use ordinary FP64, not interval certification. The result
starts at these checkpoints; deriving entry from initialization is a
separate task.

## Evidence and conventions

The main GD setting is width 177, 2048 training midpoints, learning rate
0.002, random nonzero readouts, and FP64. Evaluation uses 8192 separate
midpoints. This measures deterministic approximation error rather than
statistical generalization. Slope-population thresholds specify the
dynamical outcome of interest; constructions alone do not make them
necessary conditions for every accurate representation.

The norm table in Section 3 reads existing fields from
[the seed-0 trace](../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/curated/primary_0/trace.npz)
and its [manifest](../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/curated/primary_0/manifest.json).
The exact state is immediately before update 600,000, after 599,999
completed updates. The [runner](../experiments/expD34_readout_race/adam_run.py)
records the norms of direct fine, balanced coarse, effective, and tracking
slope gradients separately.

Signed movement comes from
[the 20k–600k table](../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/post20k_changes.csv),
using `continue600k`, seed 0, and readout-rate ratio 1. The stored direct
fine and actual coarse contributions are respectively $-0.0892611$ and
$+0.242930$ for degree 3, and $-1.07117\times10^{-4}$ and
$+4.11876\times10^{-5}$ for degree 9. The actual coarse contribution
contains balanced plus tracking terms. Construction resolution `n=128`
in this table corresponds to physical width 177.

The maximum 0.469% tracking statistic divides accumulated tracking norms
by accumulated effective-plus-tracking norms across 65 GD cases. The
earlier degree-9 median 0.00111% instead divides integrated tracking norm
by full-gradient norm. Neither is a signed-motion fraction. The generated
mode ratios concern signed mean-scale velocities at correlated checkpoints,
not independent trials or ratios of force norms.

This tutorial reorganizes existing evidence and derivations. The two-mode
interpretation was selected after inspecting starting-state spectra; the
later independent-seed forecasts were issued before their future updates.
The latter validate a fixed prediction rule and do not retroactively make
the earlier model selection independent.

The source notes retain the complete derivations and evidence:

- [Gain, tracking, and population theorem](d34_barrier_theorem_walkthrough.md).
- [Coupled effective-gradient dynamics](d34_transport_scale_barrier.md).
- [Generated-error mechanism, rates, and loss-floor bound](d34_coarse_balance_stagnation.md).
- [Signed-motion evidence](../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/README.md).
- [Expanded GD tracking and Adam comparison](../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/README.md).
- [Long continuations and independent interventions](../results/checkpoint_D_optimizers/expD34_readout_race/force_plateaus/README.md).
