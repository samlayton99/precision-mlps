# Tightening the persistence argument: signed population balances

The evidence supports slow population scale acquisition after the coarse
tracking transient. Our target-gap proof establishes one mechanism for
persistence, but its energy bound gives a short useful duration at the saved
checkpoints. **The next improvement should retain the signed evolution of
coupled population moments before bounding their size.** A small energy
reservoir alone does not preserve enough of that structure.

The literature suggests how to do this. We derive an exact tanh balance
identity, prove a conditional RMS-slope bound, and check the proposed
observables at six saved trajectories' starting and ending states. This gives
a more specific proof route. It does not establish the new assumptions
throughout those trajectories or extend the checkpoint-only guarantee to 100k
updates. The existing accumulated-feedback theorem remains our main claim.

**Notation.** Parameters and time use the ordinary GD metric; population sums
below are unnormalized sums over the $W$ neurons.

| Symbol | Meaning |
|---|---|
| $f_\theta=d+\sum_j c_j\tanh(a_jx+b_j)$ | Network; $a_j$ is the physical slope, also denoted $\gamma_j$. |
| $P_C,P_H$ | Orthogonal projections onto affine functions and their complement. |
| $f_H,g,e_H=f_H-g$ | Non-affine network output, target, and error. |
| $F,R$ | Effective fine gradient, including coarse compensation, and tracking gradient. |
| $\Delta=\tfrac12\sum_j(a_j^2+b_j^2-c_j^2)$ | Population geometry–readout imbalance, not slope energy. |
| $A=\sum_j a_j^2$, $C=\sum_j c_j^2$ | Slope and readout squared norms. |
| $m=\sum_j a_jc_j$, $\rho_{ac}=m/\sqrt{AC}$ | Linear-output coefficient and collective slope–readout alignment. |

## 1. The example that identifies what the energy bound loses

Replace tanh temporarily by its linear term. Then

$$
f_\theta(x)=d+\sum_j c_jb_j+x\sum_j c_ja_j.
$$

After the linear target component has been fitted, $m=\sum_jc_ja_j$ is
approximately fixed. Gradient flow also preserves $\Delta$ exactly in this
linear model. These facts constrain simultaneous growth of slopes and
readouts: if their norms grow together while $m$ stays fixed, their
collective alignment must decrease. No individual-neuron invariant is needed.

Exact tanh breaks the balance invariant. The useful question is therefore:
**which parts of the nonlinear error change this balance, with which sign,
and how much can their accumulated effect grow?** Section 3 answers the first
two questions exactly. Sections 4–6 identify the remaining duration question.

The current energy proof replaces signed target correlations by upper bounds
over an entire parameter neighborhood. For the width-705 degree-five
checkpoint, its refined GD bound permits about 793 updates, while a saved
continuation shows small population change through 100k additional updates.
That number is a limitation of this sufficient bound, not an observed
800-update transition. The neighborhood argument can spend loss on directions
that need not be consistent with the coupled balance and coarse fit.

## 2. What the literature contributes

**Missing target modes can suppress the next learning stage.** In Appendix
D.3 of [Berthier, Montanari, and Zhou, *Learning time-scales in two-layers
neural networks*](https://arxiv.org/html/2303.00055v2), a missing Hermite
coefficient yields a nonpositive derivative of a population readout second
moment in their simplified stage ODE: the derivative is minus a squared
generated-mode moment. This is a close precedent for our generated-error
mechanism. Their Gaussian, high-dimensional model and stage approximation
differ from ours; the general stage-failure interpretation is heuristic. We
borrow the signed moment argument, not a long-time theorem for our network.

**Layer balance is a natural observable.** [Du, Hu, and Lee, *Algorithmic
Regularization in Learning Deep Homogeneous Models*
](https://arxiv.org/abs/1806.00900) establish squared-norm balance invariants
for gradient flow in homogeneous networks. Tanh is not homogeneous, so that
invariant does not transfer. Instead, its exact failure is a useful quantity
to calculate. This motivates $\Delta$ and the identity below.

**The first nonzero coupling matters more than a generic norm.** [Ben Arous,
Gheissari, and Jagannath, *Online stochastic gradient descent on non-convex
losses from high-dimensional inference*
](https://jmlr.csail.mit.edu/papers/v22/20-1288.html) classify difficulty using
the first nonzero derivative of the population loss near weak alignment. For
us, the useful question is which signed aggregate drift survives the target
orthogonality conditions. Their SGD sample-complexity rates do not transfer
to our deterministic GD or to width scaling.

**Slow gradient-flow theory needs structure beyond a small energy budget.**
[Otto and Reznikoff, *Slow Motion of Gradient Flows*
](https://webdoc.sub.gwdg.de/ebook/serien/e/sfb611/275.pdf) combine dissipation
toward a slow set with weak energy variation along it. Their theorem supplies
long-duration control under these structural hypotheses. Our generic
energy–movement inequality uses less information. Applying the stronger
theorem would require establishing transverse relaxation in appropriate
aggregate variables. The evidence does not justify assuming a slope attractor
or uniform contraction of every nonlinear residual mode.

The common lesson is to identify a collective balance or dissipative
structure and bound its defects. A Fourier sensitivity estimate alone does
not establish that the evolving population preserves that structure.

## 3. An exact identity, including compensation and tracking

Take a probability measure on $[-1,1]$, either the empirical training measure
or a specified population measure. Let

$$
L=\tfrac12\|f_\theta-y\|^2,\qquad
e_C=P_C(f_\theta-y),\qquad g=P_Hy.
$$

Let $J_C=D_\theta(P_Cf_\theta)$ and $J_H=D_\theta f_H$. When
$K=J_CJ_C^*$ is invertible, our existing decomposition is

$$
\begin{aligned}
\ell&=K^{-1}J_CJ_H^*e_H,& z&=e_C+\ell,\\
\Pi&=I-J_C^*K^{-1}J_C,&
F&=\Pi J_H^*e_H,&R&=J_C^*z.
\end{aligned}
\tag{1}
$$

Thus $\dot\theta=-F-R$ for full gradient flow. The compensating coarse
gradient $-J_C^*\ell$ is inside $F$. A small tracking gradient does not remove
this compensation.

Write $u_j=a_jx+b_j$ and $\phi(u)=\tanh u$. Define two exact functions:

$$
\begin{aligned}
k_\theta(x)&=\sum_jc_j[\phi(u_j)-u_j\phi'(u_j)],\\
E_\theta(x)&=\sum_jc_j[3\phi(u_j)-u_j\phi'(u_j)-2u_j].
\end{aligned}
\tag{2}
$$

The first measures the failure of linear homogeneity. The second isolates
the higher-order correction to its leading cubic term. Algebra gives

$$
P_Hk_\theta=-2f_H+P_HE_\theta.
\tag{3}
$$

**Proposition 1 — signed population balance.** Along the effective flow
$\dot\theta=-F$,

$$
\boxed{
\dot\Delta_F=
\underbrace{-2\|f_H\|^2}_{\text{generated-error correction}}
+\underbrace{2\langle g,f_H\rangle}_{\text{target coupling}}
+\underbrace{\langle e_H,P_HE_\theta\rangle}_{\text{higher-order correction}}
-\underbrace{\langle\ell,P_Ck_\theta\rangle}_{\text{coarse compensation}}.
}
\tag{4}
$$

Tracking adds

$$
\dot\Delta_R=\langle z,P_Ck_\theta\rangle.
\tag{5}
$$

For full gradient flow the last terms combine to
$\langle e_C,P_Ck_\theta\rangle$. This full-flow identity remains valid even
when the coarse Gram matrix is singular; only the decomposition needs its
inverse.

**Proof.** Let $v=\nabla_\theta\Delta=(a,b,-c,0)$. Direct differentiation
gives $Df_\theta[v]=-k_\theta$. Consequently,

$$
-\langle v,F\rangle
=\langle e_H,P_Hk_\theta\rangle-\langle\ell,P_Ck_\theta\rangle.
$$

Substitute (3) and $e_H=f_H-g$. Similarly,
$-\langle v,R\rangle=\langle z,P_Ck_\theta\rangle$. $\square$

For small arguments, $E_\theta$ begins with
$-\tfrac4{15}\sum_jc_ju_j^5$, whereas $f_H$ generally begins with the
non-affine part of $-\tfrac13\sum_jc_ju_j^3$. This expansion explains (4);
the identity itself uses exact tanh and evolving parameters.

The negative term retains a sign that an absolute energy or Jacobian bound
loses. It does **not** prove that the whole generated contribution, the net
imbalance drift, or the slopes always decrease. Higher terms and compensation
matter for the first claim; target coupling matters for the second;
$\Delta$ is not slope energy for the third.

**Target-family prediction.** If $g$ is orthogonal to polynomials through
degree five, the two explicit target pairings in (4) lose their cubic and
quintic terms; their leading possible contribution has degree seven in hidden parameters.
For a gap only through degree three, quintic loading survives. Ordinary sine
need not satisfy either gap. Global remainder bounds can quantify these
statements without a per-neuron smallness assumption, but the requisite
population moments and compensation still need dynamical control.

## 4. What the saved checkpoints actually support

We post-processed existing 20k-update checkpoints and the endpoints of their
100k-additional-update continuations. The six targets are a degree-five
orthogonal-polynomial mixture, mixed sine, a shifted Gaussian, a compact bump,
a smooth step, and an absolute-value kink. All have width $W=705$ and seed 30.
The principal comparison uses ordinary GD with $\eta=0.002$, hence 200 units
of additional flow time. We also checked the saved $\eta=0.001$ GD and two
effective-flow integration endpoints.

These are post-hoc optimization diagnostics on the saved training grid. They
are neither new training runs nor independent validation of a selected
theorem premise. The six targets provide an initial check of this observable;
the broader campaign's 23-target coverage should not be attributed to it.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/population_balance_verified_20260924/population_balance.png" alt="Signed generated-error and target contributions to imbalance at six 20k checkpoints, and collective slope–readout alignment at 20k and 120k GD updates" style="max-width: 100%;">
  <figcaption>Left: exact effective-flow contributions to the derivative of the geometry–readout imbalance at the 20k checkpoint. Each contribution includes its own coarse compensation; together they equal the net effective fine contribution. The horizontal axis is symmetric-logarithmic. Right: collective alignment at the two saved ordinary-GD endpoints. The figure does not establish an interval-wide bound between those endpoints.</figcaption>
</figure>

Three findings constrain the theory:

1. **Generated correction opposes imbalance growth at all six starting
   checkpoints, but target coupling is larger in magnitude.** For degree
   five, the exact generated contribution is $-1.54\times10^{-7}$ and the
   target contribution is $+6.81\times10^{-7}$ per unit flow time. Their sum
   is positive. Universal dominance of generated-error correction is an
   unsuitable premise even at this checkpoint.
2. **The aggregate alignment changes modestly at the endpoints.** Its absolute
   value ranges from 0.215 to 0.587 initially and from 0.203 to 0.545 finally.
   The linear coefficient $m$ changes by less than 0.7% in every case. This
   supports investigating collective alignment; it does not prove persistence
   between the endpoints.
3. **Slow acquisition does not require constant or decreasing imbalance.**
   For the smooth step, $\Delta$ rises from 1.080 to 1.384 while slope RMS
   rises from 0.0593 to 0.0642. For degree five, $\Delta$ rises by only
   $1.05\times10^{-4}$ and slope RMS changes from 0.054681 to 0.054684.
   Mixed sine decreases slope RMS; the Gaussian and compact bump increase
   it. The absolute-value kink starts with negative imbalance drift but
   ends with a larger imbalance. A frozen initial sign is not a persistence
   argument.

The displayed split uses $F_{\mathrm{gen}}=\Pi J_H^*f_H$ and
$F_{\mathrm{tar}}=-\Pi J_H^*g$, whose sum is the same $F$ as before.
This is a diagnostic split inside the effective fine gradient, not a change
to the tracking/effective-fine terminology.

## 5. A precise aggregate implication for slope acquisition

The example in Section 1 can be stated without linearizing tanh. Define the
signed full-flow integrand

$$
\mathcal I(\theta)=
-2\|f_H\|^2+2\langle g,f_H\rangle
+\langle e_H,P_HE_\theta\rangle
-\langle\ell,P_Ck_\theta\rangle
+\langle z,P_Ck_\theta\rangle.
\tag{6}
$$

For effective flow, omit the final tracking term. The full-flow form may
equivalently use the coarse residual pairing and needs no inverse.

**Proposition 2 — balance and alignment limit population slopes.** Consider
either flow from a post-transient checkpoint. Suppose that, for every
$s\in[0,T]$,

$$
\int_0^s\mathcal I(\theta(t))\,dt\le B(s),\qquad
|m(s)|\le M,\qquad |\rho_{ac}(s)|\ge\alpha>0.
\tag{7}
$$

Let $D(s)=\Delta(0)+B(s)$. Then

$$
\boxed{
\operatorname{RMS}(a(s))^2
\le\frac{D(s)+\sqrt{D(s)^2+M^2/\alpha^2}}{W}.
}
\tag{8}
$$

At a time with $A=0$, the slope conclusion holds trivially; where $A>0$ the
alignment premise requires $C>0$. There is no restriction on the largest
neuron or individual slope–readout ratio.

**Proof.** Proposition 1 gives $\Delta(s)\le D(s)$. Since
$A-C=2\Delta-\sum_jb_j^2$, we have $C\ge A-2D$. Also
$AC=m^2/\rho_{ac}^2\le M^2/\alpha^2$. Thus
$A(A-2D)\le M^2/\alpha^2$; solving this quadratic gives (8). $\square$

With a bounded coarse linear coefficient, substantial RMS-slope acquisition
requires a large increase of geometry relative to readouts, or a substantial
loss of their collective alignment. Small coarse tracking alone excludes
neither route. The alignment condition allows neurons to move and exchange
influence; it does not track any particular neuron.

For scale, illustrative allowances $D\le2$, $M\le2$, and $\alpha\ge0.1$
would imply slope RMS below 0.18 at width 705. All twelve principal saved
endpoints satisfy these loose allowances. They were chosen after inspecting
the endpoints and have not been established throughout the interval. If the
allowances were uniform in width, (8) would give $O(W^{-1/2})$ RMS slopes;
this one-width audit does not establish that premise.

**Where the real theorem work remains.** Proposition 2 is a proved
implication, but (7)'s integral alone is a balance-budget assumption, not an
explanation of persistence. Its value depends on bounding (6) using signed
generated correction, target-family structure, and aggregate moments, then
establishing an alignment allowance. Replacing that work by a measured small
$B$ would recreate the weakness we are trying to remove. Proposition 2 is
not a completed replacement for the main theorem.

Fitting the coarse output does not exactly fix $m$ for nonlinear features.
For a symmetric input measure with $v_x=\|x\|^2>0$, write

$$
f_\theta=d+\sum_jc_jb_j+mx+\mathcal N_\theta(x),\qquad
\mathcal N_\theta=\sum_jc_j[\tanh u_j-u_j].
$$

Then

$$
|m|\le
\frac{|\langle y,x/\sqrt{v_x}\rangle|+\|e_C\|+
\|\mathcal N_\theta\|}{\sqrt{v_x}}.
\tag{9}
$$

Population-moment bounds control $\mathcal N_\theta$ in the broad-feature
regime. Equation (9) identifies the coupling instead of silently replacing
nonlinear coarse fitting by an exact product constraint. Targets with little
linear component can have very small $\rho_{ac}$; this route is not yet a
universal target-family argument. A slope-RMS bound alone also does not
replace the existing population-to-output-error theorem.

## 6. Ordinary GD has an exact balance correction

This observable avoids a separate trajectory-closeness argument for its GD
transfer. Let $G=\nabla L=(G_a,G_b,G_c,G_d)$ and
$\theta^+=\theta-\eta G$. Since $\Delta$ is quadratic,

$$
\Delta(\theta^+)-\Delta(\theta)
=\eta\mathcal I(\theta)
+\frac{\eta^2}{2}
\left(\|G_a\|^2+\|G_b\|^2-\|G_c\|^2\right).
\tag{10}
$$

This identity is exact for any step size. In (7), replace the integral by
the discrete sum including this correction. Proposition 2 then applies at
every GD iterate where the other premises hold.

For constant $\eta$, if the descent inequality
$L(\theta^+)\le L(\theta)-\eta\|G\|^2/2$ holds on each step, the accumulated
upper bound on the quadratic correction is

$$
\frac{\eta^2}{2}\sum_n\|G_n\|^2
\le\eta[L(\theta_0)-L(\theta_N)].
\tag{11}
$$

Discrete transfer for this moment can therefore use loss dissipation
directly. This does not establish alignment persistence or a useful loading
budget. Adam changes the coordinate weighting and carries optimizer memory,
so the unweighted GF identity is not its dynamical balance law.

## 7. Which assumptions to keep, and what would improve the proof

Keep the post-transient start, accumulated effects of tracking, and
population rather than maximum-neuron control. Keep target orthogonality
relative to the actual input measure, with explicit low-degree leakage for
other targets. Keep generated correction and compensation signed before
bounding adverse terms. These choices have mechanisms and measurable failure
modes.

Do not add universal contraction, exact layer balance for tanh, negligible
compensation, or a frozen sensitivity operator. Do not assume that every
fine mode relaxes rapidly. The evidence also does not establish alignment
preservation merely because its two endpoint values are similar.

The next useful tests have specific consequences:

1. **Recover balance and alignment along saved trajectories.** Measure
   $\Delta,m,\rho_{ac}$ and the signed terms of (4)–(5), including the exact
   GD increment. The stored dense CSV lacks these new observables; the saved
   endpoints alone cannot supply their integrals. If alignment collapses
   during still-slow training, this closure is unnecessarily restrictive and
   should not become the main theorem.
2. **Bound the first surviving loading term for a specified target family.**
   Start with the polynomial-gap family, retaining the negative generated
   term. Determine whether the signed budget is substantially smaller than
   the old ball-based target-correlation allowance. For sine and localized
   targets, measure low-degree residual coupling explicitly rather than
   declaring it zero. Success is a useful bound on an evolving aggregate
   region, not an accurate frozen-force forecast.
3. **Specify feasible controls before perturbing alignment.** One cannot
   change $\rho_{ac}$ while fixing $A,C,m$: these quantities determine it
   algebraically. Holding the coarse coefficient approximately fixed while
   increasing $A$ and $C$ together is one feasible way to lower alignment;
   it also changes the starting scales and generally the fine output. Such
   a test would probe exit from the coupled balance regime, not isolate a
   causal effect of alignment alone. First check these observables on the
   archived large-dilation experiments; design new perturbations only after
   their actual moment changes identify a distinguishable prediction.

There is an alternative if alignment fails: use a coupled low-mode/high-mode
residual system and prove dissipation of generated error with weak transfer
into the unresolved target. Slow-gradient-flow theory supplies a proof
pattern, but its dissipation and coupling assumptions must be tested here.
Neither approach needs a per-neuron persistence theorem.

## Reproducibility and status

The identities and conditional implication above are proved here. Their
interpretation as a long-duration closure remains a hypothesis. The audit is
ordinary floating-point evidence, not an interval certificate. Across six
starts and 24 saved continuation endpoints, the maximum absolute identity
discrepancy was below $4.6\times10^{-17}$. Tests at three parameter scales
also check an independent finite difference of the loss and the exact
finite-step balance identity. Together with existing target-gap checks, all
10 tests passed.

The audit ran on Modal CPU with a 4 GiB memory cap and about 185 MiB measured
peak child memory. It processed a 166 KiB starting capsule and a 403 KiB
endpoint archive, without loading a training trajectory archive or performing
training. The runner also mounts the two small CSVs used by its separate
target-gap study. No numerical analysis ran locally.

Sources, commands, environment, and hashes are recorded in
[execution.json](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/population_balance_verified_20260924/execution.json).
Numerical outputs are
[balance.csv](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/population_balance_verified_20260924/balance.csv)
and [summary.json](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/population_balance_verified_20260924/summary.json).
The existing [target-gap theorem](d34_target_gap_persistence.md) and
[accumulated-feedback theorem](d34_population_output_persistence.md) retain
their separate assumptions and empirical coverage.
