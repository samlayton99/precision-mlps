# Tightening the persistence argument: signed population balances

The evidence supports slow population scale acquisition after the coarse
tracking transient. Our target-gap proof establishes one mechanism for
persistence, but its energy bound gives a short useful duration at the saved
checkpoints. **The next improvement should retain the signed evolution of
coupled population moments before bounding their size.** A small energy
reservoir alone does not preserve enough of that structure.

The literature suggests how to do this. We derive an exact tanh balance
identity and a population growth comparison that permits positive growth.
The new comparison uses the evolving population's concentration, polynomial
output, and coupling to the target; it does not freeze the training dynamics.
Matched replays support useful conditional bounds over 71k–100k additional
GD updates for six target families. These are sampled checks of structural
conditions along trajectories, not guarantees from a checkpoint alone.
The existing accumulated-feedback theorem remains our main paper claim.

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
| $M=\sum_j(a_j^2+b_j^2+c_j^2)$ | Total hidden squared norm; the output bias is excluded. |
| $h=2/N_{\rm ref}$, $\lambda_{\rm RMS}=h\sqrt{A/W}$ | Grid spacing and normalized population slope scale. |

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
not a completed replacement for the main theorem. Section 8 supplies a
structural growth comparison and derives alignment control from it; Section
9 tests its usefulness along matched replays.

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

## 8. Growth lemmas: retain dissipation and close a population comparison

The useful strengthening allows growth. Two operations make the proof more
informative: retain the negative generated-error term before bounding its
defect, and bound the evolving population through scale-invariant polynomial
observables. This section uses exact tanh throughout. The polynomials supply
global inequalities; they are not a substitute training ODE.

Write $r_j^2=a_j^2+b_j^2+c_j^2$, $S_k=\sum_jr_j^k$, and $M=S_2$.
Define

$$
A_3=\frac{\sqrt6}{8},\quad
A_5=\frac2{15}\frac{2^{5/2}5^{5/2}}{6^3},\quad
A_7=\frac{17}{315}\frac{2^{7/2}7^{7/2}}{8^4},\qquad
C_k=(k+1)A_k.
\tag{12}
$$

**Lemma 3 — a positive allowance for generated-error correction.** Let
$\sigma=\sigma_{\min}(J_C)>0$, $K_0=2A_3S_4$,
$E_0=2A_5S_6$, and $J_0=C_3\sqrt{S_6}$. Split
$F=\Pi J_H^*f_H-\Pi J_H^*g$. The contribution of the first term to
$\dot\Delta$ is at most

$$
\mathcal D_{\rm gen}
\le\frac18\left(E_0+\frac{K_0J_0}{\sigma}\right)^2.
\tag{13}
$$

**Proof.** For $k(u)=\tanh u-u\operatorname{sech}^2u$,
$k'(u)=2u\tanh u\operatorname{sech}^2u$, so $|k(u)|\le2|u|^3/3$.
For $E(u)=3\tanh u-u\operatorname{sech}^2u-2u$,
$E'(u)=-2\tanh u\,k(u)$, giving $|E(u)|\le4|u|^5/15$.
Optimizing $|c|(a^2+b^2)^{k/2}$ at fixed $r$ gives
$\|k_\theta\|\le K_0$ and $\|E_\theta\|\le E_0$.
The Jacobian estimate below gives $\|J_H\|\le J_0$.
The generated part of $\ell$ has norm at most $J_0\|f_H\|/\sigma$.
Writing $s=\|f_H\|$, (4) therefore bounds its total contribution by

$$
-2s^2+\left(E_0+K_0J_0/\sigma\right)s
\le\left(E_0+K_0J_0/\sigma\right)^2/8.
$$

This completes the square rather than requiring the expression to be
negative. Compensation is included. $\square$

For well-distributed populations with $M=O(1)$, $S_4=O(W^{-1})$ and
$S_6=O(W^{-2})$. If the coarse singular value has an order-one lower bound,
the allowance in (13) is $O(W^{-4})$. Target loading can be larger and
positive. This explains why failure of universal contraction does not defeat
the growth argument. A width-independent conditioning margin is an additional
condition for this scaling statement, not a consequence of the identity.

### Exact remainders and scale-invariant population structure

Let
$\Psi_3=-\sum_jc_ju_j^3/3$ and
$\Psi_5=2\sum_jc_ju_j^5/15$.
Set $J_k=D_\theta(P_H\Psi_k)$ and $G_k=J_k^*g$. Global remainder bounds give

$$
\begin{aligned}
\|f_H-P_H(\Psi_3+\Psi_5)\|&\le A_7S_8,\\
\|J_H-J_3-J_5\|&\le C_7\sqrt{S_{14}},\\
\|J_H^*g-G_3-G_5\|&\le C_7\|g\|\sqrt{S_{14}}.
\end{aligned}
\tag{14}
$$

**Proof of the constants.** The scalar tanh remainders after degrees one,
three, and five are bounded by $|u|^3/3$, $2|u|^5/15$, and
$17|u|^7/315$. Their derivative remainders are bounded by the corresponding
$k$ times coefficient times $|u|^{k-1}$. For the final bound,
$\sup|\tanh^{(7)}|\le272$ suffices. In $z=\tanh^2u$, that derivative is
$-272+3968z-12096z^2+13440z^3-5040z^4$; its Bernstein coefficients on
$[0,1/2]$ and $[1/2,1]$ are respectively
$(-272,224,216,124,53)$ and $(53,-18,-68,8,0)$, proving the bound.
The lower remainders follow by integrating
$|\tanh u-u|\le|u|^3/3$ and
$|u^2-\tanh^2u|\le2|u|^4/3$.

For a scalar coefficient $t_k$, a neuron output remainder is at most
$t_k2^{k/2}|c|(a^2+b^2)^{k/2}$. Its maximum at fixed $r$ is $A_kr^{k+1}$.
The sum of squared derivative-column remainders is at most
$t_k^22^k[k^2c^2(a^2+b^2)^{k-1}+(a^2+b^2)^k]$.
Maximizing at $(a^2+b^2)/r^2=k/(k+1)$ gives $C_k^2r^{2k}$.
Summing columns bounds the Hilbert–Schmidt norm; projection cannot increase
it. This proves (14) and the $J_0$ bound in Lemma 3. $\square$

For a directly testable signed imbalance bound, put
$v=\nabla\Delta=(a,b,-c,0)$. Combining Lemma 3 and (14) gives

$$
\dot\Delta_F\le
\frac18(E_0+K_0J_0/\sigma)^2+
\langle\Pi v,G_3+G_5\rangle+
\|\Pi v\|C_7\|g\|\sqrt{S_{14}}.
\tag{14a}
$$

The polynomial target pairing retains its sign. Add $-v\cdot R$ for full
flow and the exact quadratic correction (10) for GD. This distinguishes a
negative generated correction from positive target-driven growth without
requiring either effect to dominate universally.

For $M>0$, define the dimensionless shapes

$$
\begin{aligned}
\chi_k&=W^{k/2-1}S_k/M^{k/2},\\
q_k&=W^{(k-1)/2}\|P_H\Psi_k\|/M^{(k+1)/2},\\
j_k&=W^{(k-1)/2}\|J_k\|_{\rm HS}/M^{k/2},\qquad
g_k=W^{(k-1)/2}\|G_k\|/M^{k/2},\quad k=3,5.
\end{aligned}
\tag{15}
$$

These quantities do not change under a common rescaling of all hidden
parameters. Bounding them therefore does not assume that $M$, the slopes,
or the future effective force stay small. They describe population shape,
low-order output coherence, and coupling to specified target moments. In
particular, $G_3=0$ when $g\perp\mathcal P_3$, and also $G_5=0$ when
$g\perp\mathcal P_5$. For other targets those terms remain explicit.

### A first-exit theorem for the evolving population

Choose an allowed total hidden squared norm $M_*>M(0)$. At each time define

$$
\begin{aligned}
Q_*&=q_3M_*^2/W+q_5M_*^3/W^2+A_7\chi_8M_*^4/W^3,\\
J_*&=j_3M_*^{3/2}/W+j_5M_*^{5/2}/W^2+C_7\sqrt{\chi_{14}}M_*^{7/2}/W^3,\\
G_*&=g_3M_*^{3/2}/W+g_5M_*^{5/2}/W^2
       +C_7\|g\|\sqrt{\chi_{14}}M_*^{7/2}/W^3,\\
V_*&=J_*Q_*+G_*.
\end{aligned}
\tag{16}
$$

The shapes in (16) evolve; only the allowed radius is fixed. No Jacobian is
frozen. Products are retained inside the time integral, avoiding separate
maximum-over-time bounds on concentration and coherence.

**Theorem 4 — slow population growth from accumulated structural loading.**
For effective flow, set $r=0$; for ordinary gradient flow set $r=\|R\|$.
Suppose an upper budget $B_T$ satisfies

$$
\int_0^T[V_*(t)+r(t)]\,dt\le B_T
<\sqrt{M_*}-\sqrt{M(0)}.
\tag{17}
$$

Then $M(t)<M_*$ throughout $[0,T]$, total hidden parameter travel is at most
$B_T$, and $\lambda_{\rm RMS}(t)<h\sqrt{M_*/W}$.
The same result holds for ordinary GD with the integral replaced by
$\sum_{n<N}\eta_n(V_{*,n}+\|R_n\|)$, at every iterate through $N$.

**Proof.** While $M\le M_*$, (14)–(16) and $\|\Pi\|\le1$ give
$\|F\|\le\|J_H\|\|f_H\|+\|J_H^*g\|\le V_*$.
Consequently $D^+\sqrt M\le V_*+r$. At a first exit, integration contradicts
(17). The same bound then controls travel on the entire interval. For GD,
the exact triangle inequality
$\|p_{n+1}\|\le\|p_n\|+\eta_n(V_{*,n}+\|R_n\|)$ for the hidden blocks
gives the conclusion by induction. There is no continuous-trajectory
approximation or enlarged step tube in this discrete argument. $\square$

If an initial fraction $p_0$ has $h|a_j(0)|\ge\lambda_0<\lambda_*$, the
travel conclusion also implies

$$
p_{\rm ever}(T)\le p_0+
\frac{h^2B_T^2}{W(\lambda_*-\lambda_0)^2}.
\tag{18}
$$

Indeed the squared accumulated slope travels sum to at most $B_T^2$ by
Minkowski; count the labels requiring travel at least
$(\lambda_*-\lambda_0)/h$. This is an aggregate counting argument, not an
individual-neuron premise. A bound above one may be clipped at one.

The theorem's premise is an integral of explicit dimensionless polynomial
shapes, moment concentration, and tracking. Its conclusion is a bound on
physical size and force. Establishing useful structural budgets remains a
substantive empirical or theoretical task; inserting the future true force
in place of $V_*$ would not establish this mechanism.

### Coarse fitting preserves the product, which preserves alignment

**Lemma 5 — product and alignment preservation.** Assume Theorem 4's
conditions and a symmetric input measure with $v_x=\|x\|^2>0$. Let

$$
J_{N,*}=C_3\sqrt{\chi_6}M_*^{3/2}/W,\qquad
B_m(T)=\frac1{\sqrt{v_x}}\int_0^T
\left[\sqrt{1+2M_*}\,r+J_{N,*}(V_*+r)\right]dt.
\tag{19}
$$

Then $|m(t)-m(0)|\le B_m(T)$ and

$$
|\rho_{ac}(t)|\ge\frac{2[|m(0)|-B_m(T)]_+}{M_*}
\tag{20}
$$

where $A,C>0$. Equivalently $m^2\ge\alpha_T^2AC$ with the right side of
(20) as $\alpha_T$. A positive lower bound follows when $B_m<|m(0)|$;
small-coarse-component targets need not meet that additional condition.

**Proof.** With $\mathcal N_\theta$ from (9), the linear output coefficient
is $\sqrt{v_x}m+\langle\mathcal N_\theta,x/\sqrt{v_x}\rangle$.
Its derivative under effective flow is zero because $J_CF=0$; under full
flow its absolute derivative is at most $\|J_C\|\|R\|$.
The same global cubic remainder proof gives
$\|D\mathcal N\|\le C_3\sqrt{S_6}\le J_{N,*}$, and
$\|J_C\|\le\sqrt{1+2M_*}$. Differentiating the coefficient and integrating
proves (19). Finally $\sqrt{AC}\le(A+C)/2\le M_*/2$ proves (20).
$\square$

For GD use the corresponding left-point sum in (19) and add
$\tfrac12\sum_n\eta_n^2(V_{*,n}+\|R_n\|)^2$.
This follows from the exact identity
$m_{n+1}-m_n=-\eta_n(c_n\cdot G_{a,n}+a_n\cdot G_{c,n})+
\eta_n^2G_{a,n}\cdot G_{c,n}$ and
$|G_a\cdot G_c|\le\|G\|^2/2$.

Thus alignment can be a consequence of a growth theorem rather than an
independent assumed invariant. The primary population conclusion still
applies when the alignment lower bound is zero. In particular, a loss of
useful alignment control does not by itself refute slow scale acquisition.

## 9. Matched replays test a useful duration

The degree-five example makes the gain concrete. Its hidden squared norm
increases by only 0.003% during 100k additional GD updates. The structural
comparison allows a 2% increase and consumes about 13% of the corresponding
radius budget. This closes the implication of Theorem 4 over the measured
interval without assuming the actual force stays at its starting value.
For mixed sine, the analogous allowance is 10%. For the bump and kink,
a factor-two allowance closes over the whole interval. Gaussian and step
loading exhaust this particular sufficient comparison after approximately
84k and 71k updates, respectively; their observed population growth is
still only about 9.3% by 100k updates.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/balance_comparison_verified_20260924/population_comparison.png" alt="Accumulated structural budgets, RMS slope bounds, population growth and alignment along six GD continuations" style="max-width:100%;">
  <figcaption>Width 705, seed 30, starting after 20k GD updates; another 100k updates at learning rate 0.002. A compares the structural integral to the available radius increase; the theorem applies while the curve stays below one. B shows actual normalized slope RMS and the corresponding upper bound, drawn only over its supported duration. C shows that the result allows positive population growth. D compares alignment to the lower bound derived from coarse fitting and limited travel. Bounds use sampled structural quadrature, not interval arithmetic.</figcaption>
</figure>

Over their supported durations, the normalized slope-RMS bounds lie between
$3.7\times10^{-4}$ and $5.5\times10^{-4}$, far below $\lambda=0.25$.
These numbers describe this width and initialization. The dependence on
width is explicit in the theorem, but uniform structural allowances across
widths still require evidence. The radius allowance was selected from a
reported finite grid to assess usefulness; it was not fixed as a prediction
before observing the trajectories.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/balance_comparison_verified_20260924/signed_growth.png" alt="Signed generated, target, higher-order, compensation and tracking contributions to imbalance change, with analytic upper allowances" style="max-width:100%;">
  <figcaption>The same six 100k-update GD continuations. Left: integrated terms in the exact imbalance identity. Right: actual imbalance change and the allowance from (14a), including tracking and the exact discrete correction. Mixed sine has a negative upper allowance; the localized targets permit positive growth. A small allowance does not imply universal contraction.</figcaption>
</figure>

The signed bound explains more than a small measured derivative: it
identifies which generated correction can be adverse, bounds that defect,
and leaves target loading explicit. The population comparison then limits
how much size can build up under the accumulated structural loading. Its
remaining assumption is persistence of dimensionless population structure,
which is less restrictive than assuming small future force or small future
slopes, but is still a condition to validate.

## Reproducibility and status

The identities, global remainder inequalities, and conditional comparisons
are proved here. Their numerical application is ordinary floating-point
evidence, not an interval certificate or an initial-data-only guarantee.
The replay study has six targets, effective-flow RK4 steps 0.02 and 0.01,
and GD steps 0.002 and 0.001, each through physical time 200. Scalar
diagnostics are recorded every unit of flow time; exact GD moment increments
are accumulated at every update. Only starting and ending parameter vectors
are retained.

Seven focused tests passed, including independent automatic derivatives,
global remainder bounds at three parameter scales, exact finite-step moment
identities, a zero-alignment case, and a stationary exact fit. Across the
replays, accumulated balance discrepancies are below $7\times10^{-14}$.
Halving the GD step changes population size and normalized slope RMS by
less than $4.4\times10^{-7}$ relative; RK4 differences are near floating-point
roundoff. Halving the structural sampling density changes the integrated
budget by less than $2.7\times10^{-5}$ relative in the six principal runs.
This tests numerical resolution but does not enclose every unsampled state.

The replay used 495 GPU-seconds on Modal with an 8 GiB memory cap and about
4.23 GiB measured peak child memory. Budget analysis runs on Modal CPU with
a 4 GiB cap. No numerical analysis or archive-array loading ran locally.
The prior endpoint audit remains a separate correctness check; the new
replays supply the missing time-resolved balances.

Sources, commands, environment, and hashes are recorded in
[execution.json](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/population_balance_verified_20260924/execution.json).
Numerical outputs are
[balance.csv](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/population_balance_verified_20260924/balance.csv)
and [summary.json](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/population_balance_verified_20260924/summary.json).
The existing [target-gap theorem](d34_target_gap_persistence.md) and
[accumulated-feedback theorem](d34_population_output_persistence.md) retain
their separate assumptions and empirical coverage.
