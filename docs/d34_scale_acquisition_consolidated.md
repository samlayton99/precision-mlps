# Why useful feature acquisition is slow: consolidated theory and evidence

**Working extended note, September 26, 2026.** This document is the starting
point for revising the paper's feature-learning theorem. It states the
mechanism, gives a complete basic population proof, explains the refinements
that make the bound useful, and records the empirical support and its limits.
It also preserves the new Adam findings, including results that changed our
initial explanation. The argument is self-contained; the final provenance
section identifies the more detailed derivations and experiment artifacts.

**Notation used consistently below.**

| Symbol | Meaning |
|---|---|
| $a_j=\gamma_j,b_j,c_j,d$ | Physical slope, hidden bias, readout, and output bias. |
| $P_C,P_H$ | Affine-output projection and its orthogonal complement. |
| $F,R$ | Effective fine gradient, including compensation, and coarse tracking gradient. |
| $M,S_k,\chi_k$ | Total hidden energy, its population moments, and dimensionless concentration. |
| $B_n,Q_n(B_n)$ | Scalar hidden-radius upper bound and resulting non-affine output bound. |
| $\lambda_{\rm RMS}=h\|a\|/\sqrt W$ | Normalized population slope RMS; $h$ is the construction's reference spacing. |
| $D_n$ | Inverse Adam denominator, shared by the fine and tracking update components. |

## 1. The claim we want to make

The numerical construction supplies features at scales that support accurate
output approximation. Training must discover useful features as well as fit
their readouts. Our question is whether first-order training accomplishes
that discovery within a specified budget, measured by the raw output error
of the network it actually trains.

The strongest current answer has two parts. **For GD, distributed moderate
parameter energy limits nonlinear sensitivity; a conditional population
comparison then bounds feature growth and gives an output-error floor.** The
conditions concern observable population organization and coarse tracking,
not an assumed small future learning force. **Adam reaches a different energy
regime. Its measured coarse curvature nearly saturates the adaptive stability
boundary, and tracking restricts fine motion through the shared denominator.**
The latter mechanism has causal evidence and a proved illustrative model,
but not a neural-network rate theorem comparable to the GD result.

Neither statement is permanent stagnation. Both allow slow growth, escaping
neurons, target dependence, and eventual acquisition. A useful finite-time
theorem need only explain why enough of the population cannot change enough
to attain the desired accuracy during the interval under discussion.

For context, the paper's separate five-million-update mixed-sine experiment
reports median raw relative error $1.81\times10^{-3}$ for joint Adam,
$0.244$ for joint GD, and $6.62\times10^{-7}$ for the supplied dictionary.
The Adam endpoint is below 1%; it still leaves a substantial precision gap.
Those long runs motivate the question. The conditional theorem is validated
on separately identified continuation intervals, not on every update of the
five-million-step experiment.

### The argument in five steps

1. Remove the affine output to identify the part of the error that needs
   nonlinear features. Decompose the gradient into effective fine learning
   and coarse tracking, retaining the compensating coarse response exactly.
2. Bound nonlinear sensitivity using population energy and concentration.
   Affine projection and the target's low-degree moments sharpen this bound.
3. Use the bound in the evolving ODE or native GD recurrence. This produces
   an upper bound on population radius and therefore on nonlinear output.
4. Check the structural conditions, their persistence, and perturbation
   margins empirically. Derive the output-error conclusion from the proved
   comparison, with the conditions and their measured status explicit.
5. Study Adam's actual processed motion separately. Its coarse stability
   constraint is informative even though the GD bound becomes uninformative
   at Adam's much larger parameter energies.

The paper can present the first four steps as one informal theorem and one
validation figure. The Adam paragraph should communicate the additional
mechanism without pretending that it is already the same theorem.

## 2. The exact ODE and the meaning of the two forces

Let $\mu$ be a probability measure on $[-1,1]$, such as the normalized
training grid. The network and loss are

$$
q_\theta(x)=d+\sum_{j=1}^W c_j\tanh(a_jx+b_j),\qquad
L(\theta)=\tfrac12\|q_\theta-y\|_{L^2(\mu)}^2.
$$

Here $a_j=\gamma_j$ is the physical slope, $b_j$ the hidden bias, and $c_j$
the readout. All parameter norms use these physical coordinates and the
ordinary Euclidean metric. Writing $e=q_\theta-y$ and $u_j=a_jx+b_j$, ordinary
gradient flow is exactly

$$
\dot a_j=-c_j\langle e,x\operatorname{sech}^2u_j\rangle,\quad
\dot b_j=-c_j\langle e,\operatorname{sech}^2u_j\rangle,\quad
\dot c_j=-\langle e,\tanh u_j\rangle,\quad
\dot d=-\langle e,1\rangle.
$$

The same equations are the characteristics of a population transport
description, with velocity depending on the current residual and hence on
the whole population. For our theorem, the useful transport observables are
population moments. We do not need to constrain every characteristic or
prove that every neuron stays near its initial position.

Let $P_C$ project onto affine functions and $P_H=I-P_C$. Put
$e_C=P_C(q-y)$, $e_H=P_H(q-y)$, $J_C=D(P_Cq)$, and $J_H=D(P_Hq)$.
Use an orthonormal affine basis, so $J_C$ has two rows. Where
$K_C=J_CJ_C^*$ is invertible, define

$$
\Pi=I-J_C^*K_C^{-1}J_C,\qquad
g_H=J_H^*e_H,\qquad F=\Pi g_H,
$$
$$
z=e_C+K_C^{-1}J_Cg_H,\qquad R=J_C^*z.
$$

Then

$$
\boxed{\nabla L=F+R,\qquad J_CF=0.}
\tag{1}
$$

The **effective fine gradient** $F$ contains the raw non-affine gradient
$g_H$ and its compensating coarse response $-J_C^*K_C^{-1}J_Cg_H$.
The **tracking gradient** $R$ measures departure from that compensating
equilibrium. Compensation does not disappear when tracking becomes small.
The two names refer to different objects throughout this note.

For example, a feature change that reduces fine error can also alter the
linear output. Compensation removes this first-order affine disturbance.
It is part of the coupled learning direction, rather than an additional
force to discard after the coarse fit is good. The projection gives
$\|F\|\le\|g_H\|$ in the full parameter norm; it does not imply that each
slope component is reduced or that compensation always opposes expansion.

An older polynomial decomposition also isolated an orthogonal residual
remainder. Equation (1) uses the **entire** fine residual, so no residual
remainder has been set to zero. Polynomial terms below are analytical
bounds on this exact object, with global remainder inequalities.

## 3. Why population organization can make fine learning slow

Write $p_j=(a_j,b_j,c_j)$, $r_j=\|p_j\|$, and

$$
M=\sum_j r_j^2,\qquad S_k=\sum_j r_j^k,\qquad
\chi_k=\frac{W^{k/2-1}S_k}{M^{k/2}}.
$$

$M$ is total hidden energy. $\chi_k$ describes its concentration, independently
of a common rescaling of all hidden parameters. Equal energy on every neuron
gives $\chi_6=1$. Equal energy on $k$ active neurons gives
$\chi_6=(W/k)^2$. Thus $W/\sqrt{\chi_6}$ is an effective energy-sharing count,
not a count of fixed active neurons and not an amount of motion.

This distinction answers an apparent paradox. Distributed energy does not
prevent all neurons from growing together. Rather, **at a specified total
energy**, spreading it over many broad features limits the sensitivity of
their combined nonlinear output. The dynamics then limit how quickly that
total energy can grow. Both ingredients are necessary.

### Lemma: population moments control exact tanh sensitivity

Let $q_H=P_Hq$. For every parameter state,

$$
\|J_H\|\le C_3\sqrt{S_6},\qquad
\|q_H\|\le A_3 S_4,\qquad
C_3=\frac{\sqrt6}{2},\quad A_3=\frac{\sqrt6}{8}.
\tag{2}
$$

**Proof.** The affine term $d+\sum_jc_j(a_jx+b_j)$ vanishes under $P_H$.
For real $u$,
$|\tanh u-u|\le |u|^3/3$ and
$|\operatorname{sech}^2u-1|\le u^2$.
Also $|ax+b|\le\sqrt2\sqrt{a^2+b^2}$ on $[-1,1]$. At fixed
$a^2+b^2+c^2=r^2$, maximizing the resulting products bounds one neuron's
non-affine output by $A_3r^4$ and the sum of squared derivative-column bounds
by $C_3^2r^6$. Summing outputs by the triangle inequality and derivatives
by the Hilbert–Schmidt bound proves (2). Orthogonal projection cannot
increase the norms. These inequalities are global; they do not assume a
Taylor convergence neighborhood. $\square$

If $\|e_H\|\le E_0$, projection contractivity gives

$$
\boxed{\|F\|\le\frac{C_3E_0}{W}\sqrt{\chi_6}\,M^{3/2}.}
\tag{3}
$$

The residual bound follows from full-loss dissipation under gradient flow,
or from a loss-descent condition under GD. For the effective flow
$\dot\theta=-F$, it follows directly from
$\frac{d}{dt}\|e_H\|^2=-2\|F\|^2$.

Equation (3) derives weak sensitivity from population structure. It does
not assume that the future force is small. At bounded $M$ and $\chi_6$,
the full force bound scales as $W^{-1}$. A typical per-neuron force can have
an additional $W^{-1/2}$ factor, but the theorem is about aggregate norms;
these scalings should not be conflated or transferred unchanged to Adam.

## 4. A complete conditional population theorem

We restart at a post-transient checkpoint, called time zero, where the
population has not already acquired useful large scales. Early tracking
may have helped form this checkpoint; the theorem need not explain that
earlier regime. Tracking remains in the subsequent bound as a disturbance
budget rather than being silently removed.

**Theorem A: accumulated concentration limits acquisition.** Suppose $M_0>0$
and $\|e_H(t)\|\le E_0$. Let $S_R(T)$ be an upper allowance for
$\int_0^T\|R(t)\|dt$, and set

$$
I_6(T)=\int_0^T\sqrt{\chi_6(t)}\,dt,\quad
\beta=\sqrt{M_0}+S_R(T),\quad
D_T=1-\frac{2C_3E_0\beta^2}{W}I_6(T).
$$

If $D_T>0$, then for every $t\le T$,

$$
\sqrt{M(t)}\le\frac{\beta}{\sqrt{D_T}},\qquad
\lambda_{\rm RMS}(t)=\frac{h\|a(t)\|}{\sqrt W}
\le\frac{h\beta}{\sqrt{W D_T}}.
\tag{4}
$$

The same conclusion holds for ordinary GD with left-point sums
$I_6=\sum_n\eta_n\sqrt{\chi_{6,n}}$ and
$S_R\ge\sum_n\eta_n\|R_n\|$, provided the residual bound holds at its iterates.
No approximation of GD by continuous flow is needed.

**Proof.** Put $b=\sqrt M$. The exact dynamics and (3) imply
$\dot b\le a(t)b^3+\|R\|$, where
$a=C_3E_0\sqrt{\chi_6}/W$.
Reserve the entire tracking budget at the start. More explicitly,
$\widetilde b(t)=b(t)+S_R(T)-\int_0^t\|R\|ds$ is at least $b(t)$,
starts at $\beta$, and satisfies
$\dot{\widetilde b}\le a\widetilde b^3$.
Integration of $(\widetilde b^{-2})'\ge-2a$ gives (4).

For GD, the triangle inequality gives
$b_{n+1}\le b_n+\eta_n a_nb_n^3+\eta_n\|R_n\|$.
Adding the remaining tracking reserve gives
$\widetilde b_{n+1}\le\widetilde b_n+\eta_na_n\widetilde b_n^3$.
The exact solution of $v'=a_nv^3$ over one step dominates this Euler
increment. Composing those exact scalar steps gives the same reciprocal
bound with left-point sums. Finally $\|a\|^2\le M$. $\square$

This is a rate statement. If the average of $\sqrt{\chi_6}$ is bounded by
$\bar z$, its basic allowed flow time is of order
$W/(E_0\beta^2\bar z)$. Constants and starting energy matter. There is no
universal 200k-update barrier, and a theorem whose denominator eventually
vanishes has not predicted permanent trapping.

A population conclusion also follows without controlling individual escape:

$$
\frac{\#\{j:h|a_j|\ge\lambda_*\}}{W}
\le\frac{\lambda_{\rm RMS}^2}{\lambda_*^2}.
\tag{5}
$$

Each counted neuron contributes at least $\lambda_*^2$ to the normalized
slope-square sum. This is the relevant use of the construction reference
$\lambda_*=0.25$. It corresponds to slope 64 when $h=2/512$, and 128 when
$h=2/1024$. It is not assumed to be a necessary accuracy threshold.

## 5. Turn slow acquisition into an output-error statement

Let $g=P_Hy$. If $B_n$ bounds $\sqrt{M_n}$ and $Q_n(B_n)$ bounds
$\|q_{H,n}\|$, then orthogonal projection and the reverse triangle inequality
give the paper's central conclusion:

$$
\boxed{\frac{\|q_n-y\|}{\|y\|}
\ge\frac{[\|g\|-Q_n(B_n)]_+}{\|y\|}.}
\tag{6}
$$

For example, (2) gives
$Q_n(B)=A_3\chi_{4,n}B^4/W$.
An output floor therefore uses a population moment at its evaluation time
as well as the accumulated moments controlling travel. Accumulated
concentration alone does not bound every endpoint moment.

This is why the population theorem is more useful than saying that slopes
remain below a chosen number. It limits the actual non-affine output of the
jointly evolving network, including readouts and biases. It also explains why
readout refitting is a diagnostic, not training success: the raw trained
network and a separately refitted network are different objects.

The generic constants in (2) waste structure removed by the affine
projection. For a symmetric measure, define

$$
U_2=x^2-\mu_2,\quad U_3=x^3-\frac{\mu_4}{\mu_2}x,\quad
s_k=\|U_k\|^2,
$$

where $\mu_k=\int x^k d\mu$. The cubic part is exactly

$$
q_3=-\Big(\sum_jc_ja_j^2b_j\Big)U_2
      -\frac13\Big(\sum_jc_ja_j^3\Big)U_3.
\tag{7}
$$

The two shapes are orthogonal. Direct differentiation and optimization at
fixed neuron energy give

$$
\|Dq_3\|\le D_3\sqrt{S_6},\quad
\|q_3\|\le B_3S_4,\quad
D_3^2=\frac8{27}s_2+\frac3{16}s_3,\quad
B_3^2=\frac{s_2}{64}+\frac{3s_3}{256}.
$$

Global tanh remainder inequalities supply constants $A_5,C_5$ such that
$\|q_H-q_3\|\le A_5S_6$ and
$\|J_H-Dq_3\|\le C_5\sqrt{S_{10}}$.
One valid choice is
$A_5=(2/15)2^{5/2}5^{5/2}/6^3$ and $C_5=6A_5$.
Let $\tau_3$ be the norm of the target's projection onto the degree-two/three
fine polynomial space. Splitting the exact residual before taking norms yields

$$
\|F\|\le D_3\sqrt{S_6}(\tau_3+Q)
             +C_5E_0\sqrt{S_{10}},\quad
Q=\min\{A_3S_4,B_3S_4+A_5S_6\}.
\tag{8}
$$

Indeed, write $J_H^*e_H=(Dq_3)^*q_H-(Dq_3)^*g+
(J_H-Dq_3)^*e_H$. The target term only sees the two surviving low-degree
shapes. Bound the three terms and apply $\Pi$. This explains the improvement
without assuming a small future force.

Equation (8) applies to sine as well as polynomial targets. A vanishing
$\tau_3$ gives an additional order of width suppression at bounded moments;
a nonzero $\tau_3$ remains explicitly in the bound. The generated quadratic
or cubic output may create a force opposing some expansion, but the theorem
does not require its universal dominance. Signed measurements show such a
requirement would exclude many observed trajectories.

For the practical comparison, substitute
$S_k=\chi_k B^k/W^{k/2-1}$ into (3), its projection refinement, and (8),
and take the minimum to obtain $V_n(B)$. All are nonnegative and
nondecreasing in $B$. The recurrence

$$
B_{n+1}=B_n+\eta_n\{V_n(B_n)+\|R_n\|\},\qquad B_0=\sqrt{M_0}
\tag{9}
$$

bounds the actual radius by induction. Equation (6) then bounds raw output
error. This is an evolving comparison, not a frozen Jacobian forecast.

## 6. What supports the persistence assumption?

The premise is about **parameter energy distribution**, not output error
being spread over neurons. We measure its accumulated concentration over
the subsequent interval. A short initial window supplies a reference level;
it does not logically certify the future allowance. The theorem is
conditional, and the empirical persistence check is part of the scientific
claim.

We also have an exact explanation of what can change concentration. Put
$e_j=r_j^2$ and $\rho_j=\dot e_j/e_j$ for positive energies. Differentiation
gives

$$
\frac{d}{dt}\log\chi_6
=3\left\{\frac{\sum_j e_j^3\rho_j}{\sum_j e_j^3}
          -\frac{\sum_j e_j\rho_j}{M}\right\}.
\tag{10}
$$

Concentration grows when already energetic neurons have a relative growth
advantage. Common expansion cancels. Individual identities can turn over
without changing this population statement.

There is a further aggregate closure. Define
$K=\chi_{10}/\chi_6^2\ge1$, $z_6=\sqrt{\chi_6}$, and $Y=Mz_6$.
Direct differentiation gives

$$
\|\nabla\log\chi_6\|^2=36(K-1)/M,\qquad
\langle\nabla\log\chi_6,\theta_{\rm hidden}\rangle=0.
$$

**Theorem B: dispersion limits coupled energy–concentration growth.** Under
effective flow, with $a_0=C_3E_0/W$,

$$
Y(t)\le
\frac{Y_0}{1-a_0Y_0\int_0^t\sqrt{9K(s)-5}\,ds}
\tag{11}
$$

while the denominator is positive.

**Proof.** The hidden score is
$6(e_j^2/\sum_i e_i^3-1/M)p_j$, which proves the two identities above.
Thus the radial and concentration pieces of
$\nabla Y=2z_6\theta_{\rm hidden}+(Mz_6/2)\nabla\log\chi_6$
are orthogonal, and
$\|\nabla Y\|^2=Mz_6^2(9K-5)$.
Using (3), $\dot Y\le a_0\sqrt{9K-5}Y^2$. Integrate its reciprocal.
$\square$

$K-1$ is a squared coefficient of variation of $e_j^2$ when neurons are
sampled with weights $e_j/M$. It controls capacity for differential growth.
It is not concentration itself: equal energy on any active subset gives
$K=1$, even if that subset is small. Theorem B derives concentration growth
from an accumulated dispersion condition; it does not remove every future
structural premise. Its value is an independently interpretable mechanism,
rather than replacing one measured future force by another.

### What the GD experiments establish

The population-concentration study covers 23 targets and two width-705 seeds.
Projection and target-moment refinements give informative conditional slope
and output bounds for at least **79k additional updates in all 46 cases**,
and for **100k in 32 cases**. At width 1409, the projection refinement alone
covers all six tested targets for 100k. These are durations of the comparison
bound under measured structural coefficients, not predictions made solely
from a starting checkpoint.

For degree five in the development panel, the useful interval increases
from 16.5k under the generic bound to 89.5k with affine projection and 100k
with target moments. For mixed sine the corresponding durations are 16k,
91.5k, and 100k, with its nonzero low-degree loading retained. The same
trajectory is used in each comparison. The improvement comes from the proof,
not from changing the experiment to fit the theorem.

The paper's narrower figure reports ten continuations across six target
families, where the output floor captures at least **97.7% of executed
error at each stored diagnostic time**. That statistic is a different
coverage claim from the 46-case duration result. Neither makes sparse
diagnostics an outward-rounded certificate of every floating-point update.

Matched redistribution tests change concentration while preserving total
energy, each parameter-block norm, and the leading affine output. Doses of
2, 4, and 10 in $\sqrt{\chi_6}$ substantially change the available sensitivity
budget. Subsequent acquisition depends on how the redistributed features
align with the target. In the Gaussian example, equal concentration doses
can yield very different slope responses, both with substantial remaining
output error. Thus concentration is a useful structural bottleneck, not a
sufficient state variable for exact force prediction.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/energy_final_20260925/causal_evolution.png" alt="Two matched concentration doses yield different later concentration, RMS slope, and raw output error" style="max-width:100%;">
  <figcaption>Gaussian target, width 705, seed 30; native GD restarted at update 25k. The two perturbations have equal initial hidden energy, slope RMS, and concentration. Different later acquisition at the same initial concentration demonstrates the role of feature–target alignment. Both remain far above the dotted 1% error reference.</figcaption>
</figure>

The signed audit further rules out a universal contraction story. Generated
output correction contributes nonpositively to the measured geometry–readout
imbalance at all 216 checked states, but target forcing exceeds it at 135
states. That imbalance is not total hidden energy. A negative contribution
to one cannot be imported into the other's proof.

Our confidence is therefore strongest in the chain **population structure
$\to$ limited nonlinear coupling $\to$ bounded finite-time growth $\to$
output floor**, with empirical support for the persistence of the premise.
We do not need an initial-data-only certificate or a universal equilibrium
to make this a useful explanation.

## 7. What changes under Adam

The gradient decomposition (1) still holds, but the actual increments are

$$
u_n^F=-\eta D_n\bar m_n^F,\qquad
u_n^R=-\eta D_n\bar m_n^R,\qquad u_n=u_n^F+u_n^R,
$$

where $D_n$ is the inverse diagonal denominator and the bars denote
bias-corrected first moments. Both components use the same denominator,
whose second moment comes from the full gradient, including cross terms.
Removing tracking only from signed displacement accounting does not remove
its influence on $D_n$.

Adam's population energy is already much larger and more concentrated than
GD's in the audited regime. Moderate subsequent concentration growth does
not make (3) small at that starting level, and Adam's processed update is
not bounded by the GD force without controlling its adaptive metric.
It would be incorrect to call insufficient concentration the established
Adam bottleneck.

The earlier intervention panel identifies an amplitude restriction. Reducing
tracking's contribution in a passive denominator comparison exposes a
capped mean fine-update gain of 5.30–9.95 across the six targets, two widths,
and two seeds. Actually using larger fine proposals helps some targets;
others mainly produce additional cancelling motion. Native fine motion is
often coherent, and removing momentum reduces early acquisition in all
24 cases. These results disfavor a universal explanation based on random
fine directions or intrinsically harmful momentum.

### Coarse-output disturbance and tracking renewal are distinct

Write $e_C^*=-K_C^{-1}J_Cg_H$, so $z=e_C-e_C^*$. For an actual update $u$,

$$
\Delta z=J_Cu+\mathcal E_C(\theta,u)-\Delta e_C^*,\quad
\mathcal E_C=\int_0^1(1-s)D^2(P_Cq)(\theta+su)[u,u]ds.
\tag{12}
$$

A fresh GD fine step has $J_Cu^F=0$. Adam's processed fine step generally
does not, because momentum and anisotropic scaling change its direction.
Even if that term is removed, movement of the compensating equilibrium can
renew tracking. Equation (12) identifies three separately measurable sources
instead of inferring a feedback loop from a correlated trace.

For slope energy $A=\sum_j a_j^2$, the exact discrete identity is

$$
\Delta A=2\langle a,u_a^F\rangle+
         2\langle a,u_a^R\rangle+\|u_a\|^2.
\tag{13}
$$

A negative linear tracking contribution need not mean permanent shrinkage.
For a sign-flipping coordinate, the positive squared-step term can cancel
it exactly. This is why native-step accounting and net population movement
are more informative than force magnitude or one signed term alone.

## 8. The new stability experiments: what changed our explanation

**Coverage of the current consolidated snapshot.** The campaign has completed
96 state audits, 288 GD continuation branches, and 72 Adam branches in the
`mid_analysis` snapshot. Audits cover six targets, two widths (705, 1409),
two seeds, two ages (25k, 130k), and both optimizers. The GD results here
cover three width–seed cohorts; Adam interventions here cover the two
width-705 seeds. Remaining primary cohorts and the secondary controls are
running. The numerical conclusions below are restricted to those populations.

### Observation: Adam is close to a coarse adaptive stability boundary

For a fixed preconditioner and stationary EMA momentum, linear stability
of a positive quadratic mode requires
$\eta\lambda<2(1+\beta_1)/(1-\beta_1)$, which equals 38 at
$\beta_1=0.9$. We measure the leading eigenvalue of the **full** scaled
Hessian $D^{1/2}\nabla^2L D^{1/2}$, not just its coarse approximation.

Across all 24 late Adam states, the ratio to this threshold has median
**0.9865** (range **0.2728–1.0277**). The coarse Gauss–Newton term accounts
for a median **0.9880** of its leading Rayleigh curvature, and the top mode
has median overlap **0.9992** with the scaled coarse-normal subspace.
Native GD lies much farther below its own threshold: median ratio **0.00524**.

This supports a coarse stability constraint. The moving Adam system is
nonlinear, so the frozen threshold remains a diagnostic rather than a full
stability proof. Its explanatory force comes from jointly identifying the
boundary and the direction that sets it.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/coarse_feedback_20260926/mid_analysis/stability_and_tracking.png" alt="Late Adam curvature is mostly coarse and near its local threshold; tracking oscillates with a target-dependent signed population contribution" style="max-width:100%;">
  <figcaption>Left: 24 Adam states at update 130k, spanning six targets, two widths, and two seeds. Right: 12 completed width-705 native continuations over the next 10k updates. Negative one-step tracking correlation indicates oscillation. Its signed linear contribution to slope energy is not the whole finite-step change; equation (13) retains the squared-step term.</figcaption>
</figure>

### Causal comparison: large-step GD develops a related coarse response

From 18 completed Adam-geometry starts, GD at 1.05 times its initial
stability threshold has 65–159 loss increases and moves its final normalized
sharpness to median 0.9599. Reducing tracking to one tenth at the same rate
produces zero loss increases in all 18 cases. Tracking therefore causes the
observed large-step response. This makes the relation between optimizers
more concrete than saying that their slope traces both look flat.

The attenuated algorithm has a different update rule; the full-loss Hessian
ratio is only a reference diagnostic for it. Increasing ordinary GD's rate
also accelerates acquisition for some targets, so this experiment does not
establish a universal contraction law or rate-independent failure.

### Negative result: native fine disturbance is not yet the persistence explanation

Project the processed fine proposal to remove its first-order coarse change,
then match its designated hidden-update norm. This intervention achieves
$J_Cu^F=0$ while retaining native tracking and inherited optimizer history.
Across the 12 completed native-amplitude pairs, the 10k endpoint slope RMS
ratio has median **1.00012** and range **0.99953–1.00118**. Accumulated
tracking energy has median ratio **1.0124**. Best raw error changes little.

Thus the simple story “fine displacement repeatedly drives tracking, which
blocks acquisition” is not established at native amplitude. This is a
successful discriminating experiment, not evidence to omit. At tenfold
amplitude, balancing does matter for some targets: the mixed-sine development
case loses 96.9% of tracking energy and gains 3.31% in slope RMS, with a
36.1% improvement in best error. Across all 12 amplified pairs the median
RMS gain is only 0.145%, so that example cannot carry a universal claim.

### The denominator cannot simply be removed without changing stability

At tenfold fine amplitude, 20 of the 24 completed frozen-denominator branches
fail: both balanced and unbalanced variants for all five non-polynomial
targets in each seed. Degree five survives both variants. All 48 dynamic
denominator branches complete. These failed trajectories are retained as
outcomes; they cannot identify a long-horizon mediation effect by comparing
only surviving checkpoints.

Two remaining controls sharpen the interpretation. **Fine-off** suppresses
active fine displacement while retaining native full-gradient moment updates;
it tests whether tracking can persist without that displacement. **Unit-gain
frozen denominators**, balanced and unbalanced, separate native-amplitude
dependence on adaptation from instability caused by tenfold amplification.
They are follow-ups selected after the development results, not prespecified
independent confirmation.

## 9. A proved model showing how tracking can maintain the restriction

Consider collective coarse and fine coordinates with
$L(x,s)=\kappa x^2/2+fs$. Use stationary EMA momentum and a shared scalar
RMS denominator, without bias correction or epsilon:

$$
m_{n+1}=\beta m_n+(1-\beta)(\kappa x_n,f),\quad
v_{n+1}=\beta_2v_n+(1-\beta_2)(\kappa^2x_n^2+f^2)/2,
$$
$$
(x_{n+1},s_{n+1})=(x_n,s_n)-\eta m_{n+1}/\sqrt{v_{n+1}}.
$$

**Proposition C: an exact coarse oscillation with slow fine drift.** Put
$T_\beta=2(1+\beta)/(1-\beta)$. If
$|f|<\sqrt2\eta\kappa/T_\beta$, this system has the exact solution

$$
x_n=(-1)^n b,\quad b^2=2\eta^2/T_\beta^2-f^2/\kappa^2,\quad
\sqrt v=\eta\kappa/T_\beta,\quad
s_{n+1}-s_n=-T_\beta f/\kappa.
\tag{14}
$$

**Proof.** Initialize $x_0=b$, $v_0=(\eta\kappa/T_\beta)^2$,
$m_0^s=f$, and $m_0^x=-(1-\beta)\kappa b/(1+\beta)$.
The squared-gradient input is constant and equals $v_0$.
The recurrence gives
$m_{n+1}^x=(1-\beta)\kappa x_n/(1+\beta)$, so the coarse update subtracts
$2x_n$. The fine update is $-\eta f/\sqrt v=-T_\beta f/\kappa$.
These relations preserve themselves at each step. $\square$

The coarse oscillation exists even at $f=0$. At small nonzero $f$, the
coarse stability scale fixes the denominator and fine drift is proportional
to the ratio $f/\kappa$. Increasing the nominal learning rate on this exact
solution increases oscillation amplitude, not fine drift. Suppressing the
active $s$ update while retaining its passive gradient input also leaves the
coarse cycle unchanged.

This is a concrete mechanism rather than a force-accounting identity. Its
limits are equally concrete: it proves existence, not attraction; its fine
loss is a local linear model, not a globally bounded neural loss; and it uses
a shared scalar RMS rather than arbitrary coordinatewise Adam. In rotated
physical coordinates, the stationary coordinatewise second moments for the
prescribed alternating input differ from their shared mean by relative
amount at most $(1-\beta_2)/(1+\beta_2)$, approximately $0.0005$ here.
That motivates the simplification without proving long-time shadowing.

The useful next neural inference is therefore modest: can ongoing coarse
activity maintain a mobility restriction even after fine displacement is
removed? The queued fine-off experiment directly distinguishes that
possibility from the original forcing-loop hypothesis.

## 10. What should survive into the paper theorem?

The main theorem should remain an output statement for GD:

> **Informal theorem.** After the coarse transient, suppose accumulated
> population concentration and coarse tracking satisfy the stated bounds.
> Then an explicit scalar recurrence bounds the joint hidden-parameter norm
> during GD. Its non-affine output remains below an explicit capacity bound,
> giving a lower bound on raw output error and an upper bound on population
> slope acquisition. The coefficients depend on population moments and the
> target's low-degree loading, rather than on a presumed small future force.

Use (6) in the main text, define $B_n$ and $Q_n$ in words, and put (8)–(9)
and the exact conditions in the appendix. Show the observation plots first,
then the theorem, then validation of both its premises and conclusion.
RMS is the principal slope observable; maxima and 99th percentiles add
little to this claim. Centers and readouts enter the hidden-radius and output
bounds; the theorem does not prove that centers become well distributed.

The Adam paragraph can now say that a distinct coarse adaptive stability
constraint limits how increased fine motion is used. The curvature evidence,
large-step GD comparison, and denominator interventions support this claim.
The stronger statement that fine displacement is what continually sustains
native tracking must await its targeted test. A proved reduced example is
appropriate in the appendix as an explanatory possibility, with its scope
stated explicitly.

What remains to improve the GD theorem is precision in the *aggregate*
premises and constants: preserving signed target coupling, bounding
accumulated dispersion, and exploiting orthogonality without demanding
per-neuron control. Exact future force forecasts and initial-data-only
certificates are optional strengthenings. They are not the standard that
the existing conditional explanation must meet to be useful.

## 11. Evidence status, reproducibility, and handoff

The mathematical statements proved in this note are the exact force
decomposition, population sensitivity lemma, Theorem A with its native-GD
transfer, population-to-output implication, concentration score identities,
Theorem B for effective flow, and the stated reduced Adam cycle. Neural
Adam persistence and a quantitative Adam acquisition bound remain open.

The GD empirical persistence and causal concentration studies are complete.
The latest Adam/GD stability campaign is partially complete as described in
Section 8. It uses full-batch float64 training on 2048 points and evaluates
raw error on an independent 8192-point grid at endpoints. Adam clones retain
their first moments, second moments, and update counters. Main comparisons
use equal update budgets; different GD step sizes imply different flow times.
Best-within-budget training error and current endpoint error are distinct.

Fifteen focused tests have passed remotely for the current feedback code.
They check native optimizer replay, exact signed population accounting,
balanced proposals, retained history, full Hessian products, batched/serial
agreement, sparse-checkpoint advancement, and the reduced cycle. All scientific
numerics and plotting run on Modal with memory caps; no large checkpoint
arrays are loaded on the local machine.

The detailed source notes and evidence below preserve reproducibility; the
definitions and arguments above do not require a reader to open them:

- [Population concentration: refinements, proofs, and broad GD validation](d34_population_concentration.md).
- [Adam coarse feedback: full derivations, protocols, and current results](d34_adam_coarse_feedback.md).
- [Adam population evidence: concentration, motion, and denominator studies](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/ADAM_POPULATION.md).
- [Completed campaign snapshot](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/coarse_feedback_20260926/mid_analysis/highlights.json),
  with the separate curvature and verification summaries in its sibling `summary.json`.
- [Paper appendix source for the new Adam argument](d34_adam_feedback_paper_appendix.tex).

The immediate handoff is to replace the explicitly partial campaign snapshot
when the remaining cohorts and secondary controls finish, then choose the
short Adam interpretation supported by those results. No additional
large training campaign is needed before writing the GD conditional theorem.
