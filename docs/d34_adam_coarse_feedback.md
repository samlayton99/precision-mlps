# Coarse feedback, adaptive scaling, and slow population acquisition

We want to explain why Adam's larger, more concentrated parameters still do
not acquire geometry that delivers the desired output accuracy. The GD theorem
already explains a different, quantitatively verified regime: moderately
distributed parameter energy limits nonlinear sensitivity and population
growth. Adam's measured energy and concentration make that estimate
uninformative. This does not invalidate the GD mechanism, but it prevents
using it as the explanation of Adam.

The most useful next hypothesis is a feedback loop: processed fine updates
disturb coarse balance; the resulting tracking activity contributes to Adam's
second moment; the shared denominator then restricts subsequent fine motion.
We have measured the last link and some responses to releasing that
restriction. We have not established the first link as the dominant source
of tracking, or that the loop explains persistence. This note derives the
identities needed to test those claims, proves an illustrative coupled model,
and specifies experiments whose negative outcomes would change the paper's
interpretation. No new training results are reported here.

**Notation.** All output inner products use the same normalized training
measure as the squared loss. Coarse outputs are represented in an orthonormal
basis of constant and linear functions, so their Jacobian has two rows.

| Symbol | Meaning |
|---|---|
| $q_\theta$, $L=\frac12\|q_\theta-y\|^2$ | Network output and loss. Parameters include the output bias. |
| $C=P_Cq_\theta$, $H=P_Hq_\theta$ | Coarse and non-affine output; $P_H=I-P_C$. |
| $e_C=C-P_Cy$, $e_H=H-P_Hy$ | Output errors, with the output-minus-target sign convention. |
| $J_C=DC$, $J_H=DH$ | Current output Jacobians. |
| $F$, $R$ | Existing effective fine and tracking gradients; compensation remains inside $F$. |
| $D_n$ | Positive diagonal inverse Adam denominator used for update $n$. |
| $\bar m_n^F$, $u_n^F=-\eta D_n\bar m_n^F$ | Bias-corrected fine first moment and its actual parameter update. |
| $A=\sum_j a_j^2$, $\lambda_{\rm RMS}=h\sqrt{A/W}$ | Population slope energy and normalized RMS slope. |

## 1. What the current interventions establish

The most revealing example is the compact bump at width 705, seed 30. In the
130k--140k intervention, scalar amplification of the fine update produces
15.6 times the native fine slope path. Directional coherence falls from 0.529
to 0.081, and endpoint slope RMS exceeds the native control by only 0.048%.
More fine activity does not produce proportionate population expansion.
However, this cancelling response is induced by the intervention: native
fine motion in the failing families is largely coherent.

Across six targets, two widths, and two seeds, attenuating tracking inside
the denominator used for the fine proposal exposes a mean 5.30--9.95-fold
available gain after capping each gain at ten. Scalar amplification improves
pulse-end RMS for all degree-five, sine, and kink cases. Thus the restriction
does matter to acquisition; it is not universally harmless. Bump, step, and
Gaussian responses differ, and the altered trajectories can renew tracking.
Replacing fine momentum with the current gradient lowers early acquisition
in all 24 cases, so removing momentum is not the supported remedy.

These observations identify an amplitude restriction and coupled responses.
They do not establish that the denominator is necessary for stability, that
all cancellation comes from tracking, or that a single mechanism explains
every target. In particular, the previous crossed geometry--residual test
did not find universal cancellation between those two raw force changes.

The existing experiments at 25k--150k must also remain separate from the
five-million-update mixed-sine observation in manuscript version 27. They
support mechanisms in their measured regime, not continuous coverage of the
entire longer run. The long-run Adam error quoted in Section 3.5 is
$1.81\times10^{-3}$, below 1% but far above the supplied dictionary's
$6.62\times10^{-7}$. The relevant paper claim is an accuracy gap at matched
budget, not that every Adam experiment misses a 1% requirement.

## 2. The same gradient decomposition, with optimizer-dependent motion

Assume $K=J_CJ_C^*$ is invertible. Define

$$
g_H=J_H^*e_H,\qquad
\Pi=I-J_C^*K^{-1}J_C,\qquad F=\Pi g_H.
$$

Then the decomposition and its coarse constraint are exact:

$$
\nabla L=F+R,\qquad
R=J_C^*z,\qquad
z=e_C+K^{-1}J_Cg_H,\qquad J_CF=0.
\tag{1}
$$

Indeed, subtracting $F$ from $J_C^*e_C+g_H$ gives the expression for $R$.
Multiplying $F$ by $J_C$ gives zero. This retains the existing meaning of
tracking as departure from the compensating coarse equilibrium.

For GD the component updates are $-\eta F$ and $-\eta R$. For Adam they
are $-\eta D_n\bar m_n^F$ and $-\eta D_n\bar m_n^R$, with the same actual
denominator. The identity $J_CF=0$ generally does not imply
$J_CD_n\bar m_n^F=0$. The old decomposition remains valid as gradient and
update accounting; it does not become a coarse-preserving Adam update.

### Proposition 1: exactly how fine motion disturbs coarse output

Let $u$ be any parameter increment for a twice continuously differentiable
network. Define the directional remainder

$$
\mathcal E_C(\theta,u)=
\int_0^1(1-s)D^2C(\theta+su)[u,u]\,ds.
$$

Then

$$
C(\theta+u)-C(\theta)=J_Cu+\mathcal E_C(\theta,u).
\tag{2}
$$

Consequently an isolated GD fine step satisfies

$$
C(\theta-\eta F)-C(\theta)=\mathcal E_C(\theta,-\eta F),
\tag{3}
$$

whereas an isolated Adam fine step satisfies

$$
C(\theta+u_n^F)-C(\theta)
=-\eta J_CD_n\bar m_n^F+\mathcal E_C(\theta,u_n^F).
\tag{4}
$$

**Proof.** Integrate the derivative of $C(\theta+su)$ twice and use
$J_CF=0$. $\square$

Thus a fresh GD fine step changes coarse output only through a second-order
step effect. Adam also has a possible first-order effect from scaling and
history. The remainder is directional and evaluated on the actual step; no
bound on the largest neuron or a global Hessian maximum is required to
measure it. For the full update, (2) includes both component increments and
their nonlinear interaction. Separate isolated-step remainders must not be
added as if there were no cross terms.

The history contribution can also be separated exactly. Write the fine
moment used at update $n$ as
$\bar m_n^F=I_n+\sum_{k\le n}w_{nk}F_k$, where $I_n$ contains any inherited
moment and the weights include bias correction. For any scalar $d$,

$$
J_{C,n}D_n\bar m_n^F
=J_{C,n}(D_n-dI)\bar m_n^F
+d\sum_{k\le n}w_{nk}(J_{C,n}-J_{C,k})F_k
+dJ_{C,n}I_n.
\tag{5}
$$

This follows by adding and subtracting $dJ_{C,n}\bar m_n^F$ and using
$J_{C,k}F_k=0$. It separates anisotropic scaling, movement of the coarse
tangent space, and inherited history. It does not say that momentum is
harmful overall; our direction interventions show its benefit.

### Coarse output preservation is not tracking-equilibrium preservation

Set $e_C^*(\theta)=-K^{-1}J_Cg_H$, so $z=e_C-e_C^*$. Equation (2) gives
the exact additional identity

$$
z(\theta+u)-z(\theta)
=J_Cu+\mathcal E_C(\theta,u)
-\big[e_C^*(\theta+u)-e_C^*(\theta)\big].
\tag{6}
$$

The equilibrium itself moves. Even under a GD fine step, its movement can
be first order in the step. We must therefore measure both the disturbance
of coarse output and the movement of its compensating equilibrium before
claiming to explain renewed tracking. Finally,
$R(\theta+u)=J_C(\theta+u)^*z(\theta+u)$ also includes Jacobian movement.
Equations (2) and (6) provide a common, exact comparison for GD and Adam.

## 3. A coupled model proves what correction can do, and what it cannot

Consider two collective coordinates: a coarse error coordinate $x$ and a
coordinate $s$ along a fine direction. On a finite region take

$$
L(x,s)=\frac\kappa2x^2+fs,\qquad
B=\begin{pmatrix}d_c&d_{cf}\\d_{cf}&d_f\end{pmatrix}\succ0,
\qquad \kappa>0.
$$

$f$ is a constant local fine gradient, not a target function. The linear
fine potential describes a finite segment of motion, not a globally bounded
loss. $B$ is a fixed preconditioner in these collective coordinates. A
diagonal preconditioner in physical parameter coordinates need not be
diagonal after rotating to coarse and fine directions.

### Proposition 2: correction reduces the available fine mobility

For preconditioned GD with step $\eta$, suppose
$0<\eta\kappa d_c<2$. Define

$$
x_*=-\frac{d_{cf}f}{\kappa d_c},\qquad
\alpha=1-\eta\kappa d_c,\qquad
\mu=d_f-\frac{d_{cf}^2}{d_c}>0.
$$

The iterates satisfy

$$
x_n-x_*=\alpha^n(x_0-x_*),
$$

$$
s_N-s_0=-N\eta\mu f
-\frac{d_{cf}}{d_c}(1-\alpha^N)(x_0-x_*).
\tag{7}
$$

**Proof.** The two update equations are

$$
x_{n+1}=x_n-\eta(\kappa d_cx_n+d_{cf}f),\qquad
s_{n+1}=s_n-\eta(\kappa d_{cf}x_n+d_ff).
$$

Subtract $x_*$ in the first equation. Substitute its geometric solution in
the second and sum, using $1-\alpha=\eta\kappa d_c$. Positivity of the
Schur complement of $B$ gives $\mu>0$. $\square$

After the coarse transient, the direct preconditioned fine step would move
$s$ by $-\eta d_ff$. Its induced coarse response supplies
$+\eta d_{cf}^2f/d_c$, leaving $-\eta\mu f$. This is a precise example of
correction consuming part of the available fine motion. There is no
equilibrium in $s$ when $f\ne0$: motion continues at a reduced rate.

This proposition does not imply universal slow acquisition. The mobility
$\mu$ need not be small, and it can exceed the corresponding GD mobility.
It establishes a mechanism and identifies the coupling that would have to
be measured. It is an exact fixed-preconditioner model, not a proof for
Adam's changing moments or for the neural network.

The analogous instantaneous coarse-preserving mobility in the full model is

$$
M_D=D-DJ_C^*(J_CDJ_C^*)^{-1}J_CD.
\tag{8}
$$

The velocity $-M_Dg_H$ uniquely minimizes
$\frac12v^*D^{-1}v+g_H^*v$ subject to $J_Cv=0$.
Lagrange multipliers prove the formula directly. Moreover
$M_D=D^{1/2}(I-P)D^{1/2}$ for an orthogonal projection $P$, hence
$0\preceq M_D\preceq D$. This compares constrained fine dissipation to
unconstrained dissipation in the same metric; it does not order their slope
components, compare Adam to GD, or bound the actual momentum-driven path.
Our earlier adaptive-access diagnostic already measures $g_H^*M_Dg_H$.
It is not a new definition of the recorded effective fine gradient $F$.

## 4. Why large-step GD is a useful, nontrivial comparison

For ordinary GD in Proposition 2, $B=I$. The coarse coordinate obeys
$x_{n+1}=(1-\eta\kappa)x_n$, and the fine coordinate moves by $-\eta f$.
Coarse oscillation begins when $\eta\kappa>1$; it decays when
$\eta\kappa<2$, persists at equality, and grows above two unless its
initial amplitude is zero. It does not slow $s$ in this separable model.
Thus oscillation alone is not the missing explanation. Coupling through
nonlinear geometry, a moving equilibrium, or adaptive history is essential.

On nonlinear losses, large-step GD can instead exhibit self-stabilization:
oscillations feed back on curvature. This has been studied theoretically by
[Damian, Nichani, and Lee](https://arxiv.org/abs/2209.15594) and empirically
in the [GD edge-of-stability study](https://arxiv.org/abs/2103.00065).
These results make the proposed comparison well motivated, but they do not
identify our tracking gradient with the unstable Hessian mode or imply that
the stabilizing response opposes population slope expansion.

For a fixed positive preconditioner and EMA momentum coefficient $\beta_1$,
the positive-curvature quadratic stability threshold is

$$
\eta\,\lambda_{\max}(D^{1/2}\nabla^2L D^{1/2})
<\frac{2(1+\beta_1)}{1-\beta_1}.
\tag{9}
$$

It follows from the scalar characteristic polynomial
$r^2-[1+\beta_1-\eta(1-\beta_1)\lambda]r+\beta_1$.
With $\beta_1=0.9$ the right side is 38. For GD the corresponding threshold
is two. [Cohen et al.](https://arxiv.org/abs/2207.14484) find that actual
full-batch Adam often follows the associated adaptive stability boundary.
For moving, bias-corrected Adam, (9) is a diagnostic from the locally frozen
recurrence, not a global stability certificate. Negative-curvature modes
also require separate treatment.

Unlike GD, Adam additionally remembers oscillatory tracking in its second
moment. For the illustrative inputs $F=(f,f)$ and
$R_n=(-1)^n(r,-r)$, the raw gradients are orthogonal but each coordinate's
steady second moment is

$$
v_n=f^2+r^2
\pm 2fr\frac{1-\beta_2}{1+\beta_2}(-1)^n.
\tag{10}
$$

This is obtained by substituting a constant plus an alternating term into
the second-moment recurrence. The fine first moment approaches $f$.
When the alternating cross term and denominator offset are negligible,
the fine update magnitude is $\eta|f|/\sqrt{f^2+r^2}$. GD processing the
same fine input gives $\eta|f|$, without this denominator dependence.
This is a downstream filter calculation with prescribed inputs. Closing
the neural feedback loop requires explaining the endogenous tracking that
produces those inputs. Orthogonality alone does not control each
coordinate's squared-gradient cross term in general.

## 5. The missing tests, in an order that can change our conclusion

### First: determine what renews tracking at the saved states

Use the existing early and late checkpoints across all six targets, widths
705 and 1409, and seeds 30 and 31. No new long training is needed for this
part. Evaluate the following on Modal, loading one small checkpoint at a
time and returning scalar diagnostics:

- Current fine gradient, scaled current fine gradient, fine momentum, and
  scaled fine momentum, with their native magnitudes and a second comparison
  at a common proposal norm. This separates scaling from history.
- $J_Cu^F$, the exact isolated fine-step change in $C$, and its directional
  remainder. Also evaluate (6), so movement of the compensating equilibrium
  cannot be misattributed to coarse output disturbance.
- The native second moment, the existing shadow second moment with attenuated
  tracking, and their effect on the same fine proposal. Retain squared-gradient
  cross terms; do not infer this from a ratio of raw force norms.
- The leading positive full-Hessian curvature and its adaptively scaled
  counterpart using converged matrix-free products. Measure how much of the
  unstable candidate direction lies in the coarse-normal subspace. A coarse
  Gauss--Newton matrix alone is not the full Hessian at nonzero residual.

The exact decompositions are implementation checks. The scientific question
is which measured term dominates renewed tracking. If adaptive disturbance
is small, the proposed first link is unsupported even though the denominator
effect remains real. If the moving equilibrium dominates for both optimizers,
the next theory should study that common population response instead.

### Second: test the GD learning-rate explanation directly

Start with the six width-705, seed-30 late states from each optimizer.
Fork ordinary GD from both sources so differences of geometry are not
confounded with optimizer choice. Use the native rate and rates at
0.25, 0.5, 0.9, 1.05, and 1.25 times the local positive-curvature reference
$2/\lambda_{\max}(\nabla^2L)$, removing duplicate rates. If there is no
positive leading curvature, label that state outside this calibration
instead of inventing a threshold. These are diagnostic doses; the reference
is not a promise that a nonlinear continuation is stable.

Use 2k-update pilots to locate monotone, oscillatory bounded, and divergent
responses; extend the informative neighboring rates to 10k. Track raw
output error, slope RMS, signed fine and tracking contributions, accumulated
fine path, and the finite-step remainder. For GD compare both update count
and elapsed flow time $n\eta$ over overlapping intervals. Greater progress
at the same number of larger updates is not by itself a new mechanism.
Record divergent and unresolved runs as outcomes.

The hypothesis predicts a transition in which renewed tracking accompanies
loss of proportional growth in useful fine motion. If larger stable steps
simply accelerate acquisition, the tested GD baseline was partly limited
by its chosen step size, and that must qualify the paper. If they diverge,
that establishes a stability limit, not self-stabilization. If they oscillate
without appreciable changes in signed acquisition, oscillation is not the
proposed bottleneck. Only a measured population effect justifies connecting
the stability phenomenon to scale failure.

### Third: remove the proposed coupling rather than just amplifying activity

For an actual processed fine proposal $u^F$, use the minimum-$D^{-1}$-norm
coarse correction

$$
u^{F,\mathrm{balanced}}
=u^F-DJ_C^*(J_CDJ_C^*)^{-1}J_Cu^F.
\tag{11}
$$

It satisfies $J_Cu^{F,\mathrm{balanced}}=0$ by multiplication. This is an
intervention on the update, not a renamed fine gradient. It removes the
first-order coarse-output disturbance but leaves the moving-equilibrium
effect in (6), a distinction that makes the test informative.

Compare four arms: native, the established scalar fine amplification,
coarse-balanced fine motion at the native available norm, and coarse-balanced
fine motion at the amplified available norm. Match hidden-update norms
locally, as in the existing intervention. Apply (11) to the full fine
proposal, including its output-bias component, then scale that entire
corrected vector to match the hidden-update norm. Scaling the entire vector
preserves its zero coarse response. Unlike the previous intervention, this
allows the fine output-bias proposal to change; the tracking proposal remains
native in every coordinate. Report full and hidden motion budgets separately.
Verify rank, and report unresolved solves or a zero corrected hidden norm
instead of silently substituting another policy.
Use the same physical coordinates and exact $F,R$ definitions in every arm.
Keep native full moment recurrences and the native tracking proposal at the
clone's current state. Do not reset optimizer history.

Run a 10k pulse and 10k native release, first on the six width-705, seed-30
cases, then on the second seed and width 1409 if the mechanism is observable.
Equality of local proposal norms does not imply equal cumulative budgets;
report both. Sparse endpoint checkpoints and online scalar accumulators are
sufficient. This stage requires new short continuations: existing scalar
summaries cannot recover its counterfactual trajectories.

If the loop is acquisition-limiting, removing the disturbance should reduce
renewed tracking and denominator restriction, then increase sustained signed
fine expansion relative to its matched-amplitude control. If the first two
effects occur without additional acquisition, the loop exists but is not the
main acquisition bottleneck. If tracking persists through movement of
$e_C^*$, output balancing has not broken the relevant loop; (6) identifies
why. If the amplified coarse-balanced arm still generates fine reversals
with little tracking, study fine residual--geometry feedback separately.
These outcomes must not all be reported as confirmation of one hypothesis.

Projection itself changes direction. Record its immediate radial slope
contribution and linearized loss change at every common starting state, so
an immediate directional benefit is not credited to later denominator
feedback. If the balanced arm shows the predicted tracking and acquisition
changes, add a paired check that freezes the fine denominator in both the
balanced and unbalanced arms, with identical initial amplitude matching.
The native full second moment still evolves passively and the tracking
proposal remains native. A reduced advantage when this denominator mediator
is held fixed would support its causal role; an unchanged advantage would
point to direction or direct tracking effects. Report the resulting unequal
cumulative paths rather than claiming they were controlled.

Before any continuation, verify the balance solve, the exact finite-step
population identity, native Adam replay, and retention of inherited moments.
Measure attached-network raw training error every update and independent-grid
error at common endpoints. Use the same current-checkpoint and best-within-
budget rules for every arm, reported separately. Keep the 1% diagnostic and
construction scale $0.25$ distinct from the paper's high-precision objective.
Report RMS population slopes, not a maximum or a 99th percentile.

The first seed is a development panel; the second seed and wider cases test
transfer of the selected mechanistic prediction. Do not present a rate or
intervention chosen on the development runs as prespecified held-out evidence.
Measure pilot runtime before allocating continuations within the current
authorized compute balance; no new multi-hour budget is assumed here.
All numerical execution is on Modal. No dense local checkpoint loading or
new million-update traces are needed.

### Secondary controls motivated by the first completed panel

The width-705, seed-30 development panel creates two sharper questions.
Balancing at native amplitude eliminates $J_Cu^F$ but changes accumulated
tracking little in these six cases. Freezing the fine denominator at tenfold
amplitude makes five of the six target cases leave the numerically resolved
regime in both balanced and unbalanced branches. These observations do not
establish that ongoing fine disturbance sustains native tracking, or that a
frozen denominator would be unstable at native amplitude.

We therefore add four paired branches, using the same 130k checkpoints,
10k pulse, and 10k native release: native; zero fine displacement; a fine
denominator frozen at its fork value with gain one; and the same unit-gain
frozen proposal with coarse balancing. This is a follow-up chosen after the
development results, separate from the original six-arm comparison. Apply
it across the same six targets, two widths, and two seeds within the approved
three additional GPU-hours.

The predictions distinguish two explanations. If continuing fine displacement
is necessary to maintain tracking, removing that displacement should substantially
deplete tracking during the pulse. If tracking remains comparable to native,
the proposed fine-forcing loop is not necessary for its persistence on this
interval. The full native gradient still updates the passive moments, and
tracking motion still changes the parameters: this test removes active fine
displacement, not every dependence on the fine residual.

The unit-gain frozen branches distinguish failure caused by tenfold
amplification from a need for denominator adaptation already at the native
amplitude. Record early exits as outcomes, not missing successful comparisons.
If both frozen branches fail, their long-horizon acquisition difference is
not an identified mediation effect. In all branches, retain the signed fine
and tracking terms and the shared squared-step term in the exact $A$ balance.

## 6. How Section 3.5 should tell the argument

Use the title **Why joint training acquires useful geometry slowly**. The
current phrase “fails to self-reinforcing slope acquisition” is awkward and
suggests a single established explanation that the Adam results do not yet
supply. The argument should proceed in this order:

1. **Observation:** at the same output criterion and budget, learned features
   leave an accuracy gap relative to supplied geometry. Show raw error and
   RMS slope acquisition. Preserve the learned-center/refit evidence;
   slopes alone are not a universal accuracy certificate.
2. **Common question:** the effective fine gradient supplies the main positive
   acquisition channel in the audited regimes, but useful cumulative motion
   depends on both its geometry and how the optimizer processes it.
3. **GD result:** keep Theorem 3.3 as a conditional, quantitative result.
   Distributed moderate energy limits nonlinear sensitivity, controls growth,
   and yields the output-error floor. Its proof and empirical condition checks
   are the strongest formal part of this section.
4. **Adam mechanism:** state the measured denominator restriction and paired
   intervention response. Add the feedback explanation only to the extent
   supported by the new tests. Put (2)--(11), moment details, and full target
   coverage in the appendix; a second large theorem is not needed in the
   one-page main argument.
5. **Consequence:** adaptive training improves acquisition but does not, in
   the observed budgets, reproduce the useful supplied geometry. QUILL avoids
   having to discover that geometry through these dynamics.

A main-text paragraph supported by the evidence already available is:

> Adam reaches a different parameter regime, so the preceding GD bound does
> not explain its remaining acquisition gap. We therefore examine the updates
> that actually move the slopes. Tracking can contribute little net slope growth
> while restricting the effective fine update through Adam's shared second
> moment. Attenuating tracking in that denominator exposes substantially
> larger fine proposals. Using the additional budget accelerates acquisition
> for some targets, while for others it mainly creates cancelling motion or
> a renewed tracking response. Thus concentrated parameter energy and large
> update activity do not by themselves ensure useful population growth.

If the proposed tests support the full loop, the stronger explanatory sentence
would be: **Adaptive fine motion repeatedly renews coarse tracking, whose
second-moment contribution limits subsequent fine acquisition.** This is a
prediction to validate, not yet a paper conclusion.

For figures, retain the two observation panels: raw error and RMS scale.
The GD figure should continue to show the population upper bound and output
lower bound. A compact Adam mechanism pair can show tracking-denominator
restriction and paired population/output responses to breaking the coupling.
Curvature diagnostics belong in the appendix unless they are the decisive
explanation. A visually similar plateau is not evidence of identical causes.

The objective is a common mathematical language with optimizer-specific
mechanisms where the data require them. We should not weaken the useful GD
theorem to fit Adam, or label a bookkeeping identity as an Adam persistence
theorem. The added value would be identifying which part of the coupled
response restricts useful motion and demonstrating that changing it has the
predicted population consequence.

## Evidence and source record

- Existing measurements, protocols, and artifact provenance:
  [Adam population study](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/ADAM_POPULATION.md).
- GD concentration theorem and its empirical coverage:
  [population concentration note](d34_population_concentration.md).
- Manuscript inspected: version 27, Section 3.5, Theorem 3.3, and Figures 4--5
  on pages 7--8. The older `d34_scale_acquisition_paper_main.tex` retains a
  previous theorem and is not the source of this latest section.
- Literature is used for stability mechanisms and diagnostics, not as evidence
  about these particular networks. The propositions in this note have the
  explicit proofs above; the neural Adam feedback and persistence claims
  remain experimental hypotheses.
