# A population theorem from preserved readout–slope structure

The central question is why a small outward slope force does not quickly
reinforce itself. The coupled transport ODE gives a concrete answer in an
aligned sector: **the compensation that preserves the coarse fit cannot
exceed the largest local penalty for the generated fine error.** Consequently,
smaller slopes can grow while the largest slope cannot advance. The readouts
continue moving, and the state need not approach an equilibrium.

This note proves that statement for a heterogeneous population, derives a
finite persistence horizon in the presence of a competing target loading,
and couples it to a tracking and approximation argument for tanh training.
The structural conditions are propagated by the equations; future small fine
force is not an assumption of the reduced-system theorem.

There is an important distinction between a proved mechanism and its empirical
coverage. The [population audit](../results/checkpoint_D_optimizers/expD34_readout_race/population_coverage/README.md)
found **zero of 223 archived states in the exact aligned sector**. Mixed
readout–slope signs and nonzero hidden biases are substantial, not rare
exceptions. The proof below therefore remains a sufficient mechanism example
and a conditional transfer theorem; it does not certify the explanation of
the archived plateaus. The full heterogeneous polynomial force remains
accurate at the tested wide states, motivating a theorem that retains those
signs and biases.

## 1. Example and objective: redistribution without scale acquisition

Consider two populations of small positive slopes with positive readouts.
Removing a generated cubic error favors reducing the larger slopes. Keeping
the linear output fixed requires compensating readout and slope motion.
That common compensation can increase some smaller slopes. The relevant
question is therefore whether it can also move the upper edge of the slope
distribution. A mean contraction argument would not answer this question.

This distinction matters empirically. The
[mechanism note](d34_coupled_ode_mechanisms.md#6-which-assumptions-does-the-evidence-actually-support)
records weak reinforcement at wide states and a later correction-dominated
regime. Neither establishes universal slope contraction. The theorem below
instead studies the moving upper edge, and its perturbation version counts
distinct neurons that ever cross a threshold.

Use the same notation and mean normalization as the
[rescaled transport model](d34_rescaled_transport_model.md):

| Symbol | Meaning |
|---|---|
| $W$, $\mathbb E_W$ | Number of particles and their equal-weight average. |
| $X_j=(\alpha_j,\beta_j,\zeta_j)=\sqrt W(a_j,b_j,c_j)$ | Rescaled slope, hidden bias, and readout. |
| $p=\mathbb E_W\zeta\alpha$ | Conserved linear-output contribution in the odd reduced model. |
| $P_H$, $y_H$ | Empirical orthogonal projection away from $1,x$, and $P_Hy$. |
| $v=\langle x^2\rangle_m>0$ | Input variance on a symmetric grid with $\lvert x_i\rvert\le1$. |
| $\lambda_j=h\lvert a_j\rvert=h\lvert\alpha_j\rvert/\sqrt W$ | Normalized acquisition scale; $h=2/N_{\rm ref}$. |
| $t$, $\tau_q=t/W^q$ | Physical gradient-flow time and the relevant reduced time. For GD, $t=\eta n$. |

All results about particle maxima below are finite-population results, valid
at any finite $W$. The empirical measure of the particles satisfies the
nonlinear transport equation weakly. No density, diffusion, or stochastic
forcing is introduced.

## 2. The coupled compensation has a maximum principle

**Purpose.** We first isolate the structural fact common to two different
ODEs. Their coefficients will then verify its hypotheses. Throughout this
section hidden biases vanish, the target is odd, and the output constant is
zero. Parity preserves this subspace in both reduced models.

Consider the equations

$$
\alpha_j'=\zeta_j[k-f(\alpha_j^2)],\qquad
\zeta_j'=\alpha_j[k-g(\alpha_j^2)],
\tag{1}
$$

where $f,g$ can depend on the current population and

$$
k=\frac{\mathbb E_W[\zeta^2f(\alpha^2)+\alpha^2g(\alpha^2)]}
        {\mathbb E_W(\zeta^2+\alpha^2)}.
\tag{2}
$$

The shared term $k$ is the compensation required to preserve $p$. It is
part of the effective fine force at exact coarse balance, not a tracking
disequilibrium force. Direct differentiation gives

$$
p'=k\mathbb E_W(\zeta^2+\alpha^2)
   -\mathbb E_W[\zeta^2f(\alpha^2)+\alpha^2g(\alpha^2)]=0.
\tag{3}
$$

### Lemma: preserved alignment prevents the upper edge from advancing

Suppose the vector field is locally Lipschitz, initially

$$
0\le\alpha_j(0)\le R,\qquad \zeta_j(0)\ge\alpha_j(0),\qquad p>0.
\tag{4}
$$

On the sector $0\le\alpha_j\le R$, $\zeta_j\ge\alpha_j$, suppose
$f(0)=0$, $f$ is nondecreasing on $[0,R^2]$, $0\le g(s)\le f(s)$ there, and
$f(R^2)\le B(\tau)$ for a nonnegative locally integrable bound $B$.
Then the sector is preserved, and

$$
\max_j\alpha_j(\tau)\le R,\qquad
Z(\tau):=\mathbb E_W\zeta^2(\tau)
\le Z_0+2p\int_0^\tau B(s)\,ds.
\tag{5}
$$

Moreover,

$$
(\zeta_j^2-\alpha_j^2)'=
2\alpha_j\zeta_j[f(\alpha_j^2)-g(\alpha_j^2)]\ge0.
\tag{6}
$$

**Proof.** Equation (2) is a weighted average of values of $f$ and $g$,
with nonnegative weights. Hence $0\le k\le f(R^2)$. At a boundary particle
with $\alpha_j=R$, equation (1) gives $\alpha_j'\le0$. At $\alpha_j=0$,
it gives $\alpha_j'=\zeta_jk\ge0$. At $\zeta_j=\alpha_j$,

$$
(\zeta_j-\alpha_j)'=\alpha_j(f-g)\ge0.
$$

These boundary signs preserve the sector; they also justify the positivity
used in (6). Conservation of $p$ and (1) give

$$
Z'=2pk-2\mathbb E_W[\zeta\alpha g(\alpha^2)]\le2pB.
$$

This proves (5). The denominator in (2) stays at least $2p$. At finite
$W$, (5) bounds every readout on every finite interval, so the solution
continues as long as the coefficient hypotheses apply. In addition,
$|\zeta_j'|\le RB$, giving the useful support bound

$$
\max_j\zeta_j(\tau)\le\max_j\zeta_j(0)+R\int_0^\tau B(s)\,ds.
\tag{7}
$$

This last bound is uniform in width when the initial support and $B$ are.
The full affine coarse Gram on this odd subspace obeys

$$
K_{00}=1+Z\ge1,\qquad K_{11}=v\mathbb E_W(\zeta^2+\alpha^2)\ge2vp,
\qquad K_{01}=0.
\tag{8}
$$

Thus conditioning is also preserved, rather than imposed along the future
trajectory. $\square$

**What this says.** Readout compensation is not being neglected. Its exact
coupling through the population moments is what proves the boundary sign.
Individual slopes may increase when their local penalty is below $k$.
Equation (6) also permits readouts to grow relative to slopes. This is a
moving, heterogeneous system, not an equilibrium construction.

## 3. Theorem 1: two mechanisms discharge the structural conditions

### An accessible cubic component can oppose the generated cubic output

**Example.** In the aligned sector, the leading fine output is proportional
to $-M_3P_Hx^3$, where $M_3=\mathbb E_W\zeta\alpha^3>0$. A target with a
positive pairing against $x^3$ then asks the network to remove a wrong-sign
contribution. This is different from requiring the target to be invisible
to current features.

Let $m_3=-\langle y_H,x^3\rangle_m<0$ and write $m=-m_3>0$.
The leading transport ODE, in time $\tau_1=t/W$, is exactly (1)–(2) with

$$
f(s)=ms,\qquad g(s)=ms/3.
\tag{9}
$$

Under (4), all reduced times satisfy

$$
\max_j\alpha_j(\tau_1)\le R,\qquad
Z(\tau_1)\le Z_0+2mpR^2\tau_1.
\tag{10}
$$

**Proof.** The leading constrained potential is $-m_3M_3/3$.
Its raw gradients are $\zeta f(\alpha^2)$ and $\alpha g(\alpha^2)$;
projection to $p=\text{constant}$ gives (1)–(2). Apply the lemma with
$B=mR^2$. $\square$

The result is not restricted to a ninth-degree target. It covers an
accessible cubic loading with a particular sign. After normalizing the
global output sign so $p>0$, that sign condition is $m_3p<0$ in the original
coordinates. The opposite sign can drive outward acquisition even at exact
coarse balance, as the one-atom solution in the
[persistence note](d34_state_dependent_persistence.md#1-an-accessible-target-component-need-not-produce-outward-motion)
shows. No sign or alignment claim about the archived sine states is made here.

### Generated-error correction can withstand a competing fifth-order load

**Example.** If the cubic target pairing vanishes, the preceding leading
force vanishes. The next clock retains both the network's generated cubic
error and a possible fifth-order target force. The issue is now whether
correction loses its advantage as the readouts and slopes evolve.

Assume $y_H$ is orthogonal to polynomials through degree three and the target
is odd. Set

$$
t_3=\|P_Hx^3\|_m^2>0,\qquad
\mu=\langle y_H,x^5\rangle_m,\qquad
M_r=\mathbb E_W\zeta\alpha^r.
$$

The next-order potential, in time $\tau_2=t/W^2$, is

$$
\mathcal E_2=\frac{t_3M_3^2}{18}-\frac{2\mu M_5}{15}.
\tag{11}
$$

Its constrained gradient equations have the form (1)–(2), with

$$
f(s)=\frac{t_3M_3}{3}s-\frac{2\mu}{3}s^2,\qquad
g(s)=\frac{t_3M_3}{9}s-\frac{2\mu}{15}s^2.
\tag{12}
$$

Under (4), the following conclusions hold.

1. If $\mu\le0$, the upper-edge bound $\max_j\alpha_j\le R$ holds for
   every reduced time, with
   $Z(\tau_2)\le Z_0+\frac{2pR^4}{3}(t_3p+2|\mu|)\tau_2$.
2. If $\mu>0$ and

$$
t_3p^3>4\mu R^2Z_0,
\tag{13}
$$

then the same upper-edge and alignment bounds hold through every $T<T_*$,
where

$$
C_Z=\frac{2t_3p^2R^4}{3},\qquad
T_*=
\frac{t_3p^3/(4\mu R^2)-Z_0}{C_Z},\qquad
Z(\tau_2)\le Z_0+C_Z\tau_2.
\tag{14}
$$

**Proof.** Differentiating (11) in the empirical-average particle metric
gives the two raw gradients in (12). If $\mu\le0$, $f$ is nondecreasing
and $0\le g\le f$. Conservation of $p$ and the sector give $M_3\le pR^2$,
so the lemma applies with $B=(t_3p+2|\mu|)R^4/3$.

For $\mu>0$, bootstrap the strict inequality

$$
t_3M_3>4\mu R^2.
\tag{15}
$$

It makes $f$ nondecreasing and $0\le g\le f$ on $[0,R^2]$. The lemma
then preserves the sector, and $f(R^2)\le t_3pR^4/3$ gives $Z'\le C_Z$.
Hölder's inequality and Cauchy–Schwarz give

$$
p^3=(\mathbb E_W\zeta\alpha)^3
\le M_3(\mathbb E_W\zeta)^2\le M_3Z.
\tag{16}
$$

Consequently, for every $\tau_2\le T<T_*$,

$$
t_3M_3(\tau_2)\ge\frac{t_3p^3}{Z_0+C_ZT}>4\mu R^2.
\tag{17}
$$

Condition (13) starts the bootstrap, and (17) prevents its first failure.
The lemma supplies continuation and coarse conditioning. This proves the
finite-time conclusion. $\square$

**Prediction.** The outward fifth loading competes with a generated-error
penalty whose strength has an evolving lower bound. Larger loading shortens
the guaranteed horizon. The theorem does not predict escape at $T_*$, and
does not assume that correction wins forever. Its useful advance is (17):
the coupled dynamics preserve the required advantage from initial data.

The permanently bounded cases concern the exact reduced equations. Their
readouts may grow without bound as time tends to infinity. They therefore
do not establish permanent trapping for the tanh network.

## 4. Theorem 2: couple persistence to tracking and population acquisition

**Purpose.** A reduced upper-edge theorem is useful only if small tracking
and omitted terms do not invalidate it immediately. We now state precisely
what a finite-time transfer requires. Its assumptions are spatial estimates
on a specified neighborhood, rather than observations about an unknown
future trajectory.

Let $\bar Y(\tau)=(\bar X_1,\ldots,\bar X_W,\bar d)$ be a reference solution
from Theorem 1 on $[0,T]$, using its appropriate clock $q=1$ or $q=2$.
Choose $T<T_*$ in the positive fifth-loading case. Equations (7)–(8) give
explicit bounds on its support and coarse conditioning.

Use two norms on labeled states:

$$
\|Y\|_2^2=\mathbb E_W|X_j|^2+|d|^2,\qquad
\|Y\|_\infty=\max\{\max_j|X_j|,|d|\}.
$$

Fix an outer tube of radius $r$ in the second norm around the reference
path. All bounds below must hold throughout this tube, including the
segments used in the comparisons. The reference has $\bar\beta=\bar d=0$;
the nearby actual state may have nonzero biases and arbitrary small
misalignment. Let $V_q$ denote the full corresponding reduced vector field.
Suppose its Lipschitz constants are $L_2,L_\infty$ on the tube.

For exact tanh training define, in orthonormal empirical coordinates,

$$
J=J_C,\quad H=J_H,\quad u=e_H,\quad K=JJ^T,\quad
\ell=K^{-1}JH^Tu,\quad F=H^Tu-J^T\ell,\quad z=e_C+\ell.
\tag{18}
$$

Thus the full gradient is $F+J^Tz$ and $JF=0$. Suppose the following
regional bounds have been established:

$$
K\succeq\kappa I,\quad \|J\|\le J_0,\quad
\|F\|\le f_0W^{-q},\quad \|D\ell\|\le d_0W^{-q}.
\tag{19}
$$

In addition, let $A_2,A_\infty$ bound the tracking velocity after rescaling:

$$
\|W^q\mathcal S_W(J^Tz)\|_\nu
\le W^q A_\nu\|z\|,\qquad \nu\in\{2,\infty\},
\tag{20}
$$

where $\mathcal S_W$ multiplies neuron coordinates by $\sqrt W$ and leaves
$d$ unchanged. One can take $A_2=J_0$; bounded rescaled coarse-Jacobian
columns give $A_\infty$. Finally require the regional Taylor estimate

$$
\|-W^q\mathcal S_WF-V_q(Y)\|_\nu\le C_\nu/W.
\tag{21}
$$

For $q=1$, these are the usual bounded-rescaled-parameter estimates for
the full fine force. For $q=2$, the target cancellations of Section 3 are
essential. The appendix explains the loaded derivative estimate in (19).
These conditions must be bounded or certified on the tube; a fit to one
trajectory is insufficient.

Assume $W^q\ge2d_0J_0/\kappa$, set $a_*=\kappa/2$, and define

$$
D_\nu(T)=e^{L_\nu T}\left[
\|Y(0)-\bar Y(0)\|_\nu+\frac{C_\nu T}{W}
+\frac{A_\nu\|z(0)\|}{a_*}
+\frac{A_\nu d_0f_0T}{a_*W^q}\right].
\tag{22}
$$

If $D_\infty(T)<r$, the actual gradient-flow trajectory exists in the tube
through physical time $W^qT$. Throughout that time,

$$
\|z(t)\|\le e^{-a_*t}\|z(0)\|
+\frac{d_0f_0}{a_*W^{2q}}.
\tag{23}
$$

For every acquisition threshold $\lambda_*$ with
$G:=\sqrt W\lambda_*/h-R>0$, the fraction of distinct neurons that
ever acquire it is bounded by

$$
\boxed{\quad p_{\rm ever}(W^qT;\lambda_*)
\le\min\left\{1,\frac{D_2(T)^2}{G^2}\right\}.\quad}
\tag{24}
$$

If $D_\infty(T)<G$, none acquires it. No monotonicity of the actual slopes
is required.

### Proof: derive tracking, then close the comparison

Differentiate $z=e_C+\ell$ along exact gradient flow. Since $JF=0$,

$$
\dot z=-Kz-D\ell[F+J^Tz].
\tag{25}
$$

Up to first exit from the tube, (19) gives

$$
D^+\|z\|\le-(\kappa-d_0J_0/W^q)\|z\|
+d_0f_0/W^{2q}.
$$

Comparison with this scalar linear equation proves (23). In particular,
small future tracking has been derived, rather than assumed. Its total
contribution in reduced time is bounded by

$$
\int_0^T W^q A_\nu\|z(W^q\tau)\|\,d\tau
\le\frac{A_\nu\|z(0)\|}{a_*}
+\frac{A_\nu d_0f_0T}{a_*W^q}.
\tag{26}
$$

The initial transient consumes an explicit allowance; it need not be
included in the slow mechanism itself.
Small initial tracking also requires the reference's coarse contribution
to be compatible with the target's coarse fit; $p$ cannot be chosen
independently of that requirement.

Subtract the reference ODE from the actual rescaled ODE. Equations
(20)–(21), the Lipschitz bound, and Grönwall's inequality give (22) up to
first exit. The strict inequality $D_\infty(T)<r$ prevents that exit.
Boundedness in the tube and coarse conditioning give continuation.

For the population claim we need a bound on distinct first hits, not only
the population at one instant. Set

$$
Q_2(T)^2=\mathbb E_W\sup_{0\le\tau\le T}
 |X_j(\tau)-\bar X_j(\tau)|^2
+\sup_{0\le\tau\le T}|d(\tau)-\bar d(\tau)|^2.
$$

Applying Minkowski's inequality to the integral equation, and then
Grönwall, gives $Q_2(T)\le D_2(T)$ by the same calculation. Each acquired
neuron must have a labelwise discrepancy at least $G$, because its
reference slope never exceeds $R$. Summing these squared discrepancies
gives $p_{\rm ever}G^2\le Q_2(T)^2$, proving (24). $\square$

**Meaning of the bound.** Theorem 1 supplies a reference whose structural
conditions persist. The tracking equation supplies a decaying transient and
a small forced floor. The approximation theorem supplies the remaining
perturbation budget. Together they bound actual population acquisition.
The threshold itself can lie outside the Taylor region: we only have to
prove that the actual trajectory stays inside a smaller region below it.

This is a fixed-$T$ reduced-time theorem. A width-dependent $T$, or a claim
of an enormous numerical training horizon, additionally requires controlling
the growing readout support, tube constants, and the exponential in (22).
No such numerical certification is claimed here.

## 5. Ordinary GD: an explicit version of the same transfer

**Purpose.** Gradient flow alone does not settle the user's deterministic
GD problem. Here is a sufficient discrete condition. It uses a stronger
initial tracking requirement to keep the statement short.
The requirement $z_0=O(W^{-2q})$ below is stronger than merely observing
that the tracking correction is small relative to the effective force.

On a padded tube as above, also suppose
$K\preceq K_{\max}I$ and $\|D^2z\|\le Q$ in physical parameter coordinates.
Take a fixed $\eta\le\eta_{\max}\le1/K_{\max}$ and choose $Z_*>0$ such that

$$
\|z_0\|\le Z_*/W^{2q},\qquad W^q\ge Z_*,\qquad
Z_*\ge\frac{d_0f_0+\eta_{\max}Q(f_0+J_0)^2/2}{a_*}.
\tag{27}
$$

Let $\Delta\tau=\eta/W^q$, $T=N\Delta\tau$. Let $C_{E,\nu}$ bound the
reference's one-step Euler defect divided by $\Delta\tau^2$ in norm $\nu$.
For example $C_{E,\nu}=L_\nu B_\nu/2$ suffices if the reference speed is
at most $B_\nu$. Replace (22) by

$$
D_\nu^{\rm GD}(T)=e^{L_\nu T}\left[
\|Y_0-\bar Y_0\|_\nu+\frac{C_\nu T}{W}
+\frac{A_\nu Z_*T}{W^q}
+C_{E,\nu}T\frac{\eta}{W^q}\right].
\tag{28}
$$

For segment containment, let $B_\infty$ bound $\|V_q\|_\infty$ throughout
the outer tube and impose

$$
D_\infty^{\rm GD}(T)+\Delta\tau
 [B_\infty+C_\infty/W+A_\infty Z_*/W^q]<r.
\tag{29}
$$

Then all $N$ GD updates remain in the tube, $\|z_n\|\le Z_*/W^{2q}$,
and (24) holds with $D_2^{\rm GD}(T)$ and acquisition measured at iterates.

**Proof.** Write $g_n=F_n+J_n^Tz_n$. Taylor expansion of the exact map $z$
along the proposed GD segment gives

$$
z_{n+1}=(I-\eta K_n)z_n-\eta D\ell_n[g_n]+r_n,
\qquad \|r_n\|\le\tfrac12\eta^2Q\|g_n\|^2.
$$

Under the induction hypothesis and $W^q\ge Z_*$,
$\|g_n\|\le(f_0+J_0)/W^q$. Consequently

$$
\|z_{n+1}\|\le(1-\eta a_*)\|z_n\|
+\frac{\eta}{W^{2q}}
 [d_0f_0+\eta Q(f_0+J_0)^2/2].
\tag{30}
$$

Condition (27) closes the tracking induction. Comparing each actual step
with the exact reference step gives (28) by discrete Grönwall. Condition
(29) ensures that the next entire update segment is in the region where
the Taylor bounds hold, closing both inductions without presuming segment
containment. Applying the same argument to labelwise maxima over earlier
steps proves the distinct-ever version of (24). $\square$

**Width-dependent delay.** Consider a family with bounded initial reference
support, $p$ bounded away from zero, and width-uniform tube constants. Fix
$T$ in the proved reduced interval, assume an initial discrepancy
$O(W^{-1})$, and impose the discrete tracking conditions above. Then
$D_\infty^{\rm GD}(T)=O(W^{-1})$. For sufficiently large $W$, the first-exit
condition holds and

$$
\max_{n\le\lfloor TW^q/\eta\rfloor}\max_j\lambda_{j,n}
\le\frac{h}{\sqrt W}\left[R+O(W^{-1})\right].
\tag{31}
$$

Thus the proved delay is at least $\lfloor TW^q/\eta\rfloor$ updates for
any fixed normalized threshold above this bound: the $W$ clock for the
signed accessible-cubic mechanism, or the $W^2$ clock under the stated
target cancellations. If $h=\Theta(W^{-1})$, the right side is
$O(W^{-3/2})$. This is a conditional rate statement; without certified
constants it does not assign a practical update count to an archived run.

The theorem is for ordinary Euclidean GD. Adam's preconditioner and stored
momenta change both the compensation and the boundary signs. Its observed
saturation motivates a separate qualitative test; it is not a consequence
of this theorem.

## 6. What has been established, and what must be checked next?

The new result is the structural bootstrap, especially (16)–(17). We do
not assume that the fine correction remains small. We prove that the
readout–slope sector and an ordering of competing loads persist, and use
that ordering to prevent the upper edge from advancing. Tracking is then
controlled by its own stable equation. This closes a real part of the
coupled argument missing from a force bound at one checkpoint.

The assumption audit now separates a failed sector condition from transfer
conditions that remain to be established:

| Condition | Why it enters | Empirical status or remaining obligation |
|---|---|---|
| Common product sign and $\lvert\zeta_j\rvert\ge\lvert\alpha_j\rvert$ after sign normalization | Preserves the boundary sign and prevents compensating readouts from reversing it. | Failed in all 223 audited states; at least 23.7% of neurons per state have the wrong product sign. Hidden biases also violate the zero-bias sector. |
| Cubic sign $m_3p<0$ | Accessible cubic force removes a wrong-sign contribution. | A favorable sign alone does not restore the failed alignment assumption. |
| Cubic cancellation and the fifth-load margin (13) | Makes generated correction dominate an outward target force for a derived horizon. | A positive computed margin outside the sector does not justify $T_*$; the proof also uses alignment and parity. |
| Coarse conditioning and bounded support on a padded tube | Makes compensation, Taylor errors, and tracking derivatives controllable. | Numerical constants for (19)–(21), rather than only pointwise fitted rates. |
| Small initial tracking and a successful first-exit inequality | Transfers reduced persistence to actual training. | A certified finite-time acquisition bound in (24), with ordinary-GD conditions if appropriate. |

Neuron sign symmetry $(\alpha,\zeta)\mapsto(-\alpha,-\zeta)$ allows
nonpositive slopes to be reoriented before checking the sector. A common
output/target sign reversal normalizes $p>0$. Neither operation removes
mixed signs of $\alpha_j\zeta_j$. Those violations are substantive.

The sector is substantially different from the audited states. The median
negative-product fraction is 44.6%, and the median rescaled hidden-bias RMS
is 1.465. A rare-exception argument is not supported. The next extension
must retain mixed signs and slope–bias–readout moments, then derive bounds
on their contribution to the evolving shared compensation. Likewise, the opposite
cubic sign requires a slow-passage argument, not this maximum principle.
The scalar outward-delay result in the
[coupled-ODE note](d34_coupled_ode_mechanisms.md#2-readout-compensation-can-make-outward-acquisition-slow)
is a useful starting point, but its heterogeneous extension is still open.

The broad empirical observation can therefore remain correct even where
this particular sufficient mechanism does not apply. The note proves a
nonempty, genuinely coupled class of persistent states, together with an
explicit path from such a state to a population acquisition theorem. The
empirical audit shows that this class does not contain the observed states;
it does not invalidate the broader observation of slow scale acquisition.

## Appendix: why the tracking estimates are structural

For $q=1$, bounded rescaled parameters and bounded target norm give
$\|H\|\le H_0/W$, $\|DH\|\le H_1/W$. Assume also
$\|DJ\|\le J_1$ and $\|u\|\le E$ on the specified tube. Since $F$ is an
orthogonal projection of $H^Tu$,

$$
f_0=H_0E,\qquad C_\ell=J_0H_0E/\kappa
$$

give $\|F\|\le f_0/W$ and $\|\ell\|\le C_\ell/W$. Differentiating
$K\ell=JH^Tu$ gives the exact identity

$$
D\ell[w]=K^{-1}\left\{
(DJ[w])F+
J[(DH[w])^Tu-(DJ[w])^T\ell+H^THw]\right\}.
\tag{32}
$$

Thus, for $W\ge1$,

$$
\|D\ell\|\le d_0/W,\qquad
d_0=\kappa^{-1}
 [J_1f_0+J_0(H_1E+J_1C_\ell+H_0^2)].
\tag{33}
$$

These constants come from the model and a region. They do not assume the
realized force remains small. Fine quantities are independent of the
output bias; its evolution is nevertheless included in the tracking and
comparison equations.

For $q=2$, one must use a sharper *loaded* estimate. Orthogonality through
degree three annihilates the target pairing with the cubic feature term
and its parameter derivatives. The fine network output is $O(W^{-1})$;
its product with $H=O(W^{-1})$ is $O(W^{-2})$. The first surviving target
term comes from the quintic expansion and has the same order. Hence
$F,\ell,D\ell=O(W^{-2})$ on a bounded conditioned rescaled domain. The
unloaded norm $\|H\|$ need not improve. Expanding one further order gives
(21) for $V_2$, as in the
[next-order transport derivation](d34_rescaled_transport_model.md#5-the-next-clock-generated-error-competes-with-fourth--and-fifth-degree-target-load).
Constants still have to be bounded on the chosen tube. Applying the
$q=2$ clock to a generic target with a nonzero cubic pairing would discard
its leading drive and would be invalid.
