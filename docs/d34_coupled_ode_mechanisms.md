# When coupled readout and geometry dynamics make acquisition slow

A small slope force explains a small step. To explain a long delay, we must
show why subsequent learning does not quickly create a larger, sustained
outward force. This note derives three such conclusions from specified ODEs:
readout compensation can impede escape even when the target pushes outward;
generated-error correction can progressively weaken the rates while parameters
keep changing; and a population can spend only a limited amount of motion on
correction before a weaker competing drive determines its progress.

These are theorems about the stated reduced systems. They do not assume that
their entire future fine force stays small. Their substantive assumptions
concern constrained geometry, target loadings, or the conditioning of a
generated-error map. The evidence motivates studying these conditions but
does not establish all of them for the trained networks. In particular, a
single balanced channel is an illustrative sector, not an established
reduction of a heterogeneous network.

**Notation.** Scalar physical coordinates and rescaled particle coordinates
are kept distinct, including their time variables.

| Symbol | Meaning |
|---|---|
| $a,c$ | Physical slope and readout of a scalar channel; hidden bias is fixed to zero in that example. |
| $P_H$, $y_H=P_Hy$ | Empirical orthogonal projection away from $1,x$, and the full fine target. |
| $\langle\cdot,\cdot\rangle_m$ | Empirical mean inner product on a symmetric grid, with $|x_i|\le1$. |
| $p$ | Conserved affine contribution in the displayed scalar or odd-sector model; its normalization is specified locally. |
| $X_j=(\alpha_j,\beta_j,\zeta_j)=\sqrt W(a_j,b_j,c_j)$ | Rescaled slope, bias and readout of particle $j$. |
| $\lambda_j=h|a_j|=h|\alpha_j|/\sqrt W$ | Normalized scale, with $h=2/N_{\rm ref}$ and physical width $W$. |
| $\Phi$, $J$ | Independent generated-error coefficients and their Jacobian tangent to the coarse-output constraint. |
| $\varepsilon G_{\rm drv}$ | Competing tangent drive in Theorem 3; its smallness must come from the model, not the observed plateau. |

## 1. What the perturbations tell us to retain

**Example.** In the six-target width-177 physical-perturbation panel, the
scalar amplification model has median slope-response error near 88–89%.
A frozen full Hessian has about 93% error. Evolving the anchored fine-force
quintic model reduces the error to about 14% in both cohorts. Its late pure
sine predictions nevertheless remain poor or fail. These retrospective
comparisons support evolving the readouts and geometry together; they do not
validate every further scalar reduction. The
[persistence report](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_persistence/README.md#3-physical-perturbations-expose-what-scalar-matching-omits)
retains the failures and finite-kick tracking disturbances.

**Mechanism to explain.** At coarse balance the effective fine field includes
the compensation required to preserve the fitted coarse outputs. That
compensation is present even when coarse disequilibrium is zero. It changes
which joint readout and slope motions are available. Generated errors and
accessible target components then compete within this constrained motion.

The three results below concern rates and accumulated motion. Theorem 1 gives
a slow passage without requiring contraction. Theorem 2 bounds passage through
a scale interval from the signed competition in the ODE. Theorem 3 bounds
correction travel and subsequent drift for a heterogeneous population under
a conditioning condition. A long observed plateau need not be an equilibrium.
The appendix treats an attracting scale only as a contrasting special case.

## 2. Readout compensation can make outward acquisition slow

### Example: keeping a coarse fit makes slope motion expensive

Consider the affine constraint $ac=p\ne0$ on a positive-slope branch. A
change in slope requires $dc/da=-p/a^2$. The squared Euclidean length of
that joint parameter displacement is therefore

$$
da^2+dc^2=\left(1+\frac{p^2}{a^4}\right)da^2.
\tag{1}
$$

If $E(a)$ is the loss restricted to this curve, orthogonally constrained
gradient flow obeys

$$
\dot a=-\frac{E'(a)}{1+p^2/a^4}.
\tag{2}
$$

Thus a favorable loss derivative need not produce rapid acquisition. At
$a^2\ll|p|$, the readout adjustment dominates the length of a scale-changing
step. Equation (2) follows by projecting the ambient gradient onto the
tangent $(1,-p/a^2)$; its squared norm supplies the denominator. Both
coordinates evolve. No readout refitting operation has been inserted.

The affine constraint is a useful approximation, but the same mechanism
exists with an exact tanh coarse constraint. The following theorem makes
that version explicit.

### Theorem 1: an exact constrained-tanh exit delay

Let $v=\langle x^2\rangle_m>0$ and define the linear regression coefficient
of one feature and its fine part by

$$
\ell(a)=\frac{\langle x,\tanh(ax)\rangle_m}{v},\qquad
k(a)=P_H\tanh(ax).
$$

Fix $p\ne0$ and constrain $c\ell(a)=p$. This keeps the orthonormal linear
output coefficient equal to $\sqrt v\,p$. The scalar fine output, restricted
loss, and induced metric are

$$
f_H(a)=\frac{p\,k(a)}{\ell(a)},\qquad
E(a)=\tfrac12\|f_H(a)-y_H\|_m^2,\qquad
g(a)=1+\frac{p^2\ell'(a)^2}{\ell(a)^4}.
\tag{3}
$$

Consider $\dot a=-E'(a)/g(a)$ from $0<a_0<r$. Put

$$
\sigma_r=\operatorname{sech}^2r,\qquad
C_r=\left(\|y_H\|_m+\frac{|p|r^2}{3\sigma_r}\right)
\frac{4|p|}{3\sigma_r^2},\qquad
B_r=\frac{C_r}{p^2\sigma_r^2}.
$$

Then, throughout $0<a\le r$,

$$
|\dot a|\le B_ra^5.
\tag{4}
$$

If the trajectory first reaches $r$ at time $T_r$, then

$$
\boxed{\quad
T_r\ge\frac{a_0^{-4}-r^{-4}}{4B_r}.
\quad}
\tag{5}
$$

If it never reaches $r$, take $T_r=\infty$. The estimate allows reversals
and inward motion. It does not require monotone expansion or a small target
projection onto cubic features.

**Proof.** For $0<a\le r$, the derivative of $\ell$ and $|x_i|\le1$ give

$$
\sigma_ra\le\ell(a)\le a,\qquad
\sigma_r\le\ell'(a)\le1.
$$

The identities $P_Hx=0$, $|\tanh u-u|\le|u|^3/3$, and
$|\operatorname{sech}^2u-1|=\tanh^2u\le u^2$ imply

$$
\|k(a)\|_m\le a^3/3,\qquad \|k'(a)\|_m\le a^2.
$$

The quotient rule consequently yields

$$
\|f_H(a)\|_m\le\frac{|p|a^2}{3\sigma_r},\qquad
\|f_H'(a)\|_m\le\frac{4|p|a}{3\sigma_r^2}.
$$

Hence $|E'(a)|\le C_ra$, while $g(a)\ge p^2\sigma_r^2/a^4$,
proving (4). Integrate
$d(a^{-4})/dt=-4a^{-5}\dot a\ge-4B_r$ until the first hit to obtain (5).
The opposite inequality shows that $a$ cannot reach zero in finite time
while (4) holds. On every finite interval before exit, $a$ stays positive
and $c=p/\ell(a)$ stays finite, so ordinary ODE continuation applies.

The projected-gradient derivation is exact: the curve tangent is
$(1,c'(a))$, its squared norm is $g(a)$, and its pairing with the loss
gradient is $E'(a)$. Coarse loss is constant on the curve. On an odd target
and symmetric grid the zero-bias sector is also invariant. For a general
target the theorem describes the explicitly restricted zero-bias system;
it does not assert that free biases stay zero.

**Prediction.** At fixed $p$, target and outer slope $r$, decreasing the
starting slope strengthens the lower bound as $a_0^{-4}$. The suppression
comes from a necessary joint readout motion and the affine cancellation in
the fine feature. If $p$ varies with initialization or width, its dependence
must be substituted into $B_r$ before interpreting this as a rate law.

### The existing transport ODE gives explicit clocks

The leading odd one-atom model uses rescaled coordinates,
$\alpha\zeta=p$, and time $\tau=\eta n/W$. With
$m_3=-\langle y_H,x^3\rangle_m$, its constrained potential and motion are

$$
E_1(\alpha)=-\frac{m_3p}{3}\alpha^2,\qquad
\alpha'=\chi\frac{\alpha^5}{\alpha^4+p^2},\qquad
\chi=\frac{2m_3p}{3}.
\tag{6}
$$

For $\chi>0$, direct integration gives the exact first-hit time

$$
\tau_A=\frac1\chi\left[
\log\frac A{\alpha_0}
+\frac{p^2}{4}(\alpha_0^{-4}-A^{-4})\right],\qquad A>\alpha_0.
\tag{7}
$$

For $\chi<0$, slopes instead contract with
$\alpha(\tau)\sim[p^2/(4|\chi|\tau)]^{1/4}$, while
$|\zeta|=|p|/\alpha$ grows. These claims follow by separating (6), using
$d\tau=(\alpha^{-1}+p^2\alpha^{-5})d\alpha/\chi$.

The next-order odd model makes the competition with generated error explicit.
Assume an odd target orthogonal through degree three, so the zero-bias
sector is invariant. More generally the additional condition
$\langle y_H,x^4\rangle_m=0$ suffices at this order. Put

$$
t_3=\|P_Hx^3\|_m^2,\qquad
D=\frac{t_3p^2}{18}-\frac{2p\langle y_H,x^5\rangle_m}{15}.
$$

On its clock $\tau_2=\eta n/W^2$, the one-atom potential is
$E_2=D\alpha^4$ and

$$
\alpha'=-\frac{4D\alpha^7}{\alpha^4+p^2}.
\tag{8}
$$

If $D>0$, generated-error correction wins: $\alpha$ decreases and
$\alpha(\tau_2)\sim[p^2/(24D\tau_2)]^{1/6}$. If $D=0$, this reduced
motion vanishes. If $D<0$, the target wins, but the exact exit time is still

$$
(\tau_2)_A=
\frac{\alpha_0^{-2}-A^{-2}}{8|D|}
+\frac{p^2(\alpha_0^{-6}-A^{-6})}{24|D|}.
\tag{9}
$$

To prove these statements, integrate
$d\tau_2=(\alpha^{-3}+p^2\alpha^{-7})d\alpha/(4|D|)$ on the outward
branch, and reverse the sign on the contracting branch. The small-slope
asymptotic follows from $d(\alpha^{-6})/d\tau_2\to24D/p^2$.
The expanding polynomial ODE can blow up at finite reduced time; (9) is a
finite-threshold formula, not a global boundedness assertion.

Equations (7) and (9) exhibit a race between an outward target advantage
and the compensating readout motion. An outward advantage need not remove
the delay. Conversely, an inward sign can persist while the readout grows,
eventually leaving a bounded-parameter approximation domain.

**Clock convention.** Theorem 1 uses the Euclidean metric of one physical
channel and physical gradient-flow time. Equations (6)–(9) use the
empirical-average particle metric and the rescaled potentials in the
[transport note](d34_rescaled_transport_model.md). A tied network of $W$
identical physical neurons has metric $W(da^2+dc^2)$, so its equation for
the total physical loss has an additional factor $1/W$. These conventions
must not be interchanged. Multiplying (7) or (9) by $W/\eta$ or
$W^2/\eta$ gives an update clock only over an interval on which the
reduced-to-GD approximation is controlled.

### What survives for heterogeneous neurons?

The constraint $a^Tc=p$ preserves only the aggregate affine contribution;
it does not preserve each $a_jc_j$. Let $r_a=\|a\|_2>0$ and let $P$ be
the orthogonal tangent projector for this aggregate constraint. Then

$$
\|P\nabla r_a\|_2^2
=1-\frac{p^2}{r_a^2(r_a^2+\|c\|_2^2)}
=\frac{r_a^2+\|c\|_2^2(1-\varrho^2)}{r_a^2+\|c\|_2^2},
\qquad \varrho=\frac{p}{r_a\|c\|_2}.
\tag{10}
$$

The first expression also covers $c=0$; $\varrho$ is then undefined.
The proof projects $(a/r_a,0)$ off the normal $(c,a)$. The same formula
holds with consistently normalized empirical-average inner products.
Along constrained flow of a general loss,

$$
|\dot r_a|\le\|P\nabla L\|_2\,\|P\nabla r_a\|_2.
\tag{11}
$$

Thus readout dominance suppresses radial mobility only with sufficiently
strong slope–readout alignment. For a radial restricted loss $E(r_a)$,
the exact equation is $\dot r_a=-E'(r_a)\|P\nabla r_a\|_2^2$;
the squared mobility factor cannot be used for a general nonradial loss.

This identifies the missing population hypothesis. Other neurons can
redistribute the coarse fit and bypass a single-channel bottleneck.
Persistence of an aligned sector, or a separate bound on this redistribution,
requires proof. Neither a large readout RMS nor the observed cancellation in
matched perturbation responses establishes it. Also, a bound on radial
motion alone does not count distinct neurons that acquire scale at different
times.

## 3. Correction can progressively slow the rates without reaching equilibrium

### Example: slopes slow while readouts keep moving

For $D>0$, equation (8) gives, as $\tau_2\to\infty$ in that reduced ODE,

$$
\alpha\asymp\tau_2^{-1/6},\qquad
|\zeta|\asymp\tau_2^{1/6},\qquad
\|\psi_3\|_m\asymp\tau_2^{-1/3},\qquad
|\alpha'|\asymp\tau_2^{-7/6},\quad
\|X'\|\asymp\tau_2^{-5/6}.
\tag{12}
$$

Here $\asymp$ states the power law up to positive constants; assume
$p\ne0$ and $\|P_Hx^3\|_m>0$. The error law follows from
$\psi_3=-p\alpha^2P_Hx^3/3$. The velocity laws follow by substituting
the slope asymptotic into (8) and differentiating $\zeta=p/\alpha$.
Total slope travel is finite, but full parameter path length diverges because
the readout keeps growing. There is no limiting finite parameter equilibrium.
The force becomes small as a consequence of the coupled dynamics.

The pure generated-error case, with zero fifth-degree target pairing and
$D=\|P_Hx^3\|_m^2p^2/18$, exposes the changing correction rate. Put
$\Phi=-p\alpha^2\|P_Hx^3\|_m/3$, with tangent Jacobian $J$ under
the metric (1). Then

$$
JJ^*=\frac{4\|P_Hx^3\|_m^2p^2\alpha^6}
{9(\alpha^4+p^2)}\sim\frac1{3\tau_2},\qquad
\Phi'=-JJ^*\Phi.
$$

This identity follows by squaring $d\Phi/d\alpha$ and dividing by
$1+p^2/\alpha^4$. Correction changes the geometry in a way that weakens
its own sensitivity: the normal relaxation rate tends to zero. A uniform
positive relaxation rate would miss this mechanism. The statement holds for
the exact reduced ODE; growing readouts limit its eventual transfer to tanh.

With an outward target advantage, (9) instead gives a long passage whose
duration grows as $\alpha_0^{-6}$ at fixed coefficients. Thus both shrinking
and outward-moving branches can exhibit a long rate bottleneck. The next
result includes competing accessible target terms without asking whether
the trajectory approaches an equilibrium.

### Theorem 2: target and generated-error coefficients bound passage time

Fix $p\ne0$ and two empirical fine feature shapes
$\phi_3=-P_Hx^3/3$, $\phi_5=2P_Hx^5/15$. On the affine-balanced
curve $ac=p$, define $s=a^2$ and

$$
f_H(s)=p(s\phi_3+s^2\phi_5),\qquad
E(s)=\tfrac12\|f_H(s)-y_H\|_m^2.
$$

This is the full squared loss of the restricted cubic–quintic feature model,
not the leading homogeneous potential of the transport expansion. Its exact
constrained flow is

$$
\dot s=-M(s)E'(s),\qquad M(s)=\frac{4s^3}{s^2+p^2}.
\tag{13}
$$

Direct differentiation gives

$$
\begin{aligned}
E'(s)={}&-p\langle y_H,\phi_3\rangle_m
+[p^2\|\phi_3\|_m^2-2p\langle y_H,\phi_5\rangle_m]s\\
&+3p^2\langle\phi_3,\phi_5\rangle_m s^2
+2p^2\|\phi_5\|_m^2s^3.
\end{aligned}
\tag{14}
$$

Write $H(s)=-E'(s)$ using the explicit cubic polynomial (14). Permit a
time-dependent perturbation of its driving coefficient:

$$
\dot s=M(s)[H(s)+\delta(t,s)],\qquad
|\delta(t,s)|\le\delta_0\quad(0\le s\le S).
\tag{15}
$$

Assume $\delta$ is continuous in time and locally Lipschitz in $s$.
For $0<s_0<S$, define the coefficient bound

$$
K_S=\max_{0\le s\le S}[H(s)+\delta_0]_+.
$$

If $K_S>0$, every first passage from $s_0$ to $S$ satisfies

$$
\boxed{\quad
T_{s_0\to S}\ge
\frac{\log(S/s_0)+\frac{p^2}{2}(s_0^{-2}-S^{-2})}{4K_S}.
\quad}
\tag{16}
$$

If $K_S=0$, there is no outward passage. All coefficients of $H$ are
specified by the target, feature shapes and conserved $p$; its maximum is
found from the endpoints and the real roots of $H'$ in the interval. Thus
$K_S$ is a structural coefficient calculation, not an assumed bound on
future observed force. The theorem remains valid with any independently
derived upper bound on this maximum.

**Proof.** The increasing function

$$
\mathcal G(s)=\tfrac14\log s-\frac{p^2}{8s^2}
\quad\text{satisfies}\quad \mathcal G'(s)=1/M(s).
$$

Along (15), $d\mathcal G(s(t))/dt=H(s(t))+\delta(t,s(t))\le K_S$.
Integration up to the first hit proves (16), even if the slope first moves
inward or reverses repeatedly. When $K_S=0$, $\mathcal G$ cannot increase.
The polynomial $H$ and the bounded perturbation also give
$|\dot s|\le C s^3$ near zero. Integrating
$d(s^{-2})/dt\le2C$ prevents finite-time arrival at zero. This establishes
continuation along the positive branch through every finite pre-exit interval.

For an unperturbed branch with $H(s)>0$ on $[s_0,S]$, the stronger exact
expression is

$$
T_{s_0\to S}=\int_{s_0}^S\frac{ds}{M(s)H(s)}.
\tag{17}
$$

This exhibits both contributions to the delay: the readout metric makes
$M(s)$ small near zero, while generated-error correction can reduce the
remaining outward advantage $H(s)$ as the slope grows. A narrow interval
where $0<H(s)\le\varepsilon$ gives an additional passage delay proportional
to $1/\varepsilon$ by the same integral, without an equilibrium in that
interval. The condition is checked on the polynomial coefficients, not
inferred from a slow trajectory.

**What the coefficients mean.** Generated cubic output supplies
$p^2\|\phi_3\|^2s$ in $E'$, opposing expansion. Cubic target loading
supplies the constant outward drive when its signed pairing is positive.
Quintic target loading and the cross term can change their competition.
The theorem tracks this signed balance instead of assuming that correcting
a lower mode always dominates. A perturbation allowance must be justified
in the displayed coefficient units; a small additive slope-force error
need not remain small after division by $M(s)$.

The polynomial must approximate the actual features on
$|a|\le\sqrt S$, and the readout–geometry restriction must remain
appropriate, before the bound explains a tanh trajectory. The anchored
heterogeneous quintic predictor tested in the campaign is a different model;
its successful forecasts do not establish this scalar sector or a bound
on $\delta$. The appendix states stronger conditions for an actual attracting
scale as a contrast, rather than interpreting observed slow motion as one.

There is an additional width issue. For a rescaled one-atom state with
$\alpha\zeta=p$, the small-parameter fine output is
$W^{-1}p\alpha^2\phi_3+W^{-2}p\alpha^4\phi_5+\cdots$.
The first two loss orders, multiplied by $W^2$, are

$$
E_W(\alpha)=-b_W\alpha^2+D\alpha^4,\qquad
b_W=Wp\langle y_H,\phi_3\rangle_m,
\quad D=\tfrac12p^2\|\phi_3\|_m^2-p\langle y_H,\phi_5\rangle_m.
\tag{18}
$$

When $D,b_W>0$, their candidate balance lies at
$\alpha_*^2=b_W/(2D)$. A width-uniform bounded-rescaled balance therefore
requires weak cubic loading, or another identified scaling of these
coefficients. With fixed nonzero cubic loading and fixed $p,D$, the proposed
balance moves to $\alpha_*=O(\sqrt W)$, outside the bounded-rescaled
expansion. Equation (18) does not justify continuing that expansion to the
balance. In particular, a generic sine target does not automatically satisfy
the requirements for a small-scale plateau.

**Prediction.** Within a validated scalar sector, reducing generated-error
penalties increases $H$ where those terms oppose expansion and decreases
the passage time in (17). Changing target loading can reverse that sign.
The test is a predicted change in rates and passage times, with the coupled
readout response retained; return to an equilibrium is not required.

## 4. A heterogeneous population can have a limited correction-travel budget

### Example: a slowly shrinking error need not be a replenished equilibrium

In the archived degree-nine analysis, the hard mode replenishes only
$3.77\times10^{-6}$ to $3.63\times10^{-5}$ of the generated-error
self-relaxation energy rate. The frozen two-error model's forced-equilibrium
norm is $6.30\times10^{-8}$ to $1.58\times10^{-7}$, whereas the starting
quadratic/cubic residual norm is $9.45\times10^{-4}$ to $1.34\times10^{-3}$.
Those observations support a slowly relaxing transient far above its floor,
rather than an already balanced equilibrium. The
[underlying audit](d34_coarse_balance_stagnation_details.md#10-testing-what-keeps-the-force-small)
also explains the frozen-model limitations.

Can correction itself guarantee that the future force becomes small? The
next theorem gives a sufficient condition while allowing the feature
Jacobian to evolve. Its cost is a regional conditioning assumption, which
must be checked rather than inferred from the small observed motion.

### Theorem 3: stable correction implies finite travel and a residence time

Let $\mathcal M$ be a regular coarse-output constraint manifold of the
finite-particle model. Give it the induced metric

$$
\|\dot Z\|^2=\frac1W\sum_{j=1}^W|\dot X_j|^2+|\dot d|^2,
\qquad Z=(X_1,\ldots,X_W,d).
$$

Let $\Phi:\mathcal M\to\mathbb R^k$ collect independent generated
lower-mode coefficients, $J=d\Phi|_{T\mathcal M}$, and let $J^*$ denote
the adjoint for this metric. Consider

$$
\dot Z=-J^*\Phi+\varepsilon G_{\rm drv}(Z,t),\qquad\varepsilon\ge0,
\tag{19}
$$

where $G_{\rm drv}$ is tangent to $\mathcal M$. The first term is the
constrained gradient of $\frac12\|\Phi\|^2$; the second is the competing
target or model drive. Assume smoothness sufficient for a unique local
solution and continuation on compact subsets.

Choose an open neighborhood $\mathcal U$ of $Z_0$ whose closure is compact
inside the regular part of $\mathcal M$. Let
$d_{\mathcal U}>0$ be the intrinsic distance from $Z_0$ to its boundary.
Suppose, throughout that closure and the times considered,

$$
JJ^*\succeq\kappa I_k,\qquad
\|J\|\le L_\Phi,\qquad
\|G_{\rm drv}\|\le B_0,\qquad
\|JG_{\rm drv}\|\le B_1,\qquad \kappa>0.
\tag{20}
$$

Write $e_0=\|\Phi(Z_0)\|$. Define

$$
\begin{aligned}
\mathcal A(T)={}&\frac{L_\Phi e_0}{\kappa}(1-e^{-\kappa T})
+\varepsilon B_0T\\
&+\frac{\varepsilon L_\Phi B_1}{\kappa}
\left[T-\frac{1-e^{-\kappa T}}{\kappa}\right].
\end{aligned}
\tag{21}
$$

If $\mathcal A(T)<d_{\mathcal U}$, the solution stays in $\mathcal U$
through $T$ and obeys

$$
\|\Phi(Z(t))\|\le e_0e^{-\kappa t}
+\frac{\varepsilon B_1}{\kappa}(1-e^{-\kappa t}),\qquad
\int_0^T\|\dot Z\|\,dt\le\mathcal A(T).
\tag{22}
$$

In particular, put $A_0=L_\Phi e_0/\kappa$ and
$B_*=B_0+L_\Phi B_1/\kappa$. If $A_0<d_{\mathcal U}$, then:

- For $\varepsilon=0$, the solution exists for all time, has total path
  length at most $A_0$, and converges to a point of $\Phi^{-1}(0)$.
- For $\varepsilon B_*>0$, it remains confined for every
  $T<(d_{\mathcal U}-A_0)/(\varepsilon B_*)$.

If $\varepsilon B_*=0$, the same infinite-time conclusion holds; there
is no nonzero drive on the region. No hypothesis asserts smallness of the
future generated error or its correction force.

**Proof.** The chain rule on the constraint manifold gives the exact identity

$$
\frac d{dt}\Phi=-JJ^*\Phi+\varepsilon JG_{\rm drv}.
\tag{23}
$$

No frozen Jacobian is used. For $e=\|\Phi\|$, (20) implies the upper
Dini-derivative bound $D^+e\le-\kappa e+\varepsilon B_1$, including
at $e=0$. Scalar comparison proves the first part of (22). Since
$\|\dot Z\|\le L_\Phi e+\varepsilon B_0$, integration gives (21).

Apply these inequalities up to a hypothetical first exit. Any path from
$Z_0$ to $\partial\mathcal U$ has length at least $d_{\mathcal U}$,
contradicting the strict bound on $\mathcal A(T)$. Compact continuation
then extends the solution through $T$. When $\varepsilon=0$ and
$A_0<d_{\mathcal U}$, the uniform remaining margin and precompactness give
global continuation. Finite total length makes $Z(t)$ Cauchy; its limit is
inside the regular neighborhood, and exponential decay of $\Phi$ places
the limit in $\Phi^{-1}(0)$. Finally,
$\mathcal A(T)\le A_0+\varepsilon B_*T$ gives the stated residence time.

**Population consequence.** For rescaled slopes and $r>r_0\ge0$, let
$p_{\rm ever}(r,T)$ count distinct particles that reach $|\alpha_j|\ge r$
at any time through $T$. Then

$$
p_{\rm ever}(r,T)\le
\min\left\{1,\;p_0(|\alpha|>r_0)
+\frac{\mathcal A(T)^2}{(r-r_0)^2}\right\}.
\tag{24}
$$

Indeed, define each particle's slope path length
$\ell_j(T)=\int_0^T|\dot\alpha_j|dt$. Minkowski's inequality gives
$[W^{-1}\sum_j\ell_j(T)^2]^{1/2}\le\int_0^T\|\dot Z\|dt$.
Every newly acquired particle initially below $r_0$ has
$\ell_j(T)\ge r-r_0$. Counting those particles proves (24), including
particles that cross at different times. With
$\lambda=h|\alpha|/\sqrt W$, the additional fraction is

$$
\frac{h^2\mathcal A(T)^2}{W(\lambda_*-\lambda_0)^2}.
\tag{25}
$$

For zero drive, this allowance is bounded independently of time. The theorem
therefore improves the existing generated-energy action bound, whose travel
allowance grows as $\sqrt T$, when its stronger conditioning and closure
conditions hold.

### What must make the competing drive small?

For the next-order transport potential
$\frac12\|\psi_3\|^2-\langle y_H,\psi_5\rangle$, take $\Phi$ to be
the independent coefficients of $\psi_3$. Targets orthogonal through
degree five remove the target term exactly at this order. Nearby loading
classes can provide an explicit small parameter from the target coefficients;
Taylor remainders can contribute another such term on a controlled domain.
One must bound the gradient of that target term, not merely its scalar
pairing at the starting state.

The hypothesis $JJ^*\succeq\kappa I$ concerns only independent generated
channels. Applying it to the entire fine sample-space vector would be
rank-deficient by construction. It says that correction remains effective
in each retained error direction as geometry changes. A small $\kappa$
permits slow relaxation but also enlarges the total-travel allowance; its
effect cannot be discarded when claiming a plateau.

A generic sine target need not have small competing drive in this
decomposition. Moreover, the width-1409 interventions show negligible
relaxation effects relative to geometry feedback over their tested window.
They do not establish this correction-dominated regime. Theorem 3 is a
specific mechanism to recognize where its hypotheses hold, not a universal
explanation for all wide states.

**Prediction.** Perturb a generated-error direction while preserving coarse
constraints. Uniform normal conditioning predicts restoration toward an
$O(\varepsilon)$ error floor and a bounded correction displacement.
A perturbation tangent to $\Phi^{-1}(0)$ instead tests the remaining drive.
The existing kicks matched other quantities and do not directly verify these
normal and tangent response predictions. A positive eigenvalue at one
checkpoint also does not establish the regional condition (20).

### When correction loses sensitivity: an algebraic rate law

The uniform lower bound in (20) is not necessary for a rate explanation.
The explicit slowing example in Section 3 has a normal Jacobian that tends
to zero as correction proceeds. The following version derives algebraic
rates from that degeneration rather than assuming a persistent positive gap.

For zero competing drive, suppose on a declared region

$$
k_d\|\Phi\|^\nu I\preceq JJ^*\preceq K_d\|\Phi\|^\nu I,
\qquad 0<k_d\le K_d,\quad \nu>0.
\tag{23a}
$$

For $e_0>0$, throughout the solution's interval in that region,

$$
(e_0^{-\nu}+\nu K_dt)^{-1/\nu}
\le\|\Phi(t)\|\le
(e_0^{-\nu}+\nu k_dt)^{-1/\nu}.
\tag{23b}
$$

**Proof.** Equation (23) gives
$-K_de^{1+\nu}\le e'\le-k_de^{1+\nu}$ for $e=\|\Phi\|>0$.
Differentiate $e^{-\nu}$ and integrate. The resulting positive lower bound
also prevents a finite-time zero. Furthermore, the exact identity
$\|\dot Z\|^2=\Phi^TJJ^*\Phi$ gives

$$
\sqrt{k_d}\,e^{1+\nu/2}\le\|\dot Z\|
\le\sqrt{K_d}\,e^{1+\nu/2}.
$$

If the region persists, these imply error of order $t^{-1/\nu}$ and full
speed of order $t^{-1/2-1/\nu}$. No equilibrium assumption was made.
A finite-window path bound is the integral of the displayed upper speed
with the upper envelope from (23b). For $\nu=3$, it is explicitly

$$
\mathcal A_3(T)=\frac{2\sqrt{K_d}}{k_d}
\left[(e_0^{-3}+3k_dT)^{1/6}-e_0^{-1/2}\right].
\tag{23c}
$$

The same first-exit argument applies when $\mathcal A_3(T)<d_{\mathcal U}$,
and (24)–(25) apply with this path allowance. Its growth is only $T^{1/6}$;
the added population allowance grows as $T^{1/3}$ on the established
interval. A diverging allowance does not prove acquisition, but it does
distinguish slow continuing motion from finite total motion or trapping.

For the pure scalar correction example, (23a) with $\nu=3$ follows
directly from its geometry. With $t_3=\|P_Hx^3\|_m^2$,

$$
\frac{JJ^*}{|\Phi|^3}
=\frac{12}{|p|\sqrt{t_3}(\alpha^4+p^2)}.
\tag{23d}
$$

On $0<\alpha\le A$ this lies between the explicit positive constants
$12/[|p|\sqrt{t_3}(A^4+p^2)]$ and $12/(|p|^3\sqrt{t_3})$.
Thus the slow law is a consequence of how the generated feature changes
with slope and readout. It is not a fitted decay exponent or an assumption
that the force stays small. The scalar solution exists globally in its own
noncompact parameter curve; transfer to tanh still needs control of readout
growth. No comparable regional power law has yet been established for the
heterogeneous empirical states or generic sine.

## 5. A useful obstruction: the leading cubic model cannot explain every plateau

**Rate implication.** A sine target can have a nonzero cubic loading while
remaining far from its desired scale. In the following leading model, a
bounded-rescaled domain cannot persist for all time, even though passage
through it may be very slow. Any rate theorem using that domain must account
for eventual exit rather than treating it as invariant forever.

**Proposition.** In the leading odd transport model, suppose $m_3\ne0$ and
the conserved affine slope $p=\mathbb E[\alpha\zeta]\ne0$. There is no
compactly supported stationary gradient-flow distribution. For any fixed
finite particle system, there is also no forward trajectory whose particle
coordinates remain uniformly bounded for all reduced time.

**Proof.** Put

$$
\mu=\frac{\mathbb E[\zeta^2\alpha^2+\alpha^4/3]}
{\mathbb E[\zeta^2+\alpha^2]}>0.
$$

The characteristic equations are

$$
\alpha'=m_3\zeta(\alpha^2-\mu),\qquad
\zeta'=m_3\alpha(\alpha^2/3-\mu).
\tag{26}
$$

The gradient-flow energy identity makes a compactly supported stationary
distribution satisfy both velocities equal to zero almost everywhere.
The second equation requires $\alpha=0$ or $\alpha^2=3\mu$.
In either case the first equation requires $\zeta=0$, contradicting
$\mathbb E[\alpha\zeta]\ne0$.

For fixed finitely many particles, a uniformly bounded forward trajectory
would stay in a compact set. The denominator in $\mu$ is bounded below by
$2|p|$, so the field is regular there. Its quartic potential is bounded
below on that set, and energy dissipation supplies times at which the
velocity norm tends to zero. A convergent subsequence would yield a
stationary configuration with the same nonzero $p$, which was just excluded.

This is a statement about the leading ODE, not a blowup or nonconvergence
theorem for tanh GD. It allows a very long delay, and a particle distribution
can change substantially before acquiring the desired slope scale. It rules
out proving permanent compact confinement from this truncation under these
loadings. Higher-order balance, changing residuals, or a different regime is
needed to explain an eventual equilibrium.

## 6. Which assumptions does the evidence actually support?

The existing experiments informed these theorem choices. This is a
retrospective synthesis, not a new prospective validation or a new numerical
measurement. The table distinguishes evidence for an explanatory component
from evidence for a complete theorem's hypotheses.

**Assumption ledger: support and unresolved conditions for the three mechanisms.**

| Condition or interpretation | Relevant evidence | Present conclusion |
|---|---|---|
| Effective fine dynamics dominate ordinary-GD motion after the transient. | The [cross-function force audit](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/analysis/heldout/README.md) measures small direct corrections and signed accumulated correction motion. | Supports the reduced object; sampled or signed diagnostics do not prove every future tracking allowance. |
| An evolving coupled operator improves physical-response prediction. | The [physical response comparison](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_persistence/README.md#3-physical-perturbations-expose-what-scalar-matching-omits) favors evolving quintic dynamics over frozen operators. | Motivates coupled theorems. It does not establish an invariant scalar sector or a constant coarse contribution per neuron. |
| Readout-dominated, aligned geometry suppresses radial mobility. | Frozen large-slope [readout fits](../results/checkpoint_D_optimizers/expD34_readout_race/readout_scale/README.md) recover approximately $O(h)$ coefficients for tested sine dictionaries. The [clone interventions](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/README.md) find little change from readout-only mobility compensation and stronger recovery from geometry compensation. | The endpoint scale observation is real, but a universal readout-dominated explanation is unsupported. Persistence of the alignment and limited redistribution required by (10) has not been established. |
| Generated lower-mode error can oppose scale expansion. | The [degree-nine signed-force and intervention audit](d34_coarse_balance_stagnation_details.md#7-what-the-audit-and-continuations-establish) attributes contraction to that effective force. | Supports the signed competition in that regime; it does not establish a fixed scalar shape sector throughout Theorem 2's interval. |
| An observed plateau is an already replenished equilibrium. | The [degree-nine persistence audit](d34_coarse_balance_stagnation_details.md#10-testing-what-keeps-the-force-small) places actual generated error far above the frozen model's forced floor. | This interpretation is unsupported there; slow transient correction is the supported alternative. |
| Normal correction stays conditioned on a neighborhood. | Late checkpoint audits often show net force decay, but wider feedback interventions show tiny relaxation effects. | Neither observation verifies $JJ^*\succeq\kappa I$ or Theorem 3's path margin. Strong restoration cannot be assumed for the wide panel. |
| The normal correction rate degenerates with generated-error amplitude. | The scalar correction ODE gives the exact power law (23d). | Proved in that sector; a corresponding regional law for heterogeneous trajectories remains unverified. |
| A fixed small-parameter approximation describes late sine. | Late sine response errors are large, with a failed quintic family retained. | A broad late-sine plateau theorem is not established. Keep target loading and regime validity explicit. |

The cancellation between coupling and changing curvature in the matched
physical response is also informative. In one degree-five case, two terms
of size about $4.21\times10^{-6}$ sum to about $1.10\times10^{-8}$.
That supports retaining signed evolving geometry. Those are derivatives of
the response to a deliberately matched perturbation; they are not the
original slope force, and do not prove the readout alignment or generated-mode
coercivity required above.

## 7. What these results add, and what transfers to actual GD

The mechanistic statements have different logical forms. Theorem 1 derives
slow escape from the metric of coarse-preserving readout adjustment. Theorem
2 derives a passage-time bound from explicit target and generated-error
coefficients. Its motivating correction example slows algebraically while
readouts keep growing. Theorem 3 derives small future generated error and
limited correction travel from the evolving correction dynamics. Their
proofs do not require an accurate forecast of each neuron's future motion.
Its algebraic variant also allows sensitivity to weaken with the generated
error, producing slow continuing motion rather than finite total motion.

The useful remaining questions are consequently specific: does a trained
population preserve enough alignment for the first mechanism; does its
signed target/error competition slow outward passage for the second; or do
its generated channels satisfy a regional conditioning or amplitude-dependent
sensitivity law for the third? A negative answer narrows the mechanism. It is not repaired by
assuming the observed small force persists.

For ordinary tanh GD, the transfer obligations are separate:

1. The chosen reduced model must remain valid on the region used in its
   proof, including any readout growth and redistribution. Taylor validity
   cannot be inferred solely from small slopes.
2. Coarse disequilibrium, omitted-mode effects when a truncated basis is
   used, and discretization must fit the relevant boundary or travel margin.
   Coarse disequilibrium generally moves normal to the constraint manifold;
   it cannot silently be included in the tangent $G_{\rm drv}$ of (19).
   A manifold comparison or a separately controlled constraint deviation is
   required. The full-complement effective flow itself has no omitted modes.
3. Reduced time must be converted using that model's clock and metric.
   A threshold $\lambda_*>0$ corresponds to
   $|\alpha|=\lambda_*\sqrt W/h$, often well outside a bounded-rescaled
   domain. A proof of slow exit from that domain supplies a lower bound on
   acquisition time, not permission to extrapolate its initial rate all the
   way to the threshold.

The [persistence note](d34_state_dependent_persistence.md) supplies conditional
comparison and acquisition bounds for this last transfer. The role of the
present note is to identify structural reasons its small-motion regime might
last. The generic small-parameter width law remains a conditional delay of
order $W/\eta$ updates on a closed interval; it is not upgraded to a
$W^{5/2}/\eta$ acquisition lower bound by these scalar examples.

No new claim about Adam follows from Euclidean constrained-gradient flow.
Its effective metric and moment history change the movement cost; the
[Adam intervention results](d34_adam_moment_results.md) motivate a separate
coupled analysis of actual normalized updates. Likewise, small final readouts
in a construction do not impose a competing optimization objective or make
that representation necessary for every accurate network.

These results formalize routes to slow acquisition and the conditions that
distinguish them. They do not yet establish that the heterogeneous sine or
the full target panel satisfies a single persistent bottleneck sufficient
to make practical acquisition infeasible.

## Appendix: an attracting scale is a stronger, separate possibility

An equilibrium is not needed for the passage theorems. For completeness, the
same scalar polynomial model admits a precise stronger statement. Suppose

$$
E'(0)<0,\qquad E'(S)>0,\qquad
\kappa_s:=p^2\|\phi_3\|_m^2-2p\langle y_H,\phi_5\rangle_m
-6p^2|\langle\phi_3,\phi_5\rangle_m|S>0.
\tag{27}
$$

Then $E''\ge\kappa_s$ on $[0,S]$, so $E'$ has a unique zero
$s_*\in(0,S)$. Since $M(s)>0$ for $s>0$, every unperturbed solution
from $0<s_0\le S$ stays between $s_0$ and $s_*$ and converges to $s_*$.
With $M_{\min}$ the positive minimum of $M$ on that interval,
differentiating $\frac12(s-s_*)^2$ gives

$$
|s(t)-s_*|\le|s_0-s_*|e^{-\kappa_sM_{\min}t}.
$$

If $E'(0)\ge0$ and the same curvature condition holds, every solution
from $0<s_0\le S$ instead decreases to zero; smoothness of the vector field
prevents finite-time arrival. Its readout then diverges asymptotically.

There is also a robust finite-interval version. Choose
$0<s_-<s_*<S$ and add $w(t,s)$ to (13), continuous in time and locally
Lipschitz in $s$. If

$$
|w(t,s)|\le\delta<
\min\{-M(s_-)E'(s_-),\;M(S)E'(S)\}
\quad\text{on }[s_-,S],
$$

both boundaries point inward, so first exit proves confinement. The positive
lower boundary keeps the readout finite; confinement does not imply
convergence to the original equilibrium under time-dependent forcing.
These stronger conditions have not been established for the archived
heterogeneous trajectories and are not the working explanation for their
observed rate slowdown.
