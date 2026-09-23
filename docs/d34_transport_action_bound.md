# Generated-error energy limits how much mass can ever acquire scale

The [next-order transport model](d34_rescaled_transport_model.md#5-the-next-clock-generated-error-competes-with-fourth--and-fifth-degree-target-load)
has a stronger consequence than its instantaneous parameter-energy bounds:
the energy spent correcting generated error limits the fraction of particles
that can **ever** reach a distant slope during a finite window. This is a
theorem for that continuous reduced model. Its transfer to ordinary discrete
tanh GD remains conditional.

**Notation.**

| Symbol | Meaning |
|---|---|
| $X=(\alpha,\beta,\zeta)$ | Rescaled particle parameters $\sqrt W(a,b,c)$. |
| $\tau_2$, $T$ | Reduced-model time, corresponding to $\eta n/W^2$, and its terminal value. |
| $A_2$, $\ell_0^{(2)}$ | Projected effective gradient and output-bias response from the transport model; $X'=-A_2$, $d'=\ell_0^{(2)}$. |
| $\mathcal E_0$ | Initial generated-error energy $\frac12\|\psi_3(0)\|_m^2$. |
| $\rho_0$ | Initial particle distribution, used to count distinct particle labels. |
| $h=2/N_{\rm ref}$ | Construction normalization, distinct from the actual neuron count $W$. |

## 1. Example: a large target error need not supply a large movement budget

Suppose the fine target is orthogonal to every empirical polynomial through
degree five. It then has zero pairing with the quintic output $\psi_5$.
In the next-order model the potential is simply
$\mathcal E=\frac12\|\psi_3\|_m^2\ge0$. The large unresolved target
error does not enter this model's available energy. Dissipation gives

$$
\mathcal E'=-\mathbb E_{\rho_\tau}|A_2|^2
-(\ell_0^{(2)})^2,
\qquad
\int_0^T\mathbb E|A_2|^2\,d\tau\le\mathcal E_0.
\tag{1}
$$

Thus substantial movement requires spending generated-error energy even
when the target error remains large. Individual slopes may still expand.
The theorem concerns how many can travel far, not a universal inward sign.

## 2. Theory: count particles that cross at any time

**Proposition.** On any interval $[0,T]$ where the reduced characteristics
and (1) hold, let $r>r_0\ge0$. Then

$$
\rho_0\!\left(\sup_{0\le\tau\le T}|\alpha(\tau)|\ge r\right)
\le \rho_0(|\alpha(0)|>r_0)
+\frac{T\mathcal E_0}{(r-r_0)^2}.
\tag{2}
$$

The right side can be capped at one. For finite particles, the probability
is their fraction, and each particle is counted once regardless of when
or how often it crosses.

**Proof.** For each particle label, the characteristic equation and
Cauchy–Schwarz give

$$
\sup_{\tau\le T}|\alpha(\tau)-\alpha(0)|^2
\le T\int_0^T |(A_2)_\alpha|^2\,d\tau.
$$

A particle initially at or below $r_0$ that reaches $r$ must have displacement
at least $r-r_0$. Integrate the preceding inequality over initial labels,
use (1), and add the initial tail. Unlike a bound on each time's marginal
distribution, this argument controls the entire path of each label.

For normalized slopes $\lambda=h|\alpha|/\sqrt W$, (2) becomes

$$
\operatorname{fraction}_{\rm ever}(\lambda_*,T)
\le\operatorname{fraction}_{0}(\lambda>\lambda_0)
+\frac{h^2T\mathcal E_0}{W(\lambda_*-\lambda_0)^2},
\qquad \lambda_* > \lambda_0\ge0.
\tag{3}
$$

The initial tail is zero when every initial slope is at most $\lambda_0$.
An informative exclusion window follows from the initial tail, generated
energy, and threshold gap; no experiment endpoint is privileged.

## 3. The finite-particle model continues without finite-time singularity

The preceding statement presumes a well-defined flow. For fixed finite $W$
with equally weighted particles and empirical inputs $|x_i|\le1$,
nonzero conserved affine slope supplies a sufficient global condition.
Write $C_1=\mathbb E\zeta\alpha\ne0$, $v=\langle x^2\rangle_m\in(0,1]$,
and $M(\tau)=(\mathbb E|X(\tau)|^2)^{1/2}$. From (1),

$$
M(\tau)\le M(0)+\sqrt{T\mathcal E_0},\qquad
|d(\tau)-d(0)|\le\sqrt{T\mathcal E_0}
\quad(\tau\le T).
\tag{4}
$$

The first inequality follows by integrating the particle velocity in the
label-space $L^2$ norm; the second uses the other dissipative term in (1).
For the affine coarse matrix $K$ from the transport model, put
$Z=\mathbb E\zeta^2$, $A=\mathbb E\alpha^2$, $B=\mathbb E\beta^2$,
and $C=\mathbb E\alpha\beta$. Cauchy–Schwarz implies $C^2\le AB$, so

$$
\det K=v[(1+Z+B)(Z+A)-C^2]
\ge v(Z+A)\ge2v|C_1|,
\qquad \operatorname{tr}K\le1+2M^2.
$$

Consequently, throughout any finite interval,

$$
\lambda_{\min}(K)\ge
\frac{2v|C_1|}{1+2(M(0)+\sqrt{T\mathcal E_0})^2}>0.
\tag{5}
$$

At fixed $W$, (4) also bounds every particle by $\sqrt W M(\tau)$.
The rational vector field is therefore smooth on a compact neighborhood
of the trajectory with an invertible coarse matrix. The local solution
extends past every proposed finite terminal time. This proves global
existence for this finite-particle model when $C_1\ne0$; zero $C_1$ is
not claimed to imply failure. These estimates do not give width-uniform
particle support or a global-existence theorem for arbitrary measures.

## 4. Prediction and the remaining transfer requirement

Equation (3) is mechanism-specific: it charges generated lower-mode energy,
not the full unresolved target loss. A fifth-degree target generally retains
the competing pairing $-\langle y_H,\psi_5\rangle_m$, so its potential
need not be nonnegative and (1) cannot be assigned the same budget.

**Conditional ordinary-GD corollary.** Suppose the initial particles agree
and a proved comparison gives
$\max_{j,n\le N}|\alpha^{\rm GD}_{j,n}-\alpha^{\rm reduced}_j(\tau_n)|
\le\varepsilon_T$, where $\tau_n=\eta n/W^2$ and $T=\eta N/W^2$.
A bound on the full rescaled particle discrepancy also suffices. Then

$$
\operatorname{fraction}^{\rm GD}_{\rm ever}(\lambda_*,N)
\le\operatorname{fraction}_{0}(\lambda>\lambda_0)
+\frac{h^2T\mathcal E_0}
{W(\lambda_*-\lambda_0-h\varepsilon_T/\sqrt W)^2},
\tag{6}
$$

provided the denominator's gap is positive. Indeed, a discrete GD crossing
implies a reduced-model crossing of the lowered threshold at that same grid
time, hence a continuous-path crossing to which (2) applies. The allowance
must include tanh truncation, coarse tracking, and discrete-time error on a
closed conditioned domain. Continuous dissipation alone does not prove a
descent inequality for explicit Euler or ordinary GD.

Finite-particle existence in Section 3 does not discharge those comparison
conditions: its coordinate bound grows with $W$ and may leave the uniformly
small-parameter domain required by the tanh approximation. No numerical
evaluation or certified acquisition horizon is claimed here. The result
provides a nonlinear transport barrier conditional on a controlled transfer,
and identifies the generated energy and path error that such a calculation
must bound.
