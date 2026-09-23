# A target-dependent learning delay for every capped dictionary

This theorem concerns a fixed finite sample geometry and raw Euclidean readout
coordinates. It turns a slope-cap certificate into a necessary number of GD
updates without observing a training trajectory or diagonalizing the particular
dictionary being trained. Whether its best computable bound is sharp on D36 is
an experimental question; exact spectral prediction alone does not answer it.

| Symbol | Meaning |
|:--|:--|
| $x_i,c_j$ | Fixed samples and feature centers, including halo centers |
| $\Gamma$ | Maximum absolute slope; every neuron may have a different slope |
| $y$ | Nonzero target vector, divided by $\sqrt m$ |
| $K,L$ | Output kernel $JJ^T$ and its largest eigenvalue |
| $\chi$ | Normalized GD step, $\eta L$, with $0<\chi<1$ |
| $v,\delta$ | Unit witness and its absolute overlap with $y/\|y\|$ |
| $X,\beta$ | PSD certificate and its trace |
| $B,U$ | Uniform necessary updates and an admissible upper witness |

## Uniform certificate

Set $b=\mathbf1/\sqrt m$ and

$$
a_j(g)=m^{-1/2}\big(\tanh(g(x_i-c_j))\big)_{i=1}^m,
\qquad J=[b,a_1(\gamma_1),\ldots,a_W(\gamma_W)].
$$

For a unit vector $v$, choose $X\succeq0$. Suppose

$$
(v^Tb)^2-b^TXb+
\sum_{j=1}^W\sup_{0\le g\le\Gamma}
\left\{(v^Ta_j(g))^2-a_j(g)^TXa_j(g)\right\}\le0.
\tag{C}
$$

Then every $|\gamma_j|\le\Gamma$ satisfies

$$
v^TKv\le\operatorname{tr}(XK)\le L\operatorname{tr}X.
\tag{1}
$$

The first inequality follows by summing (C) for the actual slopes. Negative
slopes negate columns and preserve both quadratic forms. The second follows
from $0\preceq K\preceq LI$ and $X\succeq0$. This proof allows arbitrary,
independently selected slopes under the cap; it does not require common slopes,
translation invariance, or monotonic eigenvalues as gamma changes.

The certificate is convex in $X$ for a fixed witness. Restricting
$X=UQU^T$, $Q\succeq0$, is allowed for computational economy. It restricts the
certificate search, not the ambient sample space or the dictionary family.
One must retain the actual trace $\operatorname{tr}(UQU^T)$ if $U$ is not
exactly orthonormal.

## Target energy and necessary updates

Let $\beta=\operatorname{tr}X<1$, $\delta=|v^Ty|/\|y\|$, and let $P_s$ project
onto eigenvalues of $K/L$ no larger than $s$, where $\beta<s<1$. Equation (1)
gives $\|(I-P_s)v\|^2\le\beta/s$. Resolving the target into its component along
$v$ and its orthogonal complement, or equivalently using the angle between
$v$ and the slow subspace, gives

$$
\frac{\|P_sy\|^2}{\|y\|^2}\ge
p(s):=\left[
\delta\sqrt{1-\beta/s}-\sqrt{1-\delta^2}\sqrt{\beta/s}
\right]_+^2.
\tag{2}
$$

For zero-start GD on $\frac12\|J\theta-y\|^2$, the residual is
$(I-\eta K)^ny$. Therefore

$$
E(n)^2\ge p(s)(1-\chi s)^{2n},
\qquad
n_\epsilon\ge
\left\lceil\frac{\log(\sqrt{p(s)}/\epsilon)}
{-\log(1-\chi s)}\right\rceil
\quad\text{when }p(s)>\epsilon^2.
\tag{3}
$$

Multiple witnesses give lower bounds on the same target spectral cumulative
distribution. Take their maximum and the monotone envelope; do not add their
masses. Placing each guaranteed mass increment at its largest allowed rate,
and remaining mass at rate one, gives a combined error lower bound. This follows
because $(1-\chi s)^{2n}$ decreases with $s$.

When $v=y/\|y\|$, Jensen gives the stronger direct result

$$
E(n)^2\ge(1-\chi\beta)^{2n},
$$

because $s\mapsto(1-\chi s)^{2n}$ is convex on $[0,1]$. The implementation only
uses this shortcut when the supplied witness is exactly the supplied target
before normalization. All bounds also hold for an actual step with
$\eta L\le\chi$; decreasing the step preserves the obstruction.

## Explicit nonvacuous baseline

Let $v$ be the unit mean-zero component of $y$, and let
$\sigma_x^2=m^{-1}\sum_i(x_i-\bar x)^2$. The variance identity and the
$g$-Lipschitz property of tanh give

$$
\|(I-bb^T)a_j(g)\|^2\le g^2\sigma_x^2.
$$

Thus $X=W\Gamma^2\sigma_x^2bb^T$ satisfies (C). This gives
$\beta=W\Gamma^2\sigma_x^2$. Taking $s=4\beta/\delta^2<1$ yields
$p(s)\ge\delta^2/4$ and an $\Omega(\Gamma^{-2})$ delay for fixed geometry,
fixed $\chi$, and $\epsilon<\delta/2$. This small-cap result need not be
informative at the larger D36 caps; the optimized certificate is tested there.

For an exactly solvable control, use $x=\{-1,1\}$, all centers zero, and the
odd target. Then $K=bb^T+Svv^T$ with $S=\sum_j\tanh^2\gamma_j$. The fastest
normalized target rate under the cap is $\min(W\tanh^2\Gamma,1)$. In the regime
$W\tanh^2\Gamma<1$, the certificate and direct Jensen bound attain the exact
optimal integer learning time.

## Numerical certification and evidence roles

`cap_certificate.py` first optimizes a finite-grid candidate. It interprets the
saved factor $P$ as exact binary data and defines $X_0=PP^T$, making positivity
algebraic. Arb interval arithmetic then bounds every scalar supremum in (C)
over the entire cap interval. It bounds tanh, projections, derivatives, traces,
normalization, and target overlap. Adaptive subdivision may stop with a loose
but valid upper bound.

For cancellation-sensitive witnesses the checker can use a Taylor polynomial
of order four or six, with an interval remainder over the complete subinterval.
The normalized derivatives of $f(g)=\tanh(gd)$ follow from
$f'=d(1-f^2)$: if $a_k=f^{(k)}(g)/k!$, then
$a_1=d(1-a_0^2)$ and
$a_{k+1}=-d\sum_{i=0}^ka_i a_{k-i}/(k+1)$ for $k\ge1$.
Linear projection and polynomial convolution give the corresponding
coefficients of each scalar quadratic form. Point coefficients retain the
projection cancellations; interval coefficients bound the final derivative
remainder. Endpoint quadratic bounds also retain the sign of curvature.
Taking the minimum of these valid upper enclosures changes their sharpness,
not the certificate condition.

If the resulting constraint upper bound is $r>0$, replace $X_0$ by
$X=X_0+rbb^T$. The bias contribution falls by $r$, and every other supremum can
only fall. Thus the repaired certificate is feasible, at the price of adding
$r$ to the trace. Report this repair separately: a large repair exposes a loose
numerical enclosure or an infeasible proposed certificate.

The final CDF and integer obstruction are also evaluated with outward error
bounds. A claimed lower bound $B$ is checked by proving that the error lower
bound at update $B-1$ exceeds the tolerance. High precision without an enclosure
is a diagnostic, not a certificate.

The initial implementation certifies the real tanh dictionary on the saved
binary sample coordinates, centers, and target values. Nominal rational grids
and analytic target evaluations are distinct inputs. FP64 optimizer evolution
is also distinct from exact-arithmetic GD; independent forecasts and precision
checks measure that difference.

The campaign compares $B$ to a verified admissible upper witness $U$ for
$\inf_{|\gamma_j|\le\Gamma}n_\epsilon$. A factor-two bracket is the declared
research target, not a presupposed result. Selected dictionaries are frozen
before ordinary GD; their hidden slopes are never updated during that timing.
Training error supports an optimization statement, not a generalization claim.
Any comparison to independent-grid error retains its separate evidence role.

## Resolvent refinement of the learning-time conversion

The CDF conversion does not use all of the relationship between the witness
and target. A second inequality improves the necessary time while preserving
the same uniform certificate (1). Set $H=K/L$ and $\widehat y=y/\|y\|$.
For every $t>0$, weighted Cauchy–Schwarz gives

$$
\delta^2\le
\big(v^T(H+tI)v\big)
\big(\widehat y^T(H+tI)^{-1}\widehat y\big)
\le(\beta+t)\widehat y^T(H+tI)^{-1}\widehat y.
$$

For $z\ge0$, define

$$
a(n,t,z)=\min_{0\le s\le1}
\left\{(1-\chi s)^{2n}-\frac{z}{t+s}\right\}.
$$

Applying this scalar inequality in the eigenbasis of $H$ yields

$$
\boxed{E(n)^2\ge
a(n,t,z)+\frac{z\delta^2}{\beta+t}.}
\tag{4}
$$

Any $n$ for which the right side exceeds $\epsilon^2$ is excluded for every
dictionary under the cap. Optimize $t,z$ to propose a stronger result, then
independently enclose the scalar minimum over the entire interval $[0,1]$.
For an interval $[\ell,u]$, the elementary lower enclosure

$$
(1-\chi u)^{2n}-\frac{z}{t+\ell}
$$

is valid because the two terms have opposite monotonicities. Adaptive
subdivision and outward arithmetic establish (4) without relying on the
optimizer or a grid of eigenvalues. The implementation takes the maximum of
this result, the CDF bound, and the direct-target Jensen bound when applicable.

This is a proof improvement, not a fitted time multiplier. The parameters
$t,z$ use only $\beta,\delta,\chi,\epsilon$; no trained iterate is an input.
It does not by itself tighten the upstream uniform estimate of $\beta$.

The witness can also be searched jointly with $X$. Allow an unnormalized $v$
in (C), which still gives $v^THv\le\operatorname{tr}X$. The variational identity
for the inverse gives

$$
\widehat y^T(H+tI)^{-1}\widehat y\ge
2\widehat y^Tv-\operatorname{tr}X-t\|v\|^2.
$$

Maximizing this concave expression subject to (C) and $X\succeq0$ is a convex
optimization problem: each feature constraint is convex in $v$ and affine in
$X$. The experimental solver searches a finite-dimensional basis and slope
grid, then normalizes the proposed direction, re-solves its fixed-direction
certificate, and applies the same independent continuum checker. Absolute
solver tolerances can become large after normalizing a small witness, so the
unpolished joint objective is never reported as a proved bound.

An alternative convex search maximizes $\widehat y^Tv$ subject to (C),
$X\succeq0$, $\operatorname{tr}X\le b_0$, and $\|v\|\le1$, for a chosen
budget $b_0>0$. With $v=\sqrt{b_0}Uc$ and $X=b_0UQU^T$, an orthonormal
search basis gives the well-scaled constraints $\operatorname{tr}Q\le1$
and $\|c\|^2\le1/b_0$. A nonzero output is normalized, polished, and
checked in exactly the same way. This changes the witness search; it adds no
assumption about the actual dictionary or its training trajectory.

There are two distinct sources of slack. Condition (C) bounds a directional
curvature uniformly over the capped family. Equations (2)–(4) then turn that
directional information into a target learning delay. Improving the second
step cannot recover information already discarded by the first. For example,
when $0<\beta<\delta^2$, the abstract operator
$H=(\beta/\delta^2)\widehat y\widehat y^T$ satisfies the same directional
constraint for a witness with overlap $\delta$ and learns the target at rate
$\beta/\delta^2$. Consequently these two scalar inputs alone cannot prove a
necessary time larger than

$$
\left\lceil\frac{\log\epsilon}
{\log(1-\chi\beta/\delta^2)}\right\rceil.
$$

This is an information limit for a single $(\beta,\delta)$ pair. The abstract
operator need not be realizable by any tanh dictionary, so it is not an
admissible upper witness for the capped-family optimization problem.
If the normalization must have $\|H\|=1$ exactly, add $ww^T$ for a unit
vector $w$ orthogonal to both $v$ and $\widehat y$ (available in dimension
at least three). This leaves the directional constraint and target evolution
unchanged while supplying a unit eigenvalue.

The cap also gives useful order relations without claiming that every pair
of dictionaries is ordered. If $\Gamma_1\le\Gamma_2$, their admissible
families are nested. A bound proved at $\Gamma_2$ is valid at $\Gamma_1$,
and a dictionary executed at $\Gamma_1$ is an admissible upper witness at
$\Gamma_2$. Thus one may replace computed bounds by the nonincreasing hull
$B(\Gamma)=\max_{G\ge\Gamma}B_G$ and upper witnesses by
$U(\Gamma)=\min_{G\le\Gamma}U_G$. A comparison
$B(\Gamma_1)>U(\Gamma_2)$ proves a uniform obstruction at the smaller cap
relative to that larger-cap witness. Neither this nesting nor the theorem
says that lowering each individual slope always slows every target.

On reflection-symmetric samples and centers, a witness of definite parity
permits replacing $X$ by $(X+RXR)/2$, where $R$ reverses sample order. This
preserves the trace and feasibility of (C). Factoring the even and odd parts
separately makes the symmetry exact in the saved binary data, allowing the
checker to reuse the scalar supremum for centers $c$ and $-c$. The neuron slopes
remain independent; this computational reduction assumes no symmetry of the
actual slope vector.
