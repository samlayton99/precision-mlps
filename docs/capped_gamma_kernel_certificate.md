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
