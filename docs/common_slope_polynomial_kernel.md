# Technical note: common-slope kernels and readout acquisition times

This note proves a quantitative acquisition-time theorem for a frozen tanh
dictionary with a shared hidden slope. A polynomial approximation preserves
the kernel's signed coefficients and cross-couplings. An explicit error bound
then transfers the polynomial model's computable learning curve to the
original readout GD dynamics. The result gives both necessary and sufficient
update counts, without fitting a training trajectory.

The analytic version uses a gamma-dependent feature approximation envelope.
A sharper version also evaluates how the discarded kernel acts on the target's
retained components. In the completed fixed-geometry study, its selected
timing intervals closely enclose executed GD hits, and an independent interval
calculation certifies the primary endpoints. This is a fixed-common-slope
result; sharp uniform guarantees over an entire interval of shared slopes
remain a separate question.

The [gamma-factorized follow-up](gamma_factorized_readout.md) now makes the
mechanism explicit in the retained operator itself: a fixed geometric kernel
is filtered by the known gamma-dependent multiplier. It applies the same
transfer inequalities below to that construction and compares its acquisition
intervals with these polynomial results.

**Notation.** Norms of vectors are Euclidean; matrix norms without a subscript
are spectral norms. The target and dictionary include the same empirical
normalization.

| Symbol | Meaning |
|---|---|
| $m,W$ | Training samples and hidden features; there are $W+1$ readout parameters. |
| $x_i,c_j,\gamma$ | Fixed sample points, centers, and common hidden slope. |
| $\theta_n$ | Raw readout parameters after $n$ GD updates, initialized at zero. |
| $J_\gamma,y$ | Normalized feature matrix and target vector. |
| $K_\gamma=J_\gamma J_\gamma^T$ | Output-space kernel governing residual dynamics. |
| $D$ | Degree used to approximate every feature and retain the kernel. |
| $k,P_k$ | A separate target-difficulty cutoff and projector onto sampled degree-$k$ polynomials. |
| $e_D(\gamma)$ | Explicit best-polynomial feature approximation envelope. |
| $\widetilde J,\widetilde K$ | Polynomial synthesis matrix and its output kernel. |
| $\alpha_\ell,\widetilde\lambda_\ell$ | Target loading and eigenvalue in a retained kernel mode. |
| $E_n,n_\epsilon$ | Relative residual norm and first update with residual at most $\epsilon$. |
| $R,\Delta_D,d_{D,n}$ | Kernel difference, its analytic norm bound, and residual-error radius. |

## 1. Model, optimization clock, and the exact recurrence

Fix $x_i\in[-1,1]$, real centers $c_j$, and a common slope $\gamma\ge0$.
Only the readout coefficients are trained:

$$
f_\theta(x)=\theta_{\rm bias}+\sum_{j=1}^W\theta_j\tanh(\gamma(x-c_j)),
\qquad
J_{i0}=\frac1{\sqrt m},\quad
J_{ij}=\frac{\tanh(\gamma(x_i-c_j))}{\sqrt m},\quad
y_i=\frac{c(x_i)}{\sqrt m}.
$$

Here $c(x)$ denotes the target function, while $c_j$ denotes a center. The
empirical loss and zero-initialized, full-batch GD updates are

$$
\mathcal L(\theta)=\frac12\|J\theta-y\|^2,
\qquad
\theta_{n+1}=\theta_n-\eta J^T(J\theta_n-y),
\qquad \theta_0=0.
$$

Use the residual $r_n=y-J\theta_n$. Multiplying the parameter update by $J$
gives the exact output recurrence

$$
r_{n+1}=(I-\eta K)r_n,\qquad
r_n=(I-\eta K)^ny,\qquad K=JJ^T.
\tag{1}
$$

For $y\ne0$ and $0<\epsilon<1$, define

$$
E_n=\frac{\|r_n\|}{\|y\|},\qquad
n_\epsilon=\inf\{n\in\mathbb N_0:E_n\le\epsilon\},
\tag{2}
$$

with $\inf\varnothing=+\infty$. Thus 1% error means $E_n\le0.01$, or
relative loss $\mathcal L(\theta_n)/\mathcal L(0)\le10^{-4}$.
Assume $0<\eta\|K\|\le1$. This makes every residual mode contract without
sign oscillation and makes $E_n$ nonincreasing. The study uses each run's
saved step, approximately $0.5/\|K_\gamma\|$ with a small safety margin.
The polynomial calculation must use that same step. Readout reparameterization
or a different step would define different dynamics.

The theorem concerns empirical training error on these samples. It imposes
no smoothness or frequency-support assumption on the target vector. A target
component in the original kernel's nullspace remains forever, which the
definition of $n_\epsilon$ permits. For nonzero initialization, the same
argument applies to the initial residual in place of $y$, with normalization
adjusted to the intended error criterion.

## 2. Where gamma enters: polynomial feature approximation

For $\gamma>0$, set

$$
\begin{aligned}
\beta_\gamma&=\operatorname{asinh}\frac{\pi}{2\gamma},\\
e_D(\gamma)&=\min\left\{\tanh\gamma,
\frac{4e^{-D\beta_\gamma}}{e^{\beta_\gamma}-1}
\left[\frac1{\sqrt{\gamma^2+\pi^2/4}}+
\frac1{\pi(D+1)}\right]\right\},
\end{aligned}
\tag{3}
$$

and set $e_D(0)=0$.

**Lemma 1 (uniform feature approximation).** For every real center $c$ and
integer $D\ge0$,

$$
\inf_{p\in\mathcal P_D}
\|\tanh(\gamma(\cdot-c))-p\|_{L_\infty([-1,1])}
\le e_D(\gamma).
\tag{4}
$$

The bound is uniform in $c$ and nondecreasing in gamma. Appendix A supplies
the pole-expansion proof, so this note does not require an unstated
approximation assumption. The closest tanh poles have imaginary height
$\pi/(2\gamma)$; decreasing gamma moves them away from the real interval
and increases the decay rate $\beta_\gamma$. For large gamma,
$\beta_\gamma\sim\pi/(2\gamma)$.

For a continuous function $h:[-1,1]\to\mathbb R$, let $I_Dh$ be its
degree-at-most-$D$ polynomial interpolant at the $D+1$ Chebyshev-Lobatto
nodes $z_\ell=\cos(\ell\pi/D)$, $\ell=0,\ldots,D$, with $D\ge1$.
The estimate below is applied separately to each hidden feature:

$$
h=h_j,\qquad h_j(x)=\tanh(\gamma(x-c_j)).
$$

Thus $h-I_Dh$ is a hidden feature's approximation error. The target $c(x)$
remains unchanged. For general $h$, the interpolant's two endpoint Chebyshev
coefficients have magnitude at most
$\|h\|_\infty$ and its $D-1$ interior coefficients at most
$2\|h\|_\infty$, giving $\|I_D\|_{\infty\to\infty}\le2D$.
Since $I_Dp=p$ for $p\in\mathcal P_D$,

$$
\|h-I_Dh\|_\infty
=\|(h-p)-I_D(h-p)\|_\infty
\le(1+2D)\|h-p\|_\infty.
\tag{5}
$$

Apply (5) to each $h_j$ and use Lemma 1 to bound its best polynomial error.
Keep the bias exact and sample the interpolating features to form
$\widetilde J_{\gamma,D}$. Each hidden column's normalized error norm is
at most $(1+2D)e_D(\gamma)$, independently of $m$. Summing squared column
errors proves

$$
\boxed{\|J-\widetilde J\|\le\|J-\widetilde J\|_F
\le\delta_J:=\sqrt W(1+2D)e_D(\gamma).}
\tag{6}
$$

The interpolation factor is deliberately elementary and conservative.
Writing the polynomial synthesis as $\widetilde J=VC$, where
$V_{ik}=T_k(x_i)/\sqrt m$, gives
$\widetilde K=VCC^TV^T$. The full matrix $CC^T$ is retained; neither its
cross terms nor the signs of its coefficients are dropped. Its exact rank
is at most $\min(D+1,W+1,m)$. A DCT and Clenshaw evaluation construct it
without the unstable high-degree empirical polynomial recurrence encountered
in the exploratory study.

## 3. Explicit acquisition-curve theorem

**Theorem 1 (common-slope polynomial-kernel transfer).** Fix the setting
above, a nonzero target $y$, and $D\ge1$. Define $\widetilde J$, $\delta_J$
by (5)–(6), $\widetilde K=\widetilde J\widetilde J^T$, and

$$
\Delta_D=(2\|\widetilde J\|+\delta_J)\delta_J.
\tag{7}
$$

Assume the same positive step satisfies

$$
0\preceq\eta K\preceq I,\qquad
0\preceq\eta\widetilde K\preceq I.
\tag{8}
$$

Let $u_1,\ldots,u_r$ be an orthonormal eigenbasis for the range of
$\widetilde K$, with positive eigenvalues $\widetilde\lambda_\ell$.
Decompose the *entire* target as

$$
\alpha_\ell=u_\ell^Ty,\qquad
y_\perp=y-\sum_{\ell=1}^r\alpha_\ell u_\ell,\qquad
q_\ell=1-\eta\widetilde\lambda_\ell\in[0,1].
$$

For every integer $n\ge0$, the polynomial model's exact relative residual is

$$
\widetilde E_{D,n}
=\frac{\sqrt{\|y_\perp\|^2+
\sum_{\ell=1}^r\alpha_\ell^2q_\ell^{2n}}}{\|y\|},
\tag{9}
$$

where $q^0=1$, including at $q=0$. The original model satisfies

$$
\boxed{
\max\{0,\widetilde E_{D,n}-n\eta\Delta_D\}
\le E_n\le
\min\{1,\widetilde E_{D,n}+n\eta\Delta_D\}.}
\tag{10}
$$

All quantities in (9)–(10) are computable before executing GD. A rectangular
SVD $\widetilde J=U\Sigma V^T$ supplies
$\widetilde\lambda_\ell=\sigma_\ell^2$ and the loadings $U^Ty$. It avoids
squaring the condition number merely to obtain small eigenvalues. A sufficient
way to verify (8) from the approximant alone is
$\eta(\|\widetilde J\|+\delta_J)^2\le1$; when this is too conservative,
one may establish contraction by a separate valid norm bound.

**Theorem 2 (target-dependent action refinement).** Under the same assumptions,
put $R=K-\widetilde K$ and define

$$
S_n(q)=\sum_{j=0}^{n-1}q^j
=\begin{cases}(1-q^n)/(1-q),&0\le q<1,\\n,&q=1,\end{cases}
\tag{11}
$$

where $S_0(q)=0$. Then (10) also holds with $n\eta\Delta_D$ replaced by

$$
d^{\rm action}_{D,n}
=\frac{\eta}{\|y\|}\left[
n\|Ry_\perp\|+
\sum_{\ell=1}^r|\alpha_\ell|\|Ru_\ell\|S_n(q_\ell)\right].
\tag{12}
$$

Consequently the strongest of these two radii is

$$
d_{D,n}=\min\{n\eta\Delta_D,d^{\rm action}_{D,n}\},\qquad
\ell_D(n)=\max(0,\widetilde E_{D,n}-d_{D,n}),\quad
u_D(n)=\min(1,\widetilde E_{D,n}+d_{D,n}).
\tag{13}
$$

The action refinement needs products with the original synthesis matrix, but
no trajectory and no original-kernel eigendecomposition. It uses additional
geometry and target information, so its sharpness must be distinguished from
that of the uniform analytic remainder. With $F=J-\widetilde J$, compute

$$
Rz=F(\widetilde J^Tz)+J(F^Tz).
\tag{14}
$$

This avoids forming an $m\times m$ kernel or subtracting two large kernel
products. It also explains exactly which additional information is measured:
the discarded operator's action on retained modes and on the target remainder.

## 4. Proof of the transfer theorems

**Step 1: feature error controls kernel error.** From $J=\widetilde J+F$,

$$
K-\widetilde K
=F\widetilde J^T+\widetilde JF^T+FF^T,
\qquad
\|K-\widetilde K\|\le
2\|\widetilde J\|\|F\|+\|F\|^2\le\Delta_D.
\tag{15}
$$

**Step 2: compare the two dynamics without assuming they commute.** Set
$A=I-\eta K$ and $B=I-\eta\widetilde K$. The identity

$$
A^n-B^n=\sum_{j=0}^{n-1}A^{n-1-j}(A-B)B^j
\tag{16}
$$

follows by expanding the summands and canceling consecutive terms. It is
valid for arbitrary square matrices; $AB=BA$ is unnecessary. Since
$A-B=-\eta R$ and $\|A\|,\|B\|\le1$,

$$
\frac{\|(A^n-B^n)y\|}{\|y\|}
\le\frac{\eta}{\|y\|}\sum_{j=0}^{n-1}\|RB^jy\|
\le n\eta\Delta_D.
\tag{17}
$$

**Step 3: preserve target-dependent action before taking the norm.** The
retained spectral decomposition gives

$$
B^jy=y_\perp+\sum_\ell\alpha_\ell q_\ell^ju_\ell,
\qquad
\|RB^jy\|\le\|Ry_\perp\|+
\sum_\ell|\alpha_\ell|q_\ell^j\|Ru_\ell\|.
\tag{18}
$$

Summing (18) in (17) gives (12). This bound still uses a triangle inequality
between modes, so it is not claimed optimal, but it preserves much more than
a single directional Rayleigh quotient.

**Step 4: pass from residual vectors to residual norms.** Orthogonality gives
(9), including the persistent $y_\perp$ term. The reverse triangle inequality
gives

$$
|E_n-\widetilde E_{D,n}|
\le\frac{\|(A^n-B^n)y\|}{\|y\|}.
\tag{19}
$$

Apply either bound from (17)–(18), and use $0\le E_n\le1$. This proves both
theorems and their combined version. At $n=0$, both radii are zero and both
models have relative residual one.

## 5. Exactly how residual bounds become update counts

**Corollary 3 (integer acquisition-time bracket).** Any integers $a,b\ge0$
satisfying

$$
\ell_D(a)>\epsilon,\qquad u_D(b)\le\epsilon
\quad\Longrightarrow\quad
\boxed{a+1\le n_\epsilon\le b.}
\tag{20}
$$

The lower statement follows because the actual residual is nonincreasing:
if it is still above tolerance at $a$, every earlier iterate is above it.
The upper statement follows because it has reached tolerance by $b$.
The $+1$ is essential: a bound that excludes update 61,790 proves a necessary
time of 61,791.

Equivalently, the ideal strongest endpoints from these curves are

$$
L_D=1+\sup\{n\ge0:\ell_D(n)>\epsilon\},\qquad
U_D=\inf\{n\ge0:u_D(n)\le\epsilon\},
\tag{21}
$$

with an empty lower set giving $L_D=0$ and an empty upper set giving
$U_D=+\infty$. The lower curve is nonincreasing: (9) decreases, while both
radii increase. The upper curve need not decrease because its uncertainty
radius can grow. Any verified upper witness is sufficient, even when it is
not the earliest crossing of the upper curve. Absence of a found upper
witness is not proof of nonrepresentability.

For multiple degrees, every valid inequality holds simultaneously, so

$$
\max_D L_D\le n_\epsilon\le\min_D U_D.
\tag{22}
$$

This selection uses the bounds themselves, not agreement with a GD hit. The
same statements apply when using only the analytic radius.

The implementation proceeds as follows:

1. Fix the grid, centers, common gamma, target array, readout metric, and
   actual GD step. Construct $\widetilde J$ for each declared degree.
2. Factor $\widetilde J$, retaining target loadings and its perpendicular
   residual. Evaluate (9), using logarithms for long-time powers.
3. Compute (7), and, for the action version, the products in (14). Evaluate
   $S_n$ with `expm1`/`log1p` so small rates do not suffer cancellation.
4. Double and bisect integer times to locate the monotone lower crossing.
   Search the upper curve on a logarithmic grid, refine candidate minima
   when needed, and check every returned sufficient endpoint. Do not assume
   global upper-curve monotonicity.
5. Combine valid endpoints across degrees. The numerical search is capped
   at $10^{18}$ updates; an unresolved endpoint remains explicitly unresolved.
6. Compare with archived GD for empirical validation. Independently certify
   selected endpoint statements with interval arithmetic as in Section 7.

There is generally no single scalar logarithm for a mixture of modes. For
the special case of one normalized target mode, no floor, zero transfer error,
and $0<q<1$, the formula reduces to
$n_\epsilon=\lceil\log(\epsilon)/\log(q)\rceil$. Equations (9)–(22) are
the corresponding computable bound for the full target mixture.

## 6. Worked calculation: gamma 16 at 1% residual

Use the archived $N=512$ geometry with $m=8193$, $W=559$, zero initialization,
raw readout coordinates, and

$$
c(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x),
\qquad
\eta=0.0020298167987789703.
$$

For $D=128$, the retained synthesis norm is approximately
$15.694828735203671$. Equation (6) gives
$\delta_J=0.05430658191687469$. At update 61,790, the analytic residual radius
at this degree exceeds 214 and is therefore vacuous. The action radius is
approximately $9.7894\times10^{-14}$: the discarded operator has extremely
little action on this target's retained components despite the conservative
global remainder.

The numerical experiment additionally uses the explicit FP64 sensitivity
allowance

$$
\delta_{\rm fp}=
64\epsilon_{\rm mach}(D+1)\|\widetilde J\|_F
+\|\widetilde J-U\Sigma V^T\|_F
+\sigma_{\max}\|U^TU-I\|_F.
\tag{23}
$$

This is a numerical sensitivity allowance, **not a proved rounding-error
enclosure**. Here it equals $4.553140274285996\times10^{-11}$.
The code adds
$n\eta(2\|\widetilde J\|+\delta_{\rm fp})\delta_{\rm fp}$ to the action
radius. At update 61,790 this addition is
$1.7925556391527273\times10^{-7}$, which dominates the measured action term.
For the analytic version it substitutes $\delta_J+\delta_{\rm fp}$ in (7).

**Table 1. Computed gamma-16 envelopes at the decisive integers.** Decimals
are rounded for display; comparisons use the saved full-precision values.
These are numerical evaluations, preceding the independent certification.

| Update $n$ | $\widetilde E_{D,n}$ | Lower envelope | Upper envelope |
|---:|---:|---:|---:|
| 61,790 | 0.010000333211 | 0.010000153955 | 0.010000512467 |
| 61,791 | 0.010000134332 | 0.009999955074 | 0.010000313591 |
| 61,792 | 0.009999935463 | 0.009999756202 | 0.010000114724 |
| 61,793 | 0.009999736603 | 0.009999557339 | 0.009999915868 |

The lower envelope excludes update 61,790. The upper envelope permits update
61,793. Therefore the proposed interval is

$$
\boxed{61{,}791\le n_{0.01}\le61{,}793.}
\tag{24}
$$

The approximate center curve alone crosses at 61,792, but a center-curve
prediction is not an error-certified timing interval. The executed GD first
hit is also 61,792; it is used for validation, not to set the interval.
Using only the analytic remainder at the larger degree $D=512$ yields the
slightly wider interval 61,789–61,796 after the sensitivity allowance.

**Table 2. Primary intervals from the full degree sweep.** Both columns of
intervals have separately certified endpoints for nominal real tanh. The
original trajectory uses its archived FP64 feature matrix.

| Gamma | Analytic-remainder interval | Combined interval | Executed GD hit | Combined degree |
|---:|---:|---:|---:|---:|
| 8 | 15,561,690–16,048,061 | 15,732,978–15,864,610 | 15,798,313 | 64 |
| 12 | 186,043–186,072 | 186,054–186,061 | 186,057 | 128 |
| 16 | 61,789–61,796 | 61,791–61,793 | 61,792 | 128 |
| 64 | 16,011–16,015 | 16,013–16,013 | 16,013 | 256 |

The [study report](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/common_slope_polynomial/REPORT.md)
also covers the four control targets, every degree, and unsuccessful upper
witnesses. This is retrospective training-dynamics evidence; it does not
establish test-set generalization.

## 7. Why the reported endpoints are independently certified

The analytic theorems are exact-arithmetic statements. Evaluating them in
FP64 with (23) does not itself certify the plotted curves. To certify the
selected timing claims, the independent checker uses 192-bit Arb intervals
for the original nominal-real tanh model on exact archived binary inputs.

Let $G=J^TJ$ and $b=J^Ty$. Common slopes give a convenient finite Gram identity.
Writing $\phi_{ij}=\tanh(\gamma(x_i-c_j))$ and
$\mu_j=m^{-1}\sum_i\phi_{ij}$, for distinct centers and $\gamma>0$,

$$
G_{jk}=1-\frac{\mu_j-\mu_k}{\tanh(\gamma(c_k-c_j))}\quad(j\ne k),
\qquad
G_{jj}=\frac1m\sum_i\phi_{ij}^2,
\quad G_{0j}=\mu_j,\quad G_{00}=1.
\tag{25}
$$

The off-diagonal identity follows from the tanh subtraction formula
$\phi_{ij}-\phi_{ik}=
\tanh(\gamma(c_k-c_j))(1-\phi_{ij}\phi_{ik})$.
Repeated centers or zero gamma use direct inner products instead of the
quotient. The audited study has distinct centers and positive slopes.

The affine parameter update can be evaluated by an integer matrix power:

$$
\begin{pmatrix}\theta_n\\1\end{pmatrix}
=\begin{pmatrix}I-\eta G&\eta b\\0&1\end{pmatrix}^{n}
\begin{pmatrix}0\\1\end{pmatrix},\qquad
E_n^2=\frac{\|y\|^2-2\theta_n^Tb+\theta_n^TG\theta_n}{\|y\|^2}.
\tag{26}
$$

Repeated squaring evaluates (26) without replaying $n$ GD updates or computing
eigenvectors. Interval Gershgorin bounds verify contraction. For (24), the
checker obtains

$$
\begin{aligned}
E_{61790}^2&\in
[1.0000666433070422,1.0000666433070425]\times10^{-4}>10^{-4},\\
E_{61793}^2&\in
[9.999473213921288,9.999473213921291]\times10^{-5}<10^{-4}.
\end{aligned}
\tag{27}
$$

Thus the necessary and sufficient statements in (24) hold for the nominal
real model independently of the FP64 sensitivity heuristic. This audit
certifies the selected endpoints, not every intermediate value of (13).
The exact target here is the archived sampled vector; the checker does not
silently reevaluate its transcendental defining function on a different
platform. At gamma 64 it similarly excludes 16,012 and certifies 16,013,
establishing the exact first hit.

## 8. Significance, the target condition, and the remaining scope

The causal chain being quantified is

$$
\gamma\ \longrightarrow\ \text{feature polynomial content}
\ \longrightarrow\ \text{retained kernel and discarded action}
\ \longrightarrow\ \text{target-dependent acquisition time}.
$$

Equation (3) makes the first link explicit. The polynomial matrix retains
the geometry needed for the second link. Equations (9)–(22) give the last
link with an error budget. The previous directional C2 reduction retained
far less of this information, so it could certify an obstruction while
missing the actual timing by large factors. In the primary cases the new
necessary times improve over directional C2 by approximately 111, 31, 44,
and 52 times.

Within the retained model, a mode with amplitude $|\alpha_\ell|$ decays as
$|\alpha_\ell|(1-\eta\widetilde\lambda_\ell)^n$. For a small positive
$\eta\widetilde\lambda_\ell$ and amplitude above the requested tolerance,
its attenuation time is approximately
$\log(|\alpha_\ell|/(\epsilon\|y\|))/(\eta\widetilde\lambda_\ell)$.
The theorem accounts for the other modes and the transfer uncertainty rather
than treating that single-mode estimate as the final answer. A directional
Rayleigh quotient averages rates; late acquisition can instead depend on a
much slower component that still has relevant target energy. Equation (9)
preserves that distinction.

The target need not lie in a preselected Fourier band. The polynomial
quantity $\delta_k(y)=\|(I-P_k)y\|/\|y\|$ remains useful for interpretation.
Lemma 1 implies $\|(I-P_k)J\|\le\sqrt W e_k(\gamma)$. Consequently any
readout attaining relative residual at most $\epsilon$ must obey, when the
denominator is positive,

$$
\|\theta\|\ge
\frac{[\delta_k(y)-\epsilon]_+\|y\|}{\sqrt W e_k(\gamma)}.
\tag{28}
$$

Indeed, project $J\theta-y$ onto the polynomial complement, use the reverse
triangle inequality, and then the operator-norm bound. A relevant target
tail can therefore require large coefficients and cancellation even when
the feature span can approximate the target very accurately. This explains
why capacity and ease of learning differ. Equation (28) alone is not the
sharp timing calculation; the retained modal mixture supplies that information.

Here $k$, used to describe target difficulty, need not equal $D$, used to
accurately retain the kernel. The transfer theorem also handles targets with
little or no high-degree tail, and correctly permits fast or slow learning
depending on their loadings. The quadratic control shows why polynomial
degree alone does not determine raw-coordinate acquisition time.

For the fixed primary problems, the certified combined intervals place the
gamma-8/gamma-64 time ratio between 982.51 and 990.73; the executed ratio is
986.59. The calculation uses the known dictionary and target before GD,
although the reported validation was performed retrospectively on archived
runs. It requires a retained matrix calculation rather than a scalar formula
in gamma alone. The action refinement additionally evaluates the original
synthesis operator.

Common slopes give a controlled one-parameter family and the useful identity
(25). The transfer proof itself works for any two contracting PSD kernels;
the sharpness gain comes from preserving matrix action and target alignment.
For heterogeneous slopes the approximation budget can be changed to a sum
of squared per-column remainders. The present experiment does not isolate
equal slopes as the sole reason for improvement.

Neither a bound on the approximation remainder nor a finite gamma sweep
proves that every target learns monotonically faster as gamma increases.
Changing gamma changes eigenvectors, loadings, and, under the study's clock,
the step. A sharp result uniform over all $0\le\gamma\le\Gamma$ would need
additional control of these quantities throughout the interval. The current
claims and endpoint certificates concern the prescribed gamma values and
geometry. Polynomial approximation degree, matrix cost, and any unresolved
numerical error remain part of the result.

## Appendix A. Proof of the explicit feature envelope

This appendix restates the feature lemma underlying the
[slope-distribution theorem](slope_distribution_spectrum.md), including the
constants used by the implementation. Let $t_\ell=\pi(\ell+1/2)$. The
paired partial-fraction expansion of tanh gives

$$
\tanh z=\sum_{\ell=0}^\infty\frac{2z}{z^2+t_\ell^2},\qquad
\tanh(\gamma(x-c))=-\frac1\gamma\sum_{\ell\ge0}
\left[\frac1{z_\ell^+-x}+\frac1{z_\ell^--x}\right],\quad
z_\ell^\pm=c\pm i\frac{t_\ell}{\gamma}.
\tag{A1}
$$

The first identity follows by logarithmically differentiating the product
for cosh; conjugate pairs converge locally uniformly away from the poles.
For $z\notin[-1,1]$, choose $s=\sqrt{z^2-1}$ so that $w=z+s$ has
$|w|>1$. The geometric cosine series gives

$$
\frac1{z-x}=\frac1s\left(1+2\sum_{n\ge1}w^{-n}T_n(x)\right).
$$

Since $|T_n(x)|\le1$, truncation after degree $D$ has error at most
$2|w|^{-D}/(|s|(|w|-1))$. To make this uniform in the real center, put
$|w|=r=e^u$ and $v=|\operatorname{Im}z|$. The ellipse parametrization
gives $v\le\sinh u$ and

$$
|s|^2(r-1)^2=(r-1)^2\sinh^2u+
v^2\left(\frac{2r}{r+1}\right)^2.
\tag{A2}
$$

Both terms increase with $r>1$. The smallest permitted radius is
$\rho(v)=v+\sqrt{1+v^2}$, at which $|s|=\sqrt{1+v^2}$.
The resolvent error is therefore at most

$$
\frac{2\rho(v)^{-D}}{\sqrt{1+v^2}(\rho(v)-1)}.
$$

Set $v_\ell=t_\ell/\gamma$ and $\rho_\ell=\rho(v_\ell)$. Adding both
poles of every pair proves

$$
\inf_{p\in\mathcal P_D}\|\tanh(\gamma(\cdot-c))-p\|_\infty
\le\frac4\gamma\sum_{\ell\ge0}
\frac{\rho_\ell^{-D}}{\sqrt{1+v_\ell^2}(\rho_\ell-1)}.
\tag{A3}
$$

The remainder series converges absolutely. Paired feature sums converge
uniformly on the real interval, so the associated degree-$D$ polynomials
converge uniformly too; their limit is still in the finite-dimensional closed
space $\mathcal P_D$. This justifies the infinite-sum approximant.

The summand as a function $g(v)$ is decreasing. Its samples start at
$v_0=\pi/(2\gamma)$ with spacing $\pi/\gamma$, so (A3) is bounded by
$4g(v_0)/\gamma+(4/\pi)\int_{v_0}^\infty g(v)\,dv$. Substituting
$v=\sinh u$ gives

$$
\int_{v_0}^\infty g(v)\,dv
=\int_{\beta_\gamma}^\infty\frac{e^{-Du}}{e^u-1}\,du
=\sum_{n=D+1}^\infty\frac{e^{-n\beta_\gamma}}n
\le\frac{e^{-D\beta_\gamma}}{(D+1)(e^{\beta_\gamma}-1)}.
\tag{A4}
$$

Together with the first summand, this is the second term in (3). The midpoint
of the feature's endpoint values is a constant approximant whose error is

$$
\frac{\tanh(\gamma(1-c))-\tanh(\gamma(-1-c))}{2}
=\frac{\sinh(2\gamma)}{\cosh(2\gamma c)+\cosh(2\gamma)}
\le\tanh\gamma.
$$

Taking the better approximant proves (4). Monotonicity in gamma follows by
writing $\rho=e^{\beta_\gamma}$: the first part of the second term in (3)
is $(8/\pi)(\rho+1)\rho^{-D}/(\rho^2+1)$, and its other part is
$4\rho^{-D}/[\pi(D+1)(\rho-1)]$. Both decrease in $\rho>1$, while
$\rho$ decreases with gamma. The constant bound $\tanh\gamma$ increases
with gamma as well. The case $\gamma=0$ is exact with a constant polynomial.

## Appendix B. Code and evidence map

The implementation and evidence are linked directly below. The numerical
formulae in Section 6 were recomputed from the stored mode arrays and
checked against the recorded integer brackets; the training trajectories
were not rerun for this note.

| Item | Source |
|---|---|
| Interpolation, retained SVD, radii, and integer searches | [common_slope_poly.py](../experiments/expD36_frozen_gamma_probe/common_slope_poly.py) |
| Fixed archive comparison and plotting | [common_slope_analysis.py](../experiments/expD36_frozen_gamma_probe/common_slope_analysis.py) |
| Independent Gram/power interval checker | [common_slope_audit.py](../experiments/expD36_frozen_gamma_probe/common_slope_audit.py) |
| Every degree, baseline bound, and archived first hit | [summary.json](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/common_slope_polynomial/summary.json) |
| Certified primary endpoint enclosures | [interval_audit.json](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/common_slope_polynomial/interval_audit.json) |
| All inputs for a compact reproduction | [probe_inputs.tar.gz](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/common_slope_polynomial/probe_inputs.tar.gz) |

Extract the compact input archive into an empty directory, then run from
the repository with NumPy, SciPy, Matplotlib, and python-flint installed:

```bash
python -m experiments.expD36_frozen_gamma_probe.common_slope_analysis --root INPUT_DIRECTORY --output OUTPUT_DIRECTORY
python -m experiments.expD36_frozen_gamma_probe.common_slope_audit --root INPUT_DIRECTORY --output OUTPUT_DIRECTORY
```

Use the same output directory for both commands. The runners default to the
historical full-sweep archive when paths are omitted. The study used degrees
32 through 2048 by powers of two, gamma values 8, 12, 16, and 64, and five
targets. All original data and the three-panel report remain separate records.
