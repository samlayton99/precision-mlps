# Slope distributions force a spectral restriction on readout learning

The structural question is how restrictions on frozen tanh slopes constrain the
readout spectrum and the target's access to fast directions. The GD recurrence
then converts that restriction into necessary learning times. The results below
separate a general distribution theorem from a more detailed evaluation for the
study's finite grid. They do not assert that every target is slow, that a median
controls a maximum, or that the resulting time estimates are uniformly tight.

| Symbol | Meaning |
|---|---|
| $g_j=\lvert\gamma_j\rvert$, $W$ | Absolute hidden slope and number of hidden neurons. |
| $J$, $K=JJ^\top$, $L$ | Raw readout matrix, function-space curvature, and largest eigenvalue. |
| $\nu_i$ | Eigenvalues of $K$ in decreasing order, including zero modes. |
| $e_k(g)$ | Existing uniform degree-$k$ tanh approximation envelope. |
| $q(G)$, $B_{k,G}$ | Number of slopes exceeding $G$ and squared-tail budget of the remaining columns. |
| $\delta_{k,G}$ | Relative initial-residual distance from polynomials plus exceptional features. |
| $F(s)$ | Initial-residual energy fraction at normalized eigenvalues $\nu_i/L\le s$. |
| $\chi$, $\epsilon$ | Normalized step $\eta L$ and desired relative residual norm. |

## Setting and the existing feature estimate

Let $x_i\in[-1,1]$, and use half empirical MSE with

$$
J_{i0}=m^{-1/2},\qquad
J_{ij}=m^{-1/2}\tanh(\gamma_j(x_i-c_j)),\qquad
y_i=m^{-1/2}f(x_i).
$$

Centers are arbitrary in the general theorem. The numerical application uses
the existing evenly spaced centers and halo. Zero slopes in this parameterization
give zero hidden columns; an arbitrary constant hidden column also lies in every
polynomial space. All readout coordinates here are raw and have equal metric.

Let $P_k$ project onto sampled polynomials of degree at most $k$. The existing
pole argument proves

$$
\|(I-P_k)J_{:j}\|\le e_k(g_j),\quad
e_k(g)=\min\left\{\tanh g,
\frac{4e^{-k\beta_g}}{e^{\beta_g}-1}
\left[\frac1{\sqrt{g^2+\pi^2/4}}+\frac1{\pi(k+1)}\right]\right\},
\quad \beta_g=\operatorname{asinh}\frac\pi{2g},
$$

with $e_k(0)=0$. This envelope is nondecreasing in $g$. Its existing proof is
the feature approximation part of the bounded-slope study; the distribution
and spectral conclusions below follow from it.

## 1. Distribution-to-spectrum theorem

For a threshold $G\ge0$, define

$$
H_G=\{j:g_j>G\},\qquad q(G)=|H_G|,\qquad
B_{k,G}=\sum_{j\notin H_G}e_k(g_j)^2.
$$

For every $k\ge0$, with empty spectral tails interpreted as zero,

$$
\boxed{\sum_{i>k+1+q(G)}\nu_i
\le B_{k,G}\le[W-q(G)]e_k(G)^2.}
$$

Consequently, for each integer $s\ge1$ with a valid eigenvalue index,

$$
\nu_{k+1+q(G)+s}\le\frac{B_{k,G}}s.
$$

**Proof.** Let $S$ be the span of degree-$k$ sampled polynomials and the
exceptional columns $J_{:j}$ for $j\in H_G$. Its dimension is at most
$r=k+1+q(G)$. Orthogonal projection onto $S$ annihilates the bias and all
exceptional-column remainders. For the other columns, projection is at least as
accurate as projection onto polynomials alone. Therefore
$\|(I-P_S)J\|_F^2\le B_{k,G}$.
The minimum of $\operatorname{tr}((I-P)K)$ over rank-at-most-$r$ orthogonal
projectors is $\sum_{i>r}\nu_i$: in the eigenbasis, the removed trace is at most
the sum of the largest $r$ eigenvalues. Apply this minimum to $P_S$.
The individual-eigenvalue bound follows because the next $s$ eigenvalues are
all at least $\nu_{r+s}$. Finally use monotonicity of $e_k$.

Large-slope exceptions can therefore add at most $q(G)$ directions beyond
the polynomial approximation rank before this tail restriction applies.
This does not say those directions necessarily become fast.

### Maximum, mean, and distribution corollaries

A maximum cap $g_j\le\Gamma$ gives $q(\Gamma)=0$ and
$\sum_{i>k+1}\nu_i\le W e_k(\Gamma)^2$.
Its explicit exponential factor is $e^{-2k\beta_\Gamma}$; finite prefactors
and the finite sample dimension remain part of the statement.

If $W^{-1}\sum_j g_j\le\mu$, then for $G>0$,
$q(G)\le W\mu/G$. More generally, a $p$th moment cap
$W^{-1}\sum_jg_j^p\le M_p$ gives $q(G)\le WM_p/G^p$.
If only an upper count $q_*$ is available, the safe consequence is
$\sum_{i>k+1+q_*}\nu_i\le W e_k(G)^2$; do not substitute $q_*$ for $q(G)$
in the factor $W-q(G)$, which would incorrectly shrink the bound.
An empirical tail constraint $q(G)/W\le\tau(G)$ works the same way.
These are deterministic implications for the actual slope vector, not
probability guarantees about a sampled population.

The numerical mean-only envelope takes the minimum of
$We_k(G)^2/[i-k-1-q_*]$ over positive denominators, selected thresholds and
degrees, with $q_*=\min\{W,\lfloor W\mu/G\rfloor\}$. The raw trace bound
$\nu_i\le W+1$ supplies an initial bound. Optimizing over a finite threshold
set preserves validity but need not find the best possible moment theorem.

A median cap gives only $q(G)\le\lfloor W/2\rfloor$ under the convention that
at least half the slopes are at most $G$. It can leave many exceptional
directions. A mean cap cannot be substituted inside $e_k$ as though it were a
maximum cap; no required Jensen direction has been proved for that substitution.

**Counterexample to a target-independent mean/median delay.** Take one centered
feature $\tanh(Hx)$, a bias, and the target $y$ equal to that feature. Append
$W-1$ zero-slope, zero-bias hidden features. The kernel $K$, its largest
curvature, and GD predictions are unchanged, while the mean absolute slope is
$H/W$ and the median is zero for $W\ge3$. This construction is compatible
with a fixed list of centers containing zero. Geometry or distribution
assumptions must rule it out, or the target condition must detect it.

## 2. Force target energy into the slow spectrum

For a nonzero initial residual $r_0$, define

$$
\delta_{k,G}=\frac{\|(I-P_S)r_0\|}{\|r_0\|},\qquad
F(s)=\frac{\|1_{[0,sL]}(K)r_0\|^2}{\|r_0\|^2},\qquad 0<s\le1.
$$

If $\delta_{k,G}>0$, its normalized residual direction $v$ is orthogonal to
$S$, so $v^\top Kv\le B_{k,G}$. Set
$z=\sqrt{\min\{B_{k,G}/(sL),1\}}$. Then

$$
\boxed{F(s)\ge
\left[\delta_{k,G}\sqrt{1-z^2}
-\sqrt{1-\delta_{k,G}^2}\,z\right]_+^2.}
$$

**Proof.** The fast projection of $v$ has squared norm at most
$B_{k,G}/(sL)$ by its Rayleigh quotient. Decomposing $v$ and $r_0/\|r_0\|$
into slow and fast subspaces bounds their inner product by
$\sqrt{1-\alpha^2}\sqrt{F(s)}+\alpha\sqrt{1-F(s)}$, where $\alpha\le z$.
Solving this two-dimensional angle inequality gives the displayed bound.
For $\delta=0$ it is zero, as required by the exception-aligned counterexample.

Any sharper proved upper bound on $v^\top Kv$ can replace $B_{k,G}$.
Take the maximum over witnesses and its nondecreasing envelope in $s$.
Target alignment is essential; an eigenvalue counting theorem alone does not
give a learning delay for the study's specified target.

For normalized-step GD, a geometric lower bound $L_*\le L$ can replace $L$
in $B/(sL)$, giving a weaker but spectrum-free prediction. On the symmetric
study grid, with $\sigma$ the fraction of samples satisfying $|x_i|\ge3/4$,

$$
L_* = \max\left\{1,\ \sigma^2
\sum_{|c_j|\le1/2}\tanh^2(g_j/4)\right\}\le L.
$$

The bias gives one, and testing the sign vector gives the sum after pairing
$x$ with $-x$. Alternatively, conditioning the statement on the actual prescribed
step $\eta=\chi/L$ retains the known normalization. Report these evaluations
separately; a change in absolute eigenvalues need not change normalized rates.

## 3. Combine spectral thresholds without double counting

Suppose $0<s_1<\cdots<s_M<1$ and $F(s_j)\ge p_j$, where $p_j$ have been
replaced by their running maximum. Put $p_0=0$. For $0<\chi<1$ and every
integer $t\ge0$, ordinary GD satisfies

$$
\frac{\|r_t\|^2}{\|r_0\|^2}\ge
\sum_{j=1}^M(p_j-p_{j-1})(1-\chi s_j)^{2t}
+(1-p_M)(1-\chi)^{2t}.
$$

**Proof.** The right side places each guaranteed increment of spectral mass
at the fastest threshold allowed, and all remaining mass at normalized rate
one. Its CDF is everywhere no larger than the actual CDF. Integrating the
decreasing function $(1-\chi s)^{2t}$ against these distributions establishes
the inequality. Equivalently, integrate their CDF difference against
$2t\chi(1-\chi s)^{2t-1}$, which is nonnegative.

The first integer step at which this lower error curve is at most $\epsilon$
is a necessary-time bound. This uses all selected thresholds, improves on each
single-threshold error witness, and needs no trajectory fit. It is not
automatically stronger than the existing spectral-tail Jensen result, which
uses additional measured moments. A conservative CDF may still be very loose.

## 4. Center-aware evaluation with explicit pole remainders

For $g_j>0$, put $z_{\ell j}=c_j+i\pi(\ell+1/2)/g_j$ and choose
$s_{\ell j}=\sqrt{z_{\ell j}^2-1}$ so $|w_{\ell j}|>1$ for
$w_{\ell j}=z_{\ell j}+s_{\ell j}$. The nonconstant Chebyshev coefficients
from the first $M$ conjugate pole pairs are

$$
c_{nj}^{(M)}=-\frac{4\operatorname{sign}(\gamma_j)}{g_j}
\operatorname{Re}\sum_{\ell<M}\frac{w_{\ell j}^{-n}}{s_{\ell j}},\qquad n\ge1.
$$

Retain these signed coefficients before aggregation. The constant is irrelevant
to every projected tail. Truncating their degree at $D$ has error at most

$$
T_{Dj}^{(M)}=\frac4{g_j}\sum_{\ell<M}
\frac{|w_{\ell j}|^{-D}}{|s_{\ell j}|(|w_{\ell j}|-1)}.
$$

For $\pi(M+1/2)/g_j\ge1$, the degree-$k$ tail of the omitted poles is bounded by

$$
R_{kj}^{(M)}=\frac{16g_j^{k+1}}{\pi^{k+2}}
\left[(2M+1)^{-k-2}
+\frac{(2M+1)^{-k-1}}{2(k+1)}\right].
$$

To prove this, use $\rho\ge2v$, $d\ge v$, and $\rho-1\ge v$ in the
positive pole-tail series, then bound the decreasing remaining series by its
first term plus its integral. The omission concerns the degree-$k$ tail,
not the entire missing feature: its lower polynomial part is annihilated.

Let $\widetilde J$ be the sampled polynomial matrix from coefficients
$c_{nj}^{(M)}$, $1\le n\le D$, with the original $m^{-1/2}$ normalization.
For $k<D$, put $\rho_k=\|(T_{Dj}^{(M)}+R_{kj}^{(M)})_j\|_2$. Then

$$
\|(I-P_k)J\|_F
\le\|(I-P_k)\widetilde J\|_F+\rho_k,
\qquad
\|J^\top v\|\le\|\widetilde J^\top v\|+\rho_k
\quad(v\perp\mathcal P_k,\ \|v\|=1).
$$

The same argument applies to the low-slope columns and the enlarged exceptional
span. Taking minima with the original envelopes preserves validity. This
evaluation uses centers, slopes, samples and targets, but no measured kernel
eigenvectors. It is a geometry-specific computation, not a closed-form law
in the cap alone.

The implementation adds and records an FP64 roundoff monitor separately from
these analytic truncation remainders. It is not interval arithmetic. Tiny
projected directions, nearly dependent exceptional columns and unresolved
eigenvalues must not be promoted to numerical certificates.

## Evidence and limits

The distribution theorem and CDF conversion are mathematical consequences of
the stated assumptions. Numerical tests independently project sampled tanh
features, check the complete spectral tail, compare target mass with an
eigendecomposition, and compare the combined error curve with directly
executed GD. The full-sweep report records the subsequent data analysis.

Neither a matching upper learning-time bound nor uniform monotonicity in gamma
has been proved. The claim concerns frozen readout GD, not Adam or joint hidden
feature training. Measured spectral-tail Jensen remains a separate benchmark
for how much accuracy is lost when replacing the actual spectrum by a
slope-derived guarantee.

The [full-sweep report](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/REPORT.md#slope-distributions-spectrum-and-the-remaining-time-bound-slack)
contains the executed CPU analysis, saved-hit comparisons, distribution
controls, and remaining tightness gap. A mathematically proved zero-access
direction is allowed at threshold zero; if its forced mass is at least
$\epsilon^2$, the tolerance cannot be reached at a finite step under
$0<\chi<1$. This case is kept separate from a finite prediction cap.
