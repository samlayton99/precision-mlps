# Common-slope acquisition times from polynomial kernels

The question is whether a polynomial approximation can retain enough of a
common-slope dictionary's kernel to bound its acquisition time sharply. The
previous C2 argument reduces the target's polynomial tail to one access
number. Here the retained matrix keeps the signed coefficients and their
cross-couplings, and an explicit remainder controls the discarded action.
This is a fixed-geometry, fixed-common-slope result. It does not establish a
uniform acquisition-time bound over an interval of slopes.

| Symbol | Meaning |
|---|---|
| $m,W$ | Number of training samples and hidden features. |
| $J_\gamma$ | Raw readout design, including bias, divided by $\sqrt m$. |
| $y$ | Archived target values divided by $\sqrt m$; initialization is zero. |
| $K_\gamma$ | Output-space kernel $J_\gamma J_\gamma^T$. |
| $D$ | Degree used to approximate the whole synthesis operator. |
| $k$ | Degree used to describe the target's polynomial tail; need not equal $D$. |
| $e_D(\gamma)$ | Existing uniform degree-$D$ tanh approximation envelope. |
| $E_n$ | Relative training residual after $n$ readout GD updates. |
| $n_\epsilon$ | First integer $n$ with $E_n\le\epsilon$. |

## 1. A polynomial approximation with explicit gamma dependence

Fix sample points $x_i\in[-1,1]$, real centers $c_j$, and a shared slope
$\gamma\ge0$. Write

$$
J_\gamma=\frac1{\sqrt m}
\begin{bmatrix}1&\tanh(\gamma(x_i-c_j))\end{bmatrix}_{i,j}.
$$

For $\gamma>0$, the envelope proved in the
[slope-distribution note](slope_distribution_spectrum.md) is

$$
\begin{aligned}
\beta_\gamma&=\operatorname{asinh}\frac{\pi}{2\gamma},\\
e_D(\gamma)&=\min\left\{\tanh\gamma,
\frac{4e^{-D\beta_\gamma}}{e^{\beta_\gamma}-1}
\left[\frac1{\sqrt{\gamma^2+\pi^2/4}}+
\frac1{\pi(D+1)}\right]\right\}.
\end{aligned}
$$

Set $e_D(0)=0$. This bounds the best uniform polynomial approximation error
for each shifted feature. It is uniform in the centers. Smaller gamma gives
faster polynomial approximation decay.

Let $I_D$ interpolate at the $D+1$ Chebyshev-Lobatto points. Its endpoint
Chebyshev coefficients have absolute value at most $\|f\|_\infty$ and its
interior coefficients at most $2\|f\|_\infty$, so
$\|I_D\|_{\infty\to\infty}\le2D$ for $D\ge1$. Since interpolation fixes
every polynomial of degree at most $D$,

$$
\|f-I_Df\|_\infty\le(1+2D)\inf_{p\in\mathcal P_D}\|f-p\|_\infty.
$$

Keep the bias column exact and interpolate each hidden feature. The resulting
$\widetilde J_{\gamma,D}$ satisfies

$$
\|J_\gamma-\widetilde J_{\gamma,D}\|_2
\le\|J_\gamma-\widetilde J_{\gamma,D}\|_F
\le\delta_J:=\sqrt W(1+2D)e_D(\gamma).
$$

The interpolation factor is deliberately elementary and conservative. The
retained operator has rank at most $\min(D+1,W+1,m)$ in exact arithmetic. No
empirical high-degree orthogonal-polynomial recurrence is needed to construct
it. The implementation uses a DCT and Clenshaw evaluation.

## 2. Transfer theorem, without a shared eigenbasis

Put $\widetilde K=\widetilde J\widetilde J^T$. Assume the *same* archived
step satisfies $0\preceq\eta K\preceq I$ and
$0\preceq\eta\widetilde K\preceq I$. Define

$$
\Delta=(2\|\widetilde J\|_2+\delta_J)\delta_J.
$$

Then, for every target $y\ne0$ and every integer $n\ge0$,

$$
\max(0,\widetilde E_n-n\eta\Delta)
\le E_n\le
\min(1,\widetilde E_n+n\eta\Delta),
\qquad
\widetilde E_n=\frac{\|(I-\eta\widetilde K)^ny\|}{\|y\|}.
$$

**Proof.** Writing $J=\widetilde J+F$ gives
$K-\widetilde K=F\widetilde J^T+\widetilde JF^T+FF^T$ and hence
$\|K-\widetilde K\|\le\Delta$. For $A=I-\eta K$ and
$B=I-\eta\widetilde K$, the noncommuting telescoping identity is

$$
A^n-B^n=\sum_{j=0}^{n-1}A^{n-1-j}(A-B)B^j.
$$

Both factors contract, so the norm applied to $y$ is at most
$n\eta\Delta\|y\|$. The reverse triangle inequality proves the claim.

All retained modes and their target loadings enter $\widetilde E_n$. In
particular, its perpendicular target component remains in the residual; it
is not silently discarded. Diagonalizing the retained polynomial kernel is
part of the calculation. Diagonalizing the original kernel is a separate
validation reference.

## 3. A target-dependent operator-action refinement

Let $R=K-\widetilde K$, and write
$y=y_\perp+\sum_\ell a_\ell u_\ell$ in an orthonormal eigenbasis of the
retained kernel, with $\widetilde K y_\perp=0$. Put
$q_\ell=1-\eta\widetilde\lambda_\ell\in[0,1]$ and

$$
S_n(q)=\begin{cases}(1-q^n)/(1-q),&q<1,\\n,&q=1.\end{cases}
$$

The same telescoping proof, applied to each retained component, gives

$$
|E_n-\widetilde E_n|\le d_n^{\rm action}:=
\frac{\eta}{\|y\|}\left[
n\|Ry_\perp\|+
\sum_\ell |a_\ell|\|Ru_\ell\|S_n(q_\ell)\right].
$$

Take the minimum of this and $n\eta\Delta$. The action version uses the
original synthesis matrix to evaluate the discarded action, but uses no GD
trajectory, fitted rate, or original-kernel eigendecomposition. It is richer
information than the analytic envelope, so the two versions must be reported
separately. To avoid cancellation, compute
$Rz=(J-\widetilde J)(\widetilde J^Tz)+J((J-\widetilde J)^Tz)$.

## 4. Acquisition times and interpretation

If a lower error envelope exceeds $\epsilon$ at $n$, then
$n_\epsilon\ge n+1$, because the original residual is nonincreasing. If an
upper envelope is at most $\epsilon$ at $n$, then $n_\epsilon\le n$.
The upper envelope itself need not be monotone; sufficient times must be
checked at their returned endpoints. Failure to find a sufficient-time
witness does not prove that the target is unreachable.

The target-only quantity
$\delta_k(y)=\|(I-P_k)y\|/\|y\|$ still describes how much target content
lies beyond low-degree polynomials. This theorem additionally retains its
signed coefficients and the operator coupling them. It applies to every
target, but a useful delay requires an interaction between that target and
the slowly acquired components. A nonzero tail below the requested tolerance
does not force a delay to that tolerance.

This method approaches the exact finite-model dynamics as the approximation
error shrinks. Sharpness obtained only with an almost full operator is useful
for prediction but supplies limited compression of the explanation. Neither
the theorem nor a finite gamma sweep proves that every target's acquisition
time is monotone in gamma, or proves a guarantee for every common slope below
a cap.

## 5. Numerical status and reproducibility

The inequalities above are exact-arithmetic theorems. The experiment stores
the analytic remainder, measured feature discrepancy, SVD reconstruction and
orthogonality checks, and a separate FP64 sensitivity allowance. A sensitivity
allowance is not an interval enclosure. Report floating-point evaluations as
estimates unless an independent interval calculation certifies the asserted
endpoints. Agreement at increased precision alone is not a certificate.

The fixed grid, centers, archived target arrays, readout coordinates, and
actual saved step define each comparison. Polynomial degree is swept without
fitting to the GD hits. This is retrospective verification on existing runs.

Run the archived comparison and its independent primary-target audit with:

```bash
python -m experiments.expD36_frozen_gamma_probe.common_slope_analysis
python -m experiments.expD36_frozen_gamma_probe.common_slope_audit
```

Both accept `--root` for the full-sweep archive and `--output` for a fresh
evidence directory. The default output is the `common_slope_polynomial`
refinement. They require the archived common arrays, dictionary certificates,
and capped-kernel campaign summary. The comparison verifies matrix and target
hashes and uses each trajectory's recorded step.

The independent audit encloses the common-slope finite Gram identity in Arb.
For $G=J^TJ$ and $c=J^Ty$, it raises the augmented affine update matrix

$$
\begin{pmatrix}I-\eta G&\eta c\\0&1\end{pmatrix}
$$

to the requested integer power, starting from zero readout. If the resulting
readout is $a_n$, it encloses
$E_n^2=(\|y\|^2-2a_n^Tc+a_n^TGa_n)/\|y\|^2$.
An interval Gershgorin bound checks contraction. This independently certifies
the *endpoint statements* for nominal real tanh on exact archived binary
inputs. It does not certify every FP64 error-envelope value or the numerical
evaluation of the interpolation remainder. Integer matrix powers require
logarithmically many squarings, without replaying GD or computing eigenvectors.
