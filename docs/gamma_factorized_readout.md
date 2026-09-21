# Gamma-controlled smoothing and finite readout acquisition times

The question is whether the same explicit gamma mechanism can explain and
accurately quantify a readout learning delay. With fixed centers and a common
slope, every tanh feature is a gamma-controlled smoothing of a fixed step
feature. Applying that filter to the full geometric kernel, retaining its
couplings, gives a computable learning curve. A controlled approximation then
transfers that curve to the original finite model. The construction below
connects these statements without assuming Fourier modes are the finite
model's eigenvectors.

This note concerns zero-initialized Euclidean readout GD on the training
samples. Neither hidden parameters nor the parameter metric change. The
polynomial transfer results are used as general operator comparison tools;
their polynomial approximation is not used to construct this predictor.

**Notation.** Vector norms are Euclidean and matrix norms are spectral unless
marked otherwise. The same empirical normalization applies to target and
features.

| Symbol | Meaning |
|---|---|
| $x_i,c_j,m,W$ | Samples, fixed centers, sample count, and hidden width. |
| $\gamma>0$ | Common hidden slope, the varied model parameter. |
| $J_\gamma,y$ | Raw design matrix and target, normalized by $1/\sqrt m$. |
| $K_\gamma,L_\gamma$ | $J_\gamma J_\gamma^T$ and its largest eigenvalue. |
| $S_\gamma,M_\gamma$ | Whole-line smoothing operator and its Fourier multiplier. |
| $T,R,Q$ | Auxiliary half-period, maximum sample-center displacement, retained odd harmonics. |
| $F_Q,C_Q,G_Q$ | Fixed sampling matrix, center/readout matrix, and $C_QC_Q^T$. |
| $D_{\gamma,Q}$ | Explicit diagonal gamma filter, with bias entry one. |
| $\tau_{\gamma,Q},\delta_J,\Delta$ | Feature, synthesis, and kernel approximation bounds. |
| $E_n,n_\epsilon$ | Relative residual norm and its first crossing of $\epsilon$. |

## 1. The original finite model and its clock

For samples $x_i\in[-1,1]$, arbitrary real centers $c_j$, and a nonzero
sampled target $y_i=f(x_i)/\sqrt m$, set

$$
(J_\gamma)_{i0}=m^{-1/2},\qquad
(J_\gamma)_{ij}=m^{-1/2}\tanh(\gamma(x_i-c_j)).
$$

Train only the $W+1$ raw coefficients on
$\frac12\|J_\gamma\theta-y\|^2$, with $\theta_0=0$ and a prescribed
$\eta_\gamma>0$ satisfying $\eta_\gamma\|K_\gamma\|\le1$. Then

$$
r_n=(I-\eta_\gamma K_\gamma)^ny,\qquad
E_n=\frac{\|r_n\|}{\|y\|},\qquad
n_\epsilon=\inf\{n\in\mathbb N_0:E_n\le\epsilon\}.
\tag{1}
$$

An empty crossing set has time $+\infty$. All comparisons use the actual
prescribed step, including its gamma dependence. The archived clock is
approximately $0.5/L_\gamma$, so a uniform rescaling of all eigenvalues alone
does not explain its large acquisition-time changes. A 1% residual means
relative squared loss $10^{-4}$.

## 2. Exact gamma factorization before sampling

**Lemma 1 (common smoothing of a fixed dictionary).** Let

$$
\rho_\gamma(t)=\frac\gamma2\operatorname{sech}^2(\gamma t),
\qquad S_\gamma h=\rho_\gamma*h.
$$

Convolution is well defined on bounded functions, including the constant
feature and step features $b_j(t)=\operatorname{sign}(t-c_j)$. It satisfies

$$
S_\gamma1=1,\qquad
S_\gamma b_j(x)=\tanh(\gamma(x-c_j)).
\tag{2}
$$

For $k_\infty(t,s)=1+\sum_j b_j(t)b_j(s)$, the finite tanh kernel is exactly

$$
k_\gamma(x,x')=
\iint\rho_\gamma(x-t)\rho_\gamma(x'-s)k_\infty(t,s)\,dt\,ds,
\qquad (K_\gamma)_{ii'}=\frac{k_\gamma(x_i,x_{i'})}{m}.
\tag{3}
$$

**Proof.** The density integrates to one, and its cumulative integral is
$[1+\tanh(\gamma t)]/2$. Splitting the convolution at $c_j$ proves (2).
Insert the finite sum defining $k_\infty$ into (3) and apply (2) to both
variables. Boundedness and integrability justify every exchange. This proof
does not treat the nondecaying step functions as whole-line $L^2$ vectors.

Under the convention $\widehat h(\omega)=\int h(t)e^{-i\omega t}\,dt$,
the transform of the integrable smoothing density is

$$
M_\gamma(\omega)=\widehat\rho_\gamma(\omega)
=\frac{z}{\sinh z},\qquad z=\frac{\pi|\omega|}{2\gamma},
\qquad M_\gamma(0)=1.
\tag{4}
$$

For completeness, integrate $e^{-i\xi z}/\cosh^2z$ around a rectangle of
height $\pi$ for $\xi>0$. Its upper horizontal integral is
$-e^{\pi\xi}$ times the lower one. The double pole at $i\pi/2$ has residue
$i\xi e^{\pi\xi/2}$, so the residue theorem gives
$\int\operatorname{sech}^2(t)e^{-i\xi t}dt=\pi\xi/\sinh(\pi\xi/2)$.
Evenness and continuity cover the remaining frequencies. Scaling proves (4).
In particular,

$$
M_\gamma(\omega)=1-z^2/6+O(z^4)\quad(z\to0),\qquad
M_\gamma(\omega)\sim2ze^{-z}\quad(z\to\infty).
\tag{5}
$$

Thus gamma changes frequency-dependent access to corrections. Equation (3)
applies that same explicit filter to both arguments of a fixed geometric
kernel. Restricting first to the sample interval and then convolving would
be a different operation and is not used here.

## 3. A finite filter with an explicit remainder

Choose $T>R:=\max_{i,j}|x_i-c_j|$ and define the $2T$-periodic square wave
$s_T(t)=\operatorname{sign}(\sin(\pi t/T))$. Its Fourier series has only
odd sine terms, with coefficient $4/(\pi(2\ell-1))$. Smooth it with the
whole-line density from Lemma 1. The resulting series is absolutely and
uniformly convergent because (4) decays exponentially. Its first $Q$ terms are

$$
h_{\gamma,Q}(t)=\sum_{\ell=1}^Q a_\ell
M_\gamma(\omega_\ell)\sin(\omega_\ell t),\qquad
a_\ell=\frac4{\pi(2\ell-1)},\quad
\omega_\ell=\frac{(2\ell-1)\pi}{T}.
\tag{6}
$$

**Lemma 2 (controlled approximation of nonperiodic tanh).** Put
$a=\pi^2/(2\gamma T)$ and $u=a(2Q+1)$. Uniformly for $|t|\le R$,

$$
|\tanh(\gamma t)-h_{\gamma,Q}(t)|\le\tau_{\gamma,Q}
:=4e^{-2\gamma(T-R)}+
\frac{4\pi}{\gamma T}
\frac{e^{-u}}{(1-e^{-2u})(1-e^{-2a})}.
\tag{7}
$$

**Proof.** The square wave and the ordinary sign function agree on $(-T,T)$.
Their difference has magnitude at most two outside this interval. Therefore
the difference of their smoothed versions is at most
$1-\tanh(\gamma(T-t))+1-\tanh(\gamma(T+t))$, bounded by the first term of (7).
For an odd harmonic $k$, its smoothed coefficient has magnitude
$2\pi\operatorname{csch}(ak)/(\gamma T)$. For omitted $k\ge2Q+1$,
$\operatorname{csch}(ak)\le2e^{-ak}/(1-e^{-2u})$.
Sum the geometric series with ratio $e^{-2a}$ to obtain the second term.

This auxiliary periodic expansion does not impose periodic training or move
the centers. Both its distant transition and its truncation are accounted
for relative to the original tanh model.

**Theorem 1 (finite gamma-factorized raw kernel).** Define $F_Q$ with columns
$1,\sin(\omega_1x),\cos(\omega_1x),\ldots$, sampled and divided by $\sqrt m$.
Let $C_Q$ have bias row $(1,0,\ldots,0)$, zero hidden-row bias entries, and
hidden-column entries

$$
(C_Q)_{2\ell-1,j}=a_\ell\cos(\omega_\ell c_j),\qquad
(C_Q)_{2\ell,j}=-a_\ell\sin(\omega_\ell c_j).
$$

Let $D_{\gamma,Q}$ have diagonal
$1,M_\gamma(\omega_1),M_\gamma(\omega_1),\ldots$. Then

$$
\boxed{\widetilde J_{\gamma,Q}=F_QD_{\gamma,Q}C_Q,\qquad
\widetilde K_{\gamma,Q}=F_QD_{\gamma,Q}G_QD_{\gamma,Q}^TF_Q^T,
\quad G_Q=C_QC_Q^T.}
\tag{8}
$$

Only $D_{\gamma,Q}$ depends on gamma. The original $W+1$ readout coordinates
are retained, all entries of $G_Q$ are retained, and

$$
\|J_\gamma-\widetilde J_{\gamma,Q}\|\le\delta_J:=\sqrt W\tau_{\gamma,Q},
\qquad
\|K_\gamma-\widetilde K_{\gamma,Q}\|
\le\Delta:=(2\|\widetilde J_{\gamma,Q}\|+\delta_J)\delta_J.
\tag{9}
$$

**Proof.** Expand $\sin(\omega(x-c))$ in (6). This gives exactly the stated
matrix product. The bias is exact. Each normalized hidden column has error
norm at most $\tau_{\gamma,Q}$; sum their squared norms to bound the
Frobenius, hence spectral, error. With $H=J-\widetilde J$, expand
$K-\widetilde K=H\widetilde J^T+\widetilde JH^T+HH^T$ to prove (9).

The factorization retains a generally dense $G_Q$ and a nonorthogonal
sampled $F_Q$. Consequently multiplying individual finite-kernel eigenvalues
by $M_\gamma^2$ is not justified. Computing the spectrum of (8) preserves
the explicit gamma mechanism and these finite-geometry effects together.

## 4. From the same filtered kernel to acquisition times

**Theorem 2 (gamma-filter acquisition curve).** Assume both kernels contract
at the actual prescribed step: $0\preceq\eta_\gamma K_\gamma\preceq I$ and
$0\preceq\eta_\gamma\widetilde K_{\gamma,Q}\preceq I$. Let
$\widetilde K u_\ell=\widetilde\lambda_\ell u_\ell$ be its positive modes,
$\alpha_\ell=u_\ell^Ty$, $y_\perp=y-\sum\alpha_\ell u_\ell$, and
$q_\ell=1-\eta_\gamma\widetilde\lambda_\ell$. Then

$$
\widetilde E_n^2=
\frac{\|y_\perp\|^2+\sum_\ell\alpha_\ell^2q_\ell^{2n}}{\|y\|^2},
\qquad
|E_n-\widetilde E_n|\le d_n,
\tag{10}
$$

where, writing $A=K_\gamma-\widetilde K_{\gamma,Q}$,

$$
d_n=\min\left\{n\eta_\gamma\Delta,
\frac{\eta_\gamma}{\|y\|}\left[
n\|Ay_\perp\|+\sum_\ell|\alpha_\ell|\|Au_\ell\|
\sum_{j=0}^{n-1}q_\ell^j\right]\right\}.
\tag{11}
$$

**Proof.** Diagonalize the retained PSD kernel to obtain the exact residual
formula. For $U=I-\eta K$ and $V=I-\eta\widetilde K$, telescope
$U^n-V^n=\sum_{j=0}^{n-1}U^{n-1-j}(U-V)V^j$. Contraction and (9) give the
first error bound. Applying the same identity to $y$, expanding $V^jy$
into its retained modes and perpendicular component, and using the triangle
inequality gives the second. No commutation or shared eigenvectors are
assumed. This is the operator comparison proved in the
[polynomial-kernel note](common_slope_polynomial_kernel.md), now applied to
the explicit construction (8).

The analytic radius uses only the known factorization and remainder. The
action radius additionally evaluates the original synthesis operator on
retained residual directions. Neither uses observed optimizer iterates.
Compute these actions as
$Az=(J-\widetilde J)(\widetilde J^Tz)+J((J-\widetilde J)^Tz)$ to avoid
subtracting two nearly equal kernel products.

**Corollary (necessary and sufficient updates).** Set
$\ell(n)=\max(0,\widetilde E_n-d_n)$ and
$u(n)=\min(1,\widetilde E_n+d_n)$. If $\ell(a)>\epsilon$ and
$u(b)\le\epsilon$, then

$$
\boxed{a+1\le n_\epsilon\le b.}
\tag{12}
$$

The true residual is nonincreasing. The upper envelope need not be, so the
upper count must be a checked witness. Valid endpoints can be combined across
$Q$. For positive finite bounds $L_g\le n_\epsilon(g)\le U_g$, two slopes
satisfy $L_g/U_h\le n_\epsilon(g)/n_\epsilon(h)\le U_g/L_h$.

Equations (4), (8), and (10)–(12) are the promised quantitative bridge:
gamma changes an explicit filter on a fixed geometry, and that same filtered
kernel determines a certified target-dependent acquisition interval. Sharpness
of its numerical evaluation remains an empirical question. No monotonicity
of target acquisition with gamma is assumed.

## 5. Numerical implementation and evidence roles

The implementation is
[`gamma_filter.py`](../experiments/expD36_frozen_gamma_probe/gamma_filter.py).
It stores $F_Q,C_Q$ once per resolution, retains $G_Q$ implicitly through $C_Q$,
and forms the rectangular filtered synthesis matrix. Short matrix products
and pairwise accumulation limit summation roundoff. A rectangular SVD avoids
forming the Gram matrix solely to find its small eigenvalues.

The exact-arithmetic remainder in (7) is separate from floating-point
sensitivity. The implementation exposes an operation-count allowance for
matrix summation and an assumed transcendental/phase allowance, followed by
SVD reconstruction and orthogonality diagnostics. These are not interval
enclosures. Selected endpoint claims require the independent nominal-real
Arb audit; that audit does not certify every plotted error envelope.

The empirical comparison will reuse the archived four-gamma, five-target
study with its exact targets, centers, samples, initialization, and saved
steps. Its role is retrospective validation of optimization on those training
samples, not held-out generalization or fitted prediction of GD trajectories.
