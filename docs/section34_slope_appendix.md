# Proof of the slope-dependent output-error bound

The result connects three objects: the smoothing induced by a common tanh slope, the eigenvalues of the finite readout kernel, and the output error that remains under gradient descent. The continuous-center integral exposes the slope dependence. An explicit lattice allowance transfers it to finitely many equally spaced centers; interlacing then supplies upper bounds on the relative rates in the exact GD error formula. This appendix proves that connection and separates analytic guarantees from their numerical evaluation.

## Setup and statement

Fix distinct samples $z_1,\ldots,z_m$ and $W$ consecutive centers $x_j$ on the lattice $c_0+h\mathbb Z$, with $h>0$. Assume the retained centers cover the sample interval. Fix a common slope $\gamma>0$. Include an output bias and define

$$
J_\gamma=\frac1{\sqrt m}[\mathbf1,\ (\tanh(\gamma(z_a-x_j)))_{a,j}],
\qquad K_\gamma=J_\gamma J_\gamma^T,
\qquad y_a=\frac{f(z_a)}{\sqrt m}.
$$

Thus $\|J_\gamma w-y\|^2$ is mean squared output error on the samples. Assume $y\ne0$. Write the descending eigenpairs of $K_\gamma$ as $(\mu_i,u_i)_{i=1}^m$, and set

$$
\rho_i=\frac{\mu_i}{\mu_1},\qquad
p_i=\frac{|u_i^Ty|^2}{\|y\|^2},\qquad
v_0=\frac{\mathbf1}{\sqrt m}.
$$

Here $\sum_i p_i=1$, including any target energy in the exact nullspace. Eigenvalues use $\mu$; the dimensionless bandwidth remains $\lambda=\gamma h$.

Let $Q\in\mathbb R^{m\times(m-1)}$ have orthonormal columns spanning $v_0^\perp$. For the sample vector $t_\gamma(c)=(\tanh(\gamma(z_a-c)))_a$, define

$$
H_\gamma=\frac1{hm}\int_{\mathbb R}Q^Tt_\gamma(c)t_\gamma(c)^TQ\,dc
=-\frac2{hm}Q^T[d_{ab}\coth(\gamma d_{ab})]_{a,b}Q,
\qquad d_{ab}=z_a-z_b,
$$

where the diagonal value is $1/\gamma$. Retain a finite set $\mathcal F$ of absent exterior lattice centers, so that the absent centers not in $\mathcal F$ form two tails starting at $c_L<z_{\min}$ and $c_R>z_{\max}$. Put

$$
T_\gamma=\frac1m\sum_{c\in\mathcal F}Q^Tt_\gamma(c)t_\gamma(c)^TQ,
\qquad S_\gamma=H_\gamma-T_\gamma.
$$

Let $\beta_1\ge\cdots\ge\beta_{m-1}$ be the eigenvalues of $S_\gamma$. For any $0<\vartheta<\pi/2$, define

$$
\delta_\gamma=
\frac{8\gamma\|z-\bar z\mathbf1\|^2\sec^4\vartheta}
{3hm[\exp(2\pi\vartheta/(\gamma h))-1]},
\qquad \bar z=\frac1m\sum_a z_a,
\qquad \ell_\gamma=v_0^TK_\gamma v_0>0.
$$

**Theorem (slope-dependent output-error lower bound).** Set $\bar\rho_1=1$ and

$$
\bar\rho_i(\gamma)=\min\left\{1,
\frac{\beta_{i-1}(\gamma)+\delta_\gamma}{\ell_\gamma}\right\},
\qquad 2\le i\le m.
$$

Then $\rho_i\le\bar\rho_i\le1$. Starting at $w_0=0$, gradient descent on $\frac12\|J_\gamma w-y\|^2$ with $\eta=1/(2\mu_1)$ satisfies, for every integer $n\ge0$,

$$
\boxed{
\frac{\|J_\gamma w_n-y\|^2}{\|y\|^2}
\ge\sum_{i=1}^m p_i(\gamma)
\left(1-\frac{\bar\rho_i(\gamma)}2\right)^{2n}.
}
$$

Known exact zero eigenvalues may have their rate endpoints set to zero; in particular, $\operatorname{rank}K_\gamma\le W+1$. The theorem is conditional on target energy in the bounded slow directions. It does not assert that every target is slow at small slope.

## 1. The continuous-center kernel exposes gamma

Removing the constant sample pattern makes the center integral finite: $Q^Tt_\gamma(c)$ tends exponentially to zero at both ends of the real line. For $d=z_a-z_b$, the identity

$$
1-\tanh u\tanh v=\coth(u-v)(\tanh u-\tanh v)
$$

and integration in $c$ give

$$
\int_{\mathbb R}[1-\tanh(\gamma(z_a-c))\tanh(\gamma(z_b-c))]\,dc
=2d\coth(\gamma d).
$$

The diagonal limit is $2/\gamma$. Multiplication by $Q^T$ and $Q$ cancels the constant term and proves the displayed expression for $H_\gamma$.

To see what this matrix does to frequencies, write

$$
s_\gamma(t)=\frac\gamma2\operatorname{sech}^2(\gamma t),
\qquad \tanh(\gamma\,\cdot)=s_\gamma*\operatorname{sign}.
$$

The convolution identity follows because both sides vanish at zero and have derivative $2s_\gamma$. With Fourier convention $\widehat g(\omega)=\int g(t)e^{-i\omega t}\,dt$, substitution $r=e^{2\gamma t}$ yields

$$
\widehat s_\gamma(\omega)
=\int_0^\infty\frac{r^{-i\omega/(2\gamma)}}{(1+r)^2}\,dr
=\Gamma\!\left(1-\frac{i\omega}{2\gamma}\right)
\Gamma\!\left(1+\frac{i\omega}{2\gamma}\right)
=\frac{\pi\omega/(2\gamma)}{\sinh(\pi\omega/(2\gamma))}
=:M_\gamma(\omega),
$$

with $M_\gamma(0)=1$. The middle identities follow from the beta integral and gamma-function reflection formula.

For $u=Qv$, the function $g_0(c)=\sum_a u_a\operatorname{sign}(c-z_a)$ has compact support because $\sum_a u_a=0$, and its distributional derivative is $2\sum_a u_a\delta_{z_a}$. Therefore

$$
\widehat g_0(\omega)=\frac2{i\omega}\sum_a u_a e^{-i\omega z_a},
\qquad
\boxed{
v^TH_\gamma v=\frac2{\pi hm}\int_{\mathbb R}
\frac{M_\gamma(\omega)^2}{\omega^2}
\left|\sum_a(Qv)_a e^{-i\omega z_a}\right|^2\,d\omega.
}
$$

Parseval proves the second identity. The apparent singularity at zero is removable because the sum in the numerator vanishes there. This formula identifies the direction being tested: its sampled coefficients $(Qv)_a$ determine the frequency content inside the absolute square; gamma attenuates that content through $M_\gamma^2$.

For $z>0$, $d\log(z/\sinh z)/dz=1/z-\coth z<0$. Hence increasing gamma increases $M_\gamma(\omega)$ at each fixed frequency, and

$$
\Gamma\ge\gamma\quad\Longrightarrow\quad H_\Gamma\succeq H_\gamma.
$$

The min-max principle gives ordered growth of the eigenvalues of $H_\gamma$. This statement permits its eigenvectors to change. Subtraction of exterior centers and normalization by the largest finite eigenvalue do not preserve this monotonicity in general.

## 2. Transfer from continuous centers to the finite lattice

For $u\perp\mathbf1$, set $g(\zeta)=\sum_a u_a\tanh(\gamma(\zeta-z_a))$. This function is analytic in $|\operatorname{Im}\zeta|<\pi/(2\gamma)$. Subtract the common translate at $\bar z$, express each difference as the integral of its derivative, and apply Minkowski followed by Cauchy-Schwarz. On either contour $\operatorname{Im}\zeta=\pm\vartheta/\gamma$,

$$
\int_{\mathbb R}|g(t\pm i\vartheta/\gamma)|^2\,dt
\le\frac{4\gamma}{3}\sec^4\vartheta
\|z-\bar z\mathbf1\|^2\|u\|^2.
$$

Indeed, $|\operatorname{sech}^2(t+i\vartheta)|\le\sec^2\vartheta\operatorname{sech}^2t$ and $\int_{\mathbb R}\operatorname{sech}^4t\,dt=4/3$, so the $L^2$ norm of each translate difference is at most $|z_a-\bar z|\sqrt{4\gamma/3}\sec^2\vartheta$.

Apply contour shifting to the analytic function $g(\zeta)^2$, rather than to $|g(\zeta)|^2$. Its Fourier transform at frequency $\omega$ is bounded by the preceding integral times $e^{-\vartheta|\omega|/\gamma}$. Poisson summation on $c_0+h\mathbb Z$ consequently bounds the discrepancy between the lattice sum of $g^2$ and its integral divided by $h$. Summing the nonzero frequencies $2\pi k/h$ in both signs, and dividing by $m$, gives

$$
\left\|\frac1m\sum_{c\in c_0+h\mathbb Z}
Q^Tt_\gamma(c)t_\gamma(c)^TQ-H_\gamma\right\|
\le\delta_\gamma.
$$

The lattice origin contributes only unit-modulus phases to this estimate.

The retained finite kernel removes every absent center from that full lattice. Write the remaining, unretained exterior contribution as $T_{\rm tail}$. Since $Q^T\mathbf1=0$ and $1-\tanh t\le2e^{-2t}$ for $t\ge0$, each right-tail outer product divided by $m$ has norm at most $4e^{-4\gamma(c-z_{\max})}$; the left side is analogous. Geometric summation yields

$$
0\preceq T_{\rm tail}\preceq\tau_\gamma I,
\qquad
\tau_\gamma=
\frac{4[e^{-4\gamma(c_R-z_{\max})}+e^{-4\gamma(z_{\min}-c_L)}]}
{1-e^{-4\gamma h}}.
$$

Consequently,

$$
Q^TK_\gamma Q=S_\gamma+R_{\rm lattice}-T_{\rm tail},
\qquad \|R_{\rm lattice}\|\le\delta_\gamma,
$$

and

$$
\boxed{
S_\gamma-(\delta_\gamma+\tau_\gamma)I
\preceq Q^TK_\gamma Q
\preceq S_\gamma+\delta_\gamma I.
}
$$

The bias vanishes under $Q$. The exterior tail only lowers the finite kernel, so it is not needed in the upper eigenvalue bound used for the output-error lower bound.

## 3. Relative rates and output error

Let $\kappa_i$ be the ordered eigenvalues of $Q^TK_\gamma Q$. The preceding matrix enclosure and codimension-one interlacing give

$$
\beta_i-\delta_\gamma-\tau_\gamma\le\kappa_i\le\beta_i+\delta_\gamma,
\qquad
\mu_i\ge\kappa_i\ge\mu_{i+1}.
$$

Thus $\mu_i\le\beta_{i-1}+\delta_\gamma$ for $i\ge2$. In exact arithmetic this upper endpoint is nonnegative, since it bounds a nonnegative eigenvalue. The Rayleigh quotient $\ell_\gamma=v_0^TK_\gamma v_0=\|J_\gamma^Tv_0\|^2$ is at most $\mu_1$ and is at least one because the bias is present. Dividing by $\ell_\gamma$ and also using $\rho_i\le1$ proves the stated relative-rate upper bound.

The GD residual $r_n=J_\gamma w_n-y$ obeys

$$
r_{n+1}=(I-\eta K_\gamma)r_n,\qquad r_0=-y,
\qquad
\frac{\|r_n\|^2}{\|y\|^2}=
\sum_i p_i(1-\eta\mu_i)^{2n}.
$$

At $\eta=1/(2\mu_1)$, each factor is $(1-\rho_i/2)^{2n}$, a decreasing function of $\rho_i\in[0,1]$. Replacing $\rho_i$ by its upper bound and summing against $p_i\ge0$ proves the theorem. The chosen step is a fixed conservative fraction of the stability limit $2/\mu_1$, not the maximal stable step. For any nonoscillatory step $0<\eta\mu_1\le1$, the same proof replaces $\rho/2$ by $\eta\mu_1\rho$.

Only energy in the **exact** nullspace contributes a permanent floor $p_0=\sum_{\mu_i=0}p_i$. A mode with $\mu_i>0$ is representable because $u_i=J_\gamma(J_\gamma^Tu_i/\mu_i)$, yet it may decay slowly. The theorem measures that distinction. The $p_i$ are projections onto the actual finite-kernel eigenvectors; replacing them by continuous-center eigenvectors is not justified by this proof.

## 4. Numerical evaluation and sources of slack

The analytic statement uses the exact $H_\gamma$ and $S_\gamma$. Numerical evaluation uses an equivalent positive center-quadrature factor $A$ and an exterior factor $C$, forming $AA^T-CC^T$. It does not truncate a Fourier series. The Fourier representation establishes the mechanism; quadrature evaluates the same integral.

Truncating the integral at $\pm(\max_a|z_a|+P/\gamma)$ omits a positive matrix with norm at most $2e^{-4P}/(h\gamma)$. This tail enters the numerical upper endpoint. The exterior tail enters the lower side of the kernel enclosure. If $Z=[A,C]$ and $Z_r$ is its truncated SVD, then, with $D=\operatorname{diag}(I,-I)$,

$$
\|ZDZ^T-Z_rDZ_r^T\|
\le2\sigma_1(Z)\sigma_{r+1}(Z)+\sigma_{r+1}(Z)^2.
$$

This follows by writing $Z=Z_r+E$, expanding, and using $\|E\|=\sigma_{r+1}$. The allowance applies on both sides. Pad the compressed spectrum with zero eigenvalues between its positive and negative eigenvalues to recover dimension $m-1$.

Quadrature error and floating-point arithmetic are separate from these analytic tails. Refining quadrature and checking sensitivity supports a numerical evaluation; it does not supply a rigorous error certificate. A chosen multiple of machine epsilon is an empirical guard unless accompanied by a proved arithmetic-error analysis. In particular, clipping a negative computed upper endpoint to zero cannot repair a violated enclosure. The publication should identify computed curves as checked floating-point evaluations of the theorem.

The lower curve omits energy in numerically unresolved modes; this weakens it conservatively. The directly computed projection residual and the unresolved spectral energy must remain separately reported. Neither alone establishes an exact nonrepresentability floor in finite precision.

The bound can be looser than executed GD for identifiable reasons: interlacing shifts an eigenvalue index when the constant direction is removed; $\ell_\gamma$ can underestimate $\mu_1$; positive lattice and numerical allowances can dominate very small eigenvalues; and omitting unresolved target energy lowers the prediction further. Retaining more exterior centers and improving numerical accuracy reduces some of this slack, but not the interlacing shift or the Rayleigh normalization gap. These distinctions separate an inaccurate evaluation from a conservative analytic bound.

All conclusions concern fixed features and the specified sample norm. Population error, evolving centers/slopes, and Adam require additional analysis or empirical evidence.
