# Mathematical review of the finite-gamma ratio note

Read-only audit of the supplied backbone and existing implementation/data. Only this review memo was written. No numerical files were changed and no training was run.

## Verdict and required qualifications

The backbone's spatial decomposition, all-vector finite-change Fourier identity, relative-correction min–max bound, and positive-target-band GD bound are correct. The following qualifications should be explicit in the polished note.

1. Use **\(W\)** for the total number of tanh neurons, including the halo. The experiments use \(N\) for the number of core intervals, with \(h=2/N\) and \(W=N+1+2\lceil\sqrt N\rceil\). There are \(W+1\) feature columns after adding the bias.
2. For centers \(c_j=c_1+(j-1)h\), define the integration interval by its **cell edges**, \(c_-=c_1-h/2\) and \(c_+=c_W+h/2\). Its length is \(Wh\). The interval between the first and last centers instead has length \((W-1)h\).
3. A lower bound on a new eigenvalue ratio establishes recovery only when it exceeds the old ratio. It does not establish small-gamma training delay. That delay requires an upper cutoff on positive rates together with target energy in those rates.
4. Corrections in the reported calculations are evaluated from the new finite kernel. These are evaluations of an exact inequality, not an a priori law depending on gamma alone and not interval-certified numerical bounds.
5. The full target is shown to attain the requested accuracy, not to have zero approximation error. The component in any strictly positive spectral band is exactly representable on the sample grid.

## Spatial decomposition and the whole-line deficit

Set \(d=x_a-x_b\) and \(g_{ab}(c)=\tanh(\gamma(x_a-c))\tanh(\gamma(x_b-c))\). For \(d\ne0\), the elementary identity

\[
1-\tanh u\tanh v
=\coth(u-v)(\tanh u-\tanh v)
\]

gives

\[
\int_{\mathbb R}[1-g_{ab}(c)]\,dc
=\coth(\gamma d)\int_{\mathbb R}
[\tanh(\gamma(x_a-c))-\tanh(\gamma(x_b-c))]\,dc
=2d\coth(\gamma d).
\]

The last integral is \(2d\), by evaluating the difference of the log-cosh antiderivatives at both infinities. For \(d=0\), direct integration of \(\operatorname{sech}^2(\gamma(x_a-c))\) gives \(2/\gamma\), agreeing with the continuous limit.

With midpoint-cell integration bounds, define

\[
(K_\gamma^{\rm int})_{ab}
=\frac1m\left[1+\frac1h\int_{c_-}^{c_+}g_{ab}(c)\,dc\right],
\qquad E_\gamma^{\rm grid}=K_\gamma-K_\gamma^{\rm int}.
\]

Splitting the deficit integral into the whole line minus its omitted tails yields the exact decomposition

\[
K_\gamma=K_\gamma^{(0)}+E_\gamma^{\rm halo}+E_\gamma^{\rm grid},
\qquad
(K_\gamma^{(0)})_{ab}
=\frac{W+1}{m}-\frac{2}{hm}d\coth(\gamma d),
\]

\[
(E_\gamma^{\rm halo})_{ab}
=\frac1{hm}\int_{\mathbb R\setminus[c_-,c_+]}
[1-g_{ab}(c)]\,dc.
\]

All integrals converge. The diagonal limit in \(K^{(0)}\) is \(1/\gamma\) before multiplication by \(-2/(hm)\). Neither \(K^{(0)}\) nor the full halo correction is asserted to be positive semidefinite. In particular, entrywise positivity of the halo deficit is not matrix positivity: for a zero-sum vector \(q\),

\[
q^\top E_\gamma^{\rm halo}q
=-\frac1{hm}\int_{\mathbb R\setminus[c_-,c_+]}
\left|\sum_aq_a\tanh(\gamma(x_a-c))\right|^2dc\le0.
\]

## Appendix-ready proof of the Fourier identity

Use the Fourier convention \(\widehat f(\omega)=\int_{\mathbb R}f(t)e^{-i\omega t}\,dt\). The probability density

\[
\rho_\gamma(t)=\frac\gamma2\operatorname{sech}^2(\gamma t)
\]

has cumulative distribution \((1+\tanh(\gamma t))/2\), so \(\rho_\gamma*\operatorname{sgn}=\tanh(\gamma\,\cdot)\). Its transform is

\[
M_\gamma(\omega)=\frac{z}{\sinh z},\qquad
z=\frac{\pi|\omega|}{2\gamma},\qquad M_\gamma(0)=1.
\]

One concise verification substitutes \(s=e^{2\gamma t}\), obtaining
\(\widehat\rho_\gamma(\omega)=\int_0^\infty s^{-i\nu}(1+s)^{-2}ds
=\Gamma(1-i\nu)\Gamma(1+i\nu)=\pi\nu/\sinh(\pi\nu)\), where \(\nu=\omega/(2\gamma)\). The last equality follows from the gamma reflection formula; the zero value follows by continuity.

First take \(q\) with \(\sum_aq_a=0\), and put

\[
g_{\gamma,q}(c)=\sum_aq_a\tanh(\gamma(x_a-c)),\qquad
F_q(\omega)=\sum_aq_ae^{-i\omega x_a}.
\]

Both tails of \(g_{\gamma,q}\) decay exponentially. It is square integrable, and differentiation gives
\(\widehat{g_{\gamma,q}'}=-2M_\gamma F_q\), hence
\(\widehat g_{\gamma,q}=-2M_\gamma F_q/(i\omega)\) away from zero. Parseval and the whole-line deficit identity therefore give

\[
\sum_{a,b}q_aq_b[-2d_{ab}\coth(\gamma d_{ab})]
=\int_{\mathbb R}|g_{\gamma,q}(c)|^2dc
=\frac2\pi\int_{\mathbb R}
\frac{M_\gamma(\omega)^2}{\omega^2}|F_q(\omega)|^2d\omega.
\]

The zero-sum condition is needed for this individual-energy formula. It will not be needed for a finite change.

Fix \(0<\gamma_a<\gamma_b\), and define

\[
D(t)=2[t\coth(\gamma_a t)-t\coth(\gamma_b t)],
\qquad
f(\omega)=\frac{M_{\gamma_b}(\omega)^2-M_{\gamma_a}(\omega)^2}{\omega^2}.
\]

The values at zero are

\[
D(0)=2(\gamma_a^{-1}-\gamma_b^{-1}),\qquad
f(0)=\frac{\pi^2}{12}(\gamma_a^{-2}-\gamma_b^{-2}).
\]

The second formula follows from \(M_\gamma(\omega)^2=1-\pi^2\omega^2/(12\gamma^2)+O(\omega^4)\). Thus \(f\) is continuous at zero and integrable on the whole line; its tails decay exponentially. Also \(D(t)\to0\) as \(|t|\to\infty\).

Apply the zero-sum formula at the two samples \(0,t\), with coefficients \((1,-1)\), and subtract the two gamma values. This gives

\[
D(0)-D(t)=\frac2\pi\int_{\mathbb R}f(\omega)(1-\cos(\omega t))\,d\omega.
\]

The Riemann–Lebesgue lemma and \(D(t)\to0\) imply
\(D(0)=(2/\pi)\int f\). Subtracting yields

\[
D(t)=\frac2\pi\int_{\mathbb R}f(\omega)\cos(\omega t)\,d\omega.
\]

This step fixes the possible additive constant explicitly. Expanding the finite quadratic form now proves, for **every** real vector \(u\),

\[
\boxed{
u^\top(K_{\gamma_b}^{(0)}-K_{\gamma_a}^{(0)})u
=\frac2{\pi hm}\int_{\mathbb R}f(\omega)
\left|\sum_a u_ae^{-i\omega x_a}\right|^2d\omega.
}
\]

There is no omitted constant or rank-one term and no need for a distributional transform of tanh. Since \(M_\gamma\) increases with gamma, \(f\ge0\), establishing \(\Delta K^{(0)}\succeq0\). This does not assert that the actual corrected kernel change is positive semidefinite or that normalized eigenvalues improve automatically.

For the stated relative-frequency interpretation, \(z\coth z\) is strictly increasing for \(z>0\). Equivalently, the finite gain \(M_{\gamma_b}(\omega)^2/M_{\gamma_a}(\omega)^2\) is strictly increasing with \(|\omega|>0\). No inference identifying Fourier waves with finite-matrix eigenvectors is required or justified.

## Min–max theorem with a relative correction

Let \(U_i\) be an orthonormal basis of an old top-\(i\) eigenspace and assume \(\lambda_i(K_{\gamma_a})>0\). Define

\[
C_i=\operatorname{diag}(\lambda_1(K_{\gamma_a}),\ldots,\lambda_i(K_{\gamma_a}))
+U_i^\top\Delta K^{(0)}U_i,
\qquad
R_i=U_i^\top(E_{\gamma_b}-E_{\gamma_a})U_i.
\]

Then \(C_i\succ0\) and \(U_i^\top K_{\gamma_b}U_i=C_i+R_i\). For symmetric \(R_i\),

\[
\varepsilon_i=
\sup_{z\ne0}\frac{|z^\top R_i z|}{z^\top C_i z}
=\|C_i^{-1/2}R_iC_i^{-1/2}\|_2
\]

implies \(R_i\succeq-\varepsilon_iC_i\). If \(\varepsilon_i<1\), this bounds the smallest eigenvalue of \(C_i+R_i\) below by \((1-\varepsilon_i)\lambda_{\min}(C_i)\). If \(\varepsilon_i\ge1\), use \(C_i+R_i=U_i^\top K_{\gamma_b}U_i\succeq0\) instead. Courant–Fischer therefore gives

\[
\frac{\lambda_i(K_{\gamma_b})}{\lambda_1(K_{\gamma_b})}
\ge\frac{[1-\varepsilon_i]_+\lambda_{\min}(C_i)}{L_b},
\quad L_b\ge\lambda_1(K_{\gamma_b})>0.
\]

New eigenvectors may rotate or cross. The old subspace is simply an admissible trial subspace. Valid denominator choices include \(W+1\) and the maximum **absolute** row sum of the new kernel.

For an arbitrary orthonormal trial basis, replace the diagonal old-eigenvalue block with \(U_i^\top K_{\gamma_a}U_i\) and require \(C_i\succ0\). This is the appropriate formulation for reference singular vectors that are not reliably identified numerically. The width-grid script uses this projected baseline.

## GD consequences and the direction of the bounds

With zero readout and \(\eta=1/(2\lambda_1)\), define \(r_j=\lambda_j/\lambda_1\) and \(p_j=|u_j^\top y|^2/\|y\|^2\). Exact-arithmetic GD satisfies

\[
E(n)^2=\sum_jp_j(1-r_j/2)^{2n},
\]

including the nullspace contribution. If a fraction \(p\) lies in \(0<r_j\le\rho\), then

\[
E(n)\ge\sqrt p(1-\rho/2)^n,
\qquad
n\ge\left\lceil\frac{\log(\sqrt p/\epsilon)}{-\log(1-\rho/2)}\right\rceil
\quad(\epsilon<\sqrt p).
\]

The positive-band component is representable because \(\lambda_j>0\) implies \(u_j=J(J^\top u_j/\lambda_j)\). The leading approximation \((2/\rho)\log(\sqrt p/\epsilon)\) requires small \(\rho\).

Conversely, established lower ratio bounds \(\ell_j\le r_j\), together with the corresponding target weights, give an **upper** residual bound using \((1-\ell_j/2)^{2n}\). Uncovered target energy and nullspace energy must still be retained. Thus one lower ratio bound cannot establish a full-target convergence time, and the recovery theorem cannot supply the separate necessary-delay claim at small gamma.

## Numerical support inspected

The relevant geometry in the saved data is \(N=128\), \(W=153\), \(m=263\), \(h=1/64\), with 12 halo centers per side and integration bounds \([-1.1953125,1.1953125]\).

- `finite_ratio_bound.json` supports rank 32, gamma \(8\to16\): old ratio \(4.93422493\times10^{-9}\), evaluated lower bound \(3.40343856\times10^{-6}\), actual ratio \(6.30457701\times10^{-6}\).
- `finite_ratio_width_grid.json` supports rank 32, gamma \(16\to64\): evaluated lower bound \(3.39157527\times10^{-4}\), actual ratio \(4.66982383\times10^{-4}\). This second comparison uses a reference at 16. Keeping the reference at 8 instead gives the different lower bound \(1.93032629\times10^{-4}\).
- `slow_band_check.json` reports at gamma 4 a fraction \(p=0.02735465123\) in positive per-step rates \((10^{-14},10^{-11}]\), yielding 280,573,583,534 necessary steps to 1% by the band inequality. The full spectral forecast is 1,007,616,497,993 steps. An explicit resolved-mode readout has relative residual \(0.0005850007883\), demonstrating that 1% is attainable.
- The same file reports the gamma-64 full spectral forecast as 18,272 steps. These are spectral calculations, not executed optimizer trajectories or interval-certified counts. The gamma-4 narrow band was selected during the diagnostic.

The scripts compute corrections from the new finite kernel and use its eigenvalues only for validation/comparison of the ratio lower bound. The exact reconstruction check is partly algebraic because the remainder is formed as the difference of the compressed actual kernel and \(C_i\); independent whitening/generalized-eigenvalue agreement and comparison with rectangular-SVD eigenvalues are the more substantive numerical checks. None turns FP64 output into a directed-rounding certificate.

The supported conclusion is finite and geometry-dependent: the explicit multiplier supplies a positive matrix change, and the measured correction bounds are small enough in specified comparisons to establish large increases in relevant normalized eigenvalues. Separately, positive target energy at tiny rates establishes a long necessary delay even though a sufficiently accurate readout exists. No universal ratio ordering, uniformly tight bound, or claim about momentum, preconditioning, or changing hidden features follows.

## Addition: every scalar schedule within the stability interval

The gamma-4 obstruction is not specific to the choice \(\eta_0=1/(2\lambda_1)\). Let the kernel remain fixed and permit any scalar schedule satisfying

\[
0\le\eta_t\le\frac2{\lambda_1}\qquad(t=0,1,\ldots).
\]

This includes state-dependent choices of the scalar step. Each eigencomponent of the residual is multiplied by \(\prod_{t<n}(1-\eta_t\lambda_j)\). On the saved positive band, \(\eta_0\lambda_j\le b=10^{-11}\), so

\[
0\le\eta_t\lambda_j\le4b,\qquad
\prod_{t<n}(1-\eta_t\lambda_j)\ge(1-4b)^n>0.
\]

Orthogonality therefore gives

\[
E(n)\ge\sqrt p(1-4b)^n,
\qquad
n\ge\left\lceil\frac{\log(\sqrt p/0.01)}{-\log(1-4b)}\right\rceil.
\]

Using the saved \(p=0.027354651231198325\), evaluating the denominator as \(-\operatorname{log1p}(-4b)\) gives the unrounded value \(70{,}143{,}395{,}882.35764\), hence the evaluated integer lower bound **70,143,395,883 updates**, approximately \(7.014\times10^{10}\). The count inherits the numerical, non-interval-certified status of the saved band mass. The exact mathematical inequality applies to every such scalar schedule; it makes no claim about momentum, preconditioning, changing features, or schedules that leave the stated stability interval.

The analytic deficit and finite-change Fourier formulas still do not imply normalized ordering automatically. They establish positivity of \(\Delta K^{(0)}\); actual corrections and growth of the largest eigenvalue must be controlled before concluding that a particular \(\lambda_i/\lambda_1\) increases.
