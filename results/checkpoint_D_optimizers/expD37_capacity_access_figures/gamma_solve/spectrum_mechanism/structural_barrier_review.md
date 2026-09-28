# Fourier bounds for a finite-tanh optimization barrier

Independent mathematical review, September 23, 2026. This report uses the positive Fourier quadratic form and a Fourier cutoff. It contains no polynomial approximation argument. No training was run and no other repository files were changed for this report.

**Status: exploratory; not selected as the explanation for the studied experiment.** The coordinator's finite-geometry check found the cutoff/min–max bound too loose: at gamma 8, the rank-32 normalized bound was approximately \(5.9\times10^{-6}\), versus an actual ratio of \(4.93\times10^{-9}\), a factor of about 1,200. The rank-48 bound was loose by about four orders of magnitude. These are reported numerical evaluations, not independently certified intervals. The valid inequalities below are retained as review material; they should not be presented as solving the requested quantitative mechanism problem.

The cutoff comparison below is exact. Combined with an explicit bound for sampling the center line, it gives a sufficient obstruction to fast Euclidean readout GD without computing the actual kernel eigenvectors at every slope. Its practical strength depends on measured low-frequency leakage and target overlap; the inequalities alone do not establish that the bound is tight on the studied target.

## Setting and exact comparison

Let there be (m) samples (x_i\in[-1,1]), (W) retained centers forming a finite subset of a lattice (c_0+h\mathbb Z), and

\[
J_\gamma=m^{-1/2}[\mathbf1,\Phi_\gamma],\qquad
(\Phi_\gamma)_{ij}=\tanh(\gamma(x_i-c_j)),\qquad
K_\gamma=J_\gamma J_\gamma^\top.
\]

Write (Q=I-\mathbf1\mathbf1^\top/m). For a real vector (q\perp\mathbf1), define

\[
F_q(\omega)=\sum_iq_i e^{-i\omega x_i},\qquad
M_\gamma(\omega)=\frac{z}{\sinh z},\qquad
z=\frac{\pi|\omega|}{2\gamma}.
\]

The whole-center-line matrix (A_\gamma), acting on \(\mathbf1^\perp\), satisfies

\[
q^\top A_\gamma q
=\frac1{2\pi hm}\int_{\mathbb R}
\frac4{\omega^2}M_\gamma(\omega)^2|F_q(\omega)|^2\,d\omega.
\]

Define (A_\infty) by replacing (M_\gamma^2) with one. Define the gamma-independent matrix (B_\Omega) by keeping only \(|\omega|<\Omega\) in that same integral. Both matrices are positive semidefinite. The zero-sum condition makes the integrals finite at zero. Because (M_\gamma) decreases with frequency magnitude, putting (s_\gamma^2=M_\gamma(\Omega)^2) gives

\[
\boxed{
A_\gamma\preceq B_\Omega+s_\gamma^2(A_\infty-B_\Omega)
=(1-s_\gamma^2)B_\Omega+s_\gamma^2A_\infty.
}
\]

The first term retains all low-frequency leakage. The second has the explicit tanh suppression factor, which behaves as

\[
s_\gamma^2\sim (\pi\Omega/\gamma)^2e^{-\pi\Omega/\gamma}
\quad\text{when }\Omega/\gamma\to\infty.
\]

There is no Fourier-eigenvector assumption here. (B_\Omega) and (A_\infty) need not commute.

For constructing these matrices, with (r_{ij}=x_i-x_j),

\[
A_\infty=-\frac2{hm}Q[|r_{ij}|]Q,
\]

\[
B_\Omega=\frac4{\pi hm}Q
\left[\frac{1-\cos(\Omega r_{ij})}{\Omega}
-r_{ij}\operatorname{Si}(\Omega r_{ij})\right]Q.
\]

The expression in brackets is zero at (r_{ij}=0); subtraction of a constant kernel is harmless on the zero-mean subspace.

## Finite centers: an upper bound without a halo approximation

For (q\perp\mathbf1), let

\[
g_q(c)=\sum_iq_i\tanh(\gamma(x_i-c)).
\]

Adding omitted lattice columns adds nonnegative squares to the quadratic form. Thus the finite-halo kernel is bounded above by the infinite-lattice kernel, with no approximation of the finite halo:

\[
q^\top K_\gamma q\le\frac1m\sum_{j\in\mathbb Z}g_q(c_0+jh)^2.
\]

The infinite lattice can be compared with the center integral by analytic trapezoidal quadrature. For any (0<v<\pi/2), define

\[
T_v=\max(1,\tan v),\qquad
d_v=\begin{cases}1,&0<v\le\pi/4,\\
\sin(2v),&\pi/4<v<\pi/2.
\end{cases}
\]

For a unit vector (q\perp\mathbf1), the following allowance is valid:

\[
\boxed{
\epsilon_{\rm lat}(\gamma,h;v)=
\frac4h\frac{T_v^2+1/(\gamma d_v^2)}
{\exp(2\pi v/(\gamma h))-1},\qquad
q^\top K_\gamma q\le q^\top A_\gamma q+\epsilon_{\rm lat}.
}
\]

To verify the allowance, shift the analytic function (g_q(c)^2) to the lines \(\operatorname{Im}c=\pm v/\gamma\). On the real part interval \([-1,1]\), its absolute value is at most (mT_v^2). Outside that interval, subtract the common limiting sign and use

\[
|g_q(c\pm iv/\gamma)|^2
\le4m d_v^{-2}e^{-4\gamma(|c|-1)}.
\]

Therefore each shifted line has absolute integral at most
\(N=2m[T_v^2+1/(\gamma d_v^2)]\).
Contour shifting bounds the Fourier transform of (g_q^2) at frequency \(\omega\) by (Ne^{-v|\omega|/\gamma}). Summing the nonzero Poisson aliases bounds the trapezoidal error by \(2N/[e^{2\pi v/(\gamma h)}-1]\); division by (hm) gives the displayed allowance. The lattice offset changes only phase factors.

The simple choice (v=\pi/4) gives

\[
\epsilon_{\rm lat}
=\frac{4(1+1/\gamma)}{h[\exp(\pi^2/(2\gamma h))-1]}.
\]

One may minimize the explicit allowance over (v). Moving toward the pole improves the exponential factor but increases the prefactor. This bound controls an upper quadratic form; it does not claim that every tiny eigenvalue has a small relative error.

## A spectral bound computed from gamma-independent matrices

All eigenvalues below are in decreasing order on \(\mathbf1^\perp\). Weyl's inequality gives, for (j+k-1\le m-1),

\[
\lambda_{j+k-1}(A_\gamma)
\le(1-s_\gamma^2)\lambda_k(B_\Omega)
+s_\gamma^2\lambda_j(A_\infty).
\]

The same upper bound plus \(\epsilon_{\rm lat}\) holds for the corresponding eigenvalue of \(QK_\gamma Q|_{\mathbf1^\perp}\). Compression interlacing then gives

\[
\boxed{
\lambda_{j+k}(K_\gamma)
\le(1-s_\gamma^2)\lambda_k(B_\Omega)
+s_\gamma^2\lambda_j(A_\infty)+\epsilon_{\rm lat}.
}
\]

For uniform samples with spacing \(\Delta\), the second spectrum is explicit:

\[
\lambda_j(A_\infty)
=\frac{\Delta}{hm\sin^2(j\pi/(2m))},\qquad j=1,\ldots,m-1.
\]

Indeed, on the zero-mean subspace \(A_\infty=(4\Delta/(hm))L_{\rm path}^{\dagger}\), where (L_{\rm path}) is the path-graph Laplacian. Thus only (B_\Omega) needs a spectral computation, once per cutoff and independently of gamma.

For normalization, a readily computed lower bound is

\[
\lambda_1(K_\gamma)\ge
1+\sum_{j=1}^W\left(\frac1m\sum_i(\Phi_\gamma)_{ij}\right)^2
\ge1.
\]

Dividing the spectral upper bound by this lower bound controls normalized rates. A small upper bound alone does not establish that a particular target has energy in those modes, or that the modes are positive rather than null.

## A fixed target witness avoids actual kernel eigenvectors

Choose a unit (q\perp\mathbf1) independently of gamma, with positive correlation

\[
d=\frac{q^\top y}{\|y\|}>\varepsilon.
\]

For example, project the target onto a subspace of small eigenvalues of (B_\Omega), then normalize. This involves only the gamma-independent cutoff matrix. Compute

\[
b=q^\top B_\Omega q,\qquad a=q^\top A_\infty q,
\qquad 0\le b\le a.
\]

The finite-model directional curvature obeys

\[
\mu_\gamma=\|J_\gamma^\top q\|^2
\le b+s_\gamma^2(a-b)+\epsilon_{\rm lat}=:U_\gamma(q).
\]

Run GD from zero readout with \(\eta\lambda_1(K_\gamma)=1/2\). Reaching relative residual at most \(\varepsilon\) requires

\[
\boxed{
n\ge\frac{3\lambda_1(K_\gamma)(d-\varepsilon)^2}{\mu_\gamma}
\ge\frac{3\lambda_{\rm top,lower}(d-\varepsilon)^2}{U_\gamma(q)}.
}
\]

For completeness, exact quadratic descent gives
\(\sum_{t<n}\|J_\gamma^\top r_t\|^2\le2\|y\|^2/(3\eta)\).
Cauchy–Schwarz along the coefficient path then gives
\(\|w_n\|^2\le n\|y\|^2/(3\lambda_1)\).
Successful prediction implies
\((d-\varepsilon)\|y\|\le q^\top J_\gamma w_n\le\sqrt{\mu_\gamma}\|w_n\|\), proving the stated constant.

This is a sufficient, explicitly gamma-dependent obstruction for the specified target. It does not assume target alignment with an actual kernel eigenvector. To attribute the obstruction to optimization access rather than capacity, separately exhibit a readout achieving the desired tolerance, or otherwise establish that the relevant target component is representable. If the target cannot reach that tolerance, the necessary-time inequality remains true but supplies no such separation by itself.

The unresolved practical issue is whether a useful witness has both (d>\varepsilon) and sufficiently small (b). Finite-window leakage can dominate the exponentially attenuated term and make a mathematically correct bound too loose. This must be checked numerically on the studied geometry; it cannot be inferred from nominal Fourier frequency. Likewise, ordinary floating-point evaluations of these exact inequalities are not interval certificates.
