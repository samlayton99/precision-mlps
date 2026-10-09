# Angular quadrature certificates and allocation of directions and centers

This note proves the angular part of the known-function QI–Radon construction. It does not assume that a scalar QI construction has an exponential convergence theorem: the allocation results explicitly state the scalar error certificate they require. All estimates below concern exact arithmetic until a separately bounded coefficient-evaluation error is added. No Lean formalization has been started.

The main limitation is substantive: analyticity of the target in its spatial coordinates does **not** imply analyticity of its canonical Fourier–Radon profiles in direction. Section 5 supplies an explicit counterexample. An exponential angular theorem needs a quantitative angular regularity assumption; it cannot be inferred from the word “analytic” applied only to the target.

## 1. Objects and the reconstruction interface

Let \(d\ge2\), let \(D\subseteq B_R^d=\{x\in\mathbb R^d:|x|\le R\}\), and let \(\mu\) be normalized surface measure on \(S^{d-1}\). Thus \(\mu(S^{d-1})=1\). The exact ridge profiles are scalar functions \(g_\theta:\mathbb R\to\mathbb R\), indexed by \(\theta\in S^{d-1}\), satisfying

\[
f(x)=\int_{S^{d-1}}g_\theta(\theta\cdot x)\,d\mu(\theta),\qquad x\in D.
\]

For the canonical Fourier–Radon representation, with Fourier convention \(\widehat f(\xi)=\int f(x)e^{-i\xi\cdot x}\,dx\),

\[
g_\theta(s)=\frac{C_d}{2\pi}\int_{\mathbb R}|\kappa|^{d-1}\widehat f(\kappa\theta)e^{i\kappa s}\,d\kappa,
\qquad C_d=\frac{|S^{d-1}|}{2(2\pi)^{d-1}}.
\]

Assume enough integrability for this formula and Fourier inversion whenever this particular representation is used. The angular theorem itself only needs the first reconstruction identity and the stated regularity of its integrand.

For a fixed input \(x\), define the function of direction

\[
A_x(\theta)=g_\theta(\theta\cdot x).
\]

If we add a constant so that the final network matches \(f(0)\), it is more economical to analyze

\[
D_x(\theta)=g_\theta(\theta\cdot x)-g_\theta(0),
\qquad f(x)=f(0)+\int D_x(\theta)\,d\mu(\theta).
\]

Here \(0\in D\) is assumed only for this choice of anchor; any fixed reference point can replace it. Applying the angular theorem directly to \(D_x\) avoids bounding two separate quadrature errors for \(A_x\) and \(A_0\). Every theorem below applies to either family; write \(H_x\) for whichever one is used consistently.

The canonical profiles satisfy

\[
g_{-\theta}(-s)=g_\theta(s).
\]

Indeed, substitute \(u=-\kappa\) into the defining Fourier integral. Consequently both \(A_x\) and \(D_x\) are even functions of direction: \(H_x(-\theta)=H_x(\theta)\).

## 2. The actual positive sphere rule and its exactness

Fix an integer \(q\ge1\). In dimension two, use \(2q\) equally spaced circle points, all with weight \(1/(2q)\). In dimension \(d>2\), use

\[
\theta=(t,\sqrt{1-t^2}\,\nu),\qquad \nu\in S^{d-2},\ -1\le t\le1.
\]

Normalized sphere integration factors as

\[
\int_{S^{d-1}}H\,d\mu
=\frac1{Z_d}\int_{-1}^1(1-t^2)^{(d-3)/2}
\int_{S^{d-2}}H(t,\sqrt{1-t^2}\nu)\,d\mu_{d-2}(\nu)\,dt,
\]

where \(Z_d=\int_{-1}^1(1-t^2)^{(d-3)/2}\,dt\).

Use the \(q\)-node Gauss–Jacobi rule with both Jacobi parameters \((d-3)/2\), normalize its weights by dividing by \(Z_d\), and take products with the lower-dimensional rule. The one-dimensional Gaussian weights are positive and integrate constants exactly. Induction therefore shows that all sphere weights \(\alpha_m\) are positive and sum to one.

**Exactness claim.** The resulting full-sphere rule integrates every polynomial in \(\theta\) of total degree at most \(2q-1\) exactly.

Proof. In two dimensions, a polynomial of degree at most \(2q-1\), restricted to \((\cos\varphi,\sin\varphi)\), is a trigonometric polynomial with frequencies of magnitude at most \(2q-1\). The average of \(e^{ik\varphi}\) on \(2q\) equally spaced points is zero for \(0<|k|<2q\), just as its integral is zero. This proves the base case.

For the induction step, it suffices to integrate a monomial

\[
\theta_1^a\prod_{j=2}^d\theta_j^{\beta_j}
=t^a(1-t^2)^{|\beta|/2}\nu^\beta,
\qquad a+|\beta|\le2q-1.
\]

If \(|\beta|\) is odd, the lower-dimensional integral of \(\nu^\beta\) is zero by antipodal symmetry, and its quadrature sum is also zero by the induction hypothesis. If \(|\beta|\) is even, the remaining factor \(t^a(1-t^2)^{|\beta|/2}\) is a polynomial of degree \(a+|\beta|\le2q-1\); the Gaussian rule integrates it exactly. These cases cover every monomial. Linearity completes the proof.

The full rule has \(2q^{d-1}\) nodes and is antipodally symmetric. Pair each node with its negative and add their equal weights. For even integrands this leaves

\[
\boxed{M=q^{d-1}}
\]

stored directions, with positive weights summing to one. Folding need not integrate arbitrary odd functions correctly, but the unfolded rule integrates their odd part to zero. All relevant Radon integrands are even, so folding is exact for the application.

## 3. Why polynomial approximation controls quadrature error

Let \(Q_qH=\sum_m\alpha_m H(\theta_m)\), using either the full rule or its folded version for even functions. Suppose \(p\) is a spherical polynomial of degree at most \(L=2q-1\). Exactness gives \(Q_qp=\int p\,d\mu\). Therefore

\[
\int H\,d\mu-Q_qH
=\int(H-p)\,d\mu-Q_q(H-p).
\]

Taking absolute values and using positivity,

\[
\left|\int H\,d\mu-Q_qH\right|
\le \|H-p\|_\infty+\sum_m\alpha_m\|H-p\|_\infty
=2\|H-p\|_\infty.
\]

For an even \(H\), replace \(p\) by \([p(\theta)+p(-\theta)]/2\) if needed; this does not increase its error or degree. Hence the folded rule has the same estimate.

Uniformly over \(x\in D\),

\[
\boxed{
\sup_{x\in D}\left|\int H_x\,d\mu-Q_qH_x\right|
\le2\sup_{x\in D}\inf_{p\in\Pi_{2q-1}(S^{d-1})}\|H_x-p\|_\infty.
}
\]

This inequality is an immediate theorem, but it is not yet a usable rate assumption. The next section provides quantitative regularity conditions and explicit constants for its right-hand side.

## 4. Explicit analytic and algebraic angular certificates

Let \(\mathcal H_\ell\) be the space of spherical harmonics of degree \(\ell\). Let \(H_{x,\ell}\) be the orthogonal projection of \(H_x\) onto this space in \(L^2(S^{d-1},\mu)\). Define its dimension as \(D_\ell\), with \(D_0=1\). For \(\ell\ge1\),

\[
D_\ell=\binom{\ell+d-1}{d-1}-\binom{\ell+d-3}{d-1},
\]

where a binomial coefficient with upper index smaller than its nonnegative lower index is zero. In dimension two this gives \(D_\ell=2\) for \(\ell\ge1\).

Choose an orthonormal basis \(Y_{\ell,j}\) in normalized sphere measure. Rotation invariance makes \(\sum_j|Y_{\ell,j}(\theta)|^2\) constant in \(\theta\); integrating shows that this constant is \(D_\ell\). Cauchy–Schwarz therefore yields

\[
\boxed{\|H_{x,\ell}\|_\infty\le\sqrt{D_\ell}\,\|H_{x,\ell}\|_2.}
\]

The dimension generating function is

\[
\sum_{\ell=0}^\infty D_\ell z^\ell
=\frac{1-z^2}{(1-z)^d}
=\frac{1+z}{(1-z)^{d-1}},\qquad 0\le z<1.
\]

The first equality follows by summing the two shifted binomial series in the dimension formula.

### 4.1 Angular analytic norm

Assume there are specified constants \(a>0\) and \(B_a<\infty\) such that

\[
\boxed{
\sup_{x\in D}\sum_{\ell=0}^\infty e^{2a\ell}\|H_{x,\ell}\|_2^2\le B_a^2.
}
\]

This is a quantitative analytic regularity assumption in the angular variable. It defines the regularity being used rather than hiding it inside an unspecified “analytic” constant. Establishing it for a target or target class is a separate obligation; spatial analyticity alone is insufficient.

Take the explicit polynomial \(p_L=\sum_{\ell=0}^L H_{x,\ell}\). Using the addition bound and then Cauchy–Schwarz,

\[
\begin{aligned}
\|H_x-p_L\|_\infty
&\le\sum_{\ell>L}\sqrt{D_\ell}\,\|H_{x,\ell}\|_2\\
&\le B_a\left(\sum_{\ell>L}D_\ell e^{-2a\ell}\right)^{1/2}.
\end{aligned}
\]

This is already an explicit tail certificate. It can be evaluated numerically from the known dimensions. For a simpler closed bound, use \(\ell\ge L+1\) to write

\[
e^{-2a\ell}\le e^{-a(L+1)}e^{-a\ell}.
\]

Sum the remaining series using its generating function:

\[
\|H_x-p_L\|_\infty
\le B_a e^{-a(L+1)/2}
\left[\frac{1+e^{-a}}{(1-e^{-a})^{d-1}}\right]^{1/2}.
\]

Putting \(L=2q-1\) gives

\[
\boxed{
E_{\rm ang}(q)
\le2B_a C_d^{\rm ang}(a)e^{-aq},\qquad
C_d^{\rm ang}(a)=\left[\frac{1+e^{-a}}{(1-e^{-a})^{d-1}}\right]^{1/2}.
}
\]

There is a free constant improvement: retain the exact harmonic tail above. More generally, for any \(0<\eta<1\),

\[
E_{\rm ang}(q)
\le2B_a e^{-2(1-\eta)aq}
\left[\frac{1+e^{-2\eta a}}{(1-e^{-2\eta a})^{d-1}}\right]^{1/2}.
\]

This follows by splitting \(e^{-2a\ell}=e^{-2(1-\eta)a\ell}e^{-2\eta a\ell}\). Optimizing \(\eta\) is a one-variable bound calculation; it changes neither the quadrature nor the network.

The dimension dependence is visible: for the simpler certificate, a sufficient order is

\[
q\ge\frac1a\left[
\log\frac{2B_a}{\varepsilon_{\rm ang}}
+\frac12\log(1+e^{-a})
-\frac{d-1}{2}\log(1-e^{-a})
\right].
\]

Thus \(M=O(\log^{d-1}(1/\varepsilon))\) is a fixed-dimension asymptotic. The constants are not uniform in dimension, nor can the target's angular norm be silently assumed independent of dimension.

### 4.2 Finite angular smoothness

Instead assume a specified \(s>(d-1)/2\) and \(B_s\) satisfy

\[
\sup_{x\in D}\sum_{\ell=0}^\infty(1+\ell)^{2s}\|H_{x,\ell}\|_2^2\le B_s^2.
\]

This is a spherical Sobolev norm, using \(1+\ell\) in place of the equivalent Laplace–Beltrami spectral weight. The same argument gives

\[
\|H_x-p_L\|_\infty
\le B_s\left[\sum_{\ell>L}D_\ell(1+\ell)^{-2s}\right]^{1/2}.
\]

For \(\ell\ge1\) and \(d\ge2\),

\[
D_\ell\le\frac{2(\ell+d-2)^{d-2}}{(d-2)!}
\le\frac{2(d-1)^{d-2}}{(d-2)!}(1+\ell)^{d-2}.
\]

For \(d=2\), the first bound simply reads \(D_\ell\le2\). Since \(d-2-2s<-1\), the decreasing-series integral bound gives

\[
\sum_{\ell>L}(1+\ell)^{d-2-2s}
\le\int_{L+1}^\infty t^{d-2-2s}\,dt
=\frac{(L+1)^{d-1-2s}}{2s-d+1}.
\]

Consequently,

\[
\boxed{
E_{\rm ang}(q)
\le2B_s
\left[\frac{2(d-1)^{d-2}}{(d-2)!(2s-d+1)}\right]^{1/2}
(2q)^{(d-1)/2-s}.
}
\]

The loss of \((d-1)/2\) derivatives is the cost of this elementary \(L^2\)-to-uniform estimate. It should not be claimed to be the sharp Jackson rate for a \(C^s\) class. A sharper uniform approximation theorem can improve it, but would be a distinct imported theorem with its own norm and constants.

## 5. An analytic target whose canonical angular profiles have cusps

Consider

\[
f(x)=\prod_{j=1}^d\frac1{1+x_j^2}.
\]

This function is integrable and holomorphic in the Cartesian strip \(|\operatorname{Im}x_j|<1\) for every coordinate. Its Fourier transform is

\[
\widehat f(\xi)=\pi^d\exp\left(-\sum_{j=1}^d|\xi_j|\right).
\]

Set \(u(\theta)=\sum_j|\theta_j|\). Along a frequency spoke,

\[
\widehat f(\kappa\theta)=\pi^d e^{-|\kappa|u(\theta)}.
\]

Substitute into the profile integral and combine its positive and negative halves:

\[
\begin{aligned}
g_\theta(s)
&=C_d\pi^{d-1}\int_0^\infty
\kappa^{d-1}e^{-u(\theta)\kappa}\cos(\kappa s)\,d\kappa\\
&=C_d\pi^{d-1}\Gamma(d)\operatorname{Re}
\bigl(u(\theta)-is\bigr)^{-d}.
\end{aligned}
\]

The last step uses \(\int_0^\infty t^{d-1}e^{-zt}\,dt=\Gamma(d)z^{-d}\), valid for \(\operatorname{Re}z>0\). In particular,

\[
g_\theta(0)=C_d\pi^{d-1}\Gamma(d)u(\theta)^{-d}.
\]

The absolute values in \(u\) create cusps on coordinate hyperplanes. Therefore the canonical angular integrand need not even be \(C^1\), despite the target having a uniform complex spatial neighborhood.

Anchoring does not generally remove the problem. In dimension two, take \(x=(v,0)\) with \(0<v<1\), and approach \(\theta=(1,0)\) through \(\theta(t)=(\sqrt{1-t^2},t)\). Then

\[
u(\theta(t))=1+|t|+O(t^2),\qquad \theta(t)\cdot x=v+O(t^2).
\]

The anchored angular function is a constant multiple of

\[
\operatorname{Re}(u-is)^{-2}-u^{-2}.
\]

Its derivative with respect to \(u\) at \((u,s)=(1,v)\) is

\[
2\left[1-\operatorname{Re}(1-iv)^{-3}\right]
=2\left[1-\frac{1-3v^2}{(1+v^2)^3}\right]>0.
\]

Thus a nonzero \(|t|\) term remains. This example disproves any implication from spatial analyticity alone to the angular analytic certificate used above. It does not claim that an entirely different ridge decomposition could not represent the same target efficiently.

## 6. A sufficient certificate stated directly on the known Fourier data

This optional condition gives one route from \(\widehat f\) to an exponential angular estimate. It is stronger than necessary and is not asserted for every analytic target.

For each real \(\kappa\), expand the function \(\theta\mapsto\widehat f(\kappa\theta)\) into spherical harmonics:

\[
\widehat f(\kappa\theta)=\sum_{\ell=0}^\infty F_{\kappa,\ell}(\theta).
\]

Choose \(\zeta>1\), and assume the explicit weighted integral

\[
J_{\zeta,R}
=\frac{C_d}{2\pi}\int_{\mathbb R}|\kappa|^{d-1}e^{\zeta R|\kappa|}
\sum_{\ell=0}^\infty\zeta^\ell\sqrt{D_\ell}\,
\|F_{\kappa,\ell}\|_2\,d\kappa
\]

is finite. The quantities here are derived from the known target; there is no target-fitting linear system in this assumption. Proving or certifying the integral is still part of checking the target class.

Expand the plane wave as

\[
e^{i\kappa\theta\cdot x}
=\sum_{k=0}^\infty\frac{(i\kappa\theta\cdot x)^k}{k!}.
\]

The product of its degree-\(k\) term with \(F_{\kappa,\ell}\) is a spherical polynomial of degree at most \(k+\ell\). Keep only terms with \(k+\ell\le L\). For an omitted term, \(\zeta^{k+\ell}\ge\zeta^{L+1}\). Since \(|\theta\cdot x|\le R\), the uniform omitted tail is bounded by

\[
\begin{aligned}
&\zeta^{-(L+1)}\frac{C_d}{2\pi}
\int|\kappa|^{d-1}
\sum_{\ell,k}\zeta^{\ell+k}\sqrt{D_\ell}\|F_{\kappa,\ell}\|_2
\frac{(|\kappa|R)^k}{k!}\,d\kappa\\
&\hspace{2cm}=\zeta^{-(L+1)}J_{\zeta,R}.
\end{aligned}
\]

Absolute convergence is supplied by the assumed finite \(J_{\zeta,R}\), so the exchanges of sums and integrals are justified. The angular quadrature therefore satisfies

\[
E_{\rm ang}(q)\le2J_{\zeta,R}\zeta^{-2q}.
\]

For anchored profiles \(D_x\), expand \(e^{i\kappa\theta\cdot x}-1\), which removes the \(k=0\) terms. The same bound holds, and replacing \(e^{\zeta R|\kappa|}\) by \(e^{\zeta R|\kappa|}-1\) improves this particular certificate.

For a centered isotropic Gaussian target, the directional Fourier function is constant in \(\theta\), only \(\ell=0\) is present, and the radial Gaussian decay makes this condition finite for every fixed \(\zeta,R\). This is one direct check that the sufficient class is nonempty and practically evaluable. The preceding rational counterexample does not satisfy the exponential angular condition.

## 7. Combining this with a scalar certificate

Let the scalar construction on direction \(m\) have an anchored uniform error at most \(\delta_m\) on the projected input interval. Assume its neurons, correction coefficients, and constant are all actually computed by the stated finite algorithm. Positivity gives

\[
\left|\sum_m\alpha_m\bigl(q_m(\theta_m\cdot x)-g_m(\theta_m\cdot x)\bigr)\right|
\le\sum_m\alpha_m\delta_m.
\]

Thus the whole-network certificate is

\[
\boxed{
\|q-f\|_{L^\infty(D)}
\le E_{\rm ang}(q)+\sum_m\alpha_m\delta_m+E_{\rm numerical},
}
\]

when the angular certificate is applied to the same anchored representation. Extension or target-preprocessing errors must also be added if the reconstructed function is a surrogate for the original target.

No factor \(M\) appears for a uniform scalar tolerance: if every \(\delta_m\le\delta\), then \(\sum\alpha_m\delta_m\le\delta\). If signed angular weights are used instead, replace this by \(\delta\sum_m|\alpha_m|\); positivity is a real stability advantage.

Let \(n\) be the number of interior intervals, so there are \(n+1\) interior centers. With \(R_h(n)\) halo centers per side, the per-spoke width is

\[
w(n)=n+1+2R_h(n).
\]

The total neuron count is \(W=\sum_m w(n_m)\); in the uniform case it is \(q^{d-1}w(n)\). A conventional fully expanded scalar-output, one-hidden-layer MLP has \((d+2)W+1\) scalar parameters. Structured storage can share directions and bandwidths, but must not be confused with this nominal parameter count.

## 8. Allocation when both certified errors are exponential

Suppose, after all halo and scalar saturation terms have been dealt with, we have the actual certificates

\[
E_{\rm ang}\le B e^{-bq},\qquad E_{\rm scalar}\le A e^{-an},
\]

with fixed positive \(A,B,a,b\). These constants include the target and dimension dependence. Suppose also that the actual per-spoke width is bounded by \(cn+c_0\), where \(c,c_0\) are fixed and already account for the halo choice. The clean asymptotic allocation below optimizes the leading width \(c q^{d-1}n\); bounded offsets and integer roundings are handled afterwards.

Assign error budgets \(\varepsilon_a+\varepsilon_s=\varepsilon\). The continuous minimal resolutions for these two bounds are

\[
q=\frac1b\log\frac B{\varepsilon_a},\qquad
n=\frac1a\log\frac A{\varepsilon_s}.
\]

Write \(p=d-1\), \(L_a=\log(B/\varepsilon_a)\), and \(L_s=\log(A/\varepsilon_s)\). Apart from fixed factors, the leading width is \(L_a^pL_s\). Set \(\varepsilon_a=\varepsilon-\varepsilon_s\). Differentiating its logarithm gives

\[
\frac{d}{d\varepsilon_s}\log(L_a^pL_s)
=\frac p{\varepsilon_aL_a}-\frac1{\varepsilon_sL_s}.
\]

Therefore an interior optimum satisfies

\[
\boxed{\varepsilon_aL_a=(d-1)\varepsilon_sL_s.}
\]

If \(\varepsilon<\min(A/e,B/e)\), this stationary point is unique: the function \(\varepsilon_aL_a-p\varepsilon_sL_s\) decreases strictly from positive to negative as \(\varepsilon_s\) increases from zero to \(\varepsilon\), because its derivative is \(-(L_a-1)-p(L_s-1)<0\).

As \(\varepsilon\to0\), the logarithms have the same leading term, hence

\[
\varepsilon_s\sim\frac\varepsilon d,
\qquad \varepsilon_a\sim\frac{d-1}{d}\varepsilon.
\]

The resulting scales are

\[
n\sim\frac1a\log\frac1\varepsilon,
\qquad q\sim\frac1b\log\frac1\varepsilon,
\qquad \frac qn\sim\frac ab,
\]

and

\[
\boxed{W\sim\frac{c}{a b^{d-1}}\log^d(1/\varepsilon)}
\]

for this leading certified optimization, at fixed dimension and fixed class constants. Round \(q,n\) upward and check the complete scalar certificate rather than relying on the asymptotic formula.

This is not a theorem that every activation or the current fixed-bandwidth scalar scheme has this rate. If the scalar certificate has an \(n^{-1}\) saturation term, or if suppressing it changes \(a\), \(c\), or the halo budget with \(\varepsilon\), those dependences must be inserted before optimizing. One cannot remove them by calling their constants small.

The halo can also be retained in the allocation calculation instead of approximating its cost. For a differentiable actual-width relaxation \(w(n)>0\), minimize \(q^{d-1}w(n)\) with the same error budgets. Its logarithmic derivative is

\[
\frac{d-1}{\varepsilon_aL_a}
-\frac{w'(n)}{a\varepsilon_s w(n)}.
\]

Thus the exact relaxed balance is

\[
\boxed{\varepsilon_aL_a=(d-1)a\varepsilon_s\frac{w(n)}{w'(n)}.}
\]

For \(w(n)=cn+c_0\), this changes the previous right-hand side to \((d-1)\varepsilon_s[L_s+ac_0/c]\). For the square-root-halo relaxation \(w(n)=n+1+2\sqrt n\), it becomes \((d-1)\varepsilon_s[L_s+\sqrt{aL_s}]\). Integer ceilings and minimum halo counts are handled by evaluating the actual width and full error certificate at nearby integers; the relaxed stationarity equation is not claimed to optimize a discontinuous ceiling function exactly.

More generally, if the actual scalar rate is \(A e^{-a n^\beta}\), its leading width contribution is \(a^{-1/\beta}L_s^{1/\beta}\). The stationary condition becomes

\[
\varepsilon_aL_a=\beta(d-1)\varepsilon_sL_s,
\]

and the logarithmic power is \((d-1)+1/\beta\), assuming width remains linear in \(n\).

## 9. More centers on more difficult spokes

For fixed directions and weights, suppose the scalar certificates are

\[
\delta_m\le A_m e^{-a_mn_m},\qquad a_m>0,
\]

and the counted variable width is \(\sum_m c_m n_m\). Here \(c_m>0\) converts one unit of the resolution variable into its actual leading neuron cost, including any linear halo overhead. Let \(n_m\ge n_{m,0}\) be required baseline resolutions.

Put \(D_m=\alpha_mA_m\). The continuous problem is

\[
\min\sum_m c_m n_m
\quad\text{subject to}\quad
\sum_mD_m e^{-a_mn_m}\le\varepsilon_s,
\qquad n_m\ge n_{m,0}.
\]

The cost is linear and the feasible set is convex. When the baseline does not already meet the error budget, the error constraint is active at the optimum. A Lagrange multiplier \(\mu>0\) gives the active-coordinate condition

\[
c_m-\mu a_mD_m e^{-a_mn_m}=0.
\]

Hence

\[
\boxed{
n_m=\max\left\{n_{m,0},\ \frac1{a_m}\log\frac{\mu a_mD_m}{c_m}\right\}.
}
\]

Determine \(\mu\) from the monotone scalar equation

\[
\sum_m\min\left\{D_m e^{-a_mn_{m,0}},\ \frac{c_m}{\mu a_m}\right\}=\varepsilon_s.
\]

This is a small allocation calculation, not a solve for neuron coefficients. If every coordinate is active, define \(S=\sum_m c_m/a_m\); then \(\mu=S/\varepsilon_s\), giving

\[
\boxed{
n_m=\frac1{a_m}\log\left(\frac{\alpha_m A_m a_mS}{c_m\varepsilon_s}\right),
\qquad
\alpha_m\delta_m^{\rm bound}=\frac{\varepsilon_s c_m}{a_m S}.
}
\]

Smaller \(a_m\), corresponding to more difficult scalar approximation, increases the required resolution. Larger angular weight or amplitude also increases it. The error allowed on an individual low-weight spoke need not equal the error allowed on a high-weight spoke.

Round each \(n_m\) upward. This only decreases the certified error and adds less than \(\sum_m c_m\) to the linear cost. Add the actual \(+1\) interior-center offsets and integer halo counts afterward.

If the exact width is nonlinear, for example \(n+1+2\lceil\sqrt n\rceil\), use that width in the discrete optimization. For a differentiable relaxation \(w_m(n)\), the stationarity equation is instead \(w_m'(n_m)=\mu a_mD_m e^{-a_mn_m}\). The preceding closed formula is exact for linear costs and only a leading approximation for nonlinear halo costs.

## 10. Allocation for algebraic certificates

Suppose

\[
E_{\rm ang}\le Bq^{-t},\qquad E_{\rm scalar}\le An^{-r},\qquad r,t>0.
\]

The minimal continuous resolutions for budgets \(\varepsilon_a,\varepsilon_s\) are \(q=(B/\varepsilon_a)^{1/t}\), \(n=(A/\varepsilon_s)^{1/r}\). Thus leading width is proportional to

\[
\varepsilon_a^{-(d-1)/t}\varepsilon_s^{-1/r}.
\]

Differentiating its logarithm under \(\varepsilon_a+\varepsilon_s=\varepsilon\) yields

\[
\frac{d-1}{t\varepsilon_a}-\frac1{r\varepsilon_s}=0.
\]

Therefore

\[
\varepsilon_s=\frac{t}{t+r(d-1)}\varepsilon,
\qquad
\varepsilon_a=\frac{r(d-1)}{t+r(d-1)}\varepsilon,
\]

and

\[
\boxed{W=O\left(\varepsilon^{-[1/r+(d-1)/t]}\right).}
\]

If the two rates agree, \(r=t\), this becomes \(W=O(\varepsilon^{-d/r})\). This allocation theorem uses actual error exponents, not the number of derivatives before converting an angular Sobolev norm to a uniform rate.

## 11. What this certificate does and does not establish

Proved here: positivity and polynomial exactness of the recursive Gaussian rule; its folded direction count; explicit analytic and Sobolev angular error bounds; the analytic-target counterexample; a stronger sufficient Fourier-side certificate; propagation of scalar errors; and conditional resource-allocation formulas.

Still required for a complete construction theorem: the scalar activation-specific finite-center and finite-halo certificate, the available numerical access to \(f\) or its Fourier/Radon profiles, an explicit extension if the target is initially only defined on a bounded domain, and a coefficient-evaluation/roundoff budget. These requirements cannot be discharged by the angular theorem.

Parameter-count bounds also do not bound the cost of obtaining the profiles. Direct high-dimensional integration may be expensive even if the final network has relatively few neurons. The theorem must state its data-access model separately.

The finite-ball Radon/orthogonal-polynomial reconstruction in Yuan Xu's work is a possible alternative for avoiding an arbitrary global extension, but it changes the profile filter. It has not been substituted into the main construction in this note.

## References and provenance

- Gaussian quadrature positivity and degree \(2q-1\) exactness: [NIST DLMF, §3.5(v)](https://dlmf.nist.gov/3.5#v). The recursive spherical exactness proof is given above.
- Spherical harmonics and their dimensions: [Dunkl and Xu, *Orthogonal Polynomials of Several Variables*, chapter on the unit sphere](https://doi.org/10.1017/CBO9781107786134.006). The particular tail estimates used here are derived explicitly above.
- Alternative finite-ball Radon reconstruction: [Yuan Xu, *Reconstruction from Radon projections and orthogonal expansion on a ball*, arXiv:0705.1984](https://arxiv.org/abs/0705.1984), Proposition 3.1 and Theorem 3.2. This reference motivates an optional alternative; it is not an assumption of the main theorem.
