# Finite QI–Radon construction: uniform accuracy, dimension, and allocation

Date: 2026-10-08. Status: mathematical derivation and independent audit; not Lean formalized. No new PDE solver or training result is claimed. This document concerns a known target and the actual one-hidden-layer network produced by an explicit construction.

## 1. What can and cannot be claimed

For a known target holomorphic and bounded on a complex neighborhood of a slightly larger real ball, a specified radial boundary extension supplies analytic scalar and angular profiles. The existing finite, corrected-halo tanh construction then has

\[
W=O_{d,\text{regularity}}\bigl(\log^d(1/\varepsilon)\bigr)
\]

hidden neurons. The statement includes the halos and requires an accuracy-dependent bandwidth/spacing schedule. It is not true for every activation: a one-layer ReLU network needs at least order \(\varepsilon^{-1/2}\) neurons to approximate \(x^2\) uniformly on an interval. Nor does input-space analyticity imply angular analyticity for an arbitrary global extension. The prescribed radial extension below supplies that implication in the target-side corollary.

The general theorem is a composition of (i) an explicit angular quadrature certificate and (ii) an activation-specific, computable scalar construction certificate. Tanh and a Gaussian-derivative family have complete scalar certificates here. A generic kernel error ledger is given separately so that other activations can be assessed without assuming a rate.

## 2. Target, domain, normalization, and access

The target domain is the closed unit ball \(\mathbb B^d\), \(d\ge2\). An affine rescaling handles another ball; enclosing a different compact domain in a ball retains a valid, potentially conservative bound. Let \(F:\mathbb R^d\to\mathbb R\) be a **specified known extension** of the desired target. Require absolute convergence of Fourier inversion and the weighted line integrals below, sufficient for the stated Fubini and polar-coordinate interchanges (for example \(F,\widehat F\in L^1\) together with the supplied uniform line-moment bound). Extension error, if nonzero, is a separate tolerance expenditure; it is not implicitly zero. One may not change the extension with tolerance and then silently treat its regularity constants as fixed.

Use \(\widehat F(\xi)=\int F(x)e^{-i\xi\cdot x}\,dx\) and normalized surface measure \(\mu\) on \(\mathbb S^{d-1}\). Define

\[
C_d=\frac{|\mathbb S^{d-1}|}{2(2\pi)^{d-1}},\qquad
g_\theta(s)=\frac{C_d}{2\pi}\int_{\mathbb R}|\omega|^{d-1}\widehat F(\omega\theta)e^{i\omega s}\,d\omega.
\]

Polar Fourier inversion gives \(F(x)=\int g_\theta(\theta\cdot x)\,d\mu(\theta)\). This fixes all normalization factors. Define the anchored angular integrand

\[
H_x(\theta)=g_\theta(\theta\cdot x)-g_\theta(0),\qquad
F(x)=F(0)+\int H_x\,d\mu.
\]

It is even in \(\theta\), since \(g_{-\theta}(s)=g_\theta(-s)\).

**Access is part of the input specification.** One must evaluate these profiles with controlled error, either from explicit formulas or from the displayed one-dimensional Fourier integrals with certified tails/quadrature. A sufficient scalar-strip bound is

\[
\frac{C_d}{2\pi}\sup_\theta\int |\omega|^{d-1}e^{\delta_0|\omega|}|\widehat F(\omega\theta)|\,d\omega\le B,
\]

which bounds profiles throughout \(|\operatorname{Im}s|\le\delta_0\); use a strictly smaller strip if necessary for an open analytic neighborhood. When only real-space evaluations of \(F\) are supplied, acquiring \(\widehat F\) or Radon profiles can itself require high-dimensional integration. The neuron-count theorem does **not** bound that cost by \(O(W)\), and finite noisy samples do not provide analytic continuation for free.

### A prescribed boundary setup from a standard target-side assumption

There is a direct sufficient condition that does not assume good profiles in advance. Suppose the original target \(f\) is holomorphic and bounded by \(B_f\) on the complex tube of radius \(\sigma>0\) around a real ball \(B_R\), with \(R>1\). Choose fixed radii \(1<T<S<R_0<R\), and a nonnegative normalized smooth bump \(w(a)\) on \([R_0,R]\). Define
\[
\chi(r)=\int_{R_0}^R w(a)\mathbf1_{r\le a}\,da,\qquad F(x)=f(x)\chi(|x|).
\]
Outside \(B_R\) set \(F=0\). This is a specified smooth compact extension, agreeing exactly with \(f\) on the unit ball. Its scalar Radon projection, for \(|s|<R_0\), has the finite representation
\[
p_\theta(s)=\int_{R_0}^R w(a)(a^2-s^2)^{(d-1)/2}
\int_{B^{d-1}} f\bigl(s\theta+\sqrt{a^2-s^2}V_\theta u\bigr)\,du\,da,
\]
where \(V_\theta\) is an orthonormal frame of \(\theta^\perp\). The cutoff is evaluated only at real radii; the known analytic target supplies the complex dependence.

`radial_extension_certificate.md` proves, with every margin and constant explicit, that this formula is analytic both in the interior scalar coordinate and under small complex rotations of the direction. For odd \(d\), the canonical filter is an ordinary derivative of \(p_\theta\). For even \(d\), the proof gives a finite, regularized Hilbert-transform formula and then differentiates it. Both produce a scalar tube \(\delta=\eta>0\), real margin \(\Delta=T-1\), and bound \(B_g\). Mixed rotation derivatives then give assumption (A) below with
\[
K=d(d-1)/2,\quad b=\min\{t_0,\log(1+\eta/2)\},\quad
a=\frac{b}{4e\sqrt K},\quad
A=\frac{2e^2B_g}{\sqrt{1-e^{-2a}}}.
\]
Here \(t_0,\eta,B_g\) are explicit functions of \(R,\sigma,B_f,T,S,R_0,d\) given in that proof. This establishes a target-side version of the main theorem in every dimension. Those generic constants are deliberately conservative; direct certified profile bounds can be much tighter.

The compact cutoff does **not** make the global Fourier integral converge at complex scalar arguments. For this boundary construction, evaluate complex profiles by the finite Radon/Hilbert continuation formulas just described. Substituting complex \(s\) into an unjustified full Fourier integral would invalidate the construction. The angular and scalar arguments below otherwise remain unchanged.

## 3. Angular quadrature and an explicit analytic certificate

At each polar-coordinate recursion use \(q\) Gauss–Jacobi nodes for weight \((1-t^2)^{(d-3)/2}\), and use \(2q\) equally spaced nodes on the terminal circle. Normalize all weights. The full product rule has \(2q^{d-1}\) nodes, positive weights summing to one, and integrates every spherical polynomial of degree at most \(2q-1\) exactly. For the even integrand \(H_x\), fold antipodal pairs and add their weights: the retained number of directions is

\[
M=q^{d-1}.
\]

The folded rule is asserted exact for even polynomials, not arbitrary odd ones. To prove full-rule exactness, induct on dimension and check coordinate monomials. A monomial of odd total degree in the lower-dimensional coordinates has zero lower-sphere average, exactly reproduced by the lower rule. For even degree \(2j\), the remaining factor \((1-t^2)^j\) is a polynomial, leaving a one-dimensional polynomial of degree at most \(2q-1\) for Gauss–Jacobi. The circle base case follows from exact integration of Fourier modes \(|k|<2q\).

Require \(H_x\) to be continuous in direction. This fixes its pointwise representative: an \(L^2\) statement alone cannot control arbitrarily redefined values at quadrature nodes. For a fully quantitative angular class, let \(H_{x,\ell}\) be the degree-\(\ell\) spherical-harmonic component, with \(L^2\) defined using normalized \(\mu\). Assume, with a positive upper bound \(A\),

\[
\sup_{x\in\mathbb B^d}\sum_{\ell\ge0}e^{2a\ell}\|H_{x,\ell}\|_2^2\le A^2,
\qquad a>0.
\tag{A}
\]

This is a quantitative angular analytic regularity assumption. It concerns variation of the profiles **with direction**, not only their scalar analyticity. It is not a computational instruction to fit spherical harmonics.

Write \(D_{d,\ell}\) for the dimension of degree-\(\ell\) spherical harmonics. The addition theorem and Cauchy–Schwarz imply

\[
\|H_{x,\ell}\|_\infty\le\sqrt{D_{d,\ell}}\|H_{x,\ell}\|_2,
\quad
\Big\|\sum_{\ell\ge2q}H_{x,\ell}\Big\|_\infty
\le A\Big(\sum_{\ell\ge2q}D_{d,\ell}e^{-2a\ell}\Big)^{1/2}.
\]

An exact positive quadrature has error at most twice the best uniform error of an exactly integrated polynomial: subtract that polynomial, then bound the integral and weighted sum separately. Thus the **sharper computable** angular certificate is

\[
E_{\rm ang}(q)\le2A\Big(\sum_{\ell\ge2q}D_{d,\ell}e^{-2a\ell}\Big)^{1/2}.
\tag{1}
\]

The generating function \(\sum_{\ell\ge0}D_{d,\ell}z^\ell=(1+z)/(1-z)^{d-1}\) gives the simpler certificate

\[
E_{\rm ang}(q)\le2A\,C_{\rm ang}(d,a)e^{-aq},\qquad
C_{\rm ang}(d,a)=\sqrt{\frac{1+e^{-a}}{(1-e^{-a})^{d-1}}}.
\tag{2}
\]

Indeed, for \(\ell\ge2q\), \(e^{-2a\ell}\le e^{-2aq}e^{-a\ell}\). The dimension-dependent constant is explicit and must not be hidden when dimension varies.

An optional stronger assumption can be stated directly on the known Fourier data and supplies both angular convergence and a scalar strip bound. Let \(F_{\omega,\ell}\) be the degree-\(\ell\) harmonic component of \(\theta\mapsto\widehat F(\omega\theta)\). For a fixed \(\zeta>1\), suppose

\[
J_\zeta=\frac{C_d}{2\pi}\int_{\mathbb R}|\omega|^{d-1}e^{\zeta|\omega|}
\sum_{\ell\ge0}\zeta^\ell\sqrt{D_{d,\ell}}\|F_{\omega,\ell}\|_2\,d\omega<\infty.
\tag{2a}
\]

Then \(|g_\theta(s)|\le J_\zeta\) for \(|\operatorname{Im}s|\le\zeta\), and
\(E_{\rm ang}(q)\le2J_\zeta\zeta^{-2q}\). For the latter, multiply the harmonic expansion of \(\widehat F(\omega\theta)\) by the Taylor expansion of \(e^{i\omega\theta\cdot x}-1\); terms of combined degree less than \(2q\) are integrated exactly. Each omitted term gains a factor at most \(\zeta^{-2q}\), and the remaining absolute sum is bounded by \(J_\zeta\). This gives a direct sufficient Fourier-data class for the main rate, independently of assumption (A). It is stronger than needed; centered Gaussian targets satisfy it. The theorem does not require computing a spherical-harmonic expansion as part of coefficient construction.

## 4. Complete scalar tanh certificate, including the finite halo

Assume every \(g_\theta\) is holomorphic and bounded by \(B\) on the complex tube of radius \(\delta>0\) around \([-1-\Delta,1+\Delta]\), where \(\Delta>0\) is a supplied real extension margin. Choose a fixed physical halo design length

\[
0<r\le\min\{\Delta/2,\delta/2\},\qquad
\gamma=2\pi/r,\qquad \beta=\pi r/4.
\]

For \(n\) interior intervals (\(n+1\) interior centers), set

\[
h=2/n,\quad \lambda=\gamma h=4\pi/(rn),\quad
R=\lceil rn/2\rceil,\quad c_j=-1+jh\quad(-R\le j\le n+R).
\]

Require

\[
n\ge n_0:=\left\lceil\max\{16,4\pi/r,3/(\Delta-r)\}\right\rceil.
\]

Then \((R+1/2)h\le\Delta\), \(0<\lambda\le1\), \(\gamma\delta\ge4\pi\), and \(\lambda R\ge2\log2\). These are the sufficient conditions of the existing contour/halo proof.

Compute the baseline readouts using

\[
\tau=\pi/(2\gamma),\quad
\rho(c)=\frac{g(c+i\tau)-g(c-i\tau)}{2i\tau},\quad
u_j=h\rho(c_j)/2.
\]

Apply the explicit endpoint-moment/rational correction of `tanh_certificate.md` to the outermost \(\lfloor(R+1)/2\rfloor\) halo slots on each side. This is finite quadrature and prescribed polynomial/partial-fraction algebra, with no fitted readouts or new activation. The unchanged scalar network is a sum of ordinary tanh neurons plus one bias.

The existing proof bounds its error by

\[
\frac{32B}{\delta}e^{-\pi\delta n/4}
+\frac{64\pi B}{\lambda\delta n}e^{-\pi^2/\lambda}
+\frac{32BD_\lambda}{\delta n}e^{-\lambda R^2/4},
\quad D_\lambda=\frac\pi{2\lambda}+\frac{4\log2}{\pi}.
\tag{3}
\]

Under the stated design, the three exponents are at least \(\beta n\), while their prefactors sum to at most

\[
\frac B\delta\left(32+20r+\frac{128\log2}{\pi n}\right)
\le B C_{\rm sc},\qquad C_{\rm sc}=\frac{34+20r}{\delta}.
\]

Consequently

\[
\boxed{E_{\rm sc}(n)\le B C_{\rm sc}e^{-\beta n},\qquad
w(n)=n+1+2\lceil rn/2\rceil\le(1+r)n+3.}
\tag{4}
\]

This includes every halo neuron. It changes the asymptotic schedule from fixed \(\lambda\), \(R\asymp\sqrt n\) to fixed physical width, fixed \(\gamma\), \(\lambda\asymp1/n\), and \(R\asymp n\). Leaving \(\lambda\) fixed retains an actual algebraic alias contribution; retaining a \(\sqrt n\) halo while shrinking \(\lambda\) gives only a \(\log^2(1/\varepsilon)\) scalar certificate from (3).

## 5. Finite multidimensional network and the main theorem

For each retained direction construct a scalar tanh sum \(S_m(s)=\sum_j u_{mj}\tanh(\gamma(s-c_j))\). Define ordinary MLP readouts and output bias by

\[
v_{mj}=\alpha_m u_{mj},\qquad
b=F(0)-\sum_{m,j}v_{mj}\tanh(-\gamma c_j),
\]
\[
Q(x)=b+\sum_{m,j}v_{mj}\tanh(\gamma\theta_m\cdot x-\gamma c_j).
\tag{5}
\]

There is one hidden layer and one output constant. Subtracting scalar values at zero is only part of the error proof, not an extra feature in the implementation.

If a scalar construction approximates \(g_m\) within \(E_m\), its difference between \(s\) and zero approximates the corresponding target difference within \(2E_m\). Positive normalized angular weights therefore give

\[
\boxed{\|F-Q\|_{\infty,\mathbb B^d}
\le E_{\rm ang}(q)+2\sum_m\alpha_m E_m+E_{\rm num}.}
\tag{6}
\]

Here \(E_{\rm num}\) contains certified coefficient, profile-evaluation, geometry, bias, and inference arithmetic errors. It is not silently assumed negligible. With common scalar parameters, (2), (4) imply

\[
\boxed{\|F-Q\|_\infty
\le Ue^{-aq}+Ve^{-\beta n}+E_{\rm num},
\quad U=2AC_{\rm ang}(d,a),\quad V=2BC_{\rm sc}.}
\tag{7}
\]

Reserve \(\varepsilon/3\) for each term. A completely explicit choice is

\[
q=\max\left\{1,\left\lceil a^{-1}\log(3U/\varepsilon)\right\rceil\right\},
\quad
n=\max\left\{n_0,\left\lceil\beta^{-1}\log(3V/\varepsilon)\right\rceil\right\},
\]

with \(E_{\rm num}\le\varepsilon/3\). Then

\[
\boxed{
W=q^{d-1}\bigl(n+1+2\lceil rn/2\rceil\bigr)
\le q^{d-1}[(1+r)n+3].}
\tag{8}
\]

Thus, at fixed dimension and fixed regularity/access constants,

\[
W=O\!\left(\frac{1+r}{a^{d-1}\beta}\log^d\frac C\varepsilon\right).
\tag{9}
\]

The actual dense parameter count is \((d+2)W+1\). Structured storage can share a direction and bandwidth across its row, but that does not alter the identity of the final MLP. The statement is geometric in angular order and in scalar resolution; in total width it is of stretched-exponential form \(e^{-cW^{1/d}}\), not \(e^{-cW}\).

## 6. Optimal allocation within this error certificate

Let \(\varepsilon_a+\varepsilon_s=\varepsilon_{\rm app}\), after reserving numerical/extension error. Ignoring integer rounding and the minimum admissible order,

\[
q=a^{-1}\log(U/\varepsilon_a),\qquad
n=\beta^{-1}\log(V/\varepsilon_s),\quad
W\approx(1+r)q^{d-1}n.
\]

Differentiation of this expression yields the exact interior stationary condition

\[
\boxed{
\varepsilon_a\log(U/\varepsilon_a)
=(d-1)\varepsilon_s\log(V/\varepsilon_s).}
\tag{10}
\]

For high accuracy with fixed constants, \(\varepsilon_a\sim(d-1)\varepsilon_{\rm app}/d\) and \(\varepsilon_s\sim\varepsilon_{\rm app}/d\). The leading balance is \(aq\sim\beta n\), hence \(n/q\sim a/\beta\). Small scalar analytic width lowers \(\beta\), requiring more centers; harder angular variation lowers \(a\), requiring more directions. Amplitudes \(U,V\) also matter, including frequency through complex bounds such as \(|\sin(ks)|\le e^{|k|\delta}\) on a strip.

For a finite target tolerance, evaluate the explicit integer bounds rather than using these asymptotics as equality. (10) optimizes this sufficient bound, not the unknown best approximation among all networks.

For spoke-dependent costs, write a certified scalar bound as \(E_m\le C_m e^{-b_m w_m}\), where \(w_m\) counts **all** neurons in that row. Set \(\eta=\varepsilon_s/2\) to account for the factor two in (6). For fixed directions, minimize \(\sum_mw_m\) subject to \(\sum_m\alpha_m C_m e^{-b_mw_m}\le\eta\). If no minimum-width constraint is active, let \(S=\sum_m b_m^{-1}\). Lagrange multipliers give

\[
w_m=\frac1{b_m}\log\frac{\alpha_m C_m b_m S}{\eta},\qquad
\alpha_m C_m e^{-b_mw_m}=\frac{\eta}{b_mS}.
\tag{11}
\]

Impose each row's minimum admissible width, recompute the active set, then round upward to a feasible geometry. No evaluation of the unknown approximation error is needed if the constants are certified. This gives more centers to slower-converging or larger-amplitude profiles. Nonuniform **numbers** of centers across directions do not require moving centers within a row away from its prescribed uniform grid.

## 7. Finite regularity and algebraic rates

For angular regularity retain pointwise continuity and replace (A) with

\[
\sup_x\sum_{\ell\ge0}(1+\ell)^{2\sigma}\|H_{x,\ell}\|_2^2\le A_\sigma^2,
\qquad \sigma>(d-1)/2.
\]

The same proof yields

\[
E_{\rm ang}(q)\le
2A_\sigma
\sqrt{\frac{2(d-1)^{d-2}}{(d-2)!(2\sigma-d+1)}}
(2q)^{-\mu},\qquad \mu=\sigma-(d-1)/2>0.
\tag{12}
\]

A nonanalytic profile must not be passed to the complex-shift formula. A fully finite real-sample alternative is described in `activation_certificate.md`: construct a bounded filtered Chebyshev polynomial from \(4L\) real samples; its error is at most four times the best degree-\(L\) polynomial error. Assume the real interval extension and the \(s\)-th smoothness/Jackson bounds hold **uniformly over directions**, with a fixed constant. Jackson approximation then gives \(O(L^{-s})\). Encode that known polynomial with the analytic tanh construction. Its bound on a fixed complex neighborhood grows at most \(\operatorname{poly}(L)e^{cL}\); therefore \(O(L+\log(1/\varepsilon))\) tanh neurons suffice. Choosing \(L\asymp\varepsilon^{-1/s}\) gives a genuine algebraic scalar construction. The final evaluator is still only the tanh network; the polynomial is a coefficient-construction intermediate, not a substitute output model.

This local finite-smooth route has a precision cost: the available complex-growth and conditioning estimates allow coefficient magnitudes \(\exp(O(L))\), so sufficient working precision is \(O(L+\log(1/\varepsilon))\) bits. It is a calculable width theorem, but not a claim that the algebraic regime is well conditioned in fixed precision. The Fourier-moment constructions in the activation module offer different, more stable certificates when their stronger global input assumptions hold.

A second, more stable **tanh** algebraic construction is proved in `finite_smooth_tanh.md`. For \(p=g'\), suppose
\[
A_s=\frac1{2\pi}\int (1+|\omega|)^s|\widehat p(\omega)|\,d\omega<\infty.
\]
Take a cutoff \(B_f\ge\max(1,e^{1/s}-1)\), \(t=s\log(1+B_f)\), \(\gamma=\pi B_f/t\), \(h=\pi/(4B_f)\), and physical halo \(\ell_h=t/\gamma\). Deconvolve only the finite frequency interval \([-B_f,B_f]\), sample that bounded density, retain centers \(|jh|\le1+\ell_h\), and anchor at zero. The actual network obeys
\[
\|g-Q\|_\infty\le9A_s(1+B_f)^{-s},\qquad
w\le\frac{8B_f}{\pi}+\frac{8s^2\log^2(1+B_f)}{\pi^2}+1.
\]
Its readout absolute sum is at most \(A_s(1+\ell_h+h/2)\); it does not require the exponentially large complex polynomial bound of the local fallback. This gives a practicable \(O(\varepsilon^{-1/s})\) scalar-width theorem for the stated global Fourier class. It is an explicit separate discretization schedule with ordinary finite halos, not a new claim about the fixed-\(\lambda\) analytic scheme.

If the scalar certificate is \(E_{\rm sc}\le Vn^{-s}\), combining it with (12) gives

\[
W=O\!\left(\varepsilon^{-[(d-1)/\mu+1/s]}\right).
\tag{13}
\]

The optimum error split for this algebraic certificate is exactly

\[
\frac{\varepsilon_a}{\varepsilon_s}=\frac{(d-1)s}{\mu}.
\]

When the effective angular and scalar orders agree, \(\mu=s\), this becomes \(W=O(\varepsilon^{-d/s})\). Finite regularity gives these algebraic bounds; an arbitrary nonanalytic continuous function need not have any prescribed algebraic rate. A nonanalytic function can also converge faster than every algebraic power, so analyticity/nonanalyticity is not a universal two-way classification.

## 8. What generalizes to other activations

For an activation \(\phi\), choose \(r\ge0\) such that \(K=\phi^{(r)}/m\), \(m=\int\phi^{(r)}\ne0\), is a usable kernel. The scalar density and sampled readouts are

\[
\widehat\rho(\omega)=\frac{(i\omega)^r\widehat g(\omega)}{\widehat K(\omega/\gamma)},\qquad
u_j=\frac{h\rho(c_j)}{m\gamma^{r-1}}.
\]

This is a conditional inverse: required frequencies must be representable and numerical amplification controlled. For \(r>1\), the lower-degree integration polynomial must also be realized using the selected activation.

`activation_certificate.md` provides an explicit finite Fourier-cutoff, sampled-kernel, finite-halo construction for exact GELU, Gaussian bumps, and Gaussian-integral activations. For \(p=g^{(r)}\), a finite exponential Fourier moment

\[
A=\frac1{2\pi}\int e^{a|\omega|}|\widehat p(\omega)|\,d\omega
\]

supports \(O(\log(1/\varepsilon))\) scalar neurons with an explicitly growing cutoff and prescribed gamma/halo. A finite algebraic Fourier moment gives an algebraic scalar rate. These are stronger global access assumptions than tanh's local tube assumption, and are not interchangeable with it. GELU's affine anchor is exactly representable by two extra neurons; affine contributions from all directions may be combined before adding those neurons.

For ReLU, \(g(s)=s^2\) gives the lower bound \(\|g-Q\|_\infty\ge1/[2(W+1)^2]\): at least one affine interval has length \(2/(W+1)\), and the best affine error on an interval of length \(\ell\) is \(\ell^2/8\). Thus no arbitrary-activation geometric theorem is possible. Polynomial activations of a fixed degree fail to approximate general functions altogether.

## 9. Boundary, precision, and computational costs

There are two different boundary questions. The supplied multidimensional extension determines the Radon profiles. The scalar real extension margin \(\Delta\), complex radius \(\delta\), and explicit halo correction make each finite neuron row approximate its known profile. One does not automatically solve the other. A compactly supported smooth cutoff is not an analytic extension, and its errors/regularity effects must be included.

The corrected tanh moment-to-readout map has a certified bound

\[
C_{\rm bd}(\lambda)\le
\frac{\sqrt\pi e^\lambda}{2\sqrt\lambda}
\exp\!\left(\frac{5\pi^2}{24\lambda}\right).
\]

For \(\lambda=4\pi/(rn)\), this is at most

\[
\frac e4\sqrt{rn}\,\exp(5\pi rn/96).
\]

The conditioning exponent is \(5/24\) of the scalar error exponent \(\beta n\). In the **fixed analytic target class**, sufficient working precision therefore grows linearly in \(\log(1/\varepsilon)\), with target-dependent constants; a universal float64 floor is not promised. This logarithmic bit bound does not apply unchanged to the finite-smooth polynomial regularization described in §7. Moment quadrature is one-dimensional and can be tolerance-controlled after rescaling its width-\(h\) boundary layer. No global readout solve appears.

For an ordinary tanh MLP on the unit ball, parameter perturbations obey

\[
\|\widetilde Q-Q\|_\infty\le |\Delta b|+
\sum_j|\Delta v_j|+
\sum_j|\widetilde v_j|(\|\Delta a_j\|_2+|\Delta\beta_j|),
\]

because \(|\tanh|\le1\) and tanh is 1-Lipschitz. Add library activation and summation errors to certify \(E_{\rm num}\). This supplies a concrete precision target, not merely the phrase 'high precision'.

Current halo algebra costs \(O(n^3)\) arithmetic to prepare; with common geometry its linear map can be precomputed once and applied in \(O(n^2)\) per spoke. Profile and moment evaluations remain separate costs. Structured inference costs \(O(dM+W)\) operations by sharing each dot product within a row. Dense unshared inference costs \(O(dW)\). At \(b\) working bits, storage in bits is proportional to parameter count times \(b\); \(\log^d\) width does not mean \(\log^d\) bit complexity.

## 10. Essential counterexample to an overly broad assumption

Take \(F(x)=\prod_{j=1}^d(1+x_j^2)^{-1}\). It is real analytic near the real domain and

\[
\widehat F(\xi)=\pi^d e^{-\|\xi\|_1}.
\]

Writing \(a(\theta)=\|\theta\|_1\), its profiles are

\[
g_\theta(s)=C_d\pi^{d-1}\Gamma(d)\operatorname{Re}(a(\theta)-is)^{-d}.
\]

At zero, \(g_\theta(0)\) is proportional to \(\|\theta\|_1^{-d}\), which has directional cusps at coordinate hyperplanes. The anchored \(H_x\) generically retains such cusps. Hence the separate angular assumption is genuinely needed for this canonical construction. This does not prove that the function itself lacks another efficient ridge representation.

## 11. Improvements that preserve the construction

- Apply angular bounds to the anchored integrand \(H_x\) directly, avoiding a gratuitous extra factor two and removing direction-dependent constants from that error.
- Use the exact harmonic tail in (1) rather than the convenient bound (2) when selecting a finite quadrature order.
- Fold antipodal directions and merge their weights.
- Optimize the physical halo length up to its analytic allowance, retaining slack for integer rounding. The balanced gamma is \(2\pi/r\); the available scalar certificate stops benefiting from halo lengths beyond approximately \(\delta/2\).
- Allocate different row widths via (11), and precompute geometry-only boundary algebra.
- Optimize the sufficient bound numerically over a small integer set after the theorem supplies a guaranteed feasible upper limit; do not call that a proof of globally optimal neural approximation.

## 12. Formalization boundary

No Lean code has been started. A natural later proof decomposition is: positive quadrature transport; spherical product exactness; harmonic tail bounds; scalar contour/quadrature identity; finite rational halo correction; explicit parameter inequalities; final neuron-count theorem. The complex analysis and computable quadrature layer are substantial obligations, not solved by formalizing only the final algebra.

Supporting derivations and audits: `tanh_certificate.md`, `angular_certificate.md`, `activation_certificate.md`, `finite_smooth_tanh.md`, `radial_extension_certificate.md`. External mathematical references are recorded in those modules. The completed argument is to be presented to Sam in chat before any permission to formalize is sought.
