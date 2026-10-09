# A finite, explicitly computable tanh spoke with logarithmic width

This note establishes the scalar component needed by the QI–Radon scaling argument. It uses the existing complex-shift density and rational halo correction. It does not introduce fitted coefficients, a different activation, an auxiliary representation, or a PDE solver. No Lean formalization is undertaken here.

The principal change is the asymptotic parameter schedule: keep the physical bandwidth and halo extent fixed while refining the center grid. Consequently the dimensionless overlap parameter decreases and the number of halo neurons increases proportionally to the number of interior neurons. This differs from keeping the overlap parameter fixed and using a square-root halo.

## 1. Objects, information, and assumptions

The target is a real function \(g:[-1,1]\to\mathbb R\). It is supplied with a complex-evaluation procedure on a specified neighborhood. Choose constants \(B>0\), \(\delta>0\), and \(\Delta>0\), and assume that \(g\) extends holomorphically to

\[
U_\delta(I_\Delta)
=\{z\in\mathbb C:\operatorname{dist}(z,I_\Delta)<\delta\},
\qquad I_\Delta=[-1-\Delta,1+\Delta],
\]

is real on \(I_\Delta\), and satisfies \(|g(z)|\le B\) on that neighborhood.

The extension is an input assumption. It cannot be inferred from arbitrary real samples. The construction below does not need a full-line integral of \(g\), global decay, or a Fourier transform of the original scalar target. Its proof uses finite contours inside this neighborhood.

There are \(n+1\) interior centers, where \(n\) counts intervals. There are \(R\) halo centers per side. The resulting scalar network has

\[
W_s=n+1+2R
\]

tanh neurons and one output bias:

\[
q(s)=b_g+\sum_{j=-R}^{n+R}u_j\tanh(\gamma(s-c_j)).
\]

Here \(u_j\in\mathbb R\) are readouts; \(c_j\in\mathbb R\) are centers; and \(\gamma>0\) is the shared inverse width. All parameters are explicitly calculated below.

Computable target access means that complex values and the finite one-dimensional integrals can be evaluated with certified absolute tolerances. A closed-form integral is not required. The error from numerical evaluation is accounted for separately from the exact-arithmetic approximation error.

## 2. Geometry and a convenient parameter schedule

Choose any fixed

\[
0<r\le\min\{\Delta/2,\delta/2\}.
\]

This \(r\) is a nominal *physical* halo length. Set

\[
\gamma=\frac{2\pi}{r},\qquad
\tau=\frac{\pi}{2\gamma}=\frac r4,\qquad
\beta=\frac{\pi r}{4}.
\]

For an integer \(n\), define

\[
h=\frac2n,\qquad
\lambda=\gamma h=\frac{4\pi}{rn},\qquad
R=\left\lceil\frac{rn}{2}\right\rceil,
\qquad c_j=-1+jh.
\]

Take

\[
n\ge n_0
:=\left\lceil\max\left\{16,\frac{4\pi}{r},\frac3{\Delta-r}\right\}\right\rceil.
\]

The actual half-cell halo extent is

\[
\ell_h=(R+\tfrac12)h.
\]

Because \(R\le rn/2+1\),

\[
\ell_h\le r+\frac3n\le r+(\Delta-r)=\Delta.
\]

The other conditions used in the finite-contour proof hold as well:

\[
\lambda\le1,
\qquad \gamma\delta=\frac{2\pi\delta}{r}\ge4\pi,
\qquad \lambda R\ge\gamma r=2\pi\ge2\log2.
\]

Thus every center, endpoint, and complex contour used below remains in the prescribed analytic neighborhood. The neuron count satisfies

\[
W_s\le(1+r)n+3.
\]

The original encoder accepts \(R\) as an override. Its current input guard also asks for \(\ell_h\le\delta\), which this schedule already satisfies:

\[
\ell_h\le r+3/n
\le r\left(1+\frac3{4\pi}\right)<1.24r\le0.62\delta.
\]

Thus this particular schedule can run through the existing guard without modification. Other schedules that use distinct \(\Delta\) and \(\delta\) may require separating the physical-halo and analytic-strip checks; this certificate does not.

## 3. Full finite coefficient recipe

The scalar reference point is \(s_0=-1\). Define

\[
a=-1-\ell_h,\qquad b=1+\ell_h,
\]

so the centers are the midpoints of equal cells covering \([a,b]\). Define the analytic coefficient density

\[
\rho(z)=\frac{g(z+i\tau)-g(z-i\tau)}{2i\tau}.
\]

For real \(z\), conjugate symmetry makes \(\rho(z)\) real. Initialize the neuron readouts by

\[
u_j^{\rm base}=\frac h2\rho(c_j).
\]

These baseline readouts must be supplemented by the following boundary correction.

Choose the contour height and number of corrected slots:

\[
k_0=\left\lceil\frac{\delta}{8\tau}\right\rceil,
\qquad \sigma=2k_0\tau,
\qquad L=\left\lfloor\frac{R+1}{2}\right\rfloor.
\]

Then \(\delta/4\le\sigma\le\delta/2\) and \(\sigma+\tau\le5\delta/8\).

For each endpoint pair \((e,\epsilon)=(a,+1)\) or \((b,-1)\), and for \(\ell=1,\ldots,L\), compute the real moments

\[
\mu_\ell(e,\epsilon)
=\frac1{2\tau}\int_{-\tau}^{\tau}
g(e+iy)e^{2i\epsilon\ell\gamma y}\,dy,
\]

\[
\nu_\ell(e,\epsilon)
=2(-1)^\ell\operatorname{Im}\int_0^\sigma
\frac{\rho(e+iy)e^{2i\epsilon\ell\gamma y}}
{1+e^{2\pi y/h}}\,dy,
\]

\[
G_\ell^-=\mu_\ell(a,+1)-\nu_\ell(a,+1),\qquad
G_\ell^+=\mu_\ell(b,-1)-\nu_\ell(b,-1).
\]

For stable evaluation of \(\mu_\ell\), its constant component may be removed exactly:

\[
\mu_\ell(e,\epsilon)
=\frac1\tau\operatorname{Re}\int_0^\tau
[g(e+iy)-g(e)]e^{2i\epsilon\ell\gamma y}\,dy.
\]

To calculate the rational-correction coefficients without evaluating cancelling polynomials, put

\[
\zeta=e^{-2\lambda},\qquad
r_i=\zeta^{i-1/2},\qquad
p_j=\prod_{k=1}^j(1-\zeta^k),\quad p_0=1.
\]

For \(i=1,\ldots,L\), let \(e_k(r_{-i})\) be the coefficient of \(z^k\) in

\[
\prod_{j\ne i}(1+r_jz).
\]

These coefficients are obtained by finite polynomial multiplication using nonnegative numbers. Define

\[
\boxed{
\kappa_i^\pm
=\frac{(-1)^i\zeta^{[i(i+1)-1]/2}}
{p_{i-1}p_{L-i}}
\sum_{\ell=1}^L G_\ell^\pm e_{\ell-1}(r_{-i}).
}
\]

Start with the baseline readouts and modify the existing outer halo slots:

\[
u_{-R+i-1}\mathrel{+}=\kappa_i^-/2,
\qquad
u_{n+R-i+1}\mathrel{-}=\kappa_i^+/2,
\qquad i=1,\ldots,L.
\]

Finally set

\[
\boxed{
b_g=g(-1)-\sum_{j=-R}^{n+R}
u_j\tanh(\gamma(-1-c_j)).
}
\]

The output is the ordinary tanh network stated above. There is no fitting or matrix solve. The elementary symmetric polynomials and moment integrals determine all the readouts.

## 4. The finite-contour error lemma being used

The main PDF note, section “The corrected tanh scalar construction,” proves the following finite-contour estimates. They are the scalar lemma imported here; this scaling argument does not silently assume that bare midpoint quadrature converges geometrically.

Define

\[
D_s(z)=\tfrac12[\tanh(\gamma(s-z))-\tanh(\gamma(s_0-z))],
\qquad F_s(z)=\rho(z)D_s(z).
\]

The exact finite-contour identity is

\[
g(s)=g(s_0)+\int_a^bF_s(t)\,dt+\mathcal B[g](s),
\]

where

\[
\mathcal B[g](s)=\frac1{4\tau}\int_{-\tau}^{\tau}
\left[g(a+iy)H_s(a+iy)-g(b+iy)H_s(b+iy)\right]dy,
\]

\[
H_s(z)=\coth(\gamma(s-z))-\coth(\gamma(s_0-z)).
\]

The identity follows by closing the rectangle \([a,b]+i[-\tau,\tau]\), using \(H_s(t\pm i\tau)=2D_s(t)\), and taking the residues at \(s\) and \(s_0\).

The signed midpoint error separates into horizontal-contour, vertical-endpoint, and tanh-pole terms. Combining that error with \(\mathcal B[g]\) gives

\[
g-q=(\mathcal B_{\rm eff}-q_{\rm bd})+\mathcal P-\mathcal H.
\]

Here \(\mathcal B_{\rm eff}\) is the physical boundary contribution minus the vertical midpoint contribution. The moments \(G_\ell^\pm\) above are precisely the power-series coefficients of this combined boundary contribution. Correcting only the physical tail would omit part of the error.

The relevant estimates are

\[
\boxed{
E_{\rm hor}\le\frac{32B}{\delta}e^{-\pi\delta n/4},\qquad
E_{\rm alias}\le\frac{64\pi B}{\lambda\delta n}e^{-\pi^2/\lambda},
}
\]

\[
\boxed{
E_{\rm halo}\le\frac{32BD_\lambda}{\delta n}e^{-\lambda R^2/4},
\qquad D_\lambda=\frac\pi{2\lambda}+\frac{4\log2}{\pi}.
}
\]

Their sum bounds \(\|g-q\|_{L^\infty([-1,1])}\).

For clarity, the origin of the halo exponent is as follows. The moments satisfy

\[
|G_\ell^\pm|\le M_h:=\frac{BhD_\lambda}{\delta}.
\]

Set \(\vartheta=e^{-\gamma\ell_h}\), and write \(G^\pm(z)=\sum_{\ell\ge1}G_\ell^\pm z^\ell\). The physical input gives \(0\le z\le\vartheta^2\). With \(t_i=1/r_i\), let

\[
D_L(z)=\prod_{i=1}^L(1+t_i z),\qquad
P_L^\pm=[D_LG^\pm]_{\le L}.
\]

Since \(t_i\vartheta\le1\), Cauchy estimates on \(|z|=\vartheta\) give

\[
\left|G^\pm(z)-P_L^\pm(z)/D_L(z)\right|
\le\frac{M_h\vartheta}{(1-\vartheta)^2}(2\vartheta)^L.
\]

There are four anchored endpoint evaluations. Since \(\lambda R\ge2\log2\), \(L\ge R/2\), and \(\vartheta\le e^{-\lambda R}\), their total is bounded by

\[
16M_he^{-\lambda R^2/4}.
\]

The partial fractions of \(P_L^\pm/D_L\) are exactly the coefficients \(\kappa_i^\pm\) above. Each fraction is a constant plus or minus half a tanh centered at an existing outer halo node. Thus this estimate applies to the actual stated network, not to an unevaluated rational surrogate.

## 5. Derivation of a single exponential certificate

With \(\gamma=2\pi/r\), \(\lambda=2\gamma/n\), and \(R\ge rn/2\),

\[
\frac{\pi\delta n}{4}\ge2\beta n,
\qquad
\frac{\pi^2}{\lambda}=\beta n,
\qquad
\frac{\lambda R^2}{4}\ge\beta n.
\]

The horizontal error is consequently at most

\[
E_{\rm hor}\le\frac{32B}{\delta}e^{-2\beta n}.
\]

The alias prefactor simplifies without growing with \(n\):

\[
\frac{64\pi B}{\lambda\delta n}
=\frac{32\pi B}{\gamma\delta}
=\frac{16rB}{\delta}.
\]

For the halo prefactor, use \(n\ge2\gamma\):

\[
\begin{aligned}
\frac{32BD_\lambda}{\delta n}
&=\frac{8\pi B}{\gamma\delta}
+\frac{128B\log2}{\pi\delta n}\\
&\le\frac{rB}{\delta}
\left(4+\frac{32\log2}{\pi^2}\right).
\end{aligned}
\]

Combining the three terms proves

\[
\|g-q\|_\infty
\le\frac B\delta
\left[32+20r+\frac{128\log2}{\pi n}\right]e^{-\beta n}.
\]

Since \(n\ge16\), \(128\log2/(\pi n)<2\). A particularly simple certificate is therefore

\[
\boxed{
\|g-q\|_\infty\le C_{\rm simple}B e^{-\beta n},
\qquad C_{\rm simple}=\frac{34+20r}{\delta}.
}
\]

Using \(n\ge2\gamma\) instead also gives the alternative constant

\[
\boxed{
\|g-q\|_\infty\le C_{\delta,r}B e^{-\beta n},
\qquad
C_{\delta,r}=\frac{32}{\delta}
+\left(20+\frac{32\log2}{\pi^2}\right)\frac r\delta.
}
\]

Because \(r/\delta\le1/2\), the simpler choice

\[
C_\delta=32/\delta+12
\]

also works.

To achieve an exact-arithmetic scalar error tolerance \(\varepsilon_s>0\), choose

\[
\boxed{
n=\left\lceil\max\left\{
n_0,\ \beta^{-1}\log\frac{C_\delta B}{\varepsilon_s}
\right\}\right\rceil.
}
\]

Then \(\|g-q\|_\infty\le\varepsilon_s\), and

\[
W_s\le(1+r)n+3
=O\!\left(\frac{1+r}{r}\log\frac{C_\delta B}{\varepsilon_s}+n_0\right).
\]

The constants \(r,\delta,\Delta,B\) belong to the supplied analytic class. If they depend on frequency, dimension, a cutoff, or the tolerance, those dependencies must be retained; they cannot be hidden while asserting an asymptotic law.

An equivalent direct tolerance schedule, which does not first select a fixed physical halo, is

\[
T=\max\left\{\pi^2,\log\frac{B(12+32/\delta)}{\varepsilon_s}\right\},
\qquad \lambda=\pi^2/T,\qquad R=\lceil2T/\pi\rceil,
\]

\[
n=\left\lceil\max\left\{
16,\frac{8T}{\pi\delta},\frac{2(R+1/2)}\Delta
\right\}\right\rceil.
\]

The original three-term estimates then give respectively
\((32B/\delta)e^{-2T}\), \(8Be^{-T}\), and \(4Be^{-T}\). This also achieves \(\varepsilon_s\) with logarithmic total scalar width. The fixed-\(\gamma\) schedule above is more convenient for presenting a single rate \(\beta\) and optimizing the angular-versus-scalar allocation.

## 6. Why this schedule is different from the square-root halo schedule

At fixed \(\lambda\), the existing certificate contains a nonzero \(e^{-\pi^2/\lambda}/n\) term. It is not a purely exponential estimate as \(n\to\infty\). For the sampled-density infinite lattice, a single sinusoidal target also exhibits a genuine nonzero alias coefficient proportional to \(1/n\) at fixed \(\lambda\).

To make the alias exponent proportional to \(\log(1/\varepsilon_s)\), choose \(\lambda\asymp1/\log(1/\varepsilon_s)\). If one additionally enforces \(R\asymp\sqrt n\), the corrected boundary exponent is \(\lambda n\), and this certificate requires \(n\asymp\log^2(1/\varepsilon_s)\).

The present schedule instead uses \(R\asymp n\) and \(\lambda\asymp1/n\). Both the alias and corrected boundary exponents then grow linearly in \(n\). The physical halo length remains fixed. Bare truncation without rational correction would still have only the constant exponent \(2\lambda R\), so that correction is indispensable.

The balance \(\gamma=2\pi/r\) equates the alias rate \(\pi^2/(2\gamma)\) and halo rate \(\gamma r^2/8\). It optimizes those two sufficient exponential factors for fixed physical halo. It is not an optimality theorem over all neural constructions.

## 7. Conditioning of the explicit coefficients

The original note contains a stronger bound than the crude \(2^L\) estimate. It is valid and has been independently checked. The endpoint moment-to-readout map satisfies

\[
\sum_{i=1}^L|\kappa_i^\pm|
\le C_{\rm bd}(\lambda)\max_\ell|G_\ell^\pm|,
\]

\[
\boxed{
C_{\rm bd}(\lambda)
\le\frac{\sqrt\pi e^\lambda}{2\sqrt\lambda}
\exp\left(\frac{5\pi^2}{24\lambda}\right).
}
\]

Here is a proof, including the dependence on \(\lambda\). Let

\[
p_\infty=\prod_{j\ge1}(1-e^{-2\lambda j}),\qquad
L_+=\prod_{j\ge1}(1+e^{-2\lambda(j-1/2)}).
\]

Summing the positive elementary coefficients in the scaled formula gives

\[
C_{\rm bd}\le
p_\infty^{-2}L_+e^\lambda
\sum_{i\ge1}e^{-\lambda i(i+1)}.
\]

For the first product, the logarithmic series gives

\[
\begin{aligned}
\log p_\infty^{-1}
&=\sum_{m\ge1}\frac1{m(e^{2\lambda m}-1)}\\
&\le\frac1{2\lambda}\sum_{m\ge1}\frac1{m^2}
=\frac{\pi^2}{12\lambda}.
\end{aligned}
\]

For the second product, \(t\mapsto\log(1+e^{-t})\) is convex. Each midpoint rectangle lies below its corresponding integral. Therefore

\[
\log L_+
\le\frac1{2\lambda}\int_0^\infty\log(1+e^{-t})\,dt
=\frac{\pi^2}{24\lambda}.
\]

Finally,

\[
\sum_{i\ge1}e^{-\lambda i(i+1)}
\le\sum_{i\ge1}e^{-\lambda i^2}
\le\int_0^\infty e^{-\lambda t^2}\,dt
=\frac{\sqrt\pi}{2\sqrt\lambda}.
\]

The three factors establish the stated bound.

Under the chosen schedule, \(\lambda=4\pi/(rn)\le1\), so

\[
\boxed{
C_{\rm bd}\le\frac e4\sqrt{rn}
\exp\left(\frac{5\pi rn}{96}\right)
=\frac e4\sqrt{rn}\,e^{(5/24)\beta n}.
}
\]

Thus the available condition bound grows exponentially in width, but at a quantified rate. At \(n\) proportional to \(\log(1/\varepsilon_s)\), it grows at most like a power of \(1/\varepsilon_s\), times a square-root logarithm. This does not give uniform double-precision stability. It does imply that a number of working bits proportional to \(\log(1/\varepsilon_s)\) suffices for this conditioning allowance.

The full readout bound is

\[
\sum_j|u_j|
\le\frac{2B(1+\Delta)}\delta
+C_{\rm bd}(\lambda)\frac{BhD_\lambda}{\delta},
\qquad
|b_g|\le B+\sum_j|u_j|.
\]

## 8. Turning the exact formulas into a certified numerical construction

Suppose the complex values used for real-center density evaluations have absolute error at most \(\eta_g\), every computed boundary moment has error at most \(\eta_G\), and the anchor value has error at most \(\eta_0\). In exact subsequent arithmetic,

\[
\sum_j|\Delta u_j|
\le\frac{W_sh}{2\tau}\eta_g+C_{\rm bd}(\lambda)\eta_G.
\]

Recomputing the anchor bias therefore gives function error at most

\[
\boxed{
\eta_0+\frac{W_sh}{\tau}\eta_g+2C_{\rm bd}(\lambda)\eta_G.
}
\]

For example, to reserve a tolerance \(\varepsilon_{\rm data}\) for these three sources, choose

\[
\eta_0\le\varepsilon_{\rm data}/3,\qquad
\eta_g\le\frac{\varepsilon_{\rm data}\tau}{3W_sh},\qquad
\eta_G\le\frac{\varepsilon_{\rm data}}{6C_{\rm bd}(\lambda)}.
\]

Moment quadrature error is included in \(\eta_G\). Errors in the partial-fraction arithmetic itself, geometry storage, and network evaluation are additional allowances. On real inputs \(|\tanh|\le1\) and \(|\tanh'|\le1\), so each can be propagated using coefficient absolute sums and absolute preactivation errors. Sequential floating summation contributes the usual factor proportional to unit roundoff times the number of operations; pairwise summation reduces that dependence.

The endpoint integrals are finite and computable. In particular, the vertical moment has a shrinking real boundary layer, so a fixed quadrature order is not a proof of numerical accuracy. With \(y=ht\), it becomes

\[
\nu_\ell
=2h(-1)^\ell\operatorname{Im}
\int_0^{\sigma/h}
\frac{\rho(e+iht)e^{2i\epsilon\ell\lambda t}}
{1+e^{2\pi t}}\,dt.
\]

The phase frequency \(\ell\lambda\) stays bounded under the schedule. The tail beyond \(t=T_0\) is bounded by

\[
\frac{hM_\sigma}{\pi}e^{-2\pi T_0},
\qquad M_\sigma\le4B/\delta,
\]

as long as it is within the stated integration interval. After this tail is accounted for, composite quadrature on bounded-length panels has a uniform separation from the logistic poles. Analytic bounds or interval quadrature can certify the required \(\eta_G\). Increasing quadrature until two numerical answers agree is a diagnostic, not a certificate by itself.

The present reference implementation recomputes excluded-node elementary polynomials and uses \(O(L^3)\) scalar arithmetic operations for the correction. The logarithmic-width theorem does not claim linear-time construction. Both the coefficient calculation and certified quadrature are finite, use no hidden solve, and can be implemented with polynomial work in the requested accuracy logarithm under the stated target-access assumptions.

## 9. Limits that must accompany the Radon theorem

1. This scalar result is specific to tanh and its explicit finite-boundary correction. Other activations need their own computable scalar theorem and boundary treatment.
2. A theorem claiming geometric convergence for every activation and analytic target is false. A width-\(W\) ReLU network has at most \(W\) knots. Approximating \(s^2\) on \([-1,1]\) incurs uniform error at least \(1/[2(W+1)^2]\), because one affine interval has length at least \(2/(W+1)\), and the best affine error on an interval of length \(\ell\) is \(\ell^2/8\).
3. The final positive angular sum inherits the weighted sum of the scalar errors. Angular approximation, profile acquisition, extension error, and floating arithmetic remain separate terms.
4. The supplied analytic bound \(B\) may encode large oscillation or complex growth. Its dependence must remain in the final constants and allocation rule.
5. The proof dependency is explicit: the finite-contour lemma is the existing scalar theorem; the new result here is the parameter schedule, constant reduction, width estimate, and quantified precision consequences.
