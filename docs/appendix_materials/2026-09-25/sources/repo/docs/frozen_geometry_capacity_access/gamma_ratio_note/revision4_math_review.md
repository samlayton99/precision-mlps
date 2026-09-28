# Revision 4 mathematical audit

Independent review, 23 September 2026. Scope: the direct interval and direct upper reports, their implementation, the revised `gamma_ratio_note_v4.md`, and the new `note_interval_figures.py` implementation. No training or new numerical sweep was run. The central formulas in revised Sections 3–4 and Appendices A–E were checked directly; the final numerical text is recorded below.

## Consequential requirements for the revision

1. The residual bounds must use the actual finite kernel's target weights on both sides. Replacing those weights by eigenvector weights of the corrected integral matrix needs an additional projector-perturbation result and is not justified by eigenvalue intervals alone.
2. A full residual upper bound must retain all target energy. Numerically unresolved modes can receive a zero lower rate bound, leaving their energy constant in the upper curve. Dropping that energy invalidates a sufficient-time claim.
3. Numerically unresolved singular values are not proved zero eigenvalues. A truncated numerical projection remainder should be called unresolved energy unless its nullspace status is established separately.
4. The lower residual curve uses upper ratio endpoints; the upper residual curve uses lower ratio endpoints. Their first crossings bracket the actual crossing in that same order. An upper curve that never crosses supplies no finite sufficient-time guarantee; it does not prove that actual GD never crosses.
5. The exact normalized first eigenvalue is one. The last eigenvalue has a zero universal lower bound, together with the stated one-sided upper bound. These endpoints must be supplied when summing all modes.
6. Analytic alias and center-tail allowances do not enclose quadrature or floating-point error. Exact theorems and numerically evaluated intervals must remain distinct, especially in a claim of necessary or sufficient integer update counts.
7. The current source constructs a contiguous finite center block. The theorem extends to arbitrary finite lattice subsets only when every omitted interior center is explicitly included in the subtraction.

## Integral and finite-geometry calculations

For a zero-sum vector \(u\), put \(F_u(\omega)=\sum_a u_a e^{-i\omega x_a}\) and \(g(c)=\sum_a u_a\tanh(\gamma(c-x_a))\). Since \(\widehat{\gamma\operatorname{sech}^2(\gamma\cdot)}=2M_\gamma\),

\[
\widehat g(\omega)=\frac{2M_\gamma(\omega)}{i\omega}F_u(\omega).
\]

The zero-sum assumption removes the apparent singularity at zero and cancels both constant tails. Parseval gives

\[
\frac1{hm}\int |g|^2
=\frac2{\pi hm}\int
\frac{M_\gamma(\omega)^2}{\omega^2}|F_u(\omega)|^2\,d\omega.
\]

The normalization is correct. Projecting the product-minus-one identity gives
\(H_\gamma=-2Q^\top[d_{ab}\coth(\gamma d_{ab})]Q/(hm)\), with diagonal bracket value \(1/\gamma\). This is positive semidefinite on the projected space. Monotonicity of \(M_\gamma^2\) proves Loewner monotonicity of \(H_\gamma\), not of the corrected finite normalized spectrum.

For \(G_{ab}(c)=\tanh(\gamma(x_a-c))\tanh(\gamma(x_b-c))-1\), the product identity and the transform above yield

\[
\widehat G_{ab}(\nu)=
-\frac{4M_\gamma(\nu)}\nu
\sin(\nu d_{ab}/2)\coth(\gamma d_{ab})
e^{-i\nu(x_a+x_b)/2}.
\]

The diagonal limit is \(-2M_\gamma(\nu)e^{-i\nu x_a}/\gamma\). With \(\nu_k=2\pi k/h\), Poisson summation therefore gives the paired real correction

\[
A_{\gamma,k}=\frac2{hm}Q^\top
\operatorname{Re}\{e^{i\nu_kc_0}\widehat G(\nu_k)\}Q.
\]

Both its sign and factor of two agree with the source implementation: the code's sinc form expands to the same expression. The correction need not be positive semidefinite.

For strip height \(d=\vartheta/\gamma\), \(0<\vartheta<\pi/2\),
\(\|T'(\cdot\pm id)\|_2^2\le(4\gamma/3)\sec^4\vartheta\).
Subtracting \(T(c-\bar x)\), then using Minkowski and Cauchy–Schwarz, bounds each shifted integral of \(|g|^2\) by
\(A=(4\gamma/3)\sec^4\vartheta\|x-\bar x\mathbf1\|^2\|u\|^2\).
Contour shifting applies to the analytic function \(g(z)^2\). Summing only aliases \(|k|>p\) gives

\[
\delta_{\gamma,p}=
\frac{8\gamma\|x-\bar x\mathbf1\|^2\sec^4\vartheta}
{3hm[\exp(2\pi\vartheta/(\gamma h))-1]}
\exp\!\left(-\frac{2\pi\vartheta p}{\gamma h}\right).
\]

The constant, geometric-series index, and exponent are correct.

For a unit zero-sum vector, cancellation of the saturated constant and \(1-\tanh t\le2e^{-2t}\) give a single right-center contribution at most \(4e^{-4\gamma(c-x_{\max})}\) after division by \(m\). Summing the unretained right and left lattice tails proves

\[
0\preceq T_{\rm tail}\preceq\tau_\gamma I,
\qquad
\tau_\gamma=
\frac{4[e^{-4\gamma(c_R-x_{\max})}+e^{-4\gamma(x_{\min}-c_L)}]}
{1-e^{-4\gamma h}}.
\]

Thus with \(S_\gamma=H_\gamma+\sum_{k=1}^pA_{\gamma,k}-T_F\),

\[
Q^\top K_\gamma Q=S_\gamma+R_p-T_{\rm tail},
\qquad \|R_p\|\le\delta_{\gamma,p}.
\]

This proves the stated asymmetric enclosure. The sign of the exterior-center subtraction is essential and correct.

## Eigenvalue indices and normalization

Write \(\mu_j\) for the descending eigenvalues of \(Q^\top KQ\), and \(\beta_j\) for those of \(S\). Interlacing gives \(\lambda_j(K)\ge\mu_j\ge\lambda_{j+1}(K)\). Weyl's inequalities therefore give, for \(2\le i\le m-1\),

\[
[\beta_i-\delta-\tau]_+\le\lambda_i(K)
\le\beta_{i-1}+\delta.
\]

For \(i=m\), keep zero as the lower bound and \(\beta_{m-1}+\delta\) as the upper bound. For \(i=1\), the ratio equals one exactly. In exact arithmetic the upper numerators are nonnegative; clipping a negative computed numerator does not by itself certify a rounded interval.

In the basis \([e,Q]\), \(e=\mathbf1/\sqrt m\), the kernel has blocks \([a,b^\top;b,C]\). Here \(a=\|B^\top e\|^2\ge1\). Let \(c=\beta_1+\delta\), so \(C\preceq cI\). Cauchy–Schwarz on the off-diagonal term bounds every Rayleigh quotient by that of the two-by-two matrix with entries \(a,\|b\|,c\). Hence

\[
\ell=a\le\lambda_1(K)\le
L=\frac{a+c+\sqrt{(a-c)^2+4\|b\|^2}}2.
\]

Dividing the nonnegative eigenvalue lower bound by \(L\), and the upper bound by \(\ell\), is correct. The mean and coupling blocks come from finite feature products, not from an actual finite eigendecomposition; they remain gamma dependent.

The implementation approximates the integral over a finite center interval. If its integral-tail allowance is \(\zeta\), the correct additional asymmetry is upper error \(\delta+\zeta\) and lower error \(\delta+\tau\). The code uses this direction. This argument treats the retained integral as exact; numerical quadrature error still has no proved sign.

## Residual and threshold transfer

For each actual eigenmode let \(0\le l_i\le\rho_i=\lambda_i/\lambda_1\le u_i\le1\), and let \(p_i=|u_i(K)^\top y|^2/\|y\|^2\) be its actual target weight. To avoid overloading the eigenvector notation, the revised note should use visually distinct symbols for ratio upper endpoints and eigenvectors. Under zero initialization and \(\eta=1/(2\lambda_1)\), the function \((1-\rho/2)^{2n}\) is nonincreasing on \([0,1]\). Therefore

\[
\underbrace{\sum_i p_i(1-u_i/2)^{2n}}_{E_{\rm low}(n)^2}
\le E(n)^2\le
\underbrace{\sum_i p_i(1-l_i/2)^{2n}}_{E_{\rm up}(n)^2}.
\]

This theorem needs neither corresponding eigenvectors of \(S\) nor a pairing of eigenvectors across different gammas. It does require the actual target weights. Actual zero modes may use \(l_i=u_i=0\), preserving their constant contribution. If only a subset of modes is resolved and the remaining mass is \(q\), a safe upper curve adds \(q\) as constant energy. A safe lower curve may omit it; alternatively it may add \(q\,2^{-2n}\) using the universal upper rate one. These choices provide weaker but valid envelopes. Tighter upper endpoints for the unresolved mass may be supplied by eigenvalue ordering when justified.

Define first crossings over nonnegative integer updates, with the infimum of the empty set equal to infinity. Pointwise ordering gives

\[
\inf\{n:E_{\rm low}(n)\le\epsilon\}
\le n_\epsilon\le
\inf\{n:E_{\rm up}(n)\le\epsilon\}.
\]

At a particular budget \(N\), \(E_{\rm low}(N)>\epsilon\) proves failure by that budget, while \(E_{\rm up}(N)\le\epsilon\) proves success by that budget. Equality at an asymptotic residual floor needs care: a curve can approach the tolerance forever without reaching it at a finite integer update. A search cap also differs from infinity; failure to cross before a finite computational cap is only that statement.

For any selected positive-spectrum tail beginning at rank \(i\), ordering gives every selected ratio at most the valid upper endpoint \(u_i\). Its mass \(P_i\) then supplies

\[
E(n)^2\ge P_i(1-u_i/2)^{2n},
\qquad
n_\epsilon\ge
\frac{\log(\sqrt{P_i}/\epsilon)}{-\log(1-u_i/2)}
\]

when \(P_i>\epsilon^2\). Excluding unresolved positive modes only reduces the mass and weakens the necessary bound, provided the retained weights and positivity are valid. Including nullspace mass also gives a valid error lower bound, but it cannot then be presented as a delay confined to representable components. If \(u_i=0\) and \(P_i>\epsilon^2\), the obstruction is infinite; the displayed quotient needs this limiting interpretation.

## Audit status

No consequential algebraic defect found in the reviewed Fourier, Poisson, remainder, interval, or scalar normalization formulas. The important review items for the revised note are the completeness and meaning of the numerical residual envelopes, the distinction between unresolved and null energy, and the separation of checked floating-point evidence from exact numerical certification.

The revised main argument meets the mathematical requirements above: (10) is a two-sided absolute finite-ratio interval, (11) uses the same actual weights on both sides, and (12) has the correct first-crossing direction. Endpoint and structural zero-eigenvalue treatment is explicit. Equation (13) has the correct necessary-time direction and positive-target-mass condition. The scalar-schedule bound in Appendix E.2 is valid under the stated per-step stability condition. The two-sample counterexample in E.3 has the displayed even/odd eigenvalues and asymptotic ratio; their difference is \(\operatorname{sech}^2\gamma+2ab>0\), validating their ordering.

The new figure implementation uses upper ratio endpoints for the lower residual and lower endpoints for the upper residual. It forces unresolved lower rates to zero and preserves all their energy in the upper curve. Following the coordinator's adjustment, its lower-curve weights also set the appended computed feature-span residual to zero. I checked this last setting directly in the saved data at all five gammas. The upper curve retains that energy. This avoids treating a numerically estimated null projection as a proved lower residual floor.

The extra numerical allowance \(64\epsilon_{64}\|S\|_2\) is an empirical guard, not an established roundoff bound. The recorded quadrature-refinement operator discrepancies are smaller than this allowance at all five evaluated slopes. Strict ratio ordering, independent SVD, and residual ordering checks support the numerical evaluation, but do not convert it into interval arithmetic. The revised prose must preserve that distinction.

One build defect and two minor precision edits were sent to the coordinator: the draft Figure 3 filename differed from the actual generated `full_residual_and_steps.png`; the exact theorem should call \(\beta_i\) the eigenvalues of \(S\), separating their numerical evaluation; and the scalar-schedule discussion should describe an approximately fourfold weakening of this necessary lower bound rather than suggest a general fourfold actual-speedup theorem. All three are corrected in the final Markdown reviewed here.

The numerical routines are scoped to the plotted geometry, where \(m>W+1\) and centers are consecutive and ordered. Their array slices are not a general implementation for \(m\le W+1\), although the theorem itself includes that case with the stated endpoint rules. No such broader software claim is needed for this note.

Final check: the inserted Section 4 table agrees with the saved numerical results. The gamma-4 endpoints \(9.20\times10^{11}\) and \(3.31\times10^{12}\) are rounded outward from 920,663,607,184 and 3,303,588,843,394; the remaining integer entries match the saved table. Appendix F.2 correctly describes the empirical guard, the extra integral-tail direction, the conservative handling of all unresolved energy, the removal of computed projection-remainder energy from the lower curve, and the finite search cap. The placeholders are removed and Figure 3 uses the generated file. No remaining consequential mathematical correction was identified. The numerical-certification limitation is explicitly disclosed and remains a limitation of the evidence, not an algebraic gap in the stated theorem.
