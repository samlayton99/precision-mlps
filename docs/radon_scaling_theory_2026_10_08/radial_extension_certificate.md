# A radial extension that supplies both scalar and angular analyticity

Date: 2026-10-08. This is a target-side corollary for the canonical whole-space Fourier–Radon representation. It uses a specified radial smooth cutoff and finite integrals of the known analytic target. It does not replace the output network, fit coefficients, or change the Radon reconstruction filter. No Lean formalization has been started.

The central point is that a nonanalytic cutoff need not destroy the **interior** analyticity of the Radon profiles. A radial cutoff can be expressed as an average of concentric balls. Its coefficients then remain on a real radius-integration interval while the target itself is evaluated analytically. This also preserves analytic dependence on rotations. The proof below makes those statements quantitative in every dimension, including the nonlocal filter in even dimensions.

## 1. Supplied target and an explicitly chosen extension

The target is required on the unit ball in \(\mathbb R^d\), with \(d\ge2\). Assume a known real-valued function \(f\) has a holomorphic extension to

\[
\mathcal T_\sigma(B_R)=\{z\in\mathbb C^d:\operatorname{dist}(z,B_R)<\sigma\},
\qquad R>1,\quad \sigma>0,
\]

and \(|f(z)|\le B\) throughout that set. The distance and vector norm are Euclidean. This requires the target on a slightly larger real ball and a specified complex neighborhood; no knowledge of the target arbitrarily far away is needed. Analytic evaluation with certified errors is part of the construction's access model.

Choose fixed real numbers

\[
1<T<S<R_0<R.
\]

For a completely prescribed choice, set \(h_0=(R-1)/4\), \(T=1+h_0\), \(S=1+2h_0\), and \(R_0=1+3h_0\). The more general notation permits optimization of constants.

Let \(w\ge0\) be a normalized \(C^\infty\) function supported in \([R_0,R]\), positive in its interior, flat at both endpoints, and satisfying \(\int_{R_0}^R w(a)\,da=1\). One explicit option is the normalized version of

\[
w_0(a)=\begin{cases}
\exp[-1/((a-R_0)(R-a))],&R_0<a<R,\\
0,&\text{otherwise}.
\end{cases}
\]

Define a radial cutoff

\[
\chi(r)=\int_{R_0}^R w(a)\mathbf 1_{\{r\le a\}}\,da.
\]

Then \(\chi(r)=1\) for \(r\le R_0\), \(\chi(r)=0\) for \(r\ge R\), and \(0\le\chi\le1\). Define the known extension

\[
F(x)=\begin{cases}f(x)\chi(|x|),&|x|<R,\\0,&|x|\ge R.\end{cases}
\]

It is smooth and compactly supported, and agrees **exactly** with \(f\) on the target unit ball. Thus this particular extension spends no approximation-error budget. It is intentionally not holomorphic globally. The proof only needs the reconstructed profiles to be holomorphic on the scalar interval used by their finite neuron rows.

## 2. The ball-mixture formula for the Radon projection

For a real unit direction \(\theta\), let \(V_\theta:\mathbb R^{d-1}\to\theta^\perp\) be any real orthonormal frame. The Radon projection of \(F\) is

\[
p_\theta(s)=\int_{\theta^\perp}F(s\theta+y)\,dy.
\]

Insert the definition of \(\chi\) and exchange the finite integrals. The indicator restricts the cross-section to \(|y|\le\sqrt{a^2-s^2}\). For real \(|s|<R_0\), substitute \(y=\sqrt{a^2-s^2}\,V_\theta u\):

\[
\boxed{
p_\theta(s)=\int_{R_0}^R w(a)(a^2-s^2)^{(d-1)/2}
\int_{B^{d-1}}f\bigl(s\theta+\sqrt{a^2-s^2}\,V_\theta u\bigr)\,du\,da.
}
\tag{R1}
\]

The cutoff is represented entirely by \(w(a)\) at real radii. Formula (R1) never evaluates \(\chi\) at a complex argument.

For other real \(s\), integrate only over \(a\ge|s|\); the projection is zero when \(|s|\ge R\). This gives a finite real formula for all projection values needed later by the Hilbert transform.

## 3. Quantitative scalar and rotation neighborhoods for \(p\)

Write \(V_j=\operatorname{vol}(B^j)\). Define

\[
t_0=\log\left(1+\frac\sigma{4R}\right),\qquad
Q_*=(S+R_0)/2,
\qquad
L_*=1+\frac{Q_*}{\sqrt{R_0^2-Q_*^2}},
\]

and choose

\[
\boxed{
\eta=\min\left\{\frac{R_0-S}{8},\frac{S-T}{4},\frac\sigma{16e^{t_0}L_*}\right\}>0.
}
\tag{R2}
\]

Let \(\mathcal U_j\) denote the complex tube of radius \(j\eta\) around the real interval \([-S,S]\). If \(z\in\mathcal U_4\), then \(|z|<S+4\eta\le Q_*<R_0\). Consequently \(\sqrt{a^2-z^2}\), using its positive-real branch on \((-a,a)\), is holomorphic for every \(a\in[R_0,R]\).

Choose a nearest point \(s_0\in[-S,S]\) to \(z\). Along the segment from \(s_0\) to \(z\),

\[
\left|\frac{d}{dz}\sqrt{a^2-z^2}\right|
=\frac{|z|}{|\sqrt{a^2-z^2}|}
\le\frac{Q_*}{\sqrt{R_0^2-Q_*^2}}.
\]

For \(|u|\le1\), this bounds the displacement of the target's argument in (R1):

\[
\left|z\theta+\sqrt{a^2-z^2}V_\theta u
-\bigl(s_0\theta+\sqrt{a^2-s_0^2}V_\theta u\bigr)\right|
\le L_*|z-s_0|<4\eta L_*.
\]

The parenthesized real point lies in \(B_R\).

We also allow complex rotations. Let \(J_{ij}\) be the real skew-symmetric coordinate-plane generator of operator norm one. For a product

\[
O=\exp(z_1J_{i_1j_1})\cdots\exp(z_kJ_{i_kj_k}),
\qquad \sum_{j=1}^k|z_j|<t_0,
\]

the matrix exponential series and a telescoping product give

\[
\|O\|\le e^{t_0},\qquad \|O-I\|\le e^{t_0}-1.
\]

Replace \(\theta,V_\theta\) in (R1) by \(O\theta,OV_\theta\). The distance of the new argument of \(f\) to its original real point is at most

\[
4e^{t_0}\eta L_*+R(e^{t_0}-1)\le\sigma/4+\sigma/4<\sigma.
\]

Thus the target is evaluated within its supplied holomorphic domain. Formula (R1) supplies a holomorphic lift in the complex rotation parameters and in \(z\in\mathcal U_4\). On real rotations it is exactly the usual rotated Radon projection. This is all that is needed for the angular derivative proof; no global choice of a complex transverse frame is required.

Since \(|a^2-z^2|\le R^2+Q_*^2\), the same calculation yields the explicit bound

\[
\boxed{
|p_O(z)|\le P:=B V_{d-1}(R^2+Q_*^2)^{(d-1)/2},\qquad z\in\mathcal U_4.
}
\tag{R3}
\]

Here \(p_O\) denotes the holomorphic rotated lift for any fixed initial real direction. The bound is uniform in that direction and in the allowed rotations.

For real \(s\), the original real cross-section formula, with \(f\) evaluated after the complex rotation, also gives

\[
|p_O(s)|\le B V_{d-1}(R^2-s^2)^{(d-1)/2}\mathbf 1_{\{|s|\le R\}}.
\]

Integrating the cross-section volumes,

\[
\boxed{\int_{\mathbb R}|p_O(s)|\,ds\le V:=B V_d R^d.}
\tag{R4}
\]

Smoothness and compact support of the real extension ensure that these real projection functions and their needed derivatives are well defined. The complex-rotation version inherits the same property because only the analytic factor \(f\), not the radial cutoff, is rotated in the lifted integral.

## 4. The canonical reconstruction filter in odd dimensions

Set

\[
C_d=\frac{|S^{d-1}|}{2(2\pi)^{d-1}},\qquad r=d-1.
\]

The canonical scalar profile is \(g_\theta=C_d|D_s|^r p_\theta\), where \(|D_s|^r\) means multiplication by \(|\omega|^r\) in Fourier space. This is exactly the whole-space Radon filter used in the original construction.

If \(d\) is odd, \(r=2k\) is even and

\[
|D_s|^r p=(-1)^k p^{(r)}.
\]

Let \(z\) lie in the complex tube of radius \(\eta\) around \([-T,T]\). A circle of radius \(\eta\) centered at \(z\) remains in \(\mathcal U_4\). Cauchy's derivative formula and (R3) give

\[
\boxed{|g_O(z)|\le B_g:=C_d r!\eta^{-r}P\qquad(d\text{ odd}).}
\tag{R5}
\]

The estimate is conservative: larger available Cauchy circles improve the constant. It already supplies the scalar tube and real extension margin required by the finite tanh construction: \(\delta=\eta\) and \(\Delta=T-1\).

## 5. The nonlocal filter in even dimensions

For even \(d\), write \(r=2k+1\). Define the Hilbert transform by

\[
(\mathcal Hp)(s)=\frac1\pi\operatorname{PV}\int_{\mathbb R}\frac{p(t)}{s-t}\,dt.
\]

Its Fourier multiplier is \(-i\operatorname{sgn}\omega\). Since \((i\omega)^{2k+1}=i(-1)^k\omega^{2k+1}\),

\[
|D_s|^r p=(-1)^k\frac{d^r}{ds^r}\mathcal Hp.
\tag{R6}
\]

The Hilbert transform involves all real projection values, so local analyticity of \(p\) alone is not the complete argument. Split its integral at \(-S,S\). For real \(|s|<S\), subtract \(p(s)\) from the inner numerator. Since

\[
\operatorname{PV}\int_{-S}^S\frac1{s-t}\,dt=\log\frac{S+s}{S-s},
\]

the following formula agrees with \(\mathcal Hp\) on \((-S,S)\):

\[
\boxed{
\begin{aligned}
\mathcal H p_O(z)=\frac1\pi\Bigg[&
\int_{-S}^S\frac{p_O(t)-p_O(z)}{z-t}\,dt
+p_O(z)\log\frac{S+z}{S-z}\\
&+\int_{[-R,-S]\cup[S,R]}\frac{p_O(t)}{z-t}\,dt\Bigg].
\end{aligned}
}
\tag{R7}
\]

This is also an explicit analytic continuation for \(z\) in a tube of radius \(2\eta\) around \([-T,T]\):

- The first integrand has a removable singularity at \(z=t\); it is a holomorphic difference quotient of the locally analytic projection.
- The logarithm has an analytic branch for \(|z|<S\).
- The outer integral stays separated from \(z\); it requires only real values of \(p_O(t)\).

To bound it, put \(Q_z=T+2\eta<S\). The straight segment between \(z\) and \(t\in[-S,S]\) lies within the radius-\(2\eta\) tube around \([-S,S]\). Cauchy's estimate from the larger radius-\(4\eta\) tube gives \(|p'_O|\le P/\eta\) along this segment. Therefore

\[
\left|\int_{-S}^S\frac{p_O(t)-p_O(z)}{z-t}\,dt\right|\le 2SP/\eta.
\]

The power series of \(\log[(S+z)/(S-z)]\) bounds its modulus by \(\log[(S+Q_z)/(S-Q_z)]\), and the outer integral is bounded by \(V/(S-Q_z)\). Consequently

\[
\boxed{
|\mathcal Hp_O(z)|\le J:=\frac1\pi\left[
\frac{2SP}{\eta}
+P\log\frac{S+Q_z}{S-Q_z}
+\frac{V}{S-Q_z}
\right].
}
\tag{R8}
\]

Cauchy's formula on a radius-\(\eta\) circle now bounds the derivative in (R6), for \(z\) in the radius-\(\eta\) tube around \([-T,T]\):

\[
\boxed{|g_O(z)|\le B_g:=C_d r!\eta^{-r}J\qquad(d\text{ even}).}
\tag{R9}
\]

This proves the same scalar tube statement in even dimensions. The use of a Hilbert transform is not an uncomputed oracle: (R7) is a finite-integral formula with a removable local singularity, and the required derivatives can be obtained by analytic differentiation or Cauchy integrals inside the certified neighborhood.

## 6. A quantitative angular analytic norm from the same target assumption

The previous sections give a holomorphic lift of \(g\) under complex rotations, jointly with its scalar argument. We now derive the angular certificate used by the sphere quadrature, without assuming it independently.

For real \(|x|\le1\), set

\[
H_x(\theta)=g_\theta(\theta\cdot x)-g_\theta(0).
\]

Define

\[
b=\min\{t_0,\log(1+\eta/2)\}>0,
\qquad K=\frac{d(d-1)}2.
\]

For products of complex rotations with \(\sum_j|z_j|<b\),

\[
|(O\theta)\cdot x-\theta\cdot x|\le\|O-I\|\le e^b-1\le\eta/2.
\]

Because \(\theta\cdot x\in[-1,1]\subset[-T,T]\), the scalar argument stays inside the certified scalar tube. Hence the holomorphic lift of \(H_x\) has the uniform bound

\[
|H_x(O\theta)|\le2B_g.
\tag{R10}
\]

Here the notation means the lift defined by the finite integral formulas; it agrees with the ordinary angular function for real rotations.

Let \(L_{ij}\) be the infinitesimal rotation field on the real sphere associated with \(J_{ij}\). Consider any word of \(k\) such fields. Realize it by differentiating a product of \(k\) independently parameterized rotations at zero. The polydisc \(|z_j|<b/(2k)\) lies in \(\sum|z_j|<b\). Applying the one-variable Cauchy estimate in each coordinate gives

\[
\boxed{
\|L_{i_1j_1}\cdots L_{i_kj_k}H_x\|_\infty
\le2B_g(2k/b)^k.
}
\tag{R11}
\]

The Laplace–Beltrami operator is \(\Delta_S=\sum_{i<j}L_{ij}^2\). Expanding its \(m\)-th power produces \(K^m\) words of length \(2m\). No commutation of the rotation fields is assumed. Since normalized sphere measure has mass one,

\[
\|\Delta_S^mH_x\|_2
\le\|\Delta_S^mH_x\|_\infty
\le2B_g K^m(4m/b)^{2m}.
\tag{R12}
\]

Let \(H_{x,\ell}\) be the degree-\(\ell\) spherical-harmonic projection. Its eigenvalue under \(-\Delta_S\) is \(\lambda_\ell=\ell(\ell+d-2)\ge\ell^2\). Orthogonal projection is a contraction in \(L^2\), and it commutes with \(\Delta_S\). For \(\ell\ge1\),

\[
\begin{aligned}
\|H_{x,\ell}\|_2
&\le\lambda_\ell^{-m}\|\Delta_S^mH_x\|_2\\
&\le2B_g\left(\frac{4m\sqrt K}{b\ell}\right)^{2m}.
\end{aligned}
\tag{R13}
\]

When \(b\ell/(4e\sqrt K)\ge1\), choose \(m=\lfloor b\ell/(4e\sqrt K)\rfloor\). The base in (R13) is at most \(1/e\). Since \(m\ge b\ell/(4e\sqrt K)-1\),

\[
\|H_{x,\ell}\|_2
\le2e^2B_g\exp\left[-\frac{b}{2e\sqrt K}\ell\right].
\]

For smaller \(\ell\), the same upper bound follows from \(\|H_{x,\ell}\|_2\le2B_g\): its right-hand side is at least \(2B_g\) in that range. It also covers \(\ell=0\).

Define

\[
a=\frac{b}{4e\sqrt K},
\qquad A=\frac{2e^2B_g}{\sqrt{1-e^{-2a}}}.
\]

Then a geometric series yields

\[
\boxed{
\sup_{|x|\le1}\sum_{\ell\ge0}e^{2a\ell}\|H_{x,\ell}\|_2^2
\le A^2.
}
\tag{R14}
\]

Indeed, the summand is bounded by \(4e^4B_g^2e^{-2a\ell}\). This proves the required angular analytic assumption with explicit constants derived entirely from the target-side \(R,\sigma,B\) and the chosen radial-window margins.

The factor \(\sqrt K\) in this conservative rate results from bounding all rotation words separately. Sharper complex-sphere harmonic estimates can improve it, but are not needed to establish the claimed logarithmic exponent.

## 7. Consequence for the finite QI–Radon theorem

The radial extension agrees with the target on the unit ball, and its canonical profiles satisfy both:

1. A uniform scalar analytic tube with \(\delta=\eta\), \(\Delta=T-1\), and bound \(B_g\), by (R5) or (R9).
2. The quantitative angular analytic certificate with \(a,A\) given in (R14).

Insert those constants into the existing positive angular quadrature and finite corrected-halo tanh theorem. The result is a genuine target-side sufficient condition for

\[
W=O_{d,R,\sigma,B}(\log^d(1/\varepsilon)),
\]

with every hidden constant obtainable from the displayed choices. This is fixed-dimension width scaling; the explicit bounds can be large as \(d\) increases or the analytic neighborhood shrinks. The proof does not claim these conservative constants describe practical best performance.

The final evaluator remains the same one-hidden-layer tanh network. Its coefficient construction evaluates known-target integrals over real balls, one real radius integral, and, in even dimensions, the finite Hilbert-continuation formula. These operations are calculable but may be expensive in high dimension. The width theorem does not assert that obtaining all coefficients costs only as much as evaluating the finished network.

The rational-target counterexample from the angular note remains correct for its unwindowed global extension. The present result shows why the extension choice matters: multiplying that target by the specified radial window changes its canonical directional profiles while preserving the target exactly on the unit ball. Their analytic regularity is then supplied by this proof.

## 8. Scope and audit points

- The target-side analyticity assumption is local to a larger real ball, not an assumed exponentially accurate angular approximation.
- The cutoff is smooth and radial, not analytic; analyticity is proved for the interior profiles using the ball-mixture formula.
- The even-dimensional nonlocal filter is handled explicitly, including its outer real tail.
- Derivatives of the cutoff do not enter the displayed interior analyticity bounds. Positivity and normalization of the ball-mixture density control its contribution.
- The holomorphic complex-rotation lift is used only to bound real angular derivatives; no arbitrary choice of global transverse coordinates is imposed.
- The extension has no target error on the requested unit ball. Numerical quadrature and analytic evaluation errors must still be bounded in the numerical part of the construction.
- Finite differentiability would require a separate quantitative derivative/Hilbert-transform estimate. This note proves the analytic corollary only and does not silently infer a finite-smoothness rate from it.
