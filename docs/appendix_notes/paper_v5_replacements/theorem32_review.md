# Theorem 3.2 appendix review

Independent mathematical review, 26 September 2026. Scope: the finite-kernel proof supporting Theorem 3.2 in paper version 5. The coordinator owns the replacement TeX. This report reviews the exact mathematical statements, not floating-point certification of the existing iteration-count example.

**Final status: accepted.** The coordinator applied the requested precision edits, and I verified them in the final replacement. No consequential mathematical correction remains within this review's scope.

## Source verification

Read `../2026-09-25-revised/02_finite_kernel_ratios.tex`, including its center-integral, Fourier, lattice, tail, eigenvalue, and normalization derivations, and `../../frozen_geometry_capacity_access/gamma_ratio_note/revision4_math_review.md`. Independently checked the constants and signs below against the derivations. No consequential defect found in these source formulas.

The supplied main-paper excerpt leaves the feature matrix normalization implicit. Defining the matrix entries as the feature values divided by the square root of the sample count, and minimizing half the squared Euclidean norm of the corresponding normalized residual, makes the theorem's stated step size and RMS-error interpretation consistent.

## Proof requirements and checked formulas

For sample count `m`, lattice spacing `h`, sample locations `x`, and an orthonormal zero-sum basis `Q`, the whole-line matrix is

\[
H_\gamma=-\frac{2}{hm}Q^\top[d_{ab}\coth(\gamma d_{ab})]Q,
\qquad d_{ab}=x_a-x_b,
\]

where the diagonal bracket is `1/gamma`. Its positive Fourier quadratic form has prefactor `2/(pi h m)`, multiplier

\[
M_\gamma(\omega)=\frac{\pi|\omega|/(2\gamma)}{\sinh(\pi|\omega|/(2\gamma))},
\]

and integrand `M_gamma(omega)^2 |sum_a (Qv)_a exp(-i omega x_a)|^2 / omega^2`. The zero-sum condition removes the apparent singularity at zero and both constant tails. It applies only to this compression; it imposes no zero-mean condition on actual finite-kernel eigenvectors.

For the product deficit `F_ab(c)=tanh(gamma(x_a-c))tanh(gamma(x_b-c))-1`, the transform is

\[
\widehat F_{ab}(\nu)=
-\frac{4M_\gamma(\nu)}{\nu}
\sin(\nu d_{ab}/2)\coth(\gamma d_{ab})
e^{-i\nu(x_a+x_b)/2}.
\]

Its diagonal limit is `-2 M_gamma(nu) exp(-i nu x_a) / gamma`. A paired lattice correction at `nu_k=2 pi k/h` is `2/(hm) Q^T Re[exp(i nu_k c_0) Fhat(nu_k)] Q`. Both the positive lattice phase and the factor of two are required.

With `a=2 pi theta/(gamma h)`, `0<theta<pi/2`, retaining paired lattice frequencies `1,...,p` leaves

\[
\delta_\gamma=
\frac{8\gamma\|x-\bar x\mathbf1\|^2\sec^4\vartheta}{3hm}
\frac{e^{-a(p+1)}}{1-e^{-a}}.
\]

This follows from the shifted derivative bound `||T'(.-i theta/gamma)||_2^2 <= (4 gamma/3) sec^4 theta`, zero-sum subtraction, and contour shifting of the analytic function `g(z)^2`. The contour does not apply to the nonanalytic function `|g(z)|^2`; its absolute value only bounds the shifted integral.

The retained absent-center set must include every omitted center within the sample interval. The remaining absent centers must lie in the two lattice tails starting at `c_R>x_max` and `c_L<x_min`. Saturated constant cancellation and `1-tanh(t)<=2 exp(-2t)` give

\[
0\preceq T_{\rm tail}\preceq\tau_\gamma I,
\qquad
\tau_\gamma=
\frac{4\{e^{-4\gamma(c_R-x_{\max})}+e^{-4\gamma(x_{\min}-c_L)}\}}
{1-e^{-4\gamma h}}.
\]

For each exterior center, the normalized projected outer product has norm at most `4 exp(-4 gamma distance)`. The saturated, unprojected constant part must not be bounded by this decaying expression.

Putting `S=H+sum_k G_k-T_retained` gives the exact relation

\[
Q^\top KQ=S+R-T_{\rm tail},\qquad
S-(\delta+\tau)I\preceq Q^\top KQ\preceq S+\delta I.
\]

The retained corrections can be large even when the unretained remainder bounds are small. In particular, `tau` does not enter the upper eigenvalue numerator.

Write `beta_j` for the descending eigenvalues of `S`, `kappa_j` for those of `Q^T K Q`, and `lambda_i` for those of `K`. Interlacing is `lambda_j >= kappa_j >= lambda_{j+1}`. Thus full-kernel upper bounds use `beta_{i-1}+delta`, while lower bounds use `[beta_i-delta-tau]_+`. The bottom eigenvalue has only the universal lower bound zero. The first ratio is exactly one. These endpoint conventions also cover `m=2`.

The normalization bounds are `ell=e^T K e>=1`, where `e=1/sqrt(m)`, and

\[
L=\frac{\ell+c+\sqrt{(\ell-c)^2+4b^2}}2,
\qquad c=\beta_1+\delta,\quad b=\|Q^\top Ke\|.
\]

The ratio lower bound divides the nonnegative lower numerator by `L`; the upper bound divides its upper numerator by `ell` and clips at one. Neither construction needs the actual small eigenvalues. Known structural zero modes may receive exact zero endpoints; numerically unresolved modes are not automatically structural zeros.

Finally, all target weights must come from the actual finite kernel. For zero initialization, the exact normalized squared error is `sum_i p_i (1-lambda_i/(2 lambda_1))^(2n)`. Upper ratio endpoints therefore give an error lower bound. A necessary threshold time is the first integer crossing of that lower curve, including the convention that an empty crossing set has infimum infinity.

## Gamma dependence and scope

The exact fixed-vector gain `M_(2 gamma)^2/M_gamma^2=cosh^2(pi |omega|/(4 gamma))` explains why different frequency contributions grow by different factors. It does not identify Fourier waves with finite eigenvectors, show that small eigenvalues always have high-frequency mass, or prove universal monotonicity of finite normalized ratios.

If local eigenvalue derivatives are retained, require simple positive eigenvalues and retain the mean/coupling, finite-lattice, exterior-center, and denominator derivatives when discussing actual finite ratios. An operator-norm remainder bound cannot be differentiated into a derivative bound. The pointwise finite-ratio interval needs none of this derivative machinery.

## Replacement draft

Reviewed the complete first draft of `theorem32_appendix.tex`. Its Fourier normalization, multiplier gain, product-deficit transform, paired Poisson terms, strip estimate, exterior-tail norm, enclosure signs, full-eigenvalue indices, scalar upper normalization, and residual/first-crossing directions are correct. The proof covers actual finite-kernel eigenvectors without imposing zero mean, includes endpoint and rank-deficiency cases, and does not assert universal monotonicity of finite normalized ratios.

The following small corrections were sent to the coordinator and are now verified as resolved:

1. The PDF text extraction suppresses the main theorem's overbar. The supplied manuscript uses the upper endpoint, and the replacement now explicitly identifies its upper endpoints as the main theorem's rate bounds. Exact ratios and upper bounds are distinguished.
2. The assumptions allow repeated sample locations. Interpret the center-integral bracket as `1/gamma` at every zero separation, not only on the diagonal.
3. The monotonicity statement for the scalar remainder allowances requires fixed strip angle as well as fixed retained counts and centers.
4. A truncated whole-line integral needs its positive tail allowance added to the upper eigenvalue numerators **and** the restricted-block upper bound used in the normalization `L`. The analogous empirical numerical allowance is added to these upper quantities and subtracted from lower numerators.
5. Specify that unresolved modes receive zero lower rates and retain all their target energy in the upper curve. Their energy may be dropped from the lower curve.

The numerical configuration, mixed-sine target, counts, refinement settings, and stated floating-point limitation agree with the revised source and saved record. The first draft correctly separates this width-153 example from the width-512 main experiment. The values are checked floating-point evaluations, not directed-rounding certificates or executed trillion-step training runs.

## Separate circuit-proof observation

The coordinator asked whether arbitrary error majorants satisfying recursive inequalities must be polynomial in the local tolerance. They need not be. Defining the majorants by recursive **equalities**, with fixed nonnegative constants and local tolerance `eta`, fixes the issue: induction produces nonnegative polynomials with zero constant term. For a fixed finite circuit and `0<=eta<=1`, its output polynomial satisfies `P_C(eta)<=eta P_C(1)`.

For a multiplication gate with exact input magnitude bounds `B_s,B_t`, the valid propagated error is `B_s E_t+B_t E_s+E_s E_t+eta`. The computed input magnitudes are bounded by `B_s+E_s` and `B_t+E_t`. Domain admissibility for approximate arguments remains a separate assumption unless established by a margin/closure argument. This observation reviews the stated algebraic issue only; it is not a full audit of the circuit appendix.

I subsequently read `circuit_error_appendix.tex`. Its recursive equalities, exact-value magnitude bounds, explicit domain-admissibility assumption, and definition `kappa_Phi=E_o(delta_max)/delta_max` are sufficient. The monotonicity of `E_o(delta)/delta` follows from its nonnegative polynomial coefficients and proves the stated error bound for every `0<delta<=delta_max`. The reported polynomial-majorant issue is resolved. No additional size/depth theorem was audited.
