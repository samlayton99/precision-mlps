# General-kernel QI–Radon theorem map — awaiting approval

Date: 2026-10-08. This is the proposed scope before Lean execution. No Lean development is authorized by this document or has been started. The general derivative-kernel construction is primary. Tanh's special rational halo correction is a corollary, not an assumption of the main theory.

## Conventions and dependencies

The target domain is the unit ball in dimension d>=2. M is the number of retained directions. N_m counts interior centers on spoke m, R_m counts halo centers on each side, and H_m=N_m+2R_m counts its grid neurons. W=sum_m H_m+W_anchor counts all neurons, including any units needed to represent integration polynomials. In the uniform case W=M H+W_anchor. A dense scalar-output MLP has (d+2)W+1 scalar parameters.

For activation phi, choose derivative order r and nonzero mass m so phi^(r)=mK and integral K=1. The order r is not the number of directions or a halo length. The primary smooth-kernel theorem assumes K is Schwartz; ReLU's delta kernel is handled by a separate spline theorem.

Dependency order:

1. Radon reconstruction and prescribed domain extension produce scalar profiles and angular regularity.
2. A general finite derivative-kernel certificate identifies cutoff, alias, and halo errors.
3. Kernel decay properties and a local-profile preparation lemma give scalar rates.
4. Angular quadrature and positive error transport give the multidimensional laws.
5. Resource allocation, activation examples/obstructions, numerical stability, and sample-based selection are consequences.

## T1. Radon reconstruction with a specified finite-domain extension

**Assumptions.** A known target f is bounded and holomorphic in a complex neighborhood of a real ball of radius R>1. Its evaluations have a specified error model. Choose a nonnegative normalized smooth radius density w supported in [R0,R], with 1<R0<R. For finite-regularity versions replace holomorphy by explicit regularity or post-Radon regularity bounds; do not infer a rate from mere nonanalyticity.

**Claim.** The radial extension F(x)=f(x) integral w(a) 1_{|x|<=a} da agrees exactly with f on the target ball. Its canonical profiles satisfy

\[
F(x)=\int_{S^{d-1}}g_\theta(\theta\cdot x)\,d\mu(\theta),\qquad
\widehat g_\theta(\omega)=C_d|\omega|^{d-1}\widehat F(\omega\theta).
\]

The analytic version gives explicit local scalar and angular analytic bounds from the supplied target domain and magnitude bounds. It gives finite formulas to evaluate the profiles and their local complex continuations.

**Proof steps.** Prove the Fourier slice identity by an orthonormal change of variables; derive the reconstruction filter by polar Fourier inversion. Express the radial window as a positive mixture of balls and obtain the finite cross-section formula. Use analyticity of f to continue the interior scalar coordinate and rotations. In odd dimensions differentiate the projection; in even dimensions first use the finite Hilbert formula with its local singularity removed. Convert complex-rotation derivative bounds into an angular harmonic-tail bound.

**Boundary scope.** The compact extension is not globally analytic. Never evaluate its full Fourier integral at a complex scalar argument without a separate justification. The local continuation formulas in `radial_extension_certificate.md` are the valid route.

## T2. General finite-kernel construction and its exact error sources

**Assumptions.** phi^(r)=mK, m!=0, integral K=1; K is real and Schwartz; p=g^(r) has an absolutely integrable Fourier representation. On the chosen frequency band [-B/gamma,B/gamma], Khat is nonzero. Supply the anchor derivatives through order r-1 and an explicit realization of their polynomial in the selected activation, or charge its approximation error and neurons separately.

**Construction.** Define
\[
\rho_B(c)=\frac1{2\pi}\int_{-B}^{B}
\frac{\widehat p(\omega)}{\widehat K(\omega/\gamma)}e^{i\omega c}\,d\omega.
\]
Retain centers jh with |jh|<=T+D, use readouts h rho_B(jh)/(m gamma^(r-1)), and add the prescribed integration constants.

**Claim.** For the actual finite network,
\[
\|g-Q\|_{\infty,[-T,T]}
\le H_r(T)(E_{\rm cut}+E_{\rm alias}+E_{\rm halo})+E_{\rm anchor},
\quad H_0=1,\ H_r=T^r/r!\ (r>0).
\]
The three explicit quantities are
\[
E_{\rm cut}=\frac1{2\pi}\int_{|\omega|>B}|\widehat p(\omega)|\,d\omega,
\]
\[
E_{\rm alias}=\frac1{2\pi}\int_{-B}^{B}|\widehat p(\omega)|
\sum_{k\ne0}\left|\frac{\widehat K((\omega+2\pi k/h)/\gamma)}{\widehat K(\omega/\gamma)}\right|d\omega,
\]
\[
E_{\rm halo}=h\|\rho_B\|_\infty
\sup_{|s|\le T}\sum_{|jh|>T+D}|K_\gamma(s-jh)|.
\]

**Proof steps.** Differentiate r times; invert convolution only on the retained band; derive all aliases by Poisson summation; bound the omitted grid centers; integrate the derivative error r times using the anchors. This theorem assumes no convergence rate and no boundary correction oracle.

## L1. Explicit preparation of a locally analytic profile

**Assumptions.** g is bounded and analytic on a supplied Bernstein ellipse surrounding [-T0,T0], where T0>T. Real evaluations are available with controlled error.

**Claim.** From L+1 Chebyshev samples construct a polynomial P_L of degree L. Define
\[
w_L(z)=e^{-Lz^2/T_0^2}\sum_{k=0}^{L}\frac{(Lz^2/T_0^2)^k}{k!},
\qquad G_L=P_Lw_L.
\]
Then ||g-G_L|| on [-T,T] is at most C exp(-cL), and for any fixed 0<sigma<delta and fixed r,
\[
\frac1{2\pi}\int e^{\sigma|\omega|}|\widehat{G_L^{(r)}}(\omega)|\,d\omega
\le \operatorname{poly}(L)e^{C_\delta L}.
\]
G_L is a Gaussian times a finite polynomial, so its transform is explicitly evaluable. It is only an intermediate used to calculate network coefficients; it is absent from the final evaluator.

**Proof steps.** Use the Chebyshev ellipse bound. On the target interval, bound 1-w_L by a Poisson tail with exponent c_win=u-1-log(u)>0, u=(T/T0)^2. Bound P_L off the real interval and the finite exponential series, retaining Gaussian decay to obtain shifted-line L1 bounds poly(L)exp(CL). Shift the Fourier contour and integrate |omega|^r exp(-(delta-sigma)|omega|). Thus log of the new Fourier-moment constant is O(L), preserving the later logarithmic rate.

**Finite-smooth variant.** A real-sample filtered polynomial with degree O(L) and error O(L^-s) gives the same global preparation at algebraic approximation error. Its cost must be propagated: the generic scalar width becomes O(epsilon^(-kappa/s)), not automatically O(epsilon^(-1/s)) when kappa>1.

## T3. Kernel-property analytic rate theorem

**Assumptions.** T2, and explicit positive constants satisfying
\[
c_0e^{-A_0|\xi|^\nu}\le|\widehat K(\xi)|\le C_0e^{-a_0|\xi|^\nu},
\qquad |K(u)|\le C_1e^{-b_0|u|^\mu},\qquad \mu,\nu>0.
\]
The target derivative has an exponential Fourier moment; alternatively use local analyticity and L1, carrying its accuracy-dependent moment bound explicitly. The integration polynomial has an admissible finite-network realization.

**Claim.** Let
\[
\kappa=\max\{1,1/\nu+1/\mu\}.
\]
An explicit finite construction reaches error epsilon with H=O(log^kappa(C/epsilon)) grid neurons, including halos. In the matched fast-decay class kappa=1, this is geometric scalar convergence. For kappa>1 it is a stretched-exponential sufficient bound, not an algebraic one and not a lower bound on every construction.

**Proof steps.** Let L be proportional to log(C/epsilon), choose cutoff B proportional to L, gamma=B/L^(1/nu), spacing h proportional to 1/B, and physical halo D proportional to L^(1/mu)/gamma. Spectral lower bounds control the density amplification exp(O(L)); spectral upper bounds suppress every alias by exp(-cL); the spatial envelope suppresses the finite tail enough to overcome that amplification. Count H=O(B+BD)=O(L+L^(1/nu+1/mu)). Under L1, log of the prepared-profile norm grows only O(L), so the same power remains.

## T4. Kernel-property algebraic rate theorem

**Assumptions.** The same kernel bounds, with a global weighted Fourier moment
\[
A_s=\frac1{2\pi}\int(1+|\omega|)^s|\widehat{g^{(r)}}(\omega)|\,d\omega<\infty,
\qquad s>0,
\]
or a local polynomial approximation bound with L1's finite-smooth preparation. All directional constants must be uniform for a uniform-width Radon theorem.

**Claim.** The global moment class gives H=O(epsilon^(-1/s)); the local polynomial class gives H=O(epsilon^(-kappa/s)) using this particular generic preparation. For kappa=1 they agree. These are sufficient rates; stronger boundary constructions can improve specific activations.

**Proof steps.** In the global class choose B proportional to epsilon^(-1/s) and L proportional to log(C/epsilon). The cutoff error is O(B^-s), aliases and halos remain exponentially small in L, and the count is O(B+log^(1/nu+1/mu)(1/epsilon)). For the local class, choose polynomial order L proportional to epsilon^(-1/s), track the exp(O(L)) Fourier-moment bound, and apply T3 with that cost. Do not insert a nonanalytic target into a complex-shift formula.

## T5. Positive angular quadrature and error transport

**Assumptions.** A valid Radon decomposition; continuous angular integrand H_x(theta)=gtheta(theta.x)-gtheta(0); quantitative analytic harmonic norm or finite Sobolev regularity; scalar approximants satisfying their certified errors. Use the recursive positive Gauss–Jacobi rule.

**Claim.** The full rule is exact through degree 2q-1; antipodal folding is valid for the even Radon integrand and leaves M=q^(d-1). Angular errors are bounded by U exp(-a q) or U q^-t under the respective regularity. Scalar errors enter as their weighted average, not multiplied by M:
\[
\|f-Q\|_\infty\le E_{\rm ang}+\sum_m\alpha_m\widetilde E_m+E_{\rm num}.
\]
Here the tilde means the already-anchored scalar error; if only an unanchored uniform error is supplied, use twice that value.

**Proof steps.** Prove product-rule polynomial exactness by induction; retain positivity and mass one. Subtract an exactly integrated spherical polynomial, giving quadrature error at most twice its uniform approximation error. Bound the harmonic tail from the relevant norm. Apply the triangle inequality with positive weights to the finite scalar rows.

## T6. Full multidimensional scaling laws — the headline approximation results

**Assumptions.** T1 and L1 when using local analytic targets, a kernel satisfying T3, the angular certificate T5, bounded anchor-neuron overhead, and a numerical error budget. For finite-regularity results use the explicitly proved scalar exponent s_rad and angular exponent t.

**Claim.** Analytic angular and scalar construction give
\[
W=O\bigl(\log^{d-1+\kappa}(C/\varepsilon)\bigr).
\]
In the matched class kappa=1,
\[
\boxed{W=O(\log^d(C/\varepsilon)).}
\]
With angular error O(q^-t) and scalar error O(H^-s_rad),
\[
\boxed{W=O(\varepsilon^{-[(d-1)/t+1/s_{\rm rad}]}).}
\]
If t=s_rad=s, the latter is O(epsilon^(-d/s)). Constants expose dimension, target frequency/regularity bounds, kernel envelopes, boundary margins, and arithmetic tolerance. Actual finite integer bounds precede Big-O corollaries.

**Proof steps.** Divide tolerance among preparation, angular, scalar, anchor, and numerical errors. Solve each proved inequality for its finite resolution. Multiply the direction count by the actual row width, then add anchor units. Derive the asymptotic expression only afterward. This is fixed-d asymptotics, not a d-independent complexity claim.

## T7. Optimal allocation within the proved certificate

**Assumptions.** Positive certified constants and actual width accounting. In the analytic family use
\[
E\le Ue^{-aM^{1/(d-1)}}+Ve^{-bH^{1/\kappa}},\qquad W=MH.
\]
For the algebraic family use the corresponding power-law bounds. Treat the continuous optimization first, then explicit rounding and minimum-width conditions.

**Claim.** The analytic interior optimum with budgets epsilon_a+epsilon_s=epsilon_app obeys
\[
\boxed{\kappa\varepsilon_a\log(U/\varepsilon_a)
=(d-1)\varepsilon_s\log(V/\varepsilon_s).}
\]
At high accuracy epsilon_a/epsilon_s tends to (d-1)/kappa. The matched class recovers the earlier split and a M^(1/(d-1)) approximately b H. The algebraic optimum obeys epsilon_a/epsilon_s=(d-1)s_rad/t. Also prove the per-spoke allocation and weighted-error variant, rather than forcing identical rows.

**Proof steps.** Express M and H in terms of their allocated errors. Differentiate the logarithm of total width; establish the minimizer in the stated range; handle admissibility constraints and rounding. This is optimality of the certificate, not a global neural-network lower bound.

## Activation corollaries and obstruction theorem

**Assumptions.** The specific activation definition and the target class of the invoked parent theorem.

**Claims.** Gaussian derivative kernels (including Gaussian bumps, Gaussian CDF/erf, exact GELU) satisfy mu=nu=2 and hence kappa=1. The generic tail-truncation theorem gives tanh mu=nu=1, kappa=2; the independently proved tanh boundary correction improves its rate to kappa=1 as a separate corollary. ReLU is handled by splines, not by pretending its delta kernel is a Schwartz function: it has constructive algebraic approximation and the lower bound
\[
\inf_{\operatorname{width}(Q)\le W}\|x^2-Q\|_{[-1,1]}
\ge1/[2(W+1)^2].
\]
Fixed-degree polynomial activations do not approximate general targets at all.

**Proof steps.** Calculate derivative kernels and verify their envelopes and anchor identities. For ReLU restrict to a line, find its longest affine interval, and bound affine approximation of x² there; pair with the explicit spline construction. Failure of a sufficient envelope condition is not itself an impossibility result. Spectral zeros obstruct division on that used band, not every possible choice of bandwidth.

## T8. Numerical realizability and perturbation stability

**Assumptions.** Certified profile/coefficient evaluations, explicit bounds on kernel inversion and finite-sum weights, known activation and derivative bounds over the used preactivation ranges, and accounted anchor realization.

**Claim.** Give tolerances on target samples, quadrature, weights, directions, and activation evaluation that imply E_num<=epsilon_num. State required working bits and finite computation costs. No fixed-float64 claim is substituted for arbitrary-epsilon accuracy.

**Proof steps.** Bound the density and total readout magnitude using kernel inequalities; propagate coefficient and hidden-parameter perturbations through activation Lipschitz bounds; include the output constant and summation. For L1, count the Gaussian-polynomial preparation and evaluate its transform by explicit finite formulas or certified quadrature. Profile acquisition from f remains a counted cost, not an assumed O(W) oracle.

## C_data. Selecting the split from samples — last priority

**Assumptions.** A finite set of candidates with recorded actual widths and an explicit data-based candidate-building procedure. The selection result itself does not recover Radon profiles from arbitrary samples.

**Practical procedure.** Compare a modest grid of angular orders and row widths, including separate angular and scalar refinements. Measure validation error versus added total neurons. Choose the smallest candidate meeting the error certificate. Adaptive search uses tuning data followed by fresh certification data.

**Uniform-error claim.** With sample covering radius delta_X, noise bounded by eta, known target Lipschitz bound L_f, and calculated candidate bound L_j,
\[
\|f-Q_j\|_\infty\le
\max_i|y_i-Q_j(x_i)|+\eta+(L_f+L_j)\delta_X=:U_j.
\]
The least-width candidate with U_j<=epsilon has certified uniform accuracy and is no larger than any tested candidate whose true error clears the same coverage/noise margin.

**Statistical alternative.** With independent validation, bounded noiseless targets/predictions, and a fixed finite candidate family, use a simultaneous Bernstein upper confidence bound for every candidate's squared L2 error. Minimum-width selection gives a near-best-among-tested-candidates guarantee, with an explicit sample/confidence margin. This avoids estimating theoretical strip widths from data. It is not a supremum-norm guarantee.

**Proof steps.** The uniform version follows by moving from an arbitrary point to its nearest sample and applying Lipschitz/noise bounds. The statistical version uses bounded squared loss, Bernstein concentration, a union bound, and minimum-width selection. A short impossibility lemma shows why finitely many unqualified samples cannot certify uniform error: an unsampled narrow bump can be arbitrarily large.

## Approval boundary

The main theory is now stated in terms of kernel invertibility, spectral suppression, spatial localization, target regularity, and explicit boundary preparation. Tanh's special identities are optional stronger corollaries. The new local-profile preparation lemma is a proposed formal proof obligation, not an already machine-checked result. Approval is requested only for subsequent formalization; no Lean files or proof stubs have been created.
