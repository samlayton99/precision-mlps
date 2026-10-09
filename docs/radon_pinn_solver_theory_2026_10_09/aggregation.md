# Sample–direction aggregation: a certified transform and its unresolved dimensional cost

Date: 2026-10-09. Scope: evaluation and adjoint evaluation of the actual fixed-geometry tanh network at arbitrary scattered points. This is not a complete nonlinear solver or a dimension-uniform complexity theorem.

## Result

There is an admissible, explicit reduction of the complete derivative-bank operator to (i) independent one-dimensional coefficient transforms and (ii) a single nonuniform Fourier synthesis on polar frequencies. It does not restrict readouts or assume the unknown solution is bandlimited. Its approximation error is certified against the same finite tanh network, including derivatives. Consequently local nonlinearities, scattered boundary/data/initial constraints, and inverse fields can retain this factorization.

A standard type-3 NUFFT removes the explicit sample-count times direction-count product. However, its auxiliary grid and spreading constants introduce dimensional costs that cannot be omitted. For fixed dimension and sufficiently populated angular geometry, this gives an honest conditional near-linear-width operator application. It does **not** presently establish Sam's desired approximately O(M N²) cost with increasing dimension. A recent tensor-butterfly paper does not close that gap either: its published cost contains rank to the power 2d, and its rigorous rank argument does not cover the arbitrary nonuniform geometry required here.

These limitations identify an open transform theorem, not an impossibility theorem.

## 1. The actual network and the required operator

Let

\[
q_w(x)=b+\sum_{m=1}^{M}\sum_{j=1}^{N}w_{mj}\tanh(\gamma(\theta_m\cdot x-c_j)),\qquad \|\theta_m\|_2=1.
\]

Assume the physical samples lie in the radius-R ball (translation of the origin is allowed), and |c_j|≤C. Centers may be nonuniform. Uniform centers permit ordinary FFTs in one stage; nonuniform centers require a one-dimensional nonuniform coefficient transform instead.

Let P=MN, let K denote all sampled locations, and let H denote the number of scalar derivative channels the PDE/data declaration actually needs. Pointwise nonlinear arithmetic has a separate declared cost. A fixed-order PDE can still have H growing with dimension, e.g. a fully uncontracted Hessian has O(d²) channels; this cannot be hidden as a universal constant.

The operating schedule is approximately fixed λ=γh. With a fixed center range and h proportional to 1/N, γ is proportional to N. This schedule is used throughout the scaling estimates below. The latest representation theorem also discusses a different schedule; we do not substitute it for Sam's requested regime.

## 2. Periodize the derivative, not tanh

Directly periodizing tanh would introduce an artificial discontinuity. Instead use

\[
\kappa_\gamma(t)=\gamma\operatorname{sech}^2(\gamma t)=\frac{d}{dt}\tanh(\gamma t).
\]

With Fourier convention \(\widehat f(\omega)=\int f(t)e^{-i\omega t}\,dt\), direct integration gives

\[
\widehat\kappa_\gamma(\omega)
=\frac{\pi\omega/\gamma}{\sinh(\pi\omega/(2\gamma))},\qquad
\widehat\kappa_\gamma(0)=2.
\]

Choose T>2U with U=R+C. Define the periodic sum

\[
\kappa_T(t)=\sum_{p\in\mathbb Z}\kappa_\gamma(t+pT)
=\sum_{\ell\in\mathbb Z}\beta_\ell e^{i\omega_\ell t},\quad
\omega_\ell=2\pi\ell/T,\quad
\beta_\ell=T^{-1}\widehat\kappa_\gamma(\omega_\ell).
\]

This is a convergent identity for real t. Retain |ℓ|≤L to obtain κ_{T,L}.

Here are finite, computable uniform error bounds, avoiding an unquantified use of spectral convergence. Define polynomials

\[
P_0(z)=1,\qquad P_{s+1}(z)=(1-z^2)P_s'(z)-2zP_s(z),\qquad
D_s=\max_{|z|\le1}|P_s(z)|.
\]

Then

\[
\kappa_\gamma^{(s)}(t)
=\gamma^{s+1}\operatorname{sech}^2(\gamma t)P_s(\tanh(\gamma t)),
\quad
|\kappa_\gamma^{(s)}(t)|\le4D_s\gamma^{s+1}e^{-2\gamma|t|}.
\]

Summing image tails and omitted Fourier modes gives, for r≥1,

\[
E_r(T,L)=
\frac{8D_{r-1}\gamma^r e^{-2\gamma(T-U)}}{1-e^{-2\gamma T}}
+\frac2T\sum_{\ell>L}\omega_\ell^{r-1}\widehat\kappa_\gamma(\omega_\ell),
\tag{1}
\]

and

\[
\sup_{|t|\le U}
|\kappa_\gamma^{(r-1)}(t)-\kappa_{T,L}^{(r-1)}(t)|\le E_r(T,L).
\]

The infinite positive tail in (1) is bounded by elementary exponential sums or incomplete-gamma integrals; it is not a numerical cancellation problem. Indeed \(\widehat\kappa_\gamma(\omega)\) decays as a linear polynomial times \(e^{-\pi|\omega|/(2\gamma)}\). One can choose T and L directly from the demanded derivative tolerances.

For each neuron define its approximation by an exact anchor and an integrated derivative,

\[
\widetilde\psi_j(t)=\tanh(-\gamma c_j)
+\int_0^t\kappa_{T,L}(u-c_j)\,du.
\]

For |t|≤R this has

\[
|\widetilde\psi_j(t)-\tanh(\gamma(t-c_j))|\le R E_1,
\quad
|\widetilde\psi_j^{(r)}(t)-\partial_t^r\tanh(\gamma(t-c_j))|\le E_r.
\tag{2}
\]

Thus for any real readouts, with B_w=Σ_{m,j}|w_{mj}|,

\[
\|q_w-\widetilde q_w\|_\infty\le B_w R E_1,
\qquad
\|\partial^\alpha q_w-\partial^\alpha\widetilde q_w\|_\infty\le B_w E_{|\alpha|}.
\tag{3}
\]

The direction factors have absolute value at most one. No bound on B_w is imposed on the model; rather the required approximation precision depends on B_w, or on a declared operator norm suitable for the solver. Treating B_w as one without justification would be an inadmissible hidden stability assumption.

Equation (3) is an approximation of the existing network, not a prior on the solution. It introduces no new physical boundary condition: the periodic kernel and its affine integral are internal evaluation devices on a bounded interval.

## 3. The explicit plane-wave factorization

Define independent radial transforms

\[
S_{m\ell}=\sum_j w_{mj}e^{-i\omega_\ell c_j},\quad
 a_{m\ell}=\frac{\beta_\ell}{i\omega_\ell}S_{m\ell}\quad(\ell\ne0),\quad
\xi_{m\ell}=\omega_\ell\theta_m.
\]

Integrating the zero Fourier mode separately gives

\[
\widetilde q_w(x)=q_w(0)
+\frac2T x\cdot\sum_m S_{m0}\theta_m
+\sum_{m,\,0<|\ell|\le L}a_{m\ell}
\left(e^{i\xi_{m\ell}\cdot x}-1\right).
\tag{4}
\]

The anchor q_w(0) is computable in O(P) work. The affine term costs O(Md) to construct and O(Kd) to evaluate. Its gradient is constant and its higher derivatives vanish. The remaining derivative banks are

\[
\partial^\alpha\sum a_{m\ell}e^{i\xi_{m\ell}\cdot x}
=\sum (i\xi_{m\ell})^\alpha a_{m\ell}e^{i\xi_{m\ell}\cdot x}.
\tag{5}
\]

Thus all derivative channels share the same Fourier synthesis matrix

\[
F_{k,(m,\ell)}=e^{i\xi_{m\ell}\cdot x_k}.
\]

The original Φ^(r)D structure has become the same shared synthesis F with frequency-diagonal derivative factors, preceded by block-diagonal one-dimensional transforms. Anchors add only low-rank terms.

Forward and adjoint use the same factorization in opposite order. For real readouts take the real-adjoint of the conjugate-paired complex representation; do not confuse complex conjugate transpose with ordinary transpose. An implementation should transpose the actual approximation factors, ensuring exact adjoint consistency up to arithmetic. Error bounds follow from (2) entrywise; sharper weighted 2-norm bounds would be necessary for a final solver theorem.

Uniform c_j=c_0+jh and T chosen as an integer multiple of h make S_{mℓ} samples of a padded ordinary DFT. The one-dimensional cost is approximately O(M(N+L)log(N+L)). Uneven centers do not destroy (4), but replace this by the corresponding certified one-dimensional NUFFT cost.

## 4. Nonlinearity and useful PINN constraints survive

For a local nonlinear residual R(w)=F_local(x,{A_αw}_α), its linearization is

\[
J(w)v=\sum_\alpha\operatorname{diag}(\partial F_{local}/\partial u_\alpha)A_\alpha v.
\]

The diagonal pointwise factors are applied after synthesis; their nonsmoothness in the sample index does not require a new Fourier expansion. The adjoint multiplies by those factors before applying A_α^*. Consequently variable coefficients and arbitrary local nonlinear coupling do not produce M² frequency interactions. No nonlinear convolution is formed.

Boundary and initial point rows simply use another list of x_k and the appropriate derivative contractions. Normal derivatives add known normal components. Inverse coefficient fields require their own synthesis and derivative channels; finitely many unknown physical parameters add ordinary dense columns whose cost must be counted. No low-rank restriction on these fields has been introduced.

The total number of sampled points remains K. Random/adaptive validation at new points requires a new target geometry or transform setup, whose cost belongs in the ledger. Geometry-dependent plans can be reused only while samples remain fixed.

## 5. Full cost of standard Fourier aggregation

The number of polar frequencies is S=2ML. A type-3 NUFFT evaluates their exponential sum at arbitrary x_k. Its cost depends on the enclosing physical and frequency boxes, not only S+K. The official FINUFFT mathematical description explicitly warns about this space–bandwidth dependence; its implemented dimensions are one through three. [FINUFFT mathematical definitions](https://finufft.readthedocs.io/en/latest/math.html)

A representative tensor-spreading/FFT ledger is

\[
T_{\rm aggregate}
=O\left((K+S)p^d+G\log G+d(K+S)\right),\qquad
G=\prod_{i=1}^d O(1+R_i\Omega_i+p),
\tag{6}
\]

where p is the per-coordinate spreading width required by the transform tolerance, R_i and Ω_i are physical and frequency coordinate extents, and constants include the chosen oversampling factor. This is a method-class estimate, not a guarantee that the current FINUFFT implementation supports arbitrary d. Memory includes O(G+K+S). Multiply appropriate parts by H, although common setup and transforms can be batched. One-time frequency storage itself is O(dS) unless generated implicitly.

Let χ denote the logarithmic accuracy factor needed in the Fourier cutoff, including B_w, derivative scaling γ^r, and the chosen absolute tolerance. Then Ω=O(γχ), L=O(Tγχ). At fixed center/physical ranges and fixed λ, γ=O(N), so L=O(Nχ), S=O(MNχ), and a generic box estimate is G=O((cNχ)^d).

The repo's explicit angular quadrature has M=q^(d−1); see [angular rule](../radon_scaling_theory_2026_10_08/theory.md#3-angular-quadrature-and-an-explicit-analytic-certificate). Therefore the grid-to-width ratio behaves schematically as

\[
\frac{G}{MN}
=O\left((c\chi)^d\left(\frac Nq\right)^{d-1}\right).
\tag{7}
\]

When d and χ are fixed and q is proportional to N, G is proportional to the existing representation width, with a dimension-dependent constant. If K is also O(MN), this is a genuine near-linear-width evaluator for that regime. It is not merely an O(M) statement at fixed K. For a fixed absolute derivative tolerance, χ generally grows logarithmically with N because γ grows; the resulting logarithmic factors must also be retained, raised to the dimension in the Cartesian-grid estimate.

But the explicit factor in (7), and p^d in (6), can be prohibitive as dimension grows. Even a fixed factor of two per coordinate becomes 2^d. If radial resolution N greatly exceeds angular order q, filling a Cartesian frequency box can be much more expensive than retaining the polar representation. This is precisely the situation an improved aggregation theorem must avoid.

To claim an O(MN²) application from (6), one must actually verify, among other terms,

\[
(K+2ML)p^d+G\log G=O(MN^2).
\tag{8}
\]

For K proportional to MN and L proportional to Nχ, sufficient conditions include χp^d=O(N) and GlogG=O(MN²). These are explicit feasibility conditions, not automatically valid facts about the construction.

Fast aggregation is still not a solve theorem: setup, number of linearized iterations, preconditioner, nonlinear globalization and independent residual certification remain additional costs.

The bounds in (1)–(3) concern exact arithmetic. A machine-precision result also needs a backward-error analysis of the coefficient transforms, phase evaluation, NUFFT spreading/deconvolution, and final accumulation. In particular, perturbing a physical coordinate by δx changes a plane-wave phase by ξ·δx, which can be as large as Ω||δx||. A requested transform tolerance equal to machine epsilon is not itself a theorem that physical fields or high derivatives have machine-epsilon error. Cancellation and input conditioning must be accounted for in the same original physical norm.

## 6. Can butterfly methods avoid the auxiliary Cartesian grid?

For a frequency block centered at ξ_0 and a physical block centered at x_0,

\[
e^{ix\cdot\xi}
=e^{ix\cdot\xi_0}e^{ix_0\cdot(\xi-\xi_0)}
 e^{i(x-x_0)\cdot(\xi-\xi_0)}.
\]

The final factor is slowly varying when the block diameters have small product. Expanding its exponential through total degree p−1 gives a separated representation with at most

\[
\binom{d+p-1}{p-1}
\]

terms and scalar remainder bounded by e^ηη^p/p! when |(x−x_0)·(ξ−ξ_0)|≤η. This is an elementary dimension-explicit origin of complementary low rank. Tensor interpolation instead produces p^d terms. A multilevel butterfly reuses such local representations rather than claiming the full operator is globally low rank.

The original butterfly FIO paper proves a fast algorithm for fixed two-dimensional Cartesian discretizations. Its result motivates this route but does not directly prove the arbitrary-scattered, dimension-uniform claim here. [Candès, Demanet and Ying, 2009](https://arxiv.org/abs/0809.0719)

A newer tensor-butterfly method addresses higher-dimensional Fourier/Radon operators. Its explicit published bounds are O(d n^d r_t^(2d+1)) setup, O(n^d r_t^(2d)) memory and application per vector. It proves a favorable rank statement for uniform Fourier structure; for more general Radon/nonuniform cases it relies on numerical evidence. Its nonuniform Fourier experiments use tolerance 10^-3. Thus neither dimension-uniform overhead nor machine-epsilon accuracy for our polar-to-arbitrary-point map follows from that paper. This is a valuable research lead, not the missing theorem. [Kielstra et al., 2025, §§3.2.1–3.2.2 and 4.3](https://arxiv.org/html/2411.03029v3)

The source directions already have a product angular parameterization, but the target locations are arbitrary. Merely reshaping a scattered point list into a tensor does not create coordinate separability. We cannot replace those points with a tensor grid without changing the allowed use case. Auxiliary grids are admissible computationally only with their full memory, interpolation errors and dimensional costs included.

## 7. Precise open theorem and what is not being claimed

The missing aggregation theorem can be stated without choosing a PDE solver:

> For the actual fixed angular rule, all center readouts, K arbitrary physical targets, and derivatives through order r, construct a data-independent factorization of the synthesis map and its adjoint, with certified physical derivative error and total setup/application cost bounded by poly(d,r,log(1/ε)) times (K+MN) times at most an additional factor N. The theorem must count dependence on coefficient cancellation and must not fill a substantially larger tensor volume or assume low-rank unknown profiles.

The one-dimensional reduction above supplies the input to this theorem: a polar Fourier synthesis with S=O(MNχ) sources and known derivative multipliers. It does not solve the theorem.

No theorem in this report proves Ω(KM) lower bounds for all possible algorithms. In particular, the failure of a tensor NUFFT bound to fit (8) does not establish impossibility. Conversely, announcing near-linear complexity while suppressing p^d, G, or the iteration count would not establish feasibility.

Promising next admissible work is a dimension-explicit block-rank and tree-count analysis specialized to the existing product angular rule and radial frequency sets, retaining arbitrary target lists. If that requires a restricted target distribution, a low-rank solution class, a different physical norm or weak constraints, it is a scope decision for Sam; none has been adopted here.

## Verification and status

All derivations are analytical; no benchmark optimization or PDE success case was selected. The derivative polynomial recurrence, image-tail geometric sum, anchored integration and Fourier derivative factors were checked algebraically. Numerical transform stability and high-dimensional butterfly rank bounds remain unproved. No claim of a complete method, machine-epsilon implementation, or width-uniform solve convergence is made.
