# A theory framework for scalable Radon PINN inference

The objective is a useful nonlinear PINN method whose numerical overhead grows roughly linearly with the number of directions M, while retaining the existing finite tanh representation, data and inverse constraints, irregular samples, and a machine-epsilon-scale physical accuracy target. The desired cost is of the order of M N², with N centers per direction, rather than treating all M N unknowns as an unstructured dense problem. This investigation establishes several exact transformations and conditional stability results. It does not yet establish that complete complexity theorem.

No PDE benchmark was tuned in this investigation. No low-degree function family, coefficient penalty, domain simplification, discarded nonzero mode or external PDE solution was adopted. All mathematical statements below are analytical derivations, not newly Lean-verified theorems. The existing Lean work proves known-target construction and approximation results; it does not establish unknown-PDE inference complexity.

Sam's clarified hard requirements are a standard one-hidden-layer MLP and the ability to resolve realistic general PDE solutions, including difficult Navier–Stokes flows. Tanh is the initial activation, with later extension to other supported activations. Internal transforms and residual products must not become extra learned layers or product gates in the returned model. Preserving every finite readout combination is the conservative branch studied here; it is not a substitute for judging whether the method retains the physically important solution classes and scales.

## Current continuation results

The continuation adds an exact nonlinear angular-rank statement, a full Navier–Stokes operator derivation, a resolution-aware angular-cluster analysis and a conditional native-neural stability-transfer theorem. These strengthen the framework but do not complete the overall O(M N²) solve bound.

- [Angular coefficient theorem](angular_coefficients.md): for an arbitrary differentiable local order-r PDE, the point–direction coefficient modulation of its Jacobian is a degree-r polynomial in direction. Its separation rank is bounded by the number of degree-r monomials, independent of M. This is not a low-rank assumption on the field or complete Jacobian. Nonlinear coefficients can change while a certified factorization of the geometric derivative banks is reused.
- [Navier–Stokes compatibility](navier_stokes_compatibility.md): exact residual/Jacobian/transpose formulas retain nonlinear advection, pressure, continuity, boundary/initial/data rows and inverse viscosity. With spacetime direction (a_m,τ_m), advection/time uses τ_m+u·a_m, and spatial diffusion uses |a_m|². The former has exact point–direction rank at most d+1. No low-Reynolds-number or simple-flow approximation is made. Variable-viscosity stress and the time-dependent pressure gauge are handled explicitly.
- [Angular resolution](angular_resolution.md): the actual product angular rule has polar clusters even when radial and angular resolution are matched. A growing cluster can have bounded local synthesis rank at matched physical frequencies. Exact moment/detail coordinates retain every coefficient; a certified source–target expansion exploits the cluster without deleting a solution feature. The missing result is a complete nested factorization whose total rank, tree and precision costs meet the target for scattered samples.
- [Operator stability transfer](operator_transfer.md): if a reference right preconditioner gives singular bounds [μ,ν] and the preconditioned actual neural operator differs by η<μ in the relevant derivative/constraint norm, the actual neural bounds lie in [μ−η,ν+η]. The nonlinear version includes both encoding defects and the change in equation sensitivities. All residuals remain evaluated on the actual network; no external solution is simply encoded. A fast general-domain reference transform remains to be supplied.

No current result proves that a turbulent/high-Reynolds-number solve will converge cheaply or reach machine precision. The established result is that those nonlinear equations do not algebraically destroy the shared derivative-bank structure; the global aggregation, physical stability and approximation certificates remain separate proof obligations.

## 1. What has been established

### Every profile has exact positive local coordinates

For one profile g(t)=b+Σ_j w_j tanh(γ(t−c_j)), define

\[
B_0(t)=\frac{1-\tanh(\gamma(t-c_1))}{2},\qquad
B_j(t)=\frac{\tanh(\gamma(t-c_j))-\tanh(\gamma(t-c_{j+1}))}{2},\qquad
B_N(t)=\frac{1+\tanh(\gamma(t-c_N))}{2}.
\]

Every B_j is nonnegative and their sum is exactly one. The coordinate change

\[
a_0=b-\sum_jw_j,\qquad a_j=a_0+2\sum_{k\le j}w_k
\]

gives g=Σ_j a_j B_j, with inverse w_j=(a_j−a_{j−1})/2 and b=(a_0+a_N)/2. Every original readout is retained. A multi-direction implementation keeps one global bias and handles only the redundant allocation of constants among profiles as a gauge; this does not remove genuine neuron functions.

Consequently ||g||∞≤||a||∞, independent of N. The interior basis functions are integrals of the normalized logistic derivative over center cells. This is a partition-of-unity interpretation of the original neural network, not replacement by a spline model. Data values and derivative constraints now share radial locality. Stable hyperbolic identities avoid subtracting rounded saturated tanh values.

### Radial conditioning can be separated from width

For uniform spacing and fixed λ=γh, the full-line Gram matrix of the interior basis is Toeplitz. Fourier analysis gives a positive explicit constant c_λ such that, for every finitely supported sequence a,

\[
h c_\lambda\|a\|_2^2\le
\left\|\sum_j a_j B((\cdot)/h-j)\right\|_{L^2(\mathbb R)}^2
\le h\|a\|_2^2.
\]

The constant is independent of row length. It can nevertheless be small: at λ=0.25, the derived lower bound is about 4.52×10⁻¹⁵, and high-frequency alternating coefficient packets show substantial attenuation. This is an admissible route to structured radial normalization, not a proof of well-conditioned physical-domain inference. Finite endpoints, angular coupling and the physical PDE norm still matter.

### All local nonlinear equations share one linear derivative map

Stack the required field values and derivatives at all constraint locations into U. There is a fixed linear map A such that U=Aa. A generic local nonlinear PDE/data system is R(a,p)=F(Aa,p), where p denotes finite-dimensional unknown physical parameters; unknown fields get their own columns and derivative channels.

The Jacobian is exactly J=C A, with C block diagonal over physical sample locations. Each small block contains derivatives of the local equation with respect to its requested field values and derivatives. Boundary/initial/data rows can use different sample lists and derivative contractions. Nonlocal observations must retain their actual operators; pointwise locality cannot be assumed for arbitrary covariance or integral measurements.

Introducing U as a variable with the exact constraint U=Aa is an equivalent lifted problem. It separates local nonlinearity from a global linear consistency operator. It does not automatically decouple directions: eliminating U still couples all directional profiles, and storing separate per-direction jets costs K M times the number of channels.

The interior part of a single-direction linearized Gram block is approximately banded after certified localization; anchors give a small border. The tail bound for derivative order r has the form C_r γ^r exp(−2T) times a cumulative coefficient amplitude, without an extra N factor. At fixed λ and fixed absolute derivative tolerance, T generally contains a log N contribution. Full formulas, proofs and costs are in [local structure](local_structure.md).

### Angular coupling sometimes has an exact transform

For an actual rotationally invariant physical norm and a uniform full-circle direction grid in two dimensions, the exact physical Gram matrix is block-circulant in direction. Angular Fourier transformation reduces it to M radial N-by-N blocks. No nonzero angular mode is discarded. After setup, the physical-coordinate map costs O(M N²+M N log M) per application.

This is an exact example of the requested dependence on M, not merely a claim that the Gram is approximately sparse. Its generic radial factorization setup costs O(M N³), plus certified acquisition of M N² Gram entries. It is a special-case result, not an adopted restriction to disks. On a different physical domain, using a larger-ball reference norm need not preserve uniform conditioning. The report gives an explicit tanh counterexample with exponentially deteriorating norm equivalence as N grows.

### There is a certified reduction of all derivative banks to polar Fourier synthesis

Periodize the decaying derivative γ sech²(γt), truncate its Fourier series with explicit tail bounds, and integrate using the exact network value at an anchor. Every finite readout vector then has a certified plane-wave approximation

\[
\widetilde q(x)=q(0)+\ell\cdot x+
\sum_{m,k}a_{mk}\bigl(e^{i\omega_k\theta_m\cdot x}-1\bigr).
\]

The approximation error is bounded in physical derivatives and depends explicitly on the original readout absolute sum. This is an approximation of the same network, not a bandlimited prior on the unknown solution. All derivatives become diagonal frequency factors applied to one synthesis operator. The transpose uses the same factorization in reverse; nonlinear coefficients remain local physical-point multipliers.

This identifies precisely the aggregation problem that could remove the sample-count times direction-count cost. Standard nonuniform FFT methods do remove explicit K M interactions, but their auxiliary Cartesian grid and spreading constants can grow exponentially with dimension beyond the representation cost. A recent high-dimensional tensor-butterfly paper retains powers r_t^(2d) in its costs, and its nonuniform examples do not establish machine-precision polar-to-arbitrary-point performance. These costs were not accepted as meeting the target. Derivations and primary sources are in [aggregation](aggregation.md).

## 2. A precise stability theorem for evaluating candidate methods

Let S map the independent finite readouts to their physical field, and let X be the physical function norm relevant to the declared PDE. Let L_q be the full linearization, including boundary, data and gauges. Suppose

\[
a\|v\|_X\le\|L_qv\|_Y\le b\|v\|_X
\]

on the finite network space, and the sampled residual norm is equivalent to Y with constants c_s,C_s. If a computable coordinate transform P satisfies

\[
\alpha I\preceq P^*S^*SP\preceq\beta I,
\]

then the transformed sampled Jacobian has condition number bounded by

\[
\frac ba\sqrt{\frac{C_s\beta}{c_s\alpha}}.
\]

This separates physical ill-conditioning, insufficient samples, and coordinate conditioning. A fast method must bound these constants as M,N grow and count the setup/application cost of P. Dense physical Gram whitening establishes existence, but fails the complexity requirement. The same theorem applies locally to a nonlinear system; global convergence requires additional hypotheses and cannot be inferred from a fast Jacobian action. The derivation and counterexamples are in [stability](stability.md).

## 3. What the exploration ruled out

- Cheap direction-wise blocks alone do not imply cheap global inference. Two nearly parallel directions have a small but nonzero physical difference; the normalized difference approaches a genuine transverse derivative. Independent block scaling cannot remove that correlation.
- The continuous Radon representation's redundancy does not usually create a large exact redundancy in a fixed finite tanh bank. Apart from duplicates and antipodal identities, distinct fixed-slope finite tanh features are linearly independent on an open domain. A nontrivial projection onto fewer coordinates therefore generally changes its function space.
- A small absolute approximation error for a Gram kernel is insufficient to justify whitening its tiny eigenvalues. A relative norm guarantee or a controlled physical error bound is needed.
- Sampling cannot be hidden. A locally stable discrete inverse with P independent unknowns needs at least P independent scalar constraints unless further information is supplied. Direct synthesis at K≈MN costs approximately M²N times stencil/channel factors.
- A declared transform tolerance is not an end-to-end machine-epsilon theorem. Phase errors, cancellation, derivative scaling, nonlinear sensitivity, physical inverse conditioning and validation must enter one error budget.

These are obstructions to particular arguments, not an impossibility proof for the desired method.

## 4. The unresolved admissibility decision

The unrestricted finite-network branch remains open: it needs a fast physical-norm coordinate transform and dimension-explicit aggregation that retain every nonzero mode. None of the routes examined so far proves the complete general O(M N²) result.

A different possible branch would use coordinates defined by the construction's underlying regularity class, with a proof that all original target functions retain the same value and derivative approximation guarantees. That could exclude finite readout combinations outside that class. It is materially different from an invertible reparameterization, even when it preserves approximation rates, and has not been adopted.

Sam clarified that the admissibility criterion is retaining realistic solutions of useful PDEs, not retaining artificial readout configurations for their own sake. The abstract choice between coefficient families did not identify a meaningful physical tradeoff and has been withdrawn. Before adopting a restriction, its consequences for thin layers, oscillations, nonlinear regimes and inverse recovery must be established and explained. No polynomial degree, low-rank hypothesis or smoothness prior has been chosen. Restrictions on the actual finite network's complex analyticity are especially dangerous: its tanh poles approach the real axis as γ grows. Regularity of an underlying construction target cannot simply be imposed on its finite neural encoding.

## 5. Next work allowed without changing the problem

The exact-coordinate branch has two concrete targets: a multilevel angular/radial transform with a relative physical-norm stability bound, and a polar Fourier aggregation algorithm whose precision and dimension dependence fit the full cost budget. The local partition-of-unity basis and nonlinear factorization can be used in either. The symmetry result provides a verified analytical test case, but any extension must retain the real physical domain, sample geometry and all finite-network functions.

No end-to-end success is claimed at this stage. The work completed here turns the research question into explicit operator-factorization and stability statements, with several exact supporting results and a documented decision boundary.
