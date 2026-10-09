# Angular resolution, exact moment coordinates, and the remaining aggregation theorem

Date: 2026-10-09. This note refines the angular-coupling analysis for fixed λ=γh, γ proportional to N. It keeps the original one-hidden-layer tanh MLP, all its readout directions, arbitrary physical constraint points, and general local nonlinear PDEs. Results about a smooth analysis weight below are explicitly auxiliary; they are not substitutions for a physical domain, boundary condition, or data norm.

## 1. The resolution variable is frequency times angle, not angle alone

For unit directions θ and θ′, let δ=||θ−θ′||, and consider one radial Fourier component. On a radius-R physical ball,

\[
|e^{i\omega\theta\cdot x}-e^{i\omega\theta'\cdot x}|
\le\min\{2,|\omega|R\delta\}.
\tag{1}
\]

Thus the angular resolution at radial frequency |ω| is approximately 1/(R|ω|). A ridge representation with many angular nodes contains substantial angular redundancy at its low radial frequencies, regardless of whether its finest radial scale is well matched to angular order.

For a localized profile with width a, the significant radial frequencies are of order 1/a, so the comparable angle is a/R. Under fixed λ and h proportional to 1/N, a proportional to 1/γ is proportional to 1/N. If a quasiuniform angular grid has separation and covering radius proportional to 1/q, q proportional to N matches this finest physical scale. At lower frequencies, the full angular grid is more highly resolved than needed for synthesis accuracy.

This does **not** prove that the original tanh columns become well conditioned. A tanh step has a nonlocal constant tail and L² mass of order one. Rotating a sharp step by δ can change it mainly in a wedge of volume of order δ. Its normalized L² difference can therefore vanish when δ is proportional to 1/N even though γδ stays of order one. The frequency-angle argument applies most directly to localized, normalized radial combinations with coarse/moment components separated and retained.

For completeness, differentiating a single tanh ridge through order r gives a pointwise pair difference bounded by

\[
\gamma^r\left(r\|\psi^{(r)}\|_\infty\delta
+\gamma R\|\psi^{(r+1)}\|_\infty\delta\right).
\]

This is only a continuity bound for columns, not a lower frame bound or solver conditioning theorem.

## 2. The actual angular product rule has severe polar clustering

The repo's angular rule recursively writes

\[
\theta=(\cos\phi,\;\sin\phi\,\vartheta),\qquad\vartheta\in S^{d-2},
\]

uses q Gauss–Jacobi nodes in t=cosφ with weight (1−t²)^((d−3)/2), and terminates with 2q equally spaced circle angles. Antipodal folding yields M=q^(d−1) retained directions. This is a positive quadrature rule of the stated polynomial order. Polynomial exactness and total node count do not imply quasiuniform physical angular separation.

For each fixed dimension, the first polar angle satisfies φ_min proportional to 1/q; this is the endpoint Bessel asymptotic for Jacobi zeros. [NIST DLMF, Jacobi zeros](https://dlmf.nist.gov/18.16#ii)

In d=3, two neighboring terminal-circle angles on the first polar ring have physical chord distance

\[
\delta=2\sin\phi_{\min}\sin(\pi/(2q))=\Theta(q^{-2}).
\tag{2}
\]

Their γδ tends to zero when γ is proportional to q. Moreover, the entire first ring contains 2q directions inside a cap of radius O(1/q). Antipodal folding changes constants, not this clustering.

In general fixed d, choosing the first node at every polar recursion gives neighboring terminal-circle points with separation

\[
\delta=2\left(\prod_{k=3}^d\sin\phi_{k,\min}\right)
\sin(\pi/(2q))=\Theta_d(q^{-(d-1)}).
\tag{3}
\]

This exhibits pairs at this separation; it is not asserted to be a sharp lower bound for every pair in the rule. Under γ proportional to q, their γδ is Θ_d(q^(−(d−2))). The top polar cap alone contains Θ(q^(d−2)) directions inside angular radius O_d(1/q).

The normalized quadrature weights near the poles are correspondingly small. In a k-dimensional recursion the first Gauss–Jacobi weight scales as q^(−(k−1)) for fixed k; multiplying the endpoint weights along the full recursion and the circle weight gives q^(−d(d−1)/2) for the extreme chain, versus average weight q^(−(d−1)). This follows from the endpoint Jacobi polynomial/derivative asymptotics and the Gauss weight formula. [NIST DLMF, Gauss–Jacobi quadrature](https://dlmf.nist.gov/3.5#v)

Small quadrature weights can stabilize integration of controlled profiles. They do not remove near-dependence of normalized free-readout columns: allowing the readout to compensate a small weight restores the original column. Likewise, changing coefficient scaling is not a proof of a physical lower stability bound.

Consequently matched q and N invalidate neither all nearby-direction concerns nor all prospects for fast structure. They identify a specific transform opportunity: physically unresolved polar clusters should be handled by aggregate moments and their retained angular details, rather than independent identical-cost direction blocks.

All Θ_d assertions fix dimension while q grows. Their constants contain Jacobi/Bessel parameter dependence and must not be advertised as dimension-uniform.

## 3. Exact radial changes of coordinates can expose vanishing moments without deleting modes

Let a finite radial combination be

\[
g(t)=\sum_{j\in B}v_j\tanh(\gamma(t-c_j)).
\]

If

\[
\sum_{j\in B}v_jc_j^k=0,\qquad k=0,\ldots,p,
\tag{4}
\]

then g decays exponentially at both ends and has p vanishing moments,

\[
\int_{\mathbb R}t^k g(t)\,dt=0,\qquad k=0,\ldots,p-1.
\tag{5}
\]

To see this, the coefficient exponential sum S(ω)=Σv_j e^(−iωc_j) is O(ω^(p+1)) at zero, while

\[
\widehat g(\omega)=\widehat\kappa_\gamma(\omega)S(\omega)/(i\omega),
\quad\kappa_\gamma=\gamma\operatorname{sech}^2(\gamma t),
\]

extends smoothly with a zero of order p. Its derivatives at zero give (5). For p=0 the sole condition Σv_j=0 makes a localized bump, not a zero-integral bump. A second finite difference is the first simple zero-integral example.

For a center block B, form the finite moment matrix V_{kj}=c_j^k, k=0,…,p. An orthogonal factorization splits coefficient space into its row span and nullspace. This is an exact invertible coordinate change. The nullspace columns generate (5); the at most p+1 complementary moment channels retain the remaining functions. Applying this hierarchically produces a multiscale center transform while retaining all details, the coarsest channels, and every endpoint center. It does not assume periodic physical boundaries or delete halo corrections.

Orthogonalization should use scaled center coordinates, avoiding an unnecessarily ill-conditioned raw Vandermonde matrix. Orthogonality in coefficient space alone does not establish physical frame conditioning. No minimum-norm selection or penalty is implied by the change of coordinates.

## 4. A dimension-explicit overlap estimate, with its exact limitations

This section supplies an auxiliary localization theorem. It explains why moment coordinates help, but it is not by itself a PDE preconditioner theorem.

Let φ be a real exponentially decaying profile with p vanishing moments and finite μ_p=∫|s|^p|φ(s)|ds. Let

\[
f_{\theta,b,a}(x)=a^{-1/2}\phi((\theta\cdot x-b)/a).
\]

Let χ_R(x)=χ(x/R), where χ is a smooth compactly supported weight. For directions making angle δ with 0<δ<π, coordinates u=θ·x and v=θ′·x have Jacobian factor 1/|sinδ| after integrating the remaining d−2 coordinates. Thus

\[
\langle f_{\theta,b,a},f_{\theta',b',a}\rangle_{\chi_R}
=\frac{a^{-1}}{|\sin\delta|}
\int\phi((u-b)/a)\phi((v-b')/a)W(u,v)\,du\,dv,
\]

where W is the cross-sectional integral of χ_R. Its mixed derivative satisfies

\[
\|\partial_u^p\partial_v^p W\|_\infty
\le C_{d,p,\chi}R^{d-2-2p}|\sin\delta|^{-2p}.
\]

Taylor expansion in u and v through degree p−1, with each polynomial term killed by the corresponding moments, gives

\[
|\langle f_{\theta,b,a},f_{\theta',b',a}\rangle_{\chi_R}|
\le\frac{C_{d,p,\chi}\mu_p^2}{(p!)^2}
R^{d-1}\left(\frac{a}{R|\sin\delta|}\right)^{2p+1}.
\tag{6}
\]

Combine this with the trivial O(R^(d−1)) diagonal-size bound near parallel/antiparallel directions. If the profiles' centers stay in a region where their diagonal norms are bounded below by cR^(d−1), (6) also bounds normalized correlation with the corresponding factor 1/c. Profiles grazing a boundary do not automatically have this lower bound.

For a quasiuniform angular grid with spacing Δ comparable to a/R, the number of directions in the j-th angular ring is bounded by a dimension-dependent constant times (j+1)^(d−2). Therefore the absolute angular overlap sum is bounded independently of angular refinement when

\[
2p+1>d-1.
\tag{7}
\]

This gives a possible upper frame/Schur bound under the stated hypotheses. It does not give a lower frame bound: off-diagonal interactions might still cancel, and the constants need not imply diagonal dominance. The required moment order grows at least linearly with dimension in this elementary argument, and the constants are explicit dependencies rather than harmless universals.

Two important restrictions prevent substituting (6) directly into the final solver:

1. The actual polar product grid is not quasiuniform; its clusters must first be addressed with retained angular moment/detail channels.
2. A sharp domain indicator, arbitrary scattered data measure, or rapidly varying nonlinear Jacobian coefficient need not have the smooth cross-sectional derivative bounds used above. Replacing it by χ_R would change the task. The theorem can support interior localization or a component of an exact partition with separately handled boundaries; it cannot silently replace the physical norm.

Classical ridgelet frames use precisely a radial-scale-dependent angular resolution. Their established continuous/discrete frame results provide relevant guidance, but their atoms or frame spaces do not automatically coincide with this finite tanh span. [Candès and Donoho, ridgelet review](https://candes.su.domains/publications/downloads/RoySoc.pdf) [Donoho, orthonormal ridgelets](https://doi.org/10.1137/S0036141098344403)

## 5. Constructive local angular moments with a certified derivative error

The plane-wave reduction in `aggregation.md` supplies sources ξ=ωθ, with arbitrary transformed amplitudes. Work first at one fixed ω. Group directions into a cap centered at θ_0 with ||θ−θ_0||≤δ. For |x|≤R,

\[
\sum_{m\in B}a_m e^{i\omega\theta_m\cdot x}
=e^{i\omega\theta_0\cdot x}
\sum_{|\beta|<p}\frac{(i\omega x)^\beta}{\beta!}
\underbrace{\sum_{m\in B}a_m(\theta_m-\theta_0)^\beta}_{\text{angular moment}}
+\mathcal E_B(x).
\tag{8}
\]

For real phases the scalar Taylor remainder has magnitude at most |z|^p/p!, so

\[
\|\mathcal E_B\|_\infty
\le\frac{(|\omega|R\delta)^p}{p!}\sum_{m\in B}|a_m|.
\tag{9}
\]

The number of ambient monomials is

\[
Q(d,p)=\binom{d+p-1}{p-1}.
\tag{10}
\]

For a derivative α of order r, apply the same expansion after multiplying each source amplitude by (iωθ_m)^α. The error is bounded by

\[
|\omega|^r\frac{(|\omega|R\delta)^p}{p!}\sum_{m\in B}|a_m|.
\tag{11}
\]

Channel-specific moments retain rank Q(d,p); a common angular moment basis through degree p+r−1 covers all derivative multipliers and has at most Q(d,p+r) entries. The same bound controls arbitrary physical points in the ball, including boundary, initial, data, and validation points. No smooth sampling density is assumed.

A weighted orthogonal angular transform can expose these moments while preserving all coefficients. If a_m=√α_m z_m with positive angular quadrature weights α_m, factor the matrix with rows √α_m(θ_m−θ_0)^β. The complement consists of angular detail coefficients annihilating all these moments. Retain that complement. An approximation may defer its contribution at a coarse target scale only using (9)–(11); sufficiently fine target scales must resolve it or maintain a certified bound. Dropping all details as a permanent prior is not the proposed transformation.

This exact moment/detail split is especially relevant to the Θ(q^(d−2)) nodes in the first polar cap. At ω proportional to q, its phase radius |ω|Rδ is O_d(1), although the cap contains increasingly many nodes. Its local synthesis admits a rank bounded by Q(d,p) independent of q for fixed d,p. Thus product-grid clustering is a genuine source of compressibility, not only an ill-conditioning objection.

At the precision-required cutoff Ω proportional to γχ, the phase radius can be O(χ), so p must grow accordingly. The binomial factor (10), coefficient cancellation, and requested derivative tolerance must all remain in the cost. This is a concrete local rank theorem, not yet a global near-linear algorithm.

## 6. Source/target complementarity avoids pretending every point visits every cap

A single global angular cap partition with radius 1/(RΩ) still requires K times the number of caps. To improve the whole operator, introduce target clusters as well.

Let a target cluster have center x_0 and radius a_x. Let a source cluster contain frequencies with |ω−ρ|≤Δρ and ||θ−θ_0||≤δ. Then

\[
\|\omega\theta-\rho\theta_0\|\le\Delta\rho+\rho\delta.
\]

Factor the kernel as

\[
e^{ix\cdot\xi}
=e^{ix\cdot\xi_0}e^{ix_0\cdot(\xi-\xi_0)}
 e^{i(x-x_0)\cdot(\xi-\xi_0)},\qquad\xi_0=\rho\theta_0.
\]

The last factor admits a rank-Q(d,p) expansion with entrywise error at most

\[
\frac{\eta^p}{p!},\qquad
\eta=a_x(\Delta\rho+\rho\delta).
\tag{12}
\]

Derivative channels have the additional source multiplier bound (ρ+Δρ)^r. This is the complementary-resolution condition needed by a butterfly-style factorization: fine targets permit larger angular/frequency groups; coarse targets use smaller groups. Arbitrary target locations are compatible with the local theorem, because only their enclosing radii enter it.

The missing global step is not the Taylor expansion. It is a complete nested construction proving that the sum of retained interaction ranks, transfer costs, and numerical errors across all source/target tree levels is bounded by the desired budget for the actual angular product rule and any allowed target set. It must not hide a p^d Cartesian volume, a large tree branching constant, K M interactions in setup, or O(M) outer iterations.

The dimension-explicit local rank (10) is polynomial in d for fixed p, but p itself depends on precision, derivative order, and block phase radius. This is a more informative starting bound than calling the rank a constant. Whether structured transfers can apply these moment bases with less cost than dense Q² is also part of the constructive question.

## 7. Why general nonlinear PDEs need more than angularly local Gram matrices

Fast synthesis and its adjoint survive arbitrary samplewise nonlinear Jacobian multipliers. Angular near-diagonality of the resulting normal operator generally does not.

For plane-wave channels ξ and η, a weighted Gram entry contains

\[
\int_\Omega a(x)e^{i(\eta-\xi)\cdot x}\,dx.
\]

Even a bounded positive coefficient a(x)=1+εcos((η−ξ)·x), 0<ε<1, can induce a substantial interaction between widely separated frequency directions. On a compatible periodic box the cosine contribution is exactly ε times half the box volume for distinct nonzero Fourier modes. This example diagnoses frequency coupling; no periodic-domain restriction is proposed for the actual method. On general domains the same shifted spectral interaction remains.

Therefore neither vanishing radial moments nor angular separation establishes a uniform banded Gram matrix for all nonlinear states. Navier–Stokes transport, coefficient inverse fields, and general nonlinear constitutive terms can supply physically meaningful mixing. A method that discards these interactions would sacrifice the intended PDE capability.

Equations (8)–(12), in contrast, compress only the shared synthesis kernel. They apply the full nonlinear multiplier at physical samples and then the full compressed adjoint, preserving the interaction to certified error. Preconditioning and nonlinear convergence must be handled separately from this operator-compression theorem.

## 8. Next constructive target

The defensible next target is an exact multiscale coefficient organization plus a certified approximate operator, in this order:

1. Orthogonal radial moment/detail transforms on actual center blocks, retaining coarse and endpoint channels.
2. Frequency-dependent angular moment/detail transforms on actual geometric caps, including polar clusters, retaining all transformed coordinates.
3. A nested source/target factorization using (12), with explicit Q(d,p), tree counts, derivative errors and coefficient norm dependence.
4. A stability analysis in the original physical PDE/data norm, including boundary pieces and nonlinear multipliers; the smooth-window overlap estimate is an ingredient only where its hypotheses hold.

This keeps the final model a standard one-hidden-layer MLP because every coordinate operation maps back to its original readouts. It supplies two concrete new theoretical facts: the actual quadrature's polar clustering persists under matched refinement, and those growing clusters have bounded local synthesis rank at matched physical frequencies. What remains open is converting this local fact into the full requested near-linear-in-M solve complexity without changing the allowed physical problem.
