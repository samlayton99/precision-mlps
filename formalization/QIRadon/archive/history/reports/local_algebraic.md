# Local finite-smooth construction from real samples

Completed 2026-10-09. New modules:

- `QIRadon/ProfilePreparation/FilteredSamples.lean`
- `QIRadon/LocalAlgebraic.lean`
- `QIRadon/LocalAlgebraicScaling.lean`
- `QIRadon/LocalAlgebraicConstruction.lean`

## The regularity class and what the construction actually reads

The target profile f is real-valued on [-1,1] and bounded by M. Its local polynomial approximation class is specified by fixed A≥0 and s>0: for each integer N≥1, there exists a polynomial of degree less than N with uniform approximation error at most A N^(-s).

That comparator polynomial is a proof witness, never an input to the construction. The construction reads exactly **2N real Chebyshev-node values of f**. No complex evaluation, global Fourier moment, derivative of f, or fitted polynomial oracle is required. `filteredSamples_congr` and `localAlgebraicNetwork_congr` prove that matching those samples produces exactly the same computed polynomial and activation row.

## Stable filter: the missing step now proved

For each positive integer m, define the finite Chebyshev Fejér kernel

\[
K_m(x,y)=\frac1m\sum_{a,b=0}^{m-1}T_{a-b}(x)T_{a-b}(y).
\]

Integer-index Chebyshev polynomials satisfy T_(-k)=T_k. Substituting x=cos(t), y=cos(u) expands the double sum into four real squares. This proves K_m≥0 on [-1,1]^2 without relying on an absent Fejér library.

For L≥m Chebyshev nodes y_j, define

\[
F_{m,L}f(x)=\frac1L\sum_{j=0}^{L-1}f(y_j)K_m(x,y_j).
\]

The actual discrete Chebyshev orthogonality proves that the sample kernel has mass one. Positivity therefore gives ||F_{m,L}f||∞≤||f|| on the sampled nodes. Exact finite diagonal counts prove the multiplier

\[
F_{m,L}T_k=(1-k/m)T_k,\qquad k<m.
\]

The actual sample filter is

\[
P_N=2F_{2N,2N}f-F_{N,2N}f.
\]

It has norm at most 3, degree at most 2N, and reproduces every polynomial of degree less than N. Consequently

\[
\|f-P_N\|_\infty\le4A N^{-s}.
\]

This preserves the full s exponent. The earlier unfiltered interpolation bound carried a factor N; that loss is not used here.

Principal declarations: `fejerKernel_nonneg`, `fejerKernel_sample_mass`, `fejerSamples_bound`, `fejerSamples_T`, `filteredSamples_bound`, `filteredSamples_reproduces`, `filteredSamples_degree_le`, `filteredSamples_error_of_comparator`, and `filteredSamples_algebraic_error`.

## Gaussian-window preparation and the actual activation row

Use the existing exact Gaussian-window/Fourier preparation at order 2N on the **computed polynomial** P_N. `filtered_reinterpolation` proves that the existing preparation's sample-polynomial step reproduces P_N exactly. Those evaluations are evaluations of the known finite polynomial, not new samples of f.

The prepared profile is exactly P_N(x) times the existing finite Gaussian flat window. Its Fourier-moment bound is the already proved explicit exponential-in-order bound, now using the uniform polynomial bound 3M. All complex contour steps concern this entire Gaussian-polynomial preparation, never the original nonanalytic f.

`local_algebraic_error_terms` proves the actual finite activation row error

\[
4A N^{-s}
+3M e^{-2N c_{\rm win}}
+\frac{C_{\rm moment}(3M,T_0,r)C_{\rm kernel}}{r!}e^{-4N}.
\]

The two exponential terms are converted to N^(-s) with the explicit elementary constant

\[
D(b,s)=\frac{\lceil s\rceil!}{b^{\lceil s\rceil}},
\qquad e^{-bN}\le D(b,s)N^{-s}\quad(N\ge1).
\]

Thus the row satisfies error≤C_alg N^(-s), where the exact fixed constant in the code is

\[
C_{\rm alg}=4A+3M D(2c_{\rm win},s)
+\frac{C_{\rm moment}(3M,T_0,r)C_{\rm kernel}}{r!}D(4,s).
\]

This bound is conservative, particularly when the window margin is small. It is explicit and introduces no new target regularity assumption.

`local_algebraic_accuracy_and_width` chooses the actual integer N from max(1,C_alg), s, and epsilon, proves uniform scalar accuracy, and counts the actual grid at preparation order 2N, including all halos. `local_algebraic_grid_epsilon_bigO` gives

\[
H=O(\epsilon^{-\kappa/s}),\qquad
\kappa=\max(1,1/\mu+1/\nu).
\]

A separate theorem includes any fixed, explicitly realized anchor-unit count. Generic r≥2 still requires its declared anchor realization; no arbitrary activation is assumed to realize polynomials exactly.

## Centered multidimensional construction

`LocalAlgebraicConstruction.local_centered_uniform_accuracy_and_width` combines these rows with the actual positive folded Gauss–Jacobi rule. Scalar polynomial approximation constants are uniform over directions. Angular assumptions are the actual canonical Sobolev norm of the **centered** function

`theta -> g_theta(theta·x) - g_theta(0)`,

uniformly for all x in the target ball, with fixed constants before the network is chosen. No origin-profile angular regularity is required.

If the ambient dimension is d=k+2 and the angular Sobolev exponent is s_ang, the proved angular uniform exponent is

`t = s_ang - (d-1)/2 > 0`.

The network is one fixed finite sum followed by the output-constant adjustment

`Q_centered(x)=f0+Q_full(x)-Q_full(0)`.

It exactly hits f0. Its error bound is the actual angular power-law error plus twice the actual scalar row error. The construction allocates epsilon/2 angularly and epsilon/4 to the uncentered scalar rows. Positive max-with-one constants handle zero error constants without changing asymptotics.

The exact all-halo tensor cardinality satisfies

\[
W=O\!\left(\epsilon^{-[(d-1)/t+\kappa/s]}\right).
\]

This is deliberately distinguished from the stronger global polynomial-Fourier class's exponent `(d-1)/t+1/s`. They agree when kappa=1, including Gaussian/erf/exact GELU.

`centeredTarget_canonical_on_ball` proves exact equality to the actual canonical Radon target from profile agreement only on |s|≤1. `local_centered_gelu_is_network` proves the entire centered construction is exactly an ordinary single-hidden-layer standard-GELU MLP with **two global anchor neurons total**, plus every direction/center/halo neuron. It uses the already proved architecture identity; no fitting or extra polynomial evaluator remains.

## Scope and verification

The theorem assumes the stated local polynomial approximation class and actual post-Radon angular Sobolev class. It does not derive those classes from a generic finite-smooth multidimensional original target; the approved map permits explicit post-Radon regularity assumptions. These are fixed-d asymptotics with dimension- and regularity-dependent constants.

The filter and all preparation steps are finite explicit operations/integrals. The formal kernel definition uses finite multiple sums and is not claimed to be an optimized implementation. Working-bit and total arithmetic-cost guarantees remain the separate T8 obligation. Neuron/sample counts here are not a claim of optimal numerical runtime.

Build and axiom audit: all four modules are built with the pinned Lean 4.33.0 toolchain. Principal declarations are inspected in `scratch/AuditLocalAlgebraic.lean`. No existing shared source module was edited.

Build succeeded (8778-job dependency closure). All 17 audited principal declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`. No proof holes, imported `sorryAx`, or project axioms were introduced.
