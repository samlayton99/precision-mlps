# Angular construction: formal status

The actual finite sphere quadrature, positivity, exactness, antipodal folding,
and analytic/algebraic angular convergence from spectral regularity are proved.
The target-side implication from local analytic/smooth function assumptions to
the required expansion and energy remains a separate T1 obligation.

No new axiom, sorry, or admit is used. Standard Lean foundations are
propext, Classical.choice, and Quot.sound. All listed modules compile against
the pinned mathlib. A local heartbeat increase in HarmonicQuadrature changes only
elaboration resources, not its mathematical assumptions.

## Concrete Gaussian/Jacobi rule

- Gaussian.lean defines integrated Lagrange weights and proves degree<2q exactness,
  positivity and normalization from orthogonality and positive moments.
- OrthogonalPolynomial.lean constructs the monic orthogonal polynomial by the
  finite moment Gram form and proves uniqueness.
- OrthogonalRoots.lean proves every degree-many root is real, simple and interior
  using a constant-sign factorization and positivity contradiction.
- Construction.lean combines these into a Gaussian quadrature existence theorem.
- JacobiGeneral.lean instantiates the actual normalized integral of (1-x^2)^a
  on [-1,1], for every a>-1. Nodes/exactness/positivity are conclusions.
- BetaMoments.lean (kernel worker) proves the actual Gamma/Beta moments and
  singular-endpoint integrability. JacobiSymmetry.lean (kernel worker) proves
  actual reflected nodes/weights, including cancellation of arbitrary odd functions.

Main names: exists_gaussJacobi_rule_general, exists_positive_gaussian_rule,
monic_orthogonal_unique, orthogonal_roots_nodup.

## Actual sphere integration and finite rule

GaussianMoments and SphereMoments derive moments of Mathlib volume.toSphere
using actual polar integration, radial Gamma integrals and product Fubini.
SphereLift proves unit norm and monomial factorization of the coordinate lift.
BaseSphere supplies the two-point S^0 rule. JacobiScaledMoments and SphereRecursion
match the actual Jacobi moments to the actual sphere moments.

SphereRule defines all directions/weights recursively and proves positivity,
normalization, unit norms and spherical-polynomial exactness for degree<2q.
SphereFolding constructs a genuine antipodal involution with equal weights and
proves exactly q^(d-1) representatives. Folding preserves even integrands; no
claim of preserving arbitrary odd polynomials is made.

SphereMeasure defines the actual normalized probability measure and proves
foldedSphere_polynomial_error: error <=2e for a degree<2q comparison polynomial
within e of an even target. All rule properties are established, not assumed.

The recursion starts at S^0, using a=-1/2 for its first lift. Its exact
identification with a half-offset equally spaced circle array (and then the
original zero-offset array by rotation) is still an algorithm-identification
corollary. Positivity/exactness/count do not assume that identification.

## Actual polynomial L2 spaces and point bounds

HomogeneousSpace builds the actual L2 image of homogeneous polynomial restrictions,
recovers its unique continuous representative, and proves integral(ev(v)^2)=norm(v)^2.
SphereRotation proves actual normalized sphere measure invariance and transitivity
of ambient orthogonal maps. HomogeneousRotation proves closure under rotation and
the actual L2 isometric action. Addition.lean (root) then gives the derived point
bound, instantiated in HomogeneousBound without covariance/norm assumptions.

HomogeneousDimension proves exact dimension choose(d+k-1,k), including radial
restriction injectivity and stars-and-bars coefficient counting.
HarmonicSpace defines actual orthogonal layers H_k=E_k orthogonal to E_(k-2),
proves the inclusion by multiplication with squared radius, and derives exact
dimension differences. HarmonicRotation proves their actual invariant isometric
action. HarmonicBound combines the addition estimate with the dimension helper
(root) to obtain the sharp d-2 dimension growth exponent.

These layers are intrinsically defined orthogonal polynomial restrictions.
CasimirPolynomial, PlaneRotation, CasimirRotation, and CasimirHarmonic now prove
that their unique homogeneous representatives satisfy the ambient Laplace
equation. The argument derives rotational integration by parts from actual
invariance of surface measure, uses the polynomial Casimir identity, and tests
against the lower-degree Laplacian. The ordered-pair Casimir eigenvalue is
-2*k*(k+d-2); the factor two is explicit.

## Verified angular convergence

All theorem names below are in namespace QIRadon.Gaussian.

foldedSphere_analytic_series_error (HomogeneousQuadrature.lean) takes actual
homogeneous L2 components, an integrable even F with their pointwise expansion,
and summable exponential energy sum exp(2*a*k)||v_k||^2 <= A^2. It proves:

  error <= 2*A*[d!/(a/2)^d * exp(a/2)]*exp(-a*q)/(1-exp(-a/2)).

The finite partial sum is proved exactly integrated. Its approximation error is
derived from actual point bounds and L2 energy; it is not a supplied hypothesis.

foldedSphere_sobolev_series_error (HarmonicQuadrature.lean) uses actual orthogonal
layers and sum[((k+1)^s)||v_k||]^2 <= A^2. For d=m+2 and s>(m+1)/2 it proves:

  error <= 2*C_m*A*sqrt((2*q-1)^(m+1-2*s)/(2*s-m-1)),
  C_m = sqrt(2*(m+1)^m).

This preserves the sharp uniform Sobolev exponent s-(d-1)/2; no additional half
of a derivative is lost through an inflated homogeneous dimension bound.

## Canonical components (now verified)

HarmonicExpansionDensity proves Stone–Weierstrass density of actual polynomial
restrictions and their L2 density. HarmonicExpansionOrthogonal derives pairwise
orthogonality and the actual Hilbert direct sum. HarmonicExpansion defines
canonical coordinates in this direct sum. HarmonicExpansionEnergy proves that
exponential or Sobolev weighted energy upgrades the L2 expansion to uniform
pointwise equality with the original continuous function. CanonicalConvergence
therefore has no supplied expansion witness: its hypotheses are continuous even
F and its canonical weighted harmonic norm.

The root's HarmonicGenerating and SharpAngular strengthen the analytic constant
using the exact dimension generating function, preserving the rate a:

  error <= A*sqrt((1+exp(-a))/(1-exp(-a))^(d-1))*exp(-a*q).

The 2026-10-09 polish proves that every nonempty canonical harmonic truncation
preserves the exact sphere mean. A shared transfer theorem therefore bounds
quadrature error by the uniform tail only. Both this analytic bound and
`canonical_foldedSphere_sobolev_error` lose their former factor two; the older
noncanonical series theorem displayed above retains its original bound.

## Bounded orbit extensions to canonical energy (verified)

OrbitFamily.lean now proves BoundedOrbitExtension.canonical_energy from scalar
complex orbit extensions on a uniform total-angle budget. Its source hypotheses
are value-level ones: each finite rotation-word profile is complex C-infinity,
bounded by B on sum|angle|<δ, and agrees on real angles with the original
continuous sphere function evaluated at the actual rotated point.

The proof does not assume a continuously varying transverse frame or joint
complex-profile continuity in the sphere point. The root's OrbitContinuity
lemma derives derivative continuity from uniform finite differences;
OrbitMixedContinuity applies it to every mixed derivative. MixedRestriction
proves the exact real-restriction and coordinate-insertion calculus.
OrbitLie identifies mixed partials with the actual recursively defined Lie
rotation derivatives. OrbitFamily constructs their continuous jet and derives
its Cauchy bound. CasimirEstimate then proves actual canonical coefficient
bounds and exponential energy with a=δ*log(2)/(8*d), with explicit constant.
No spectral norm, coefficient decay, or approximation error is an assumption.

OrbitOrder.lean provides the exact source-convention adapter: reverse both
pairs and angles to change outermost-first products into innermost-first Lie
words. The l1 angle budget is unchanged. Its constructor
boundedOrbitExtension_of_outer accepts actual outermost-first profiles,
complex smoothness, value bounds, and real agreement at the actual complexified
rotation point. It produces the OrbitFamily interface without new assumptions.

AmbientRotationJet and ComplexGenerator are also verified supporting results
for an actual ambient smooth/complex extension. They are NOT used to replace
the source's orbit-family assumptions or counted as proving such an ambient
extension exists. The main OrbitFamily proof uses the intended orbit families.

## Remaining obligations

1. T1 must supply the bounded orbit-extension fields from the original local
   target regularity and actual boundary construction. The kernel worker owns
   Radon/RotationLift.lean. The entire subsequent canonical-energy implication
   is now complete, including derivative continuity and mixed-Lie identification.
2. Connect the source's actual constants and real profile to the completed
   OrbitOrder adapter and then to the global construction theorem. This is a
   genuine assembly/source obligation, not an assumed spectral certificate.
3. Explicit original circle phase/array identification, described above. Actual
   recursive Jacobi quadrature, antipodal folding, positive weights, count,
   polynomial exactness, and angular error estimates are already verified.

QIRadon/Quadrature.lean retains the supporting positive quadrature and anchored
error-transport lemmas. They are not substitutes for the concrete rule/theory.
