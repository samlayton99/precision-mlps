# Canonical Radon reconstruction and the original analytic target source certificate

Owned files:

- `QIRadon/Radon.lean`
- `QIRadon/Radon/Slice.lean`
- `QIRadon/Radon/Boundary.lean`
- `QIRadon/Radon/CrossSection.lean`
- `QIRadon/Radon/Analytic.lean`
- `QIRadon/Radon/HilbertAbel.lean`
- `QIRadon/Radon/HilbertLocal.lean`
- `QIRadon/Radon/HilbertAnalytic.lean`
- `QIRadon/Radon/HilbertLogSharp.lean`
- `QIRadon/Radon/Filter.lean`
- `QIRadon/Radon/FilteredContinuation.lean`
- `QIRadon/Radon/ProjectionBounds.lean`
- `QIRadon/Radon/Source.lean`
- `QIRadon/Radon/RotationLift.lean`
- `QIRadon/Radon/RotationTail.lean`
- `QIRadon/Radon/RotationFilter.lean`
- `QIRadon/Radon/UniformVolume.lean`
- `QIRadon/Radon/SourceParameters.lean`
- `QIRadon/Radon/LocalSource.lean`
- `QIRadon/Radon/SourceProfiles.lean`
- `QIRadon/Radon/SourceOrbit.lean`
- `QIRadon/Radon/RadiusDensity.lean`
- `QIRadon/Radon/AutomaticSource.lean`

Dependency: coordinator's `QIRadon/Polar.lean`, which proves the actual polar change of coordinates from Euclidean volume.

## What is fully proved

### Actual hyperplane Fourier slice

`Hyperplane θ` is the actual orthogonal complement of the unit direction. `frameLp` is explicitly constructed using the unit-vector span isometry and the orthogonal decomposition into the span and its complement. Its ordinary product-coordinate version is proved to be `frame θ (s,y)=s*θ+y` and to preserve actual Euclidean volume.

`hyperplaneRadon f θ s = ∫_{y∈θ⊥} f(s*θ+y)` is an actual hyperplane integral.

`fourier_slice` proves `F[hyperplaneRadon f θ](κ)=F[f](κ*θ)` under integrability of f. The proof derives the coordinate integrability, applies Fubini, and uses the explicitly proved frame-volume identity. It does not assume the Fourier-slice conclusion or a geometric coordinate-change oracle.

### Canonical normalized real full-frequency reconstruction

`surfaceMeasure` is actual `volume.toSphere`; `normalizedSphere` divides by its finite nonzero total mass and is proved a probability measure.

`surfaceArea_gamma` proves the exact dimension constant:

`surfaceArea = 2 * sqrt(pi)^d / Gamma(d/2)`.

The real profile is first defined from the real part of the actual positive-frequency Fourier ray. Actual Fourier conjugate symmetry then proves that this profile equals half the sum of the positive ray and its opposite. The full signed radial integral is proved equal to that pair. Every ray is proved integrable for Schwartz targets by explicitly restricting the Fourier-Schwartz function along the unit line.

`real_ridge_antipodal` proves `g_{-θ}(-s)=g_θ(s)`.

`filteredRadonProfile` is defined by the full signed-frequency reconstruction filter applied to the actual hyperplane Radon projection. `real_ridge_eq_filtered_radon` proves its equality with the Fourier-derived real profile using the actual slice theorem.

`normalized_filtered_radon_reconstruction` proves exact normalized-sphere reconstruction for every real Schwartz target. No ridge-representation identity is an assumption.

The frequency convention is mathlib's cycles convention `exp(2π i ξ·x)`. The profile therefore has coefficient `surfaceArea/2` before the signed-frequency integral. Under `ω=2πξ`, its Jacobian and radial power give the paper's coefficient `surfaceArea/[2(2π)^d]`. This change of frequency units must remain explicit when comparing formulas.

### The prescribed finite boundary extension

`radialCutoff w R0 R r = ∫_[R0,R] 1_{r≤a} w(a) da` is the actual ball-mixture cutoff from the approved proof.

Proved:

- It is exactly one for `r≤R0` if the radius density is normalized.
- It is zero for `r>R`.
- It lies between zero and one for nonnegative normalized integrable density.
- The resulting extension agrees exactly with the target on the inner ball, has support in the closed outer ball, and retains the supplied real magnitude bound.
- For a smooth integrable radius density supported in `[R0,R]`, the cutoff equals the ordinary tail integral `∫_r^R w`, hence is C∞ by FTC.
- Its composition with `norm x` is C∞: away from the origin the norm is smooth; near the origin it is the constant one.
- If the target is smooth on an open neighborhood of the closed outer ball, the extension is smooth globally. Outside the target's neighborhood it is locally zero; no global target regularity is imposed.
- `radialExtensionSchwartz` explicitly bundles this actual smooth compactly supported extension as a Schwartz function.

`local_radial_filtered_reconstruction` combines those facts into exact canonical Radon reconstruction of a target known only near the finite outer ball, on the region where the extension equals the target.

The scalar radius density w is an input subject to the approved smoothness/support/normalization conditions. This module has not yet supplied a particular normalized bump and proved all its conditions from its elementary expression.

### Actual finite cross-section and scalar complex continuation

`ballMixture_crossSection` proves the real ball-mixture cross-section identity with dominated Fubini. `ball_crossSection_scaled` proves the true hyperplane change of scale, including the derived transverse dimension d−1. `ballMixture_unitBall_formula` combines them into the approved finite integral over radii and a fixed transverse unit ball. Only target regularity near the finite outer ball is used; a zero extension removes any requirement on its values elsewhere.

`complexCrossSection` is the same finite radius/unit-ball integral with the complex point `z θ + sqrt(a²−z²) u`. Its safe domain is derived from the explicit gap Q<R0 and target tube margin. The square-root derivative and displacement bounds are proved, giving the actual condition `(1 + Q/sqrt(R0²−Q²))*δ ≤ σ`. The holomorphic parameter-integral theorem is proved using Cauchy derivative estimates and dominated differentiation, rather than accepting profile holomorphy as a premise.

`complexCrossSection_analytic` proves this actual finite integral holomorphic on the scalar tube. `complexCrossSection_norm_le` gives the explicit bound `volume(transverse unit ball) * sqrt(R²+Q²)^(d−1) * B` for nonnegative normalized radius density and target bound B. `complexCrossSection_eq_radon_real` identifies it with the actual real hyperplane Radon projection. This proves scalar-position continuation of the unfiltered projection; it does not yet prove rotation-parameter continuation or filtered-profile regularity.

### Actual Fourier-to-Hilbert bridge

`HilbertAbel.lean` derives the inverse Fourier transform of `−i sign(ξ) exp(−a|ξ|)` from two convergent half-line exponential integrals. It is exactly `4πx/[a²+(2πx)²]`. Fubini then proves the explicit physical convolution identity. Dominated convergence proves that removing the damping recovers the canonical sign Fourier multiplier.

`HilbertLocal.lean` rescales a=2πε, derives the endpoint logarithm by FTC, and removes the damping separately on the inner and outer domains. `hilbertFourier_local_formula` proves, for p and its Fourier transform integrable and an actual local Lipschitz bound,

`H p(s) = (1/π) [ ∫_[−L,L] (p(t)−p(s))/(s−t) dt + p(s)(log(L+s)−log(L−s)) + ∫_{outside [−L,L]} p(t)/(s−t) dt ]`, for |s|<L.

The singularity-subtracted formula is a conclusion, not an assumption. No external Hilbert-transform identity is trusted.

### Local Hilbert continuation and the sharp endpoint constant

`hilbertContinuation` uses a removable divided difference (`dslope`) for the finite inner integral, two principal logarithms in the right half-plane, and the ordinary real outer integral. Its analyticity is proved by actual dominated holomorphic integration, the removable-singularity theorem, and denominator separation. `hilbertContinuation_eq_hilbertFourier` proves that this is a continuation of the canonical Fourier Hilbert transform; its local Lipschitz hypothesis is derived from bounded local holomorphy.

`hilbertLog_norm_le_sharp` proves the exact approved bound `|log(L+z)−log(L−z)| ≤ log((L+Q)/(L−Q))` for |z|≤Q<L. The proof integrates the derivative along tz and evaluates the real majorant by FTC. It does not replace the logarithm with the preliminary coarser rational bound.

`hilbertContinuation_norm_le_sharp` gives

`|H p(z)| ≤ (1/π)[4 L B/δ + B log((L+Q)/(L−Q)) + ||p||₁/(L−Q)]`.

At δ=4η its inner contribution is LB/η, which is stronger than the certificate R8 term 2LB/η. The prior conservative endpoint estimate remains an explicitly named intermediate theorem only.

### Both parities of the actual canonical filter

`projectionSchwartz f θ` is explicitly the inverse Fourier transform of the true Fourier slice. `fourier_projectionSchwartz` and `projectionSchwartz_eq_radon` prove its canonical identification. For supported targets the actual projection is proved globally continuous by uniform transverse-ball domination, so the inverse slice agrees with it at every real offset, including the Hilbert tail.

`radialFilter_even` proves that order 2k filtering is `(2πi)^(-2k)` times the real derivative. `radialFilter_odd_deriv_hilbert` proves that order 2k+1 filtering is `i(2πi)^(-2k−1)` times the corresponding derivative of H p. Hilbert/derivative commutation is derived by differentiating actual inverse Fourier integrals with integrable moments. Thus the bound uses the unfiltered L1 mass, as required by R4/R8, rather than inserting an uncontrolled derivative L1 mass.

`filteredRadonProfile_eq_radialFilter` connects these identities to the original full signed-frequency Radon profile, retaining the sphere-area/2 factor. `iteratedDeriv_real_of_complex` proves that complex derivatives of a local continuation restrict to the actual real derivatives of the agreeing profile.

`FilteredContinuation.lean` gives concrete continuations for both parities, proves agreement with the canonical filter on the real interval, and derives Cauchy derivative bounds on explicit smaller tubes. The sharp even-dimensional bound is `oddFilterContinuation_norm_le_sharp`.

### Actual local target supplies the projection certificate

`Source.lean` expresses the approved analytic extension assumption as `AnalyticOnNhd ℂ F (realBallTube ...)`: local convergent complex power series around the finite real outer ball. It does not assume global analyticity or analyticity of the compact extension. This exact analytic hypothesis implies the real smoothness needed to construct the finite supported extension as a Schwartz function. No unproved general finite-dimensional Frechet-holomorphic→analytic equivalence is claimed.

`canonicalProjection_certificate` starts with this target, a supplied smooth normalized nonnegative finite radius density, the explicit scalar/tube margins, and the target magnitude bound. It returns, for the actual canonical inverse Fourier slice p and actual finite cross-section P:

- P is holomorphic on the certified scalar tube.
- P has the explicit transverse-volume magnitude bound.
- p and P agree at every real point in the certified interval.
- The actual mass satisfies `∫|p| ≤ B * volume(outer ball)`.

The last bound is proved using the actual coordinate frame, Fubini, pointwise support, and the target bound. These conclusions provide the inputs to the proved parity-specific filtered continuations. No projection regularity or inverse identity is supplied as a premise.

### Actual joint rotation families and angular regularity

`RotationLift.lean` constructs the actual finite section integral under complex rotation words, using a fixed real transverse ball at each real base direction. The scalar and rotation margins are proved sufficient to keep every target evaluation inside its supplied complex neighborhood. The integrated result is jointly complex C∞ in angles and scalar offset using the checked parameter-integral theorem. Real angles agree with the actual real isometry; diagonal rotation pairs act as identity. Canonical filtered profiles intertwine with these real isometries by the Fourier slice/change-of-variables theorem.

The joint Hilbert outer tail is handled without assuming global scalar analyticity. `hilbertOuter_eq_spatial_finite` proves by the actual volume-preserving frame and Fubini that

`∫_{|t|>S} p_O(t)/(z−t) dt = ∫_{|x|≤R, |θ·x|>S} f(O·ιx) χ(|x|)/(z−θ·x) dx`.

This fixed finite real region supplies a genuine jointly smooth angular/scalar tail and the explicit bound `B*volume(B_R)/(S−Q)`. The cutoff and integration domain are never evaluated at complex radii.

`RotationFilter.lean` combines this with the checked parameterized removable divided difference, exact endpoint logarithm, and actual scalar partial derivatives. Both parity-specific filtered families are jointly smooth, have proved common bounds, and agree with the canonical filtered projection at every real scalar offset in the final closed interval. The scalar derivatives are actual Fréchet-coordinate derivatives of the joint lift, identified with the ordinary scalar derivatives.

No global ambient analytic extension of the angular function is assumed. `SourceOrbit.lean` instead supplies the exact object the angular proof needs: one actual bounded complex extension for each real base direction and each rotation word, agreeing with the original centered angular function at every real angle tuple. The real angular function is independently proved continuous by the Fourier-Schwartz integral (the angular-continuity worker). The checked reversal adapter handles the Lie-word ordering convention.

`LocalAnalyticSource.boundedOrbit x hx` is an actual constructor of

`BoundedOrbitExtension (canonicalAngular src.extendedReal x) src.params.b (2*src.profileBound)`.

The angular worker's `BoundedOrbitExtension.canonical_energy` then derives the exponential spherical-harmonic energy bound. Neither an angular norm bound nor a profile reconstruction identity is an input to this constructor.

### Input-only source setup and an explicit radius density

`LocalAnalyticSource n` uses dimension n+2, covering the approved d≥2 class. Its inputs are the original real target, its bounded local analytic complex extension agreeing on B_R, explicit numerical construction margins, and a smooth finite radius density. It stores no profile analyticity or angular regularity conclusions.

`extendedReal_eq_target_unitBall` proves exact agreement with the original target. `profile_analytic`, `profile_bound`, and `profile_eq_realRidge` supply the actual scalar continuation, uniform bound, and closed-interval canonical agreement. `jointProfile_bound` is uniform over real directions, rotation words of every length, all admissible complex angles, and the final scalar tube.

The numerical margins have an automatic checked constructor in the angular worker's `SourceParameterChoice.lean`. The available operator estimate uses exp(2 sum|angles|), so the certified angular budget is conservatively half the paper's t₀. This factor is explicit; no change to the target class or logarithmic exponent is claimed.

`RadiusDensity.lean` proves every property of the explicit normalized bump

`w(a) = expNegInvGlue((a−R₀)(R−a)) / ∫_[R₀,R] expNegInvGlue((t−R₀)(R−t)) dt`.

The numerator is exactly exp(−1/[(a−R₀)(R−a)]) inside the interval and zero outside. It is C∞, compactly supported, nonnegative, and positive inside. Its normalization is a strictly positive finite one-dimensional integral, and the resulting density integrates to one.

`LocalAnalyticSource.ofKnownTarget` in `AutomaticSource.lean` combines the automatic margins and this density. The caller therefore supplies only R>1, σ>0, a bound B≥0, the known real target, a local analytic extension, agreement on the real outer ball, and the magnitude bound. There are no density, scalar-profile, angular-norm, or reconstruction oracles among its inputs.

## Remaining scope outside this source certificate

The **analytic local-target source certificate is now proved**. The coordinator owns its final composition with finite quadrature, kernel construction, explicit M/N schedules, actual MLP realization, and width Big-O theorems. These source modules do not themselves claim that final theorem assembly is finished.

A separate finite-smooth local-target→algebraic angular/Sobolev certificate has not been derived here. As the approved radial extension note states, that would require quantitative finite-derivative/Hilbert estimates. Any algebraic approximation theorem must retain its actual stated profile/Sobolev hypotheses until that further bridge is proved. The exact general Radon reconstruction and the analytic source theorem are not a substitute for that distinction.

The sharper exp(sum|angles|) operator constant from the informal matrix-exponential argument remains an optimization; the proved factor-two angular bound is recorded in the automatic margins.

## Verification

The complete source chain through `SourceOrbit.lean` builds. `#print axioms` audits for exact extension agreement, scalar continuation, scalar bound, canonical real agreement, actual joint filtered bound, real/angular tail identity, and `LocalAnalyticSource.boundedOrbit` report only `[propext, Classical.choice, Quot.sound]`. Earlier reconstruction, cross-section, Hilbert, sharp-log, and parity-filter audits report the same. No sorry/admit/custom axioms are present. RadiusDensity and AutomaticSource also build, and their normalization, smoothness, positive-integral and automatic-constructor audits report only the same three standard axioms.
