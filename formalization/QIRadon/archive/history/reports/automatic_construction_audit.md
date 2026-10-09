# Automatic known-target endpoint: repair and assumption audit

Date: 2026-10-09. Reviewer: `lean_local_scaling`.

## Minimal elaboration repair

Only `QIRadon/AutomaticConstruction.lean` was edited. The raw source constructor was changed from a directly applied term inside a let-binding to an expected-type tactic block:

```lean
let src : LocalAnalyticSource n := by
  apply LocalAnalyticSource.ofKnownTarget (n := n) R σ B hR hσ hB f F hF hagree
  intro z hz
  exact hbound z hz
```

This constructs exactly the same automatic source and changes no mathematical assumption or conclusion. It prevents the constructor's proof-argument elaboration from being delayed into the dependent neuron-index/typeclass context. The isolated source constructor already compiled with the full imports; the unseparated endpoint previously timed out after 1.6 million heartbeats.

Both custom heartbeat and transparency overrides were removed. The entire file now checks with default settings in about **3.3 seconds**, and `lake build QIRadon.AutomaticConstruction` succeeds. No bundled replacement conclusion or weaker predicate was needed.

## Independent statement-level audit

I inspected `KnownFunctionConstruction`, `AutomaticConstruction`, `AutomaticSource`, `LocalSource`, `SourceParameters`, `SourceParameterChoice`, `RadiusDensity`, `SourceProfiles`, `SourceOrbit`, and the actual bounded-orbit-to-canonical-energy adapter. I did not implement those source and endpoint arguments. Some lower-level scalar/count/centered-architecture modules were implemented by me earlier, so review of those dependencies is not independent.

### Exactly what the raw theorem assumes

`AutomaticConstruction.known_target_gelu` takes:

1. Dimension d=n+2, covering every integer d≥2.
2. R>1 and a positive complex tube radius sigma.
3. A real target f on real Euclidean d-space and a complex function F on complex Euclidean d-space.
4. F analytic on the tube of radius sigma around the closed real ball of radius R.
5. F agrees with f on that real ball.
6. A nonnegative uniform magnitude bound B for F on that tube.

There is no supplied ridge profile, approximation inequality, harmonic norm, derivative-growth bound, frame, radius density, fitted coefficient vector, or quadrature error among the inputs.

These are the approved bounded local analytic-source assumptions, expressed in Lean's `AnalyticOnNhd` language. No global analyticity or exponential Fourier bound is imposed on the compact extension. The target can be arbitrary outside the outer ball; the construction does not rely on those values.

### Exactly what it proves

The theorem constructs one fixed source from those inputs. For every epsilon>0 it supplies arrays of ordinary one-hidden-layer GELU weights, hidden biases, readouts, and one output bias. The arrays are chosen **before** the universal quantifier over x. The same MLP satisfies error≤epsilon at every x in the unit ball and equals f(0) exactly at the origin.

The same source and prescribed integer orders determine the actual `NeuronIndex`. Its cardinality is separately proved to be

`O(|log epsilon|^d)`

as epsilon decreases to zero, for fixed d and fixed source bounds. The index includes all direction/center combinations, both halos, and **two ordinary GELU anchor neurons total**. It is not a continuum-width surrogate or an uncounted polynomial feature space.

The raw theorem is a GELU corollary. The parent `KnownFunctionConstruction.general_kernel_accuracy_and_width` remains the general derivative-kernel theorem, with its correct exponent d−1+kappa and explicit anchor-realization obligation. The ordinary-MLP first-order corollary covers general r=0 or r=1 kernels. There is no false claim that every activation has the GELU exponent or can represent every integration polynomial exactly.

### The source-to-regularity bridge is now real, not assumed

`AutomaticSource.ofKnownTarget` supplies a particular normalized smooth radius density and explicit source margins from R and sigma. `standardRadiusDensity` is the stated finite-interval exponential bump divided by its proved positive one-dimensional normalizing integral. `standardSourceParameters` proves all scalar, Hilbert, and rotation margins for its actual formulas. These are construction choices, not extra target hypotheses.

`LocalSource.extendedComplex` is the actual compact radial extension and is proved Schwartz. Its real part equals the original target on the unit ball, and its complexification agrees with the original complex extension. The source is never declared globally analytic.

`SourceProfiles` derives the local scalar continuation and its bound from finite cross-sections and the dimension-dependent filter. Odd/even dimension selects ordinary derivatives or the proved singularity-subtracted finite Hilbert formula. `KnownFunctionConstruction.scalar_profile_certificate` transfers the resulting scalar tube to the actual Bernstein ellipse used by sampled preparation.

`SourceOrbit` constructs actual complex rotation-word families, including the input-dependent bilinear scalar coordinate. Its common bound is derived from the source target bound and finite cross-section/Hilbert estimates. At real angles it agrees exactly with the actual canonical centered angular function. `BoundedOrbitExtension.canonical_energy` then derives actual canonical harmonic energy through the proved rotation-derivative/Casimir argument. No derivative jet, spectrum, angular expansion, or energy bound is assumed by the original source.

Thus the earlier acknowledged source-to-angular-regularity gap is closed in these theorem statements. The scalar and angular hypotheses of the lower-level finite-network theorem are actual conclusions from the raw source.

### Frequency, boundary, architecture, and constants

The source and finite-kernel constructions consistently use cycles-frequency Fourier normalization. The actual Radon factor and derivatives are carried through the source profiles; the previously audited conversion of the older angular-frequency grid-count helper remains explicit.

The automatic endpoint uses T0=2 for local Gaussian-window preparation, the proved `oversampling geluSpectral` choice, and the proved `haloScale geluSpectral geluSpatial`. These choices satisfy the same alias/halo inequalities used by the actual construction. They are sufficient choices, not a newly claimed optimal lambda rule.

The angular and scalar budgets are epsilon/2 and epsilon/4, respectively. The latter pays for exact global centering. The function is anchored by changing the output constant; there is no nonstandard difference activation in the MLP.

All margins, angular norms, and scalar constants are fixed independently of epsilon and can depend strongly on dimension and the target domain. The theorem is fixed-d asymptotics; it does not remove the curse of dimensionality or assert dimension-independent constants. The explicit finite width bound remains in the parent theorem, preceding the Big-O corollary.

## Remaining limits that this endpoint does not claim to solve

This is an **exact-arithmetic known-function approximation theorem**. It does not solve an unknown PDE, provide a PINN inverse solver, establish fixed-float64 performance, or finish the full T8 bit-complexity and profile-acquisition-cost analysis. Quadrature nodes, finite integrals, and activation values are mathematical real quantities here. Their existence and identities are verified; certified numerical acquisition and total runtime are separate obligations.

Likewise, the automatic choices are not the exact discrete global optimum in neurons. The independent allocation theorems optimize specified continuous error/cost certificates. No neural-network-wide optimality claim follows merely from the automatic endpoint.

I found no new mathematical-scope defect in `KnownFunctionConstruction` or the repaired raw-input theorem. The repair addresses elaboration only.

## Verification

- `lake env lean QIRadon/AutomaticConstruction.lean`: succeeds with default options, about 3.3 seconds.
- `lake build QIRadon.AutomaticConstruction`: succeeds (8823-job dependency closure).
- `scratch/AuditAutomaticConstruction.lean`: all 11 audited declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`, including the final raw-input `known_target_gelu`.
- No `sorry`, `admit`, project axiom, or weakening of the original theorem statement was introduced.
