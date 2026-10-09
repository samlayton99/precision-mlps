# Fresh semantic audit, 2026-10-09

Independent worker `/root/polish_fresh_audit`; reviewed the decisive source → scalar/angular certificates → fixed network chain against the original approved theorem map. No correctness or hidden-assumption finding in that chain. This is a targeted semantic review, not a second complete rereading or independent rebuild of all files.

## Evidence

- `Radon/LocalSource.lean:16–32` supplies only the original bounded local analytic target, the boundary density, and explicit geometric margins. No profile approximation or harmonic-energy oracle occurs in that structure. `Radon/AutomaticSource.lean:16–38` constructs its density and margins from `R>1` and `σ>0`; `AutomaticConstruction.known_target_gelu` (line 54) exposes only the raw target assumptions.
- `Radon/LocalSource.lean:39–76` constructs an actual compact smooth extension and proves exact target agreement on the whole closed unit ball. Compact support is not combined with an impossible claim of global analyticity. `Radon/Source.lean:25–64` derives smoothness using local analyticity and the flat radial cutoff.
- `Radon.lean:278–325` derives normalized Fourier reconstruction and the actual Fourier-slice/filter identification. `Radon/SourceProfiles.lean:190–209` identifies the supplied local continuation with the real canonical profile over the full used scalar interval. `KnownFunctionConstruction.lean:45–82` then proves scalar elliptic regularity, centered profile agreement and uniform angular energy. `Radon/SourceOrbit.lean:23–89` derives the latter from one common bounded complex rotation family for every word length.
- `Kernel.lean:792–837` proves the finite spectral certificate from integrability, Schwartz regularity and used-band nonvanishing. Its hypotheses contain no approximation bound. Cutoff, all nonzero integer aliases, and the complete omitted-center complement are explicit. `ScalarRate.lean:127–189` bijectively reindexes both omitted half-grids; `LocalProfileRate.lean:257–280` uses the exact symmetric finite grid including both halos.
- `LocalProfileRate.lean:17–55,191–254` derives the actual prepared derivative spectrum and its inversion, including the accuracy-dependent preparation norm. The local profile theorem does not silently assume a globally analytic original profile or an epsilon-independent preparation norm.
- `KnownFunctionConstruction.lean:86–170` fixes direction/row orders before quantifying over every point in the unit ball. `Architecture.lean:16–19` is an ordinary shallow network on raw coordinates; `GenericArchitecture.lean:104–109` proves the Euclidean inner-product/coordinate-sum conversion, preventing a sup-norm ball substitution.
- `GenericAnchoredConstruction.centeredSpokeAnchor` (line 17) and `known_target_with_anchor` (line 127) defines the actual integration polynomial and assumes approximation of that polynomial only. Its own error and units are added explicitly. No unconditional arbitrary-order ordinary-MLP rate follows without a suitable anchor realization. The r≤1 endpoint needs no additional units. `Activations/Anchors.lean:8–43` realizes all GELU affine anchors with two neurons total, using exact GELU reflection.
- `LocalAlgebraicConstruction.local_centered_uniform_accuracy_and_width` (line 109) assumes uniform polynomial approximation of each real scalar profile and the actual centered angular Sobolev energy. These are explicit source-class assumptions, not the full target approximation conclusion. `LocalAlgebraic.lean:25–31` proves dependence only on the stated real samples; the comparator polynomial is not an input to the network. Its grid count uses `2*N`, as does the actual preparation. The algebraic exponent is `(d−1)/t+κ/s`; the stronger global Fourier class has its separate exponent.

## Scope and wording

“Local finite-smooth class” should be understood as the explicit uniform polynomial-approximation class above. This endpoint does not independently prove a Jackson theorem from a raw target's C^s derivatives. General arbitrary-order results count grid units separately unless an anchor realization is supplied. The current status discloses the optional stronger tanh correction and full numerical coefficient acquisition as deferred; neither was used in this audit.

## Safe simplifications and bound improvement

1. Canonical harmonic truncation already preserves the spherical integral. Prove this once from the actual orthogonal L2 decomposition, then estimate only the quadrature of the tail. This removes the extraneous factor two in the analytic angular bound without changing assumptions, rates or the grid. The shared `Gaussian.canonical_partial_integral` and `canonical_foldedSphere_error_of_partial` in `HarmonicExpansion.lean` now implement this step for both the sharp analytic and canonical Sobolev endpoints. `SharpAngular.canonical_quadrature_error` and `Gaussian.canonical_foldedSphere_sobolev_error` both have half their former prefactor. This was an explicitly authorized follow-up to the read-only audit.
2. Use one positive-weight transport lemma for centered analytic and algebraic assemblies; their only mathematical difference is the scalar and angular certificates. The coordinator owns this refactor.
3. The accuracy-order rounding constant can be sharpened directly, retaining the exact same order selection; the coordinator owns this change.

No numerical acquisition, improved tanh-boundary argument, commits or publication were performed by this worker.

## Follow-up verification

The final `lake build QIRadon.Quadrature.SharpAngular` passed, including the changed `HarmonicExpansion`, `CanonicalConvergence` and `SharpAngular` modules. Direct checking of `CanonicalConvergence.lean` also passed. Its temporary elaboration timeout was eliminated by restricting two scalar rewrites with `simp only`; the original heartbeat limit was retained and no hypotheses changed. The coordinator is responsible for the final whole-package build and axiom audit.

The worker also independently checked the coordinator's propagation of the improved analytic constant, common error-transport lemma and rounding constant. The actual angular-order selectors and all corresponding count statements match. The scalar centering factor two, epsilon/4 scalar allocation and source orbit bound `2*profileBound` remain intact. No semantic mismatch was found.
