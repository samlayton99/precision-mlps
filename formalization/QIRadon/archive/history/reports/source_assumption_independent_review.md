# Independent source-to-network assumption review

Date: 2026-10-09. Reviewer: `lean_angular`.

I inspected the current statements and proof links in `Radon/LocalSource`, `AutomaticSource`, `SourceProfiles`, `SourceOrbit`, `KnownFunctionConstruction`, `AutomaticConstruction`, `GenericArchitecture`, and `LocalAlgebraicConstruction`. I did not implement those modules. The underlying spherical harmonic/rotation results and explicit source-margin/ellipse helpers were my work, so this is an independent review of their source integration and final quantifiers, not independent review of every dependency.

I found no hidden approximation-error premise or new source-regularity assumption in the automatic analytic endpoint. Its raw inputs are a known real target, an agreeing bounded holomorphic function on a positive complex tube around a real ball of radius R>1, and epsilon>0. `AutomaticSource.ofKnownTarget` supplies the smooth radius mixture and all margins. Local profiles and their quantitative bound are derived, including the dimension-parity Hilbert branch. The source's actual centered complex rotation family supplies the angular norm; the final statement does not ask the caller for that norm or a desired convergence rate.

The endpoint quantifiers are correct: one fixed collection of finite network weights is chosen before all x in the unit ball. The same ordinary shallow GELU MLP attains the uniform bound and exactly anchors at the origin. Its counted index includes all grid halos and two global GELU anchor neurons. Dimension is d=n+2, so the raw endpoint covers exactly d≥2; the scalar theory is separate. Big-O constants are fixed with respect to epsilon and may grow substantially with d and inverse analytic margins.

The generic theorem is correctly more conditional than the named GELU endpoint. It requires genuine spatial/spectral envelopes of the actual derivative kernel and uses exponent d−1+kappa. Its higher-order integration polynomial remains an explicit architectural obligation, rather than an uncounted feature. The first-order theorem realizes r≤1 with only the actual grid and output bias; GELU realizes its affine anchor with two counted units. No statement justifies saying every activation enjoys log^d width.

The finite-regularity construction uses explicit post-Radon local polynomial approximation and canonical angular Sobolev assumptions. That is permitted by T1/T4; it does not assert an algebraic rate merely because the target is nonanalytic. Its approximating polynomials witness a regularity class, while the actual network uses the proved finite real-sample filter, not those witnesses as an oracle.

The exact construction uses mathematical integrals, roots and real coefficients. This review does not certify finite-arithmetic acquisition work or a PDE solver. Per the user's latest steering, numerical cubature/cost extensions are deferred; this does not change the exact approximation theorems. The previously started numerical helper was moved out of the proof library into `scratch/deferred_numerics` and is not part of the deliverable.

No mathematical source/automatic-endpoint defect was found in this review. The corrected tanh boundary corollary remains a separate active proof obligation; it is not covered by the generic kappa=2 tanh rate.
