# Continuity of the actual canonical angular function

Implemented `QIRadon/Radon/AngularContinuity.lean`, for any finite-dimensional real inner-product space V and real Schwartz target f.

The central object is

`canonicalAngular f x : C(sphere (0:V) 1, ℝ)`

with actual value

`realRidgeProfile f theta (inner theta x) - realRidgeProfile f theta 0`.

Continuity is derived, not supplied as a premise. For the full signed-frequency formula, let k=dim(V)-1 and F be the actual Schwartz Fourier transform. Two actual Schwartz seminorms give the uniform bound

`|r|^k * norm(F(r theta)) <= (seminorm(k,0,F) + seminorm(k+2,0,F))/(1+r^2)`

for every unit theta. The right side is integrable on the real line and independent of both theta and the scalar profile coordinate. Dominated continuity therefore gives joint continuity of the Fourier-ray integral in (theta,s), and the already proved Fourier/Radon identity gives joint continuity of the real canonical profile. Composing s=inner theta x and subtracting s=0 gives the requested continuous angular function.

Principal interfaces, in namespace `QIRadon.Radon`:

- `schwartz_ray_majorant`
- `fullFrequencyIntegral_continuous`
- `realRidgeProfile_joint_continuous`
- `realRidgeProfile_angular_continuous`
- `canonicalAngular`, `canonicalAngular_apply`
- `canonicalAngular_eq_filtered`: exact complexification to the difference of the source's canonical `filteredRadonProfile (complexifySchwartz f)` evaluations.
- `canonicalAngular_antipodal`: actual evenness needed by folded sphere quadrature.
- `filteredRadonProfile_centered_continuous`: continuity in the complex filtered-profile notation itself.

There is no assumed angular regularity, approximation error, chosen ambient analytic extension, or desired coefficient bound. This module proves continuity; quantitative analytic angular energy remains the separate source/orbit calculation.

Verification: `lake build QIRadon.Radon.AngularContinuity`; principal declaration audit in `scratch/AuditAngularContinuity.lean`.

Build succeeded (8710-job dependency closure), with no warnings in the new module. All seven audited declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`. No proof holes or project axioms were introduced.
