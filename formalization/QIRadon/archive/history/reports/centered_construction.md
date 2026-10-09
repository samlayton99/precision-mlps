# Uniform centered construction

The approved angular object is `H_x(theta)=g_theta(theta·x)-g_theta(0)`. The new `CenteredConstruction.lean` implements exactly this version:

`Q_centered(x) = f0 + Q_local(x) - Q_local(0)`.

The subtraction is a number in the output bias. It creates no new hidden activation type and adds no hidden neurons. `centeredLocalNetwork_zero` proves exact anchoring at the origin.

`centered_analytic_finite_error` proves the error bound

`U exp(-a q) + 2 V exp(-b N)`

from the actual centered canonical harmonic energy and the actual local scalar preparation. The factor two follows by subtracting the two already proved scalar errors. It does not assume an angular regularity bound on the separate origin profile.

`centered_analytic_uniform_accuracy_and_width` chooses q at angular budget epsilon/2 and N at scalar budget epsilon/4. It fixes those orders and the entire network before quantifying over every input in the unit ball. It proves uniform accuracy and the exact halo-inclusive finite grid bound. `centered_total_grid_epsilon_bigO` proves the same logarithmic exponent with any separately realized fixed anchor-unit count.

`canonical_ridge_integrable` derives angular integrability of the actual Fourier/Radon profile from Schwartz integrability. `centeredReconstructed_canonical_on_ball` then proves that the centered reconstructed target is the actual original function when the scalar continuations agree with its canonical profiles only on |s|≤1. No global complex continuation agreement is required.

`CenteredGELU.canonical_uniform_accuracy_and_width` composes these results with the explicit GELU realization. It produces one ordinary one-hidden-layer GELU MLP, uniformly approximates the actual canonical Schwartz target, exactly hits f(0), and counts q^(d−1) times the complete center grid plus two anchor neurons. `canonical_total_epsilon_bigO` is the actual halo-and-anchor-inclusive O(|log epsilon|^d) statement for the same order choices.

`UniformConstruction.lean` also supplies the simpler uniform wrappers for the existing full-profile analytic and algebraic theorems. These are separate from the centered version; they require uniform norms of the full angular profile.

All these theorems remain conditional on the local scalar analytic class and actual canonical angular norm bounds. They do not claim that those bounds have already been derived from the original finite-domain target. The source bridge is now proved separately in `KnownFunctionConstruction.lean` and instantiated automatically in `AutomaticConstruction.lean`. For generic derivative order r≥2, the generic evaluator still carries its explicitly displayed polynomial-anchor obligation; the actual GELU r=2 realization is complete.

Historical milestone verification: `lake build QIRadon.CenteredConstruction QIRadon.CenteredGELU QIRadon.UniformConstruction` and the principal-declaration audit in `scratch/AuditCentered.lean`. No existing source module was edited.

Build succeeded (8776 jobs in the dependency closure). All ten audited principal declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`; no proof holes or new axioms were introduced.
