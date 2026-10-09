# Mixed Cauchy estimate for complex rotation parameters

`QIRadon/Quadrature/MixedCauchy.lean` proves the mixed derivative estimate needed to convert local complex-rotation analyticity into angular derivative control. The file builds cleanly and has no additional theorem assumptions that assert derivative growth.

## Interface

Namespace: `QIRadon.MixedCauchy`.

- Domain: `Fin n → ℂ`, with the ordinary function-space norm for Lean's analytic structure.
- Actual analytic region: `l1Ball n δ = {z | sum_i norm(z_i) < δ}`. This is proved open by `isOpen_l1Ball`.
- Codomain: any complete complex normed vector space `V`.
- Hypotheses: `AnalyticOnNhd ℂ F (l1Ball n δ)` and `norm(F z) ≤ M` on that region.
- Actual derivative: `mixedPartial [] F = F`; `mixedPartial (i::w) F z = fderiv ℂ (mixedPartial w F) z (Pi.single i 1)`.

`mixedPartial_analytic` proves these successive coordinate derivatives remain analytic on the same region.

## Primary bound and proof

`mixedPartial_bound` proves, for `r>0` and any index list `w`,

`l1Size z + length(w)*r < δ  =>  norm(mixedPartial w F z) ≤ M/r^length(w)`.

The proof is by induction on the actual derivative list. For a derivative in coordinate `i`, restrict the remaining derivative function to the complex line `t ↦ z+t e_i`. On the circle `|t|=r`, the elementary bound

`l1Size(z+t e_i) ≤ l1Size(z)+|t|`

leaves enough radius for the inductive hypothesis. Analyticity justifies the ordinary first-derivative Cauchy estimate. That estimate contributes precisely one factor `1/r`. The chain rule identifies the line derivative with the actual coordinate derivative in the definition. Repeating produces `r^-length(w)` and no extra factorial.

`mixedPartial_zero_bound` takes `r=δ/(2 length(w))`, for nonempty `w`, to obtain

`norm(mixedPartial w F 0) ≤ M (2 length(w)/δ)^length(w)`.

The half-radius leaves strict room inside the open domain, so no analyticity assumption on its boundary is introduced. Repeated indices are allowed; no distinctness assumption is needed.

`allCoordinatePartial F` uses the list `List.ofFn id`, taking exactly one derivative in each of the `n` parameters. `allCoordinatePartial_zero_bound` gives

`norm(allCoordinatePartial F 0) ≤ M (2n/δ)^n`.

For the rotation-orbit application, the other modules must still prove that the actual orbit is holomorphic and bounded on this domain, and identify its coordinate derivative with the corresponding ordered real rotation-generator word. This file does not assume those identification results or assert the full angular convergence theorem.

## Verification

`lake build QIRadon.Quadrature.MixedCauchy` succeeds without warnings. The four primary analytic/bound declarations were inspected with `#print axioms`; only `propext`, `Classical.choice`, and `Quot.sound` occur.

## Direct C∞ variant for integrated profiles

The `_of_smooth` versions now derive the identical bound from actual `ContDiffOn ℂ ∞` on the open ℓ¹ domain. Successive partials stay C∞ there (`mixedPartial_smooth`), hence are differentiable on each required complex line. The one-variable Cauchy estimate then gives the same constant and factorial-free result. `Radon/ParameterIntegral.lean` derives this joint C∞ hypothesis from the actual dominated integrands; it is not assumed as an integral oracle. See `reports/parameter_integral.md` for that chain.
