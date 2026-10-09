import QIRadon.Quadrature.HarmonicExpansionEnergy

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section
set_option maxHeartbeats 1200000
set_option backward.isDefEq.respectTransparency false

/-- Geometric convergence of the actual positive folded Gauss–Jacobi sphere
rule from the canonical spectral norm of the target F. No expansion witness,
quadrature exactness, pointwise component bound or approximation rate is assumed.
-/
theorem canonical_foldedSphere_analytic_error (q : ℕ) (hq : 0 < q) (n : ℕ)
    (F : C(UnitSphere n, ℝ)) (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    (a A : ℝ) (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ ↦ Real.exp (2 * a * k) * ‖continuousHarmonicCoordinates n F k‖ ^ 2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * ‖continuousHarmonicCoordinates n F k‖ ^ 2) ≤ A ^ 2) :
    |(∫ θ, F θ ∂normalizedSphere n) -
      ∑ i : SphereHalfIndex q n, foldedSphereWeight q hq n i * F (sphereNodeOnSphere q hq n i.val)| ≤
    2 * ((A * (((n + 1).factorial : ℝ) / (a / 2) ^ (n + 1) * Real.exp (a / 2))) *
      Real.exp (-a * q) / (1 - Real.exp (-(a / 2)))) := by
  apply foldedSphere_analytic_series_error q hq n
    (fun k ↦ (continuousHarmonicCoordinates n F k).val) F
    (F.continuous.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)) heven
    (canonical_harmonic_pointwise_analytic n F a A ha hA hsum henergy) a A ha hA
  · exact hsum
  · exact henergy

/-- Algebraic convergence of the same actual rule from the canonical Sobolev
spectral norm, with the sharp s-(d-1)/2 uniform tail exponent.
-/
theorem canonical_foldedSphere_sobolev_error (q : ℕ) (hq : 0 < q) (m : ℕ)
    (F : C(UnitSphere (m + 1), ℝ)) (heven : ∀ θ, F (sphereNeg (m + 1) θ) = F θ)
    (s A : ℝ) (hs : ((m : ℝ) + 1) / 2 < s) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ ↦
      (((k : ℝ) + 1) ^ s * ‖continuousHarmonicCoordinates (m + 1) F k‖) ^ 2))
    (henergy : (∑' k : ℕ,
      (((k : ℝ) + 1) ^ s * ‖continuousHarmonicCoordinates (m + 1) F k‖) ^ 2) ≤ A ^ 2) :
    |(∫ θ, F θ ∂normalizedSphere (m + 1)) -
      ∑ i : SphereHalfIndex q (m + 1), foldedSphereWeight q hq (m + 1) i *
        F (sphereNodeOnSphere q hq (m + 1) i.val)| ≤
    harmonicGrowthConstant m * A *
      Real.sqrt ((((2 * q - 1 : ℕ) : ℝ) ^ ((m : ℝ) + 1 - 2 * s)) /
        (2 * s - (m : ℝ) - 1)) := by
  refine canonical_foldedSphere_error_of_partial q hq (m+1) F heven _ ?_
  intro θ
  let v := continuousHarmonicCoordinates (m+1) F
  have h := QIRadon.SobolevTail.sobolev_component_series_tail
    (fun k θ => harmonicEvaluation (m+1) k θ (v k))
    (ρ := (m:ℝ)/2) (by linarith) hA (harmonicGrowthConstant_nonneg m)
    (fun k => norm_nonneg (v k)) hsum henergy
    (fun k θ => harmonicEvaluation_bound m k θ (v k)) θ
    (n := 2*q-1) (by omega)
  simp only [Nat.sub_add_cancel (by omega : 1 ≤ 2*q)] at h
  have hexp : 1+2*((m:ℝ)/2-s) = (m:ℝ)+1-2*s := by ring
  have hden : -2*((m:ℝ)/2-s)-1 = 2*s-(m:ℝ)-1 := by ring
  simp only [hexp,hden] at h
  have hrep := canonical_harmonic_pointwise_sobolev m F s A hs hA hsum henergy θ
  simpa only [v, ← hrep] using h.2

end
end QIRadon.Gaussian
