import QIRadon.Quadrature.SphereFolding

open MeasureTheory Metric Set
open scoped BigOperators

namespace QIRadon.Gaussian

abbrev UnitSphere (n : ℕ) := sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1

noncomputable def normalizedSphere (n : ℕ) : Measure (UnitSphere n) :=
  (((volume : Measure (EuclideanSpace ℝ (Fin (n + 1)))).toSphere) Set.univ)⁻¹ •
    (volume : Measure (EuclideanSpace ℝ (Fin (n + 1)))).toSphere

theorem sphere_mass_ne_zero (n : ℕ) :
    ((volume : Measure (EuclideanSpace ℝ (Fin (n + 1)))).toSphere) Set.univ ≠ 0 :=
  Measure.measure_univ_ne_zero.mpr (Measure.toSphere_ne_zero _)

instance normalizedSphere_probability (n : ℕ) : IsProbabilityMeasure (normalizedSphere n) := by
  apply isProbabilityMeasure_iff.mpr
  rw [normalizedSphere, Measure.smul_apply, smul_eq_mul]
  exact ENNReal.inv_mul_cancel (sphere_mass_ne_zero n) (measure_ne_top _ _)

instance normalizedSphere_openPos (n : ℕ) : (normalizedSphere n).IsOpenPosMeasure where
  open_pos U hU hnon := by
    rw [normalizedSphere, Measure.smul_apply, smul_eq_mul]
    exact mul_ne_zero (ENNReal.inv_ne_zero.mpr (measure_ne_top _ _))
      (hU.measure_ne_zero _ hnon)

/-- The probability-measure integral is the earlier normalized surface average. -/
theorem normalizedSphere_integral (n : ℕ) (f : UnitSphere n → ℝ) :
    (∫ θ, f θ ∂normalizedSphere n) = sphereAverage f := by
  rw [normalizedSphere, integral_smul_measure]
  simp only [ENNReal.toReal_inv, smul_eq_mul, sphereAverage, Measure.real,
    div_eq_mul_inv, mul_comm]

noncomputable def sphereNodeOnSphere (q : ℕ) (hq : 0 < q) (n : ℕ)
    (i : SphereIndex q n) : UnitSphere n :=
  ⟨sphereNode q hq n i, by simpa using sphereNode_norm q hq n i⟩

def sphereNeg (n : ℕ) (θ : UnitSphere n) : UnitSphere n :=
  ⟨-θ.val, by simp⟩

theorem sphereNodeOnSphere_reflection (q : ℕ) (hq : 0 < q) (n : ℕ) (i : SphereIndex q n) :
    sphereNodeOnSphere q hq n (sphereReflection q hq n i) =
      sphereNeg n (sphereNodeOnSphere q hq n i) :=
  Subtype.ext (sphereNode_reflection q hq n i)

theorem foldedSphere_sum_even_onSphere (q : ℕ) (hq : 0 < q) (n : ℕ)
    (F : UnitSphere n → ℝ) (heven : ∀ θ, F (sphereNeg n θ) = F θ) :
    (∑ i, sphereWeight q hq n i * F (sphereNodeOnSphere q hq n i)) =
      ∑ i : SphereHalfIndex q n,
        foldedSphereWeight q hq n i * F (sphereNodeOnSphere q hq n i.val) := by
  have h := sum_invariant_eq_twice_half q hq n
    (fun i ↦ sphereWeight q hq n i * F (sphereNodeOnSphere q hq n i)) (by
      intro i
      rw [sphereWeight_reflection, sphereNodeOnSphere_reflection, heven])
  rw [h, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  unfold foldedSphereWeight
  ring

/-- Certified angular integration error for the actual folded sphere rule.
Only the comparison polynomial and its approximation error are supplied;
positivity, normalization and polynomial exactness are all proved constructions. -/
theorem foldedSphere_polynomial_error {τ : Type*} [Fintype τ]
    (q : ℕ) (hq : 0 < q) (n : ℕ) (F : UnitSphere n → ℝ)
    (hF : Integrable F (normalizedSphere n))
    (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    (coefficient : τ → ℝ) (exponent : τ → Fin (n + 1) → ℕ)
    (hdegree : ∀ t, (∑ j, exponent t j) < 2 * q)
    (err : ℝ)
    (happrox : ∀ θ, |F θ - ∑ t, coefficient t * coordinateMonomial (exponent t) θ.val| ≤ err) :
    |(∫ θ, F θ ∂normalizedSphere n) -
      ∑ i : SphereHalfIndex q n,
        foldedSphereWeight q hq n i * F (sphereNodeOnSphere q hq n i.val)| ≤ 2 * err := by
  let P : UnitSphere n → ℝ := fun θ ↦
    ∑ t, coefficient t * coordinateMonomial (exponent t) θ.val
  have hPc : Continuous P := by
    unfold P coordinateMonomial
    fun_prop
  have hPi : Integrable P (normalizedSphere n) :=
    hPc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hexact : (∫ θ, P θ ∂normalizedSphere n) =
      ∑ i, sphereWeight q hq n i * P (sphereNodeOnSphere q hq n i) := by
    rw [normalizedSphere_integral]
    exact (sphereRule_polynomial_exact q hq n coefficient exponent hdegree).symm
  have h := QIRadon.positive_quadrature_error (normalizedSphere n)
    (sphereNodeOnSphere q hq n) (sphereWeight q hq n) F P err
    (fun i ↦ (sphereWeight_positive q hq n i).le)
    (sphereWeight_normalized q hq n) hF hPi hexact happrox
  rwa [foldedSphere_sum_even_onSphere q hq n F heven] at h

end QIRadon.Gaussian
