import QIRadon.Quadrature.HarmonicExpansion

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section
set_option maxHeartbeats 1200000
set_option backward.isDefEq.respectTransparency false

/-- Exponential weighted energy of the canonical coefficients implies their
pointwise expansion is the original continuous function. No representation
witness or approximation-error hypothesis is supplied. -/
theorem canonical_harmonic_pointwise_analytic (n : ℕ) (F : C(UnitSphere n, ℝ))
    (a A : ℝ) (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ ↦ Real.exp (2 * a * k) * ‖continuousHarmonicCoordinates n F k‖ ^ 2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * ‖continuousHarmonicCoordinates n F k‖ ^ 2) ≤ A ^ 2)
    (θ : UnitSphere n) :
    F θ = ∑' k, harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k) := by
  let C := A * (((n + 1).factorial : ℝ) / (a / 2) ^ (n + 1) * Real.exp (a / 2))
  let r := Real.exp (-(a / 2))
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hr : 0 ≤ r := (Real.exp_pos _).le
  have hr1 : r < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  have hsr : Summable (fun k : ℕ ↦ C * r ^ k) :=
    (summable_geometric_of_lt_one hr hr1).mul_left C
  apply canonical_harmonic_pointwise_of_majorant n F (fun k ↦ C * r ^ k)
    (fun k ↦ mul_nonneg hC (pow_nonneg hr _)) hsr _ θ
  intro k x
  have hb := QIRadon.SeriesTail.component_bound
    (fun k ↦ norm_nonneg (continuousHarmonicCoordinates n F k)) hA hsum henergy k
  have he := polynomialEvaluation_bound_power n k x
    (continuousHarmonicCoordinates n F k).val
  change |harmonicEvaluation n k x (continuousHarmonicCoordinates n F k)| ≤
    ((k : ℝ) + 1) ^ (n + 1) * ‖continuousHarmonicCoordinates n F k‖ at he
  calc
    _ ≤ ((k : ℝ) + 1) ^ (n + 1) * ‖continuousHarmonicCoordinates n F k‖ := he
    _ ≤ ((k : ℝ) + 1) ^ (n + 1) * (A * Real.exp (-a * k)) :=
      mul_le_mul_of_nonneg_left hb (by positivity)
    _ = A * (((k : ℝ) + 1) ^ (n + 1) * Real.exp (-a * k)) := by ring
    _ ≤ A * ((((n + 1).factorial : ℝ) / (a / 2) ^ (n + 1) * Real.exp (a / 2)) *
      Real.exp (-(a / 2) * k)) :=
      mul_le_mul_of_nonneg_left (QIRadon.SeriesTail.polynomial_times_exponential_bound (n + 1) k ha) hA
    _ = C * r ^ k := by
      dsimp [C, r]
      rw [← Real.exp_nat_mul]
      have hexp : -(a / 2) * (k : ℝ) = (k : ℝ) * (-(a / 2)) := by ring
      rw [hexp]
      ring

/-- Sobolev weighted energy of the canonical coefficients, above the sharp
uniform embedding threshold, likewise gives the actual pointwise expansion. -/
theorem canonical_harmonic_pointwise_sobolev (m : ℕ) (F : C(UnitSphere (m + 1), ℝ))
    (s A : ℝ) (hs : ((m : ℝ) + 1) / 2 < s) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ ↦
      (((k : ℝ) + 1) ^ s * ‖continuousHarmonicCoordinates (m + 1) F k‖) ^ 2))
    (henergy : (∑' k : ℕ,
      (((k : ℝ) + 1) ^ s * ‖continuousHarmonicCoordinates (m + 1) F k‖) ^ 2) ≤ A ^ 2)
    (θ : UnitSphere (m + 1)) :
    F θ = ∑' k, harmonicEvaluation (m + 1) k θ (continuousHarmonicCoordinates (m + 1) F k) := by
  let u := fun k : ℕ ↦ ((k : ℝ) + 1) ^ s * ‖continuousHarmonicCoordinates (m + 1) F k‖
  let v := fun k : ℕ ↦ ((k : ℝ) + 1) ^ ((m : ℝ) / 2 - s)
  have hu : ∀ k, 0 ≤ u k := by intro k; dsimp [u]; positivity
  have hv : ∀ k, 0 ≤ v k := by intro k; dsimp [v]; positivity
  have hsv : Summable (fun k ↦ v k ^ 2) :=
    QIRadon.SobolevTail.summable_shape_squares (by linarith)
  have hh := QIRadon.SobolevTail.energy_times_shape_tail hu hv hA hsum hsv henergy 0
  let b := fun k ↦ harmonicGrowthConstant m * (u k * v k)
  have hb : ∀ k, 0 ≤ b k := fun k ↦
    mul_nonneg (harmonicGrowthConstant_nonneg m) (mul_nonneg (hu k) (hv k))
  have hbs : Summable b := hh.1.mul_left (harmonicGrowthConstant m)
  apply canonical_harmonic_pointwise_of_majorant (m + 1) F b hb hbs _ θ
  intro k x
  have hid : u k * v k = ((k : ℝ) + 1) ^ ((m : ℝ) / 2) *
      ‖continuousHarmonicCoordinates (m + 1) F k‖ := by
    dsimp [u, v]
    rw [mul_right_comm, ← Real.rpow_add (by positivity : 0 < (k : ℝ) + 1)]
    congr 2
    ring
  dsimp only [b]
  rw [hid]
  exact harmonicEvaluation_bound m k x (continuousHarmonicCoordinates (m + 1) F k)

end
end QIRadon.Gaussian
