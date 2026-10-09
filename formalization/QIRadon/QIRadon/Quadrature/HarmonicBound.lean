import QIRadon.Quadrature.HarmonicRotation
import QIRadon.Quadrature.HarmonicDimensionBound

namespace QIRadon.Gaussian

/-- Uniform sharp-exponent dimension bound in ambient dimension m+2. -/
theorem harmonic_finrank_bound (m k : ℕ) :
    Module.finrank ℝ (HarmonicSpace (m + 1) k) ≤ 2 * (m + 1) ^ m * (k + 1) ^ m := by
  cases k with
  | zero =>
    rw [harmonic_finrank_zero]
    simp only [Nat.zero_add, one_pow, mul_one]
    have hp := Nat.one_le_pow m (m + 1) (by omega)
    omega
  | succ k =>
    cases k with
    | zero =>
      rw [harmonic_finrank_one]
      cases m with
      | zero => norm_num
      | succ m =>
        change m + 3 ≤ 2 * (m + 2) ^ (m + 1) * 2 ^ (m + 1)
        have hp : m + 2 ≤ (m + 2) ^ (m + 1) := by
          rw [pow_succ]
          have hh := Nat.one_le_pow m (m + 2) (by omega)
          nlinarith
        have htwo : 1 ≤ 2 ^ (m + 1) := Nat.one_le_pow _ _ (by omega)
        have hmul : (m + 2) ^ (m + 1) ≤ (m + 2) ^ (m + 1) * 2 ^ (m + 1) := by
          nlinarith
        nlinarith
    | succ k =>
      have hdim := harmonic_finrank_add (m + 1) k
      have heq : Module.finrank ℝ (HarmonicSpace (m + 1) (k + 2)) =
          (m + k + 3).choose (k + 2) - (m + k + 1).choose k := by
        have h₁ : m + 1 + k = m + k + 1 := by omega
        have h₂ : m + 1 + (k + 2) = m + k + 3 := by omega
        rw [h₁, h₂] at hdim
        omega
      rw [heq]
      exact QIRadon.HarmonicDimensionBound.homogeneous_difference_bound m k

noncomputable def harmonicGrowthConstant (m : ℕ) : ℝ :=
  Real.sqrt ((2 * (m + 1) ^ m : ℕ) : ℝ)

theorem harmonicGrowthConstant_nonneg (m : ℕ) : 0 ≤ harmonicGrowthConstant m :=
  Real.sqrt_nonneg _

/-- Sharp pointwise growth exponent (ambient dimension minus two)/2, with an
explicit dimension-only constant and an actual L2 norm. -/
theorem harmonicEvaluation_bound (m k : ℕ) (θ : UnitSphere (m + 1))
    (v : HarmonicSpace (m + 1) k) :
    |harmonicEvaluation (m + 1) k θ v| ≤
      harmonicGrowthConstant m * (((k : ℝ) + 1) ^ ((m : ℝ) / 2) * ‖v‖) := by
  have hd : (Module.finrank ℝ (HarmonicSpace (m + 1) k) : ℝ) ≤
      ((2 * (m + 1) ^ m : ℕ) : ℝ) * ((k : ℝ) + 1) ^ m := by
    exact_mod_cast harmonic_finrank_bound m k
  have hs := Real.sqrt_le_sqrt hd
  rw [Real.sqrt_mul (by positivity)] at hs
  have hp : Real.sqrt (((k : ℝ) + 1) ^ m) = ((k : ℝ) + 1) ^ ((m : ℝ) / 2) := by
    rw [Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]
    congr 1
    ring
  rw [hp] at hs
  have h := (harmonicEvaluation_bound_dimension (m + 1) k θ v).trans
    (mul_le_mul_of_nonneg_right hs (norm_nonneg v))
  simpa only [harmonicGrowthConstant, mul_assoc] using h

end QIRadon.Gaussian
