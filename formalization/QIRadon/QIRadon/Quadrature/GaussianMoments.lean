import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Tactic

/-!
# Gaussian radial moments used to identify spherical polynomial moments
-/

open MeasureTheory Set Real

namespace QIRadon.Gaussian

/-- The radial Gaussian moment, with arbitrary real exponent above -1. -/
theorem integral_rpow_gaussian_Ioi (s : ℝ) (hs : -1 < s) :
    (∫ x in Ioi (0 : ℝ), x ^ s * Real.exp (-(x ^ 2))) =
      Real.Gamma ((s + 1) / 2) / 2 := by
  have hs' : 0 < (s + 1) / 2 := by linarith
  have h := Real.Gamma_eq_integral hs'
  rw [← integral_comp_rpow_Ioi_of_pos (by norm_num : (0 : ℝ) < 2)] at h
  have heq :
      (∫ x in Ioi (0 : ℝ), (2 * x ^ ((2 : ℝ) - 1)) •
        (Real.exp (-(x ^ (2 : ℝ))) *
          (x ^ (2 : ℝ)) ^ ((s + 1) / 2 - 1))) =
      2 * ∫ x in Ioi (0 : ℝ), x ^ s * Real.exp (-(x ^ 2)) := by
    rw [← integral_const_mul]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro x hx
    dsimp only
    have hxpos : 0 < x := hx
    rw [← Real.rpow_mul hxpos.le]
    have hm : (2 : ℝ) * ((s + 1) / 2 - 1) = s - 1 := by ring
    rw [hm, Real.rpow_two, smul_eq_mul]
    norm_num only [show (2 : ℝ) - 1 = 1 by norm_num, Real.rpow_one]
    have hxpow : x * x ^ (s - 1) = x ^ s := by
      calc
        x * x ^ (s - 1) = x ^ (1 : ℝ) * x ^ (s - 1) := by rw [Real.rpow_one]
        _ = x ^ ((1 : ℝ) + (s - 1)) := (Real.rpow_add hxpos _ _).symm
        _ = x ^ s := by congr 1; ring
    calc
      (2 * x) * (Real.exp (-(x ^ 2)) * x ^ (s - 1)) =
          2 * (x * x ^ (s - 1)) * Real.exp (-(x ^ 2)) := by ring
      _ = 2 * (x ^ s * Real.exp (-(x ^ 2))) := by rw [hxpow]; ring
  rw [heq] at h
  linarith

/-- Natural-exponent version for the homogeneous-degree radial factor. -/
theorem integral_pow_gaussian_Ioi (n : ℕ) :
    (∫ x in Ioi (0 : ℝ), x ^ n * Real.exp (-(x ^ 2))) =
      Real.Gamma (((n : ℝ) + 1) / 2) / 2 := by
  have h := integral_rpow_gaussian_Ioi (n : ℝ)
    (lt_of_lt_of_le (by norm_num : (-1 : ℝ) < 0) (Nat.cast_nonneg n))
  simpa only [Real.rpow_natCast] using h

/-- Full-line even Gaussian moments. -/
theorem integral_even_pow_gaussian (k : ℕ) :
    (∫ x : ℝ, x ^ (2 * k) * Real.exp (-(x ^ 2))) =
      Real.Gamma ((k : ℝ) + 1 / 2) := by
  have h := integral_comp_abs (f := fun x : ℝ ↦ x ^ (2 * k) * Real.exp (-(x ^ 2)))
  simp only [(even_two_mul k).pow_abs, sq_abs] at h
  rw [h, integral_pow_gaussian_Ioi]
  have he : (((2 * k : ℕ) : ℝ) + 1) / 2 = (k : ℝ) + 1 / 2 := by push_cast; ring
  rw [he]
  ring

/-- Full-line odd Gaussian moments vanish by reflection. -/
theorem integral_odd_pow_gaussian (k : ℕ) :
    (∫ x : ℝ, x ^ (2 * k + 1) * Real.exp (-(x ^ 2))) = 0 := by
  have h := MeasureTheory.integral_neg_eq_self
    (fun x : ℝ ↦ x ^ (2 * k + 1) * Real.exp (-(x ^ 2))) volume
  have heq : (fun x : ℝ ↦ (-x) ^ (2 * k + 1) * Real.exp (-((-x) ^ 2))) =
      (fun x : ℝ ↦ -(x ^ (2 * k + 1) * Real.exp (-(x ^ 2)))) := by
    funext x
    simp [pow_add, pow_mul]
  change (∫ x : ℝ, (-x) ^ (2 * k + 1) * Real.exp (-((-x) ^ 2))) = _ at h
  rw [heq, integral_neg] at h
  linarith

/-- Integrability of all natural Gaussian moments, needed for product Fubini. -/
theorem integrable_pow_gaussian (n : ℕ) :
    Integrable (fun x : ℝ ↦ x ^ n * Real.exp (-(x ^ 2))) := by
  have h := integrable_rpow_mul_exp_neg_mul_sq
    (b := 1) (by norm_num) (s := (n : ℝ))
    (lt_of_lt_of_le (by norm_num : (-1 : ℝ) < 0) (Nat.cast_nonneg n))
  simpa only [Real.rpow_natCast, neg_mul, one_mul] using h

end QIRadon.Gaussian
