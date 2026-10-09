import QIRadon.Quadrature.JacobiGeneral

open Polynomial MeasureTheory

namespace QIRadon.Gaussian

noncomputable def jacobiTestPolynomial (h k : ℕ) : ℝ[X] := X ^ h * (1 - X ^ 2) ^ k

theorem jacobiTestPolynomial_degree (h k : ℕ) :
    (jacobiTestPolynomial h k).natDegree ≤ h + 2 * k := by
  have hd : (1 - (X : ℝ[X]) ^ 2).natDegree ≤ 2 := by
    have h := Polynomial.natDegree_sub_le (1 : ℝ[X]) (X ^ 2)
    simpa only [Polynomial.natDegree_one, Polynomial.natDegree_X_pow, Nat.zero_max] using h
  calc
    _ ≤ (X ^ h : ℝ[X]).natDegree + ((1 - (X : ℝ[X]) ^ 2) ^ k).natDegree :=
      Polynomial.natDegree_mul_le
    _ = h + k * (1 - (X : ℝ[X]) ^ 2).natDegree := by
      simp only [Polynomial.natDegree_pow, Polynomial.natDegree_X, mul_one]
    _ ≤ h + 2 * k := by
      simpa [Nat.mul_comm] using Nat.add_le_add_left (Nat.mul_le_mul_left k hd) h

theorem rawJacobiMomentGeneral_test (a : ℝ) (ha : -1 < a) (h k : ℕ) :
    rawJacobiMomentGeneral a ha (jacobiTestPolynomial h k) =
      ∫ x in (-1 : ℝ)..1, x ^ h * (1 - x ^ 2) ^ (a + (k : ℝ)) := by
  change (∫ x in (-1 : ℝ)..1,
    jacobiDensity a x * (jacobiTestPolynomial h k).eval x) = _
  apply intervalIntegral.integral_congr_uIoo
  intro x hx
  have hx' : x ∈ Set.Ioo (-1 : ℝ) 1 := by simpa using hx
  have hrad : 0 < 1 - x ^ 2 := by nlinarith [hx'.1, hx'.2]
  dsimp [jacobiDensity, jacobiTestPolynomial]
  rw [Real.rpow_add hrad, Real.rpow_natCast]
  simp only [Polynomial.eval_mul, Polynomial.eval_pow, Polynomial.eval_X,
    Polynomial.eval_sub, Polynomial.eval_one]
  ring

theorem rawJacobiMomentGeneral_mass (a : ℝ) (ha : -1 < a) :
    rawJacobiMomentGeneral a ha 1 =
      Real.Gamma (1 / 2) * Real.Gamma (a + 1) / Real.Gamma (a + 3 / 2) := by
  have h := QIRadon.Quadrature.jacobi_even_moment 0 ha
  simpa [rawJacobiMomentGeneral, jacobiDensity] using h

theorem jacobiMomentGeneral_even_test (a : ℝ) (ha : -1 < a) (j k : ℕ) :
    jacobiMomentGeneral a ha (jacobiTestPolynomial (2 * j) k) =
      Real.Gamma ((j : ℝ) + 1 / 2) * Real.Gamma (a + (k : ℝ) + 1) *
        Real.Gamma (a + 3 / 2) /
      (Real.Gamma ((j : ℝ) + (a + (k : ℝ)) + 3 / 2) *
        Real.Gamma (1 / 2) * Real.Gamma (a + 1)) := by
  have hak : -1 < a + (k : ℝ) := by have := Nat.cast_nonneg (α := ℝ) k; linarith
  change (rawJacobiMomentGeneral a ha 1)⁻¹ *
    rawJacobiMomentGeneral a ha (jacobiTestPolynomial (2 * j) k) = _
  rw [rawJacobiMomentGeneral_mass, rawJacobiMomentGeneral_test,
    QIRadon.Quadrature.jacobi_even_moment j hak]
  have hg₀ : Real.Gamma (1 / 2) ≠ 0 := (Real.Gamma_pos_of_pos (by norm_num)).ne'
  have hg₁ : Real.Gamma (a + 1) ≠ 0 := (Real.Gamma_pos_of_pos (by linarith)).ne'
  have hg₂ : Real.Gamma (a + 3 / 2) ≠ 0 := (Real.Gamma_pos_of_pos (by linarith)).ne'
  have hg₃ : Real.Gamma ((j : ℝ) + (a + (k : ℝ)) + 3 / 2) ≠ 0 :=
    (Real.Gamma_pos_of_pos (by have := Nat.cast_nonneg (α := ℝ) j; linarith)).ne'
  field_simp

theorem jacobiMomentGeneral_odd_test (a : ℝ) (ha : -1 < a) (j k : ℕ) :
    jacobiMomentGeneral a ha (jacobiTestPolynomial (2 * j + 1) k) = 0 := by
  change (rawJacobiMomentGeneral a ha 1)⁻¹ *
    rawJacobiMomentGeneral a ha (jacobiTestPolynomial (2 * j + 1) k) = _
  rw [rawJacobiMomentGeneral_test, QIRadon.Quadrature.jacobi_odd_moment, mul_zero]

end QIRadon.Gaussian
