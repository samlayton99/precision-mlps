import QIRadon.Quadrature.SphereLift
import QIRadon.Quadrature.BaseSphere
import QIRadon.Quadrature.JacobiScaledMoments

open Polynomial MeasureTheory Metric
open scoped BigOperators

namespace QIRadon.Gaussian

/-- Jacobi exponent when lifting from ambient dimension n+1 to n+2. -/
noncomputable def sphereJacobiExponent (n : ℕ) : ℝ := (n : ℝ) / 2 - 1 / 2

theorem sphereJacobiExponent_gt (n : ℕ) : -1 < sphereJacobiExponent n := by
  unfold sphereJacobiExponent
  have := Nat.cast_nonneg (α := ℝ) n
  linarith

/-- Exact even-monomial sphere recursion, proved by matching the actual Gaussian
sphere moments to the actual Beta/Jacobi integral, not assumed disintegration. -/
theorem sphere_even_moment_recursion (n : ℕ) (k : Fin (n + 2) → ℕ) :
    sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 2))) 1 ↦
      coordinateMonomial (fun i ↦ 2 * k i) θ.val) =
    jacobiMomentGeneral (sphereJacobiExponent n) (sphereJacobiExponent_gt n)
      (jacobiTestPolynomial (2 * k 0) (∑ i : Fin (n + 1), k i.succ)) *
    sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
      coordinateMonomial (fun i : Fin (n + 1) ↦ 2 * k i.succ) θ.val) := by
  rw [sphereAverage_even_monomial, sphereAverage_even_monomial,
    jacobiMomentGeneral_even_test]
  simp only [Fintype.card_fin, Fin.sum_univ_succ, Fin.prod_univ_succ,
    Nat.cast_add, Nat.cast_ofNat, Nat.cast_sum, sphereJacobiExponent]
  have hsum : (0 : ℝ) ≤ ∑ i : Fin (n + 1), (k i.succ : ℝ) :=
    Finset.sum_nonneg (fun i _ ↦ Nat.cast_nonneg _)
  have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hk : (0 : ℝ) ≤ k 0 := Nat.cast_nonneg _
  have h₀ : Real.Gamma (1 / 2) ≠ 0 := (Real.Gamma_pos_of_pos (by norm_num)).ne'
  have h₁ : Real.Gamma (((n : ℝ) + 1) / 2) ≠ 0 :=
    (Real.Gamma_pos_of_pos (by linarith)).ne'
  have h₂ : Real.Gamma ((∑ i : Fin (n + 1), (k i.succ : ℝ)) + ((n : ℝ) + 1) / 2) ≠ 0 :=
    (Real.Gamma_pos_of_pos (by linarith)).ne'
  have h₃ : Real.Gamma ((k 0 : ℝ) + (∑ i : Fin (n + 1), (k i.succ : ℝ)) +
      ((n : ℝ) + 2) / 2) ≠ 0 :=
    (Real.Gamma_pos_of_pos (by linarith)).ne'
  ring_nf at h₀ h₁ h₂ h₃ ⊢
  field_simp

end QIRadon.Gaussian
