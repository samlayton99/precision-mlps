import QIRadon.Quadrature.SphereMoments

/-!
# Exact two-point quadrature on the zero-dimensional sphere

This provides the base of a Jacobi recursion including the arcsine (a=-1/2)
step. The resulting circle rule must still be identified with the equally
spaced half-offset circle rule before claiming that algorithmic identification.
-/

open scoped BigOperators
open Metric

namespace QIRadon.Gaussian

def axisPoint (b : Bool) : EuclideanSpace ℝ (Fin 1) :=
  WithLp.toLp 2 (fun _ ↦ if b then 1 else -1)

theorem axisPoint_norm (b : Bool) : ‖axisPoint b‖ = 1 := by
  apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [EuclideanSpace.real_norm_sq_eq]
  cases b <;> norm_num [axisPoint]

theorem base_sphere_monomial_exact (k : Fin 1 → ℕ) :
    (∑ b : Bool, (1 / 2 : ℝ) * coordinateMonomial k (axisPoint b)) =
      sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin 1)) 1 ↦
        coordinateMonomial k θ.val) := by
  rcases Nat.even_or_odd (k 0) with heven | hodd
  · obtain ⟨l, hl⟩ := heven
    have hk : k = fun _ ↦ 2 * l := by
      funext i
      have hi : i = 0 := Fin.eq_zero i
      rw [hi, hl]
      omega
    rw [hk]
    have h := sphereAverage_even_monomial (ι := Fin 1) (fun _ ↦ l)
    have hg : Real.Gamma ((l : ℝ) + 1 / 2) ≠ 0 :=
      (Real.Gamma_pos_of_pos (by have := Nat.cast_nonneg (α := ℝ) l; linarith)).ne'
    have hg₀ : Real.Gamma (1 / 2) ≠ 0 := (Real.Gamma_pos_of_pos (by norm_num)).ne'
    simp only [Fin.sum_univ_one, Fin.prod_univ_one, Fintype.card_fin, Nat.cast_one, pow_one] at h
    rw [h]
    have hc : Real.Gamma (1 / 2) * Real.Gamma ((l : ℝ) + 1 / 2) /
        (Real.Gamma ((l : ℝ) + 1 / 2) * Real.Gamma (1 / 2)) = 1 := by
      field_simp
    rw [hc]
    have hn : (-1 : ℝ) ^ (2 * l) = 1 := by rw [pow_mul]; norm_num
    norm_num [coordinateMonomial, axisPoint, hn]
  · rw [sphereAverage_monomial_zero_of_odd k 0 hodd]
    obtain ⟨l, hl⟩ := hodd
    simp only [coordinateMonomial, Fin.prod_univ_one, Fintype.sum_bool, axisPoint]
    simp [hl, pow_add, pow_mul]

end QIRadon.Gaussian
