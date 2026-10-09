import QIRadon.Quadrature.SphereMoments

/-!
# Recursive coordinates for the actual unit sphere
-/

open scoped BigOperators
open Metric

namespace QIRadon.Gaussian

/-- Append the Jacobi coordinate to a scaled lower-dimensional direction. -/
noncomputable def liftPoint {n : ℕ} (t : ℝ) (v : EuclideanSpace ℝ (Fin n)) :
    EuclideanSpace ℝ (Fin (n + 1)) :=
  WithLp.toLp 2 (Fin.cons t (fun i ↦ Real.sqrt (1 - t ^ 2) * v i))

@[simp] theorem liftPoint_zero {n : ℕ} (t : ℝ) (v : EuclideanSpace ℝ (Fin n)) :
    liftPoint t v 0 = t := rfl

@[simp] theorem liftPoint_succ {n : ℕ} (t : ℝ) (v : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    liftPoint t v i.succ = Real.sqrt (1 - t ^ 2) * v i := rfl

/-- This is genuinely a sphere point, rather than a formal tuple of coordinates. -/
theorem norm_liftPoint {n : ℕ} (t : ℝ) (ht : t ∈ Set.Icc (-1 : ℝ) 1)
    (v : EuclideanSpace ℝ (Fin n)) (hv : ‖v‖ = 1) :
    ‖liftPoint t v‖ = 1 := by
  have hrad : 0 ≤ 1 - t ^ 2 := by nlinarith [ht.1, ht.2]
  have hvsum : ∑ i, v i ^ 2 = 1 := by
    rw [← EuclideanSpace.real_norm_sq_eq, hv]
    norm_num
  apply (sq_eq_sq₀ (norm_nonneg _) zero_le_one).mp
  rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_succ]
  simp only [liftPoint_zero, liftPoint_succ, mul_pow]
  rw [← Finset.mul_sum, Real.sq_sqrt hrad, hvsum]
  ring

/-- Evaluation of any monomial under the recursive coordinate map. -/
theorem monomial_liftPoint {n : ℕ} (t : ℝ) (v : EuclideanSpace ℝ (Fin n))
    (k : Fin (n + 1) → ℕ) :
    coordinateMonomial k (liftPoint t v) =
      t ^ k 0 * Real.sqrt (1 - t ^ 2) ^ (∑ i : Fin n, k i.succ) *
        coordinateMonomial (fun i : Fin n ↦ k i.succ) v := by
  simp only [coordinateMonomial, Fin.prod_univ_succ, liftPoint_zero, liftPoint_succ,
    mul_pow, Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]
  ring

/-- For even exponents the square root disappears, leaving a polynomial in the
Jacobi coordinate whose degree is precisely the total monomial degree. -/
theorem even_monomial_liftPoint {n : ℕ} (t : ℝ) (ht : t ∈ Set.Icc (-1 : ℝ) 1)
    (v : EuclideanSpace ℝ (Fin n)) (k : Fin (n + 1) → ℕ) :
    coordinateMonomial (fun i ↦ 2 * k i) (liftPoint t v) =
      (t ^ (2 * k 0) * (1 - t ^ 2) ^ (∑ i : Fin n, k i.succ)) *
        coordinateMonomial (fun i : Fin n ↦ 2 * k i.succ) v := by
  rw [monomial_liftPoint, ← Finset.mul_sum,
    pow_mul (Real.sqrt (1 - t ^ 2)) 2 (∑ i : Fin n, k i.succ),
    Real.sq_sqrt (by nlinarith [ht.1, ht.2] : 0 ≤ 1 - t ^ 2)]

end QIRadon.Gaussian
