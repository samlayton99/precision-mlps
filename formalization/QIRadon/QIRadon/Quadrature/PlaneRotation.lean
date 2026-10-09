import QIRadon.Quadrature.SphereRotation
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-! Actual coordinate-plane rotations. The sign convention has infinitesimal
generator `x_i ∂_j - x_j ∂_i`. -/

open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

def planeRotate (i j : ι) (t : ℝ) (x : EuclideanSpace ℝ ι) : EuclideanSpace ℝ ι :=
  WithLp.toLp 2 (fun k ↦ if k = i then Real.cos t * x i - Real.sin t * x j
    else if k = j then Real.sin t * x i + Real.cos t * x j else x k)

omit [Fintype ι] in
theorem planeRotate_apply (i j k : ι) (t : ℝ) (x : EuclideanSpace ℝ ι) :
    planeRotate i j t x k = if k = i then Real.cos t * x i - Real.sin t * x j
    else if k = j then Real.sin t * x i + Real.cos t * x j else x k := rfl

theorem planeRotate_norm (i j : ι) (hij : i ≠ j) (t : ℝ) (x : EuclideanSpace ℝ ι) :
    ‖planeRotate i j t x‖ = ‖x‖ := by
  apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  apply sub_eq_zero.mp
  rw [← Finset.sum_sub_distrib]
  have hterm (k : ι) : (planeRotate i j t x k)^2 - (x k)^2 =
      (if k = i then (Real.cos t*x i-Real.sin t*x j)^2-(x i)^2 else 0) +
      (if k = j then (Real.sin t*x i+Real.cos t*x j)^2-(x j)^2 else 0) := by
    by_cases hki : k = i
    · subst k; simp [planeRotate_apply, hij]
    · by_cases hkj : k = j
      · subst k; simp [planeRotate_apply, hki]
      · simp [planeRotate_apply, hki, hkj]
  simp_rw [hterm]
  rw [Finset.sum_add_distrib]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true]
  nlinarith [Real.sin_sq_add_cos_sq t]

omit [Fintype ι] in
theorem planeRotate_inverse (i j : ι) (hij : i ≠ j) (t : ℝ)
    (x : EuclideanSpace ℝ ι) : planeRotate i j (-t) (planeRotate i j t x) = x := by
  ext k
  by_cases hki : k = i
  · subst k
    simp only [planeRotate_apply, if_true, hij.symm, if_false, Real.cos_neg, Real.sin_neg]
    calc
      _ = (Real.sin t ^ 2 + Real.cos t ^ 2) * x i := by ring
      _ = x i := by rw [Real.sin_sq_add_cos_sq]; ring
  · by_cases hkj : k = j
    · subst k
      simp only [planeRotate_apply, hki, if_false, if_true, Real.cos_neg, Real.sin_neg]
      calc
        _ = (Real.sin t ^ 2 + Real.cos t ^ 2) * x j := by ring
        _ = x j := by rw [Real.sin_sq_add_cos_sq]; ring
    · simp [planeRotate_apply, hki, hkj]

def planeRotation (i j : ι) (hij : i ≠ j) (t : ℝ) :
    EuclideanSpace ℝ ι ≃ₗᵢ[ℝ] EuclideanSpace ℝ ι where
  toFun := planeRotate i j t
  invFun := planeRotate i j (-t)
  map_add' x y := by
    ext k
    simp only [planeRotate_apply, PiLp.add_apply]
    split_ifs <;> ring
  map_smul' c x := by
    ext k
    simp only [planeRotate_apply, PiLp.smul_apply, smul_eq_mul, RingHom.id_apply]
    split_ifs <;> ring
  left_inv := planeRotate_inverse i j hij t
  right_inv x := by simpa only [neg_neg] using planeRotate_inverse i j hij (-t) x
  norm_map' := planeRotate_norm i j hij t

omit [Fintype ι] in
@[simp] theorem planeRotate_zero (i j : ι) (_hij : i ≠ j) (x : EuclideanSpace ℝ ι) :
    planeRotate i j 0 x = x := by
  ext k
  by_cases hki : k = i
  · subst k; simp [planeRotate_apply]
  · by_cases hkj : k = j
    · subst k; simp [planeRotate_apply, hki]
    · simp [planeRotate_apply, hki, hkj]

omit [Fintype ι] in
theorem planeRotate_hasDerivAt_coordinate (i j : ι) (hij : i ≠ j)
    (x : EuclideanSpace ℝ ι) (k : ι) (t : ℝ) :
    HasDerivAt (fun t ↦ planeRotate i j t x k)
      (if k = i then -planeRotate i j t x j
        else if k = j then planeRotate i j t x i else 0) t := by
  by_cases hki : k = i
  · subst k
    simp only [planeRotate_apply, if_true, hij.symm, if_false]
    convert ((Real.hasDerivAt_cos t).mul_const (x i)).sub
      ((Real.hasDerivAt_sin t).mul_const (x j)) using 1 <;> first | rfl | ring
  · by_cases hkj : k = j
    · subst k
      simp only [planeRotate_apply, hki, if_false, if_true]
      convert ((Real.hasDerivAt_sin t).mul_const (x i)).add
        ((Real.hasDerivAt_cos t).mul_const (x j)) using 1 <;> first | rfl | ring
    · simpa only [planeRotate_apply, hki, hkj, if_false] using hasDerivAt_const t (x k)

end
end QIRadon.Gaussian
