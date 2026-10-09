import QIRadon.Quadrature.SphereMeasure
import Mathlib.Analysis.InnerProductSpace.Projection.Reflection

open MeasureTheory Set Metric
open scoped Pointwise

namespace QIRadon.Gaussian

noncomputable def sphereRotation {n : ℕ}
    (U : EuclideanSpace ℝ (Fin (n + 1)) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin (n + 1))) :
    UnitSphere n ≃ₜ UnitSphere n where
  toFun θ := ⟨U θ.val, by
    change ‖U θ.val - 0‖ = 1
    simp⟩
  invFun θ := ⟨U.symm θ.val, by
    change ‖U.symm θ.val - 0‖ = 1
    simp⟩
  left_inv θ := Subtype.ext (U.symm_apply_apply θ.val)
  right_inv θ := Subtype.ext (U.apply_symm_apply θ.val)
  continuous_toFun := by fun_prop
  continuous_invFun := by fun_prop

theorem sphereRotation_cone_preimage {n : ℕ}
    (U : EuclideanSpace ℝ (Fin (n + 1)) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin (n + 1)))
    (s : Set (UnitSphere n)) :
    Ioo (0 : ℝ) 1 • (Subtype.val '' ((sphereRotation U) ⁻¹' s)) =
      U ⁻¹' (Ioo (0 : ℝ) 1 • (Subtype.val '' s)) := by
  ext x
  constructor
  · rintro ⟨r, hr, y, ⟨θ, hθ, rfl⟩, rfl⟩
    exact ⟨r, hr, U θ.val, ⟨sphereRotation U θ, hθ, rfl⟩, (U.map_smul r θ.val).symm⟩
  · rintro ⟨r, hr, y, ⟨θ, hθ, rfl⟩, hx⟩
    refine ⟨r, hr, U.symm θ.val,
      ⟨(sphereRotation U).symm θ, ?_, rfl⟩, ?_⟩
    · simpa using hθ
    · have h := congrArg U.symm hx
      simpa using h

theorem sphereRotation_measurePreserving_surface {n : ℕ}
    (U : EuclideanSpace ℝ (Fin (n + 1)) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin (n + 1))) :
    MeasurePreserving (sphereRotation U)
      ((volume : Measure (EuclideanSpace ℝ (Fin (n + 1)))).toSphere)
      ((volume : Measure (EuclideanSpace ℝ (Fin (n + 1)))).toSphere) := by
  refine ⟨(sphereRotation U).measurable, ?_⟩
  apply Measure.ext
  intro s hs
  rw [Measure.map_apply (sphereRotation U).measurable hs,
    Measure.toSphere_apply' _ ((sphereRotation U).measurable hs),
    Measure.toSphere_apply' _ hs, sphereRotation_cone_preimage]
  rw [U.measurePreserving.measure_preimage_emb U.toHomeomorph.measurableEmbedding]

theorem sphereRotation_measurePreserving {n : ℕ}
    (U : EuclideanSpace ℝ (Fin (n + 1)) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin (n + 1))) :
    MeasurePreserving (sphereRotation U) (normalizedSphere n) (normalizedSphere n) :=
  (sphereRotation_measurePreserving_surface U).smul_measure _

/-- Actual transitivity of the orthogonal action on the Euclidean unit sphere. -/
theorem sphereRotation_transitive {n : ℕ} (x y : UnitSphere n) :
    ∃ U : EuclideanSpace ℝ (Fin (n + 1)) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin (n + 1)),
      sphereRotation U x = y := by
  let U := (ℝ ∙ (x.val - y.val))ᗮ.reflection
  refine ⟨U, Subtype.ext ?_⟩
  apply Submodule.reflection_sub
  have hx : ‖x.val‖ = 1 := by simp
  have hy : ‖y.val‖ = 1 := by simp
  exact hx.trans hy.symm

end QIRadon.Gaussian
