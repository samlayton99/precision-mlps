import QIRadon.Quadrature.HarmonicSpace

open MeasureTheory
open scoped RealInnerProductSpace

namespace QIRadon.Gaussian

noncomputable section

theorem polynomialRepresentative_raise (n k : ℕ) (v : PolynomialL2Space n k) :
    polynomialRepresentative n (k + 2) (polynomialDegreeRaise n k v) =
      polynomialRepresentative n k v := by
  apply ContinuousMap.toLp_injective (normalizedSphere n) (p := 2) (𝕜 := ℝ)
  rw [toLp_polynomialRepresentative, toLp_polynomialRepresentative]
  rfl

theorem polynomialRotation_raise {n : ℕ} (U : Rotation n) (k : ℕ) (v : PolynomialL2Space n k) :
    polynomialRotation U (k + 2) (polynomialDegreeRaise n k v) =
      polynomialDegreeRaise n k (polynomialRotation U k v) := by
  have hrep : polynomialRepresentative n (k + 2)
      (polynomialRotation U (k + 2) (polynomialDegreeRaise n k v)) =
      polynomialRepresentative n (k + 2)
      (polynomialDegreeRaise n k (polynomialRotation U k v)) := by
    ext θ
    change polynomialRepresentative n (k + 2)
      (polynomialRotationLinear U (k + 2) (polynomialDegreeRaise n k v)) θ = _
    rw [polynomialRepresentative_rotation, polynomialRepresentative_raise,
      polynomialRepresentative_raise]
    exact (polynomialRepresentative_rotation U k v θ).symm
  have h := congrArg (ContinuousMap.toLp 2 (normalizedSphere n) ℝ) hrep
  rw [toLp_polynomialRepresentative, toLp_polynomialRepresentative] at h
  exact Subtype.ext h

theorem lowerDegreeSpace_rotation {n : ℕ} (U : Rotation n) (k : ℕ)
    (v : PolynomialL2Space n (k + 2)) (hv : v ∈ lowerDegreeSpace n k) :
    polynomialRotation U (k + 2) v ∈ lowerDegreeSpace n k := by
  obtain ⟨w, rfl⟩ := hv
  exact ⟨polynomialRotation U k w, (polynomialRotation_raise U k w).symm⟩

theorem lowerDegreeSpace_rotation_symm {n : ℕ} (U : Rotation n) (k : ℕ)
    (v : PolynomialL2Space n (k + 2)) (hv : v ∈ lowerDegreeSpace n k) :
    (polynomialRotation U (k + 2)).symm v ∈ lowerDegreeSpace n k := by
  obtain ⟨w, rfl⟩ := hv
  refine ⟨(polynomialRotation U k).symm w, ?_⟩
  apply (polynomialRotation U (k + 2)).injective
  rw [polynomialRotation_raise, LinearIsometryEquiv.apply_symm_apply,
    LinearIsometryEquiv.apply_symm_apply]

theorem isometry_preserves_orthogonal {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (S : Submodule ℝ E) (U : E ≃ₗᵢ[ℝ] E)
    (hinv : ∀ w ∈ S, U.symm w ∈ S) (v : E) (hv : v ∈ Sᗮ) : U v ∈ Sᗮ := by
  apply (S.mem_orthogonal (U v)).mpr
  intro w hw
  have h := U.inner_map_map (U.symm w) v
  rw [U.apply_symm_apply] at h
  rw [h]
  exact (S.mem_orthogonal v).mp hv (U.symm w) (hinv w hw)

theorem harmonicSubspace_rotation {n : ℕ} (U : Rotation n) (k : ℕ)
    (v : PolynomialL2Space n k) (hv : v ∈ harmonicSubspace n k) :
    polynomialRotation U k v ∈ harmonicSubspace n k := by
  cases k with
  | zero => trivial
  | succ k =>
    cases k with
    | zero => trivial
    | succ k =>
      exact isometry_preserves_orthogonal (lowerDegreeSpace n k) (polynomialRotation U (k + 2))
        (lowerDegreeSpace_rotation_symm U k) v hv

theorem harmonicSubspace_rotation_symm {n : ℕ} (U : Rotation n) (k : ℕ)
    (v : PolynomialL2Space n k) (hv : v ∈ harmonicSubspace n k) :
    (polynomialRotation U k).symm v ∈ harmonicSubspace n k := by
  cases k with
  | zero => trivial
  | succ k =>
    cases k with
    | zero => trivial
    | succ k =>
      exact isometry_preserves_orthogonal (lowerDegreeSpace n k) (polynomialRotation U (k + 2)).symm
        (by simpa using lowerDegreeSpace_rotation U k) v hv

def harmonicRotation {n : ℕ} (U : Rotation n) (k : ℕ) :
    HarmonicSpace n k ≃ₗᵢ[ℝ] HarmonicSpace n k where
  toFun v := ⟨polynomialRotation U k v.val, harmonicSubspace_rotation U k v.val v.property⟩
  invFun v := ⟨(polynomialRotation U k).symm v.val, harmonicSubspace_rotation_symm U k v.val v.property⟩
  left_inv v := Subtype.ext ((polynomialRotation U k).symm_apply_apply v.val)
  right_inv v := Subtype.ext ((polynomialRotation U k).apply_symm_apply v.val)
  map_add' v w := Subtype.ext ((polynomialRotation U k).map_add v.val w.val)
  map_smul' c v := Subtype.ext ((polynomialRotation U k).map_smul c v.val)
  norm_map' v := (polynomialRotation U k).norm_map v.val

theorem harmonicEvaluation_covariant (n k : ℕ) (x y : UnitSphere n) :
    ∃ U : HarmonicSpace n k ≃ₗᵢ[ℝ] HarmonicSpace n k,
      ∀ v, harmonicEvaluation n k y v = harmonicEvaluation n k x (U v) := by
  obtain ⟨U, hU⟩ := sphereRotation_transitive x y
  refine ⟨harmonicRotation U k, ?_⟩
  intro v
  change polynomialRepresentative n k v.val y =
    polynomialRepresentative n k (polynomialRotationLinear U k v.val) x
  rw [polynomialRepresentative_rotation, hU]

/-- The addition estimate with the sharp harmonic-layer dimension, obtained by
restricting the actual L2 rotation action to the actual orthogonal complement. -/
theorem harmonicEvaluation_bound_dimension (n k : ℕ) (θ : UnitSphere n) (v : HarmonicSpace n k) :
    |harmonicEvaluation n k θ v| ≤
      Real.sqrt (Module.finrank ℝ (HarmonicSpace n k)) * ‖v‖ := by
  have h := QIRadon.Addition.evaluation_bound
    (stdOrthonormalBasis ℝ (HarmonicSpace n k))
    (harmonicEvaluation n k) (normalizedSphere n)
    (fun w ↦ ⟨harmonicEvaluation_square_integrable n k w, harmonicEvaluation_L2 n k w⟩)
    (harmonicEvaluation_covariant n k) θ v
  simpa only [Fintype.card_fin] using h

end
end QIRadon.Gaussian
