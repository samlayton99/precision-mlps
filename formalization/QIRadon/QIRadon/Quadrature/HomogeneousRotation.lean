import QIRadon.Quadrature.HomogeneousSpace
import QIRadon.Quadrature.SphereRotation

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section

variable {n : ℕ}
abbrev Rotation (n : ℕ) :=
  EuclideanSpace ℝ (Fin (n + 1)) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin (n + 1))

def linearCoordinatePolynomial (U : Rotation n) (i : Fin (n + 1)) :
    MvPolynomial (Fin (n + 1)) ℝ :=
  ∑ j, (U (EuclideanSpace.basisFun (Fin (n + 1)) ℝ j) i) • MvPolynomial.X j

theorem linearCoordinatePolynomial_homogeneous (U : Rotation n) (i : Fin (n + 1)) :
    (linearCoordinatePolynomial U i).IsHomogeneous 1 := by
  change linearCoordinatePolynomial U i ∈ MvPolynomial.homogeneousSubmodule _ ℝ 1
  apply Submodule.sum_mem
  intro j _
  apply Submodule.smul_mem
  exact MvPolynomial.isHomogeneous_X ℝ j

theorem linearCoordinatePolynomial_eval (U : Rotation n) (i : Fin (n + 1))
    (x : EuclideanSpace ℝ (Fin (n + 1))) :
    MvPolynomial.eval (fun j ↦ x j) (linearCoordinatePolynomial U i) = U x i := by
  have h := congrArg (fun y : EuclideanSpace ℝ (Fin (n + 1)) ↦ U y i)
    ((EuclideanSpace.basisFun (Fin (n + 1)) ℝ).sum_repr x)
  simp only [map_sum, map_smul, EuclideanSpace.basisFun_repr] at h
  simpa [linearCoordinatePolynomial, mul_comm] using h

def rotatePolynomial (U : Rotation n) (p : MvPolynomial (Fin (n + 1)) ℝ) :
    MvPolynomial (Fin (n + 1)) ℝ := MvPolynomial.aeval (linearCoordinatePolynomial U) p

theorem rotatePolynomial_homogeneous (U : Rotation n) {k : ℕ}
    (p : HomogeneousPolynomials n k) : (rotatePolynomial U p.val).IsHomogeneous k := by
  simpa only [rotatePolynomial, one_mul] using p.property.aeval
    (linearCoordinatePolynomial U) (linearCoordinatePolynomial_homogeneous U)

theorem rotatePolynomial_eval (U : Rotation n) (p : MvPolynomial (Fin (n + 1)) ℝ)
    (x : EuclideanSpace ℝ (Fin (n + 1))) :
    MvPolynomial.eval (fun j ↦ x j) (rotatePolynomial U p) =
      MvPolynomial.eval (fun j ↦ U x j) p := by
  unfold rotatePolynomial
  induction p using MvPolynomial.induction_on with
  | C c => simp
  | add p q hp hq => simp only [map_add, hp, hq]
  | mul_X p i hp =>
    simp only [map_mul, MvPolynomial.aeval_X, hp,
      MvPolynomial.eval_X, linearCoordinatePolynomial_eval]

def continuousRotation (U : Rotation n) : C(UnitSphere n, ℝ) →ₗ[ℝ] C(UnitSphere n, ℝ) where
  toFun f := ⟨fun θ ↦ f (sphereRotation U θ), f.continuous.comp (sphereRotation U).continuous⟩
  map_add' := by intro f g; ext θ; rfl
  map_smul' := by intro c f; ext θ; rfl

theorem continuousRotation_preserves (U : Rotation n) (k : ℕ)
    (f : ContinuousPolynomialSpace n k) :
    continuousRotation U f.val ∈ ContinuousPolynomialSpace n k := by
  obtain ⟨p, hp⟩ := f.property
  refine ⟨⟨rotatePolynomial U p.val, rotatePolynomial_homogeneous U p⟩, ?_⟩
  ext θ
  change MvPolynomial.eval (fun i ↦ θ.val i) (rotatePolynomial U p.val) = _
  rw [rotatePolynomial_eval]
  have h := congrArg (fun g : C(UnitSphere n, ℝ) ↦ g (sphereRotation U θ)) hp
  exact h

def continuousPolynomialRotation (U : Rotation n) (k : ℕ) :
    ContinuousPolynomialSpace n k ≃ₗ[ℝ] ContinuousPolynomialSpace n k where
  toFun f := ⟨continuousRotation U f.val, continuousRotation_preserves U k f⟩
  invFun f := ⟨continuousRotation U.symm f.val, continuousRotation_preserves U.symm k f⟩
  map_add' := by intro f g; apply Subtype.ext; ext θ; rfl
  map_smul' := by intro c f; apply Subtype.ext; ext θ; rfl
  left_inv f := by
    apply Subtype.ext
    ext θ
    change f.val (sphereRotation U (sphereRotation U.symm θ)) = f.val θ
    congr 1
    exact Subtype.ext (U.apply_symm_apply θ.val)
  right_inv f := by
    apply Subtype.ext
    ext θ
    change f.val (sphereRotation U.symm (sphereRotation U θ)) = f.val θ
    congr 1
    exact Subtype.ext (U.symm_apply_apply θ.val)

def polynomialRotationLinear (U : Rotation n) (k : ℕ) :
    PolynomialL2Space n k ≃ₗ[ℝ] PolynomialL2Space n k :=
  (polynomialL2Equiv n k).symm.trans
    ((continuousPolynomialRotation U k).trans (polynomialL2Equiv n k))

theorem polynomialRepresentative_rotation (U : Rotation n) (k : ℕ)
    (v : PolynomialL2Space n k) (θ : UnitSphere n) :
    polynomialRepresentative n k (polynomialRotationLinear U k v) θ =
      polynomialRepresentative n k v (sphereRotation U θ) := by
  change ((polynomialL2Equiv n k).symm
    ((polynomialL2Equiv n k) ((continuousPolynomialRotation U k)
      ((polynomialL2Equiv n k).symm v)))).val θ = _
  rw [LinearEquiv.symm_apply_apply]
  rfl

theorem polynomialRotation_norm (U : Rotation n) (k : ℕ) (v : PolynomialL2Space n k) :
    ‖polynomialRotationLinear U k v‖ = ‖v‖ := by
  apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [← polynomialEvaluation_L2 n k (polynomialRotationLinear U k v),
    ← polynomialEvaluation_L2 n k v]
  change (∫ θ, (polynomialRepresentative n k (polynomialRotationLinear U k v) θ) ^ 2
    ∂normalizedSphere n) = _
  simp_rw [polynomialRepresentative_rotation]
  exact (sphereRotation_measurePreserving U).integral_comp
    (sphereRotation U).measurableEmbedding
    (fun θ ↦ (polynomialRepresentative n k v θ) ^ 2)

def polynomialRotation (U : Rotation n) (k : ℕ) :
    PolynomialL2Space n k ≃ₗᵢ[ℝ] PolynomialL2Space n k where
  __ := polynomialRotationLinear U k
  norm_map' := polynomialRotation_norm U k

/-- All structural hypotheses of the addition estimate are verified for the
actual homogeneous polynomial restriction space. -/
theorem polynomialEvaluation_covariant (n k : ℕ) (x y : UnitSphere n) :
    ∃ U : PolynomialL2Space n k ≃ₗᵢ[ℝ] PolynomialL2Space n k,
      ∀ v, polynomialEvaluation n k y v = polynomialEvaluation n k x (U v) := by
  obtain ⟨U, hU⟩ := sphereRotation_transitive x y
  refine ⟨polynomialRotation U k, ?_⟩
  intro v
  change polynomialRepresentative n k v y =
    polynomialRepresentative n k (polynomialRotationLinear U k v) x
  rw [polynomialRepresentative_rotation, hU]

end
end QIRadon.Gaussian
