import QIRadon.Quadrature.SphereMeasure
import QIRadon.Quadrature.Addition
import Mathlib.RingTheory.MvPolynomial.Homogeneous
import Mathlib.Topology.Algebra.MvPolynomial
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Function.LpSpace.ContinuousFunctions

open MeasureTheory
open scoped RealInnerProductSpace

namespace QIRadon.Gaussian

noncomputable section

abbrev HomogeneousPolynomials (n k : ℕ) :=
  MvPolynomial.homogeneousSubmodule (Fin (n + 1)) ℝ k

instance homogeneousPolynomialsFinite (n k : ℕ) :
    Module.Finite ℝ (HomogeneousPolynomials n k) :=
  Module.Finite.of_fg (MvPolynomial.homogeneousSubmodule_fg (Fin (n + 1)) ℝ k)

/-- Restrict an actual homogeneous multivariate polynomial to the actual sphere. -/
noncomputable def homogeneousRestriction (n k : ℕ) :
    HomogeneousPolynomials n k →ₗ[ℝ] C(UnitSphere n, ℝ) where
  toFun p := ⟨fun θ ↦ MvPolynomial.eval (fun i ↦ θ.val i) p.val, by
    apply p.val.continuous_eval.comp
    fun_prop⟩
  map_add' p q := by ext θ; simp
  map_smul' c p := by ext θ; simp

abbrev ContinuousPolynomialSpace (n k : ℕ) := (homogeneousRestriction n k).range

instance continuousPolynomialSpaceFinite (n k : ℕ) :
    Module.Finite ℝ (ContinuousPolynomialSpace n k) :=
  Module.Finite.range (homogeneousRestriction n k)

noncomputable def polynomialL2Map (n k : ℕ) :
    ContinuousPolynomialSpace n k →ₗ[ℝ] Lp ℝ 2 (normalizedSphere n) :=
  (ContinuousMap.toLp 2 (normalizedSphere n) ℝ).toLinearMap.comp
    (ContinuousPolynomialSpace n k).subtype

theorem polynomialL2Map_injective (n k : ℕ) : Function.Injective (polynomialL2Map n k) :=
  (ContinuousMap.toLp_injective (normalizedSphere n)).comp Subtype.val_injective

abbrev PolynomialL2Space (n k : ℕ) := (polynomialL2Map n k).range

instance polynomialL2SpaceFinite (n k : ℕ) : Module.Finite ℝ (PolynomialL2Space n k) :=
  Module.Finite.range (polynomialL2Map n k)

noncomputable def polynomialL2Equiv (n k : ℕ) :
    ContinuousPolynomialSpace n k ≃ₗ[ℝ] PolynomialL2Space n k :=
  LinearEquiv.ofInjective (polynomialL2Map n k) (polynomialL2Map_injective n k)

/-- Recover the unique continuous polynomial representative of an L2 class. -/
noncomputable def polynomialRepresentative (n k : ℕ) :
    PolynomialL2Space n k →ₗ[ℝ] C(UnitSphere n, ℝ) :=
  (ContinuousPolynomialSpace n k).subtype.comp (polynomialL2Equiv n k).symm.toLinearMap

theorem toLp_polynomialRepresentative (n k : ℕ) (v : PolynomialL2Space n k) :
    ContinuousMap.toLp 2 (normalizedSphere n) ℝ (polynomialRepresentative n k v) = v.val := by
  exact congrArg Subtype.val ((polynomialL2Equiv n k).apply_symm_apply v)

noncomputable def polynomialEvaluation (n k : ℕ) (θ : UnitSphere n) :
    PolynomialL2Space n k →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := fun v ↦ polynomialRepresentative n k v θ
      map_add' := by intro v w; simp
      map_smul' := by intro c v; simp }

/-- The norm on this concrete finite-dimensional space is genuinely the sphere
L2 norm, not a coefficient norm asserted to be equivalent to it. -/
theorem polynomialEvaluation_L2 (n k : ℕ) (v : PolynomialL2Space n k) :
    (∫ θ, (polynomialEvaluation n k θ v) ^ 2 ∂normalizedSphere n) = ‖v‖ ^ 2 := by
  have h := ContinuousMap.inner_toLp (normalizedSphere n)
    (polynomialRepresentative n k v) (polynomialRepresentative n k v)
  rw [toLp_polynomialRepresentative] at h
  change (∫ θ, (polynomialRepresentative n k v θ) ^ 2 ∂normalizedSphere n) = _
  simpa only [RCLike.conj_to_real, real_inner_self_eq_norm_sq, pow_two, Submodule.norm_coe] using h.symm

theorem polynomialEvaluation_square_integrable (n k : ℕ) (v : PolynomialL2Space n k) :
    Integrable (fun θ ↦ (polynomialEvaluation n k θ v) ^ 2) (normalizedSphere n) := by
  have h : Continuous (fun θ ↦ (polynomialRepresentative n k v θ) ^ 2) := by fun_prop
  exact h.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)

end
end QIRadon.Gaussian
