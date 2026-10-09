import QIRadon.Quadrature.RotationFlow
import QIRadon.Quadrature.CasimirJet
import Mathlib.Analysis.Calculus.ContDiff.WithLp
import Mathlib.Analysis.Calculus.FDeriv.WithLp

/-! Construct actual continuous rotation derivatives from a smooth ambient
 extension on an open neighborhood of the real sphere. Quantitative Cauchy
 bounds and existence of this extension are separate obligations. -/

open scoped ContDiff

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false

open Gaussian

variable {n : ℕ}

def realRotationGeneratorLinear (i j : Fin (n+1)) :
    EuclideanSpace ℝ (Fin (n+1)) →ₗ[ℝ] EuclideanSpace ℝ (Fin (n+1)) where
  toFun x := if i=j then 0 else WithLp.toLp 2
    (fun k ↦ if k=i then -x j else if k=j then x i else 0)
  map_add' x y := by
    by_cases hij : i=j
    · simp [hij]
    · ext k
      simp only [hij, if_false, PiLp.add_apply, WithLp.toLp_ofLp]
      split_ifs <;> simp [add_comm]
  map_smul' c x := by
    by_cases hij : i=j
    · simp [hij]
    · ext k
      simp only [hij, if_false, PiLp.smul_apply, WithLp.toLp_ofLp, RingHom.id_apply, smul_eq_mul]
      split_ifs <;> simp

def realRotationGenerator (i j : Fin (n+1)) :
    EuclideanSpace ℝ (Fin (n+1)) →L[ℝ] EuclideanSpace ℝ (Fin (n+1)) :=
  (realRotationGeneratorLinear i j).toContinuousLinearMap

def ambientRotationDerivative (i j : Fin (n+1))
    (F : EuclideanSpace ℝ (Fin (n+1)) → ℝ) (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  fderiv ℝ F x (realRotationGenerator i j x)

theorem ambientRotationDerivative_contDiffOn {S : Set (EuclideanSpace ℝ (Fin (n+1)))}
    (hS : IsOpen S) {F : EuclideanSpace ℝ (Fin (n+1)) → ℝ}
    (hF : ContDiffOn ℝ ∞ F S) (i j : Fin (n+1)) :
    ContDiffOn ℝ ∞ (ambientRotationDerivative i j F) S :=
  (hF.fderiv_of_isOpen hS (by simp)).clm_apply (realRotationGenerator i j).contDiff.contDiffOn

def ambientRotationWord (F : EuclideanSpace ℝ (Fin (n+1)) → ℝ) :
    List (RotationPair n) → EuclideanSpace ℝ (Fin (n+1)) → ℝ
  | [] => F
  | p::w => ambientRotationDerivative p.1 p.2 (ambientRotationWord F w)

theorem ambientRotationWord_contDiffOn {S : Set (EuclideanSpace ℝ (Fin (n+1)))}
    (hS : IsOpen S) {F : EuclideanSpace ℝ (Fin (n+1)) → ℝ}
    (hF : ContDiffOn ℝ ∞ F S) (w : List (RotationPair n)) :
    ContDiffOn ℝ ∞ (ambientRotationWord F w) S := by
  induction w with
  | nil => exact hF
  | cons p w ih => exact ambientRotationDerivative_contDiffOn hS ih p.1 p.2

def restrictSmoothSphere {S : Set (EuclideanSpace ℝ (Fin (n+1)))}
    (hS : IsOpen S) (hcontains : ∀ θ : UnitSphere n, θ.val ∈ S)
    (F : EuclideanSpace ℝ (Fin (n+1)) → ℝ) (hF : ContDiffOn ℝ ∞ F S) :
    C(UnitSphere n, ℝ) where
  toFun θ := F θ.val
  continuous_toFun := hF.continuousOn.comp_continuous continuous_subtype_val hcontains

def ambientJet {S : Set (EuclideanSpace ℝ (Fin (n+1)))}
    (hS : IsOpen S) (hcontains : ∀ θ : UnitSphere n, θ.val ∈ S)
    (F : EuclideanSpace ℝ (Fin (n+1)) → ℝ) (hF : ContDiffOn ℝ ∞ F S) : RotationJet n :=
  fun w ↦ restrictSmoothSphere hS hcontains (ambientRotationWord F w)
    (ambientRotationWord_contDiffOn hS hF w)


theorem rotationFlow_hasDerivAt_zero (i j : Fin (n+1)) (θ : UnitSphere n) :
    HasDerivAt (fun t ↦ (rotationFlow i j t θ).val) (realRotationGenerator i j θ.val) 0 := by
  by_cases hij : i = j
  · subst j
    simpa [rotationFlow, realRotationGenerator, realRotationGeneratorLinear] using
      hasDerivAt_const 0 θ.val
  · have hpi : HasDerivAt (fun t k ↦ planeRotate i j t θ.val k)
        (fun k ↦ if k=i then -θ.val j else if k=j then θ.val i else 0) 0 := by
      apply hasDerivAt_pi.mpr
      intro k
      simpa only [planeRotate_zero i j hij] using
        planeRotate_hasDerivAt_coordinate i j hij θ.val k 0
    have h := (PiLp.hasFDerivAt_toLp (𝕜 := ℝ) 2
      (fun k ↦ planeRotate i j 0 θ.val k)).comp_hasDerivAt 0 hpi
    simp only [rotationFlow, dif_neg hij]
    convert h using 1 <;> first | rfl | simp [realRotationGenerator, realRotationGeneratorLinear, hij]

theorem ambientRotationDerivative_along_flow
    {S : Set (EuclideanSpace ℝ (Fin (n+1)))}
    (hS : IsOpen S) (hcontains : ∀ θ : UnitSphere n, θ.val ∈ S)
    (F : EuclideanSpace ℝ (Fin (n+1)) → ℝ) (hF : ContDiffOn ℝ ∞ F S)
    (i j : Fin (n+1)) :
    HasRotationDerivative i j (restrictSmoothSphere hS hcontains F hF)
      (restrictSmoothSphere hS hcontains (ambientRotationDerivative i j F)
        (ambientRotationDerivative_contDiffOn hS hF i j)) := by
  apply hasRotationDerivative_of_zero
  intro θ
  have hf : DifferentiableAt ℝ F θ.val :=
    (hF.differentiableOn (by simp) θ.val (hcontains θ)).differentiableAt
      (hS.mem_nhds (hcontains θ))
  have hf' : HasFDerivAt F (fderiv ℝ F θ.val) ((rotationFlow i j 0 θ).val) := by
    simpa only [rotationFlow_zero] using hf.hasFDerivAt
  exact hf'.comp_hasDerivAt 0 (rotationFlow_hasDerivAt_zero i j θ)

theorem ambientJet_isRotationJet
    {S : Set (EuclideanSpace ℝ (Fin (n+1)))}
    (hS : IsOpen S) (hcontains : ∀ θ : UnitSphere n, θ.val ∈ S)
    (F : EuclideanSpace ℝ (Fin (n+1)) → ℝ) (hF : ContDiffOn ℝ ∞ F S) :
    IsRotationJet (ambientJet hS hcontains F hF) := by
  intro w i j
  exact ambientRotationDerivative_along_flow hS hcontains (ambientRotationWord F w)
    (ambientRotationWord_contDiffOn hS hF w) i j

end
end QIRadon.Casimir
