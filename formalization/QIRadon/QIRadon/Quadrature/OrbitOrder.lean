import QIRadon.Quadrature.OrbitFamily
import QIRadon.Quadrature.ComplexGenerator

open scoped BigOperators ContDiff

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 200000

open Gaussian ComplexRotation MixedCauchy MixedRestriction

theorem ofFn_rev {α : Type*} {l : ℕ} (f : Fin l → α) :
    List.ofFn (fun i ↦ f i.rev) = (List.ofFn f).reverse := by
  apply List.ext_getElem
  · simp
  · intro i hi₁ hi₂
    simp only [List.getElem_ofFn, List.getElem_reverse, List.length_ofFn]
    congr 1
    apply Fin.ext
    simp only [Fin.rev, Fin.val_mk]
    omega

def realSteps {n : ℕ} (w : List (RotationPair n)) (t : Fin w.length → ℝ) :
    List (RotationPair n × ℝ) := List.ofFn (fun i ↦ (w.get i, t i))

theorem realSteps_cons {n : ℕ} (p : RotationPair n) (w : List (RotationPair n))
    (t : Fin (p::w).length → ℝ) :
    realSteps (p::w) t = (p,t 0) :: realSteps w (fun i ↦ t i.succ) := by
  unfold realSteps
  rw [List.ofFn_succ]
  rfl

def realStep {n : ℕ} (p : RotationPair n × ℝ) (θ : UnitSphere n) : UnitSphere n :=
  rotationFlow p.1.1 p.1.2 p.2 θ

theorem realOrbit_eq_foldl {n : ℕ} (w : List (RotationPair n)) (t : Fin w.length → ℝ)
    (θ : UnitSphere n) :
    realOrbit w t θ = (realSteps w t).foldl (fun θ p ↦ realStep p θ) θ := by
  induction w generalizing θ with
  | nil => rfl
  | cons p w ih =>
    rw [realSteps_cons, List.foldl_cons]
    exact ih _ _

theorem indexedWord_real_foldr {n l : ℕ} (pairs : Fin l → RotationPair n)
    (t : Fin l → ℝ) (word : List (Fin l)) (θ : UnitSphere n) :
    indexedWord pairs word (fun i ↦ (t i : ℂ)) (realEmbedding θ.val) =
      realEmbedding ((word.map (fun i ↦ (pairs i,t i))).foldr realStep θ).val := by
  induction word with
  | nil => rfl
  | cons i word ih =>
    rw [indexedWord, ih]
    exact complexPlaneRotate_real_flow (pairs i).1 (pairs i).2 (t i) _

theorem rotationWord_real_foldr {n l : ℕ} (pairs : Fin l → RotationPair n)
    (t : Fin l → ℝ) (θ : UnitSphere n) :
    rotationWord pairs (fun i ↦ (t i : ℂ)) (realEmbedding θ.val) =
      realEmbedding ((List.ofFn (fun i ↦ (pairs i,t i))).foldr realStep θ).val := by
  simpa only [rotationWord, List.map_ofFn, Function.comp_def] using
    indexedWord_real_foldr pairs t (List.ofFn (fun i : Fin l ↦ i)) θ

/-- Exact adapter between the source's outermost-first rotation word and the
 innermost-first Lie-word convention: reverse both pairs and their angles. -/
theorem reverse_rotationWord_realOrbit {n : ℕ} (w : List (RotationPair n))
    (t : Fin w.length → ℝ) (θ : UnitSphere n) :
    rotationWord (fun i ↦ w.get i.rev) (fun i ↦ (t i.rev : ℂ)) (realEmbedding θ.val) =
      realEmbedding (realOrbit w t θ).val := by
  rw [rotationWord_real_foldr]
  have hsteps : List.ofFn (fun i : Fin w.length ↦ (w.get i.rev,t i.rev)) =
      (realSteps w t).reverse := ofFn_rev (fun i ↦ (w.get i,t i))
  rw [hsteps, List.foldr_reverse, realOrbit_eq_foldl]

def reverseCoordinates (l : ℕ) : (Fin l → ℂ) →L[ℂ] (Fin l → ℂ) :=
  ContinuousLinearMap.pi (fun i ↦ ContinuousLinearMap.proj i.rev)

@[simp] theorem reverseCoordinates_apply (l : ℕ) (z : Fin l → ℂ) (i : Fin l) :
    reverseCoordinates l z i = z i.rev := rfl

theorem l1Size_reverseCoordinates (l : ℕ) (z : Fin l → ℂ) :
    l1Size (reverseCoordinates l z) = l1Size z :=
  Equiv.sum_comp Fin.revPerm (fun i ↦ ‖z i‖)

theorem reverseCoordinates_real (l : ℕ) (t : Fin l → ℝ) :
    reverseCoordinates l (realCoordinates l t) = realCoordinates l (fun i ↦ t i.rev) := rfl


/-- Adapter for the source's existing outermost-first convention. Real
 agreement is stated by the actual complexified point reached by the word, so
 this theorem does not depend on a particular real-isometry implementation. -/
def boundedOrbitExtension_of_outer {n : ℕ} (F : C(UnitSphere n, ℝ)) (δ B : ℝ)
    (H : (l : ℕ) → (Fin l → RotationPair n) → UnitSphere n → (Fin l → ℂ) → ℂ)
    (hsmooth : ∀ l pairs θ, ContDiffOn ℂ ∞ (H l pairs θ) (l1Ball l δ))
    (hbound : ∀ (l : ℕ) (pairs : Fin l → RotationPair n) (θ : UnitSphere n)
      (z : Fin l → ℂ), z ∈ l1Ball l δ → ‖H l pairs θ z‖ ≤ B)
    (hreal : ∀ l pairs θ (t : Fin l → ℝ) (θ' : UnitSphere n),
      realCoordinates l t ∈ l1Ball l δ →
      rotationWord pairs (realCoordinates l t) (realEmbedding θ.val) = realEmbedding θ'.val →
      H l pairs θ (realCoordinates l t) = (F θ' : ℂ)) : BoundedOrbitExtension F δ B where
  profile w θ z := H w.length (fun i ↦ w.get i.rev) θ (reverseCoordinates w.length z)
  smooth w θ := (hsmooth w.length _ θ).comp (reverseCoordinates w.length).contDiff.contDiffOn (by
    intro z hz
    change l1Size (reverseCoordinates w.length z) < δ
    rwa [l1Size_reverseCoordinates])
  bound w θ z hz := hbound w.length _ θ _ (by
    change l1Size (reverseCoordinates w.length z) < δ
    rwa [l1Size_reverseCoordinates])
  real_agreement w θ t ht := by
    rw [reverseCoordinates_real]
    apply hreal w.length _ θ (fun i ↦ t i.rev) (realOrbit w t θ)
    · rw [← reverseCoordinates_real]
      change l1Size (reverseCoordinates w.length (realCoordinates w.length t)) < δ
      rw [l1Size_reverseCoordinates]
      exact ht
    · exact reverse_rotationWord_realOrbit w t θ

end
end QIRadon.Casimir
