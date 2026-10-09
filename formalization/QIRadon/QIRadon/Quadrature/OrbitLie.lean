import QIRadon.Quadrature.MixedRestriction
import QIRadon.Quadrature.RotationFlow
import QIRadon.Quadrature.CasimirJet

open Set
open scoped ContDiff Topology

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 200000

open Gaussian MixedRestriction MixedCauchy

variable {n : ℕ}

/-- The first listed rotation acts innermost. This order makes coordinate zero
 exactly the next outer differentiation in the recursive Lie word. -/
def realOrbit : (w : List (RotationPair n)) → (Fin w.length → ℝ) → UnitSphere n → UnitSphere n
  | [], _, θ => θ
  | p::w, t, θ => realOrbit w (fun i ↦ t i.succ) (rotationFlow p.1 p.2 (t 0) θ)

theorem realOrbit_zero (w : List (RotationPair n)) (θ : UnitSphere n) : realOrbit w 0 θ = θ := by
  induction w generalizing θ with
  | nil => rfl
  | cons p w ih =>
    simp only [realOrbit, Pi.zero_apply, rotationFlow_zero]
    convert ih θ using 1
    rfl

theorem realOrbit_continuous (w : List (RotationPair n)) (t : Fin w.length → ℝ) :
    Continuous (realOrbit w t) := by
  induction w with
  | nil => exact continuous_id
  | cons p w ih => exact (ih _).comp (rotationFlow p.1 p.2 (t 0)).continuous

def realAngleDomain (l : ℕ) (δ : ℝ) : Set (Fin l → ℝ) :=
  (realCoordinates l) ⁻¹' l1Ball l δ

theorem realAngleDomain_open (l : ℕ) (δ : ℝ) : IsOpen (realAngleDomain l δ) :=
  (isOpen_l1Ball l δ).preimage (realCoordinates l).continuous

theorem realAngleDomain_zero (l : ℕ) {δ : ℝ} (hδ : 0 < δ) :
    (0 : Fin l → ℝ) ∈ realAngleDomain l δ := by
  change l1Size (realCoordinates l 0) < δ
  simpa only [map_zero, l1Size_zero] using hδ

def lieWord (F : UnitSphere n → ℝ) : List (RotationPair n) → UnitSphere n → ℝ
  | [], θ => F θ
  | p::w, θ => deriv (fun t ↦ lieWord F w (rotationFlow p.1 p.2 t θ)) 0

theorem allIndices_cons (l : ℕ) : (List.ofFn (fun i : Fin (l+1) ↦ i)) =
    0 :: (List.ofFn (fun i : Fin l ↦ i)).map Fin.succ := by
  rw [List.ofFn_succ, List.map_ofFn]
  rfl

/-- The recursively defined Lie derivatives equal the actual mixed coordinate
 derivatives of the finite rotation word. This is an exact identification. -/
theorem lieWord_eq_realMixed (F : UnitSphere n → ℝ) {δ : ℝ} (hδ : 0 < δ)
    (hsmooth : ∀ (w : List (RotationPair n)) (θ : UnitSphere n),
      ContDiffOn ℝ ∞ (fun t ↦ F (realOrbit w t θ)) (realAngleDomain w.length δ))
    (w : List (RotationPair n)) (θ : UnitSphere n) :
    lieWord F w θ = realMixedPartial (List.ofFn (fun i : Fin w.length ↦ i))
      (fun t ↦ F (realOrbit w t θ)) 0 := by
  induction w generalizing θ with
  | nil => rfl
  | cons p w ih =>
    have hindices : (List.ofFn (fun i : Fin (p::w).length ↦ i)) =
        0 :: (List.ofFn (fun i : Fin w.length ↦ i)).map Fin.succ := allIndices_cons w.length
    rw [hindices]
    rw [realMixedPartial_cons_head (realAngleDomain_open _ δ)
      (realAngleDomain_zero _ hδ) (hsmooth (p::w) θ)]
    unfold lieWord
    congr 1
    funext t
    exact ih (rotationFlow p.1 p.2 t θ)

theorem lieWord_hasDerivAt_zero (F : UnitSphere n → ℝ) {δ : ℝ} (hδ : 0 < δ)
    (hsmooth : ∀ (w : List (RotationPair n)) (θ : UnitSphere n),
      ContDiffOn ℝ ∞ (fun t ↦ F (realOrbit w t θ)) (realAngleDomain w.length δ))
    (w : List (RotationPair n)) (i j : Fin (n+1)) (θ : UnitSphere n) :
    HasDerivAt (fun t ↦ lieWord F w (rotationFlow i j t θ)) (lieWord F ((i,j)::w) θ) 0 := by
  have h := realMixedPartial_cons_head_hasDerivAt (realAngleDomain_open _ δ)
    (realAngleDomain_zero _ hδ) (hsmooth ((i,j)::w) θ)
    (List.ofFn (fun k : Fin w.length ↦ k))
  have hfun : (fun t ↦ lieWord F w (rotationFlow i j t θ)) =
      (fun t ↦ realMixedPartial (List.ofFn (fun k : Fin w.length ↦ k))
        (fun y ↦ F (realOrbit ((i,j)::w) (Fin.cons t y) θ)) 0) := by
    funext t
    exact lieWord_eq_realMixed F hδ hsmooth w (rotationFlow i j t θ)
  have hindices : (List.ofFn (fun a : Fin ((i,j)::w).length ↦ a)) =
      0 :: (List.ofFn (fun a : Fin w.length ↦ a)).map Fin.succ := allIndices_cons w.length
  rw [hfun, lieWord_eq_realMixed F hδ hsmooth ((i,j)::w) θ, hindices]
  exact h

end
end QIRadon.Casimir
