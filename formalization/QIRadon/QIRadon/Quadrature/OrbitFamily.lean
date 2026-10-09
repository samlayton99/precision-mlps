import QIRadon.Quadrature.OrbitLie
import QIRadon.Quadrature.OrbitMixedContinuity
import QIRadon.Quadrature.CasimirEstimate

open Set
open scoped ContDiff Topology

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 300000

open Gaussian MixedRestriction MixedCauchy

variable {n : ℕ}

/-- Scalar bounded complex extensions of actual finite real rotation orbits.
 These are value-level source conditions. No continuous transverse frame,
 derivative-growth estimate, expansion, or spectral norm is a field. -/
structure BoundedOrbitExtension (F : C(UnitSphere n, ℝ)) (δ B : ℝ) where
  profile : (w : List (RotationPair n)) → UnitSphere n → (Fin w.length → ℂ) → ℂ
  smooth : ∀ w θ, ContDiffOn ℂ ∞ (profile w θ) (l1Ball w.length δ)
  bound : ∀ (w : List (RotationPair n)) (θ : UnitSphere n) (z : Fin w.length → ℂ),
    z ∈ l1Ball w.length δ → ‖profile w θ z‖ ≤ B
  real_agreement : ∀ w θ (t : Fin w.length → ℝ), t ∈ realAngleDomain w.length δ →
    profile w θ (realCoordinates w.length t) = (F (realOrbit w t θ) : ℂ)

namespace BoundedOrbitExtension

variable {F : C(UnitSphere n, ℝ)} {δ B : ℝ} (H : BoundedOrbitExtension F δ B)
include H

theorem smooth_real (w : List (RotationPair n)) (θ : UnitSphere n) :
    ContDiffOn ℝ ∞ (fun t ↦ F (realOrbit w t θ)) (realAngleDomain w.length δ) := by
  have hs := (H.smooth w θ).restrict_scalars ℝ
  have hc := hs.comp (realCoordinates w.length).contDiff.contDiffOn
    (show MapsTo (realCoordinates w.length) (realAngleDomain w.length δ) (l1Ball w.length δ) from
      fun _ hx ↦ hx)
  have hr := Complex.reCLM.contDiff.comp_contDiffOn hc
  apply hr.congr
  intro t ht
  change F (realOrbit w t θ) = (H.profile w θ (realCoordinates w.length t)).re
  rw [H.real_agreement w θ t ht, Complex.ofReal_re]

theorem lie_eq_mixed (hδ : 0 < δ) (w : List (RotationPair n)) (θ : UnitSphere n) :
    lieWord F w θ = (allCoordinatePartial (H.profile w θ) 0).re := by
  rw [lieWord_eq_realMixed F hδ H.smooth_real]
  have hsame : (fun t ↦ F (realOrbit w t θ)) =ᶠ[𝓝 (0 : Fin w.length → ℝ)]
      (fun t ↦ (H.profile w θ (realCoordinates w.length t)).re) := by
    filter_upwards [(realAngleDomain_open w.length δ).mem_nhds (realAngleDomain_zero _ hδ)] with t ht
    rw [H.real_agreement w θ t ht, Complex.ofReal_re]
  have heq := (realMixedPartial_eventuallyEq hsame (List.ofFn (fun i : Fin w.length ↦ i))).self_of_nhds
  rw [heq, mixedPartial_real_restriction (isOpen_l1Ball w.length δ) (H.smooth w θ)
    (List.ofFn (fun i : Fin w.length ↦ i)) 0 (by simpa [l1Ball] using hδ)]
  simp only [map_zero, allCoordinatePartial]

theorem lie_continuous (hδ : 0 < δ) (w : List (RotationPair n)) :
    Continuous (lieWord F w) := by
  have hreal : ∀ t : Fin w.length → ℝ, realCoordinates w.length t ∈ l1Ball w.length δ →
      Continuous (fun θ ↦ H.profile w θ (realCoordinates w.length t)) := by
    intro t ht
    have heq : (fun θ ↦ H.profile w θ (realCoordinates w.length t)) =
        (fun θ ↦ (F (realOrbit w t θ) : ℂ)) := by
      funext θ
      exact H.real_agreement w θ t ht
    rw [heq]
    exact Complex.continuous_ofReal.comp (F.continuous.comp (realOrbit_continuous w t))
  have hc := OrbitMixedContinuity.continuous_mixedPartial_real (H.profile w)
    (H.smooth w) (H.bound w) hreal (List.ofFn (fun i : Fin w.length ↦ i))
    0 (by simpa [l1Ball] using hδ)
  have hcr := Complex.continuous_re.comp hc
  have heq : lieWord F w = (fun θ ↦ (allCoordinatePartial (H.profile w θ) 0).re) :=
    funext (H.lie_eq_mixed hδ w)
  rw [heq]
  simpa only [map_zero, allCoordinatePartial, Function.comp_def] using hcr

def jet (hδ : 0 < δ) : RotationJet n := fun w ↦ ⟨lieWord F w, H.lie_continuous hδ w⟩

@[simp] theorem jet_nil (hδ : 0 < δ) : H.jet hδ [] = F := by ext θ; rfl

theorem jet_isRotationJet (hδ : 0 < δ) : IsRotationJet (H.jet hδ) := by
  intro w i j
  apply hasRotationDerivative_of_zero
  intro θ
  exact lieWord_hasDerivAt_zero F hδ H.smooth_real w i j θ

theorem jet_word_bound (hδ : 0 < δ) (hB : 0 ≤ B) (w : List (RotationPair n)) :
    ‖H.jet hδ w‖ ≤ B*(2*(w.length:ℝ)/δ)^w.length := by
  apply (ContinuousMap.norm_le _ (by positivity)).mpr
  intro θ
  change |lieWord F w θ| ≤ _
  rw [H.lie_eq_mixed hδ w θ]
  apply (Complex.abs_re_le_norm _).trans
  by_cases hw : w.length = 0
  · have hw' : w = [] := List.length_eq_zero_iff.mp hw
    subst w
    simpa [allCoordinatePartial, mixedPartial] using H.bound [] θ 0 (by simpa [l1Ball] using hδ)
  · exact allCoordinatePartial_zero_bound_of_smooth (Nat.pos_of_ne_zero hw)
      hδ (H.smooth w θ) (H.bound w θ)

/-- Actual bounded complex orbit extensions imply canonical exponential
 harmonic energy. All real Lie derivatives, continuity, integration by parts,
 and projection bounds have been derived. -/
theorem canonical_energy (hn : 1 ≤ n) (hδ : 0 < δ) (hB : 0 ≤ B) :
    Summable (fun k : ℕ ↦ Real.exp (2*CasimirDecay.energyRate (4*(n+1:ℕ)/δ)*k) *
      ‖continuousHarmonicCoordinates n F k‖^2) ∧
    (∑' k : ℕ, Real.exp (2*CasimirDecay.energyRate (4*(n+1:ℕ)/δ)*k) *
      ‖continuousHarmonicCoordinates n F k‖^2) ≤
      (CasimirDecay.energyConstant B (4*(n+1:ℕ)/δ))^2 := by
  have h := canonical_jet_energy hn (H.jet hδ) (H.jet_isRotationJet hδ) hB hδ (H.jet_word_bound hδ hB)
  simpa only [H.jet_nil hδ] using h

end BoundedOrbitExtension
end
end QIRadon.Casimir
