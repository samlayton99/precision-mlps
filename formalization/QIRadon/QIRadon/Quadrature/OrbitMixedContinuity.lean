import QIRadon.Quadrature.MixedRestriction
import QIRadon.Quadrature.OrbitContinuity

open Set Metric
open scoped ContDiff Topology

namespace QIRadon.OrbitMixedContinuity

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 200000

open MixedCauchy MixedRestriction

theorem realCoordinates_shift {n : ℕ} (y : Fin n → ℝ) (i : Fin n) (t : ℝ) :
    realCoordinates n (y + t • Pi.single i 1) =
      realCoordinates n y + (t:ℂ) • Pi.single i 1 := by
  ext j
  by_cases hij : j=i <;> simp [hij]

/-- Continuity of every actual mixed derivative in an arbitrary external
 parameter follows from real-slice continuity and uniform complex bounds.
 In particular, no continuously varying transverse frame is assumed. -/
theorem continuous_mixedPartial_real {X : Type*} [TopologicalSpace X] {n : ℕ}
    (P : X → (Fin n → ℂ) → ℂ) {δ B : ℝ}
    (hP : ∀ θ, ContDiffOn ℂ ∞ (P θ) (l1Ball n δ))
    (hbound : ∀ θ z, z ∈ l1Ball n δ → ‖P θ z‖ ≤ B)
    (hreal : ∀ y : Fin n → ℝ, realCoordinates n y ∈ l1Ball n δ →
      Continuous (fun θ ↦ P θ (realCoordinates n y)))
    (w : List (Fin n)) :
    ∀ y : Fin n → ℝ, realCoordinates n y ∈ l1Ball n δ →
      Continuous (fun θ ↦ mixedPartial w (P θ) (realCoordinates n y)) := by
  induction w with
  | nil => exact hreal
  | cons i w ih =>
    intro y hy
    let z := realCoordinates n y
    let r := (δ-l1Size z)/(2*((w.length:ℝ)+1))
    have hz : l1Size z < δ := hy
    have hr : 0 < r := by dsimp [r]; positivity
    have hbudget : l1Size z + ((w.length:ℝ)+1)*r < δ := by
      have he : ((w.length:ℝ)+1)*r = (δ-l1Size z)/2 := by
        dsimp [r]
        have hn : (w.length:ℝ)+1 ≠ 0 := by positivity
        field_simp
      rw [he]
      linarith
    have hinside (t : ℂ) (ht : t ∈ ball (0:ℂ) r) :
        l1Size (z+t • Pi.single i 1) + (w.length:ℝ)*r < δ := by
      have ht' : ‖t‖ < r := by simpa only [mem_ball, dist_zero_right] using ht
      have hs := l1Size_coordinate_shift z i t
      linarith
    have hdomain (t : ℂ) (ht : t ∈ ball (0:ℂ) r) :
        z+t • Pi.single i 1 ∈ l1Ball n δ := by
      change l1Size _ < δ
      have hlen : 0 ≤ (w.length:ℝ)*r := by positivity
      have hs := hinside t ht
      linarith
    let E : X → ℂ → ℂ := fun θ t ↦ mixedPartial w (P θ) (z+t • Pi.single i 1)
    have hdiff : ∀ θ, DifferentiableOn ℂ (E θ) (ball (0:ℂ) r) := by
      intro θ t ht
      have hp := ((mixedPartial_smooth (isOpen_l1Ball n δ) (hP θ) w).differentiableOn
        (by simp) _ (hdomain t ht)).differentiableAt
          ((isOpen_l1Ball n δ).mem_nhds (hdomain t ht))
      apply (hp.comp t ?_).differentiableWithinAt
      fun_prop
    have hb : ∀ θ t, t ∈ ball (0:ℂ) r → ‖E θ t‖ ≤ B/r^w.length := by
      intro θ t ht
      exact mixedPartial_bound_of_smooth (hP θ) (hbound θ) hr w _ (hinside t ht)
    have hc : ∀ t : ℝ, |t|<r → Continuous (fun θ ↦ E θ (t:ℂ)) := by
      intro t ht
      have hin := hdomain (t:ℂ) (by simpa using ht)
      have h := ih (y+t • Pi.single i 1) (by rwa [realCoordinates_shift])
      simpa only [realCoordinates_shift] using h
    have hcont := OrbitContinuity.continuous_deriv_zero E hr hdiff hb hc
    have hder (θ : X) : deriv (E θ) 0 = mixedPartial (i::w) (P θ) z := by
      have hp := ((mixedPartial_smooth (isOpen_l1Ball n δ) (hP θ) w).differentiableOn
        (by simp) z hy).differentiableAt ((isOpen_l1Ball n δ).mem_nhds hy)
      have hline : HasDerivAt (fun t : ℂ ↦ z+t • Pi.single i 1) (Pi.single i 1) 0 := by
        convert! ((hasDerivAt_id (0:ℂ)).smul_const (Pi.single i (1:ℂ))).const_add z using 1
        simp only [one_smul]
      have hp' : HasFDerivAt (mixedPartial w (P θ)) (fderiv ℂ (mixedPartial w (P θ)) z)
          (z+(0:ℂ) • Pi.single i 1) := by simpa using hp.hasFDerivAt
      exact (hp'.comp_hasDerivAt 0 hline).deriv
    simpa only [hder] using hcont

end
end QIRadon.OrbitMixedContinuity
