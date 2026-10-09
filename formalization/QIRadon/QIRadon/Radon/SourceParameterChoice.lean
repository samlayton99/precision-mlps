import QIRadon.Radon.SourceParameters
import Mathlib

/-! An automatic, fully checked source-margin choice. Only R>1 and σ>0 are
 inputs; all scalar, Hilbert, and rotation margins are construction choices. -/

namespace QIRadon.Radon

noncomputable section
set_option maxHeartbeats 200000

/-- Explicit source parameters, with reserve in both target and Hilbert tubes.
 The factor two in the rotation exponential is the proved rotation bound. -/
def standardSourceParameters (R σ : ℝ) (hR : 1 < R) (hσ : 0 < σ) : SourceParameters R σ := by
  let gap := (R-1)/4
  let T := 1+gap
  let S := 1+2*gap
  let R₀ := 1+3*gap
  let Q := (S+R₀)/2
  let L := 1+Q/Real.sqrt (R₀^2-Q^2)
  let η := Real.log (1+σ/(4*R))/2
  let σ₀ := σ/(4*Real.exp (2*η))
  let h := min (min ((R₀-S)/8) ((S-T)/4)) (σ₀/(4*L))
  let b := min η (Real.log (1+h)/2)
  have hgap : 0 < gap := by dsimp [gap]; linarith
  have hT : 1 < T := by dsimp [T]; linarith
  have hTS : T < S := by dsimp [T,S]; linarith
  have hSR₀ : S < R₀ := by dsimp [S,R₀]; linarith
  have hR₀R : R₀ < R := by dsimp [R₀,gap]; linarith
  have hQ : 0 < Q := by dsimp [Q]; linarith
  have hQR₀ : Q < R₀ := by dsimp [Q]; linarith
  have hR₀ : 0 < R₀ := by linarith
  have hsquare : 0 < R₀^2-Q^2 := by nlinarith
  have hroot : 0 < Real.sqrt (R₀^2-Q^2) := Real.sqrt_pos.mpr hsquare
  have hL : 0 < L := by dsimp [L]; positivity
  have hRR : 0 < R := by linarith
  have hlog : 0 < Real.log (1+σ/(4*R)) := by
    apply Real.log_pos
    have hp : 0 < σ/(4*R) := by positivity
    linarith
  have hη : 0 < η := by dsimp [η]; linarith
  have hexp : Real.exp (2*η) = 1+σ/(4*R) := by
    dsimp [η]
    rw [show 2*(Real.log (1+σ/(4*R))/2) = Real.log (1+σ/(4*R)) by ring,
      Real.exp_log (by positivity)]
  have hσ₀ : 0 < σ₀ := by dsimp [σ₀]; positivity
  have hh : 0 < h := by
    dsimp [h]
    apply lt_min
    · apply lt_min <;> positivity
    · positivity
  have hh₁ : h ≤ (R₀-S)/8 := (min_le_left _ _).trans (min_le_left _ _)
  have hh₂ : h ≤ (S-T)/4 := (min_le_left _ _).trans (min_le_right _ _)
  have hh₃ : h ≤ σ₀/(4*L) := min_le_right _ _
  have hscalar : L*(4*h) ≤ σ₀ := by
    have hc := (le_div_iff₀ (show 0 < 4*L by positivity)).mp hh₃
    nlinarith
  have hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ := by
    have he : Real.exp (2*η) ≠ 0 := (Real.exp_pos _).ne'
    have hid : σ₀+(Real.exp (2*η)-1)*(R+σ₀) = σ/2 := by
      dsimp [σ₀]
      rw [hexp]
      field_simp
      ring
    rw [hid]
    linarith
  have hb : 0 < b := by
    apply lt_min hη
    have hlogh : 0 < Real.log (1+h) := Real.log_pos (by linarith)
    linarith
  have hbη : b ≤ η := min_le_left _ _
  have hbh : b ≤ Real.log (1+h)/2 := min_le_right _ _
  have horbit : Real.exp (2*b)-1 ≤ h := by
    have he := Real.exp_le_exp.mpr (show 2*b ≤ Real.log (1+h) by linarith)
    rw [Real.exp_log (by linarith : 0 < 1+h)] at he
    linarith
  exact
    { T := T
      S := S
      R₀ := R₀
      Q := Q
      h := h
      σ₀ := σ₀
      η := η
      b := b
      radius_gt_one := hR
      target_tube_pos := hσ
      one_lt_T := hT
      T_lt_S := hTS
      S_lt_R₀ := hSR₀
      R₀_lt_R := hR₀R
      h_pos := hh
      projection_margin := by dsimp [Q]; linarith
      Q_lt_R₀ := hQR₀
      hilbert_margin := by linarith
      scalar_tube_pos := hσ₀
      scalar_target_margin := hscalar
      rotation_budget_pos := hη
      rotation_target_margin := hrotation
      orbit_budget_pos := hb
      orbit_rotation_margin := hbη
      orbit_scalar_margin := horbit }

theorem exists_sourceParameters (R σ : ℝ) (hR : 1 < R) (hσ : 0 < σ) :
    Nonempty (SourceParameters R σ) := ⟨standardSourceParameters R σ hR hσ⟩

end
end QIRadon.Radon
