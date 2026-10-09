import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-! Explicit, checkable construction margins. These are choices made by the
construction, not additional regularity assumptions on the target. -/
namespace QIRadon.Radon

structure SourceParameters (R σ : ℝ) where
  T : ℝ
  S : ℝ
  R₀ : ℝ
  Q : ℝ
  h : ℝ
  σ₀ : ℝ
  η : ℝ
  b : ℝ
  radius_gt_one : 1 < R
  target_tube_pos : 0 < σ
  one_lt_T : 1 < T
  T_lt_S : T < S
  S_lt_R₀ : S < R₀
  R₀_lt_R : R₀ < R
  h_pos : 0 < h
  projection_margin : S+4*h ≤ Q
  Q_lt_R₀ : Q < R₀
  hilbert_margin : T+2*h < S
  scalar_tube_pos : 0 < σ₀
  scalar_target_margin : (1+Q/Real.sqrt (R₀^2-Q^2))*(4*h) ≤ σ₀
  rotation_budget_pos : 0 < η
  rotation_target_margin : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ
  orbit_budget_pos : 0 < b
  orbit_rotation_margin : b ≤ η
  orbit_scalar_margin : Real.exp (2*b)-1 ≤ h

namespace SourceParameters
variable {R σ : ℝ} (p : SourceParameters R σ)

theorem T_pos : 0 < p.T := by linarith [p.one_lt_T]
theorem S_pos : 0 < p.S := p.T_pos.trans p.T_lt_S
theorem R₀_pos : 0 < p.R₀ := p.S_pos.trans p.S_lt_R₀
theorem Q_pos : 0 < p.Q := by linarith [p.S_pos, p.h_pos, p.projection_margin]
end SourceParameters
end QIRadon.Radon
