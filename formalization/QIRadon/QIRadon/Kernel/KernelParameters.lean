import QIRadon.Kernel.KernelRates

/-! Explicit sufficient alias/halo constants, requiring no parameter search.
These choices certify the existing construction; they are not an optimized
lambda rule and make no optimal-bandwidth claim. -/
noncomputable section
open scoped Real
namespace QIRadon.KernelParameters
open QIRadon.KernelRates

def oversampling {Khat : ℝ → ℂ} {ν : ℝ} (E : SpectralEnvelope Khat ν) : ℝ :=
  2*max 1 (((E.A+4)/E.a)^(1/ν))

def haloScale {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ) : ℝ :=
  (2*(E.A+3)/S.b)^(1/μ)

theorem oversampling_ge_two {Khat : ℝ → ℂ} {ν : ℝ} (E : SpectralEnvelope Khat ν) :
    2 ≤ oversampling E := by
  unfold oversampling
  linarith [le_max_left 1 (((E.A+4)/E.a)^(1/ν))]

theorem oversampling_separation {Khat : ℝ → ℂ} {ν : ℝ} (E : SpectralEnvelope Khat ν) :
    E.A+4 ≤ E.a*(oversampling E/2)^ν := by
  have hp : 0 < (E.A+4)/E.a := div_pos (by linarith [E.A_nonneg]) E.a_pos
  have hh := Real.rpow_le_rpow (Real.rpow_nonneg hp.le (1/ν))
    (le_max_right 1 (((E.A+4)/E.a)^(1/ν))) E.exponent_pos.le
  rw [← Real.rpow_mul hp.le, one_div_mul_cancel E.exponent_pos.ne',Real.rpow_one] at hh
  have hE := mul_le_mul_of_nonneg_left hh E.a_pos.le
  have heq : oversampling E/2=max 1 (((E.A+4)/E.a)^(1/ν)) := by unfold oversampling; ring
  rw [heq]
  simpa only [mul_div_cancel₀ _ E.a_pos.ne'] using hE

theorem haloScale_pos {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ) : 0 < haloScale E S := by
  unfold haloScale
  apply Real.rpow_pos_of_pos
  exact div_pos (mul_pos (by norm_num) (by linarith [E.A_nonneg])) S.b_pos

theorem haloScale_suppression {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ) :
    E.A+3 = (S.b/2)*(haloScale E S)^μ := by
  have hp : 0 < 2*(E.A+3)/S.b :=
    div_pos (mul_pos (by norm_num) (by linarith [E.A_nonneg])) S.b_pos
  unfold haloScale
  rw [← Real.rpow_mul hp.le,one_div_mul_cancel S.exponent_pos.ne',Real.rpow_one]
  field_simp [S.b_pos.ne']

end QIRadon.KernelParameters
