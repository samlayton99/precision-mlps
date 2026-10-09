import Mathlib.Analysis.SpecialFunctions.SmoothTransition
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

/-! The explicit smooth finite radius density used by the boundary construction.
Only a positive finite one-dimensional normalizing integral is required. -/
noncomputable section
open MeasureTheory Set
open scoped ContDiff
namespace QIRadon.Radon

/-- Exactly exp(−1/[(a−R₀)(R−a)]) inside the interval and zero outside. -/
def rawRadiusDensity (R₀ R a : ℝ) : ℝ := expNegInvGlue ((a-R₀)*(R-a))

theorem rawRadiusDensity_smooth (R₀ R : ℝ) : ContDiff ℝ ∞ (rawRadiusDensity R₀ R) :=
  expNegInvGlue.contDiff.comp (by fun_prop)

theorem rawRadiusDensity_nonneg (R₀ R a : ℝ) : 0 ≤ rawRadiusDensity R₀ R a :=
  expNegInvGlue.nonneg _

theorem rawRadiusDensity_pos {R₀ R a : ℝ} (ha : a ∈ Ioo R₀ R) :
    0 < rawRadiusDensity R₀ R a :=
  expNegInvGlue.pos_of_pos (mul_pos (sub_pos.mpr ha.1) (sub_pos.mpr ha.2))

theorem rawRadiusDensity_support {R₀ R : ℝ} (hR : R₀ < R) :
    Function.support (rawRadiusDensity R₀ R) ⊆ Ioo R₀ R := by
  intro a ha
  by_contra hh
  have hcases : a ≤ R₀ ∨ R ≤ a := by simpa only [mem_Ioo, not_and_or, not_lt] using hh
  have hprod : (a-R₀)*(R-a) ≤ 0 := by
    rcases hcases with hlo | hhi
    · exact mul_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
    · exact mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
  exact ha (expNegInvGlue.zero_of_nonpos hprod)

theorem rawRadiusDensity_compact {R₀ R : ℝ} (hR : R₀ < R) :
    HasCompactSupport (rawRadiusDensity R₀ R) :=
  HasCompactSupport.of_support_subset_isCompact (isCompact_Icc (a := R₀) (b := R))
    ((rawRadiusDensity_support hR).trans Ioo_subset_Icc_self)

theorem rawRadiusDensity_integrable {R₀ R : ℝ} (hR : R₀ < R) :
    Integrable (rawRadiusDensity R₀ R) :=
  (rawRadiusDensity_smooth R₀ R).continuous.integrable_of_hasCompactSupport (rawRadiusDensity_compact hR)

/-- A strictly positive, finite, ordinary one-dimensional normalization integral. -/
def radiusNormalization (R₀ R : ℝ) : ℝ := ∫ a in Icc R₀ R, rawRadiusDensity R₀ R a

theorem radiusNormalization_eq_integral {R₀ R : ℝ} (hR : R₀ < R) :
    radiusNormalization R₀ R = ∫ a : ℝ, rawRadiusDensity R₀ R a := by
  unfold radiusNormalization
  rw [← integral_indicator measurableSet_Icc,
    indicator_eq_self.mpr ((rawRadiusDensity_support hR).trans Ioo_subset_Icc_self)]

theorem radiusNormalization_pos {R₀ R : ℝ} (hR : R₀ < R) :
    0 < radiusNormalization R₀ R := by
  rw [radiusNormalization_eq_integral hR]
  exact (rawRadiusDensity_smooth R₀ R).continuous.integral_pos_of_hasCompactSupport_nonneg_nonzero
    (rawRadiusDensity_compact hR) (rawRadiusDensity_nonneg R₀ R)
    (rawRadiusDensity_pos (show (R₀+R)/2 ∈ Ioo R₀ R by constructor <;> linarith)).ne'

/-- The prescribed normalized density, with no user-supplied weight function. -/
def standardRadiusDensity (R₀ R a : ℝ) : ℝ :=
  rawRadiusDensity R₀ R a / radiusNormalization R₀ R

theorem standardRadiusDensity_smooth (R₀ R : ℝ) : ContDiff ℝ ∞ (standardRadiusDensity R₀ R) :=
  (rawRadiusDensity_smooth R₀ R).div_const _

theorem standardRadiusDensity_integrable {R₀ R : ℝ} (hR : R₀ < R) :
    Integrable (standardRadiusDensity R₀ R) :=
  (rawRadiusDensity_integrable hR).div_const _

theorem standardRadiusDensity_support {R₀ R : ℝ} (hR : R₀ < R) :
    Function.support (standardRadiusDensity R₀ R) ⊆ Icc R₀ R := by
  intro a ha
  apply Ioo_subset_Icc_self (rawRadiusDensity_support hR ?_)
  intro heq
  exact ha (by simp [standardRadiusDensity, heq])

theorem standardRadiusDensity_nonneg {R₀ R : ℝ} (hR : R₀ < R) (a : ℝ) :
    0 ≤ standardRadiusDensity R₀ R a :=
  div_nonneg (rawRadiusDensity_nonneg R₀ R a) (radiusNormalization_pos hR).le

theorem standardRadiusDensity_pos {R₀ R a : ℝ} (hR : R₀ < R) (ha : a ∈ Ioo R₀ R) :
    0 < standardRadiusDensity R₀ R a :=
  div_pos (rawRadiusDensity_pos ha) (radiusNormalization_pos hR)

theorem standardRadiusDensity_normalized {R₀ R : ℝ} (hR : R₀ < R) :
    (∫ a in Icc R₀ R, standardRadiusDensity R₀ R a)=1 := by
  unfold standardRadiusDensity
  rw [integral_div]
  exact div_self (radiusNormalization_pos hR).ne'

/-- The implementation-facing interior formula is exactly the paper's bump. -/
theorem standardRadiusDensity_formula {R₀ R a : ℝ} (ha : a ∈ Ioo R₀ R) :
    standardRadiusDensity R₀ R a =
      Real.exp (-((a-R₀)*(R-a))⁻¹) / radiusNormalization R₀ R := by
  have hp : 0 < (a-R₀)*(R-a) := mul_pos (sub_pos.mpr ha.1) (sub_pos.mpr ha.2)
  simp [standardRadiusDensity, rawRadiusDensity, expNegInvGlue, not_le.mpr hp]

end QIRadon.Radon
