import Mathlib.Analysis.Complex.CauchyIntegral
import Mathlib.Analysis.Fourier.AddCircle
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.LinearCombination

/-! Analytic strip / Bernstein ellipse foundation for sampled Chebyshev preparation.
The coefficient decay is derived from Cauchy-Goursat contour translation,
not assumed as an approximation certificate. -/

namespace QIRadon.ProfilePreparation

open Set MeasureTheory
open scoped Interval

/-- Closed strip of a specified half-width in the angular variable. -/
def closedStrip (δ : ℝ) : Set ℂ := {z | |z.im| ≤ δ}

/-- Bernstein ellipse defined by the cosine image of the angular strip.
This makes the local ellipse hypothesis explicit; no enclosing radius > 1 disk
is used. -/
def bernsteinEllipse (δ : ℝ) : Set ℂ := Complex.cos '' closedStrip δ

/-- Integer Fourier exponential on the complex angular variable. -/
noncomputable def fourierFactor (k : ℤ) (z : ℂ) : ℂ :=
  Complex.exp (-(k : ℂ) * Complex.I * z)

/-- Normalized Fourier coefficient over a single real period. -/
noncomputable def stripCoefficient (F : ℂ → ℂ) (k : ℤ) : ℂ :=
  (1 / (2 * Real.pi) : ℝ) •
    ∫ t in (0 : ℝ)..(2 * Real.pi), F (t : ℂ) * fourierFactor k (t : ℂ)

theorem fourierFactor_periodic (k : ℤ) (z : ℂ) :
    fourierFactor k (z + (2 * Real.pi : ℝ)) = fourierFactor k z := by
  unfold fourierFactor
  have he := Complex.exp_periodic.int_mul (-k) (-(k : ℂ) * Complex.I * z)
  convert! he using 1
  congr 1
  push_cast
  ring

theorem norm_fourierFactor (k : ℤ) (t y : ℝ) :
    ‖fourierFactor k ((t : ℂ) + y * Complex.I)‖ = Real.exp ((k : ℝ) * y) := by
  unfold fourierFactor
  rw [Complex.norm_exp]
  congr 1
  simp [Complex.mul_re, Complex.mul_im]

/-- Periodicity cancels the vertical sides of a Cauchy rectangle. -/
theorem periodic_integral_shift {F : ℂ → ℂ} {δ y : ℝ}
    (hδ : 0 ≤ δ) (hy : |y| ≤ δ)
    (hF : DifferentiableOn ℂ F (closedStrip δ))
    (hperiod : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z) :
    (∫ t in (0 : ℝ)..(2 * Real.pi), F (t : ℂ)) =
      ∫ t in (0 : ℝ)..(2 * Real.pi), F ((t : ℂ) + y * Complex.I) := by
  have hrect : DifferentiableOn ℂ F
      ([[((0 : ℂ).re), (Complex.mk (2 * Real.pi) y).re]] ×ℂ
        [[((0 : ℂ).im), (Complex.mk (2 * Real.pi) y).im]]) := by
    apply hF.mono
    intro z hz
    change z.re ∈ uIcc 0 (2 * Real.pi) ∧ z.im ∈ uIcc 0 y at hz
    change |z.im| ≤ δ
    have hy' := abs_le.mp hy
    apply abs_le.mpr
    constructor
    · exact le_trans (le_min (by linarith) hy'.1) hz.2.1
    · exact le_trans hz.2.2 (max_le hδ hy'.2)
  have hh := Complex.integral_boundary_rect_eq_zero_of_differentiableOn F 0
    (Complex.mk (2 * Real.pi) y) hrect
  dsimp only at hh
  simp only [Complex.zero_re, Complex.zero_im,
    Complex.ofReal_zero, zero_mul, add_zero, zero_add] at hh
  have hv : (∫ t in (0 : ℝ)..y, F ((2 * Real.pi : ℝ) + t * Complex.I)) =
      ∫ t in (0 : ℝ)..y, F ((t : ℂ) * Complex.I) := by
    apply intervalIntegral.integral_congr
    intro t ht
    simpa only [add_comm] using hperiod ((t : ℂ) * Complex.I)
  rw [hv] at hh
  exact sub_eq_zero.mp (by linear_combination hh)

/-- Geometric decay of each actual Fourier coefficient of a bounded
holomorphic periodic strip function. -/
theorem stripCoefficient_bound {F : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 ≤ δ)
    (hF : AnalyticOnNhd ℂ F (closedStrip δ))
    (hperiod : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z)
    (hbound : ∀ z ∈ closedStrip δ, ‖F z‖ ≤ M) (k : ℤ) :
    ‖stripCoefficient F k‖ ≤ M * Real.exp (-|(k : ℝ)| * δ) := by
  let y : ℝ := if 0 ≤ (k : ℝ) then -δ else δ
  have hy : |y| ≤ δ := by
    dsimp [y]
    split_ifs <;> simp [abs_of_nonneg hδ]
  have hky : (k : ℝ) * y = -|(k : ℝ)| * δ := by
    dsimp [y]
    split_ifs with hk
    · rw [abs_of_nonneg hk]
      ring
    · rw [abs_of_neg (lt_of_not_ge hk)]
      ring
  have hfprod : DifferentiableOn ℂ (fun z => F z * fourierFactor k z) (closedStrip δ) := by
    apply hF.differentiableOn.mul
    have hfac : Differentiable ℂ (fourierFactor k) := by unfold fourierFactor; fun_prop
    exact hfac.differentiableOn
  have hperiodprod : ∀ z : ℂ,
      F (z + (2 * Real.pi : ℝ)) * fourierFactor k (z + (2 * Real.pi : ℝ)) =
      F z * fourierFactor k z := by
    intro z
    rw [hperiod, fourierFactor_periodic]
  have hshift := periodic_integral_shift hδ hy hfprod hperiodprod
  have hint : ‖∫ t in (0 : ℝ)..(2 * Real.pi),
      F ((t : ℂ) + y * Complex.I) * fourierFactor k ((t : ℂ) + y * Complex.I)‖ ≤
      (M * Real.exp (-|(k : ℝ)| * δ)) * (2 * Real.pi) := by
    have hraw := intervalIntegral.norm_integral_le_of_norm_le_const
      (a := (0 : ℝ)) (b := 2 * Real.pi)
      (f := fun t : ℝ => F ((t : ℂ) + y * Complex.I) * fourierFactor k ((t : ℂ) + y * Complex.I))
      (C := M * Real.exp (-|(k : ℝ)| * δ)) (by
        intro t ht
        rw [norm_mul, norm_fourierFactor, hky]
        apply mul_le_mul_of_nonneg_right _ (Real.exp_pos _).le
        apply hbound
        simpa [closedStrip] using hy)
    have hp : 0 < 2 * Real.pi := by positivity
    simpa only [sub_zero, abs_of_pos hp] using hraw
  unfold stripCoefficient
  rw [hshift, norm_smul, Real.norm_eq_abs,
    abs_of_pos (by positivity : (0 : ℝ) < 1 / (2 * Real.pi))]
  have hm := mul_le_mul_of_nonneg_left hint
    (by positivity : (0 : ℝ) ≤ 1 / (2 * Real.pi))
  convert! hm using 1
  field_simp

/-- The local Bernstein-ellipse hypothesis directly yields the strip condition
for the actual profile composition. -/
theorem ellipse_composition_analytic {g : ℂ → ℂ} {δ : ℝ}
    (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ)) :
    AnalyticOnNhd ℂ (fun z => g (Complex.cos z)) (closedStrip δ) := by
  intro z hz
  exact (hg (Complex.cos z) ⟨z, hz, rfl⟩).comp (Complex.analyticAt_cos)

/-- Actual profile Fourier coefficients decay geometrically under an ellipse
assumption, without strengthening the domain to a large disk. -/
theorem ellipseCoefficient_bound {g : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 ≤ δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M) (k : ℤ) :
    ‖stripCoefficient (fun z => g (Complex.cos z)) k‖ ≤
      M * Real.exp (-|(k : ℝ)| * δ) := by
  apply stripCoefficient_bound hδ (ellipse_composition_analytic hg)
  · intro z
    congr 1
    simpa using Complex.cos_add_two_pi z
  · intro z hz
    exact hbound (Complex.cos z) ⟨z, hz, rfl⟩


private instance twoPi_pos : Fact (0 < 2 * Real.pi) := ⟨by positivity⟩

private theorem realRestriction_periodic {F : ℂ → ℂ}
    (hp : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z) :
    Function.Periodic (fun t : ℝ => F (t : ℂ)) (2 * Real.pi) := by
  intro t
  simpa only [Complex.ofReal_add] using hp (t : ℂ)

/-- Continuous function on the actual angular quotient, used only for Fourier
inversion, not as an additional assumption on approximation. -/
noncomputable def periodicLift (F : ℂ → ℂ)
    (hp : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z)
    (hc : Continuous (fun t : ℝ => F (t : ℂ))) : C(AddCircle (2 * Real.pi), ℂ) where
  toFun := (realRestriction_periodic hp).lift
  continuous_toFun := by
    apply isQuotientMap_quotient_mk'.continuous_iff.mpr
    exact hc

@[simp]
theorem periodicLift_coe (F : ℂ → ℂ)
    (hp : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z)
    (hc : Continuous (fun t : ℝ => F (t : ℂ))) (t : ℝ) :
    periodicLift F hp hc (t : AddCircle (2 * Real.pi)) = F (t : ℂ) := rfl

/-- Normalization connector between the explicitly integrated coefficient and
mathlib's quotient-circle Fourier coefficient. -/
theorem fourierCoeff_periodicLift (F : ℂ → ℂ)
    (hp : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z)
    (hc : Continuous (fun t : ℝ => F (t : ℂ))) (k : ℤ) :
    fourierCoeff (periodicLift F hp hc) k = stripCoefficient F k := by
  rw [fourierCoeff_eq_intervalIntegral (periodicLift F hp hc) k 0]
  unfold stripCoefficient
  simp only [zero_add]
  congr 1
  apply intervalIntegral.integral_congr
  intro t ht
  dsimp only
  rw [fourier_coe_apply, periodicLift_coe]
  have he : (2 * (Real.pi : ℂ) * Complex.I * (-k) * (t : ℂ)) /
      ((2 * Real.pi : ℝ) : ℂ) = -(k : ℂ) * Complex.I * (t : ℂ) := by
    push_cast
    field_simp
  simp only [Int.cast_neg]
  rw [he]
  simp only [fourierFactor, smul_eq_mul, mul_comm]

/-- The positive exponential majorant is summable over all integer frequencies. -/
theorem summable_integer_exponential {δ : ℝ} (hδ : 0 < δ) :
    Summable (fun k : ℤ => Real.exp (-|(k : ℝ)| * δ)) := by
  have hn0 : Summable (fun n : ℕ => Real.exp ((n : ℝ) * (-δ))) :=
    Real.summable_exp_nat_mul_iff.mpr (neg_neg_of_pos hδ)
  have hn : Summable (fun n : ℕ => Real.exp (-|((n : ℤ) : ℝ)| * δ)) := by
    convert! hn0 using 1
    ext n
    simp only [Int.cast_natCast, Nat.abs_cast]
    congr 1
    ring
  have hm : Summable (fun n : ℕ => Real.exp (-|((-n : ℤ) : ℝ)| * δ)) := by
    simpa only [Int.cast_neg, abs_neg] using hn
  exact Summable.of_nat_of_neg (f := fun k : ℤ => Real.exp (-|(k : ℝ)| * δ)) hn hm

/-- Summability of the actual coefficients follows from the analytic strip. -/
theorem summable_stripCoefficient {F : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hF : AnalyticOnNhd ℂ F (closedStrip δ))
    (hp : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z)
    (hbound : ∀ z ∈ closedStrip δ, ‖F z‖ ≤ M) :
    Summable (stripCoefficient F) := by
  have hs := (summable_integer_exponential hδ).mul_left M
  apply Summable.of_norm
  exact hs.of_nonneg_of_le (fun k => norm_nonneg _)
    (fun k => stripCoefficient_bound hδ.le hF hp hbound k)

/-- Actual Fourier reconstruction from the local analytic strip assumptions. -/
theorem stripCoefficient_hasSum {F : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hF : AnalyticOnNhd ℂ F (closedStrip δ))
    (hp : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z)
    (hbound : ∀ z ∈ closedStrip δ, ‖F z‖ ≤ M) (t : ℝ) :
    HasSum (fun k : ℤ => stripCoefficient F k *
      Complex.exp ((k : ℂ) * Complex.I * (t : ℂ))) (F (t : ℂ)) := by
  have hc : Continuous (fun t : ℝ => F (t : ℂ)) := by
    apply continuous_iff_continuousAt.mpr
    intro t
    exact (hF (t : ℂ) (by simpa [closedStrip] using hδ.le)).continuousAt.comp
      Complex.continuous_ofReal.continuousAt
  let f := periodicLift F hp hc
  have hsum : Summable (fourierCoeff f) := by
    have heq : fourierCoeff f = stripCoefficient F := by
      ext k
      exact fourierCoeff_periodicLift F hp hc k
    rw [heq]
    exact summable_stripCoefficient hδ hF hp hbound
  have hs := has_pointwise_sum_fourier_series_of_summable hsum
    (t : AddCircle (2 * Real.pi))
  convert! hs using 1
  ext k
  rw [fourierCoeff_periodicLift F hp hc, fourier_coe_apply]
  simp only [smul_eq_mul]
  congr 2
  push_cast
  field_simp

end QIRadon.ProfilePreparation
