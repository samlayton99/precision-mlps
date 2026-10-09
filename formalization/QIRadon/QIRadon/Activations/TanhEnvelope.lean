import QIRadon.Activations.TanhFourier
import QIRadon.Kernel.KernelRates

/-! Quantitative exponential envelopes for the actual tanh spectrum. The upper
rate sacrifices a factor two to absorb its linear prefactor uniformly. -/
noncomputable section
open MeasureTheory Set FourierTransform
namespace QIRadon.Activations

private theorem exp_neg_mul_sinh (t : ℝ) :
    Real.exp (-t)*Real.sinh t=(1-Real.exp (-2*t))/2 := by
  rw [Real.sinh_eq, ← mul_div_assoc, mul_sub, ← Real.exp_add, ← Real.exp_add]
  rw [show -t+t=0 by ring, show -t + -t = -2*t by ring, Real.exp_zero]

private theorem exp_neg_le_div_sinh {t : ℝ} (ht : 0 < t) :
    Real.exp (-t) ≤ t/Real.sinh t := by
  have hs : 0 < Real.sinh t := Real.sinh_pos_iff.mpr ht
  rw [le_div_iff₀ hs, exp_neg_mul_sinh]
  linarith [Real.add_one_le_exp (-2*t)]

private theorem div_sinh_le_polynomial_exp {t : ℝ} (ht : 0 < t) :
    t/Real.sinh t ≤ (1+2*t)*Real.exp (-t) := by
  have hs : 0 < Real.sinh t := Real.sinh_pos_iff.mpr ht
  have he : (1+2*t)*Real.exp (-2*t) ≤ 1 := by
    have h := mul_le_mul_of_nonneg_right (Real.add_one_le_exp (2*t)) (Real.exp_pos (-2*t)).le
    rw [← Real.exp_add, show 2*t + -2*t=0 by ring, Real.exp_zero] at h
    simpa only [add_comm] using h
  rw [div_le_iff₀ hs]
  calc
    t ≤ (1+2*t)*((1-Real.exp (-2*t))/2) := by nlinarith
    _ = ((1+2*t)*Real.exp (-t))*Real.sinh t := by rw [mul_assoc, exp_neg_mul_sinh]

private theorem polynomial_exp_le_four_exp (t : ℝ) :
    (1+2*t)*Real.exp (-t) ≤ 4*Real.exp (-t/2) := by
  have hp : 1+2*t ≤ 4*Real.exp (t/2) := by linarith [Real.add_one_le_exp (t/2)]
  calc
    _ ≤ (4*Real.exp (t/2))*Real.exp (-t) := mul_le_mul_of_nonneg_right hp (Real.exp_pos (-t)).le
    _ = 4*Real.exp (-t/2) := by rw [mul_assoc, ← Real.exp_add]; congr 2; ring

theorem tanhSpectrum_even (ξ : ℝ) : tanhSpectrum (-ξ)=tanhSpectrum ξ := by
  by_cases hξ : ξ=0
  · simp [hξ]
  · simp [tanhSpectrum, hξ, neg_ne_zero.mpr hξ, Real.sinh_neg]

theorem tanhSpectrum_abs (ξ : ℝ) : tanhSpectrum |ξ|=tanhSpectrum ξ := by
  rcases le_or_gt 0 ξ with hξ | hξ
  · rw [abs_of_nonneg hξ]
  · rw [abs_of_neg hξ, tanhSpectrum_even]

/-- Actual two-sided spectrum bounds, including zero frequency. -/
theorem tanhSpectrum_bounds (ξ : ℝ) :
    Real.exp (-(Real.pi^2)*|ξ|) ≤ tanhSpectrum ξ ∧
      tanhSpectrum ξ ≤ 4*Real.exp (-(Real.pi^2/2)*|ξ|) := by
  by_cases hξ : ξ=0
  · subst ξ
    simp [tanhSpectrum]
  · have ht : 0 < Real.pi^2*|ξ| := mul_pos (sq_pos_of_pos Real.pi_pos) (abs_pos.mpr hξ)
    rw [← tanhSpectrum_abs ξ, tanhSpectrum, if_neg (abs_pos.mpr hξ).ne']
    refine ⟨?_, ?_⟩
    · simpa only [neg_mul] using exp_neg_le_div_sinh ht
    · have h := (div_sinh_le_polynomial_exp ht).trans (polynomial_exp_le_four_exp (Real.pi^2*|ξ|))
      convert h using 1 <;> congr 2; ring

theorem tanhSpectrum_pos (ξ : ℝ) : 0 < tanhSpectrum ξ :=
  (Real.exp_pos _).trans_le (tanhSpectrum_bounds ξ).1

/-- Verified ν=1 spectral envelope for the actual tanh derivative kernel. -/
def tanhSpectral : QIRadon.KernelRates.SpectralEnvelope
    (𝓕 (fun x : ℝ => (tanhKernelValue x:ℂ))) 1 where
  c := 1
  C := 4
  A := Real.pi^2
  a := Real.pi^2/2
  c_pos := by norm_num
  C_pos := by norm_num
  A_nonneg := sq_nonneg _
  a_pos := by positivity
  exponent_pos := by norm_num
  lower ξ := by
    rw [tanhKernel_fourier, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (tanhSpectrum_pos ξ)]
    simpa only [one_mul, Real.rpow_one] using (tanhSpectrum_bounds ξ).1
  upper ξ := by
    rw [tanhKernel_fourier, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (tanhSpectrum_pos ξ)]
    simpa only [Real.rpow_one] using (tanhSpectrum_bounds ξ).2

end QIRadon.Activations
