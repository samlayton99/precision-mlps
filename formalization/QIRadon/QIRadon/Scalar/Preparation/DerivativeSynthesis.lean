import QIRadon.Scalar.Preparation.HorizontalFourier
import Mathlib.Analysis.Fourier.Inversion
import Mathlib.Analysis.Fourier.FourierTransformDeriv

/-! Actual derivative-spectrum synthesis from integrable Fourier moments. This
uses mathlib's Fourier inversion and differentiation-under-the-integral theorem,
not an assumed profile reconstruction identity. -/

namespace QIRadon.ProfilePreparation

open MeasureTheory Filter
open scoped FourierTransform

/-- Exponential spectral decay supplies all the complex monomial moments
needed to differentiate inverse Fourier synthesis. -/
theorem monomial_spectrum_integrable {H : ℝ → ℂ} {D c : ℝ}
    (hH : Continuous H) (_hD : 0 ≤ D) (hc : 0 < c)
    (hbound : ∀ ξ : ℝ, ‖H ξ‖ ≤ D * Real.exp (-c * |ξ|)) (r : ℕ) :
    Integrable (fun ξ : ℝ => ξ ^ r • H ξ) := by
  have hmajor := (absolute_exponential_moment r hc).1.const_mul D
  apply hmajor.mono'
  · have hc : Continuous (fun ξ : ℝ => ξ ^ r • H ξ) := by fun_prop
    exact hc.aestronglyMeasurable
  · apply Eventually.of_forall
    intro ξ
    rw [norm_smul, Real.norm_eq_abs, abs_pow]
    have hm := mul_le_mul_of_nonneg_left (hbound ξ) (pow_nonneg (abs_nonneg ξ) r)
    convert! hm using 1
    ring

/-- The inverse Fourier derivative formula under actual integrable moments. -/
theorem iteratedDeriv_inverseFourier {H : ℝ → ℂ}
    (hH : Continuous H)
    (hMom : ∀ r : ℕ, Integrable (fun ξ : ℝ => ξ ^ r • H ξ)) (r : ℕ) :
    iteratedDeriv r (𝓕⁻ H) =
      𝓕⁻ (fun ξ : ℝ => (2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * H ξ) := by
  have href : ∀ n : ℕ, Integrable (fun ξ : ℝ => ξ ^ n • H (-ξ)) := by
    intro n
    have hi := (hMom n).norm.comp_neg
    have hm : AEStronglyMeasurable (fun ξ : ℝ => ξ ^ n • H (-ξ)) := by
      have hc : Continuous (fun ξ : ℝ => ξ ^ n • H (-ξ)) := by fun_prop
      exact hc.aestronglyMeasurable
    apply (integrable_norm_iff hm).mp
    simpa only [norm_smul, Real.norm_eq_abs, abs_pow, abs_neg] using hi
  rw [Real.fourierInv_eq_fourier_comp_neg]
  rw [Real.iteratedDeriv_fourier (N := ⊤) (fun n hn => href n) (by simp)]
  rw [Real.fourierInv_eq_fourier_comp_neg]
  congr 1
  ext ξ
  simp only [smul_eq_mul, Complex.ofReal_neg]
  congr 2
  ring

/-- A known continuous integrable profile is genuinely the inverse transform
of its Fourier transform; differentiating gives the actual target derivative. -/
theorem profile_derivative_synthesis {g : ℝ → ℂ}
    (hg : Continuous g) (hgi : Integrable g)
    (hMom : ∀ r : ℕ, Integrable (fun ξ : ℝ => ξ ^ r • (𝓕 g) ξ)) (r : ℕ) :
    iteratedDeriv r g =
      𝓕⁻ (fun ξ : ℝ => (2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * (𝓕 g) ξ) := by
  have hfi : Integrable (𝓕 g) := by simpa using hMom 0
  have hfc : Continuous (𝓕 g) := by
    change Continuous (VectorFourier.fourierIntegral Real.fourierChar volume (innerₗ ℝ) g)
    exact VectorFourier.fourierIntegral_continuous Real.continuous_fourierChar (by fun_prop) hgi
  have hinv := hg.fourierInv_fourier_eq hgi hfi
  have hder := iteratedDeriv_inverseFourier hfc hMom r
  rw [hinv] at hder
  exact hder

/-- The derivative spectrum itself is integrable, not merely its unscaled
moment. -/
theorem derivativeSpectrum_integrable {H : ℝ → ℂ}
    (hMom : ∀ r : ℕ, Integrable (fun ξ : ℝ => ξ ^ r • H ξ)) (r : ℕ) :
    Integrable (fun ξ : ℝ => (2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * H ξ) := by
  have hh := (hMom r).const_mul ((2 * Real.pi * Complex.I) ^ r)
  convert! hh using 1
  ext ξ
  simp only [mul_pow, Complex.real_smul, Complex.ofReal_pow]
  ring

end QIRadon.ProfilePreparation
