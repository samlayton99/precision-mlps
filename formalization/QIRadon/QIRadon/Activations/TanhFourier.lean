import Mathlib

/-! The actual tanh derivative kernel and its Fourier transform, derived through
an explicit logistic change of variables and the beta integral. -/
noncomputable section
open MeasureTheory Set
open scoped FourierTransform
namespace QIRadon.Activations

/-- Unit-mass derivative kernel for tanh: tanh'=2K. -/
def tanhKernelValue (x : ℝ) : ℝ := 1/(2*Real.cosh x^2)

theorem tanh_hasDerivAt (x : ℝ) : HasDerivAt Real.tanh (2*tanhKernelValue x) x := by
  have h := (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) (Real.cosh_pos x).ne'
  have hf : Real.tanh = fun x : ℝ => Real.sinh x/Real.cosh x := funext Real.tanh_eq_sinh_div_cosh
  rw [hf]
  convert! h using 1
  unfold tanhKernelValue
  rw [← pow_two, ← pow_two, Real.cosh_sq_sub_sinh_sq]
  ring

def tanhCDF (x : ℝ) : ℝ := (Real.tanh x+1)/2

theorem tanhCDF_hasDerivAt (x : ℝ) : HasDerivAt tanhCDF (tanhKernelValue x) x := by
  have h := ((tanh_hasDerivAt x).add_const 1).div_const 2
  convert! h using 1
  ring

theorem tanhCDF_mem (x : ℝ) : tanhCDF x ∈ Ioo (0:ℝ) 1 := by
  unfold tanhCDF
  constructor <;> linarith [Real.neg_one_lt_tanh x, Real.tanh_lt_one x]

theorem tanhCDF_injective : Function.Injective tanhCDF := by
  intro x y h
  apply Real.tanh_injective
  unfold tanhCDF at h
  linarith

theorem tanhCDF_image : tanhCDF '' univ=Ioo (0:ℝ) 1 := by
  ext t
  constructor
  · rintro ⟨x, _, rfl⟩
    exact tanhCDF_mem x
  · intro ht
    refine ⟨Real.artanh (2*t-1), mem_univ _, ?_⟩
    unfold tanhCDF
    rw [Real.tanh_artanh (by constructor <;> linarith [ht.1, ht.2])]
    ring

theorem tanhCDF_log_ratio (x : ℝ) : Real.log (tanhCDF x/(1-tanhCDF x))=2*x := by
  have ht := tanhCDF_mem x
  have hfrac : tanhCDF x/(1-tanhCDF x)=(1+Real.tanh x)/(1-Real.tanh x) := by
    unfold tanhCDF
    field_simp
    ring
  rw [hfrac]
  have ha := Real.artanh_tanh x
  rw [Real.artanh_eq_half_log ⟨(Real.neg_one_lt_tanh x).le, (Real.tanh_lt_one x).le⟩] at ha
  linarith

/-- Evaluation of the exact beta integral occurring after logistic substitution.
The frequency-zero value is 1 and is handled separately from the quotient. -/
theorem beta_one_conjugate_imag {y : ℝ} (hy : y ≠ 0) :
    Complex.betaIntegral (1-Complex.I*(y:ℂ)) (1+Complex.I*(y:ℂ)) =
      (Real.pi*(y:ℂ))/Complex.sinh (Real.pi*(y:ℂ)) := by
  have hs : 0 < (1-Complex.I*(y:ℂ)).re := by simp
  have ht : 0 < (1+Complex.I*(y:ℂ)).re := by simp
  have hg2 : Complex.Gamma (2:ℂ)=1 := by
    rw [show (2:ℂ)=1+1 by ring, Complex.Gamma_add_one 1 (by norm_num), Complex.Gamma_one]
    simp
  have hiy : Complex.I*(y:ℂ) ≠ 0 := mul_ne_zero Complex.I_ne_zero (Complex.ofReal_ne_zero.mpr hy)
  rw [Complex.betaIntegral_eq_Gamma_mul_div _ _ hs ht,
    show (1-Complex.I*(y:ℂ))+(1+Complex.I*(y:ℂ))=(2:ℂ) by ring, hg2, div_one]
  calc
    _ = Complex.Gamma (Complex.I*(y:ℂ)+1)*Complex.Gamma (1-Complex.I*(y:ℂ)) := by rw [add_comm]; ring
    _ = (Complex.I*(y:ℂ))*(Complex.Gamma (Complex.I*(y:ℂ))*Complex.Gamma (1-Complex.I*(y:ℂ))) := by
      rw [Complex.Gamma_add_one _ hiy]
      ring
    _ = (Complex.I*(y:ℂ))*((Real.pi:ℂ)/Complex.sin (Real.pi*(Complex.I*(y:ℂ)))) := by
      rw [Complex.Gamma_mul_Gamma_one_sub]
    _ = _ := by
      rw [show (Real.pi:ℂ)*(Complex.I*(y:ℂ))=(Real.pi*(y:ℂ))*Complex.I by ring, Complex.sin_mul_I]
      field_simp

private def tanhBetaIntegrand (ξ t : ℝ) : ℂ :=
  (t:ℂ)^(-Complex.I*((Real.pi*ξ:ℝ):ℂ)) *
    (1-(t:ℂ))^(Complex.I*((Real.pi*ξ:ℝ):ℂ))

private theorem tanhBetaIntegrand_CDF (ξ x : ℝ) :
    tanhBetaIntegrand ξ (tanhCDF x) = (Real.fourierChar (-(x*ξ)):ℂ) := by
  have ht := tanhCDF_mem x
  have hu : 0 < 1-tanhCDF x := by linarith [ht.2]
  have hratio := tanhCDF_log_ratio x
  rw [Real.log_div ht.1.ne' hu.ne'] at hratio
  unfold tanhBetaIntegrand
  rw [Complex.cpow_def_of_ne_zero (Complex.ofReal_ne_zero.mpr ht.1.ne'),
    Complex.cpow_def_of_ne_zero (by exact_mod_cast hu.ne' : (1-(tanhCDF x:ℂ)) ≠ 0),
    ← Complex.exp_add, ← Complex.ofReal_log ht.1.le,
    ← Complex.ofReal_one, ← Complex.ofReal_sub, ← Complex.ofReal_log hu.le,
    Real.fourierChar_apply]
  congr 1
  have hc := congrArg (fun a : ℝ => (a:ℂ)) hratio
  push_cast at hc ⊢
  linear_combination -Complex.I*(Real.pi:ℂ)*(ξ:ℂ)*hc

private theorem beta_one_pm_eq_integral (ξ : ℝ) :
    Complex.betaIntegral (1-Complex.I*((Real.pi*ξ:ℝ):ℂ))
      (1+Complex.I*((Real.pi*ξ:ℝ):ℂ)) = ∫ t in Ioo (0:ℝ) 1, tanhBetaIntegrand ξ t := by
  unfold Complex.betaIntegral tanhBetaIntegrand
  simp_rw [show (1-Complex.I*((Real.pi*ξ:ℝ):ℂ))-1 = -Complex.I*((Real.pi*ξ:ℝ):ℂ) by ring,
    show (1+Complex.I*((Real.pi*ξ:ℝ):ℂ))-1 = Complex.I*((Real.pi*ξ:ℝ):ℂ) by ring]
  rw [intervalIntegral.integral_of_le (by norm_num : (0:ℝ) ≤ 1), integral_Ioc_eq_integral_Ioo]

/-- The actual tanh-kernel Fourier integral equals its beta-integral expression.
The substitution is a proved real differentiable bijection, not a transform-table premise. -/
theorem tanhKernel_fourier_beta (ξ : ℝ) :
    𝓕 (fun x : ℝ => (tanhKernelValue x:ℂ)) ξ =
      Complex.betaIntegral (1-Complex.I*((Real.pi*ξ:ℝ):ℂ))
        (1+Complex.I*((Real.pi*ξ:ℝ):ℂ)) := by
  have hsub := integral_image_eq_integral_abs_deriv_smul MeasurableSet.univ
    (fun x _ => (tanhCDF_hasDerivAt x).hasDerivWithinAt) tanhCDF_injective.injOn
    (tanhBetaIntegrand ξ)
  rw [tanhCDF_image, Measure.restrict_univ] at hsub
  rw [beta_one_pm_eq_integral, hsub, Real.fourier_real_eq]
  apply integral_congr_ae
  filter_upwards with x
  have hK : 0 ≤ tanhKernelValue x := by unfold tanhKernelValue; positivity
  rw [abs_of_nonneg hK, tanhBetaIntegrand_CDF]
  simp only [Circle.smul_def, Complex.real_smul, smul_eq_mul]
  ring

/-- Explicit everywhere-defined spectrum of the unit-mass tanh derivative kernel. -/
def tanhSpectrum (ξ : ℝ) : ℝ :=
  if ξ=0 then 1 else Real.pi^2*ξ/Real.sinh (Real.pi^2*ξ)

theorem tanhKernel_fourier (ξ : ℝ) :
    𝓕 (fun x : ℝ => (tanhKernelValue x:ℂ)) ξ = (tanhSpectrum ξ:ℂ) := by
  rw [tanhKernel_fourier_beta]
  by_cases hξ : ξ=0
  · subst ξ
    simp [tanhSpectrum, Complex.betaIntegral_eval_one_right]
  · rw [beta_one_conjugate_imag (mul_ne_zero Real.pi_ne_zero hξ)]
    simp only [tanhSpectrum, if_neg hξ, Complex.ofReal_div, Complex.ofReal_mul, Complex.ofReal_pow,
      Complex.ofReal_sinh]
    have hp : (Real.pi:ℂ)*(Real.pi*(ξ:ℂ))=(Real.pi:ℂ)^2*(ξ:ℂ) := by ring
    rw [hp]

end QIRadon.Activations
