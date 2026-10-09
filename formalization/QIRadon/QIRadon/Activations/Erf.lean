import QIRadon.Activations.Gaussian
import QIRadon.Kernel.Certificate
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-! The standard normal CDF and error-function activations, with actual
Schwartz derivative kernels and explicit Fourier envelopes. -/

noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap
namespace QIRadon.Activations

/-- The usual standard-normal density, with its actual Schwartz structure. -/
def normalPDF : 𝓢(ℝ, ℝ) := (Real.sqrt (2 * Real.pi))⁻¹ • standardGaussianSchwartz

@[simp] theorem normalPDF_apply (x : ℝ) :
    normalPDF x = (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-(x^2/2)) := rfl

theorem normalPDF_pos (x : ℝ) : 0 < normalPDF x := by rw [normalPDF_apply]; positivity

theorem normalPDF_even (x : ℝ) : normalPDF (-x) = normalPDF x := by simp

theorem normalPDF_mass : (∫ x : ℝ, normalPDF x) = 1 := by
  simp only [normalPDF_apply]
  rw [integral_const_mul]
  have hi : (∫ x : ℝ, Real.exp (-(x^2/2))) = Real.sqrt (2 * Real.pi) := by
    convert! integral_gaussian (1/2 : ℝ) using 1 <;> congr 1 <;> ring_nf
  rw [hi, inv_mul_cancel₀ (by positivity)]

theorem normalPDF_scaled_gaussian (x : ℝ) :
    normalPDF x = (Real.sqrt (2 * Real.pi))⁻¹ *
      gaussianKernel ((Real.sqrt (2 * Real.pi))⁻¹ * x) := by
  rw [normalPDF_apply, gaussianKernel_apply]
  congr 2
  have hs := Real.sq_sqrt (show 0 ≤ 2 * Real.pi by positivity)
  have hp := Real.pi_ne_zero
  have hn : Real.sqrt (2 * Real.pi) ≠ 0 := by positivity
  field_simp
  nlinarith

theorem normalPDF_fourier (ξ : ℝ) :
    𝓕 (fun x : ℝ => (normalPDF x : ℂ)) ξ =
      (Real.exp (-(2 * Real.pi^2) * ξ^2) : ℂ) := by
  have he : (fun x : ℝ => (normalPDF x : ℂ)) =
      (fun x : ℝ => ((Real.sqrt (2 * Real.pi))⁻¹ : ℂ) *
        (gaussianKernel ((Real.sqrt (2 * Real.pi))⁻¹ * x) : ℂ)) := by
    funext x
    rw [normalPDF_scaled_gaussian, Complex.ofReal_mul]
    norm_cast
  rw [he]
  have hscale := QIRadon.Kernel.fourier_scaled_kernel (fun x : ℝ => (gaussianKernel x : ℂ))
    (γ := (Real.sqrt (2 * Real.pi))⁻¹) (by positivity) ξ
  simp only [Complex.ofReal_inv] at hscale
  rw [hscale, gaussianKernel_fourier]
  congr 2
  rw [div_inv_eq_mul, mul_pow, Real.sq_sqrt (by positivity)]
  ring_nf

/-- Standard Gaussian CDF, defined by its normalized integral. -/
def normalCDF (x : ℝ) : ℝ := 1/2 + ∫ t in (0 : ℝ)..x, normalPDF t

theorem normalCDF_hasDerivAt (x : ℝ) : HasDerivAt normalCDF (normalPDF x) x := by
  have h := intervalIntegral.integral_hasDerivAt_right
    (normalPDF.continuous.intervalIntegrable 0 x) (normalPDF.continuous.stronglyMeasurableAtFilter volume (nhds x))
    normalPDF.continuous.continuousAt
  exact h.const_add (1/2 : ℝ)

@[simp] theorem normalCDF_deriv : deriv normalCDF = normalPDF := by
  funext x
  exact (normalCDF_hasDerivAt x).deriv

theorem normalCDF_contDiff (n : ℕ) : ContDiff ℝ n normalCDF := by
  cases n with
  | zero =>
      rw [Nat.cast_zero, contDiff_zero]
      exact (show Differentiable ℝ normalCDF from fun x =>
      (normalCDF_hasDerivAt x).differentiableAt).continuous
  | succ n =>
      change ContDiff ℝ ((n : WithTop ℕ∞) + 1) normalCDF
      rw [contDiff_succ_iff_deriv]
      exact ⟨fun x => (normalCDF_hasDerivAt x).differentiableAt, by simp,
        by rw [normalCDF_deriv]; convert! normalPDF.smooth (n : ℕ∞) using 1⟩

theorem normalCDF_neg (x : ℝ) : normalCDF (-x) = 1 - normalCDF x := by
  have he : (∫ t in (0 : ℝ)..(-x), normalPDF t) = -(∫ t in (0 : ℝ)..x, normalPDF t) := by
    have hi := intervalIntegral.integral_comp_neg (fun t : ℝ => normalPDF t) (a := (0 : ℝ)) (b := x)
    simp only [normalPDF_even, neg_zero] at hi
    rw [intervalIntegral.integral_symm 0 (-x)] at hi
    linarith
  unfold normalCDF
  rw [he]
  ring_nf

def normalPDFSpatial : KernelRates.SpatialEnvelope normalPDF 2 where
  C := (Real.sqrt (2 * Real.pi))⁻¹
  b := 1/2
  C_pos := by positivity
  b_pos := by norm_num
  exponent_pos := by norm_num
  upper x := by
    rw [normalPDF_apply, abs_of_pos (by positivity), Real.rpow_two, sq_abs]
    convert! le_refl ((Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-(x^2/2))) using 1
    congr 2
    ring_nf

def normalPDFSpectral : KernelRates.SpectralEnvelope
    (𝓕 (fun x : ℝ => (normalPDF x : ℂ))) 2 where
  c := 1
  C := 1
  A := 2 * Real.pi^2
  a := 2 * Real.pi^2
  c_pos := by norm_num
  C_pos := by norm_num
  A_nonneg := by positivity
  a_pos := by positivity
  exponent_pos := by norm_num
  lower ξ := by
    rw [normalPDF_fourier, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), Real.rpow_two, sq_abs, one_mul]
  upper ξ := by
    rw [normalPDF_fourier, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), Real.rpow_two, sq_abs, one_mul]

/-- Standard erf, with the exact standard-normal change of variables explicit. -/
def erf (x : ℝ) : ℝ := 2 * normalCDF (Real.sqrt 2 * x) - 1

theorem erf_hasDerivAt (x : ℝ) :
    HasDerivAt erf (2 * Real.sqrt 2 * normalPDF (Real.sqrt 2 * x)) x := by
  have hc := (normalCDF_hasDerivAt (Real.sqrt 2 * x)).comp x
    ((hasDerivAt_id x).const_mul (Real.sqrt 2))
  convert! (hc.const_mul 2).sub_const 1 using 1
  ring_nf


/-- Unit-mass standard erf derivative kernel; erf' is twice this kernel. -/
def erfKernel : 𝓢(ℝ, ℝ) := Real.sqrt 2 •
  SchwartzMap.compCLMOfContinuousLinearEquiv ℝ
    (ContinuousLinearEquiv.unitsEquivAut ℝ (Units.mk0 (Real.sqrt 2) (by positivity))) normalPDF

@[simp] theorem erfKernel_scaled (x : ℝ) :
    erfKernel x = Real.sqrt 2 * normalPDF (Real.sqrt 2 * x) := by
  simp only [erfKernel, smul_apply, smul_eq_mul,
    SchwartzMap.compCLMOfContinuousLinearEquiv_apply, Function.comp_apply,
    ContinuousLinearEquiv.unitsEquivAut_apply, Units.val_mk0, mul_comm x]

theorem erfKernel_apply (x : ℝ) :
    erfKernel x = (Real.sqrt 2 * (Real.sqrt (2 * Real.pi))⁻¹) * Real.exp (-(x^2)) := by
  rw [erfKernel_scaled, normalPDF_apply, mul_pow, Real.sq_sqrt (by norm_num)]
  rw [show -(2 * x^2 / 2) = -(x^2) by ring_nf]
  ring_nf

theorem erfKernel_mass : (∫ x : ℝ, erfKernel x) = 1 := by
  simp only [erfKernel_scaled]
  rw [integral_const_mul, MeasureTheory.Measure.integral_comp_mul_left]
  rw [normalPDF_mass, smul_eq_mul, mul_one, abs_of_pos (by positivity)]
  exact mul_inv_cancel₀ (by positivity)

theorem erf_first_derivative (x : ℝ) : iteratedDeriv 1 erf x = 2 * erfKernel x := by
  rw [iteratedDeriv_one, (erf_hasDerivAt x).deriv, erfKernel_scaled]
  ring_nf

theorem normalCDF_first_derivative (x : ℝ) : iteratedDeriv 1 normalCDF x = normalPDF x := by
  rw [iteratedDeriv_one, normalCDF_deriv]

theorem erf_contDiff (n : ℕ) : ContDiff ℝ n erf := by
  exact (contDiff_const.mul ((normalCDF_contDiff n).comp
    (contDiff_const.mul contDiff_id))).sub contDiff_const

theorem erfKernel_fourier (ξ : ℝ) :
    𝓕 (fun x : ℝ => (erfKernel x : ℂ)) ξ = (Real.exp (-Real.pi^2 * ξ^2) : ℂ) := by
  have he : (fun x : ℝ => (erfKernel x : ℂ)) =
      (fun x : ℝ => (Real.sqrt 2 : ℂ) * (normalPDF (Real.sqrt 2 * x) : ℂ)) := by
    funext x
    rw [erfKernel_scaled, Complex.ofReal_mul]
  rw [he, QIRadon.Kernel.fourier_scaled_kernel (fun x : ℝ => (normalPDF x : ℂ)) (by positivity),
    normalPDF_fourier]
  congr 2
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  field_simp
  nlinarith

def erfSpatial : KernelRates.SpatialEnvelope erfKernel 2 where
  C := Real.sqrt 2 * (Real.sqrt (2 * Real.pi))⁻¹
  b := 1
  C_pos := by positivity
  b_pos := by norm_num
  exponent_pos := by norm_num
  upper x := by
    rw [erfKernel_apply, abs_of_pos (by positivity), Real.rpow_two, sq_abs]
    norm_num

def erfSpectral : KernelRates.SpectralEnvelope
    (𝓕 (fun x : ℝ => (erfKernel x : ℂ))) 2 where
  c := 1
  C := 1
  A := Real.pi^2
  a := Real.pi^2
  c_pos := by norm_num
  C_pos := by norm_num
  A_nonneg := by positivity
  a_pos := by positivity
  exponent_pos := by norm_num
  lower ξ := by
    rw [erfKernel_fourier, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), Real.rpow_two, sq_abs, one_mul]
  upper ξ := by
    rw [erfKernel_fourier, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), Real.rpow_two, sq_abs, one_mul]


/-- The conventional normalization of the erf kernel. -/
theorem erfKernel_standard_formula (x : ℝ) :
    erfKernel x = (Real.sqrt Real.pi)⁻¹ * Real.exp (-(x^2)) := by
  rw [erfKernel_apply, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2)]
  have h2 : Real.sqrt 2 ≠ 0 := by positivity
  have hp : Real.sqrt Real.pi ≠ 0 := by positivity
  field_simp

theorem erf_zero : erf 0 = 0 := by simp [erf, normalCDF]

/-- This is the standard error function, not a differently scaled sigmoid. -/
theorem erf_integral_formula (x : ℝ) :
    erf x = (2 / Real.sqrt Real.pi) * ∫ t in (0 : ℝ)..x, Real.exp (-(t^2)) := by
  have hd : ∀ t : ℝ, HasDerivAt erf ((2 / Real.sqrt Real.pi) * Real.exp (-(t^2))) t := by
    intro t
    have he := erf_hasDerivAt t
    have hv : 2 * Real.sqrt 2 * normalPDF (Real.sqrt 2 * t) =
        (2 / Real.sqrt Real.pi) * Real.exp (-(t^2)) := by
      rw [mul_assoc, ← erfKernel_scaled, erfKernel_standard_formula]
      ring_nf
    rwa [hv] at he
  have hi := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hd t)
    ((by fun_prop : Continuous (fun t : ℝ => (2 / Real.sqrt Real.pi) * Real.exp (-(t^2)))).intervalIntegrable 0 x)
  rw [erf_zero, sub_zero, intervalIntegral.integral_const_mul] at hi
  exact hi.symm

end QIRadon.Activations
