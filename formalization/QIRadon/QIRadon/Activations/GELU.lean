import QIRadon.Activations.Erf
import Mathlib.Analysis.Distribution.SchwartzSpace.Fourier

/-! Exact standard GELU: its second derivative is a signed Gaussian-polynomial
kernel, and two ordinary GELU neurons reproduce every affine anchor. -/

noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap
namespace QIRadon.Activations

def gelu (x : ℝ) : ℝ := x * normalCDF x

theorem normalPDF_hasDerivAt (x : ℝ) : HasDerivAt normalPDF (-x * normalPDF x) x := by
  have hh := ((hasDerivAt_id x).pow 2 |>.div_const 2 |>.neg).exp.const_mul
    (Real.sqrt (2 * Real.pi))⁻¹
  convert! hh using 1
  simp only [normalPDF_apply, Pi.neg_apply, Pi.pow_apply, id_eq]
  ring_nf

@[simp] theorem normalPDF_deriv : deriv normalPDF = fun x => -x * normalPDF x := by
  funext x
  exact (normalPDF_hasDerivAt x).deriv

theorem normalPDF_deriv_hasDerivAt (x : ℝ) :
    HasDerivAt (deriv normalPDF) ((x^2-1) * normalPDF x) x := by
  rw [normalPDF_deriv]
  convert! ((hasDerivAt_id x).neg.mul (normalPDF_hasDerivAt x)) using 1
  simp only [Pi.neg_apply, id_eq]
  ring_nf

@[simp] theorem normalPDF_deriv_deriv :
    deriv (deriv normalPDF) = fun x => (x^2-1) * normalPDF x := by
  funext x
  exact (normalPDF_deriv_hasDerivAt x).deriv

theorem gelu_hasDerivAt (x : ℝ) :
    HasDerivAt gelu (normalCDF x + x * normalPDF x) x := by
  convert! (hasDerivAt_id x).mul (normalCDF_hasDerivAt x) using 1
  simp

@[simp] theorem gelu_deriv : deriv gelu = fun x => normalCDF x + x * normalPDF x := by
  funext x
  exact (gelu_hasDerivAt x).deriv

theorem gelu_deriv_hasDerivAt (x : ℝ) :
    HasDerivAt (deriv gelu) ((2-x^2) * normalPDF x) x := by
  rw [gelu_deriv]
  convert! (normalCDF_hasDerivAt x).add ((hasDerivAt_id x).mul (normalPDF_hasDerivAt x)) using 1
  simp only [id_eq]
  ring_nf

/-- Actual Schwartz kernel, not merely an unbundled formula assumed to decay. -/
def geluKernel : 𝓢(ℝ, ℝ) :=
  normalPDF - SchwartzMap.derivCLM ℝ ℝ (SchwartzMap.derivCLM ℝ ℝ normalPDF)

@[simp] theorem geluKernel_apply (x : ℝ) : geluKernel x = (2-x^2) * normalPDF x := by
  simp only [geluKernel, sub_apply, SchwartzMap.derivCLM_apply]
  change normalPDF x - deriv (deriv normalPDF) x = _
  rw [normalPDF_deriv_deriv]
  ring_nf

theorem gelu_second_derivative (x : ℝ) : iteratedDeriv 2 gelu x = geluKernel x := by
  rw [iteratedDeriv_succ, iteratedDeriv_one, geluKernel_apply]
  exact (gelu_deriv_hasDerivAt x).deriv

theorem gelu_contDiff (n : ℕ) : ContDiff ℝ n gelu :=
  contDiff_id.mul (normalCDF_contDiff n)

/-- The affine anchor uses two standard GELU neurons and one output bias. -/
theorem gelu_sub_reflection (x : ℝ) : gelu x - gelu (-x) = x := by
  simp only [gelu, normalCDF_neg]
  ring_nf

theorem gelu_affine_anchor (a b x : ℝ) : a * gelu x - a * gelu (-x) + b = a * x + b := by
  rw [← mul_sub, gelu_sub_reflection]

/-- Complexification preserves the real derivative, including the second
Schwartz derivative needed for the Fourier multiplier calculation. -/
theorem complexifiedSchwartz_deriv (f : 𝓢(ℝ, ℝ)) :
    deriv (fun x : ℝ => (f x : ℂ)) = fun x : ℝ => ((SchwartzMap.derivCLM ℝ ℝ f) x : ℂ) := by
  funext x
  exact (Complex.ofRealCLM.hasFDerivAt.comp_hasDerivAt x (f.hasDerivAt x)).deriv

theorem complexifiedSchwartz_fourier_deriv (f : 𝓢(ℝ, ℝ)) (ξ : ℝ) :
    𝓕 (fun x : ℝ => ((SchwartzMap.derivCLM ℝ ℝ f) x : ℂ)) ξ =
      (2 * Real.pi * Complex.I * (ξ : ℂ)) * 𝓕 (fun x : ℝ => (f x : ℂ)) ξ := by
  have hf : Integrable (fun x : ℝ => (f x : ℂ)) :=
    Complex.ofRealCLM.integrable_comp (f.integrable (μ := volume))
  have hdf : Integrable (fun x : ℝ => ((SchwartzMap.derivCLM ℝ ℝ f) x : ℂ)) :=
    Complex.ofRealCLM.integrable_comp ((SchwartzMap.derivCLM ℝ ℝ f).integrable (μ := volume))
  have hdiff : Differentiable ℝ (fun x : ℝ => (f x : ℂ)) :=
    Complex.ofRealCLM.differentiable.comp f.differentiable
  have hfourier := Real.fourier_deriv hf hdiff
    (by simpa only [complexifiedSchwartz_deriv] using hdf)
  rw [complexifiedSchwartz_deriv] at hfourier
  exact congrFun hfourier ξ

/-- Actual standard-GELU spectrum: it is strictly positive and has no notch. -/
theorem geluKernel_fourier (ξ : ℝ) :
    𝓕 (fun x : ℝ => (geluKernel x : ℂ)) ξ =
      (((1 + (2 * Real.pi * ξ)^2) * Real.exp (-(2 * Real.pi^2) * ξ^2) : ℝ) : ℂ) := by
  let f : 𝓢(ℝ, ℂ) := normalPDF.postcompCLM Complex.ofRealCLM
  let ddf : 𝓢(ℝ, ℂ) :=
    (SchwartzMap.derivCLM ℝ ℝ (SchwartzMap.derivCLM ℝ ℝ normalPDF)).postcompCLM Complex.ofRealCLM
  have hfun : (fun x : ℝ => (geluKernel x : ℂ)) = (fun x : ℝ => (f - ddf) x) := by
    funext x
    simp [f, ddf, geluKernel, SchwartzMap.postcompCLM_apply]
  have hsub : 𝓕 (f - ddf) = 𝓕 f - 𝓕 ddf :=
    (SchwartzMap.fourierTransformCLM ℂ).map_sub f ddf
  rw [hfun, ← SchwartzMap.fourier_coe, hsub]
  change 𝓕 (fun x : ℝ => (normalPDF x : ℂ)) ξ -
    𝓕 (fun x : ℝ => ((SchwartzMap.derivCLM ℝ ℝ (SchwartzMap.derivCLM ℝ ℝ normalPDF)) x : ℂ)) ξ = _
  rw [complexifiedSchwartz_fourier_deriv, complexifiedSchwartz_fourier_deriv, normalPDF_fourier]
  simp only [Complex.ofReal_mul, Complex.ofReal_add, Complex.ofReal_pow, Complex.ofReal_one,
    Complex.ofReal_ofNat]
  calc
    _ = (1 - (2 * Real.pi * Complex.I * (ξ : ℂ))^2) *
      (Real.exp (-(2 * Real.pi^2) * ξ^2) : ℂ) := by ring_nf
    _ = _ := by rw [mul_pow, mul_pow, mul_pow, Complex.I_sq]; ring_nf


/-- Elementary polynomial-factor absorption with explicit constants. -/
theorem linear_gaussian_envelope {a b y : ℝ}
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hy : 0 ≤ y) :
    (a + b*y) * Real.exp (-2*y) ≤ (a+b) * Real.exp (-y) := by
  have h1 : 1 ≤ Real.exp y := Real.one_le_exp_iff.mpr hy
  have h2 : y ≤ Real.exp y := by linarith [Real.add_one_le_exp y]
  have hp : a+b*y ≤ (a+b)*Real.exp y := by nlinarith
  have hm := mul_le_mul_of_nonneg_right hp (Real.exp_pos (-2*y)).le
  have he : Real.exp y * Real.exp (-2*y) = Real.exp (-y) := by
    rw [← Real.exp_add]
    congr 1
    ring_nf
  simpa only [mul_assoc, he] using hm

theorem geluKernel_spatial_bound (x : ℝ) :
    |geluKernel x| ≤ (6 * (Real.sqrt (2 * Real.pi))⁻¹) * Real.exp (-(1/4 : ℝ) * |x|^2) := by
  rw [geluKernel_apply, abs_mul, abs_of_pos (normalPDF_pos x), normalPDF_apply]
  have hp : |2-x^2| ≤ 2+x^2 := by
    have hh := abs_sub_le (2 : ℝ) 0 (x^2)
    simpa only [sub_zero, zero_sub, abs_neg, abs_of_nonneg (sq_nonneg x),
      abs_of_pos (by norm_num : (0 : ℝ) < 2)] using hh
  have he := linear_gaussian_envelope (a := 2) (b := 4) (y := x^2/4)
    (by norm_num) (by norm_num) (by positivity)
  have hm := mul_le_mul_of_nonneg_left he (show 0 ≤ (Real.sqrt (2 * Real.pi))⁻¹ by positivity)
  calc
    _ ≤ (2+x^2) * ((Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-(x^2/2))) := by gcongr
    _ = (Real.sqrt (2 * Real.pi))⁻¹ * ((2+4*(x^2/4)) * Real.exp (-2*(x^2/4))) := by
      rw [show -2 * (x^2 / 4) = -(x^2 / 2) by ring_nf]
      ring_nf
    _ ≤ (Real.sqrt (2 * Real.pi))⁻¹ * ((2+4) * Real.exp (-(x^2/4))) := hm
    _ = _ := by rw [sq_abs]; ring_nf

def geluSpatial : KernelRates.SpatialEnvelope geluKernel 2 where
  C := 6 * (Real.sqrt (2 * Real.pi))⁻¹
  b := 1/4
  C_pos := by positivity
  b_pos := by norm_num
  exponent_pos := by norm_num
  upper x := by simpa only [Real.rpow_two] using geluKernel_spatial_bound x

theorem geluKernel_spectral_upper (ξ : ℝ) :
    ‖𝓕 (fun x : ℝ => (geluKernel x : ℂ)) ξ‖ ≤ 5 * Real.exp (-Real.pi^2 * |ξ|^2) := by
  rw [geluKernel_fourier, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  have hh := linear_gaussian_envelope (a := 1) (b := 4) (y := Real.pi^2 * ξ^2)
    (by norm_num) (by norm_num) (by positivity)
  simp only [sq_abs]
  convert! hh using 1 <;> congr 1 <;> ring_nf

theorem geluKernel_spectral_lower (ξ : ℝ) :
    Real.exp (-(2 * Real.pi^2) * |ξ|^2) ≤
      ‖𝓕 (fun x : ℝ => (geluKernel x : ℂ)) ξ‖ := by
  rw [geluKernel_fourier, Complex.norm_real, Real.norm_eq_abs, sq_abs]
  rw [abs_of_nonneg (by positivity : 0 ≤ (1+(2*Real.pi*ξ)^2) * Real.exp (-(2*Real.pi^2)*ξ^2))]
  nlinarith [sq_nonneg (2 * Real.pi * ξ), Real.exp_pos (-(2 * Real.pi^2) * ξ^2)]

def geluSpectral : KernelRates.SpectralEnvelope
    (𝓕 (fun x : ℝ => (geluKernel x : ℂ))) 2 where
  c := 1
  C := 5
  A := 2 * Real.pi^2
  a := Real.pi^2
  c_pos := by norm_num
  C_pos := by norm_num
  A_nonneg := by positivity
  a_pos := by positivity
  exponent_pos := by norm_num
  lower ξ := by simpa only [Real.rpow_two, one_mul] using geluKernel_spectral_lower ξ
  upper ξ := by simpa only [Real.rpow_two] using geluKernel_spectral_upper ξ

theorem geluKernel_mass : (∫ x : ℝ, geluKernel x) = 1 := by
  have hh := geluKernel_fourier 0
  have hf : 𝓕 (fun x : ℝ => (geluKernel x : ℂ)) 0 =
      ((∫ x : ℝ, geluKernel x) : ℂ) := by
    simp [Real.fourier_real_eq]
  rw [hf] at hh
  norm_num only [mul_zero, neg_zero, Real.exp_zero, zero_pow, add_zero, mul_one, Complex.ofReal_one] at hh
  exact_mod_cast hh

/-- The complete order-two anchor is exactly an ordinary GELU readout with
two added neurons; there is no external polynomial feature. -/
theorem gelu_anchorPolynomial_two (a : ℕ → ℝ) (x : ℝ) :
    QIRadon.Kernel.anchorPolynomial 2 a x = a 0 + a 1 * gelu x - a 1 * gelu (-x) := by
  simp only [QIRadon.Kernel.anchorPolynomial, Finset.sum_range_succ, Finset.sum_range_zero,
    Nat.factorial_zero, Nat.cast_one, div_one, pow_zero, mul_one, zero_add,
    Nat.factorial_one, pow_one]
  rw [add_sub_assoc, ← mul_sub, gelu_sub_reflection]


/-- Exact equality with the conventional erf formula for standard GELU. -/
theorem gelu_standard_erf_formula (x : ℝ) :
    gelu x = (x / 2) * (1 + erf (x / Real.sqrt 2)) := by
  have hx : Real.sqrt 2 * (x / Real.sqrt 2) = x := by field_simp
  simp only [erf, hx, gelu]
  ring_nf

/-- Expand the complete anchored construction into the original finite GELU
sum plus two GELU neurons and an output bias. -/
theorem gelu_anchored_network (ρ g : ℝ → ℝ) (m γ h : ℝ) (J : Finset ℤ) (x : ℝ) :
    let S := QIRadon.Kernel.rawActivationNetwork gelu ρ 2 m γ h J
    QIRadon.Kernel.anchoredActivationNetwork gelu ρ 2 m γ h J g x =
      S x + (g 0 - S 0) + (deriv g 0 - deriv S 0) * gelu x -
        (deriv g 0 - deriv S 0) * gelu (-x) := by
  dsimp only [QIRadon.Kernel.anchoredActivationNetwork]
  rw [gelu_anchorPolynomial_two]
  simp only [iteratedDeriv_zero, iteratedDeriv_one]
  ring_nf

end QIRadon.Activations
