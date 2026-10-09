import QIRadon.Kernel.KernelRates
import Mathlib.Analysis.Distribution.SchwartzSpace.Deriv
import Mathlib.Analysis.SpecialFunctions.Gaussian.FourierTransform
import Mathlib.RingTheory.Polynomial.Hermite.Gaussian
import Mathlib

noncomputable section
open MeasureTheory FourierTransform Polynomial
open scoped Real SchwartzMap

namespace QIRadon.Activations

/-- An explicit uniform bound for every polynomially weighted Gaussian monomial. -/
theorem gaussian_monomial_bound (k : ℕ) (x : ℝ) :
    |x|^k * Real.exp (-(x^2/2)) ≤ 1 + 2^k * (k.factorial : ℝ) := by
  let y := x^2/2
  have hy : 0 ≤ y := by dsimp [y]; positivity
  have hsmall : Real.exp (-y) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hcancel : Real.exp y * Real.exp (-y) = 1 := by rw [← Real.exp_add]; simp
  have hp := Real.pow_div_factorial_le_exp y hy k
  have hf : 0 < (k.factorial : ℝ) := by positivity
  have hpow := (div_le_iff₀ hf).mp hp
  have hm := mul_le_mul_of_nonneg_right hpow (Real.exp_pos (-y)).le
  have hm' : y^k * Real.exp (-y) ≤ (k.factorial : ℝ) := by nlinarith [hcancel]
  have hsimple : |x|^k ≤ 1 + (|x|^k)^2 := by nlinarith [sq_nonneg (|x|^k - 1/2)]
  have hid : (|x|^k)^2 = (2 : ℝ)^k * y^k := by
    rw [← pow_mul, mul_comm k 2, pow_mul, sq_abs, ← mul_pow]
    dsimp [y]
    congr 1
    ring
  rw [hid] at hsimple
  calc
    _ ≤ (1 + (2 : ℝ)^k * y^k) * Real.exp (-y) :=
      mul_le_mul_of_nonneg_right hsimple (Real.exp_pos _).le
    _ = Real.exp (-y) + (2 : ℝ)^k * (y^k * Real.exp (-y)) := by ring
    _ ≤ 1 + (2 : ℝ)^k * (k.factorial : ℝ) :=
      add_le_add hsmall (mul_le_mul_of_nonneg_left hm' (by positivity))

/-- Polynomial factors preserve all Gaussian weighted bounds. -/
theorem polynomial_gaussian_bound (p : ℝ[X]) (k : ℕ) :
    ∃ C : ℝ, ∀ x : ℝ, |x|^k * |p.eval x| * Real.exp (-(x^2/2)) ≤ C := by
  refine ⟨∑ i ∈ p.support, |p.coeff i| * (1 + 2^(k+i) * ((k+i).factorial : ℝ)), ?_⟩
  intro x
  have hnorm : |p.eval x| ≤ ∑ i ∈ p.support, |p.coeff i * x^i| := by
    rw [Polynomial.eval_eq_sum]
    exact Finset.abs_sum_le_sum_abs _ _
  calc
    _ ≤ |x|^k * (∑ i ∈ p.support, |p.coeff i * x^i|) * Real.exp (-(x^2/2)) := by
      gcongr
    _ = ∑ i ∈ p.support, |p.coeff i| * (|x|^(k+i) * Real.exp (-(x^2/2))) := by
      rw [Finset.mul_sum, Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_pow, pow_add]
      ring
    _ ≤ _ := Finset.sum_le_sum (fun i _ =>
      mul_le_mul_of_nonneg_left (gaussian_monomial_bound (k+i) x) (abs_nonneg _))

/-- A concrete nonzero Schwartz Gaussian, derived using the actual Hermite
formula for every derivative rather than assuming rapid decay. -/
def standardGaussianSchwartz : 𝓢(ℝ, ℝ) where
  toFun x := Real.exp (-(x^2/2))
  smooth' := by fun_prop
  decay' k n := by
    let p : ℝ[X] := (Polynomial.hermite n).map (Int.castRingHom ℝ)
    obtain ⟨C, hC⟩ := polynomial_gaussian_bound p k
    refine ⟨C, ?_⟩
    intro x
    rw [norm_iteratedFDeriv_eq_norm_iteratedDeriv, iteratedDeriv_eq_iterate,
      Polynomial.deriv_gaussian_eq_hermite_mul_gaussian]
    have he : Polynomial.aeval x (Polynomial.hermite n) = p.eval x := by
      simp [p, Polynomial.aeval_def, Polynomial.eval_map]
    rw [he]
    simpa only [Real.norm_eq_abs, abs_mul, abs_pow, abs_neg, abs_one, one_pow,
      one_mul, abs_of_pos (Real.exp_pos _), mul_assoc] using hC x

@[simp] theorem standardGaussianSchwartz_apply (x : ℝ) :
    standardGaussianSchwartz x = Real.exp (-(x^2/2)) := rfl

/-- Unit-mass self-Fourier Gaussian in the cycles convention. -/
def gaussianKernel : 𝓢(ℝ, ℝ) :=
  SchwartzMap.compCLMOfContinuousLinearEquiv ℝ
    (ContinuousLinearEquiv.unitsEquivAut ℝ
      (Units.mk0 (Real.sqrt (2 * Real.pi)) (by positivity))) standardGaussianSchwartz

@[simp] theorem gaussianKernel_apply (x : ℝ) :
    gaussianKernel x = Real.exp (-Real.pi * x^2) := by
  simp only [gaussianKernel, SchwartzMap.compCLMOfContinuousLinearEquiv_apply]
  simp only [Function.comp_apply, ContinuousLinearEquiv.unitsEquivAut_apply,
    Units.val_mk0, standardGaussianSchwartz_apply]
  rw [mul_pow, Real.sq_sqrt (by positivity)]
  congr 1
  ring

theorem gaussianKernel_mass : (∫ x : ℝ, gaussianKernel x) = 1 := by
  simp only [gaussianKernel_apply]
  rw [integral_gaussian]
  simp [Real.pi_ne_zero]

theorem gaussianKernel_fourier (ξ : ℝ) :
    𝓕 (fun x : ℝ => (gaussianKernel x : ℂ)) ξ = (Real.exp (-Real.pi * ξ^2) : ℂ) := by
  have hfun : (fun x : ℝ => (gaussianKernel x : ℂ)) =
      (fun x : ℝ => Complex.exp (-(Real.pi : ℂ) * (1 : ℂ) * (x : ℂ)^2)) := by
    funext x
    rw [gaussianKernel_apply, Complex.ofReal_exp]
    congr 1
    push_cast
    ring
  rw [hfun, fourier_gaussian_pi (b := (1 : ℂ)) (by norm_num)]
  simp only [Complex.one_cpow, one_mul, div_one]
  rw [Complex.ofReal_exp]
  congr 1
  push_cast
  ring

/-- The primary fast-decay class is inhabited by an actual normalized kernel. -/
def gaussianSpatial : KernelRates.SpatialEnvelope gaussianKernel 2 where
  C := 1
  b := Real.pi
  C_pos := by norm_num
  b_pos := Real.pi_pos
  exponent_pos := by norm_num
  upper x := by
    rw [gaussianKernel_apply, abs_of_pos (Real.exp_pos _), Real.rpow_two, sq_abs, one_mul]

/-- Both spectral envelopes are exact for this Gaussian. -/
def gaussianSpectral : KernelRates.SpectralEnvelope
    (𝓕 (fun x : ℝ => (gaussianKernel x : ℂ))) 2 where
  c := 1
  C := 1
  A := Real.pi
  a := Real.pi
  c_pos := by norm_num
  C_pos := by norm_num
  A_nonneg := Real.pi_pos.le
  a_pos := Real.pi_pos
  exponent_pos := by norm_num
  lower ξ := by
    rw [gaussianKernel_fourier, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), Real.rpow_two, sq_abs, one_mul]
  upper ξ := by
    rw [gaussianKernel_fourier, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), Real.rpow_two, sq_abs, one_mul]

/-- The matched Gaussian class has κ=1 in the proved general rate theorem. -/
theorem gaussian_rate_exponent : max (1 : ℝ) (1/(2:ℝ) + 1/(2:ℝ)) = 1 := by norm_num

end QIRadon.Activations
