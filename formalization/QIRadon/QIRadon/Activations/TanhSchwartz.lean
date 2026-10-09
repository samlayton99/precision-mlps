import QIRadon.Activations.TanhFourier
import Mathlib.Analysis.Distribution.SchwartzSpace.Deriv

/-! The actual normalized tanh derivative kernel is Schwartz. Each derivative
is the kernel times a polynomial in tanh(x); the bounded tanh range and explicit
exponential kernel decay prove every polynomially weighted derivative bound. -/
noncomputable section
open Polynomial
open scoped Real SchwartzMap
namespace QIRadon.Activations

/-- The actual tanh derivative, in the polynomial form used by the recurrence. -/
theorem tanh_hasDerivAt_polynomial (x : ℝ) :
    HasDerivAt Real.tanh (1-Real.tanh x^2) x := by
  have he : 2*tanhKernelValue x = 1-Real.tanh x^2 := by
    unfold tanhKernelValue
    rw [Real.tanh_eq_sinh_div_cosh]
    have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
    field_simp
    nlinarith [Real.cosh_sq_sub_sinh_sq x]
  rw [← he]
  exact tanh_hasDerivAt x

theorem tanhKernelValue_contDiff (n : WithTop ℕ∞) : ContDiff ℝ n tanhKernelValue := by
  unfold tanhKernelValue
  exact contDiff_const.div (contDiff_const.mul (Real.contDiff_cosh.pow 2)) (by intro x; positivity)

theorem tanh_contDiff (n : WithTop ℕ∞) : ContDiff ℝ n Real.tanh := by
  have he : Real.tanh = fun x : ℝ => Real.sinh x/Real.cosh x := funext Real.tanh_eq_sinh_div_cosh
  rw [he]
  exact Real.contDiff_sinh.div Real.contDiff_cosh (fun x => (Real.cosh_pos x).ne')

/-- The normalized kernel obeys the exact first-order derivative identity. -/
theorem tanhKernelValue_hasDerivAt (x : ℝ) :
    HasDerivAt tanhKernelValue (-2*Real.tanh x*tanhKernelValue x) x := by
  have hden := ((Real.hasDerivAt_cosh x).pow 2).const_mul 2
  have hh := (hasDerivAt_const x (1:ℝ)).div hden
    (show 2*Real.cosh x^2 ≠ 0 by positivity)
  convert! hh using 1
  unfold tanhKernelValue
  rw [Real.tanh_eq_sinh_div_cosh]
  simp only [Pi.pow_apply, Nat.cast_ofNat, Nat.reduceSub, pow_one]
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  field_simp
  ring

/-- P_0=1 and P_(n+1)=(1-t²)P_n'−2tP_n. -/
def tanhDerivativePolynomial : ℕ → ℝ[X]
  | 0 => 1
  | n+1 => (1-X^2)*(tanhDerivativePolynomial n).derivative -
      (2:ℝ[X])*X*tanhDerivativePolynomial n

/-- A polynomial-times-kernel derivative stays in the same explicit family. -/
theorem tanhKernel_polynomial_hasDerivAt (p : ℝ[X]) (x : ℝ) :
    HasDerivAt (fun y => tanhKernelValue y*p.eval (Real.tanh y))
      (tanhKernelValue x*((1-X^2)*p.derivative-(2:ℝ[X])*X*p).eval (Real.tanh x)) x := by
  have hp := (p.hasDerivAt (Real.tanh x)).comp x (tanh_hasDerivAt_polynomial x)
  have hh := (tanhKernelValue_hasDerivAt x).mul hp
  convert! hh using 1
  simp only [Polynomial.eval_sub, Polynomial.eval_mul, Polynomial.eval_pow,
    Polynomial.eval_one, Polynomial.eval_X, Polynomial.eval_ofNat, Function.comp_apply]
  ring

/-- Formula for every actual iterated derivative, not an assumed envelope. -/
theorem tanhKernelValue_iteratedDeriv (n : ℕ) :
    iteratedDeriv n tanhKernelValue =
      fun x => tanhKernelValue x*(tanhDerivativePolynomial n).eval (Real.tanh x) := by
  induction n with
  | zero => simp [iteratedDeriv_zero, tanhDerivativePolynomial]
  | succ n ih =>
    rw [iteratedDeriv_succ, ih]
    funext x
    exact (tanhKernel_polynomial_hasDerivAt (tanhDerivativePolynomial n) x).deriv

/-- A polynomial on [-1,1] is bounded by the absolute sum of its coefficients. -/
theorem polynomial_unit_abs_bound (p : ℝ[X]) {y : ℝ} (hy : |y| ≤ 1) :
    |p.eval y| ≤ ∑ j ∈ p.support, |p.coeff j| := by
  rw [Polynomial.eval_eq_sum]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  apply Finset.sum_le_sum
  intro j _
  rw [abs_mul, abs_pow]
  have hp : |y|^j ≤ 1 := pow_le_one₀ (abs_nonneg y) hy
  exact (mul_le_mul_of_nonneg_left hp (abs_nonneg _)).trans_eq (mul_one _)

/-- The actual spatial tail used by the tanh envelope theorem. -/
theorem tanhKernelValue_spatial_bound (x : ℝ) :
    |tanhKernelValue x| ≤ 2*Real.exp (-(2*|x|)) := by
  have hc : 0 < Real.cosh x := Real.cosh_pos x
  have hhalf : Real.exp |x|/2 ≤ Real.cosh x := by
    rw [← Real.cosh_abs, Real.cosh_eq]
    linarith [Real.exp_pos (-|x|)]
  have hs := mul_self_le_mul_self (show 0 ≤ Real.exp |x|/2 by positivity) hhalf
  have he : Real.exp (2*|x|) = (Real.exp |x|)^2 := by
    rw [show 2*|x|=|x|+|x| by ring, Real.exp_add, pow_two]
  rw [tanhKernelValue, abs_of_nonneg (by positivity : 0 ≤ 1/(2*Real.cosh x^2))]
  calc
    _ ≤ 2/(Real.exp |x|)^2 := by
      apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
      nlinarith
    _ = _ := by rw [Real.exp_neg, he]; ring

/-- Every polynomially weighted value of K has an explicit uniform bound. -/
theorem tanhKernelValue_weighted_bound (k : ℕ) (x : ℝ) :
    |x|^k*|tanhKernelValue x| ≤ 2*(k.factorial:ℝ) := by
  have hf : (0:ℝ) < k.factorial := by positivity
  have hp := (div_le_iff₀ hf).mp (Real.pow_div_factorial_le_exp |x| (abs_nonneg x) k)
  have hm := mul_le_mul_of_nonneg_right hp (Real.exp_pos (-|x|)).le
  have he : Real.exp |x| *Real.exp (-|x|)=1 := by rw [← Real.exp_add]; simp
  have hexp : Real.exp (-(2*|x|)) ≤ Real.exp (-|x|) := Real.exp_le_exp.mpr (by linarith [abs_nonneg x])
  calc
    _ ≤ |x|^k*(2*Real.exp (-|x|)) := mul_le_mul_of_nonneg_left
      ((tanhKernelValue_spatial_bound x).trans (mul_le_mul_of_nonneg_left hexp (by norm_num))) (by positivity)
    _ = 2*(|x|^k*Real.exp (-|x|)) := by ring
    _ ≤ 2*(k.factorial:ℝ) := by nlinarith [he]

/-- The actual tanh derivative kernel, now bundled with all Schwartz proofs. -/
def tanhKernel : 𝓢(ℝ, ℝ) where
  toFun := tanhKernelValue
  smooth' := tanhKernelValue_contDiff _
  decay' k n := by
    let p := tanhDerivativePolynomial n
    let B := ∑ j ∈ p.support, |p.coeff j|
    refine ⟨2*(k.factorial:ℝ)*B, ?_⟩
    intro x
    rw [norm_iteratedFDeriv_eq_norm_iteratedDeriv, tanhKernelValue_iteratedDeriv]
    simp only [Real.norm_eq_abs, abs_mul]
    have hp : |p.eval (Real.tanh x)| ≤ B :=
      polynomial_unit_abs_bound p (Real.abs_tanh_lt_one x).le
    have hk := tanhKernelValue_weighted_bound k x
    change |x|^k*(|tanhKernelValue x| *|p.eval (Real.tanh x)|) ≤ 2*(k.factorial:ℝ)*B
    calc
      _ = (|x|^k*|tanhKernelValue x|)*|p.eval (Real.tanh x)| := by ring
      _ ≤ _ := mul_le_mul hk hp (abs_nonneg _) (by positivity)

@[simp] theorem tanhKernel_apply (x : ℝ) : tanhKernel x = 1/(2*Real.cosh x^2) := rfl

/-- Activation/kernel matching for the actual Schwartz witness. -/
theorem tanh_first_derivative (x : ℝ) : iteratedDeriv 1 Real.tanh x = 2*tanhKernel x := by
  rw [iteratedDeriv_one]
  exact (tanh_hasDerivAt x).deriv

end QIRadon.Activations
