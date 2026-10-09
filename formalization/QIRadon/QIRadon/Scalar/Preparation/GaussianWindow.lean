import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.Basic
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-! Explicit complex growth and Gaussian decay of the finite flat window.
The truncated exponential is bounded without discarding the remaining
horizontal Gaussian decay. -/

namespace QIRadon.ProfilePreparation

open Polynomial Finset
open Polynomial.Chebyshev

/-- A direct complex-plane bound for Chebyshev basis polynomials. -/
theorem complex_chebyshev_bound (n : ℕ) (z : ℂ) :
    ‖(T ℂ n).eval z‖ ≤ (1 + 2 * ‖z‖) ^ n := by
  induction n using Nat.twoStepInduction with
  | zero => simp
  | one =>
    simp only [Nat.cast_one, T_one, Polynomial.eval_X, pow_one]
    linarith [norm_nonneg z]
  | more n hn hn1 =>
    rw [show ((n + 2 : ℕ) : ℤ) = (n : ℤ) + 2 by push_cast; rfl, T_add_two,
      Polynomial.eval_sub, Polynomial.eval_mul, Polynomial.eval_mul,
      Polynomial.eval_ofNat, Polynomial.eval_X]
    have hterm : ‖(2 : ℂ) * z * (T ℂ ((n : ℤ) + 1)).eval z‖ =
        2 * ‖z‖ * ‖(T ℂ (n + 1)).eval z‖ := by
      rw [norm_mul, norm_mul]
      norm_num
    have htri := norm_sub_le ((2 : ℂ) * z * (T ℂ ((n : ℤ) + 1)).eval z) ((T ℂ n).eval z)
    rw [hterm] at htri
    have hm := mul_le_mul_of_nonneg_left hn1 (by positivity : 0 ≤ 2 * ‖z‖)
    apply htri.trans
    calc
      _ ≤ 2 * ‖z‖ * (1 + 2 * ‖z‖) ^ (n + 1) + (1 + 2 * ‖z‖) ^ n := add_le_add hm hn
      _ ≤ (1 + 2 * ‖z‖) ^ (n + 2) := by
        rw [pow_succ, pow_add]
        have hpos : 0 ≤ (1 + 2 * ‖z‖) ^ n * ‖z‖ := by positivity
        nlinarith

/-- The finite exponential sum is bounded by a narrower exponential, paying
only 2^n. This retains half of the Gaussian spatial decay. -/
theorem finite_exp_sum_scaled_bound (n : ℕ) {x : ℝ} (hx : 0 ≤ x) :
    (∑ k ∈ range (n + 1), x ^ k / (k.factorial : ℝ)) ≤
      (2 : ℝ) ^ n * Real.exp (x / 2) := by
  have hterm : ∀ k ∈ range (n + 1),
      x ^ k / (k.factorial : ℝ) ≤ (2 : ℝ) ^ n * ((x / 2) ^ k / (k.factorial : ℝ)) := by
    intro k hk
    have hkN : k ≤ n := mem_range_succ_iff.mp hk
    have hp : (2 : ℝ) ^ k ≤ (2 : ℝ) ^ n := pow_le_pow_right₀ (by norm_num) hkN
    have hm := mul_le_mul_of_nonneg_right hp (by positivity : 0 ≤ (x / 2) ^ k / (k.factorial : ℝ))
    convert! hm using 1
    rw [← mul_div_assoc, ← mul_pow]
    congr 1
    ring
  calc
    _ ≤ ∑ k ∈ range (n + 1), (2 : ℝ) ^ n * ((x / 2) ^ k / (k.factorial : ℝ)) := sum_le_sum hterm
    _ = (2 : ℝ) ^ n * ∑ k ∈ range (n + 1), ((x / 2) ^ k / (k.factorial : ℝ)) := by rw [mul_sum]
    _ ≤ (2 : ℝ) ^ n * Real.exp (x / 2) :=
      mul_le_mul_of_nonneg_left (Real.sum_le_exp_of_nonneg (by positivity) (n + 1)) (by positivity)

/-- The complex continuation of the actual finite Gaussian-polynomial window. -/
noncomputable def complexFlatWindow (n : ℕ) (z : ℂ) : ℂ :=
  Complex.exp (-(n : ℂ) * z ^ 2) *
    ∑ k ∈ range (n + 1), ((n : ℂ) * z ^ 2) ^ k / (k.factorial : ℂ)

/-- Global bound exposing Gaussian decay along EVERY fixed horizontal line. -/
theorem complexFlatWindow_bound (n : ℕ) (z : ℂ) :
    ‖complexFlatWindow n z‖ ≤
      (2 : ℝ) ^ n * Real.exp (-(n : ℝ) * z.re ^ 2 / 2 + 3 * n * z.im ^ 2 / 2) := by
  have hsum : ‖∑ k ∈ range (n + 1), ((n : ℂ) * z ^ 2) ^ k / (k.factorial : ℂ)‖ ≤
      (2 : ℝ) ^ n * Real.exp (((n : ℝ) * ‖z‖ ^ 2) / 2) := by
    apply (norm_sum_le _ _).trans
    have heq : (∑ k ∈ range (n + 1), ‖((n : ℂ) * z ^ 2) ^ k / (k.factorial : ℂ)‖) =
        ∑ k ∈ range (n + 1), ((n : ℝ) * ‖z‖ ^ 2) ^ k / (k.factorial : ℝ) := by
      apply sum_congr rfl
      intro k hk
      simp only [norm_div, norm_pow, norm_mul, Complex.norm_natCast]
    rw [heq]
    exact finite_exp_sum_scaled_bound n (by positivity)
  unfold complexFlatWindow
  rw [norm_mul, Complex.norm_exp]
  have hm := mul_le_mul_of_nonneg_left hsum
    (Real.exp_pos (-(n : ℂ) * z ^ 2).re).le
  apply hm.trans_eq
  rw [mul_left_comm, ← Real.exp_add]
  congr 2
  rw [Complex.sq_norm]
  simp only [pow_two, Complex.mul_re, Complex.neg_re, Complex.neg_im,
    Complex.natCast_re, Complex.natCast_im, neg_zero, zero_mul, sub_zero,
    Complex.normSq_apply]
  ring

/-- Linear growth can be absorbed into a Gaussian, with an explicit constant. -/
theorem abs_le_gaussian (x : ℝ) {T : ℝ} (hT : 0 < T) :
    |x| ≤ 2 * T * Real.exp (x ^ 2 / (4 * T ^ 2)) := by
  let y : ℝ := |x| / (2 * T)
  have hy : 0 ≤ y := by dsimp [y]; positivity
  have hpoly : y ≤ 1 + y ^ 2 := by nlinarith [sq_nonneg (y - 1)]
  have he := Real.add_one_le_exp (y ^ 2)
  have hm := mul_le_mul_of_nonneg_left (hpoly.trans (by linarith)) (by positivity : 0 ≤ 2 * T)
  have hyid : (2 * T) * y = |x| := by dsimp [y]; field_simp
  have hysq : y ^ 2 = x ^ 2 / (4 * T ^ 2) := by
    dsimp [y]
    rw [div_pow, sq_abs]
    ring
  simpa only [hyid, hysq] using hm

/-- The particular polynomial growth factor coming from the Chebyshev bound
can be absorbed without losing all horizontal Gaussian decay. -/
theorem chebyshev_growth_gaussian {x y T σ : ℝ}
    (hT : 0 < T) (hσ : 0 ≤ σ) (hy : |y| ≤ σ) :
    1 + 2 * ‖(x : ℂ) + y * Complex.I‖ ≤
      (1 + 2 * σ + 4 * T) * Real.exp (x ^ 2 / (4 * T ^ 2)) := by
  have hn := Complex.norm_le_abs_re_add_abs_im ((x : ℂ) + y * Complex.I)
  simp only [Complex.add_re, Complex.add_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im, mul_zero,
    mul_one, zero_add, add_zero, sub_zero] at hn
  have hx := abs_le_gaussian x hT
  have he : 1 ≤ Real.exp (x ^ 2 / (4 * T ^ 2)) := Real.one_le_exp_iff.mpr (by positivity)
  have hm := mul_le_mul_of_nonneg_left he (show 0 ≤ 1 + 2 * σ by positivity)
  nlinarith

end QIRadon.ProfilePreparation
