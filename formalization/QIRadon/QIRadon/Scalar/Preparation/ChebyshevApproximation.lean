import QIRadon.Scalar.Preparation.PeriodicStrip
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.Basic
import Mathlib.Analysis.Normed.Group.InfiniteSum
import Mathlib.Analysis.SpecificLimits.Basic

/-! Derivation of an analytic Chebyshev comparator from actual ellipse Fourier
coefficients. The finite sampled interpolation operator is connected in the
parent ProfilePreparation module. -/

namespace QIRadon.ProfilePreparation

open Polynomial Finset Set
open Polynomial.Chebyshev

/-- Averaging the real Fourier series at opposite angles gives a cosine series
for an even analytic profile. -/
theorem stripCoefficient_cosine_hasSum {F : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hF : AnalyticOnNhd ℂ F (closedStrip δ))
    (hp : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z)
    (hbound : ∀ z ∈ closedStrip δ, ‖F z‖ ≤ M)
    (heven : ∀ t : ℝ, F ((-t : ℝ) : ℂ) = F (t : ℂ)) (t : ℝ) :
    HasSum (fun k : ℤ => (stripCoefficient F k).re * Real.cos ((k : ℝ) * t))
      (F (t : ℂ)).re := by
  have h1 := stripCoefficient_hasSum hδ hF hp hbound t
  have h2 := stripCoefficient_hasSum hδ hF hp hbound (-t)
  rw [heven] at h2
  have hav := (h1.add h2).mul_left (1 / 2 : ℂ)
  have hcos : HasSum (fun k : ℤ => stripCoefficient F k * Complex.cos ((k : ℂ) * (t : ℂ)))
      (F (t : ℂ)) := by
    convert! hav using 1
    · ext k
      have hc := Complex.two_cos ((k : ℂ) * (t : ℂ))
      have he1 : ((k : ℂ) * (t : ℂ)) * Complex.I = (k : ℂ) * Complex.I * (t : ℂ) := by ring
      have he2 : -((k : ℂ) * (t : ℂ)) * Complex.I =
          (k : ℂ) * Complex.I * ((-t : ℝ) : ℂ) := by push_cast; ring
      rw [he1, he2] at hc
      linear_combination (stripCoefficient F k) * hc / 2
    · ring
  have hr := Complex.hasSum_re hcos
  convert! hr using 1
  ext k
  have hc : Complex.cos ((k : ℂ) * (t : ℂ)) = (Real.cos ((k : ℝ) * t) : ℂ) := by
    rw [show (k : ℂ) * (t : ℂ) = (((k : ℝ) * t : ℝ) : ℂ) by push_cast; rfl]
    exact (Complex.ofReal_cos _).symm
  rw [hc]
  simp only [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, mul_zero, sub_zero]

/-- The real polynomial assembled from a finite part of the cosine expansion.
Pairing natural and negative-successor indices gives a degree <= N polynomial. -/
noncomputable def cosinePolynomial (a : ℤ → ℂ) (N : ℕ) : ℝ[X] :=
  ∑ j ∈ range N,
    ((a (j : ℤ)).re • T ℝ (j : ℤ) +
    (a (-((j : ℤ) + 1))).re • T ℝ (-((j : ℤ) + 1)))

theorem cosinePolynomial_degree_le (a : ℤ → ℂ) (N : ℕ) :
    (cosinePolynomial a N).degree ≤ N := by
  unfold cosinePolynomial
  apply (Polynomial.degree_sum_le _ _).trans
  apply Finset.sup_le
  intro j hj
  have hjN : j < N := mem_range.mp hj
  apply Polynomial.degree_add_le_of_degree_le
  · apply (Polynomial.degree_smul_le _ _).trans
    simp only [degree_T, Int.natAbs_natCast]
    exact_mod_cast hjN.le
  · apply (Polynomial.degree_smul_le _ _).trans
    simp only [degree_T, Int.natAbs_neg]
    rw [show (j : ℤ) + 1 = ((j + 1 : ℕ) : ℤ) by norm_cast, Int.natAbs_natCast]
    exact_mod_cast hjN

theorem cosinePolynomial_eval_cos (a : ℤ → ℂ) (N : ℕ) (t : ℝ) :
    (cosinePolynomial a N).eval (Real.cos t) =
      ∑ j ∈ range N,
        ((a (j : ℤ)).re * Real.cos ((j : ℝ) * t) +
          (a (-((j : ℤ) + 1))).re * Real.cos ((-((j : ℤ) + 1) : ℤ) * t)) := by
  simp [cosinePolynomial, Polynomial.eval_finsetSum, Polynomial.eval_add,
    Polynomial.eval_smul, T_real_cos, smul_eq_mul]

/-- A quantitative remainder for a genuinely convergent series. -/
theorem geometric_series_remainder {u : ℕ → ℝ} {s C q : ℝ}
    (hs : HasSum u s) (_hC : 0 ≤ C) (hq : 0 ≤ q) (hq1 : q < 1)
    (hbound : ∀ n, |u n| ≤ C * q ^ n) (N : ℕ) :
    |s - ∑ j ∈ range N, u j| ≤ C * q ^ N / (1 - q) := by
  have hgeo := hasSum_geometric_of_lt_one hq hq1
  have hmajor := hgeo.mul_left (C * q ^ N)
  have hpoint : ∀ j : ℕ, |u (j + N)| ≤ (C * q ^ N) * q ^ j := by
    intro j
    have hb := hbound (j + N)
    rw [pow_add] at hb
    convert! hb using 1
    ring
  have habs := hmajor.summable.of_nonneg_of_le (fun j => abs_nonneg _) hpoint
  have ht : Summable (fun j => u (j + N)) := (summable_nat_add_iff N).mpr hs.summable
  have hid : s - ∑ j ∈ range N, u j = ∑' j, u (j + N) := by
    have he := hs.summable.sum_add_tsum_nat_add N
    rw [hs.tsum_eq] at he
    linarith
  rw [hid]
  calc
    _ ≤ ∑' j, |u (j + N)| := by
      simpa only [Real.norm_eq_abs] using norm_tsum_le_tsum_norm (f := fun j => u (j + N)) (by simpa only [Real.norm_eq_abs] using habs)
    _ ≤ ∑' j, (C * q ^ N) * q ^ j := habs.tsum_le_tsum hpoint hmajor.summable
    _ = C * q ^ N / (1 - q) := by
      rw [hmajor.tsum_eq]
      ring

/-- Turn the actual coefficient envelope into the natural geometric form. -/
theorem coefficient_exp_eq_pow (δ : ℝ) (k : ℤ) :
    Real.exp (-|(k : ℝ)| * δ) = (Real.exp (-δ)) ^ k.natAbs := by
  rw [← Real.exp_nat_mul]
  congr 1
  have hk : (k.natAbs : ℝ) = |(k : ℝ)| := by
    simpa only [Int.cast_natCast, Int.cast_abs] using
      congrArg (fun z : ℤ => (z : ℝ)) (Int.natCast_natAbs k)
  rw [hk]
  ring

/-- A Fourier cosine term is bounded by its coefficient norm. -/
theorem cosine_term_bound (a : ℂ) (t : ℝ) : |a.re * Real.cos t| ≤ ‖a‖ := by
  rw [abs_mul]
  exact (mul_le_mul (Complex.abs_re_le_norm a) (Real.abs_cos_le_one t)
    (abs_nonneg _) (norm_nonneg _)).trans_eq (mul_one _)

/-- Geometric polynomial approximation directly from the ellipse assumptions.
This theorem supplies the comparator required by the sampled-interpolation
bound; the comparator itself is not an input hypothesis. -/
theorem analytic_cosinePolynomial_error {g : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M)
    (N : ℕ) {x : ℝ} (hx : |x| ≤ 1) :
    |(g (x : ℂ)).re -
      (cosinePolynomial (stripCoefficient (fun z => g (Complex.cos z))) N).eval x| ≤
      (2 * M) * Real.exp (-δ * N) / (1 - Real.exp (-δ)) := by
  let F : ℂ → ℂ := fun z => g (Complex.cos z)
  let a : ℤ → ℂ := stripCoefficient F
  let t : ℝ := Real.arccos x
  let q : ℝ := Real.exp (-δ)
  have hM : 0 ≤ M := by
    have hz : (0 : ℂ) ∈ closedStrip δ := by simpa [closedStrip] using hδ.le
    exact (norm_nonneg _).trans (hbound (Complex.cos 0) ⟨0, hz, rfl⟩)
  have hq : 0 ≤ q := (Real.exp_pos _).le
  have hq1 : q < 1 := Real.exp_lt_one_iff.mpr (neg_neg_of_pos hδ)
  have hF := ellipse_composition_analytic hg
  have hperiod : ∀ z : ℂ, F (z + (2 * Real.pi : ℝ)) = F z := by
    intro z
    dsimp [F]
    congr 1
    simpa using Complex.cos_add_two_pi z
  have hFb : ∀ z ∈ closedStrip δ, ‖F z‖ ≤ M := by
    intro z hz
    exact hbound (Complex.cos z) ⟨z, hz, rfl⟩
  have heven : ∀ t : ℝ, F ((-t : ℝ) : ℂ) = F (t : ℂ) := by
    intro t
    simp [F]
  have hs := (stripCoefficient_cosine_hasSum hδ hF hperiod hFb heven t).nat_add_neg_add_one
  let u : ℕ → ℝ := fun j =>
    (a (j : ℤ)).re * Real.cos ((j : ℝ) * t) +
    (a (-((j : ℤ) + 1))).re * Real.cos ((-((j : ℤ) + 1) : ℤ) * t)
  have hcoeff : ∀ k : ℤ, ‖a k‖ ≤ M * q ^ k.natAbs := by
    intro k
    have hb := ellipseCoefficient_bound hδ.le hg hbound k
    rw [coefficient_exp_eq_pow] at hb
    exact hb
  have hu : ∀ j, |u j| ≤ (2 * M) * q ^ j := by
    intro j
    have hpos := (cosine_term_bound (a (j : ℤ)) ((j : ℝ) * t)).trans (hcoeff j)
    have hneg := (cosine_term_bound (a (-((j : ℤ) + 1))) ((-((j : ℤ) + 1) : ℤ) * t)).trans
      (hcoeff (-((j : ℤ) + 1)))
    simp only [Int.natAbs_natCast] at hpos
    have hneg' : |(a (-((j : ℤ) + 1))).re * Real.cos ((-((j : ℤ) + 1) : ℤ) * t)| ≤ M * q ^ (j + 1) := by
      have hnabs : (-((j : ℤ) + 1)).natAbs = j + 1 := by
        rw [Int.natAbs_neg, show (j : ℤ) + 1 = ((j + 1 : ℕ) : ℤ) by norm_cast, Int.natAbs_natCast]
      rwa [hnabs] at hneg
    have hpow : q ^ (j + 1) ≤ q ^ j := by
      rw [pow_succ]
      exact mul_le_of_le_one_right (pow_nonneg hq j) hq1.le
    have hm := mul_le_mul_of_nonneg_left hpow hM
    have htri := abs_add_le ((a (j : ℤ)).re * Real.cos ((j : ℝ) * t))
      ((a (-((j : ℤ) + 1))).re * Real.cos ((-((j : ℤ) + 1) : ℤ) * t))
    dsimp [u]
    linarith
  have hrest := geometric_series_remainder (u := u) hs (by positivity) hq hq1 hu N
  have hxcos : Real.cos t = x := Real.cos_arccos (neg_le_of_abs_le hx) (le_of_abs_le hx)
  have hval : (F (t : ℂ)).re = (g (x : ℂ)).re := by
    dsimp [F]
    rw [← Complex.ofReal_cos, hxcos]
  rw [hval] at hrest
  have hpoly : (cosinePolynomial a N).eval x = ∑ j ∈ range N, u j := by
    rw [← hxcos, cosinePolynomial_eval_cos]
  have hpow : q ^ N = Real.exp (-δ * N) := by
    dsimp [q]
    rw [← Real.exp_nat_mul]
    congr 1
    ring
  rw [← hpoly, hpow] at hrest
  exact hrest

end QIRadon.ProfilePreparation
