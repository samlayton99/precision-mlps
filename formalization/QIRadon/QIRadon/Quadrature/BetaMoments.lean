import Mathlib.Probability.Distributions.Beta
import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts

/-! Actual Jacobi weighted moments, including integrable endpoint singularities. -/

noncomputable section
open MeasureTheory Set
open scoped Interval

namespace QIRadon.Quadrature

private theorem real_beta_integrand_eq (α β x : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) :
    ((x : ℂ)^((α : ℂ) - 1) * (1 - (x : ℂ))^((β : ℂ) - 1)).re =
      x^(α - 1) * (1 - x)^(β - 1) := by
  have heq : ((x^(α - 1) * (1 - x)^(β - 1) : ℝ) : ℂ) =
      (x : ℂ)^((α : ℂ) - 1) * (1 - (x : ℂ))^((β : ℂ) - 1) := by
    rw [Complex.ofReal_mul, Complex.ofReal_cpow hx,
      Complex.ofReal_cpow (by linarith : 0 ≤ 1 - x)]
    push_cast
    rfl
  rw [← heq]
  simp

/-- Real beta integrands are integrable even when either endpoint is singular. -/
theorem real_beta_intervalIntegrable {α β : ℝ} (hα : 0 < α) (hβ : 0 < β) :
    IntervalIntegrable (fun x : ℝ => x^(α - 1) * (1 - x)^(β - 1)) volume 0 1 := by
  have hc := Complex.betaIntegral_convergent (u := (α : ℂ)) (v := (β : ℂ))
    (by simpa) (by simpa)
  have hr : IntervalIntegrable (fun x : ℝ =>
      ((x : ℂ)^((α : ℂ) - 1) * (1 - (x : ℂ))^((β : ℂ) - 1)).re) volume 0 1 :=
    ⟨Complex.reCLM.integrable_comp hc.1, Complex.reCLM.integrable_comp hc.2⟩
  apply IntervalIntegrable.congr _ hr
  intro x hx
  have hx' : x ∈ Ioc (0 : ℝ) 1 := by simpa using hx
  exact real_beta_integrand_eq α β x hx'.1.le hx'.2

/-- Real beta integral, expressed by real Gamma functions. -/
theorem real_beta_integral {α β : ℝ} (hα : 0 < α) (hβ : 0 < β) :
    (∫ x in (0 : ℝ)..1, x^(α - 1) * (1 - x)^(β - 1)) =
      Real.Gamma α * Real.Gamma β / Real.Gamma (α + β) := by
  have hc := Complex.betaIntegral_convergent (u := (α : ℂ)) (v := (β : ℂ))
    (by simpa) (by simpa)
  change _ = ProbabilityTheory.beta α β
  rw [ProbabilityTheory.beta_eq_betaIntegralReal α β hα hβ]
  unfold Complex.betaIntegral
  calc
    _ = ∫ x in (0 : ℝ)..1,
        ((x : ℂ)^((α : ℂ) - 1) * (1 - (x : ℂ))^((β : ℂ) - 1)).re := by
      apply intervalIntegral.integral_congr_uIoo
      intro x hx
      have hx' : x ∈ Ioo (0 : ℝ) 1 := by simpa using hx
      exact (real_beta_integrand_eq α β x hx'.1.le hx'.2.le).symm
    _ = _ := Complex.reCLM.intervalIntegral_comp_comm hc

private theorem square_beta_integrand (k : ℕ) (a x : ℝ) (hx : 0 < x) :
    ((x^2)^((k : ℝ) - 1/2) * (1 - x^2)^a) * (2*x) =
      2 * (x^(2*k) * (1 - x^2)^a) := by
  have hpow : (x^2)^((k : ℝ) - 1/2) * x = x^(2*k) := by
    calc
      (x^2)^((k : ℝ) - 1/2) * x =
          x^((2 : ℝ)*((k : ℝ)-1/2)) * x^(1 : ℝ) := by
        rw [← Real.rpow_natCast_mul hx.le 2, Real.rpow_one]
        norm_num
      _ = x^((2 : ℝ)*((k : ℝ)-1/2)+1) := (Real.rpow_add hx _ _).symm
      _ = x^(2*k) := by
        have he : (2 : ℝ) * ((k : ℝ) - 1/2) + 1 = ((2*k : ℕ) : ℝ) := by
          push_cast
          ring
        rw [he, Real.rpow_natCast]
  calc
    _ = 2 * ((x^2)^((k : ℝ) - 1/2) * x) * (1 - x^2)^a := by ring
    _ = _ := by rw [hpow]; ring

/-- The positive-half Jacobi moment is integrable for every a>-1, including
negative a and therefore singular endpoint weights. -/
theorem jacobi_even_half_intervalIntegrable (k : ℕ) {a : ℝ} (ha : -1 < a) :
    IntervalIntegrable (fun x : ℝ => x^(2*k) * (1 - x^2)^a) volume 0 1 := by
  let b := fun u : ℝ => u^((k : ℝ) - 1/2) * (1-u)^a
  have hb : IntervalIntegrable b volume 0 1 := by
    have ht := real_beta_intervalIntegrable
      (α := (k : ℝ) + 1/2) (β := a+1) (by positivity) (by linarith)
    convert ht using 1
    funext u
    dsimp [b]
    congr 2 <;> ring
  have hsub : IntervalIntegrable (fun x : ℝ => b (x^2) * (2*x)) volume 0 1 := by
    have hs := intervalIntegral.integrable_comp_mul_deriv_iff_of_deriv_nonneg
      (a := (0 : ℝ)) (b := (1 : ℝ)) (f := fun x : ℝ => x^2)
      (f' := fun x : ℝ => 2*x) (g := b)
      (by fun_prop)
      (by
        intro x hx
        simpa using (hasDerivAt_id x).fun_pow 2)
      (by
        intro x hx
        have hx0 : 0 < x := by simpa using hx.1
        linarith)
    simpa only [Function.comp_apply] using hs.mpr (by simpa using hb)
  apply IntervalIntegrable.congr _ (hsub.const_mul (1/2))
  intro x hx
  have hx' : x ∈ Ioc (0 : ℝ) 1 := by simpa using hx
  dsimp [b]
  rw [square_beta_integrand k a x hx'.1]
  ring

/-- Integrability of every even weighted monomial on the full Jacobi interval. -/
theorem jacobi_even_intervalIntegrable (k : ℕ) {a : ℝ} (ha : -1 < a) :
    IntervalIntegrable (fun x : ℝ => x^(2*k) * (1 - x^2)^a) volume (-1) 1 := by
  let f := fun x : ℝ => x^(2*k) * (1 - x^2)^a
  have hp : IntervalIntegrable f volume 0 1 := jacobi_even_half_intervalIntegrable k ha
  have hn : IntervalIntegrable f volume (-1) 0 := by
    have ht := (IntervalIntegrable.iff_comp_neg).mp hp
    have heven : (fun x => f (-x)) = f := by
      funext x
      simp [f, pow_mul]
    rw [heven] at ht
    simpa using ht.symm
  exact hn.trans hp

/-- The actual Jacobi weight is integrable for the full admissible exponent range. -/
theorem jacobi_weight_intervalIntegrable {a : ℝ} (ha : -1 < a) :
    IntervalIntegrable (fun x : ℝ => (1 - x^2)^a) volume (-1) 1 := by
  simpa using jacobi_even_intervalIntegrable 0 ha

/-- Evaluation of the even weighted moment on the positive half interval. -/
theorem jacobi_even_half_moment (k : ℕ) {a : ℝ} (ha : -1 < a) :
    (∫ x in (0 : ℝ)..1, x^(2*k) * (1 - x^2)^a) =
      (Real.Gamma ((k : ℝ)+1/2) * Real.Gamma (a+1) /
        Real.Gamma ((k : ℝ)+a+3/2)) / 2 := by
  let b := fun u : ℝ => u^((k : ℝ) - 1/2) * (1-u)^a
  have hb : (∫ u in (0 : ℝ)..1, b u) =
      Real.Gamma ((k : ℝ)+1/2) * Real.Gamma (a+1) /
        Real.Gamma ((k : ℝ)+a+3/2) := by
    have ht := real_beta_integral
      (α := (k : ℝ)+1/2) (β := a+1) (by positivity) (by linarith)
    convert ht using 1
    · apply intervalIntegral.integral_congr
      intro u hu
      dsimp [b]
      congr 2 <;> ring
    · congr 2
      ring
  have hs := intervalIntegral.integral_comp_mul_deriv_of_deriv_nonneg
    (a := (0 : ℝ)) (b := (1 : ℝ)) (f := fun x : ℝ => x^2)
    (f' := fun x : ℝ => 2*x) (g := b)
    (by fun_prop)
    (by intro x hx; simpa using (hasDerivAt_id x).fun_pow 2)
    (by
      intro x hx
      have hx0 : 0 < x := by simpa using hx.1
      linarith)
  have hsub : (∫ x in (0 : ℝ)..1, b (x^2) * (2*x)) = ∫ u in (0 : ℝ)..1, b u := by
    simpa only [Function.comp_apply, zero_pow (by omega : (2 : ℕ) ≠ 0), one_pow] using hs
  have hleft : (∫ x in (0 : ℝ)..1, b (x^2) * (2*x)) =
      2 * ∫ x in (0 : ℝ)..1, x^(2*k) * (1 - x^2)^a := by
    rw [← intervalIntegral.integral_const_mul]
    apply intervalIntegral.integral_congr_uIoo
    intro x hx
    have hx0 : 0 < x := by simpa using hx.1
    exact square_beta_integrand k a x hx0
  rw [hleft, hb] at hsub
  linarith

/-- Actual Gauss--Jacobi even moment over [-1,1], for every a>-1. -/
theorem jacobi_even_moment (k : ℕ) {a : ℝ} (ha : -1 < a) :
    (∫ x in (-1 : ℝ)..1, x^(2*k) * (1 - x^2)^a) =
      Real.Gamma ((k : ℝ)+1/2) * Real.Gamma (a+1) /
        Real.Gamma ((k : ℝ)+a+3/2) := by
  let f := fun x : ℝ => x^(2*k) * (1 - x^2)^a
  have hp : IntervalIntegrable f volume 0 1 := jacobi_even_half_intervalIntegrable k ha
  have heven : (fun x => f (-x)) = f := by
    funext x
    simp [f, pow_mul]
  have hn : IntervalIntegrable f volume (-1) 0 := by
    have ht := (IntervalIntegrable.iff_comp_neg).mp hp
    rw [heven] at ht
    simpa using ht.symm
  have hleft : (∫ x in (-1 : ℝ)..0, f x) = ∫ x in (0 : ℝ)..1, f x := by
    have ht := intervalIntegral.integral_comp_neg f (a := (0 : ℝ)) (b := (1 : ℝ))
    rw [heven] at ht
    simpa using ht.symm
  have hsplit := intervalIntegral.integral_add_adjacent_intervals hn hp
  rw [hleft] at hsplit
  have hhalf := jacobi_even_half_moment k ha
  change (∫ x in (-1 : ℝ)..1, f x) = _
  dsimp only [f] at hsplit
  linarith

/-- Every odd weighted Jacobi moment vanishes by reflection symmetry. -/
theorem jacobi_odd_moment (k : ℕ) (a : ℝ) :
    (∫ x in (-1 : ℝ)..1, x^(2*k+1) * (1 - x^2)^a) = 0 := by
  let f := fun x : ℝ => x^(2*k+1) * (1 - x^2)^a
  have hodd : (fun x => f (-x)) = fun x => -f x := by
    funext x
    simp [f, pow_add, pow_mul]
  have ht := intervalIntegral.integral_comp_neg f (a := (-1 : ℝ)) (b := (1 : ℝ))
  rw [hodd, intervalIntegral.integral_neg] at ht
  simp only [neg_neg] at ht
  change (∫ x in (-1 : ℝ)..1, f x) = 0
  linarith

end QIRadon.Quadrature
