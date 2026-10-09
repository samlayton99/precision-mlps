import Mathlib

/-! Finite unqualified samples cannot certify uniform error. The counterexample
is an explicit polynomial, so even qualitative analyticity is insufficient:
quantitative regularity or coverage bounds are essential. -/

noncomputable section
open Polynomial Set
namespace QIRadon.FiniteSampleObstruction

def vanishingPolynomial (sites : Finset ℝ) : ℝ[X] := ∏ a ∈ sites, (X-C a)

theorem vanishingPolynomial_eval_zero (sites : Finset ℝ) {x : ℝ} (hx : x ∈ sites) :
    (vanishingPolynomial sites).eval x = 0 := by
  simp only [vanishingPolynomial, eval_prod, eval_sub, eval_X, eval_C]
  exact Finset.prod_eq_zero hx (sub_self x)

theorem vanishingPolynomial_eval_ne_zero (sites : Finset ℝ) {x : ℝ} (hx : x ∉ sites) :
    (vanishingPolynomial sites).eval x ≠ 0 := by
  simp only [vanishingPolynomial, eval_prod, eval_sub, eval_X, eval_C]
  apply Finset.prod_ne_zero_iff.mpr
  intro a ha
  exact sub_ne_zero.mpr (fun h => hx (h ▸ ha))

/-- The same finite sample values are compatible with a polynomial of any
prescribed height at an unsampled point in any nonempty interval. -/
theorem exists_analytic_sample_invisible (sites : Finset ℝ) {a b : ℝ} (hab : a < b)
    (height : ℝ) :
    ∃ (p : ℝ[X]) (x : ℝ), x ∈ Ioo a b ∧
      (∀ s ∈ sites, p.eval s = 0) ∧ p.eval x = height ∧
      AnalyticOnNhd ℝ (fun y : ℝ => p.eval y) univ := by
  have hinf : (Ioo a b).Infinite := Set.Ioo_infinite hab
  obtain ⟨x, hx, hxout⟩ := hinf.exists_notMem_finset sites
  let p := vanishingPolynomial sites
  have hp : p.eval x ≠ 0 := vanishingPolynomial_eval_ne_zero sites hxout
  refine ⟨C (height/p.eval x)*p, x, hx, ?_, ?_, ?_⟩
  · intro s hs
    simp only [eval_mul, eval_C]
    rw [vanishingPolynomial_eval_zero sites hs, mul_zero]
  · simp only [eval_mul, eval_C]
    exact div_mul_cancel₀ height hp
  · exact AnalyticOnNhd.eval_polynomial _

/-- At the same samples, zero and an analytic target may be separated by any
positive amount inside the target interval. No sample-only uniform bound follows. -/
theorem no_unqualified_uniform_sample_certificate (sites : Finset ℝ)
    {a b H : ℝ} (hab : a < b) (hH : 0 < H) :
    ∃ f : ℝ → ℝ, AnalyticOnNhd ℝ f univ ∧ (∀ s ∈ sites, f s=0) ∧
      ∃ x ∈ Ioo a b, H < |f x| := by
  obtain ⟨p,x,hx,hzero,hvalue,han⟩ := exists_analytic_sample_invisible sites hab (2*H)
  refine ⟨fun y => p.eval y,han,hzero,x,hx,?_⟩
  change H < |p.eval x|
  rw [hvalue,abs_of_pos (by positivity)]
  linarith

end QIRadon.FiniteSampleObstruction
