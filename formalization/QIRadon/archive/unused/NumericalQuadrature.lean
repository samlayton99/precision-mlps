import QIRadon.PreparationStability
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

/-! Actual finite-sample integration of a certified analytic integrand.
The polynomial integral below is a finite coefficient formula, not another
unevaluated integral. Its applicability to each profile/density acquisition
integrand must still be certified separately. -/

noncomputable section
open MeasureTheory Polynomial Set
open scoped Real
namespace QIRadon.NumericalQuadrature
open QIRadon.ProfilePreparation QIRadon.LocalProfileRate

def polynomialIntegral (p : ℝ[X]) (a b : ℝ) : ℝ :=
  ∑ k ∈ Finset.range (p.natDegree+1), p.coeff k *
    ((b^(k+1)-a^(k+1))/((k:ℝ)+1))

theorem polynomialIntegral_eq (p : ℝ[X]) (a b : ℝ) :
    polynomialIntegral p a b = ∫ x in a..b, p.eval x := by
  simp_rw [Polynomial.eval_eq_sum_range]
  rw [intervalIntegral.integral_finsetSum (fun i _ =>
    (show Continuous (fun x : ℝ => p.coeff i*x^i) by fun_prop).intervalIntegrable a b)]
  unfold polynomialIntegral
  apply Finset.sum_congr rfl
  intro i _
  rw [intervalIntegral.integral_const_mul, integral_pow]

def sampledIntegral (N : ℕ) (f : ℝ → ℝ) : ℝ :=
  polynomialIntegral (samplePolynomial (N+1) f) (-1) 1

/-- The actual finite integration formula has a geometric error bound for an
integrand with a certified local ellipse, using exactly N+1 real samples. -/
theorem sampledIntegral_error {g : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hb : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M) (N : ℕ) :
    |(∫ x in (-1:ℝ)..1, (g (x:ℂ)).re)-sampledIntegral N (fun x => (g (x:ℂ)).re)| ≤
      2*interpolationConstant M δ*Real.exp (-(δ/2)*N) := by
  let f : ℝ → ℝ := fun x => (g (x:ℂ)).re
  let p := samplePolynomial (N+1) f
  have hreal (x : ℝ) (hx : x ∈ Icc (-1:ℝ) 1) : (x:ℂ) ∈ bernsteinEllipse δ := by
    refine ⟨(Real.arccos x:ℂ),?_,?_⟩
    · simpa [closedStrip] using hδ.le
    · rw [← Complex.ofReal_cos, Real.cos_arccos hx.1 hx.2]
  have hc : ContinuousOn f (Icc (-1:ℝ) 1) := by
    intro x hx
    exact (Complex.continuous_re.continuousAt.comp
      ((hg (x:ℂ) (hreal x hx)).continuousAt.comp Complex.continuous_ofReal.continuousAt)).continuousWithinAt
  have hi := hc.intervalIntegrable_of_Icc (μ := volume) (by norm_num : (-1:ℝ) ≤ 1)
  have hp := p.continuous.intervalIntegrable (μ := volume) (-1:ℝ) 1
  unfold sampledIntegral
  rw [polynomialIntegral_eq,← intervalIntegral.integral_sub hi hp]
  have hh := intervalIntegral.norm_integral_le_of_norm_le_const
    (a := (-1:ℝ)) (b := 1)
    (C := interpolationConstant M δ*Real.exp (-(δ/2)*N)) (f := fun x => f x-p.eval x) (by
      intro x hx
      have hx' : |x| ≤ 1 := abs_le.mpr ⟨(by simpa using hx.1.le),by simpa using hx.2⟩
      simpa only [Real.norm_eq_abs,f,p,interpolationConstant] using
        samplePolynomial_geometric_error hδ hg hb N hx')
  have hh' : |∫ x in (-1:ℝ)..1, f x-p.eval x| ≤
      (interpolationConstant M δ*Real.exp (-(δ/2)*N))*2 := by norm_num at hh ⊢; exact hh
  exact hh'.trans_eq (by ring)

/-- Errors in the finite supplied samples add a completely explicit term. -/
theorem sampledIntegral_perturbation {f g : ℝ → ℝ} {η : ℝ} (hη : 0 ≤ η)
    (N : ℕ)
    (hdata : ∀ j < N+1, |f (sampleNode (N+1) j)-g (sampleNode (N+1) j)| ≤ η) :
    |sampledIntegral N f-sampledIntegral N g| ≤ 4*(N+1)*η := by
  unfold sampledIntegral
  rw [polynomialIntegral_eq,polynomialIntegral_eq,← intervalIntegral.integral_sub
    ((samplePolynomial (N+1) f).continuous.intervalIntegrable (-1:ℝ) 1)
    ((samplePolynomial (N+1) g).continuous.intervalIntegrable (-1:ℝ) 1)]
  have hh := intervalIntegral.norm_integral_le_of_norm_le_const
    (a := (-1:ℝ)) (b := 1) (C := 2*(N+1)*η)
    (f := fun x => (samplePolynomial (N+1) f).eval x-(samplePolynomial (N+1) g).eval x) (by
      intro x hx
      have hx' : |x| ≤ 1 := abs_le.mpr ⟨(by simpa using hx.1.le),by simpa using hx.2⟩
      simpa only [Real.norm_eq_abs, Nat.cast_succ] using
        PreparationStability.samplePolynomial_difference (Nat.succ_pos N) hη hdata hx')
  have hh' : |∫ x in (-1:ℝ)..1,
      (samplePolynomial (N+1) f).eval x-(samplePolynomial (N+1) g).eval x| ≤
      (2*(N+1)*η)*2 := by norm_num at hh ⊢; exact hh
  exact hh'.trans_eq (by ring)

end QIRadon.NumericalQuadrature
