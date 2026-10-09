import QIRadon.Quadrature.Construction
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Order.Interval.Set.Infinite

/-!
# Symmetric Gauss–Jacobi quadrature used by the sphere rule

The weight is `(1-x^2)^a`, with `a ≥ 0`. This is the required range at recursive
sphere levels d ≥ 3, since a = (d-3)/2; d = 2 uses circle quadrature directly.
We derive interval positivity for the actual weighted integral and instantiate
the Gaussian construction, so nodes/weights and exactness are not assumptions.
-/

open Polynomial MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable def jacobiDensity (a x : ℝ) : ℝ := (1 - x ^ 2) ^ a

theorem jacobiDensity_continuous (a : ℝ) (ha : 0 ≤ a) :
    Continuous (jacobiDensity a) := by
  exact (continuous_const.sub (continuous_id.pow 2)).rpow_const (fun _ ↦ Or.inr ha)

theorem jacobiDensity_pos (a x : ℝ) (hx : x ∈ Set.Ioo (-1 : ℝ) 1) :
    0 < jacobiDensity a x := by
  apply Real.rpow_pos_of_pos
  nlinarith [hx.1, hx.2, sq_nonneg x]

theorem jacobi_integrand_continuous (a : ℝ) (ha : 0 ≤ a) (p : ℝ[X]) :
    Continuous (fun x ↦ jacobiDensity a x * p.eval x) :=
  (jacobiDensity_continuous a ha).mul p.continuous

/-- Unnormalized Jacobi moment functional, using the actual interval integral. -/
noncomputable def rawJacobiMoment (a : ℝ) (ha : 0 ≤ a) : ℝ[X] →ₗ[ℝ] ℝ where
  toFun p := ∫ x in (-1 : ℝ)..1, jacobiDensity a x * p.eval x
  map_add' p q := by
    simp only [Polynomial.eval_add, mul_add]
    exact intervalIntegral.integral_add
      ((jacobi_integrand_continuous a ha p).intervalIntegrable _ _)
      ((jacobi_integrand_continuous a ha q).intervalIntegrable _ _)
  map_smul' c p := by
    simp only [Polynomial.eval_smul, smul_eq_mul, RingHom.id_apply]
    simp_rw [mul_left_comm (jacobiDensity a _) c]
    exact intervalIntegral.integral_const_mul _ _

/-- A nonzero polynomial nonnegative on the open Jacobi interval has strictly
positive weighted integral. This is the positivity needed to construct roots. -/
theorem rawJacobiMoment_pos (a : ℝ) (ha : 0 ≤ a) (p : ℝ[X]) (hp : p ≠ 0)
    (hnonneg : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 ≤ p.eval x) :
    0 < rawJacobiMoment a ha p := by
  let f : ℝ → ℝ := fun x ↦ jacobiDensity a x * p.eval x
  have hf : Continuous f := jacobi_integrand_continuous a ha p
  have hposInterior : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 ≤ f x := by
    intro x hx
    exact mul_nonneg (jacobiDensity_pos a x hx).le (hnonneg x hx)
  have hclosed : IsClosed {x | 0 ≤ f x} := isClosed_le continuous_const hf
  have hclosure : Set.Icc (-1 : ℝ) 1 ⊆ {x | 0 ≤ f x} := by
    have h := closure_minimal hposInterior hclosed
    rwa [closure_Ioo (by norm_num : (-1 : ℝ) ≠ 1)] at h
  have hinf : (Set.Ioo (-1 : ℝ) 1).Infinite := Set.Ioo_infinite (by norm_num)
  obtain ⟨c, hc, hcroot⟩ := hinf.exists_notMem_finite
    (Polynomial.finite_setOfPred_isRoot hp)
  have hpne : p.eval c ≠ 0 := hcroot
  have hfc : 0 < f c := mul_pos (jacobiDensity_pos a c hc)
    (lt_of_le_of_ne (hnonneg c hc) hpne.symm)
  have h := intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt
    (by norm_num : (-1 : ℝ) < 1)
    (f := fun _ : ℝ ↦ 0) (g := f) continuousOn_const hf.continuousOn
    (fun x hx ↦ hclosure ⟨hx.1.le, hx.2⟩)
    ⟨c, ⟨hc.1.le, hc.2.le⟩, hfc⟩
  simpa [rawJacobiMoment, f] using h

theorem rawJacobiMoment_one_pos (a : ℝ) (ha : 0 ≤ a) :
    0 < rawJacobiMoment a ha 1 :=
  rawJacobiMoment_pos a ha 1 one_ne_zero (by intro x _; simp)

/-- Normalized Jacobi moment functional. Its mass is proved strictly positive. -/
noncomputable def jacobiMoment (a : ℝ) (ha : 0 ≤ a) : ℝ[X] →ₗ[ℝ] ℝ :=
  (rawJacobiMoment a ha 1)⁻¹ • rawJacobiMoment a ha

theorem jacobiMoment_one (a : ℝ) (ha : 0 ≤ a) : jacobiMoment a ha 1 = 1 := by
  simp [jacobiMoment, (rawJacobiMoment_one_pos a ha).ne']

theorem jacobiMoment_pos (a : ℝ) (ha : 0 ≤ a) (p : ℝ[X]) (hp : p ≠ 0)
    (hnonneg : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 ≤ p.eval x) :
    0 < jacobiMoment a ha p := by
  exact mul_pos (inv_pos.mpr (rawJacobiMoment_one_pos a ha))
    (rawJacobiMoment_pos a ha p hp hnonneg)

/-- The actual symmetric Gauss–Jacobi rule: distinct interior nodes, strictly
positive normalized integrated-Lagrange weights, and exactness through 2n-1. -/
theorem exists_gaussJacobi_rule (a : ℝ) (ha : 0 ≤ a) (n : ℕ) (hn : 0 < n) :
    ∃ node : Fin n → ℝ,
      Function.Injective node ∧
      (∀ i, node i ∈ Set.Ioo (-1 : ℝ) 1) ∧
      (∀ i, 0 < weight (jacobiMoment a ha) node i) ∧
      (∑ i, weight (jacobiMoment a ha) node i = 1) ∧
      (∀ p : ℝ[X], p.natDegree < 2 * n →
        jacobiMoment a ha p =
        ∑ i, weight (jacobiMoment a ha) node i * p.eval (node i)) :=
  exists_positive_gaussian_rule (jacobiMoment a ha) (-1) 1 (by norm_num)
    (jacobiMoment_pos a ha) (jacobiMoment_one a ha) n hn

end QIRadon.Gaussian
