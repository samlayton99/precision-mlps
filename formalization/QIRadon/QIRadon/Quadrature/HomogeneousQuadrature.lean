import QIRadon.Quadrature.HomogeneousBound
import QIRadon.Quadrature.SeriesTail

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section

theorem mvPolynomial_eval_support (n : ℕ) (p : MvPolynomial (Fin (n + 1)) ℝ)
    (x : EuclideanSpace ℝ (Fin (n + 1))) :
    MvPolynomial.eval (fun i ↦ x i) p =
      ∑ s : p.support, MvPolynomial.coeff s.val p * coordinateMonomial (fun i ↦ s.val i) x := by
  classical
  rw [MvPolynomial.eval_eq,
    Finset.sum_coe_sort p.support
      (fun s ↦ MvPolynomial.coeff s p * coordinateMonomial (fun i ↦ s i) x)]
  apply Finset.sum_congr rfl
  intro s _
  congr 1
  exact Finsupp.prod_fintype s (fun i e ↦ x i ^ e) (by intro i; simp)

theorem sphereRule_homogeneous_polynomial_exact (q : ℕ) (hq : 0 < q) (n k : ℕ)
    (p : HomogeneousPolynomials n k) (hk : k < 2 * q) :
    (∑ i, sphereWeight q hq n i * homogeneousRestriction n k p (sphereNodeOnSphere q hq n i)) =
      ∫ θ, homogeneousRestriction n k p θ ∂normalizedSphere n := by
  classical
  have hdegree : ∀ s : p.val.support, (∑ i, s.val i) < 2 * q := by
    intro s
    have hdeg := p.property.degree_eq_sum_deg_support s.property
    have heq : s.val.degree = k := by simpa only [Finsupp.degree_apply] using hdeg.symm
    rw [← Finsupp.degree_eq_sum, heq]
    exact hk
  have h := sphereRule_polynomial_exact q hq n
    (fun s : p.val.support ↦ MvPolynomial.coeff s.val p.val)
    (fun s : p.val.support ↦ fun j ↦ s.val j) hdegree
  rw [normalizedSphere_integral]
  change (∑ i, sphereWeight q hq n i * MvPolynomial.eval (fun j ↦ sphereNode q hq n i j) p.val) =
    sphereAverage (fun θ ↦ MvPolynomial.eval (fun j ↦ θ.val j) p.val)
  simp_rw [mvPolynomial_eval_support]
  exact h

theorem sphereRule_polynomialL2_exact (q : ℕ) (hq : 0 < q) (n k : ℕ)
    (v : PolynomialL2Space n k) (hk : k < 2 * q) :
    (∑ i, sphereWeight q hq n i * polynomialEvaluation n k (sphereNodeOnSphere q hq n i) v) =
      ∫ θ, polynomialEvaluation n k θ v ∂normalizedSphere n := by
  obtain ⟨p, hp⟩ := ((polynomialL2Equiv n k).symm v).property
  have heq : (fun θ ↦ polynomialEvaluation n k θ v) = homogeneousRestriction n k p := by
    funext θ
    exact (congrArg (fun f : C(UnitSphere n, ℝ) ↦ f θ) hp).symm
  change (∑ i, sphereWeight q hq n i * (fun θ ↦ polynomialEvaluation n k θ v)
    (sphereNodeOnSphere q hq n i)) = _
  rw [heq]
  exact sphereRule_homogeneous_polynomial_exact q hq n k p hk

theorem polynomialEvaluation_bound_power (n k : ℕ) (θ : UnitSphere n)
    (v : PolynomialL2Space n k) :
    |polynomialEvaluation n k θ v| ≤ ((k : ℝ) + 1) ^ (n + 1) * ‖v‖ := by
  apply (polynomialEvaluation_bound n k θ v).trans
  have hx : (1 : ℝ) ≤ (((k + 1) ^ (n + 1) : ℕ) : ℝ) := by
    exact_mod_cast (Nat.one_le_pow _ _ (by omega : 1 ≤ k + 1))
  have hs := Real.sqrt_le_self_iff.mpr (Or.inr hx)
  have h := mul_le_mul_of_nonneg_right hs (norm_nonneg v)
  simpa only [Nat.cast_pow, Nat.cast_add, Nat.cast_one] using h

/-- Analytic angular convergence for an actual homogeneous-polynomial L2
expansion with exponential weighted energy. The separate target-side theorem
must still establish this expansion and energy from local holomorphic f. -/
theorem foldedSphere_analytic_series_error (q : ℕ) (hq : 0 < q) (n : ℕ)
    (v : (k : ℕ) → PolynomialL2Space n k) (F : UnitSphere n → ℝ)
    (hF : Integrable F (normalizedSphere n))
    (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    (hrep : ∀ θ, F θ = ∑' k, polynomialEvaluation n k θ (v k))
    (a A : ℝ) (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ ↦ Real.exp (2 * a * k) * ‖v k‖ ^ 2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * ‖v k‖ ^ 2) ≤ A ^ 2) :
    |(∫ θ, F θ ∂normalizedSphere n) -
      ∑ i : SphereHalfIndex q n, foldedSphereWeight q hq n i *
        F (sphereNodeOnSphere q hq n i.val)| ≤
    2 * ((A * (((n + 1).factorial : ℝ) / (a / 2) ^ (n + 1) * Real.exp (a / 2))) *
      Real.exp (-a * q) / (1 - Real.exp (-(a / 2)))) := by
  let P : UnitSphere n → ℝ := fun θ ↦
    ∑ k ∈ Finset.range (2 * q), polynomialEvaluation n k θ (v k)
  let err : ℝ := (A * (((n + 1).factorial : ℝ) / (a / 2) ^ (n + 1) * Real.exp (a / 2))) *
    Real.exp (-a * q) / (1 - Real.exp (-(a / 2)))
  have happrox : ∀ θ, |F θ - P θ| ≤ err := by
    intro θ
    have h := QIRadon.SeriesTail.analytic_component_series_tail
      (fun k θ ↦ polynomialEvaluation n k θ (v k)) (n + 1) ha hA
      (fun k ↦ norm_nonneg (v k)) hsum henergy
      (fun k θ ↦ polynomialEvaluation_bound_power n k θ (v k)) θ (2 * q)
    have hexp : -(a / 2) * ((2 * q : ℕ) : ℝ) = -a * q := by push_cast; ring
    rw [hexp] at h
    simpa only [P, err, hrep θ] using h.2
  have hPc : Continuous P := by
    unfold P polynomialEvaluation
    change Continuous (fun θ ↦ ∑ k ∈ Finset.range (2 * q), polynomialRepresentative n k (v k) θ)
    fun_prop
  have hPi : Integrable P (normalizedSphere n) :=
    hPc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hexact : (∫ θ, P θ ∂normalizedSphere n) =
      ∑ i, sphereWeight q hq n i * P (sphereNodeOnSphere q hq n i) := by
    unfold P
    have hi : ∀ k, Integrable (fun θ ↦ polynomialEvaluation n k θ (v k)) (normalizedSphere n) := by
      intro k
      have hc : Continuous (fun θ ↦ polynomialRepresentative n k (v k) θ) := by fun_prop
      exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
    rw [integral_finsetSum _ (fun k _ ↦ hi k)]
    calc
      _ = ∑ k ∈ Finset.range (2 * q), ∑ i, sphereWeight q hq n i *
          polynomialEvaluation n k (sphereNodeOnSphere q hq n i) (v k) := by
        apply Finset.sum_congr rfl
        intro k hk
        exact (sphereRule_polynomialL2_exact q hq n k (v k) (Finset.mem_range.mp hk)).symm
      _ = _ := by rw [Finset.sum_comm]; simp_rw [Finset.mul_sum]
  have h := QIRadon.positive_quadrature_error (normalizedSphere n)
    (sphereNodeOnSphere q hq n) (sphereWeight q hq n) F P err
    (fun i ↦ (sphereWeight_positive q hq n i).le)
    (sphereWeight_normalized q hq n) hF hPi hexact happrox
  rwa [foldedSphere_sum_even_onSphere q hq n F heven] at h

end
end QIRadon.Gaussian
