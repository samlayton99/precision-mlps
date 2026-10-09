import QIRadon.Quadrature.OrthogonalPolynomial
import QIRadon.Quadrature.OrthogonalRoots

/-!
# Gaussian quadrature existence from interval positivity

This constructs an orthogonal polynomial from the moment functional, proves its
roots are real, simple and interior, and derives positive normalized weights and
degree-2n-1 exactness. It does not assume any of those quadrature conclusions.
-/

open Polynomial
open scoped BigOperators

namespace QIRadon.Gaussian

/-- An arbitrary interval-positive normalized moment functional admits the
Gaussian rule. The returned weights are the defined integrated Lagrange weights,
not independent existential coefficients. -/
theorem exists_positive_gaussian_rule (L : ℝ[X] →ₗ[ℝ] ℝ)
    (a b : ℝ) (hab : a < b)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → (∀ x ∈ Set.Ioo a b, 0 ≤ u.eval x) → 0 < L u)
    (hunit : L 1 = 1) (n : ℕ) (hn : 0 < n) :
    ∃ node : Fin n → ℝ,
      Function.Injective node ∧
      (∀ i, node i ∈ Set.Ioo a b) ∧
      (∀ i, 0 < weight L node i) ∧
      (∑ i, weight L node i = 1) ∧
      (∀ p : ℝ[X], p.natDegree < 2 * n →
        L p = ∑ i, weight L node i * p.eval (node i)) := by
  classical
  have hsq : ∀ u : ℝ[X], u ≠ 0 → 0 < L (u ^ 2) := by
    intro u hu
    apply hpos (u ^ 2) (pow_ne_zero _ hu)
    intro x _
    simpa only [Polynomial.eval_pow] using sq_nonneg (u.eval x)
  obtain ⟨p, hpmonic, hpdegree, hporth⟩ := exists_monic_orthogonal L n hsq
  have horth : ∀ u : ℝ[X], u.natDegree < p.natDegree → L (p * u) = 0 := by
    intro u hu
    apply hporth u
    apply Polynomial.degree_le_natDegree.trans_lt
    rw [hpdegree] at hu
    exact_mod_cast hu
  have hcard := orthogonal_roots_card L a b hab hpos p hpmonic.ne_zero horth
  have hnodup := orthogonal_roots_nodup L a b hpos p hpmonic.ne_zero horth
  have hinside := orthogonal_roots_mem_interval L a b hab hpos p hpmonic.ne_zero horth
  let s : Finset ℝ := p.roots.toFinset
  have hs : Fintype.card s = n := by
    simp only [Fintype.card_coe, s, Multiset.toFinset_card_of_nodup hnodup,
      hcard, hpdegree]
  let e : Fin n ≃ s := (Fintype.equivFinOfCardEq hs).symm
  let node : Fin n → ℝ := fun i ↦ (e i).val
  have hinj : Function.Injective node :=
    Subtype.val_injective.comp e.injective
  have hroot : ∀ i, p.eval (node i) = 0 := by
    intro i
    have hi : node i ∈ p.roots := by
      exact Multiset.mem_toFinset.mp (e i).property
    exact (Polynomial.mem_roots hpmonic.ne_zero).mp hi
  have hpnode : p = Lagrange.nodal Finset.univ node := by
    apply Polynomial.eq_of_degree_le_of_eval_index_eq Finset.univ hinj.injOn
    · simp [Polynomial.degree_eq_natDegree hpmonic.ne_zero, hpdegree]
    · simp [Polynomial.degree_eq_natDegree hpmonic.ne_zero, hpdegree,
        Lagrange.degree_nodal]
    · exact hpmonic.trans (Lagrange.nodal_monic.leadingCoeff).symm
    · intro i _
      rw [hroot i, Lagrange.eval_nodal_at_node (Finset.mem_univ i)]
  have hncard : 0 < Fintype.card (Fin n) := by simpa using hn
  have hnodeorth : ∀ u : ℝ[X], u.degree < Fintype.card (Fin n) →
      L (Lagrange.nodal Finset.univ node * u) = 0 := by
    intro u hu
    rw [← hpnode]
    exact hporth u (by simpa using hu)
  refine ⟨node, hinj, ?_, ?_, ?_, ?_⟩
  · intro i
    exact hinside (node i) (Multiset.mem_toFinset.mp (e i).property)
  · exact weight_pos L node hinj hncard hnodeorth hsq
  · exact sum_weights L node hinj hncard hunit
  · intro u hu
    exact exact_below_twice_card L node hinj hncard hnodeorth u (by simpa using hu)

end QIRadon.Gaussian
