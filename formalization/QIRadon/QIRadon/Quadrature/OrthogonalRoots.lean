import QIRadon.Quadrature.Gaussian
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.Algebra.Polynomial

/-!
# Roots of interval-orthogonal polynomials

The intended role of this file is to discharge the root part of Gaussian
quadrature from positivity of an interval integral. It does not assume that
the polynomial already has the required real, simple, interior roots.
-/

open Polynomial

namespace QIRadon.Gaussian

/-- A continuous nonvanishing real function on an interval has constant sign. -/
theorem sign_constant_on_interval (a b : ℝ) (hab : a < b) (f : ℝ → ℝ)
    (hf : ContinuousOn f (Set.Ioo a b)) (hzero : ∀ x ∈ Set.Ioo a b, f x ≠ 0) :
    (∀ x ∈ Set.Ioo a b, 0 < f x) ∨ (∀ x ∈ Set.Ioo a b, f x < 0) := by
  obtain ⟨c, hc⟩ := Set.nonempty_Ioo.mpr hab
  rcases lt_or_gt_of_ne (hzero c hc) with hcneg | hcpos
  · right
    intro x hx
    by_contra h
    have hxpos : 0 ≤ f x := le_of_not_gt h
    obtain ⟨y, hy, hyzero⟩ :=
      isPreconnected_Ioo.intermediate_value hc hx hf ⟨hcneg.le, hxpos⟩
    exact hzero y hy hyzero
  · left
    intro x hx
    by_contra h
    have hxneg : f x ≤ 0 := le_of_not_gt h
    obtain ⟨y, hy, hyzero⟩ :=
      isPreconnected_Ioo.intermediate_value hx hc hf ⟨hxneg, hcpos.le⟩
    exact hzero y hy hyzero

/-- Factoring out every root in an interval leaves a factor with no roots there.
The roots are counted with multiplicity in the product. -/
theorem factor_interval_roots (p : ℝ[X]) (hp : p ≠ 0) (a b : ℝ) :
    ∃ q r : ℝ[X],
      p = q * r ∧ q.Monic ∧ r ≠ 0 ∧
      q.natDegree = (p.roots.filter (fun x ↦ x ∈ Set.Ioo a b)).card ∧
      (∀ x ∈ Set.Ioo a b, r.eval x ≠ 0) := by
  classical
  let s := p.roots.filter (fun x ↦ x ∈ Set.Ioo a b)
  let q : ℝ[X] := (s.map (fun x ↦ X - C x)).prod
  have hmonic : q.Monic := Polynomial.monic_multisetProd_X_sub_C s
  have hdiv : q ∣ p :=
    (Multiset.prod_X_sub_C_dvd_iff_le_roots hp s).mpr (Multiset.filter_le _ _)
  obtain ⟨r, hr⟩ := hdiv
  have hrne : r ≠ 0 := by
    intro h
    rw [h, mul_zero] at hr
    exact hp hr
  refine ⟨q, r, hr, hmonic, hrne, ?_, ?_⟩
  · exact Polynomial.natDegree_multiset_prod_X_sub_C_eq_card s
  · have hroots : p.roots = s + r.roots := by
      rw [hr, Polynomial.roots_mul (mul_ne_zero hmonic.ne_zero hrne)]
      rw [show q.roots = s from Polynomial.roots_multiset_prod_X_sub_C s]
    have hfilter : r.roots.filter (fun x ↦ x ∈ Set.Ioo a b) = 0 := by
      have h := congrArg (Multiset.filter (fun x ↦ x ∈ Set.Ioo a b)) hroots
      change s = _ at h
      rw [Multiset.filter_add] at h
      have hs : s.filter (fun x ↦ x ∈ Set.Ioo a b) = s := by
        simp [s]
      rw [hs] at h
      exact (add_eq_left.mp h.symm)
    intro x hx hz
    have hxroots : x ∈ r.roots := (Polynomial.mem_roots hrne).mpr hz
    have hmem : x ∈ r.roots.filter (fun x ↦ x ∈ Set.Ioo a b) :=
      Multiset.mem_filter.mpr ⟨hxroots, hx⟩
    rw [hfilter] at hmem
    exact Multiset.notMem_zero x hmem

/-- A polynomial orthogonal to all lower degrees has all of its roots in the
open integration interval, counted with multiplicity. The proof derives this
from interval positivity; no real-rootedness assumption is used. -/
theorem orthogonal_interval_root_count (L : ℝ[X] →ₗ[ℝ] ℝ)
    (a b : ℝ) (hab : a < b)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → (∀ x ∈ Set.Ioo a b, 0 ≤ u.eval x) → 0 < L u)
    (p : ℝ[X]) (hp : p ≠ 0)
    (horth : ∀ u : ℝ[X], u.natDegree < p.natDegree → L (p * u) = 0) :
    (p.roots.filter (fun x ↦ x ∈ Set.Ioo a b)).card = p.natDegree := by
  classical
  have hle : (p.roots.filter (fun x ↦ x ∈ Set.Ioo a b)).card ≤ p.natDegree :=
    (Multiset.card_le_card (Multiset.filter_le _ _)).trans (Polynomial.card_roots' p)
  by_contra hne
  have hlt := lt_of_le_of_ne hle hne
  obtain ⟨q, r, hpqr, hqmonic, hrne, hqdegree, hrzero⟩ := factor_interval_roots p hp a b
  have horthq : L (p * q) = 0 := horth q (by simpa [hqdegree] using hlt)
  have hpqne : p * q ≠ 0 := mul_ne_zero hp hqmonic.ne_zero
  rcases sign_constant_on_interval a b hab (fun x ↦ r.eval x) r.continuous.continuousOn hrzero with
    hrpos | hrneg
  · have hnonneg : ∀ x ∈ Set.Ioo a b, 0 ≤ (p * q).eval x := by
      intro x hx
      rw [hpqr]
      simp only [Polynomial.eval_mul]
      nlinarith [sq_nonneg (q.eval x), hrpos x hx]
    have h := hpos (p * q) hpqne hnonneg
    rw [horthq] at h
    exact (lt_irrefl 0) h
  · have hnonneg : ∀ x ∈ Set.Ioo a b, 0 ≤ (-(p * q)).eval x := by
      intro x hx
      rw [hpqr]
      simp only [Polynomial.eval_neg, Polynomial.eval_mul]
      nlinarith [sq_nonneg (q.eval x), hrneg x hx]
    have h := hpos (-(p * q)) (neg_ne_zero.mpr hpqne) hnonneg
    rw [map_neg, horthq, neg_zero] at h
    exact (lt_irrefl 0) h

/-- Every real root of an interval-orthogonal polynomial is in the open interval. -/
theorem orthogonal_roots_mem_interval (L : ℝ[X] →ₗ[ℝ] ℝ)
    (a b : ℝ) (hab : a < b)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → (∀ x ∈ Set.Ioo a b, 0 ≤ u.eval x) → 0 < L u)
    (p : ℝ[X]) (hp : p ≠ 0)
    (horth : ∀ u : ℝ[X], u.natDegree < p.natDegree → L (p * u) = 0) :
    ∀ x ∈ p.roots, x ∈ Set.Ioo a b := by
  classical
  have hc := orthogonal_interval_root_count L a b hab hpos p hp horth
  have heq : p.roots.filter (fun x ↦ x ∈ Set.Ioo a b) = p.roots := by
    apply Multiset.eq_of_le_of_card_le (Multiset.filter_le _ _)
    rw [hc]
    exact Polynomial.card_roots' p
  intro x hx
  have hmem : x ∈ p.roots.filter (fun x ↦ x ∈ Set.Ioo a b) := by rwa [heq]
  exact (Multiset.mem_filter.mp hmem).2

/-- The total real-root count reaches the polynomial degree. -/
theorem orthogonal_roots_card (L : ℝ[X] →ₗ[ℝ] ℝ)
    (a b : ℝ) (hab : a < b)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → (∀ x ∈ Set.Ioo a b, 0 ≤ u.eval x) → 0 < L u)
    (p : ℝ[X]) (hp : p ≠ 0)
    (horth : ∀ u : ℝ[X], u.natDegree < p.natDegree → L (p * u) = 0) :
    p.roots.card = p.natDegree := by
  have hc := orthogonal_interval_root_count L a b hab hpos p hp horth
  exact le_antisymm (Polynomial.card_roots' p) (by
    rw [← hc]
    exact Multiset.card_le_card (Multiset.filter_le _ _))

/-- No squared real linear factor can divide an interval-orthogonal polynomial:
orthogonality would give zero integral to a nonzero nonnegative polynomial. -/
theorem orthogonal_no_square_factor (L : ℝ[X] →ₗ[ℝ] ℝ) (a b : ℝ)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → (∀ x ∈ Set.Ioo a b, 0 ≤ u.eval x) → 0 < L u)
    (p : ℝ[X]) (hp : p ≠ 0)
    (horth : ∀ u : ℝ[X], u.natDegree < p.natDegree → L (p * u) = 0)
    (c : ℝ) : ¬ (X - C c) ^ 2 ∣ p := by
  rintro ⟨q, hpq⟩
  have hqne : q ≠ 0 := by
    intro h
    rw [h, mul_zero] at hpq
    exact hp hpq
  have hfactor : ((X - C c) ^ 2 : ℝ[X]) ≠ 0 :=
    pow_ne_zero _ (Polynomial.X_sub_C_ne_zero c)
  have hdeg : q.natDegree < p.natDegree := by
    rw [hpq, Polynomial.natDegree_mul hfactor hqne,
      Polynomial.natDegree_pow, Polynomial.natDegree_X_sub_C]
    omega
  have hzero := horth q hdeg
  have hpositive : 0 < L (p * q) := by
    apply hpos (p * q) (mul_ne_zero hp hqne)
    intro x _
    rw [hpq]
    simp only [Polynomial.eval_mul, Polynomial.eval_pow, Polynomial.eval_sub,
      Polynomial.eval_X, Polynomial.eval_C]
    nlinarith [sq_nonneg ((x - c) * q.eval x)]
  rw [hzero] at hpositive
  exact (lt_irrefl 0) hpositive

/-- The real roots of an interval-orthogonal polynomial are simple. -/
theorem orthogonal_roots_nodup (L : ℝ[X] →ₗ[ℝ] ℝ) (a b : ℝ)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → (∀ x ∈ Set.Ioo a b, 0 ≤ u.eval x) → 0 < L u)
    (p : ℝ[X]) (hp : p ≠ 0)
    (horth : ∀ u : ℝ[X], u.natDegree < p.natDegree → L (p * u) = 0) :
    p.roots.Nodup := by
  classical
  apply Multiset.nodup_iff_count_le_one.mpr
  intro c
  rw [Polynomial.count_roots]
  by_contra h
  have hmult : 2 ≤ p.rootMultiplicity c := by omega
  exact orthogonal_no_square_factor L a b hpos p hp horth c
    ((Polynomial.le_rootMultiplicity_iff hp).mp hmult)

end QIRadon.Gaussian
