import QIRadon.Quadrature.SphereRecursion
import QIRadon.Quadrature.ErrorTransport
import QIRadon.Quadrature.JacobiSymmetry

/-!
# The actual recursively constructed Gauss–Jacobi sphere rule

Depth n means ambient dimension n+1. The base is S⁰={-1,+1}; the first
Jacobi lift uses a=-1/2. This is the half-offset-circle version of the rule.
Its identification with arbitrary phase circle rules is a separate corollary.
-/

open Polynomial MeasureTheory Metric
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable def sphereJacobiNode (q : ℕ) (hq : 0 < q) (n : ℕ) : Fin q → ℝ :=
  (exists_gaussJacobi_rule_general (sphereJacobiExponent n)
    (sphereJacobiExponent_gt n) q hq).choose

noncomputable def sphereJacobiWeight (q : ℕ) (hq : 0 < q) (n : ℕ) : Fin q → ℝ :=
  weight (jacobiMomentGeneral (sphereJacobiExponent n) (sphereJacobiExponent_gt n))
    (sphereJacobiNode q hq n)

theorem sphereJacobi_inside (q : ℕ) (hq : 0 < q) (n : ℕ) (i : Fin q) :
    sphereJacobiNode q hq n i ∈ Set.Ioo (-1 : ℝ) 1 :=
  (exists_gaussJacobi_rule_general (sphereJacobiExponent n)
    (sphereJacobiExponent_gt n) q hq).choose_spec.2.1 i

theorem sphereJacobi_injective (q : ℕ) (hq : 0 < q) (n : ℕ) :
    Function.Injective (sphereJacobiNode q hq n) :=
  (exists_gaussJacobi_rule_general (sphereJacobiExponent n)
    (sphereJacobiExponent_gt n) q hq).choose_spec.1

theorem sphereJacobi_positive (q : ℕ) (hq : 0 < q) (n : ℕ) (i : Fin q) :
    0 < sphereJacobiWeight q hq n i :=
  (exists_gaussJacobi_rule_general (sphereJacobiExponent n)
    (sphereJacobiExponent_gt n) q hq).choose_spec.2.2.1 i

theorem sphereJacobi_normalized (q : ℕ) (hq : 0 < q) (n : ℕ) :
    ∑ i, sphereJacobiWeight q hq n i = 1 :=
  (exists_gaussJacobi_rule_general (sphereJacobiExponent n)
    (sphereJacobiExponent_gt n) q hq).choose_spec.2.2.2.1

theorem sphereJacobi_exact (q : ℕ) (hq : 0 < q) (n : ℕ) (p : ℝ[X])
    (hp : p.natDegree < 2 * q) :
    jacobiMomentGeneral (sphereJacobiExponent n) (sphereJacobiExponent_gt n) p =
      ∑ i, sphereJacobiWeight q hq n i * p.eval (sphereJacobiNode q hq n i) :=
  (exists_gaussJacobi_rule_general (sphereJacobiExponent n)
    (sphereJacobiExponent_gt n) q hq).choose_spec.2.2.2.2 p hp

def SphereIndex (q : ℕ) : ℕ → Type
  | 0 => Bool
  | n + 1 => Fin q × SphereIndex q n

instance sphereIndexFintype (q : ℕ) : (n : ℕ) → Fintype (SphereIndex q n)
  | 0 => inferInstanceAs (Fintype Bool)
  | n + 1 => @instFintypeProd _ _ (inferInstanceAs (Fintype (Fin q))) (sphereIndexFintype q n)

theorem sphereIndex_card (q n : ℕ) : Fintype.card (SphereIndex q n) = 2 * q ^ n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change Fintype.card (Fin q × SphereIndex q n) = _
    rw [Fintype.card_prod, Fintype.card_fin, ih, pow_succ]
    ring

noncomputable def sphereNode (q : ℕ) (hq : 0 < q) :
    (n : ℕ) → SphereIndex q n → EuclideanSpace ℝ (Fin (n + 1))
  | 0, i => axisPoint i
  | n + 1, ij => liftPoint (sphereJacobiNode q hq n ij.1) (sphereNode q hq n ij.2)

noncomputable def sphereWeight (q : ℕ) (hq : 0 < q) : (n : ℕ) → SphereIndex q n → ℝ
  | 0, _ => 1 / 2
  | n + 1, ij => sphereJacobiWeight q hq n ij.1 * sphereWeight q hq n ij.2

theorem sphereNode_norm (q : ℕ) (hq : 0 < q) (n : ℕ) (i : SphereIndex q n) :
    ‖sphereNode q hq n i‖ = 1 := by
  induction n with
  | zero => exact axisPoint_norm i
  | succ n ih =>
    apply norm_liftPoint
    · exact ⟨(sphereJacobi_inside q hq n i.1).1.le, (sphereJacobi_inside q hq n i.1).2.le⟩
    · exact ih i.2

theorem sphereWeight_positive (q : ℕ) (hq : 0 < q) (n : ℕ) (i : SphereIndex q n) :
    0 < sphereWeight q hq n i := by
  induction n with
  | zero => norm_num [sphereWeight]
  | succ n ih => exact mul_pos (sphereJacobi_positive q hq n i.1) (ih i.2)

theorem sphereWeight_normalized (q : ℕ) (hq : 0 < q) (n : ℕ) :
    ∑ i, sphereWeight q hq n i = 1 := by
  induction n with
  | zero =>
    change (∑ i : Bool, (1 / 2 : ℝ)) = 1
    rw [Fintype.sum_bool]
    norm_num
  | succ n ih =>
    change (∑ ij : Fin q × SphereIndex q n,
      sphereJacobiWeight q hq n ij.1 * sphereWeight q hq n ij.2) = 1
    simp only [Fintype.sum_prod_type, ← Finset.mul_sum, ih, mul_one,
      sphereJacobi_normalized]

/-- Exact even-coordinate moments through the quadrature resolution. -/
theorem sphereRule_even_exact (q : ℕ) (hq : 0 < q) (n : ℕ)
    (k : Fin (n + 1) → ℕ) (hk : 2 * (∑ i, k i) < 2 * q) :
    (∑ i, sphereWeight q hq n i * coordinateMonomial (fun j ↦ 2 * k j) (sphereNode q hq n i)) =
      sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
        coordinateMonomial (fun j ↦ 2 * k j) θ.val) := by
  induction n with
  | zero => exact base_sphere_monomial_exact (fun j ↦ 2 * k j)
  | succ n ih =>
    let p := jacobiTestPolynomial (2 * k 0) (∑ j : Fin (n + 1), k j.succ)
    have hk' : 2 * (∑ j : Fin (n + 1), k j.succ) < 2 * q := by
      rw [Fin.sum_univ_succ] at hk
      omega
    have hp : p.natDegree < 2 * q := by
      apply (jacobiTestPolynomial_degree _ _).trans_lt
      rw [Fin.sum_univ_succ] at hk
      omega
    calc
      _ = ∑ ij : Fin q × SphereIndex q n,
          (sphereJacobiWeight q hq n ij.1 * sphereWeight q hq n ij.2) *
          (p.eval (sphereJacobiNode q hq n ij.1) *
            coordinateMonomial (fun j : Fin (n + 1) ↦ 2 * k j.succ) (sphereNode q hq n ij.2)) := by
        apply Finset.sum_congr rfl
        intro ij _
        change _ * coordinateMonomial _ (liftPoint _ _) = _
        rw [even_monomial_liftPoint _
          ⟨(sphereJacobi_inside q hq n ij.1).1.le, (sphereJacobi_inside q hq n ij.1).2.le⟩]
        simp only [p, jacobiTestPolynomial, Polynomial.eval_mul, Polynomial.eval_pow,
          Polynomial.eval_X, Polynomial.eval_sub, Polynomial.eval_one]
        rfl
      _ = (∑ i, sphereJacobiWeight q hq n i * p.eval (sphereJacobiNode q hq n i)) *
          (∑ i, sphereWeight q hq n i *
            coordinateMonomial (fun j : Fin (n + 1) ↦ 2 * k j.succ) (sphereNode q hq n i)) :=
        QIRadon.product_sum_separable (sphereJacobiWeight q hq n) (sphereWeight q hq n)
          (fun i ↦ p.eval (sphereJacobiNode q hq n i))
          (fun i ↦ coordinateMonomial (fun j : Fin (n + 1) ↦ 2 * k j.succ) (sphereNode q hq n i))
      _ = jacobiMomentGeneral (sphereJacobiExponent n) (sphereJacobiExponent_gt n) p *
          sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
            coordinateMonomial (fun j : Fin (n + 1) ↦ 2 * k j.succ) θ.val) := by
        rw [← sphereJacobi_exact q hq n p hp, ih _ hk']
      _ = _ := (sphere_even_moment_recursion n k).symm

/-- Factor any discrete monomial moment into its top coordinate and lower
dimensional moment. No polynomial degree or parity assumption is needed. -/
theorem sphereRule_monomial_step (q : ℕ) (hq : 0 < q) (n : ℕ)
    (k : Fin (n + 2) → ℕ) :
    (∑ i, sphereWeight q hq (n + 1) i * coordinateMonomial k (sphereNode q hq (n + 1) i)) =
    (∑ i, sphereJacobiWeight q hq n i *
      (sphereJacobiNode q hq n i ^ k 0 *
        Real.sqrt (1 - sphereJacobiNode q hq n i ^ 2) ^ (∑ j : Fin (n + 1), k j.succ))) *
    (∑ i, sphereWeight q hq n i *
      coordinateMonomial (fun j : Fin (n + 1) ↦ k j.succ) (sphereNode q hq n i)) := by
  calc
    _ = ∑ ij : Fin q × SphereIndex q n,
        (sphereJacobiWeight q hq n ij.1 * sphereWeight q hq n ij.2) *
          ((sphereJacobiNode q hq n ij.1 ^ k 0 *
            Real.sqrt (1 - sphereJacobiNode q hq n ij.1 ^ 2) ^ (∑ j : Fin (n + 1), k j.succ)) *
            coordinateMonomial (fun j : Fin (n + 1) ↦ k j.succ) (sphereNode q hq n ij.2)) := by
      apply Finset.sum_congr rfl
      intro ij _
      change _ * coordinateMonomial _ (liftPoint _ _) = _
      rw [monomial_liftPoint]
      rfl
    _ = _ := QIRadon.product_sum_separable
      (sphereJacobiWeight q hq n) (sphereWeight q hq n)
      (fun i ↦ sphereJacobiNode q hq n i ^ k 0 *
        Real.sqrt (1 - sphereJacobiNode q hq n i ^ 2) ^ (∑ j : Fin (n + 1), k j.succ))
      (fun i ↦ coordinateMonomial (fun j : Fin (n + 1) ↦ k j.succ) (sphereNode q hq n i))

/-- Reflection symmetry makes every coordinate-odd monomial moment exactly zero,
regardless of its degree. -/
theorem sphereRule_odd_zero (q : ℕ) (hq : 0 < q) (n : ℕ)
    (k : Fin (n + 1) → ℕ) (j : Fin (n + 1)) (hj : Odd (k j)) :
    (∑ i, sphereWeight q hq n i * coordinateMonomial k (sphereNode q hq n i)) = 0 := by
  induction n with
  | zero =>
    rw [show j = 0 from Fin.eq_zero j] at hj
    change (∑ i : Bool, (1 / 2 : ℝ) * coordinateMonomial k (axisPoint i)) = 0
    rw [base_sphere_monomial_exact, sphereAverage_monomial_zero_of_odd k 0 hj]
  | succ n ih =>
    rw [sphereRule_monomial_step]
    refine Fin.cases ?_ (fun j ↦ ?_) j hj
    · intro hhead
      have hzero := gaussJacobi_sum_odd (sphereJacobiExponent n)
        (sphereJacobiExponent_gt n) hq (sphereJacobiNode q hq n)
        (sphereJacobi_injective q hq n) (sphereJacobi_exact q hq n)
        (fun t : ℝ ↦ t ^ k 0 * Real.sqrt (1 - t ^ 2) ^ (∑ j : Fin (n + 1), k j.succ))
        (by
          intro t
          rw [hhead.neg_pow, neg_sq]
          ring)
      change (∑ i, sphereJacobiWeight q hq n i * _) = 0 at hzero
      rw [hzero, zero_mul]
    · intro htail
      rw [ih (fun i ↦ k i.succ) j htail, mul_zero]

/-- Exactness of the actually constructed sphere rule for every coordinate
monomial of total degree below 2q. -/
theorem sphereRule_monomial_exact (q : ℕ) (hq : 0 < q) (n : ℕ)
    (k : Fin (n + 1) → ℕ) (hk : (∑ i, k i) < 2 * q) :
    (∑ i, sphereWeight q hq n i * coordinateMonomial k (sphereNode q hq n i)) =
      sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
        coordinateMonomial k θ.val) := by
  classical
  by_cases hodd : ∃ j, Odd (k j)
  · obtain ⟨j, hj⟩ := hodd
    rw [sphereRule_odd_zero q hq n k j hj, sphereAverage_monomial_zero_of_odd k j hj]
  · have heven : ∀ j, Even (k j) := by
      intro j
      rcases Nat.even_or_odd (k j) with he | ho
      · exact he
      · exact False.elim (hodd ⟨j, ho⟩)
    choose l hl using heven
    have hkl : k = fun j ↦ 2 * l j := by
      funext j
      rw [hl j]
      omega
    rw [hkl] at hk ⊢
    rw [← Finset.mul_sum] at hk
    exact sphereRule_even_exact q hq n l hk

theorem sphere_monomial_integrable {n : ℕ} (k : Fin (n + 1) → ℕ) :
    Integrable (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
      coordinateMonomial k θ.val)
      ((volume : Measure (EuclideanSpace ℝ (Fin (n + 1)))).toSphere) := by
  have hc : Continuous (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
      coordinateMonomial k θ.val) := by
    unfold coordinateMonomial
    fun_prop
  exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)

/-- Full spherical-polynomial exactness, expressed as a finite linear combination
of coordinate monomials with the explicit total-degree bound. Every node,
weight, and sphere integral in this theorem is the actual construction above. -/
theorem sphereRule_polynomial_exact {τ : Type*} [Fintype τ]
    (q : ℕ) (hq : 0 < q) (n : ℕ) (coefficient : τ → ℝ)
    (exponent : τ → Fin (n + 1) → ℕ)
    (hdegree : ∀ t, (∑ j, exponent t j) < 2 * q) :
    (∑ i, sphereWeight q hq n i *
      (∑ t, coefficient t * coordinateMonomial (exponent t) (sphereNode q hq n i))) =
      sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
        ∑ t, coefficient t * coordinateMonomial (exponent t) θ.val) := by
  calc
    _ = ∑ t, coefficient t *
        (∑ i, sphereWeight q hq n i * coordinateMonomial (exponent t) (sphereNode q hq n i)) := by
      simp only [Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro t _
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = ∑ t, coefficient t *
        sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ (Fin (n + 1))) 1 ↦
          coordinateMonomial (exponent t) θ.val) := by
      simp_rw [sphereRule_monomial_exact q hq n _ (hdegree _)]
    _ = _ := by
      unfold sphereAverage
      rw [integral_finsetSum Finset.univ (fun t _ ↦ (sphere_monomial_integrable (exponent t)).const_mul _),
        Finset.sum_div]
      apply Finset.sum_congr rfl
      intro t _
      rw [integral_const_mul]
      ring

end QIRadon.Gaussian
