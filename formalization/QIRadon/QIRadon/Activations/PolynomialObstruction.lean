import Mathlib

/-! Fixed-degree polynomial activations cannot become universal by increasing
 width. The obstruction below has an explicit finite-node error constant. -/

open Polynomial
open scoped BigOperators

namespace QIRadon.PolynomialActivation

noncomputable section

def networkPolynomial {W : ℕ} (P : Polynomial ℝ) (v a b : Fin W → ℝ) (bias : ℝ) :
    Polynomial ℝ := C bias + ∑ j, C (v j) * P.comp (C (a j)*X+C (b j))

def network {W : ℕ} (P : Polynomial ℝ) (v a b : Fin W → ℝ) (bias x : ℝ) : ℝ :=
  bias + ∑ j, v j * P.eval (a j*x+b j)

theorem networkPolynomial_eval {W : ℕ} (P : Polynomial ℝ) (v a b : Fin W → ℝ)
    (bias x : ℝ) : (networkPolynomial P v a b bias).eval x = network P v a b bias x := by
  simp [networkPolynomial, network, eval_finsetSum]

theorem networkPolynomial_degree {W m : ℕ} (P : Polynomial ℝ) (hP : P.natDegree ≤ m)
    (v a b : Fin W → ℝ) (bias : ℝ) : (networkPolynomial P v a b bias).natDegree ≤ m := by
  apply natDegree_add_le_of_degree_le (by simp)
  apply natDegree_sum_le_of_forall_le
  intro j _
  apply (natDegree_C_mul_le _ _).trans
  apply natDegree_comp_le.trans
  have haff : (C (a j)*X+C (b j) : Polynomial ℝ).natDegree ≤ 1 := by
    apply natDegree_add_le_of_degree_le
    · exact (natDegree_C_mul_le _ _).trans (by simp)
    · simp
  exact (Nat.mul_le_mul hP haff).trans_eq (Nat.mul_one m)

def testNode (m : ℕ) (i : Fin (m+2)) : ℝ := (i:ℝ)/(m+1:ℕ)

theorem testNode_injective (m : ℕ) : Function.Injective (testNode m) := by
  intro i j hij
  have hden : ((m+1:ℕ):ℝ) ≠ 0 := by positivity
  have h : (i:ℝ) = (j:ℝ) := (div_left_inj' hden).mp hij
  apply Fin.ext
  exact_mod_cast h

theorem testNode_mem (m : ℕ) (i : Fin (m+2)) : testNode m i ∈ Set.Icc (0:ℝ) 1 := by
  have hden : (0:ℝ) < (m+1:ℕ) := by positivity
  constructor
  · unfold testNode; positivity
  · apply (div_le_iff₀ hden).mpr
    have hi : (i:ℕ) ≤ m+1 := by omega
    simpa using (show (i:ℝ) ≤ (m+1:ℕ) by exact_mod_cast hi)

def nodeDenominator (m : ℕ) (i : Fin (m+2)) : ℝ :=
  ∏ j ∈ Finset.univ.erase i, (testNode m i-testNode m j)

theorem nodeDenominator_ne_zero (m : ℕ) (i : Fin (m+2)) : nodeDenominator m i ≠ 0 := by
  apply Finset.prod_ne_zero_iff.mpr
  intro j hj
  apply sub_ne_zero.mpr
  intro h
  exact (Finset.mem_erase.mp hj).1 ((testNode_injective m h).symm)

def obstructionMass (m : ℕ) : ℝ := ∑ i : Fin (m+2), |(nodeDenominator m i)⁻¹|

theorem obstructionMass_pos (m : ℕ) : 0 < obstructionMass m := by
  apply Finset.sum_pos
  · intro i _
    exact abs_pos.mpr (inv_ne_zero (nodeDenominator_ne_zero m i))
  · exact Finset.univ_nonempty

/-- No degree-m polynomial uniformly approximates x^(m+1) below this explicit
 constant, determined by m+2 fixed nodes in [0,1]. -/
theorem polynomial_error_lower_bound (m : ℕ) (Q : Polynomial ℝ) (hQ : Q.natDegree ≤ m)
    (E : ℝ) (hE : ∀ i : Fin (m+2), |testNode m i^(m+1)-Q.eval (testNode m i)| ≤ E) :
    1/obstructionMass m ≤ E := by
  let R : Polynomial ℝ := X^(m+1)-Q
  have hdeg : R.degree < (Finset.univ : Finset (Fin (m+2))).card := by
    apply lt_of_le_of_lt (degree_le_natDegree) ?_
    have hn : R.natDegree ≤ m+1 := by
      apply (natDegree_sub_le _ _).trans
      exact max_le (by simp) (hQ.trans (by omega))
    simp only [Finset.card_univ, Fintype.card_fin]
    exact_mod_cast (lt_of_le_of_lt hn (by omega : m + 1 < m + 2))
  have hc : R.coeff (m+1) = 1 := by
    simp only [R, coeff_sub, coeff_X_pow_self, coeff_eq_zero_of_natDegree_lt (by omega : Q.natDegree < m+1)]
    ring
  have hsum := Lagrange.coeff_eq_sum (testNode_injective m).injOn hdeg
  simp only [Finset.card_univ, Fintype.card_fin] at hsum
  rw [show m+2-1 = m+1 by omega, hc] at hsum
  have hsum' : (1:ℝ) = ∑ i : Fin (m+2),
      (testNode m i^(m+1)-Q.eval (testNode m i))*(nodeDenominator m i)⁻¹ := by
    simpa only [hc, R, eval_sub, eval_pow, eval_X, div_eq_mul_inv, nodeDenominator] using hsum
  have hbound : (1:ℝ) ≤ E*obstructionMass m := by
    calc
      1 = |∑ i : Fin (m+2), (testNode m i^(m+1)-Q.eval (testNode m i))*(nodeDenominator m i)⁻¹| := by
        rw [← hsum']; norm_num
      _ ≤ ∑ i : Fin (m+2), |(testNode m i^(m+1)-Q.eval (testNode m i))*(nodeDenominator m i)⁻¹| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i : Fin (m+2), E*|(nodeDenominator m i)⁻¹| := by
        apply Finset.sum_le_sum
        intro i _
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_right (hE i) (abs_nonneg _)
      _ = _ := by rw [obstructionMass, Finset.mul_sum]
  exact (div_le_iff₀ (obstructionMass_pos m)).mpr hbound

/-- Width, affine first-layer parameters, and readouts cannot remove the
 obstruction for a fixed-degree polynomial activation. -/
theorem network_error_lower_bound {W m : ℕ} (P : Polynomial ℝ) (hP : P.natDegree ≤ m)
    (v a b : Fin W → ℝ) (bias E : ℝ)
    (hE : ∀ x ∈ Set.Icc (0:ℝ) 1, |x^(m+1)-network P v a b bias x| ≤ E) :
    1/obstructionMass m ≤ E := by
  apply polynomial_error_lower_bound m (networkPolynomial P v a b bias)
    (networkPolynomial_degree P hP v a b bias) E
  intro i
  rw [networkPolynomial_eval]
  exact hE _ (testNode_mem m i)

end
end QIRadon.PolynomialActivation
