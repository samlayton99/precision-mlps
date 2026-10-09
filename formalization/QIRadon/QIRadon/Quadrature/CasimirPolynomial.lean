import Mathlib.RingTheory.MvPolynomial.EulerIdentity
import Mathlib.Tactic

/-!
# Polynomial rotation generators and the Casimir identity

We sum over all ordered coordinate pairs. The resulting Casimir is twice the
usual sum over i<j. This normalization is explicit and must be retained in the
eigenvalue and analytic-word estimates.
-/

open MvPolynomial
open scoped BigOperators

namespace QIRadon.Casimir

noncomputable section

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

def generator (i j : ι) (p : MvPolynomial ι ℝ) : MvPolynomial ι ℝ :=
  X i * pderiv j p - X j * pderiv i p

def euler (p : MvPolynomial ι ℝ) : MvPolynomial ι ℝ := ∑ i, X i * pderiv i p

def laplace (p : MvPolynomial ι ℝ) : MvPolynomial ι ℝ := ∑ i, pderiv i (pderiv i p)

def radiusSquared : MvPolynomial ι ℝ := ∑ i, X i ^ 2

def orderedCasimir (p : MvPolynomial ι ℝ) : MvPolynomial ι ℝ :=
  ∑ i, ∑ j, generator i j (generator i j p)

def firstSum (p : MvPolynomial ι ℝ) : MvPolynomial ι ℝ :=
  ∑ i, ∑ j, X i * pderiv j (generator i j p)

omit [Fintype ι] [DecidableEq ι] in
theorem generator_swap (i j : ι) (p : MvPolynomial ι ℝ) :
    generator j i p = -generator i j p := by unfold generator; ring

omit [Fintype ι] [DecidableEq ι] in
theorem generator_self (i : ι) (p : MvPolynomial ι ℝ) : generator i i p = 0 := by
  simp [generator]

omit [DecidableEq ι] in
/-- Swapping the ordered indices halves the Casimir computation. -/
theorem orderedCasimir_eq_two_firstSum (p : MvPolynomial ι ℝ) :
    orderedCasimir p = 2 * firstSum p := by
  have hswap : (∑ i, ∑ j, X j * pderiv i (generator i j p)) = -firstSum p := by
    rw [Finset.sum_comm]
    unfold firstSum
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro j _
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i _
    rw [generator_swap j i p, map_neg, mul_neg]
  unfold orderedCasimir
  simp only [generator, Finset.sum_sub_distrib]
  change firstSum p - (∑ i, ∑ j, X j * pderiv i (generator i j p)) = _
  rw [hswap]
  ring

theorem firstSum_eq (p : MvPolynomial ι ℝ) :
    firstSum p = radiusSquared * laplace p + euler p -
      (Fintype.card ι : ℝ) • euler p - ∑ i, X i * euler (pderiv i p) := by
  have hdelta : (∑ i, ∑ j, X i * pderiv j (X i) * pderiv j p) = euler p := by
    unfold euler
    apply Finset.sum_congr rfl
    intro i _
    simp [MvPolynomial.pderiv_X, Pi.single_apply]
  have hradius : (∑ i, ∑ j, X i ^ 2 * pderiv j (pderiv j p)) = radiusSquared * laplace p := by
    simp only [radiusSquared, laplace, Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
  have hcount : (∑ i, ∑ j : ι, X i * pderiv i p) = (Fintype.card ι : ℝ) • euler p := by
    simp only [Finset.sum_const, Finset.card_univ, ← Finset.smul_sum, euler,
      Nat.cast_smul_eq_nsmul]
  have hcross : (∑ i, ∑ j, X i * X j * pderiv j (pderiv i p)) =
      ∑ i, X i * euler (pderiv i p) := by
    simp only [euler, Finset.mul_sum, mul_assoc]
  unfold firstSum
  simp only [generator, map_sub, MvPolynomial.pderiv_mul, MvPolynomial.pderiv_X_self]
  have hexpand :
      (∑ i, ∑ j, X i *
        (pderiv j (X i) * pderiv j p + X i * pderiv j (pderiv j p) -
          (1 * pderiv i p + X j * pderiv j (pderiv i p)))) =
      (∑ i, ∑ j, X i * pderiv j (X i) * pderiv j p) +
      (∑ i, ∑ j, X i ^ 2 * pderiv j (pderiv j p)) -
      (∑ i, ∑ j : ι, X i * pderiv i p) -
      (∑ i, ∑ j, X i * X j * pderiv j (pderiv i p)) := by
    simp only [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [hexpand, hdelta, hradius, hcount, hcross]
  ring

omit [DecidableEq ι] in
theorem euler_homogeneous (p : MvPolynomial ι ℝ) (k : ℕ) (hp : p.IsHomogeneous k) :
    euler p = (k : ℝ) • p := by
  have h := hp.sum_X_mul_pderiv
  simpa only [euler, Nat.cast_smul_eq_nsmul] using h

/-- The polynomial Casimir identity with the explicit ordered-pair factor two.
Its restriction to the intrinsic degree-k orthogonal layer will eliminate the
lower-degree Laplacian term by orthogonality and rotational invariance. -/
theorem orderedCasimir_homogeneous (p : MvPolynomial ι ℝ) (k : ℕ)
    (hp : p.IsHomogeneous k) :
    orderedCasimir p = 2 * (radiusSquared * laplace p) -
      (2 * (k : ℝ) * ((k : ℝ) + (Fintype.card ι : ℝ) - 2)) • p := by
  have hcross : (∑ i, X i * euler (pderiv i p)) = ((k - 1 : ℕ) : ℝ) • euler p := by
    simp_rw [euler_homogeneous _ (k - 1) hp.pderiv, mul_smul_comm,
      ← Finset.smul_sum]
    rfl
  rw [orderedCasimir_eq_two_firstSum, firstSum_eq, hcross, euler_homogeneous p k hp]
  cases k with
  | zero => simp
  | succ k =>
    simp only [Nat.add_sub_cancel, Nat.cast_add, Nat.cast_one, two_mul]
    module

end
end QIRadon.Casimir
