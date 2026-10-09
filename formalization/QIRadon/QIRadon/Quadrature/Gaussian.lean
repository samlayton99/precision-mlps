import Mathlib.LinearAlgebra.Lagrange
import Mathlib.Algebra.Polynomial.Div
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Measure.Typeclasses.NullSingletonClass
import Mathlib.Tactic

/-!
# Algebraic Gaussian quadrature from an orthogonal nodal polynomial

This file proves the substantive algebraic step in Gaussian quadrature: roots of
an orthogonal degree-n nodal polynomial and integrated Lagrange basis weights
give exactness through degree 2n-1. It also derives weight positivity from
positivity on nonzero polynomial squares.

It does NOT assert existence, location, or simplicity of Jacobi roots. Those
must be established for the actual weighted Jacobi integral separately.
-/

open scoped BigOperators
open Polynomial
open MeasureTheory

namespace QIRadon.Gaussian

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The weight associated to one interpolation node, defined by a linear
functional applied to the Lagrange basis polynomial. -/
noncomputable def weight (L : ℝ[X] →ₗ[ℝ] ℝ) (node : ι → ℝ) (i : ι) : ℝ :=
  L (Lagrange.basis Finset.univ node i)

/-- Polynomial interpolation already makes the integrated Lagrange rule exact
below degree n; orthogonality is not required for this preliminary fact. -/
theorem exact_below_card (L : ℝ[X] →ₗ[ℝ] ℝ) (node : ι → ℝ)
    (hinj : Function.Injective node) (p : ℝ[X])
    (hdeg : p.degree < Fintype.card ι) :
    L p = ∑ i, weight L node i * p.eval (node i) := by
  have hi : Set.InjOn node (↑(Finset.univ : Finset ι) : Set ι) :=
    hinj.injOn
  have heq := Lagrange.eq_interpolate hi (by simpa using hdeg)
  calc
    L p = L (Lagrange.interpolate Finset.univ node (fun i ↦ p.eval (node i))) :=
      congrArg L heq
    _ = ∑ i, weight L node i * p.eval (node i) := by
      simp only [Lagrange.interpolate_apply, map_sum, ← Polynomial.smul_eq_C_mul,
        map_smul, smul_eq_mul, weight]
      apply Finset.sum_congr rfl
      intro i _
      ring

/-- Orthogonality of the nodal polynomial upgrades interpolation exactness from
degree n-1 to degree 2n-1. No quadrature exactness is assumed in this theorem. -/
theorem exact_below_twice_card (L : ℝ[X] →ₗ[ℝ] ℝ) (node : ι → ℝ)
    (hinj : Function.Injective node) (hn : 0 < Fintype.card ι)
    (horth : ∀ a : ℝ[X], a.degree < Fintype.card ι →
      L (Lagrange.nodal Finset.univ node * a) = 0)
    (p : ℝ[X]) (hdeg : p.natDegree < 2 * Fintype.card ι) :
    L p = ∑ i, weight L node i * p.eval (node i) := by
  let nodal : ℝ[X] := Lagrange.nodal Finset.univ node
  have hmonic : nodal.Monic := Lagrange.nodal_monic
  have hnd : nodal.natDegree = Fintype.card ι := by
    simp [nodal, Lagrange.natDegree_nodal]
  have hd : nodal.degree = Fintype.card ι := by
    simp [nodal, Lagrange.degree_nodal]
  have hquot : (p /ₘ nodal).degree < Fintype.card ι := by
    have hnquot : (p /ₘ nodal).natDegree < Fintype.card ι := by
      rw [Polynomial.natDegree_divByMonic p hmonic, hnd]
      omega
    exact Polynomial.degree_le_natDegree.trans_lt (by exact_mod_cast hnquot)
  have hrem : (p %ₘ nodal).degree < Fintype.card ι := by
    rw [← hd]
    exact Polynomial.degree_modByMonic_lt p hmonic
  have hvalues : ∀ i, (p %ₘ nodal).eval (node i) = p.eval (node i) := by
    intro i
    have h := congrArg (fun a : ℝ[X] ↦ a.eval (node i))
      (Polynomial.modByMonic_add_div p nodal)
    have hzero : nodal.eval (node i) = 0 :=
      Lagrange.eval_nodal_at_node (Finset.mem_univ i)
    simpa only [Polynomial.eval_add, Polynomial.eval_mul, hzero, zero_mul, add_zero] using h
  calc
    L p = L (p %ₘ nodal) := by
      have h := congrArg L (Polynomial.modByMonic_add_div p nodal)
      rw [map_add, horth _ hquot, add_zero] at h
      exact h.symm
    _ = ∑ i, weight L node i * (p %ₘ nodal).eval (node i) :=
      exact_below_card L node hinj _ hrem
    _ = ∑ i, weight L node i * p.eval (node i) := by simp_rw [hvalues]

theorem basis_eval (node : ι → ℝ) (hinj : Function.Injective node) (i j : ι) :
    (Lagrange.basis Finset.univ node i).eval (node j) = if i = j then 1 else 0 := by
  split_ifs with h
  · subst j
    exact Lagrange.eval_basis_self hinj.injOn (Finset.mem_univ i)
  · exact Lagrange.eval_basis_of_ne h (Finset.mem_univ j)

/-- Each actual integrated Lagrange weight equals the functional on a square.
This is the usual proof of positivity of Gaussian quadrature weights. -/
theorem weight_eq_square (L : ℝ[X] →ₗ[ℝ] ℝ) (node : ι → ℝ)
    (hinj : Function.Injective node) (hn : 0 < Fintype.card ι)
    (horth : ∀ a : ℝ[X], a.degree < Fintype.card ι →
      L (Lagrange.nodal Finset.univ node * a) = 0) (i : ι) :
    weight L node i = L ((Lagrange.basis Finset.univ node i) ^ 2) := by
  have hdeg : ((Lagrange.basis Finset.univ node i) ^ 2).natDegree <
      2 * Fintype.card ι := by
    rw [Polynomial.natDegree_pow,
      Lagrange.natDegree_basis hinj.injOn (Finset.mem_univ i)]
    simp only [Finset.card_univ]
    omega
  have h := exact_below_twice_card L node hinj hn horth
    ((Lagrange.basis Finset.univ node i) ^ 2) hdeg
  simp only [Polynomial.eval_pow, basis_eval node hinj] at h
  simpa using h.symm

/-- Positivity follows from positivity on nonzero polynomial squares, rather
than being assumed of the quadrature weights. -/
theorem weight_pos (L : ℝ[X] →ₗ[ℝ] ℝ) (node : ι → ℝ)
    (hinj : Function.Injective node) (hn : 0 < Fintype.card ι)
    (horth : ∀ a : ℝ[X], a.degree < Fintype.card ι →
      L (Lagrange.nodal Finset.univ node * a) = 0)
    (hpos : ∀ a : ℝ[X], a ≠ 0 → 0 < L (a ^ 2)) (i : ι) :
    0 < weight L node i := by
  rw [weight_eq_square L node hinj hn horth i]
  exact hpos _ (Lagrange.basis_ne_zero hinj.injOn (Finset.mem_univ i))

/-- Normalization of the functional implies normalization of the derived rule. -/
theorem sum_weights (L : ℝ[X] →ₗ[ℝ] ℝ) (node : ι → ℝ)
    (hinj : Function.Injective node) (hn : 0 < Fintype.card ι) (hunit : L 1 = 1) :
    ∑ i, weight L node i = 1 := by
  have hdeg : (1 : ℝ[X]).degree < Fintype.card ι := by
    simpa using (show (0 : WithBot ℕ) < (Fintype.card ι : WithBot ℕ) by exact_mod_cast hn)
  have h := exact_below_card L node hinj 1 hdeg
  simpa [hunit] using h.symm

/-- The actual integral is a linear functional on polynomials whenever every
polynomial is integrable. This holds for the finite interval Jacobi measures. -/
noncomputable def polynomialIntegral (μ : Measure ℝ)
    (hint : ∀ p : ℝ[X], Integrable (fun x ↦ p.eval x) μ) : ℝ[X] →ₗ[ℝ] ℝ where
  toFun p := ∫ x, p.eval x ∂μ
  map_add' p q := by
    simp only [Polynomial.eval_add]
    exact integral_add (hint p) (hint q)
  map_smul' c p := by
    simp only [Polynomial.eval_smul, smul_eq_mul, RingHom.id_apply]
    exact integral_const_mul c _

theorem polynomialIntegral_one (μ : Measure ℝ) [IsProbabilityMeasure μ]
    (hint : ∀ p : ℝ[X], Integrable (fun x ↦ p.eval x) μ) :
    polynomialIntegral μ hint 1 = 1 := by
  simp [polynomialIntegral]

/-- For any atomless probability measure with polynomial moments, a nonzero
polynomial has strictly positive squared integral. Its finite root set is null.
Thus the positivity hypothesis used above is verified from measure properties. -/
theorem polynomialIntegral_square_pos (μ : Measure ℝ)
    [IsProbabilityMeasure μ] [NullSingletonClass μ]
    (hint : ∀ p : ℝ[X], Integrable (fun x ↦ p.eval x) μ)
    (p : ℝ[X]) (hp : p ≠ 0) :
    0 < polynomialIntegral μ hint (p ^ 2) := by
  change 0 < ∫ x, (p ^ 2).eval x ∂μ
  simp only [Polynomial.eval_pow]
  have hroots : Set.Finite {x : ℝ | p.eval x = 0} :=
    Polynomial.finite_setOfPred_isRoot hp
  have hnull : μ {x : ℝ | p.eval x = 0} = 0 := hroots.measure_zero μ
  have hsupport : Function.support (fun x : ℝ ↦ p.eval x ^ 2) =
      {x : ℝ | p.eval x = 0}ᶜ := by
    ext x
    simp [Function.support]
  have hi : Integrable (fun x : ℝ ↦ p.eval x ^ 2) μ := by
    convert hint (p ^ 2) using 1
    funext x
    exact (Polynomial.eval_pow 2).symm
  apply (integral_pos_iff_support_of_nonneg (fun x ↦ sq_nonneg (p.eval x))
    hi).2
  rw [hsupport, measure_compl hroots.measurableSet (by simp [hnull]), hnull]
  simp

/-- An actual Gaussian integral rule has positive weights, conditional only on
the construction of its distinct orthogonal nodes. -/
theorem integral_weight_pos (μ : Measure ℝ)
    [IsProbabilityMeasure μ] [NullSingletonClass μ]
    (hint : ∀ p : ℝ[X], Integrable (fun x ↦ p.eval x) μ)
    (node : ι → ℝ) (hinj : Function.Injective node) (hn : 0 < Fintype.card ι)
    (horth : ∀ a : ℝ[X], a.degree < Fintype.card ι →
      polynomialIntegral μ hint (Lagrange.nodal Finset.univ node * a) = 0)
    (i : ι) :
    0 < weight (polynomialIntegral μ hint) node i :=
  weight_pos (polynomialIntegral μ hint) node hinj hn horth
    (polynomialIntegral_square_pos μ hint) i

end QIRadon.Gaussian
