import QIRadon.Quadrature.HomogeneousQuadrature
import Mathlib.Algebra.Order.Antidiag.FinsuppEquiv
import Mathlib.Algebra.MvPolynomial.Funext

open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section

def homogeneousExponentEquiv (n k : ℕ) :
    {s : Fin (n + 1) →₀ ℕ // s.degree = k} ≃
      ((Finset.univ : Finset (Fin (n + 1))).finsuppAntidiag k) where
  toFun s := ⟨s.val, by
    rw [Finset.mem_finsuppAntidiag']
    exact ⟨s.property, Finset.subset_univ _⟩⟩
  invFun s := ⟨s.val, (Finset.mem_finsuppAntidiag'.mp s.property).1⟩
  left_inv s := rfl
  right_inv s := rfl

instance homogeneousExponentFintype (n k : ℕ) :
    Fintype {s : Fin (n + 1) →₀ ℕ // s.degree = k} :=
  Fintype.ofEquiv ((Finset.univ : Finset (Fin (n + 1))).finsuppAntidiag k)
    (homogeneousExponentEquiv n k).symm

theorem homogeneousExponent_card (n k : ℕ) :
    Fintype.card {s : Fin (n + 1) →₀ ℕ // s.degree = k} = (n + k).choose k := by
  rw [Fintype.card_congr (homogeneousExponentEquiv n k), Fintype.card_coe,
    Finset.card_finsuppAntidiag_nat_eq_choose]
  simp

/-- Exact stars-and-bars dimension of the actual homogeneous polynomial space. -/
theorem homogeneous_finrank (n k : ℕ) :
    Module.finrank ℝ (HomogeneousPolynomials n k) = (n + k).choose k := by
  let : Fintype (↑{s : Fin (n + 1) →₀ ℕ | s.degree = k}) := homogeneousExponentFintype n k
  change Module.finrank ℝ (MvPolynomial.homogeneousSubmodule (Fin (n + 1)) ℝ k) = _
  rw [MvPolynomial.homogeneousSubmodule_eq_finsupp_supported]
  rw [(AddMonoidAlgebra.supportedEquivFinsupp (R := ℝ) (S := ℝ)
    {s : Fin (n + 1) →₀ ℕ | s.degree = k}).finrank_eq]
  rw [Module.finrank_finsupp, Module.finrank_self, mul_one]
  exact homogeneousExponent_card n k

theorem homogeneous_eval_smul (n k : ℕ) (p : HomogeneousPolynomials n k)
    (r : ℝ) (x : EuclideanSpace ℝ (Fin (n + 1))) :
    MvPolynomial.eval (fun i ↦ (r • x) i) p.val =
      r ^ k * MvPolynomial.eval (fun i ↦ x i) p.val := by
  classical
  rw [mvPolynomial_eval_support, mvPolynomial_eval_support, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro s _
  have hdeg : (∑ i, s.val i) = k := by
    rw [← Finsupp.degree_eq_sum]
    simpa only [Finsupp.degree_apply] using
      (p.property.degree_eq_sum_deg_support s.property).symm
  rw [coordinateMonomial_homogeneous, hdeg]
  ring

/-- Restriction of homogeneous polynomials to the sphere is injective. Values
away from zero follow by radial scaling; zero follows from the same homogeneity
identity at scale zero, including degree zero. -/
theorem homogeneousRestriction_injective (n k : ℕ) :
    Function.Injective (homogeneousRestriction n k) := by
  apply (LinearMap.ker_eq_bot).mp
  apply LinearMap.ker_eq_bot'.mpr
  intro p hp
  apply Subtype.ext
  apply MvPolynomial.funext
  intro x
  let y : EuclideanSpace ℝ (Fin (n + 1)) := WithLp.toLp 2 x
  have hzero : ∀ θ : UnitSphere n, MvPolynomial.eval (fun i ↦ θ.val i) p.val = 0 := by
    intro θ
    have h := congrArg (fun f : C(UnitSphere n, ℝ) ↦ f θ) hp
    exact h
  change MvPolynomial.eval (fun i ↦ y i) p.val = MvPolynomial.eval x 0
  rw [map_zero]
  by_cases hy : y = 0
  · let θ : UnitSphere n := ⟨EuclideanSpace.single 0 1, by simp⟩
    have h := homogeneous_eval_smul n k p 0 θ.val
    rw [hzero θ, mul_zero, zero_smul] at h
    simpa [hy] using h
  · let θ : UnitSphere n := ⟨‖y‖⁻¹ • y, by
      change ‖‖y‖⁻¹ • y - 0‖ = 1
      rw [sub_zero, norm_smul, Real.norm_eq_abs, abs_of_nonneg (inv_nonneg.mpr (norm_nonneg _)),
        inv_mul_cancel₀ (norm_ne_zero_iff.mpr hy)]⟩
    have h := homogeneous_eval_smul n k p ‖y‖ θ.val
    have hyθ : ‖y‖ • θ.val = y := by
      change ‖y‖ • (‖y‖⁻¹ • y) = y
      rw [smul_smul, mul_inv_cancel₀ (norm_ne_zero_iff.mpr hy), one_smul]
    rw [hyθ, hzero θ, mul_zero] at h
    exact h

/-- The concrete L2 restriction space has the same exact dimension. -/
theorem polynomialL2_finrank (n k : ℕ) :
    Module.finrank ℝ (PolynomialL2Space n k) = (n + k).choose k := by
  rw [← (polynomialL2Equiv n k).finrank_eq,
    ← (LinearEquiv.ofInjective (homogeneousRestriction n k)
      (homogeneousRestriction_injective n k)).finrank_eq,
    homogeneous_finrank]

end
end QIRadon.Gaussian
