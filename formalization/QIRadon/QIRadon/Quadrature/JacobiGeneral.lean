import QIRadon.Quadrature.Jacobi
import QIRadon.Quadrature.BetaMoments

/-!
# Symmetric Gauss–Jacobi quadrature for every exponent above -1

This includes the arcsine exponent -1/2 required at the first sphere lift. It
uses the proved integrability of the actual singular density, not a continuity
assumption at the endpoints.
-/

open Polynomial MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

theorem jacobi_polynomial_intervalIntegrable (a : ℝ) (ha : -1 < a) (p : ℝ[X]) :
    IntervalIntegrable (fun x ↦ jacobiDensity a x * p.eval x) volume (-1) 1 := by
  exact (QIRadon.Quadrature.jacobi_weight_intervalIntegrable ha).mul_continuousOn
    p.continuous.continuousOn

noncomputable def rawJacobiMomentGeneral (a : ℝ) (ha : -1 < a) : ℝ[X] →ₗ[ℝ] ℝ where
  toFun p := ∫ x in (-1 : ℝ)..1, jacobiDensity a x * p.eval x
  map_add' p q := by
    simp only [Polynomial.eval_add, mul_add]
    exact intervalIntegral.integral_add
      (jacobi_polynomial_intervalIntegrable a ha p)
      (jacobi_polynomial_intervalIntegrable a ha q)
  map_smul' c p := by
    simp only [Polynomial.eval_smul, smul_eq_mul, RingHom.id_apply]
    simp_rw [mul_left_comm (jacobiDensity a _) c]
    exact intervalIntegral.integral_const_mul _ _

theorem rawJacobiMomentGeneral_pos (a : ℝ) (ha : -1 < a) (p : ℝ[X]) (hp : p ≠ 0)
    (hnonneg : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 ≤ p.eval x) :
    0 < rawJacobiMomentGeneral a ha p := by
  let f : ℝ → ℝ := fun x ↦ jacobiDensity a x * p.eval x
  have hfi : IntervalIntegrable f volume (-1) 1 := jacobi_polynomial_intervalIntegrable a ha p
  have hnonneg_ae : 0 ≤ᵐ[volume.restrict (Set.uIoc (-1 : ℝ) 1)] f := by
    rw [Set.uIoc_of_le (by norm_num : (-1 : ℝ) ≤ 1)]
    apply ae_restrict_of_ae_eq_of_ae_restrict Ioo_ae_eq_Ioc
    rw [ae_restrict_iff' measurableSet_Ioo]
    filter_upwards [] with x hx
    exact mul_nonneg (jacobiDensity_pos a x hx).le (hnonneg x hx)
  let U : Set ℝ := Set.Ioo (-1) 1 ∩ {x | 0 < p.eval x}
  have hU : IsOpen U := isOpen_Ioo.inter (isOpen_lt continuous_const p.continuous)
  have hinf : (Set.Ioo (-1 : ℝ) 1).Infinite := Set.Ioo_infinite (by norm_num)
  obtain ⟨c, hc, hcroot⟩ := hinf.exists_notMem_finite
    (Polynomial.finite_setOfPred_isRoot hp)
  have hpne : p.eval c ≠ 0 := hcroot
  have hcpos : 0 < p.eval c := lt_of_le_of_ne (hnonneg c hc) hpne.symm
  have hUpos : 0 < volume U := hU.measure_pos volume ⟨c, hc, hcpos⟩
  have hsub : U ⊆ Function.support f ∩ Set.Ioc (-1) 1 := by
    intro x hx
    exact ⟨(mul_pos (jacobiDensity_pos a x hx.1) hx.2).ne', hx.1.1, hx.1.2.le⟩
  exact (intervalIntegral.integral_pos_iff_support_of_nonneg_ae' hnonneg_ae hfi).mpr
    ⟨by norm_num, hUpos.trans_le (measure_mono hsub)⟩

theorem rawJacobiMomentGeneral_one_pos (a : ℝ) (ha : -1 < a) :
    0 < rawJacobiMomentGeneral a ha 1 :=
  rawJacobiMomentGeneral_pos a ha 1 one_ne_zero (by intro x _; simp)

noncomputable def jacobiMomentGeneral (a : ℝ) (ha : -1 < a) : ℝ[X] →ₗ[ℝ] ℝ :=
  (rawJacobiMomentGeneral a ha 1)⁻¹ • rawJacobiMomentGeneral a ha

theorem jacobiMomentGeneral_one (a : ℝ) (ha : -1 < a) : jacobiMomentGeneral a ha 1 = 1 := by
  simp [jacobiMomentGeneral, (rawJacobiMomentGeneral_one_pos a ha).ne']

theorem jacobiMomentGeneral_pos (a : ℝ) (ha : -1 < a) (p : ℝ[X]) (hp : p ≠ 0)
    (hnonneg : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, 0 ≤ p.eval x) :
    0 < jacobiMomentGeneral a ha p := by
  exact mul_pos (inv_pos.mpr (rawJacobiMomentGeneral_one_pos a ha))
    (rawJacobiMomentGeneral_pos a ha p hp hnonneg)

/-- Full exponent range of symmetric Gauss–Jacobi quadrature, including the
arcsine/Chebyshev exponent -1/2 needed for the base circle. -/
theorem exists_gaussJacobi_rule_general (a : ℝ) (ha : -1 < a) (n : ℕ) (hn : 0 < n) :
    ∃ node : Fin n → ℝ,
      Function.Injective node ∧
      (∀ i, node i ∈ Set.Ioo (-1 : ℝ) 1) ∧
      (∀ i, 0 < weight (jacobiMomentGeneral a ha) node i) ∧
      (∑ i, weight (jacobiMomentGeneral a ha) node i = 1) ∧
      (∀ p : ℝ[X], p.natDegree < 2 * n →
        jacobiMomentGeneral a ha p =
        ∑ i, weight (jacobiMomentGeneral a ha) node i * p.eval (node i)) :=
  exists_positive_gaussian_rule (jacobiMomentGeneral a ha) (-1) 1 (by norm_num)
    (jacobiMomentGeneral_pos a ha) (jacobiMomentGeneral_one a ha) n hn

end QIRadon.Gaussian
