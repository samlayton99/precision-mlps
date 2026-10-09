import QIRadon.Quadrature.PlaneRotation
import QIRadon.Quadrature.CasimirPolynomial
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Topology.Algebra.MvPolynomial

open MeasureTheory MvPolynomial
open scoped BigOperators

namespace QIRadon.Casimir

noncomputable section

set_option backward.isDefEq.respectTransparency false

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

theorem polynomial_eval_hasDerivAt (p : MvPolynomial ι ℝ)
    (x : ℝ → ι → ℝ) (v : ι → ℝ) (t : ℝ)
    (hx : ∀ i, HasDerivAt (fun t ↦ x t i) (v i) t) :
    HasDerivAt (fun t ↦ eval (x t) p)
      (∑ i, v i * eval (x t) (pderiv i p)) t := by
  induction p using MvPolynomial.induction_on with
  | C c => simpa using hasDerivAt_const t c
  | add p q hp hq =>
    convert hp.add hq using 1 <;> try rfl
    · funext s; simp
    · simp [Finset.sum_add_distrib, mul_add]
  | mul_X p j hp =>
    convert hp.mul (hx j) using 1 <;> try rfl
    · funext s; simp
    · simp only [pderiv_mul, map_add, map_mul, eval_X, mul_add,
        Finset.sum_add_distrib]
      simp [pderiv_X, Pi.single_apply, mul_comm, mul_left_comm, Finset.mul_sum]

theorem planeRotation_polynomial_hasDerivAt (i j : ι) (hij : i ≠ j)
    (p : MvPolynomial ι ℝ) (x : EuclideanSpace ℝ ι) (t : ℝ) :
    HasDerivAt (fun t ↦ eval (fun k ↦ Gaussian.planeRotate i j t x k) p)
      (eval (fun k ↦ Gaussian.planeRotate i j t x k) (generator i j p)) t := by
  convert polynomial_eval_hasDerivAt p
    (fun s k ↦ Gaussian.planeRotate i j s x k)
    (fun k ↦ if k = i then -Gaussian.planeRotate i j t x j
      else if k = j then Gaussian.planeRotate i j t x i else 0) t
    (fun k ↦ Gaussian.planeRotate_hasDerivAt_coordinate i j hij x k t) using 1
  simp only [generator, map_sub, map_mul, eval_X]
  simp only [ite_mul, zero_mul]
  simp [Finset.sum_ite, hij.symm, sub_eq_add_neg, neg_mul, Finset.sum_filter, add_comm]


open Gaussian

def polynomialOnSphere (n : ℕ) (p : MvPolynomial (Fin (n+1)) ℝ) :
    C(UnitSphere n, ℝ) where
  toFun θ := eval (fun i ↦ θ.val i) p
  continuous_toFun := p.continuous_eval.comp (by fun_prop)

theorem polynomialOnSphere_integrable (n : ℕ) (p : MvPolynomial (Fin (n+1)) ℝ) :
    Integrable (polynomialOnSphere n p) (normalizedSphere n) :=
  (polynomialOnSphere n p).continuous.integrable_of_hasCompactSupport
    (HasCompactSupport.of_compactSpace _)

/-- Differentiating actual rotation invariance shows that a spherical rotational
 derivative has zero integral. No integration-by-parts identity is assumed. -/
theorem integral_generator_zero (n : ℕ) (i j : Fin (n+1))
    (p : MvPolynomial (Fin (n+1)) ℝ) :
    (∫ θ, polynomialOnSphere n (generator i j p) θ ∂normalizedSphere n) = 0 := by
  by_cases hij : i = j
  · subst j
    simp [generator_self, polynomialOnSphere]
  let F : ℝ → UnitSphere n → ℝ := fun t θ ↦
    polynomialOnSphere n p (sphereRotation (planeRotation i j hij t) θ)
  let D : ℝ → UnitSphere n → ℝ := fun t θ ↦
    polynomialOnSphere n (generator i j p) (sphereRotation (planeRotation i j hij t) θ)
  have hF (t : ℝ) : Continuous (F t) :=
    (polynomialOnSphere n p).continuous.comp (sphereRotation _).continuous
  have hD (t : ℝ) : Continuous (D t) :=
    (polynomialOnSphere n (generator i j p)).continuous.comp (sphereRotation _).continuous
  have hd : ∀ θ t, HasDerivAt (fun s ↦ F s θ) (D t θ) t := by
    intro θ t
    exact planeRotation_polynomial_hasDerivAt i j hij p θ.val t
  have hbound : ∀ θ t, ‖D t θ‖ ≤ ‖polynomialOnSphere n (generator i j p)‖ := by
    intro θ t
    exact (polynomialOnSphere n (generator i j p)).norm_coe_le_norm _
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := normalizedSphere n) (F := F) (F' := D) (x₀ := (0 : ℝ))
    (s := Set.univ) (bound := fun _ ↦ ‖polynomialOnSphere n (generator i j p)‖)
    (Filter.univ_mem) (Filter.Eventually.of_forall fun t ↦ (hF t).aestronglyMeasurable)
    ((hF 0).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _))
    (hD 0).aestronglyMeasurable
    (Filter.Eventually.of_forall fun θ t _ ↦ hbound θ t)
    (integrable_const _) (Filter.Eventually.of_forall fun θ t _ ↦ hd θ t)
  have hinv : (fun t ↦ ∫ θ, F t θ ∂normalizedSphere n) =
      (fun _ : ℝ ↦ ∫ θ, polynomialOnSphere n p θ ∂normalizedSphere n) := by
    funext t
    exact (sphereRotation_measurePreserving (planeRotation i j hij t)).integral_comp
      (sphereRotation _).measurableEmbedding (polynomialOnSphere n p)
  have hz := h.2
  rw [hinv] at hz
  have heq := hz.unique (hasDerivAt_const 0 _)
  change (∫ θ, eval (fun k ↦ planeRotate i j 0 θ.val k) (generator i j p)
    ∂normalizedSphere n) = 0 at heq
  simp only [planeRotate_zero i j hij] at heq
  convert heq using 1
  rfl

omit [Fintype ι] [DecidableEq ι] in
theorem generator_mul (i j : ι) (p q : MvPolynomial ι ℝ) :
    generator i j (p*q) = generator i j p * q + p * generator i j q := by
  simp only [generator, pderiv_mul]
  ring

/-- The rotation generator is skew-adjoint for actual normalized surface measure. -/
theorem generator_skew_integral (n : ℕ) (i j : Fin (n+1))
    (p q : MvPolynomial (Fin (n+1)) ℝ) :
    (∫ θ, polynomialOnSphere n (generator i j p) θ * polynomialOnSphere n q θ
      ∂normalizedSphere n) =
    -(∫ θ, polynomialOnSphere n p θ * polynomialOnSphere n (generator i j q) θ
      ∂normalizedSphere n) := by
  have h := integral_generator_zero n i j (p*q)
  have hpq : Integrable (fun θ ↦ polynomialOnSphere n (generator i j p) θ *
      polynomialOnSphere n q θ) (normalizedSphere n) := by
    have hc : Continuous (fun θ ↦ polynomialOnSphere n (generator i j p) θ *
      polynomialOnSphere n q θ) := by fun_prop
    exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hqp : Integrable (fun θ ↦ polynomialOnSphere n p θ *
      polynomialOnSphere n (generator i j q) θ) (normalizedSphere n) := by
    have hc : Continuous (fun θ ↦ polynomialOnSphere n p θ *
      polynomialOnSphere n (generator i j q) θ) := by fun_prop
    exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  simp only [generator_mul, polynomialOnSphere, ContinuousMap.coe_mk, map_add, map_mul] at h
  change (∫ θ, polynomialOnSphere n (generator i j p) θ * polynomialOnSphere n q θ +
    polynomialOnSphere n p θ * polynomialOnSphere n (generator i j q) θ
    ∂normalizedSphere n) = 0 at h
  rw [integral_add hpq hqp] at h
  linarith

end
end QIRadon.Casimir
