import QIRadon.Quadrature.CasimirRotation
import QIRadon.Quadrature.HarmonicExpansionOrthogonal

open MeasureTheory MvPolynomial
open scoped BigOperators RealInnerProductSpace

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 800000

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [Fintype ι] [DecidableEq ι] in
theorem generator_homogeneous (i j : ι) {p : MvPolynomial ι ℝ} {k : ℕ}
    (hp : p.IsHomogeneous k) : (generator i j p).IsHomogeneous k := by
  cases k with
  | zero =>
    rw [← totalDegree_zero_iff_isHomogeneous, totalDegree_eq_zero_iff_eq_C] at hp
    rw [hp]
    simpa [generator] using isHomogeneous_zero (R := ℝ) (σ := ι) 0
  | succ k =>
    have hi := (isHomogeneous_X ℝ i).mul (hp.pderiv (i := j))
    have hj := (isHomogeneous_X ℝ j).mul (hp.pderiv (i := i))
    simpa [generator, Nat.add_comm] using hi.sub hj

omit [DecidableEq ι] in
theorem orderedCasimir_homogeneous_degree {p : MvPolynomial ι ℝ} {k : ℕ}
    (hp : p.IsHomogeneous k) : (orderedCasimir p).IsHomogeneous k := by
  change orderedCasimir p ∈ homogeneousSubmodule ι ℝ k
  apply Submodule.sum_mem
  intro i _
  apply Submodule.sum_mem
  intro j _
  exact generator_homogeneous i j (generator_homogeneous i j hp)

omit [DecidableEq ι] in
theorem laplace_homogeneous {p : MvPolynomial ι ℝ} {k : ℕ}
    (hp : p.IsHomogeneous (k+2)) : (laplace p).IsHomogeneous k := by
  change laplace p ∈ homogeneousSubmodule ι ℝ k
  apply Submodule.sum_mem
  intro i _
  simpa using (hp.pderiv (i := i)).pderiv (i := i)

open Gaussian

theorem polynomialOnSphere_product_integrable (n : ℕ)
    (p q : MvPolynomial (Fin (n+1)) ℝ) :
    Integrable (fun θ ↦ polynomialOnSphere n p θ * polynomialOnSphere n q θ)
      (normalizedSphere n) := by
  have hc : Continuous (fun θ ↦ polynomialOnSphere n p θ * polynomialOnSphere n q θ) := by
    fun_prop
  exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)

theorem orderedCasimir_symmetric_integral (n : ℕ)
    (p q : MvPolynomial (Fin (n+1)) ℝ) :
    (∫ θ, polynomialOnSphere n (orderedCasimir p) θ * polynomialOnSphere n q θ
      ∂normalizedSphere n) =
    (∫ θ, polynomialOnSphere n p θ * polynomialOnSphere n (orderedCasimir q) θ
      ∂normalizedSphere n) := by
  have hl (θ : UnitSphere n) : polynomialOnSphere n (orderedCasimir p) θ *
      polynomialOnSphere n q θ = ∑ i, ∑ j,
      polynomialOnSphere n (generator i j (generator i j p)) θ * polynomialOnSphere n q θ := by
    simp [polynomialOnSphere, orderedCasimir, Finset.sum_mul]
  have hr (θ : UnitSphere n) : polynomialOnSphere n p θ *
      polynomialOnSphere n (orderedCasimir q) θ = ∑ i, ∑ j,
      polynomialOnSphere n p θ * polynomialOnSphere n (generator i j (generator i j q)) θ := by
    simp [polynomialOnSphere, orderedCasimir, Finset.mul_sum]
  simp_rw [hl, hr]
  rw [integral_finsetSum, integral_finsetSum]
  · apply Finset.sum_congr rfl
    intro i _
    rw [integral_finsetSum, integral_finsetSum]
    · apply Finset.sum_congr rfl
      intro j _
      rw [generator_skew_integral n i j (generator i j p) q,
        generator_skew_integral n i j p (generator i j q), neg_neg]
    · intro j _; exact polynomialOnSphere_product_integrable n _ _
    · intro j _; exact polynomialOnSphere_product_integrable n _ _
  · intro i _
    exact integrable_finsetSum _ (fun j _ ↦ polynomialOnSphere_product_integrable n _ _)
  · intro i _
    exact integrable_finsetSum _ (fun j _ ↦ polynomialOnSphere_product_integrable n _ _)

theorem orderedCasimir_onSphere (n k : ℕ)
    (p : MvPolynomial (Fin (n+1)) ℝ) (hp : p.IsHomogeneous k) (θ : UnitSphere n) :
    polynomialOnSphere n (orderedCasimir p) θ =
      2 * polynomialOnSphere n (laplace p) θ -
        (2*(k:ℝ)*((k:ℝ)+(n+1:ℕ)-2)) * polynomialOnSphere n p θ := by
  rw [orderedCasimir_homogeneous p k hp]
  have hr : eval (fun i ↦ θ.val i) (radiusSquared : MvPolynomial (Fin (n+1)) ℝ) = 1 := by
    change eval (fun i ↦ θ.val i) (squaredRadiusPolynomial n) = 1
    rw [squaredRadiusPolynomial_eval]
    have hθ : ‖θ.val‖ = 1 := by simp
    rw [hθ, one_pow]
  simp [polynomialOnSphere, hr]


theorem orderedCasimir_pairing (n k : ℕ)
    (p q : MvPolynomial (Fin (n+1)) ℝ) (hp : p.IsHomogeneous k) :
    (∫ θ, polynomialOnSphere n (orderedCasimir p) θ * polynomialOnSphere n q θ
      ∂normalizedSphere n) =
    2 * (∫ θ, polynomialOnSphere n (laplace p) θ * polynomialOnSphere n q θ
      ∂normalizedSphere n) -
    (2*(k:ℝ)*((k:ℝ)+(n+1:ℕ)-2)) *
      (∫ θ, polynomialOnSphere n p θ * polynomialOnSphere n q θ ∂normalizedSphere n) := by
  simp_rw [orderedCasimir_onSphere n k p hp, sub_mul, mul_assoc]
  rw [integral_sub]
  · simp only [integral_const_mul]
  · exact (polynomialOnSphere_product_integrable n _ _).const_mul 2
  · exact (((polynomialOnSphere_product_integrable n _ _).const_mul _).const_mul _).const_mul _

/-- A homogeneous polynomial orthogonal on the sphere to the degree-(k−2)
 space is genuinely ambient harmonic. This identifies the intrinsic layers
 rather than adding harmonicity as a hypothesis. -/
theorem laplace_eq_zero_of_lower_orthogonal (n k : ℕ)
    (p : MvPolynomial (Fin (n+1)) ℝ) (hp : p.IsHomogeneous (k+2))
    (horth : ∀ q : MvPolynomial (Fin (n+1)) ℝ, q.IsHomogeneous k →
      (∫ θ, polynomialOnSphere n p θ * polynomialOnSphere n q θ ∂normalizedSphere n) = 0) :
    laplace p = 0 := by
  have hq : (laplace p).IsHomogeneous k := laplace_homogeneous hp
  have hqorth := horth (laplace p) hq
  have hcorth := horth (orderedCasimir (laplace p)) (orderedCasimir_homogeneous_degree hq)
  have hsym := orderedCasimir_symmetric_integral n p (laplace p)
  rw [hcorth, orderedCasimir_pairing n (k+2) p (laplace p) hp, hqorth] at hsym
  have hsquare : (∫ θ, (polynomialOnSphere n (laplace p) θ)^2 ∂normalizedSphere n) = 0 := by
    simp only [pow_two]
    linarith
  have hint : Integrable (fun θ ↦ (polynomialOnSphere n (laplace p) θ)^2)
      (normalizedSphere n) := by
    simpa only [pow_two] using polynomialOnSphere_product_integrable n (laplace p) (laplace p)
  have hae := (integral_eq_zero_iff_of_nonneg (fun θ ↦ sq_nonneg _) hint).mp hsquare
  have hae' : (fun θ ↦ polynomialOnSphere n (laplace p) θ) =ᵐ[normalizedSphere n] 0 := by
    filter_upwards [hae] with θ hθ
    exact sq_eq_zero_iff.mp hθ
  have heq := MeasureTheory.Measure.eq_of_ae_eq hae' (polynomialOnSphere n (laplace p)).continuous continuous_const
  have hrestrict : homogeneousRestriction n k ⟨laplace p, hq⟩ = 0 := by
    ext θ
    exact congrFun heq θ
  have hpzero := homogeneousRestriction_injective n k
    (show homogeneousRestriction n k ⟨laplace p, hq⟩ = homogeneousRestriction n k 0 by
      simpa using hrestrict)
  exact congrArg Subtype.val hpzero


def homogeneousL2Iso (n k : ℕ) : HomogeneousPolynomials n k ≃ₗ[ℝ] PolynomialL2Space n k :=
  (LinearEquiv.ofInjective (homogeneousRestriction n k)
    (homogeneousRestriction_injective n k)).trans (polynomialL2Equiv n k)

theorem representative_homogeneousL2Iso (n k : ℕ) (p : HomogeneousPolynomials n k) :
    polynomialRepresentative n k (homogeneousL2Iso n k p) = homogeneousRestriction n k p := by
  apply ContinuousMap.toLp_injective (normalizedSphere n) (p := 2) (𝕜 := ℝ)
  rw [toLp_polynomialRepresentative]
  rfl

omit [DecidableEq ι] in
theorem laplace_homogeneous_low (p : MvPolynomial ι ℝ) {k : ℕ}
    (hp : p.IsHomogeneous k) (hk : k ≤ 1) : laplace p = 0 := by
  cases k with
  | zero =>
    rw [← totalDegree_zero_iff_isHomogeneous, totalDegree_eq_zero_iff_eq_C] at hp
    rw [hp]
    simp [laplace]
  | succ k =>
    have hk0 : k = 0 := by omega
    subst k
    unfold laplace
    apply Finset.sum_eq_zero
    intro i _
    have h := hp.pderiv (i := i)
    simp only [Nat.sub_self] at h
    rw [← totalDegree_zero_iff_isHomogeneous, totalDegree_eq_zero_iff_eq_C] at h
    rw [h, pderiv_C]

/-- The actual orthogonal layer, viewed through its unique homogeneous polynomial,
 consists of ambient harmonic polynomials. -/
theorem laplace_harmonicLayer (n k : ℕ) (p : HomogeneousPolynomials n k)
    (hmem : homogeneousL2Iso n k p ∈ harmonicSubspace n k) : laplace p.val = 0 := by
  by_cases hk : k ≤ 1
  · exact laplace_homogeneous_low p.val p.property hk
  · obtain ⟨l, rfl⟩ : ∃ l, k = l+2 := ⟨k-2, by omega⟩
    apply laplace_eq_zero_of_lower_orthogonal n l p.val p.property
    intro q hq
    let w : PolynomialL2Space n l := homogeneousL2Iso n l ⟨q, hq⟩
    have hw : polynomialDegreeRaise n l w ∈ lowerDegreeSpace n l := ⟨w, rfl⟩
    have h := ((lowerDegreeSpace n l).mem_orthogonal
      (homogeneousL2Iso n (l+2) p)).mp hmem (polynomialDegreeRaise n l w) hw
    change inner ℝ w.val (homogeneousL2Iso n (l+2) p).val = 0 at h
    rw [polynomial_inner_representative] at h
    dsimp only [w] at h
    rw [representative_homogeneousL2Iso, representative_homogeneousL2Iso] at h
    exact h

/-- Exact Casimir eigenvalue for the already constructed canonical harmonic layer. -/
theorem orderedCasimir_harmonicLayer (n k : ℕ) (p : HomogeneousPolynomials n k)
    (hmem : homogeneousL2Iso n k p ∈ harmonicSubspace n k) :
    orderedCasimir p.val = -(2*(k:ℝ)*((k:ℝ)+(n+1:ℕ)-2)) • p.val := by
  rw [orderedCasimir_homogeneous p.val k p.property, laplace_harmonicLayer n k p hmem]
  simp

end
end QIRadon.Casimir
