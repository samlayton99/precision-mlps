import QIRadon.Quadrature.HomogeneousRotation

open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section

abbrev BoundedExponent (n k : ℕ) := Fin (n + 1) → Fin (k + 1)

def boundedCoefficientMap (n k : ℕ) :
    HomogeneousPolynomials n k →ₗ[ℝ] (BoundedExponent n k → ℝ) where
  toFun p e := MvPolynomial.coeff
    (Finsupp.equivFunOnFinite.symm (fun i ↦ (e i).val)) p.val
  map_add' p q := by ext e; simp
  map_smul' c p := by ext e; simp

theorem homogeneous_coefficient_index_le (n k : ℕ) (p : HomogeneousPolynomials n k)
    (s : Fin (n + 1) →₀ ℕ) (hs : MvPolynomial.coeff s p.val ≠ 0) (i : Fin (n + 1)) :
    s i ≤ k := by
  have hdeg : s.degree = k := by
    simpa only [Finsupp.degree_apply] using
      (p.property.degree_eq_sum_deg_support (Finsupp.mem_support_iff.mpr hs)).symm
  exact (Finsupp.le_degree i s).trans_eq hdeg

theorem boundedCoefficientMap_injective (n k : ℕ) :
    Function.Injective (boundedCoefficientMap n k) := by
  intro p q hpq
  apply Subtype.ext
  ext s
  by_cases hs : ∀ i, s i ≤ k
  · let e : BoundedExponent n k := fun i ↦ ⟨s i, Nat.lt_succ_of_le (hs i)⟩
    have he : Finsupp.equivFunOnFinite.symm (fun i ↦ (e i).val) = s := by
      ext i
      simp [e]
    have h := congrFun hpq e
    change MvPolynomial.coeff _ p.val = MvPolynomial.coeff _ q.val at h
    rwa [he] at h
  · push Not at hs
    obtain ⟨i, hi⟩ := hs
    have hpc : MvPolynomial.coeff s p.val = 0 := by
      by_contra h
      exact (not_le_of_gt hi) (homogeneous_coefficient_index_le n k p s h i)
    have hqc : MvPolynomial.coeff s q.val = 0 := by
      by_contra h
      exact (not_le_of_gt hi) (homogeneous_coefficient_index_le n k q s h i)
    rw [hpc, hqc]

/-- A deliberately loose but explicit polynomial dimension bound. It preserves
the analytic exponential rate; it is not substituted for the sharp harmonic
dimension in an algebraic Sobolev-rate claim. -/
theorem homogeneous_finrank_le (n k : ℕ) :
    Module.finrank ℝ (HomogeneousPolynomials n k) ≤ (k + 1) ^ (n + 1) := by
  have h := LinearMap.finrank_le_finrank_of_injective (boundedCoefficientMap_injective n k)
  simpa [BoundedExponent] using h

theorem polynomialL2_finrank_le (n k : ℕ) :
    Module.finrank ℝ (PolynomialL2Space n k) ≤ (k + 1) ^ (n + 1) := by
  exact (polynomialL2Map n k).finrank_range_le.trans
    ((homogeneousRestriction n k).finrank_range_le.trans (homogeneous_finrank_le n k))

/-- Pointwise L2 evaluation bound with the actual dimension of the concrete
homogeneous-polynomial space, obtained from the verified rotation action. -/
theorem polynomialEvaluation_bound_dimension (n k : ℕ) (θ : UnitSphere n)
    (v : PolynomialL2Space n k) :
    |polynomialEvaluation n k θ v| ≤
      Real.sqrt (Module.finrank ℝ (PolynomialL2Space n k)) * ‖v‖ := by
  have h := QIRadon.Addition.evaluation_bound
    (stdOrthonormalBasis ℝ (PolynomialL2Space n k))
    (polynomialEvaluation n k) (normalizedSphere n)
    (fun w ↦ ⟨polynomialEvaluation_square_integrable n k w, polynomialEvaluation_L2 n k w⟩)
    (polynomialEvaluation_covariant n k) θ v
  simpa only [Fintype.card_fin] using h

/-- An explicit dimension-polynomial bound usable for analytic series tails. -/
theorem polynomialEvaluation_bound (n k : ℕ) (θ : UnitSphere n)
    (v : PolynomialL2Space n k) :
    |polynomialEvaluation n k θ v| ≤
      Real.sqrt (((k + 1) ^ (n + 1) : ℕ) : ℝ) * ‖v‖ := by
  apply (polynomialEvaluation_bound_dimension n k θ v).trans
  apply mul_le_mul_of_nonneg_right _ (norm_nonneg v)
  exact Real.sqrt_le_sqrt (by exact_mod_cast polynomialL2_finrank_le n k)

end
end QIRadon.Gaussian
