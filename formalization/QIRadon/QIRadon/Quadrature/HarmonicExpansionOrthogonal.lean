import QIRadon.Quadrature.HarmonicExpansionDensity
import Mathlib.Analysis.InnerProductSpace.l2Space

open MeasureTheory
open scoped BigOperators RealInnerProductSpace

namespace QIRadon.Gaussian

noncomputable section

theorem polynomialRepresentative_parity (n k : ℕ) (v : PolynomialL2Space n k) (θ : UnitSphere n) :
    polynomialRepresentative n k v (sphereNeg n θ) = (-1 : ℝ) ^ k * polynomialRepresentative n k v θ := by
  obtain ⟨p, hp⟩ := ((polynomialL2Equiv n k).symm v).property
  have hrep : homogeneousRestriction n k p = polynomialRepresentative n k v := hp
  rw [← hrep]
  have h := homogeneous_eval_smul n k p (-1) θ.val
  change MvPolynomial.eval (fun i ↦ (-θ.val) i) p.val =
    (-1 : ℝ) ^ k * MvPolynomial.eval (fun i ↦ θ.val i) p.val
  simpa only [neg_one_smul] using h

theorem polynomial_inner_representative (n k l : ℕ)
    (v : PolynomialL2Space n k) (w : PolynomialL2Space n l) :
    inner ℝ v.val w.val = ∫ θ,
      polynomialRepresentative n l w θ * polynomialRepresentative n k v θ ∂normalizedSphere n := by
  have h := ContinuousMap.inner_toLp (normalizedSphere n)
    (polynomialRepresentative n k v) (polynomialRepresentative n l w)
  rw [toLp_polynomialRepresentative, toLp_polynomialRepresentative] at h
  simpa only [RCLike.conj_to_real] using h

theorem polynomial_inner_zero_of_odd (n k l : ℕ) (v : PolynomialL2Space n k)
    (w : PolynomialL2Space n l) (hodd : Odd (k + l)) : inner ℝ v.val w.val = 0 := by
  rw [polynomial_inner_representative]
  let f : UnitSphere n → ℝ := fun θ ↦
    polynomialRepresentative n l w θ * polynomialRepresentative n k v θ
  have hf : ∀ θ, f (sphereNeg n θ) = -f θ := by
    intro θ
    dsimp [f]
    rw [polynomialRepresentative_parity, polynomialRepresentative_parity]
    calc
      _ = (-1 : ℝ) ^ (k + l) *
          (polynomialRepresentative n l w θ * polynomialRepresentative n k v θ) := by
        rw [pow_add]
        ring
      _ = _ := by rw [hodd.neg_one_pow]; ring
  have hm := (sphereRotation_measurePreserving
    (LinearIsometryEquiv.neg ℝ : Rotation n)).integral_comp
    (sphereRotation (LinearIsometryEquiv.neg ℝ : Rotation n)).measurableEmbedding f
  change (∫ θ, f (sphereNeg n θ) ∂normalizedSphere n) = ∫ θ, f θ ∂normalizedSphere n at hm
  simp_rw [hf] at hm
  rw [integral_neg] at hm
  change (∫ θ, f θ ∂normalizedSphere n) = 0
  linarith

theorem polynomialL2Space_le_add_two (n k : ℕ) :
    PolynomialL2Space n k ≤ PolynomialL2Space n (k + 2) := by
  intro v hv
  exact (polynomialDegreeRaise n k ⟨v, hv⟩).property

theorem polynomialL2Space_le_even_add (n k j : ℕ) :
    PolynomialL2Space n k ≤ PolynomialL2Space n (k + 2 * j) := by
  induction j with
  | zero => simp
  | succ j ih =>
    have h := ih.trans (polynomialL2Space_le_add_two n (k + 2 * j))
    convert h using 1
    congr 1

theorem polynomialL2Space_le_of_even_sub (n k l : ℕ) (hkl : k ≤ l) (heven : Even (l - k)) :
    PolynomialL2Space n k ≤ PolynomialL2Space n l := by
  obtain ⟨j, hj⟩ := heven
  have hl : l = k + 2 * j := by omega
  rw [hl]
  exact polynomialL2Space_le_even_add n k j

theorem harmonic_orthogonal_lt (n k l : ℕ) (hkl : k < l)
    (v : HarmonicSpace n k) (w : HarmonicSpace n l) :
    inner ℝ (harmonicToLp n k v) (harmonicToLp n l w) = 0 := by
  by_cases hodd : Odd (k + l)
  · exact polynomial_inner_zero_of_odd n k l v.val w.val hodd
  · have heven : Even (k + l) := (Nat.even_or_odd (k + l)).resolve_right hodd
    have hmod := Nat.even_iff.mp heven
    have hl : 2 ≤ l := by omega
    obtain ⟨t, rfl⟩ : ∃ t, l = t + 2 := ⟨l - 2, by omega⟩
    have hkt : k ≤ t := by omega
    have hdiff : Even (t - k) := Nat.even_iff.mpr (by omega)
    let u : PolynomialL2Space n t :=
      ⟨v.val.val, (polynomialL2Space_le_of_even_sub n k t hkt hdiff) v.val.property⟩
    have hu : polynomialDegreeRaise n t u ∈ lowerDegreeSpace n t := ⟨u, rfl⟩
    have hw : w.val ∈ (lowerDegreeSpace n t)ᗮ := w.property
    have h := ((lowerDegreeSpace n t).mem_orthogonal w.val).mp hw
      (polynomialDegreeRaise n t u) hu
    exact h

/-- The actual orthogonal layers form an orthogonal family in actual sphere L2. -/
theorem harmonic_orthogonalFamily (n : ℕ) :
    OrthogonalFamily ℝ (fun k ↦ HarmonicSpace n k) (harmonicToLp n) := by
  intro k l hne v w
  rcases lt_or_gt_of_ne hne with hkl | hlk
  · exact harmonic_orthogonal_lt n k l hkl v w
  · exact inner_eq_zero_symm.mp (harmonic_orthogonal_lt n l k hlk w v)

set_option maxHeartbeats 1200000 in
set_option backward.isDefEq.respectTransparency false in
/-- The actual Hilbert direct-sum decomposition: neither orthogonality nor total
span is a hypothesis here. -/
theorem harmonic_isHilbertSum (n : ℕ) :
    IsHilbertSum ℝ (fun k ↦ HarmonicSpace n k) (harmonicToLp n) := by
  exact @IsHilbertSum.mk ℕ ℝ _ (Lp ℝ 2 (normalizedSphere n)) _ _
    (fun k ↦ HarmonicSpace n k) (fun k ↦ inferInstance) (fun k ↦ inferInstance) _
    (harmonicToLp n) (fun k ↦ inferInstance) (harmonic_orthogonalFamily n) (by
      change ⊤ ≤ (harmonicSpan n).topologicalClosure
      rw [harmonicSpan_dense])

end
end QIRadon.Gaussian
