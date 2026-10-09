import QIRadon.Quadrature.HarmonicRotation
import Mathlib.Topology.ContinuousMap.StoneWeierstrass
import Mathlib.MeasureTheory.Function.ContinuousMapDense

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section

def sphereCoordinate (n : ℕ) (i : Fin (n + 1)) : C(UnitSphere n, ℝ) :=
  ⟨fun θ ↦ θ.val i, by fun_prop⟩

def polynomialOnSphere (n : ℕ) : MvPolynomial (Fin (n + 1)) ℝ →ₐ[ℝ] C(UnitSphere n, ℝ) :=
  MvPolynomial.aeval (sphereCoordinate n)

theorem polynomialOnSphere_eval (n : ℕ) (p : MvPolynomial (Fin (n + 1)) ℝ)
    (θ : UnitSphere n) :
    polynomialOnSphere n p θ = MvPolynomial.eval (fun i ↦ θ.val i) p := by
  induction p using MvPolynomial.induction_on with
  | C c => simp [polynomialOnSphere]
  | add p q hp hq => simp only [map_add, ContinuousMap.add_apply, hp, hq]
  | mul_X p i hp =>
    simp only [map_mul, ContinuousMap.mul_apply, polynomialOnSphere,
      MvPolynomial.aeval_X, MvPolynomial.eval_X] at hp ⊢
    rw [hp]
    rfl

theorem polynomialOnSphere_separates (n : ℕ) : (polynomialOnSphere n).range.SeparatesPoints := by
  intro x y hxy
  have hi : ∃ i, x.val i ≠ y.val i := by
    by_contra h
    apply hxy
    apply Subtype.ext
    ext i
    by_contra hne
    exact h ⟨i, hne⟩
  obtain ⟨i, hi⟩ := hi
  refine ⟨sphereCoordinate n i, ?_, hi⟩
  refine ⟨sphereCoordinate n i, ?_, rfl⟩
  exact ⟨MvPolynomial.X i, by simp [polynomialOnSphere]⟩

/-- Actual density of polynomial restrictions in the continuous functions on the
sphere, by Stone–Weierstrass; point separation is proved from coordinates. -/
theorem polynomialOnSphere_denseRange (n : ℕ) : DenseRange (polynomialOnSphere n) := by
  have h := ContinuousMap.subalgebra_topologicalClosure_eq_top_of_separatesPoints
    (polynomialOnSphere n).range (polynomialOnSphere_separates n)
  rw [denseRange_iff_closure_range]
  change closure (↑(polynomialOnSphere n).range : Set C(UnitSphere n, ℝ)) = Set.univ
  rw [← Subalgebra.topologicalClosure_coe, h]
  rfl

def polynomialToLp (n : ℕ) :
    MvPolynomial (Fin (n + 1)) ℝ →ₗ[ℝ] Lp ℝ 2 (normalizedSphere n) :=
  (ContinuousMap.toLp 2 (normalizedSphere n) ℝ).toLinearMap.comp (polynomialOnSphere n).toLinearMap

/-- Actual L2 density, combining uniform polynomial density with density of
continuous functions under the actual finite sphere measure. -/
theorem polynomialToLp_denseRange (n : ℕ) : DenseRange (polynomialToLp n) := by
  have h := (ContinuousMap.toLp_denseRange ℝ (normalizedSphere n) ℝ (p := 2) (by norm_num)).comp
    (polynomialOnSphere_denseRange n) (ContinuousMap.toLp 2 (normalizedSphere n) ℝ).continuous
  exact h

theorem polynomialToLp_homogeneous (n k : ℕ) (p : HomogeneousPolynomials n k) :
    polynomialToLp n p.val ∈ PolynomialL2Space n k := by
  refine ⟨⟨homogeneousRestriction n k p, ⟨p, rfl⟩⟩, ?_⟩
  change ContinuousMap.toLp 2 (normalizedSphere n) ℝ (homogeneousRestriction n k p) =
    ContinuousMap.toLp 2 (normalizedSphere n) ℝ (polynomialOnSphere n p.val)
  congr 1
  ext θ
  exact (polynomialOnSphere_eval n p.val θ).symm

def harmonicToLp (n k : ℕ) : HarmonicSpace n k →ₗᵢ[ℝ] Lp ℝ 2 (normalizedSphere n) :=
  (PolynomialL2Space n k).subtypeₗᵢ.comp (harmonicSubspace n k).subtypeₗᵢ

def harmonicSpan (n : ℕ) : Submodule ℝ (Lp ℝ 2 (normalizedSphere n)) :=
  ⨆ k, (harmonicToLp n k).toLinearMap.range

theorem harmonic_mem_span (n k : ℕ) (v : HarmonicSpace n k) :
    harmonicToLp n k v ∈ harmonicSpan n :=
  (le_iSup (fun k ↦ (harmonicToLp n k).toLinearMap.range) k) ⟨v, rfl⟩

/-- All homogeneous polynomial restrictions lie in the span of the orthogonal
layers; the proof uses the actual degree-two inclusion and orthogonal split. -/
theorem homogeneous_mem_harmonicSpan (n k : ℕ) (v : PolynomialL2Space n k) :
    v.val ∈ harmonicSpan n := by
  induction k using Nat.twoStepInduction with
  | zero => exact harmonic_mem_span n 0 ⟨v, by trivial⟩
  | one => exact harmonic_mem_span n 1 ⟨v, by trivial⟩
  | more k ih _ =>
    obtain ⟨u, hu, z, hz, hv⟩ := (lowerDegreeSpace n k).exists_add_mem_mem_orthogonal v
    obtain ⟨w, hw⟩ := hu
    have hu' : u.val ∈ harmonicSpan n := by
      rw [← hw]
      exact ih w
    have hz' : z.val ∈ harmonicSpan n := harmonic_mem_span n (k + 2) ⟨z, hz⟩
    rw [hv]
    exact (harmonicSpan n).add_mem hu' hz'

theorem polynomial_mem_harmonicSpan (n : ℕ) (p : MvPolynomial (Fin (n + 1)) ℝ) :
    polynomialToLp n p ∈ harmonicSpan n := by
  rw [← MvPolynomial.sum_homogeneousComponent p, map_sum]
  apply Submodule.sum_mem
  intro k _
  let pk : HomogeneousPolynomials n k :=
    ⟨MvPolynomial.homogeneousComponent k p, MvPolynomial.homogeneousComponent_isHomogeneous k p⟩
  exact homogeneous_mem_harmonicSpan n k
    ⟨polynomialToLp n pk.val, polynomialToLp_homogeneous n k pk⟩

/-- Density of the actual harmonic-layer span in actual sphere L2. Completeness
of the expansion is a consequence of this and the separately proved family
orthogonality, not an added representation hypothesis. -/
theorem harmonicSpan_dense (n : ℕ) : (harmonicSpan n).topologicalClosure = ⊤ := by
  apply Submodule.dense_iff_topologicalClosure_eq_top.mp
  apply (polynomialToLp_denseRange n).mono
  rintro y ⟨p, rfl⟩
  exact polynomial_mem_harmonicSpan n p

end
end QIRadon.Gaussian
