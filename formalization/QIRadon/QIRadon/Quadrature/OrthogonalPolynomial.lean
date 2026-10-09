import Mathlib.LinearAlgebra.BilinearForm.Properties
import Mathlib.RingTheory.Polynomial.DegreeLT
import Mathlib.Algebra.Polynomial.Monic
import Mathlib.Tactic

/-!
# Construction of monic orthogonal polynomials

This uses finite-dimensional nondegeneracy of the moment bilinear form. Its
matrix in the monomial basis is the ordinary finite moment matrix. It is not a
readout solve or an assumed orthogonal-polynomial existence theorem.
-/

open Polynomial

namespace QIRadon.Gaussian

/-- Moment Gram form on polynomials of degree below n. -/
noncomputable def momentForm (L : ℝ[X] →ₗ[ℝ] ℝ) (n : ℕ) :
    LinearMap.BilinForm ℝ (Polynomial.degreeLT ℝ n) where
  toFun p :=
    { toFun := fun q ↦ L ((p : ℝ[X]) * (q : ℝ[X]))
      map_add' := by intro q r; simp [mul_add]
      map_smul' := by intro c q; simp }
  map_add' := by intro p q; ext r; simp [add_mul]
  map_smul' := by intro c p; ext q; simp

/-- Positivity on nonzero squares proves nondegeneracy of the moment matrix. -/
theorem momentForm_nondegenerate (L : ℝ[X] →ₗ[ℝ] ℝ) (n : ℕ)
    (hpos : ∀ p : ℝ[X], p ≠ 0 → 0 < L (p ^ 2)) :
    (momentForm L n).Nondegenerate := by
  constructor
  · intro p hzero
    apply Subtype.ext
    change (p : ℝ[X]) = 0
    by_contra hp
    have h := hpos p hp
    have hz := hzero p
    change L ((p : ℝ[X]) * (p : ℝ[X])) = 0 at hz
    rw [pow_two, hz] at h
    exact (lt_irrefl 0) h
  · intro p hzero
    apply Subtype.ext
    change (p : ℝ[X]) = 0
    by_contra hp
    have h := hpos p hp
    have hz := hzero p
    change L ((p : ℝ[X]) * (p : ℝ[X])) = 0 at hz
    rw [pow_two, hz] at h
    exact (lt_irrefl 0) h

/-- The moment functional of X^n against lower-degree polynomials. -/
noncomputable def topMoment (L : ℝ[X] →ₗ[ℝ] ℝ) (n : ℕ) :
    Module.Dual ℝ (Polynomial.degreeLT ℝ n) where
  toFun q := L (X ^ n * (q : ℝ[X]))
  map_add' := by intro p q; simp [mul_add]
  map_smul' := by intro c p; simp

/-- Every strictly positive moment functional has a monic degree-n polynomial
orthogonal to every lower-degree polynomial. The proof explicitly subtracts
the moment projection of X^n onto the lower-degree subspace. -/
theorem exists_monic_orthogonal (L : ℝ[X] →ₗ[ℝ] ℝ) (n : ℕ)
    (hpos : ∀ p : ℝ[X], p ≠ 0 → 0 < L (p ^ 2)) :
    ∃ p : ℝ[X], p.Monic ∧ p.natDegree = n ∧
      ∀ u : ℝ[X], u.degree < n → L (p * u) = 0 := by
  let v : Polynomial.degreeLT ℝ n :=
    ((momentForm L n).toDual (momentForm_nondegenerate L n hpos)).symm (topMoment L n)
  have hv : (v : ℝ[X]).degree < n := Polynomial.mem_degreeLT.mp v.property
  refine ⟨X ^ n - (v : ℝ[X]), Polynomial.monic_X_pow_sub hv, ?_, ?_⟩
  · apply Polynomial.natDegree_eq_of_degree_eq_some
    rw [Polynomial.degree_sub_eq_left_of_degree_lt (by simpa using hv)]
    simp
  · intro u hu
    have h := LinearMap.BilinForm.apply_toDual_symm_apply
      (B := momentForm L n) (hB := momentForm_nondegenerate L n hpos)
      (topMoment L n) (⟨u, Polynomial.mem_degreeLT.mpr hu⟩ : Polynomial.degreeLT ℝ n)
    change L ((v : ℝ[X]) * u) = L (X ^ n * u) at h
    rw [sub_mul, map_sub, h, sub_self]

/-- The monic orthogonal polynomial is unique. This later permits symmetry of
Jacobi nodes to be derived from symmetry of the measure, not assumed. -/
theorem monic_orthogonal_unique (L : ℝ[X] →ₗ[ℝ] ℝ) (n : ℕ)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → 0 < L (u ^ 2))
    (p q : ℝ[X]) (hp : p.Monic) (hq : q.Monic)
    (hpn : p.natDegree = n) (hqn : q.natDegree = n)
    (hpo : ∀ u : ℝ[X], u.degree < n → L (p * u) = 0)
    (hqo : ∀ u : ℝ[X], u.degree < n → L (q * u) = 0) : p = q := by
  have hdeq : p.degree = q.degree := by
    rw [Polynomial.degree_eq_natDegree hp.ne_zero, Polynomial.degree_eq_natDegree hq.ne_zero,
      hpn, hqn]
  have hd : (p - q).degree < n := by
    have h := Polynomial.degree_sub_lt_left hdeq hp.ne_zero (hp.trans hq.symm)
    simpa [Polynomial.degree_eq_natDegree hp.ne_zero, hpn] using h
  have hzero : L ((p - q) ^ 2) = 0 := by
    have heq : (p - q) ^ 2 = p * (p - q) - q * (p - q) := by ring
    rw [heq, map_sub, hpo _ hd, hqo _ hd, sub_self]
  apply sub_eq_zero.mp
  by_contra hne
  have h := hpos (p - q) hne
  rw [hzero] at h
  exact (lt_irrefl 0) h

end QIRadon.Gaussian
