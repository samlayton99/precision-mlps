import QIRadon.Quadrature.CasimirHarmonic
import QIRadon.Quadrature.HarmonicExpansion

open MeasureTheory
open scoped RealInnerProductSpace

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 100000

open Gaussian

def harmonicPolynomial (n k : ℕ) (v : HarmonicSpace n k) : HomogeneousPolynomials n k :=
  (homogeneousL2Iso n k).symm v.val

theorem harmonicPolynomial_representative (n k : ℕ) (v : HarmonicSpace n k) :
    polynomialOnSphere n (harmonicPolynomial n k v).val = continuousHarmonicComponent n k v := by
  have h := representative_homogeneousL2Iso n k (harmonicPolynomial n k v)
  rw [show homogeneousL2Iso n k (harmonicPolynomial n k v) = v.val from
    (homogeneousL2Iso n k).apply_symm_apply v.val] at h
  exact h.symm

theorem harmonicPolynomial_eigen (n k : ℕ) (v : HarmonicSpace n k) :
    orderedCasimir (harmonicPolynomial n k v).val =
      -(2*(k:ℝ)*((k:ℝ)+(n+1:ℕ)-2)) • (harmonicPolynomial n k v).val := by
  apply orderedCasimir_harmonicLayer
  simpa only [harmonicPolynomial, LinearEquiv.apply_symm_apply] using v.property

theorem harmonicCoordinates_inner (n k : ℕ) (f : Lp ℝ 2 (normalizedSphere n))
    (v : HarmonicSpace n k) :
    inner ℝ f (harmonicToLp n k v) = inner ℝ (harmonicCoordinates n f k) v := by
  have hs : (harmonicCoordinates n).symm (lp.single 2 k v) = harmonicToLp n k v :=
    (harmonic_isHilbertSum n).linearIsometryEquiv_symm_apply_single v
  have hx : (harmonicCoordinates n).symm (harmonicCoordinates n f) = f :=
    (harmonicCoordinates n).symm_apply_apply f
  have hsingle : inner ℝ (harmonicCoordinates n f) (lp.single 2 k v) =
      inner ℝ (harmonicCoordinates n f k) v := lp.inner_single_right k v (harmonicCoordinates n f)
  have hmap := @LinearIsometryEquiv.inner_map_map ℝ
    (lp (fun k ↦ HarmonicSpace n k) 2) _ _ _ (Lp ℝ 2 (normalizedSphere n)) _ _
    (harmonicCoordinates n).symm (harmonicCoordinates n f) (lp.single 2 k v)
  exact (congrArg₂ (inner ℝ) hx hs).symm.trans (hmap.trans hsingle)

theorem continuous_harmonic_pairing (n k : ℕ) (F : C(UnitSphere n, ℝ))
    (v : HarmonicSpace n k) :
    (∫ θ, F θ * harmonicEvaluation n k θ v ∂normalizedSphere n) =
      inner ℝ (continuousHarmonicCoordinates n F k) v := by
  have h := ContinuousMap.inner_toLp (normalizedSphere n) F (continuousHarmonicComponent n k v)
  rw [toLp_continuousHarmonicComponent, harmonicCoordinates_inner n k] at h
  simp only [RCLike.conj_to_real, mul_comm] at h
  convert h.symm using 1 <;> rfl

theorem continuous_harmonic_pairing_self (n k : ℕ) (F : C(UnitSphere n, ℝ)) :
    (∫ θ, F θ * harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k)
      ∂normalizedSphere n) = ‖continuousHarmonicCoordinates n F k‖^2 := by
  rw [continuous_harmonic_pairing, real_inner_self_eq_norm_sq]

theorem continuous_toLp_norm_le (n : ℕ) (F : C(UnitSphere n, ℝ)) :
    ‖ContinuousMap.toLp 2 (normalizedSphere n) ℝ F‖ ≤ ‖F‖ := by
  have h := ContinuousMap.toLp_norm_le (p := 2) (𝕜 := ℝ) (E := ℝ) (normalizedSphere n)
  have h1 : ‖ContinuousMap.toLp (E := ℝ) 2 (normalizedSphere n) ℝ‖ ≤ 1 := by simpa [measureUnivNNReal] using h
  simpa only [one_mul] using
    (ContinuousMap.toLp (E := ℝ) 2 (normalizedSphere n) ℝ).le_of_opNorm_le h1 F

theorem continuous_harmonic_norm_le (n k : ℕ) (F : C(UnitSphere n, ℝ)) :
    ‖continuousHarmonicCoordinates n F k‖ ≤ ‖F‖ := by
  have h := lp.norm_apply_le_norm (by norm_num : (2 : ENNReal) ≠ 0)
    (continuousHarmonicCoordinates n F) k
  have hn : ‖continuousHarmonicCoordinates n F‖ =
      ‖ContinuousMap.toLp 2 (normalizedSphere n) ℝ F‖ :=
    (harmonicCoordinates n).norm_map _
  exact (h.trans_eq hn).trans (continuous_toLp_norm_le n F)

theorem continuous_harmonic_pairing_bound (n k : ℕ) (F : C(UnitSphere n, ℝ))
    (v : HarmonicSpace n k) :
    |∫ θ, F θ * harmonicEvaluation n k θ v ∂normalizedSphere n| ≤ ‖F‖ * ‖v‖ := by
  rw [continuous_harmonic_pairing]
  exact (abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_right (continuous_harmonic_norm_le n k F) (norm_nonneg _))

end
end QIRadon.Casimir
