import QIRadon.Quadrature.HarmonicExpansionOrthogonal
import QIRadon.Quadrature.HarmonicQuadrature

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable section

set_option maxHeartbeats 1200000
set_option backward.isDefEq.respectTransparency false

/-- Canonical coordinates in the proved Hilbert direct sum of actual sphere
polynomial layers. This replaces a supplied representation witness. -/
def harmonicCoordinates (n : ℕ) :
    Lp ℝ 2 (normalizedSphere n) ≃ₗᵢ[ℝ] lp (fun k ↦ HarmonicSpace n k) 2 :=
  (harmonic_isHilbertSum n).linearIsometryEquiv

theorem harmonicCoordinates_hasSum (n : ℕ) (f : Lp ℝ 2 (normalizedSphere n)) :
    HasSum (fun k ↦ harmonicToLp n k (harmonicCoordinates n f k)) f := by
  have h := (harmonic_isHilbertSum n).hasSum_linearIsometryEquiv_symm (harmonicCoordinates n f)
  rw [show (harmonic_isHilbertSum n).linearIsometryEquiv = harmonicCoordinates n from rfl,
    LinearIsometryEquiv.symm_apply_apply] at h
  exact h

def continuousHarmonicComponent (n k : ℕ) (v : HarmonicSpace n k) : C(UnitSphere n, ℝ) :=
  polynomialRepresentative n k v.val

theorem toLp_continuousHarmonicComponent (n k : ℕ) (v : HarmonicSpace n k) :
    ContinuousMap.toLp 2 (normalizedSphere n) ℝ (continuousHarmonicComponent n k v) =
      harmonicToLp n k v := toLp_polynomialRepresentative n k v.val

def continuousHarmonicCoordinates (n : ℕ) (F : C(UnitSphere n, ℝ)) :
    lp (fun k ↦ HarmonicSpace n k) 2 :=
  harmonicCoordinates n (ContinuousMap.toLp 2 (normalizedSphere n) ℝ F)

/-- A summable uniform majorant for the canonical components upgrades the actual
L2 expansion to equality of continuous functions and hence pointwise equality.
The analytic/Sobolev corollaries must derive this majorant from their energy. -/
theorem canonical_harmonic_pointwise_of_majorant (n : ℕ) (F : C(UnitSphere n, ℝ))
    (b : ℕ → ℝ) (hb : ∀ k, 0 ≤ b k) (hs : Summable b)
    (hbound : ∀ k θ,
      |harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k)| ≤ b k)
    (θ : UnitSphere n) :
    F θ = ∑' k, harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k) := by
  let c : ℕ → C(UnitSphere n, ℝ) := fun k ↦
    continuousHarmonicComponent n k (continuousHarmonicCoordinates n F k)
  have hc_bound : ∀ k, ‖c k‖ ≤ b k := by
    intro k
    apply (ContinuousMap.norm_le _ (hb k)).mpr
    intro x
    exact hbound k x
  have hcnorm : Summable (fun k ↦ ‖c k‖) :=
    Summable.of_nonneg_of_le (fun k ↦ norm_nonneg _) hc_bound hs
  have hc : Summable c := hcnorm.of_norm
  have hm := hc.hasSum.map (ContinuousMap.toLp 2 (normalizedSphere n) ℝ)
    (ContinuousMap.toLp 2 (normalizedSphere n) ℝ).continuous
  have hm' : HasSum (fun k ↦ harmonicToLp n k (continuousHarmonicCoordinates n F k))
      (ContinuousMap.toLp 2 (normalizedSphere n) ℝ (∑' k, c k)) := by
    simpa only [Function.comp_def, c, toLp_continuousHarmonicComponent] using hm
  have hL2 := harmonicCoordinates_hasSum n (ContinuousMap.toLp 2 (normalizedSphere n) ℝ F)
  have hcontinuous : (∑' k, c k) = F :=
    ContinuousMap.toLp_injective (normalizedSphere n) (hm'.unique hL2)
  have heval := hc.hasSum.map (ContinuousMap.evalCLM ℝ θ) (ContinuousMap.evalCLM ℝ θ).continuous
  have heval' : HasSum (fun k ↦ harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k)) (F θ) := by
    rw [hcontinuous] at heval
    exact heval
  exact heval'.tsum_eq.symm

private def constantHarmonic (n : ℕ) : HarmonicSpace n 0 :=
  ⟨(polynomialL2Equiv n 0) ⟨1, ⟨⟨1, MvPolynomial.isHomogeneous_one _ ℝ⟩, by ext θ; simp [homogeneousRestriction]⟩⟩,
    by trivial⟩

private theorem integral_eq_constant_inner (n : ℕ) (F : C(UnitSphere n, ℝ)) :
    (∫ θ, F θ ∂normalizedSphere n) =
      inner ℝ (harmonicToLp n 0 (constantHarmonic n))
        (ContinuousMap.toLp 2 (normalizedSphere n) ℝ F) := by
  change (∫ θ, F θ ∂normalizedSphere n) =
    inner ℝ (ContinuousMap.toLp 2 (normalizedSphere n) ℝ (1 : C(UnitSphere n, ℝ)))
      (ContinuousMap.toLp 2 (normalizedSphere n) ℝ F)
  have h := ContinuousMap.inner_toLp (normalizedSphere n) (1 : C(UnitSphere n, ℝ)) F
  simpa only [ContinuousMap.one_apply, RCLike.conj_to_real, mul_one] using h.symm

private theorem harmonic_integral_zero (n k : ℕ) (hk : 0 < k) (v : HarmonicSpace n k) :
    (∫ θ, harmonicEvaluation n k θ v ∂normalizedSphere n) = 0 := by
  change (∫ θ, continuousHarmonicComponent n k v θ ∂normalizedSphere n) = 0
  rw [integral_eq_constant_inner, toLp_continuousHarmonicComponent]
  exact harmonic_orthogonal_lt n 0 k hk (constantHarmonic n) v

/-- Every nonempty canonical harmonic truncation preserves the exact spherical
mean. This follows from L2 orthogonality alone, without analytic assumptions. -/
theorem canonical_partial_integral (n N : ℕ) (hN : 0 < N) (F : C(UnitSphere n, ℝ)) :
    (∫ θ, (∑ k ∈ Finset.range N, harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k))
      ∂normalizedSphere n) = ∫ θ, F θ ∂normalizedSphere n := by
  let c := harmonicToLp n 0 (constantHarmonic n)
  have hi : ∀ k, Integrable (fun θ => harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k))
      (normalizedSphere n) := by
    intro k
    exact (continuousHarmonicComponent n k _).continuous.integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hs := (harmonicCoordinates_hasSum n (ContinuousMap.toLp 2 (normalizedSphere n) ℝ F)).map
    (innerSL ℝ c) (innerSL ℝ c).continuous
  have hterm : ∀ k, inner ℝ c (harmonicToLp n k (continuousHarmonicCoordinates n F k)) =
      ∫ θ, harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k) ∂normalizedSphere n := by
    intro k
    have ht := (integral_eq_constant_inner n (continuousHarmonicComponent n k (continuousHarmonicCoordinates n F k))).symm
    rw [toLp_continuousHarmonicComponent] at ht
    exact ht
  have hs' : HasSum (fun k => ∫ θ, harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k)
      ∂normalizedSphere n) (∫ θ, F θ ∂normalizedSphere n) := by
    change HasSum (fun k => inner ℝ c (harmonicToLp n k (continuousHarmonicCoordinates n F k)))
      (inner ℝ c (ContinuousMap.toLp 2 (normalizedSphere n) ℝ F)) at hs
    simpa only [hterm, integral_eq_constant_inner] using hs
  rw [integral_finsetSum _ (fun k _ => hi k)]
  rw [Finset.sum_eq_single 0]
  · exact (tsum_eq_single 0 (fun k hk => harmonic_integral_zero n k (Nat.pos_of_ne_zero hk) _)).symm.trans hs'.tsum_eq
  · intro k hk hk0
    exact harmonic_integral_zero n k (Nat.pos_of_ne_zero hk0) _
  · intro hnot
    exact (hnot (Finset.mem_range.mpr hN)).elim

/-- Canonical truncation preserves the integral, so positive exact quadrature
spends only one uniform tail bound. The original target is the fixed F. -/
theorem canonical_foldedSphere_error_of_partial
    (q : ℕ) (hq : 0 < q) (n : ℕ) (F : C(UnitSphere n, ℝ))
    (heven : ∀ θ, F (sphereNeg n θ) = F θ) (err : ℝ)
    (hpartial : ∀ θ, |F θ - ∑ k ∈ Finset.range (2*q),
      harmonicEvaluation n k θ (continuousHarmonicCoordinates n F k)| ≤ err) :
    |(∫ θ, F θ ∂normalizedSphere n) -
      ∑ i : SphereHalfIndex q n, foldedSphereWeight q hq n i * F (sphereNodeOnSphere q hq n i.val)| ≤ err := by
  let v := continuousHarmonicCoordinates n F
  let P : UnitSphere n → ℝ := fun θ =>
    ∑ k ∈ Finset.range (2*q), harmonicEvaluation n k θ (v k)
  have hp : ∀ θ, |F θ-P θ| ≤ err := hpartial
  have hi : ∀ k, Integrable (fun θ => harmonicEvaluation n k θ (v k)) (normalizedSphere n) := by
    intro k
    have hc : Continuous (fun θ => polynomialRepresentative n k (v k).val θ) := by fun_prop
    exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hexact : (∫ θ, P θ ∂normalizedSphere n) =
      ∑ i, sphereWeight q hq n i * P (sphereNodeOnSphere q hq n i) := by
    unfold P
    rw [integral_finsetSum _ (fun k _ => hi k)]
    calc
      _ = ∑ k ∈ Finset.range (2*q), ∑ i, sphereWeight q hq n i *
          harmonicEvaluation n k (sphereNodeOnSphere q hq n i) (v k) := by
        apply Finset.sum_congr rfl
        intro k hk
        exact (sphereRule_polynomialL2_exact q hq n k (v k).val (Finset.mem_range.mp hk)).symm
      _ = _ := by rw [Finset.sum_comm]; simp_rw [Finset.mul_sum]
  have hmean : (∫ θ, P θ ∂normalizedSphere n) = ∫ θ, F θ ∂normalizedSphere n :=
    canonical_partial_integral n (2*q) (by omega) F
  have herr := QIRadon.normalized_weighted_error_bound (sphereWeight q hq n)
    (fun i => P (sphereNodeOnSphere q hq n i)) (fun i => F (sphereNodeOnSphere q hq n i)) err
    (fun i => (sphereWeight_positive q hq n i).le) (sphereWeight_normalized q hq n)
    (fun i => by rw [abs_sub_comm]; exact hp (sphereNodeOnSphere q hq n i))
  rw [← hexact,hmean,foldedSphere_sum_even_onSphere q hq n F heven] at herr
  exact herr

end
end QIRadon.Gaussian
