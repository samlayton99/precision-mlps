import QIRadon.Quadrature.HarmonicBound
import QIRadon.Quadrature.SobolevTail

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

set_option maxHeartbeats 1200000 in
/-- Algebraic angular convergence for actual orthogonal polynomial/harmonic
layers with weighted L2 energy. The growth exponent is the sharp ambient d-2,
so the resulting tail loses only the standard (d-1)/2 from Sobolev to uniform
control; no additional half derivative is discarded.

The separate target-side theorem must still produce the expansion/energy from
its stated smoothness assumptions. They are explicit hypotheses here.
-/
theorem foldedSphere_sobolev_series_error (q : ℕ) (hq : 0 < q) (m : ℕ)
    (v : (k : ℕ) → HarmonicSpace (m + 1) k) (F : UnitSphere (m + 1) → ℝ)
    (hF : Integrable F (normalizedSphere (m + 1)))
    (heven : ∀ θ, F (sphereNeg (m + 1) θ) = F θ)
    (hrep : ∀ θ, F θ = ∑' k, harmonicEvaluation (m + 1) k θ (v k))
    (s A : ℝ) (hs : ((m : ℝ) + 1) / 2 < s) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ ↦ (((k : ℝ) + 1) ^ s * ‖v k‖) ^ 2))
    (henergy : (∑' k : ℕ, (((k : ℝ) + 1) ^ s * ‖v k‖) ^ 2) ≤ A ^ 2) :
    |(∫ θ, F θ ∂normalizedSphere (m + 1)) -
      ∑ i : SphereHalfIndex q (m + 1), foldedSphereWeight q hq (m + 1) i *
        F (sphereNodeOnSphere q hq (m + 1) i.val)| ≤
    2 * (harmonicGrowthConstant m * A *
      Real.sqrt ((((2 * q - 1 : ℕ) : ℝ) ^ ((m : ℝ) + 1 - 2 * s)) /
        (2 * s - (m : ℝ) - 1))) := by
  let P : UnitSphere (m + 1) → ℝ := fun θ ↦
    ∑ k ∈ Finset.range (2 * q), harmonicEvaluation (m + 1) k θ (v k)
  let err : ℝ := harmonicGrowthConstant m * A *
    Real.sqrt ((((2 * q - 1 : ℕ) : ℝ) ^ ((m : ℝ) + 1 - 2 * s)) /
      (2 * s - (m : ℝ) - 1))
  have happrox : ∀ θ, |F θ - P θ| ≤ err := by
    intro θ
    have h := QIRadon.SobolevTail.sobolev_component_series_tail
      (fun k θ ↦ harmonicEvaluation (m + 1) k θ (v k))
      (ρ := (m : ℝ) / 2) (by linarith) hA (harmonicGrowthConstant_nonneg m)
      (fun k ↦ norm_nonneg (v k)) hsum henergy
      (fun k θ ↦ harmonicEvaluation_bound m k θ (v k)) θ
      (n := 2 * q - 1) (by omega)
    rw [Nat.sub_add_cancel (by omega : 1 ≤ 2 * q)] at h
    have hexp : 1 + 2 * ((m : ℝ) / 2 - s) = (m : ℝ) + 1 - 2 * s := by ring
    have hden : -2 * ((m : ℝ) / 2 - s) - 1 = 2 * s - (m : ℝ) - 1 := by ring
    rw [hexp, hden] at h
    simpa only [P, err, hrep θ] using h.2
  have hPc : Continuous P := by
    unfold P harmonicEvaluation polynomialEvaluation
    change Continuous (fun θ ↦ ∑ k ∈ Finset.range (2 * q),
      polynomialRepresentative (m + 1) k (v k).val θ)
    fun_prop
  have hPi : Integrable P (normalizedSphere (m + 1)) :=
    hPc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hexact : (∫ θ, P θ ∂normalizedSphere (m + 1)) =
      ∑ i, sphereWeight q hq (m + 1) i * P (sphereNodeOnSphere q hq (m + 1) i) := by
    unfold P
    have hi : ∀ k, Integrable (fun θ ↦ harmonicEvaluation (m + 1) k θ (v k))
        (normalizedSphere (m + 1)) := by
      intro k
      have hc : Continuous (fun θ ↦ polynomialRepresentative (m + 1) k (v k).val θ) := by fun_prop
      exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
    rw [integral_finsetSum _ (fun k _ ↦ hi k)]
    calc
      _ = ∑ k ∈ Finset.range (2 * q), ∑ i, sphereWeight q hq (m + 1) i *
          harmonicEvaluation (m + 1) k (sphereNodeOnSphere q hq (m + 1) i) (v k) := by
        apply Finset.sum_congr rfl
        intro k hk
        exact (sphereRule_polynomialL2_exact q hq (m + 1) k (v k).val (Finset.mem_range.mp hk)).symm
      _ = _ := by rw [Finset.sum_comm]; simp_rw [Finset.mul_sum]
  have h := QIRadon.positive_quadrature_error (normalizedSphere (m + 1))
    (sphereNodeOnSphere q hq (m + 1)) (sphereWeight q hq (m + 1)) F P err
    (fun i ↦ (sphereWeight_positive q hq (m + 1) i).le)
    (sphereWeight_normalized q hq (m + 1)) hF hPi hexact happrox
  rwa [foldedSphere_sum_even_onSphere q hq (m + 1) F heven] at h

end QIRadon.Gaussian
