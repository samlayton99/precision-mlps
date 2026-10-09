import QIRadon.Quadrature.HarmonicSpace
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib

noncomputable section
namespace QIRadon.HarmonicGenerating
open QIRadon.Gaussian

/-- Exact positive generating function for homogeneous polynomial dimensions. -/
theorem polynomial_dimension_hasSum (n : ℕ) {r : ℝ} (hr : 0 ≤ r) (hr1 : r < 1) :
    HasSum (fun k : ℕ => ((n+k).choose k : ℝ) * r^k) (1/(1-r)^(n+1)) := by
  have hn : ‖r‖ < 1 := by simpa only [Real.norm_eq_abs, abs_of_nonneg hr] using hr1
  have h := hasSum_choose_mul_geometric_of_norm_lt_one n hn
  have heq : (fun k : ℕ => ((n+k).choose k : ℝ) * r^k) =
      (fun k : ℕ => ((k+n).choose n : ℝ) * r^k) := by
    funext k
    rw [Nat.add_comm k n, Nat.choose_symm_add]
  rw [heq]
  exact h

/-- Harmonic dimensions are bounded by the enclosing homogeneous space. -/
theorem harmonic_dimension_le (n k : ℕ) :
    Module.finrank ℝ (HarmonicSpace n k) ≤ (n+k).choose k := by
  have h := (harmonicSubspace n k).finrank_le
  rwa [polynomialL2_finrank] at h

/-- Exact canonical harmonic trace generating function on the n-sphere.
This supplies a substantially sharper dimension constant than bounding each
component separately by a polynomial and summing a factorial majorant. -/
theorem harmonic_dimension_hasSum (n : ℕ) {r : ℝ} (hr : 0 ≤ r) (hr1 : r < 1) :
    HasSum (fun k : ℕ => (Module.finrank ℝ (HarmonicSpace n k) : ℝ) * r^k)
      ((1+r)/(1-r)^n) := by
  let P := fun k : ℕ => ((n+k).choose k : ℝ) * r^k
  let H := fun k : ℕ => (Module.finrank ℝ (HarmonicSpace n k) : ℝ) * r^k
  have hP : HasSum P (1/(1-r)^(n+1)) := polynomial_dimension_hasSum n hr hr1
  have hle : ∀ k, H k ≤ P k := by
    intro k
    apply mul_le_mul_of_nonneg_right _ (pow_nonneg hr _)
    exact_mod_cast harmonic_dimension_le n k
  have hH : Summable H := hP.summable.of_nonneg_of_le
    (fun k => mul_nonneg (Nat.cast_nonneg _) (pow_nonneg hr _)) hle
  have hPt := (summable_nat_add_iff 2).mpr hP.summable
  have hHt := (summable_nat_add_iff 2).mpr hH
  have hshift := hP.summable.mul_left (r^2)
  have hcoeff : ∀ k, H (k+2) + r^2 * P k = P (k+2) := by
    intro k
    have hd : (Module.finrank ℝ (HarmonicSpace n (k+2)) : ℝ) + ((n+k).choose k : ℝ) =
        ((n+(k+2)).choose (k+2) : ℝ) := by exact_mod_cast harmonic_finrank_add n k
    dsimp [H,P]
    rw [pow_add]
    nlinarith [congrArg (fun v : ℝ => v*(r^k*r^2)) hd]
  have heq : (∑' k, H (k+2)) + r^2 * (∑' k, P k) = ∑' k, P (k+2) := by
    rw [← tsum_mul_left, ← hHt.tsum_add hshift]
    exact tsum_congr hcoeff
  have hh := hH.sum_add_tsum_nat_add 2
  have hp := hP.summable.sum_add_tsum_nat_add 2
  have hhead : (∑ k ∈ Finset.range 2, H k) = ∑ k ∈ Finset.range 2, P k := by
    simp [H,P, Finset.sum_range_succ, harmonic_finrank_zero, harmonic_finrank_one]
  rw [hhead] at hh
  have htotal : (∑' k, H k) = (1-r^2) * (1/(1-r)^(n+1)) := by
    rw [hP.tsum_eq] at heq hp
    nlinarith
  have hrnz : 1-r ≠ 0 := ne_of_gt (sub_pos.mpr hr1)
  have hid : (1-r^2) * (1/(1-r)^(n+1)) = (1+r)/(1-r)^n := by
    rw [pow_succ]
    field_simp
    ring
  exact htotal.trans hid ▸ hH.hasSum

end QIRadon.HarmonicGenerating
