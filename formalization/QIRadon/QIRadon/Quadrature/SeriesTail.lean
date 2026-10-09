import QIRadon.Quadrature.Addition
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Topology.Algebra.InfiniteSum.NatInt

noncomputable section
open MeasureTheory

namespace QIRadon.SeriesTail

/-- Every component of a weighted L2 expansion is controlled by its actual
summed energy. This is not a supplied uniform approximation bound. -/
theorem component_bound {b : ℕ → ℝ} {a A : ℝ}
    (hb : ∀ k, 0 ≤ b k) (hA : 0 ≤ A)
    (hs : Summable (fun k : ℕ => Real.exp (2 * a * k) * (b k)^2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * (b k)^2) ≤ A^2) (k : ℕ) :
    b k ≤ A * Real.exp (-a * k) := by
  have ht := (hs.le_tsum k (fun i _ => by positivity)).trans henergy
  have hexp : Real.exp (2 * a * k) = (Real.exp (a * k))^2 := by
    rw [← Real.exp_nat_mul]
    congr 1
    norm_num
    ring
  have hsq : (Real.exp (a * k) * b k)^2 ≤ A^2 := by
    simpa only [hexp, mul_pow] using ht
  have hl := (sq_le_sq₀ (mul_nonneg (Real.exp_pos _).le (hb k)) hA).mp hsq
  have hd := (le_div_iff₀ (Real.exp_pos (a * k))).mpr (by simpa [mul_comm] using hl)
  calc
    b k ≤ A / Real.exp (a * k) := hd
    _ = A * Real.exp (-a * k) := by
      rw [show -a * (k : ℝ) = -(a * k) by ring, Real.exp_neg]
      rfl

/-- Explicit absorption of the dimension's polynomial growth into half of the
analytic exponential rate. The loss changes constants, not the geometric class. -/
theorem polynomial_times_exponential_bound (d k : ℕ) {a : ℝ} (ha : 0 < a) :
    ((k : ℝ) + 1)^d * Real.exp (-a * k) ≤
      ((d.factorial : ℝ) / (a / 2)^d * Real.exp (a / 2)) * Real.exp (-(a / 2) * k) := by
  have ha2 := half_pos ha
  have hp := Real.pow_div_factorial_le_exp ((a / 2) * ((k : ℝ) + 1))
    (show 0 ≤ (a / 2) * ((k : ℝ) + 1) by positivity) d
  have hf : 0 < (d.factorial : ℝ) := by positivity
  have had : 0 < (a / 2)^d := pow_pos ha2 _
  have hp' : ((k : ℝ) + 1)^d ≤
      ((d.factorial : ℝ) / (a / 2)^d) * Real.exp ((a / 2) * ((k : ℝ) + 1)) := by
    calc
      ((k : ℝ) + 1)^d =
          (((a / 2) * ((k : ℝ) + 1))^d / (d.factorial : ℝ)) *
            ((d.factorial : ℝ) / (a / 2)^d) := by
        rw [mul_pow]
        field_simp
      _ ≤ Real.exp ((a / 2) * ((k : ℝ) + 1)) *
            ((d.factorial : ℝ) / (a / 2)^d) :=
        mul_le_mul_of_nonneg_right hp (by positivity)
      _ = _ := by ring
  calc
    _ ≤ (((d.factorial : ℝ) / (a / 2)^d) *
        Real.exp ((a / 2) * ((k : ℝ) + 1))) * Real.exp (-a * k) :=
      mul_le_mul_of_nonneg_right hp' (Real.exp_pos _).le
    _ = _ := by
      simp only [mul_assoc, ← Real.exp_add]
      congr 2
      ring

/-- An explicit geometric tail for the actual series and its finite partial sum. -/
theorem geometric_partial_sum_error {u : ℕ → ℝ} {C r : ℝ}
    (_hC : 0 ≤ C) (hr : 0 ≤ r) (hr1 : r < 1)
    (hu : ∀ k, |u k| ≤ C * r^k) (n : ℕ) :
    Summable u ∧ |(∑' k, u k) - ∑ k ∈ Finset.range n, u k| ≤
      C * r^n / (1 - r) := by
  have hgeom := summable_geometric_of_lt_one hr hr1
  have hs : Summable u := (hgeom.mul_left C).of_norm_bounded
    (fun k => by simpa only [Real.norm_eq_abs] using hu k)
  refine ⟨hs, ?_⟩
  have htail : Summable (fun k => u (k + n)) := (summable_nat_add_iff n).mpr hs
  have hid := hs.sum_add_tsum_nat_add n
  have heq : (∑' k, u k) - ∑ k ∈ Finset.range n, u k = ∑' k, u (k + n) := by linarith
  rw [heq]
  have hle : ∀ k : ℕ, ‖u (k + n)‖ ≤ (C * r^n) * r^k := by
    intro k
    simpa only [Real.norm_eq_abs, pow_add, mul_comm, mul_left_comm, mul_assoc] using hu (k + n)
  calc
    |∑' k, u (k + n)| ≤ ∑' k, ‖u (k + n)‖ := norm_tsum_le_tsum_norm htail.norm
    _ ≤ ∑' k : ℕ, (C * r^n) * r^k :=
      htail.norm.tsum_le_tsum hle (hgeom.mul_left _)
    _ = _ := by rw [tsum_mul_left, tsum_geometric_of_lt_one hr hr1]; rfl

/-- Analytic weighted L2 energy and the proved polynomial evaluation growth
imply an actual uniform partial-sum approximation. Polynomial nature/degree is
not needed for this analytic lemma, and is supplied by the angular instantiation. -/
theorem analytic_component_series_tail {X : Type*} (P : ℕ → X → ℝ)
    {b : ℕ → ℝ} {a A : ℝ} (d : ℕ)
    (ha : 0 < a) (hA : 0 ≤ A) (hb : ∀ k, 0 ≤ b k)
    (henergy_sum : Summable (fun k : ℕ => Real.exp (2 * a * k) * (b k)^2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * (b k)^2) ≤ A^2)
    (heval : ∀ k x, |P k x| ≤ ((k : ℝ) + 1)^d * b k)
    (x : X) (n : ℕ) :
    Summable (fun k => P k x) ∧
      |(∑' k, P k x) - ∑ k ∈ Finset.range n, P k x| ≤
        (A * ((d.factorial : ℝ) / (a / 2)^d * Real.exp (a / 2))) *
          Real.exp (-(a / 2) * n) / (1 - Real.exp (-(a / 2))) := by
  let C := A * ((d.factorial : ℝ) / (a / 2)^d * Real.exp (a / 2))
  let r := Real.exp (-(a / 2))
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hr : 0 ≤ r := (Real.exp_pos _).le
  have hr1 : r < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  have hterm : ∀ k, |P k x| ≤ C * r^k := by
    intro k
    have hb' := component_bound hb hA henergy_sum henergy k
    have hpoly := polynomial_times_exponential_bound d k ha
    calc
      |P k x| ≤ ((k : ℝ) + 1)^d * b k := heval k x
      _ ≤ ((k : ℝ) + 1)^d * (A * Real.exp (-a * k)) :=
        mul_le_mul_of_nonneg_left hb' (by positivity)
      _ = A * (((k : ℝ) + 1)^d * Real.exp (-a * k)) := by ring
      _ ≤ A * (((d.factorial : ℝ) / (a / 2)^d * Real.exp (a / 2)) *
          Real.exp (-(a / 2) * k)) := mul_le_mul_of_nonneg_left hpoly hA
      _ = C * r^k := by
        dsimp [C, r]
        rw [← Real.exp_nat_mul]
        have heq : Real.exp (-(a / 2) * (k : ℝ)) = Real.exp ((k : ℝ) * (-(a / 2))) := by
          congr 1
          ring
        rw [heq]
        ring
  have hf := geometric_partial_sum_error hC hr hr1 hterm n
  have hid : r^n = Real.exp (-(a / 2) * n) := by
    dsimp [r]
    rw [← Real.exp_nat_mul]
    congr 1
    ring
  simpa only [C, hid, r] using hf

end QIRadon.SeriesTail
