import QIRadon.Quadrature.SeriesTail
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.PSeries
import Mathlib.Analysis.SumIntegralComparisons
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals

noncomputable section
open MeasureTheory Set

namespace QIRadon.SobolevTail

/-- The complete positive power tail, with its exact integral-test constant. -/
theorem power_tail_bound {p : ℝ} (hp : 1 < p) {n : ℕ} (hn : 1 ≤ n) :
    (∑' j : ℕ, ((j + n + 1 : ℕ) : ℝ) ^ (-p)) ≤
      (n : ℝ)^(1 - p) / (p - 1) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast lt_of_lt_of_le Nat.zero_lt_one hn
  have ha : AntitoneOn (fun x : ℝ => x^(-p)) (Ici (n : ℝ)) := by
    apply (Real.antitoneOn_rpow_Ioi_of_exponent_nonpos (by linarith : -p ≤ 0)).mono
    intro x hx
    exact lt_of_lt_of_le hn0 hx
  have hint := integrableOn_Ioi_rpow_of_lt (by linarith : -p < -1) hn0
  have hb := ha.tsum_comp_add_le_integral n hint
    (fun x hx => Real.rpow_nonneg (le_of_lt (lt_trans hn0 hx)) _)
  rw [integral_Ioi_rpow_of_lt (by linarith : -p < -1) hn0] at hb
  have hid : -(n : ℝ)^(-p + 1) / (-p + 1) = (n : ℝ)^(1 - p) / (p - 1) := by
    rw [show -p + 1 = -(p - 1) by ring, div_neg]
    ring_nf
  simpa only [hid] using hb

/-- Removing finitely many nonnegative terms cannot increase the total energy. -/
theorem shifted_energy_le {u : ℕ → ℝ} (hu : ∀ k, 0 ≤ u k)
    (hs : Summable u) (n : ℕ) :
    (∑' j, u (j + n)) ≤ ∑' j, u j := by
  have hi := hs.sum_add_tsum_nat_add n
  have hp : 0 ≤ ∑ k ∈ Finset.range n, u k := Finset.sum_nonneg (fun k _ => hu k)
  linarith

/-- Cauchy–Schwarz for an actual coefficient tail. The first factor is bounded
by the full supplied energy; the second retains its exact summed shape. -/
theorem energy_times_shape_tail {u v : ℕ → ℝ} {A : ℝ}
    (hu : ∀ k, 0 ≤ u k) (hv : ∀ k, 0 ≤ v k) (hA : 0 ≤ A)
    (hsu : Summable (fun k => (u k)^2))
    (hsv : Summable (fun k => (v k)^2))
    (henergy : (∑' k, (u k)^2) ≤ A^2) (n : ℕ) :
    Summable (fun k => u k * v k) ∧
    (∑' j, u (j + n) * v (j + n)) ≤
      A * Real.sqrt (∑' j, (v (j + n))^2) := by
  have hsu' : Summable (fun k => (u k)^(2 : ℝ)) := by simpa only [Real.rpow_two] using hsu
  have hsv' : Summable (fun k => (v k)^(2 : ℝ)) := by simpa only [Real.rpow_two] using hsv
  have hall := Real.summable_and_inner_le_Lp_mul_Lq_tsum_of_nonneg
    Real.HolderConjugate.two_two hu hv hsu' hsv'
  refine ⟨hall.1, ?_⟩
  have ht := Real.summable_and_inner_le_Lp_mul_Lq_tsum_of_nonneg
    Real.HolderConjugate.two_two (fun j => hu (j + n)) (fun j => hv (j + n))
    ((summable_nat_add_iff n).mpr hsu') ((summable_nat_add_iff n).mpr hsv')
  simp only [Real.rpow_two, ← Real.sqrt_eq_rpow] at ht
  have hen := (shifted_energy_le (fun k => sq_nonneg (u k)) hsu n).trans henergy
  have hsqrt : Real.sqrt (∑' j, (u (j + n))^2) ≤ A := by
    exact (Real.sqrt_le_sqrt hen).trans_eq (Real.sqrt_sq hA)
  exact ht.2.trans (mul_le_mul_of_nonneg_right hsqrt (Real.sqrt_nonneg _))

theorem shifted_power_square {t : ℝ} (k : ℕ) :
    (((k : ℝ) + 1)^t)^2 = ((k : ℝ) + 1)^(2*t) := by
  rw [show 2*t = t*2 by ring, Real.rpow_mul (by positivity), Real.rpow_two]

/-- The squared shape is summable at exactly the Sobolev threshold. -/
theorem summable_shape_squares {t : ℝ} (ht : t < -(1 / 2 : ℝ)) :
    Summable (fun k : ℕ => (((k : ℝ) + 1)^t)^2) := by
  have hs : Summable (fun k : ℕ => (k : ℝ)^(2*t)) :=
    Real.summable_nat_rpow.mpr (by linarith)
  have hh := (summable_nat_add_iff 1).mpr hs
  simpa only [Nat.cast_add, Nat.cast_one, shifted_power_square] using hh

/-- Quantified tail of the squared reciprocal Sobolev weight. -/
theorem shape_squared_tail_bound {t : ℝ} (ht : t < -(1 / 2 : ℝ))
    {n : ℕ} (hn : 1 ≤ n) :
    (∑' j : ℕ, ((((j + (n + 1) : ℕ) : ℝ) + 1)^t)^2) ≤
      (n : ℝ)^(1 + 2*t) / (-2*t - 1) := by
  have hs := (summable_nat_add_iff (n+1)).mpr (summable_shape_squares ht)
  have hm : Summable (fun j : ℕ => ((j + n + 1 : ℕ) : ℝ)^(2*t)) := by
    have hh := (summable_nat_add_iff (n+1)).mpr
      (Real.summable_nat_rpow.mpr (show 2*t < -1 by linarith))
    simpa only [Nat.add_assoc] using hh
  have hp : ∀ j : ℕ, ((((j + (n+1) : ℕ) : ℝ) + 1)^t)^2 ≤
      ((j + n + 1 : ℕ) : ℝ)^(2*t) := by
    intro j
    rw [show ((((j + (n+1) : ℕ) : ℝ) + 1)^t)^2 =
      (((j + (n+1) : ℕ) : ℝ) + 1)^(2*t) from shifted_power_square (j+(n+1))]
    apply Real.rpow_le_rpow_of_nonpos (by positivity)
    · push_cast
      linarith
    · linarith
  have ht' := hs.tsum_le_tsum hp hm
  have hb := power_tail_bound (p := -(2*t)) (by linarith) hn
  have he : (∑' j : ℕ, ((j+n+1:ℕ):ℝ)^(2*t)) ≤
      (n : ℝ)^(1 + 2*t) / (-2*t - 1) := by
    simpa only [neg_neg, sub_neg_eq_add, neg_mul] using hb
  exact ht'.trans he

/-- Uniform Sobolev tail from actual weighted L2 coefficients. The exponent is
s−ρ−1/2. With the sharp harmonic evaluation exponent ρ=(d−2)/2, this is exactly
s−(d−1)/2, preserving the approved regularity loss. -/
theorem sobolev_component_series_tail {X : Type*} (P : ℕ → X → ℝ)
    {b : ℕ → ℝ} {s ρ A C : ℝ}
    (hreg : ρ + 1/2 < s) (hA : 0 ≤ A) (hC : 0 ≤ C) (hb : ∀ k, 0 ≤ b k)
    (hsum : Summable (fun k : ℕ => (((k : ℝ)+1)^s * b k)^2))
    (henergy : (∑' k : ℕ, (((k : ℝ)+1)^s * b k)^2) ≤ A^2)
    (heval : ∀ k x, |P k x| ≤ C * (((k : ℝ)+1)^ρ * b k))
    (x : X) {n : ℕ} (hn : 1 ≤ n) :
    Summable (fun k => P k x) ∧
    |(∑' k, P k x) - ∑ k ∈ Finset.range (n+1), P k x| ≤
      C * A * Real.sqrt ((n : ℝ)^(1 + 2*(ρ-s)) / (-2*(ρ-s)-1)) := by
  let u := fun k : ℕ => ((k : ℝ)+1)^s * b k
  let v := fun k : ℕ => ((k : ℝ)+1)^(ρ-s)
  have hu : ∀ k, 0 ≤ u k := by intro k; dsimp [u]; exact mul_nonneg (Real.rpow_nonneg (by positivity) _) (hb k)
  have hv : ∀ k, 0 ≤ v k := by intro k; dsimp [v]; positivity
  have ht : ρ-s < -(1/2 : ℝ) := by linarith
  have hsv : Summable (fun k => (v k)^2) := summable_shape_squares ht
  have hholder := energy_times_shape_tail hu hv hA hsum hsv henergy (n+1)
  have hid : ∀ k, u k * v k = ((k : ℝ)+1)^ρ * b k := by
    intro k
    dsimp [u, v]
    rw [mul_right_comm, ← Real.rpow_add (by positivity : 0 < (k : ℝ)+1)]
    rw [show s + (ρ-s) = ρ by ring]
  have hpoint : ∀ k, ‖P k x‖ ≤ C * (u k * v k) := by
    intro k
    rw [Real.norm_eq_abs, hid]
    exact heval k x
  have hsP : Summable (fun k => P k x) := (hholder.1.mul_left C).of_norm_bounded hpoint
  refine ⟨hsP, ?_⟩
  have hsPt : Summable (fun j => P (j+(n+1)) x) := (summable_nat_add_iff (n+1)).mpr hsP
  have herr : (∑' k, P k x) - ∑ k ∈ Finset.range (n+1), P k x =
      ∑' j, P (j+(n+1)) x := by
    have hh := hsP.sum_add_tsum_nat_add (n+1)
    linarith
  rw [herr]
  have hmajor := ((summable_nat_add_iff (n+1)).mpr hholder.1).mul_left C
  have htail := hsPt.norm.tsum_le_tsum (fun j => hpoint (j+(n+1))) hmajor
  rw [tsum_mul_left] at htail
  have hshape := shape_squared_tail_bound ht hn
  calc
    |∑' j, P (j+(n+1)) x| ≤ ∑' j, ‖P (j+(n+1)) x‖ := norm_tsum_le_tsum_norm hsPt.norm
    _ ≤ C * ∑' j, u (j+(n+1)) * v (j+(n+1)) := htail
    _ ≤ C * (A * Real.sqrt (∑' j, (v (j+(n+1)))^2)) :=
      mul_le_mul_of_nonneg_left hholder.2 hC
    _ ≤ C * A * Real.sqrt ((n : ℝ)^(1 + 2*(ρ-s)) / (-2*(ρ-s)-1)) := by
      rw [← mul_assoc]
      exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hshape) (mul_nonneg hC hA)

end QIRadon.SobolevTail
