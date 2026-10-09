import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Tactic

/-! Quantitative optimization of all-orders projection estimates. This file
proves the scalar optimization and the weighted energy consequence; deriving
the projection estimates from an actual rotation/Casimir operator belongs to
the angular analysis and is not assumed to have already been done here. -/

noncomputable section
open scoped BigOperators
namespace QIRadon.CasimirDecay

/-- Exponential decay rate produced by optimizing the number of integrations
by parts. -/
def decayRate (c : ℝ) : ℝ := Real.log 2 / c

def energyRate (c : ℝ) : ℝ := Real.log 2 / (2*c)

theorem decayRate_pos {c : ℝ} (hc : 0 < c) : 0 < decayRate c :=
  div_pos (Real.log_pos (by norm_num)) hc

theorem energyRate_pos {c : ℝ} (hc : 0 < c) : 0 < energyRate c := by
  unfold energyRate
  positivity

theorem twice_energyRate (c : ℝ) : 2 * energyRate c = decayRate c := by
  unfold energyRate decayRate
  ring_nf

/-- The floor choice works even when it selects zero derivative orders. -/
theorem optimized_positive_index {b B c : ℝ} {k : ℕ}
    (hB : 0 ≤ B) (hc : 0 < c) (hk : 0 < k)
    (horders : ∀ j : ℕ, b ≤ B * (c * (j : ℝ) / k)^(2*j)) :
    b ≤ 4 * B * Real.exp (-decayRate c * k) := by
  let j : ℕ := ⌊(k : ℝ)/(2*c)⌋₊
  have hkR : 0 < (k : ℝ) := Nat.cast_pos.mpr hk
  have hj : (j : ℝ) ≤ (k : ℝ)/(2*c) := Nat.floor_le (by positivity)
  have hj' : (k : ℝ)/(2*c) < (j : ℝ)+1 := Nat.lt_floor_add_one _
  have hratio : c*(j : ℝ)/(k : ℝ) ≤ (1/2 : ℝ) := by
    apply (div_le_iff₀ hkR).mpr
    have hh := (le_div_iff₀ (show 0 < 2*c by positivity)).mp hj
    nlinarith
  have hpow := pow_le_pow_left₀ (show 0 ≤ c*(j : ℝ)/(k : ℝ) by positivity) hratio (2*j)
  have hbase : (1/2 : ℝ) = Real.exp (-Real.log 2) := by
    rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  have hpower : (1/2 : ℝ)^(2*j) = Real.exp (-(2*Real.log 2)*(j : ℝ)) := by
    rw [hbase, ← Real.exp_nat_mul]
    congr 1
    push_cast
    ring_nf
  have hexponent : -(2*Real.log 2)*(j : ℝ) ≤ 2*Real.log 2 - decayRate c*k := by
    have hl : 0 < Real.log (2 : ℝ) := Real.log_pos (by norm_num)
    have hm := mul_le_mul_of_nonneg_left hj'.le (show 0 ≤ 2*Real.log 2 by positivity)
    have he : (2*Real.log 2)*((k : ℝ)/(2*c)) = decayRate c*k := by
      unfold decayRate
      field_simp
    rw [he] at hm
    nlinarith
  have he4 : Real.exp (2*Real.log (2 : ℝ)) = 4 := by
    rw [show 2*Real.log (2 : ℝ) = Real.log 2 + Real.log 2 by ring_nf, Real.exp_add,
      Real.exp_log (by norm_num : (0 : ℝ) < 2)]
    norm_num
  calc
    b ≤ B * (c*(j : ℝ)/(k : ℝ))^(2*j) := horders j
    _ ≤ B * (1/2 : ℝ)^(2*j) := mul_le_mul_of_nonneg_left hpow hB
    _ = B * Real.exp (-(2*Real.log 2)*(j : ℝ)) := by rw [hpower]
    _ ≤ B * Real.exp (2*Real.log 2-decayRate c*k) :=
      mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hexponent) hB
    _ = 4 * B * Real.exp (-decayRate c*k) := by
      rw [sub_eq_add_neg, Real.exp_add, he4]
      ring_nf

/-- Full coefficient decay, including the constant harmonic layer. The
all-orders hypotheses concern projection estimates, not an assumed rate. -/
theorem all_orders_decay {b : ℕ → ℝ} {B c : ℝ}
    (hB : 0 ≤ B) (hc : 0 < c) (hb0 : b 0 ≤ B)
    (horders : ∀ k : ℕ, 0 < k → ∀ j : ℕ, b k ≤ B * (c*(j : ℝ)/k)^(2*j)) :
    ∀ k : ℕ, b k ≤ 4*B*Real.exp (-decayRate c*k) := by
  intro k
  cases k with
  | zero => simpa using hb0.trans (show B ≤ 4*B by linarith)
  | succ k => exact optimized_positive_index hB hc (Nat.succ_pos k) (horders _ (Nat.succ_pos k))

/-- A positive exponential rate controls the complete weighted square series. -/
theorem exponential_weighted_energy {b : ℕ → ℝ} {B rate : ℝ}
    (_hB : 0 ≤ B) (hrate : 0 < rate) (hb : ∀ k, 0 ≤ b k)
    (hdecay : ∀ k : ℕ, b k ≤ 4*B*Real.exp (-rate*k)) :
    Summable (fun k : ℕ => Real.exp (rate*k) * (b k)^2) ∧
      (∑' k : ℕ, Real.exp (rate*k) * (b k)^2) ≤ 16*B^2 / (1-Real.exp (-rate)) := by
  let q := Real.exp (-rate)
  have hq : 0 ≤ q := (Real.exp_pos _).le
  have hq1 : q < 1 := Real.exp_lt_one_iff.mpr (neg_neg_of_pos hrate)
  have hs := (summable_geometric_of_lt_one hq hq1).mul_left (16*B^2)
  have hmajor : ∀ k : ℕ, Real.exp (rate*k) * (b k)^2 ≤ 16*B^2*q^k := by
    intro k
    have hp := pow_le_pow_left₀ (hb k) (hdecay k) 2
    have hm := mul_le_mul_of_nonneg_left hp (Real.exp_pos (rate*k)).le
    apply hm.trans_eq
    have he : Real.exp (rate*k) * (Real.exp (-rate*k))^2 = Real.exp (-rate*k) := by
      rw [← Real.exp_nat_mul, ← Real.exp_add]
      congr 1
      push_cast
      ring_nf
    have hqpow : q^k = Real.exp (-rate*k) := by
      dsimp [q]
      rw [← Real.exp_nat_mul]
      congr 1
      ring_nf
    calc
      _ = (16*B^2) * (Real.exp (rate*k) * (Real.exp (-rate*k))^2) := by ring_nf
      _ = _ := by rw [he, hqpow]
  have hsum : Summable (fun k : ℕ => Real.exp (rate*k) * (b k)^2) :=
    hs.of_nonneg_of_le (fun _ => by positivity) hmajor
  refine ⟨hsum, ?_⟩
  calc
    _ ≤ ∑' k : ℕ, 16*B^2*q^k := hsum.tsum_le_tsum hmajor hs
    _ = _ := by
      rw [tsum_mul_left, tsum_geometric_of_lt_one hq hq1]
      rfl

/-- Explicit weighted ℓ² energy derived from the complete family of projection
estimates. Nonnegativity records that b_k is a coefficient norm. -/
theorem all_orders_energy {b : ℕ → ℝ} {B c : ℝ}
    (hB : 0 ≤ B) (hc : 0 < c) (hb : ∀ k, 0 ≤ b k) (hb0 : b 0 ≤ B)
    (horders : ∀ k : ℕ, 0 < k → ∀ j : ℕ, b k ≤ B * (c*(j : ℝ)/k)^(2*j)) :
    Summable (fun k : ℕ => Real.exp (2*energyRate c*k) * (b k)^2) ∧
      (∑' k : ℕ, Real.exp (2*energyRate c*k) * (b k)^2) ≤
        16*B^2/(1-Real.exp (-decayRate c)) := by
  simpa only [twice_energyRate] using exponential_weighted_energy hB (decayRate_pos hc) hb
    (all_orders_decay hB hc hb0 horders)

/-- The explicit nonnegative A used in the canonical harmonic-energy interface. -/
def energyConstant (B c : ℝ) : ℝ := 4*B / Real.sqrt (1-Real.exp (-decayRate c))

theorem energyConstant_nonneg {B c : ℝ} (hB : 0 ≤ B) : 0 ≤ energyConstant B c := by
  unfold energyConstant
  positivity

theorem energyConstant_sq {B c : ℝ} (hc : 0 < c) :
    (energyConstant B c)^2 = 16*B^2/(1-Real.exp (-decayRate c)) := by
  have hd : 0 < 1-Real.exp (-decayRate c) :=
    sub_pos.mpr (Real.exp_lt_one_iff.mpr (neg_neg_of_pos (decayRate_pos hc)))
  unfold energyConstant
  rw [div_pow, mul_pow, Real.sq_sqrt hd.le]
  norm_num

theorem all_orders_energy_certificate {b : ℕ → ℝ} {B c : ℝ}
    (hB : 0 ≤ B) (hc : 0 < c) (hb : ∀ k, 0 ≤ b k) (hb0 : b 0 ≤ B)
    (horders : ∀ k : ℕ, 0 < k → ∀ j : ℕ, b k ≤ B * (c*(j : ℝ)/k)^(2*j)) :
    Summable (fun k : ℕ => Real.exp (2*energyRate c*k) * (b k)^2) ∧
      (∑' k : ℕ, Real.exp (2*energyRate c*k) * (b k)^2) ≤ (energyConstant B c)^2 := by
  rw [energyConstant_sq hc]
  exact all_orders_energy hB hc hb hb0 horders

end QIRadon.CasimirDecay
