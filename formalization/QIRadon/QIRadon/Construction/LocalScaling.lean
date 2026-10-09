import QIRadon.Construction.LocalMultidimensional
import QIRadon.Scalar.ScalarScaling

/-!
# Actual halo-inclusive local Radon construction: logarithmic width

The integer orders here are exactly the orders used by
`LocalMultidimensional.local_analytic_accuracy_and_width`. The count is the
cardinality of its finite direction × center index type, not an asymptotic
surrogate. Kernel and target constants, including angular regularity constants,
may depend on the fixed dimension. No dimension-independent constant is claimed.
-/

noncomputable section
open Filter Asymptotics
open scoped Topology Real
namespace QIRadon.LocalScaling
open QIRadon.Gaussian QIRadon.LocalProfileRate QIRadon.LocalMultidimensional
open QIRadon.ProfilePreparation

/-- Rounding and the enforced positive preparation order cost only this explicit
factor. This holds even when the error constant is zero. -/
theorem localAccuracyOrder_le {C b ε : ℝ} (hb : 0 < b) :
    (localAccuracyOrder C b ε : ℝ) ≤
      (1 + 1 / b) * max 1 (Real.log (C / ε)) := by
  let L := max 1 (Real.log (C / ε))
  have hL : 1 ≤ L := le_max_left _ _
  have hL0 : 0 ≤ L := by linarith
  by_cases hsmall : L/b ≤ 1
  · have horder : localAccuracyOrder C b ε=1 := by
      change ⌈max 1 (L/b)⌉₊=1
      rw [max_eq_left hsmall]
      norm_num
    rw [horder,Nat.cast_one]
    have hpositive := div_nonneg hL0 hb.le
    change 1 ≤ (1+1/b)*L
    calc
      1 ≤ L := hL
      _ ≤ L+L/b := le_add_of_nonneg_right hpositive
      _ = _ := by ring
  · have hlarge : 1 ≤ L/b := (lt_of_not_ge hsmall).le
    have hceil := (Nat.ceil_lt_add_one (show 0 ≤ L/b by positivity)).le
    change (⌈max 1 (L/b)⌉₊:ℝ) ≤ (1+1/b)*L
    rw [max_eq_right hlarge]
    calc
      _ ≤ L/b+1 := hceil
      _ ≤ L/b+L := add_le_add le_rfl hL
      _ = _ := by ring

/-- A finite constant valid also for zero angular or scalar error. The factor
2 inside the logarithm accounts for assigning half the requested error budget
to that part of the construction. -/
def orderConstant (C b : ℝ) : ℝ :=
  (1 + 1 / b) * (|Real.log (max 1 (2 * C))| + 1)

theorem orderConstant_pos {C b : ℝ} (hb : 0 < b) : 0 < orderConstant C b := by
  unfold orderConstant
  positivity

/-- Exact order used at tolerance ε/2 is bounded explicitly by |log ε|.
Zero target constants require no artificial positive lower bound. -/
theorem localAccuracyOrder_log_bound {C b ε : ℝ}
    (hC : 0 ≤ C) (hb : 0 < b) (hε : 0 < ε) (hsmall : ε ≤ Real.exp (-1)) :
    (localAccuracyOrder C b (ε / 2) : ℝ) ≤ orderConstant C b * |Real.log ε| := by
  let D := max 1 (2 * C)
  have hD : 0 < D := lt_of_lt_of_le zero_lt_one (le_max_left _ _)
  have hlog : max 1 (Real.log (C / (ε / 2))) ≤ max 1 (Real.log (D / ε)) := by
    by_cases hC0 : C = 0
    · simp only [hC0, zero_div, Real.log_zero, max_eq_left zero_le_one]
      exact le_max_left _ _
    · have hCp : 0 < C := lt_of_le_of_ne hC (Ne.symm hC0)
      apply max_le_max_left
      apply Real.log_le_log (div_pos hCp (half_pos hε))
      rw [div_div_eq_mul_div]
      exact div_le_div_of_nonneg_right (by simpa only [D, mul_comm] using le_max_right 1 (2*C)) hε.le
  have hs := ScalarScaling.logarithmic_order_bound hD hε hsmall
  calc
    _ ≤ (1 + 1/b) * max 1 (Real.log (C / (ε/2))) := localAccuracyOrder_le hb
    _ ≤ (1 + 1/b) * ((|Real.log D| + 1) * |Real.log ε|) :=
      mul_le_mul_of_nonneg_left (hlog.trans hs) (by positivity)
    _ = _ := by simp only [orderConstant, D]; ring

/-- Explicit finite width bound for exactly the two integer orders and the
halo-inclusive finite grid of the local-profile accuracy theorem. -/
theorem local_total_log_width (n : ℕ) {T₀ H Z ν μ U V a b ε : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 < a) (hb : 0 < b)
    (hε : 0 < ε) (hsmall : ε ≤ Real.exp (-1)) :
    let q := localAccuracyOrder U a (ε/2)
    let N := localAccuracyOrder V b (ε/2)
    let κ := max 1 (1/μ + 1/ν)
    (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) : ℝ) ≤
      (orderConstant U a ^ n *
        (((2*H/Real.pi + 2*H*Z + 1) * (preparationGrowth T₀ + 3)^κ) *
          orderConstant V b ^ κ)) * |Real.log ε| ^ ((n : ℝ) + κ) := by
  let q := localAccuracyOrder U a (ε/2)
  let N := localAccuracyOrder V b (ε/2)
  let κ : ℝ := max 1 (1/μ+1/ν)
  let B : ℝ := (2*H/Real.pi + 2*H*Z + 1) * (preparationGrowth T₀ + 3)^κ
  have hκ : 0 ≤ κ := le_trans zero_le_one (le_max_left _ _)
  have hB : 0 ≤ B := by
    dsimp [B]
    have hg := preparationGrowth_pos hT
    positivity
  have hq := localAccuracyOrder_log_bound hU ha hε hsmall
  have hN := localAccuracyOrder_log_bound hV hb hε hsmall
  have hqp := pow_le_pow_left₀ (Nat.cast_nonneg q) hq n
  have hNp := Real.rpow_le_rpow (Nat.cast_nonneg N) hN hκ
  rw [mul_pow] at hqp
  rw [Real.mul_rpow (orderConstant_pos hb).le (abs_nonneg _)] at hNp
  have hlog : 0 < |Real.log ε| := by
    have hh := Real.log_le_log hε hsmall
    rw [Real.log_exp] at hh
    rw [abs_of_nonpos (by linarith)]
    linarith
  have hw := local_total_grid_width (N := N) (ν := ν) (μ := μ) q
    (localAccuracyOrder_pos _ _ _) n hT (localAccuracyOrder_pos _ _ _) hH hZ
  apply hw.trans
  change (q : ℝ)^n * (B * (N : ℝ)^κ) ≤ _
  have hUa := orderConstant_pos (C := U) ha
  have hp := mul_le_mul hqp (mul_le_mul_of_nonneg_left hNp hB)
    (by positivity : 0 ≤ B * (N : ℝ)^κ) (by positivity)
  apply hp.trans_eq
  rw [Real.rpow_add hlog, Real.rpow_natCast]
  dsimp only [B, κ]
  ring

/-- The local construction's actual total finite grid satisfies the analytic
law W = O(|log ε|^(d−1+κ)), d=n+1. All halo centers and both integer roundings are
included. Neither positive target constants nor a global Fourier bound for the
original profiles are required. -/
theorem local_total_grid_epsilon_bigO (n : ℕ) {T₀ H Z ν μ U V a b : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 < a) (hb : 0 < b) :
    (fun ε : ℝ =>
      let q := localAccuracyOrder U a (ε/2)
      let N := localAccuracyOrder V b (ε/2)
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) : ℝ))
      =O[𝓝[>] (0 : ℝ)]
        (fun ε : ℝ => |Real.log ε| ^ ((n : ℝ) + max 1 (1/μ + 1/ν))) := by
  let κ : ℝ := max 1 (1/μ+1/ν)
  apply IsBigO.of_bound (orderConstant U a ^ n *
    (((2*H/Real.pi + 2*H*Z + 1) * (preparationGrowth T₀ + 3)^κ) *
      orderConstant V b ^ κ))
  filter_upwards [show Set.Ioc 0 (Real.exp (-1)) ∈ 𝓝[>] (0 : ℝ) from
    Ioc_mem_nhdsGT (Real.exp_pos _)] with ε hε
  rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _),
    Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg (abs_nonneg _) _)]
  exact local_total_log_width n hT hH hZ hU hV ha hb hε.1 hε.2

/-- A fixed, explicitly realized number of anchor units preserves the law.
This does not assume that arbitrary activation families realize all anchors:
the architecture theorem must supply that fact separately. -/
theorem local_grid_with_anchors_epsilon_bigO (n anchors : ℕ)
    {T₀ H Z ν μ U V a b : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 < a) (hb : 0 < b) :
    (fun ε : ℝ =>
      let q := localAccuracyOrder U a (ε/2)
      let N := localAccuracyOrder V b (ε/2)
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ ⊕ Fin anchors) : ℝ))
      =O[𝓝[>] (0 : ℝ)]
        (fun ε : ℝ => |Real.log ε| ^ ((n : ℝ) + max 1 (1/μ + 1/ν))) := by
  let κ : ℝ := max 1 (1/μ+1/ν)
  let C : ℝ := orderConstant U a ^ n *
    (((2*H/Real.pi + 2*H*Z + 1) * (preparationGrowth T₀ + 3)^κ) * orderConstant V b ^ κ)
  apply IsBigO.of_bound (C + anchors)
  filter_upwards [show Set.Ioc 0 (Real.exp (-1)) ∈ 𝓝[>] (0 : ℝ) from
    Ioc_mem_nhdsGT (Real.exp_pos _)] with ε hε
  rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _),
    Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg (abs_nonneg _) _)]
  simp only [Fintype.card_sum, Fintype.card_fin, Nat.cast_add]
  have hw := local_total_log_width (ν := ν) (μ := μ) n hT hH hZ hU hV ha hb hε.1 hε.2
  have hl : 1 ≤ |Real.log ε| := by
    have hh := Real.log_le_log hε.1 hε.2
    rw [Real.log_exp] at hh
    rw [abs_of_nonpos (by linarith)]
    linarith
  have hp : 0 ≤ (n : ℝ) + κ := by dsimp [κ]; positivity
  have hr : 1 ≤ |Real.log ε| ^ ((n : ℝ) + κ) := Real.one_le_rpow hl hp
  have hc := mul_le_mul_of_nonneg_left hr (Nat.cast_nonneg anchors : (0 : ℝ) ≤ anchors)
  change _ ≤ (C + anchors) * |Real.log ε| ^ ((n : ℝ) + κ)
  change _ ≤ C * |Real.log ε| ^ ((n : ℝ) + κ) at hw
  nlinarith

end QIRadon.LocalScaling
