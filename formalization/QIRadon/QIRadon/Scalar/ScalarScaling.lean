import QIRadon.Scalar.ScalarRate
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

noncomputable section
open Filter Asymptotics
open scoped Topology

namespace QIRadon.ScalarScaling

/-- The actual symmetric-grid count in the cycles-frequency convention. -/
def gridWidth (T B L μ ν Z H : ℝ) : ℕ :=
  2 * ⌊(T + Z * L ^ (1 / μ) / (B / L ^ (1 / ν))) / (1 / (H * B))⌋₊ + 1

/-- The logarithmic halo is lower order than the passband cost for algebraic data. -/
theorem logarithmic_halo_bigO {s p : ℝ} (hs : 0 < s) (_hp : 0 ≤ p) :
    (fun B : ℝ => (max 1 ((s / 2) * Real.log (1 + B))) ^ p) =O[atTop]
      (fun B : ℝ => B) := by
  have hshift : Tendsto (fun B : ℝ => 1 + B) atTop atTop :=
    tendsto_atTop_add_const_left atTop 1 tendsto_id
  have hlog : (fun B : ℝ => (Real.log (1 + B)) ^ p) =o[atTop]
      (fun B : ℝ => 1 + B) := by
    simpa only [Function.comp_def, Real.rpow_one] using
      (isLittleO_log_rpow_rpow_atTop p (s := 1) (by norm_num)).comp_tendsto hshift
  have hgrowth : Tendsto (fun B : ℝ => (s / 2) * Real.log (1 + B)) atTop atTop :=
    (Real.tendsto_log_atTop.comp hshift).const_mul_atTop (half_pos hs)
  apply IsBigO.of_bound (2 * (s / 2) ^ p)
  filter_upwards [hlog.bound (show (0 : ℝ) < 1 by norm_num),
    hgrowth.eventually_ge_atTop 1, eventually_ge_atTop (1 : ℝ)] with B hb hmax hB
  have hlog0 : 0 ≤ Real.log (1 + B) := Real.log_nonneg (by linarith)
  rw [Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg (by positivity) _),
    Real.norm_eq_abs, abs_of_nonneg (by linarith : 0 ≤ B), max_eq_right hmax,
    Real.mul_rpow (half_pos hs).le hlog0]
  have hlogpow : (Real.log (1 + B))^p ≤ 1 + B := by
    simpa only [Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg hlog0 _),
      abs_of_nonneg (show 0 ≤ 1 + B by linarith), one_mul] using hb
  have hc : 0 ≤ (s / 2)^p := Real.rpow_nonneg (half_pos hs).le _
  calc
    _ ≤ (s / 2)^p * (1 + B) := mul_le_mul_of_nonneg_left hlogpow hc
    _ ≤ 2 * (s / 2)^p * B := by nlinarith

/-- An asymptotic statement about the actual floor-counted neuron grid, obtained
after the explicit finite inequality. The accuracy bound is separately proved
by `algebraic_scheduled_certificate`, with error O(B^(-s)). -/
theorem algebraic_gridWidth_bigO {T μ ν Z H s : ℝ}
    (hT : 0 ≤ T) (hμ : 0 < μ) (hν : 0 < ν)
    (hZ : 0 ≤ Z) (hH : 0 < H) (hs : 0 < s) :
    (fun B : ℝ => (gridWidth T B
      (max 1 ((s / 2) * Real.log (1 + B))) μ ν Z H : ℝ)) =O[atTop]
      (fun B : ℝ => B) := by
  have hp : 0 ≤ 1 / μ + 1 / ν := by positivity
  have hlog := logarithmic_halo_bigO hs hp
  obtain ⟨C, hC, hbound⟩ := hlog.exists_pos
  apply IsBigO.of_bound (2 * H * T + 2 * H * Z * C + 1)
  filter_upwards [hbound.bound, eventually_ge_atTop (1 : ℝ)] with B hb hB
  have hB0 : 0 < B := by linarith
  have hL : 1 ≤ max 1 ((s / 2) * Real.log (1 + B)) := le_max_left _ _
  have hw := ScalarRate.grid_width_cycles (μ := μ) (ν := ν) hT hB0
    (lt_of_lt_of_le zero_lt_one hL) hZ hH
  dsimp only at hw
  rw [ScalarRate.symmetric_grid_card] at hw
  push_cast at hw
  change 2 * (⌊_⌋₊ : ℝ) + 1 ≤ _ at hw
  have hlogb : (max 1 ((s / 2) * Real.log (1 + B))) ^ (1 / μ + 1 / ν) ≤ C * B := by
    simpa only [Real.norm_eq_abs,
      abs_of_nonneg (Real.rpow_nonneg (le_trans zero_le_one hL) _),
      abs_of_nonneg hB0.le] using hb
  rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _),
    Real.norm_eq_abs, abs_of_nonneg hB0.le]
  unfold gridWidth
  push_cast
  apply hw.trans
  have hm := mul_le_mul_of_nonneg_left hlogb (show 0 ≤ 2 * H * Z by positivity)
  nlinarith

/-- Cutoff actually used by the algebraic accuracy theorem. -/
def algebraicCutoff (C s ε : ℝ) : ℝ :=
  Real.exp (max 1 (Real.log (C / ε)) / s)

/-- At high accuracy the explicit cutoff has exactly the power-law form. -/
theorem algebraicCutoff_eventually_eq {C s : ℝ} (hC : 0 < C) (_hs : 0 < s) :
    (fun ε => algebraicCutoff C s ε) =ᶠ[𝓝[>] (0 : ℝ)]
      (fun ε => C ^ (1 / s) * ε ^ (-(1 / s))) := by
  have hratio : Tendsto (fun ε : ℝ => C / ε) (𝓝[>] 0) atTop := by
    simpa only [div_eq_mul_inv] using tendsto_inv_nhdsGT_zero.const_mul_atTop hC
  have hlog := Real.tendsto_log_atTop.comp hratio
  filter_upwards [hlog.eventually_ge_atTop 1,
    eventually_mem_nhdsWithin] with ε hεlog hε
  have hε0 : 0 < ε := hε
  dsimp only [Function.comp_apply] at hεlog
  unfold algebraicCutoff
  rw [max_eq_right hεlog]
  have hid : Real.exp (Real.log (C / ε) / s) = (C / ε) ^ (1 / s) := by
    rw [Real.rpow_def_of_pos (div_pos hC hε0)]
    congr 1
    ring
  rw [hid, Real.div_rpow hC.le hε0.le, Real.rpow_neg hε0.le]
  rfl

/-- The cutoff tends to infinity, so the proved grid asymptotic can be composed. -/
theorem algebraicCutoff_tendsto {C s : ℝ} (hC : 0 < C) (hs : 0 < s) :
    Tendsto (fun ε => algebraicCutoff C s ε) (𝓝[>] (0 : ℝ)) atTop := by
  have hpow : Tendsto (fun ε : ℝ => ε ^ (-(1 / s))) (𝓝[>] 0) atTop := by
    have ht := (tendsto_rpow_atTop (one_div_pos.mpr hs)).comp tendsto_inv_nhdsGT_zero
    apply ht.congr'
    filter_upwards [eventually_mem_nhdsWithin] with ε hε
    simp only [Function.comp_apply, Real.inv_rpow (le_of_lt hε), Real.rpow_neg (le_of_lt hε)]
  have ht := hpow.const_mul_atTop (Real.rpow_pos_of_pos hC (1 / s))
  exact ht.congr' (algebraicCutoff_eventually_eq hC hs).symm

/-- The requested algebraic neuron law for the actual grid as ε decreases to 0.
The constant C is the fixed positive target/kernel/anchor factor in the actual
accuracy theorem, not an assumed approximation constant for a numerical fit. -/
theorem algebraic_gridWidth_epsilon_bigO {T μ ν Z H s C : ℝ}
    (hT : 0 ≤ T) (hμ : 0 < μ) (hν : 0 < ν)
    (hZ : 0 ≤ Z) (hH : 0 < H) (hs : 0 < s) (hC : 0 < C) :
    (fun ε : ℝ =>
      let B := algebraicCutoff C s ε
      (gridWidth T B (max 1 ((s / 2) * Real.log (1 + B))) μ ν Z H : ℝ))
      =O[𝓝[>] (0 : ℝ)] (fun ε : ℝ => ε ^ (-(1 / s))) := by
  have hw := (algebraic_gridWidth_bigO hT hμ hν hZ hH hs).comp_tendsto
    (algebraicCutoff_tendsto hC hs)
  have hb := (algebraicCutoff_eventually_eq hC hs).isBigO.trans
    (isBigO_const_mul_self (C ^ (1 / s)) (fun ε : ℝ => ε ^ (-(1 / s))) (𝓝[>] 0))
  exact hw.trans hb

/-- The prescribed logarithmic accuracy order is controlled by |log ε|,
with a target-dependent constant made explicit. -/
theorem logarithmic_order_bound {C ε : ℝ} (hC : 0 < C)
    (hε : 0 < ε) (hεsmall : ε ≤ Real.exp (-1)) :
    max 1 (Real.log (C / ε)) ≤ (|Real.log C| + 1) * |Real.log ε| := by
  have hl := Real.log_le_log hε hεsmall
  rw [Real.log_exp] at hl
  have ha : |Real.log ε| = -Real.log ε := abs_of_nonpos (by linarith)
  rw [ha, Real.log_div hC.ne' hε.ne']
  apply max_le
  · nlinarith [abs_nonneg (Real.log C)]
  · nlinarith [le_abs_self (Real.log C), abs_nonneg (Real.log C)]

/-- The requested analytic neuron law for the actual floor-counted grid:
width O(|log ε|^κ), with κ=max(1,1/μ+1/ν). All halo units are included. -/
theorem analytic_gridWidth_epsilon_bigO {T σ μ ν Z H C : ℝ}
    (hT : 0 ≤ T) (hσ : 0 < σ) (hZ : 0 ≤ Z) (hH : 0 < H) (hC : 0 < C) :
    (fun ε : ℝ =>
      let L := max 1 (Real.log (C / ε))
      (gridWidth T (L / σ) L μ ν Z H : ℝ))
      =O[𝓝[>] (0 : ℝ)]
      (fun ε : ℝ => |Real.log ε| ^ max 1 (1 / μ + 1 / ν)) := by
  let κ : ℝ := max 1 (1 / μ + 1 / ν)
  let A : ℝ := 2 * H * T / σ + 2 * H * Z + 1
  have hA : 0 ≤ A := by dsimp [A]; positivity
  have hκ : 0 ≤ κ := le_trans zero_le_one (le_max_left _ _)
  apply IsBigO.of_bound (A * (|Real.log C| + 1) ^ κ)
  filter_upwards [show Set.Ioc 0 (Real.exp (-1)) ∈ 𝓝[>] (0 : ℝ) from
    Ioc_mem_nhdsGT (Real.exp_pos _)] with ε hε
  let L := max 1 (Real.log (C / ε))
  have hL : 1 ≤ L := le_max_left _ _
  have hw := ScalarRate.analytic_grid_width_cycles (μ := μ) (ν := ν)
    hT hσ hL hZ hH
  dsimp only at hw
  rw [ScalarRate.symmetric_grid_card] at hw
  push_cast at hw
  have hlo := logarithmic_order_bound hC hε.1 hε.2
  have hp := Real.rpow_le_rpow (le_trans zero_le_one hL) hlo hκ
  rw [Real.mul_rpow (by positivity : 0 ≤ |Real.log C| + 1) (abs_nonneg _)] at hp
  rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _),
    Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg (abs_nonneg _) _)]
  unfold gridWidth
  push_cast
  apply hw.trans
  have hm := mul_le_mul_of_nonneg_left hp hA
  simpa only [A, κ, mul_assoc] using hm

end QIRadon.ScalarScaling
