import QIRadon.Scalar.LocalAlgebraic

/-! Halo-inclusive scaling for the actual locally polynomial-approximable,
real-sample construction. The exponent is kappa/s; no analytic target data or
global target Fourier moment is assumed. -/
noncomputable section
open Filter Asymptotics
open scoped Real Topology
namespace QIRadon.LocalAlgebraicScaling
open QIRadon.LocalProfileRate QIRadon.LocalAlgebraic QIRadon.ProfilePreparation

/-- Actual finite row width at the selected integer algebraic order, including
all halos: O(epsilon^(-kappa/s)). The fixed constant C is chosen from the actual
construction error constant; using max(1,C) handles zero targets as well. -/
theorem local_algebraic_grid_epsilon_bigO {T₀ H Z ν μ C s : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) (hC : 0 < C) (hs : 0 < s) :
    (fun ε : ℝ =>
      let N := ⌈ScalarScaling.algebraicCutoff C s ε⌉₊
      ((localGrid (2*N) T₀ H Z ν μ).card:ℝ))
      =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => ε^(-(max 1 (1/μ+1/ν)/s))) := by
  let κ : ℝ := max 1 (1/μ+1/ν)
  let D : ℝ := (2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^κ
  have hκ : 0 ≤ κ := le_trans zero_le_one (le_max_left _ _)
  have horder := MultidimensionalScaling.algebraic_direction_order_bigO hC hs
  have hnonneg : 0 ≤ᶠ[𝓝[>] (0:ℝ)] (fun ε : ℝ => ε^(-(1/s))) := by
    filter_upwards [eventually_mem_nhdsWithin] with ε hε
    exact Real.rpow_nonneg (le_of_lt hε) _
  have hNpow := IsBigO.rpow hκ hnonneg (horder.const_mul_left 2)
  have hgrid : (fun ε : ℝ =>
      ((localGrid (2*⌈ScalarScaling.algebraicCutoff C s ε⌉₊) T₀ H Z ν μ).card:ℝ))
      =O[𝓝[>] (0:ℝ)] (fun ε : ℝ => (2*(⌈ScalarScaling.algebraicCutoff C s ε⌉₊:ℝ))^κ) := by
    apply IsBigO.of_bound D
    filter_upwards with ε
    have hN : 0 < ⌈ScalarScaling.algebraicCutoff C s ε⌉₊ := Nat.ceil_pos.mpr (Real.exp_pos _)
    have hh := local_analytic_grid_width (ν := ν) (μ := μ) hT
      (show 0 < 2*⌈ScalarScaling.algebraicCutoff C s ε⌉₊ by omega) hH hZ
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _), Real.norm_eq_abs,
      abs_of_nonneg (Real.rpow_nonneg (by positivity) _)]
    simpa only [Nat.cast_mul, Nat.cast_ofNat, D, κ] using hh
  have hh := hgrid.trans hNpow
  apply hh.congr'
  · filter_upwards with ε
    rfl
  · filter_upwards [eventually_mem_nhdsWithin] with ε hε
    have hε0 : 0 < ε := hε
    rw [← Real.rpow_mul hε0.le]
    congr 1
    dsimp [κ]
    ring

/-- A separately realized fixed anchor overhead preserves the same actual
width exponent. This theorem counts those units; it does not assume their
realizability for an arbitrary activation. -/
theorem local_algebraic_grid_with_anchors_epsilon_bigO (anchors : ℕ)
    {T₀ H Z ν μ C s : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) (hC : 0 < C) (hs : 0 < s) :
    (fun ε : ℝ =>
      let N := ⌈ScalarScaling.algebraicCutoff C s ε⌉₊
      (Fintype.card (localGrid (2*N) T₀ H Z ν μ ⊕ Fin anchors):ℝ))
      =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => ε^(-(max 1 (1/μ+1/ν)/s))) := by
  have hcount : (fun ε : ℝ =>
      let N := ⌈ScalarScaling.algebraicCutoff C s ε⌉₊
      (Fintype.card (localGrid (2*N) T₀ H Z ν μ ⊕ Fin anchors):ℝ))
      =O[𝓝[>] (0:ℝ)] (fun ε : ℝ =>
        ((localGrid (2*⌈ScalarScaling.algebraicCutoff C s ε⌉₊) T₀ H Z ν μ).card:ℝ)) := by
    apply IsBigO.of_bound (1+(anchors:ℝ))
    filter_upwards with ε
    have hpos : 1 ≤ (localGrid (2*⌈ScalarScaling.algebraicCutoff C s ε⌉₊) T₀ H Z ν μ).card := by
      unfold localGrid
      rw [ScalarRate.symmetric_grid_card]
      omega
    have hposR : (1:ℝ) ≤ (localGrid (2*⌈ScalarScaling.algebraicCutoff C s ε⌉₊) T₀ H Z ν μ).card := by
      exact_mod_cast hpos
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _),
      Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
    simp only [Fintype.card_sum, Fintype.card_coe, Fintype.card_fin, Nat.cast_add]
    nlinarith [Nat.cast_nonneg anchors (α := ℝ)]
  exact hcount.trans (local_algebraic_grid_epsilon_bigO hT hH hZ hC hs)

end QIRadon.LocalAlgebraicScaling
