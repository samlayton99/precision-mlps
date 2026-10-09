import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog
import Mathlib.Analysis.Calculus.LocalExtr.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Allocation of a certified approximation error budget

This module proves optimization statements for explicit analytic and algebraic
error certificates. These results concern the continuous certificate, not a
lower bound over all neural-network representations. Angular and scalar
approximation theorems must supply the positive constants independently.
-/

namespace QIRadon.Allocation

open Set Filter

/-- A differentiable objective decreasing to `c` and increasing after `c` has a
strict global minimum at `c` on the specified open interval. -/
theorem strict_min_of_deriv_sign {f : ℝ → ℝ} {a b c : ℝ}
    (hc : c ∈ Ioo a b)
    (hd : ∀ x ∈ Ioo a b, DifferentiableAt ℝ f x)
    (hneg : ∀ x ∈ Ioo a c, deriv f x < 0)
    (hpos : ∀ x ∈ Ioo c b, 0 < deriv f x) :
    ∀ y ∈ Ioo a b, y ≠ c → f c < f y := by
  intro y hy hne
  rcases lt_or_gt_of_ne hne with hyc | hcy
  · have hanti : StrictAntiOn f (Icc y c) := by
      apply strictAntiOn_of_deriv_neg (convex_Icc y c)
      · intro x hx
        exact (hd x ⟨lt_of_lt_of_le hy.1 hx.1, lt_of_le_of_lt hx.2 hc.2⟩).continuousAt.continuousWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        exact hneg x ⟨lt_trans hy.1 hx.1, hx.2⟩
    exact hanti ⟨le_rfl, hyc.le⟩ ⟨hyc.le, le_rfl⟩ hyc
  · have hmono : StrictMonoOn f (Icc c y) := by
      apply strictMonoOn_of_deriv_pos (convex_Icc c y)
      · intro x hx
        exact (hd x ⟨lt_of_lt_of_le hc.1 hx.1, lt_of_le_of_lt hx.2 hy.2⟩).continuousAt.continuousWithinAt
      · intro x hx
        rw [interior_Icc] at hx
        exact hpos x ⟨hx.1, lt_trans hx.2 hy.2⟩
    exact hmono ⟨le_rfl, hcy.le⟩ ⟨hcy.le, le_rfl⟩ hcy

/-- Logarithm of the algebraic width objective, up to a positive constant.
`A = (d-1)/t` and `B = 1/s`; `x` is the angular error budget. -/
noncomputable def algebraicLogCost (A B ε x : ℝ) : ℝ :=
  -A * Real.log x - B * Real.log (ε - x)

/-- The actual positive algebraic width objective. -/
noncomputable def algebraicCost (A B ε x : ℝ) : ℝ :=
  Real.exp (algebraicLogCost A B ε x)

/-- Closed form optimum of the algebraic budget allocation. -/
noncomputable def algebraicSplit (A B ε : ℝ) : ℝ := A * ε / (A + B)

theorem algebraicSplit_mem {A B ε : ℝ} (hA : 0 < A) (hB : 0 < B) (hε : 0 < ε) :
    algebraicSplit A B ε ∈ Ioo 0 ε := by
  have hAB : 0 < A + B := add_pos hA hB
  constructor
  · exact div_pos (mul_pos hA hε) hAB
  · unfold algebraicSplit
    apply (div_lt_iff₀ hAB).2
    nlinarith

theorem hasDerivAt_algebraicLogCost {A B ε x : ℝ} (hx : x ∈ Ioo 0 ε) :
    HasDerivAt (algebraicLogCost A B ε)
      (-A / x + B / (ε - x)) x := by
  have hx0 : x ≠ 0 := ne_of_gt hx.1
  have hes : ε - x ≠ 0 := ne_of_gt (sub_pos.mpr hx.2)
  have h1 := ((hasDerivAt_id x).log hx0).const_mul (-A)
  have h2 := (((hasDerivAt_const x ε).sub (hasDerivAt_id x)).log hes).const_mul B
  convert! h1.sub h2 using 1
  dsimp [algebraicLogCost]
  ring

/-- The algebraic optimizer is a strict GLOBAL minimizer, not just stationary. -/
theorem algebraicSplit_strict_min {A B ε : ℝ}
    (hA : 0 < A) (hB : 0 < B) (hε : 0 < ε) :
    ∀ x ∈ Ioo 0 ε, x ≠ algebraicSplit A B ε →
      algebraicCost A B ε (algebraicSplit A B ε) < algebraicCost A B ε x := by
  have hc := algebraicSplit_mem hA hB hε
  have hAB : 0 < A + B := add_pos hA hB
  have hsign : ∀ x ∈ Ioo 0 ε,
      deriv (algebraicLogCost A B ε) x =
        ((A + B) * x - A * ε) / (x * (ε - x)) := by
    intro x hx
    rw [(hasDerivAt_algebraicLogCost hx).deriv]
    have hx0 : x ≠ 0 := ne_of_gt hx.1
    have hes : ε - x ≠ 0 := ne_of_gt (sub_pos.mpr hx.2)
    field_simp
    ring
  have hmin : ∀ x ∈ Ioo 0 ε, x ≠ algebraicSplit A B ε →
      algebraicLogCost A B ε (algebraicSplit A B ε) < algebraicLogCost A B ε x := by
    apply strict_min_of_deriv_sign hc
    · intro x hx
      exact (hasDerivAt_algebraicLogCost hx).differentiableAt
    · intro x hx
      have hxe : x < ε := lt_trans hx.2 hc.2
      rw [hsign x ⟨hx.1, hxe⟩]
      apply div_neg_of_neg_of_pos
      · have ht : x * (A + B) < A * ε := (lt_div_iff₀ hAB).mp hx.2
        nlinarith
      · exact mul_pos hx.1 (sub_pos.mpr hxe)
    · intro x hx
      have hx0 : 0 < x := lt_trans hc.1 hx.1
      rw [hsign x ⟨hx0, hx.2⟩]
      apply div_pos
      · have ht : A * ε < x * (A + B) := (div_lt_iff₀ hAB).mp hx.1
        nlinarith
      · exact mul_pos hx0 (sub_pos.mpr hx.2)
  intro x hx hne
  exact Real.exp_lt_exp.mpr (hmin x hx hne)

/-- The closed-form optimal angular/scalar budget ratio. -/
theorem algebraicSplit_ratio {A B ε : ℝ}
    (hA : 0 < A) (hB : 0 < B) (hε : 0 < ε) :
    algebraicSplit A B ε / (ε - algebraicSplit A B ε) = A / B := by
  have hAB : A + B ≠ 0 := ne_of_gt (add_pos hA hB)
  have he : ε ≠ 0 := ne_of_gt hε
  have hb : B ≠ 0 := ne_of_gt hB
  have hs : ε - algebraicSplit A B ε ≠ 0 :=
    ne_of_gt (sub_pos.mpr (algebraicSplit_mem hA hB hε).2)
  unfold algebraicSplit at *
  field_simp [hAB, he, hb, hs]
  ring


/-- Logarithm of the ratio, written as a difference to expose derivatives. -/
noncomputable def logRatio (U x : ℝ) : ℝ := Real.log U - Real.log x

theorem logRatio_eq {U x : ℝ} (hU : 0 < U) (hx : 0 < x) :
    logRatio U x = Real.log (U / x) :=
  (Real.log_div hU.ne' hx.ne').symm

theorem hasDerivAt_logRatio {U x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (logRatio U) (-1 / x) x := by
  convert! (hasDerivAt_const x (Real.log U)).sub ((hasDerivAt_id x).log hx) using 1
  dsimp [logRatio]
  ring

theorem logRatio_gt_one {U ε x : ℝ} (hx : x ∈ Ioo 0 ε)
    (hU : 1 < logRatio U ε) : 1 < logRatio U x := by
  have hl : Real.log x < Real.log ε := Real.log_lt_log hx.1 hx.2
  dsimp [logRatio] at *
  linarith

/-- Logarithm of the analytic width certificate. Here `p = d-1` and `κ`
is the scalar logarithmic-width exponent. -/
noncomputable def analyticLogCost (p κ U V ε x : ℝ) : ℝ :=
  p * Real.log (logRatio U x) + κ * Real.log (logRatio V (ε - x))

/-- The positive analytic width objective, up to fixed factors independent of x. -/
noncomputable def analyticCost (p κ U V ε x : ℝ) : ℝ :=
  Real.exp (analyticLogCost p κ U V ε x)

/-- The stationarity balance extended continuously to both budget endpoints. -/
noncomputable def analyticBalance (p κ U V ε x : ℝ) : ℝ :=
  κ * (x * Real.log U - x * Real.log x) -
  p * ((ε - x) * Real.log V - (ε - x) * Real.log (ε - x))

theorem analyticBalance_eq (p κ U V ε x : ℝ) :
    analyticBalance p κ U V ε x =
      κ * x * logRatio U x - p * (ε - x) * logRatio V (ε - x) := by
  unfold analyticBalance logRatio
  ring

theorem continuous_analyticBalance (p κ U V ε : ℝ) :
    Continuous (analyticBalance p κ U V ε) := by
  have hc : Continuous (fun x : ℝ => ε - x) := continuous_const.sub continuous_id
  exact ((continuous_id.mul continuous_const).sub Real.continuous_mul_log).const_mul κ |>.sub
    (((hc.mul continuous_const).sub (Real.continuous_mul_log.comp hc)).const_mul p)

theorem hasDerivAt_analyticBalance {p κ U V ε x : ℝ} (hx : x ∈ Ioo 0 ε) :
    HasDerivAt (analyticBalance p κ U V ε)
      (κ * (logRatio U x - 1) + p * (logRatio V (ε - x) - 1)) x := by
  have hx0 : x ≠ 0 := ne_of_gt hx.1
  have hs0 : ε - x ≠ 0 := ne_of_gt (sub_pos.mpr hx.2)
  have hc := (hasDerivAt_const x ε).sub (hasDerivAt_id x)
  have h1 := (((hasDerivAt_id x).mul_const (Real.log U)).sub
    (Real.hasDerivAt_mul_log hx0)).const_mul κ
  have h2 := ((hc.mul_const (Real.log V)).sub
    ((Real.hasDerivAt_mul_log hs0).comp x hc)).const_mul p
  convert! h1.sub h2 using 1
  dsimp [analyticBalance, logRatio]
  ring

theorem analyticBalance_strictMonoOn {p κ U V ε : ℝ}
    (hp : 0 < p) (hκ : 0 < κ)
    (hU : 1 < logRatio U ε) (hV : 1 < logRatio V ε) :
    StrictMonoOn (analyticBalance p κ U V ε) (Icc 0 ε) := by
  apply strictMonoOn_of_deriv_pos (convex_Icc 0 ε)
    (continuous_analyticBalance p κ U V ε).continuousOn
  intro x hx
  rw [interior_Icc] at hx
  rw [(hasDerivAt_analyticBalance hx).deriv]
  have hs : ε - x ∈ Ioo 0 ε := ⟨sub_pos.mpr hx.2, by linarith [hx.1]⟩
  exact add_pos (mul_pos hκ (sub_pos.mpr (logRatio_gt_one hx hU)))
    (mul_pos hp (sub_pos.mpr (logRatio_gt_one hs hV)))

/-- Existence AND uniqueness of the analytic stationarity root in the high-accuracy
regime. No stationary-point equation is assumed. -/
theorem existsUnique_analyticBalance_zero {p κ U V ε : ℝ}
    (hp : 0 < p) (hκ : 0 < κ) (hε : 0 < ε)
    (hU : 1 < logRatio U ε) (hV : 1 < logRatio V ε) :
    ∃! x, x ∈ Ioo 0 ε ∧ analyticBalance p κ U V ε x = 0 := by
  have hl : analyticBalance p κ U V ε 0 < 0 := by
    rw [analyticBalance_eq]
    simp only [mul_zero, sub_zero]
    have : 0 < p * ε * logRatio V ε := mul_pos (mul_pos hp hε) (lt_trans zero_lt_one hV)
    linarith
  have hr : 0 < analyticBalance p κ U V ε ε := by
    rw [analyticBalance_eq]
    simp only [sub_self, mul_zero, zero_mul, sub_zero]
    exact mul_pos (mul_pos hκ hε) (lt_trans zero_lt_one hU)
  obtain ⟨x, hx, hzero⟩ := intermediate_value_Ioo hε.le
    (continuous_analyticBalance p κ U V ε).continuousOn ⟨hl, hr⟩
  refine ⟨x, ⟨hx, hzero⟩, ?_⟩
  intro y hy
  exact (analyticBalance_strictMonoOn hp hκ hU hV).injOn
    ⟨hy.1.1.le, hy.1.2.le⟩ ⟨hx.1.le, hx.2.le⟩ (hy.2.trans hzero.symm)

/-- Direct differentiation of the analytic objective. -/
theorem hasDerivAt_analyticLogCost {p κ U V ε x : ℝ}
    (hx : x ∈ Ioo 0 ε)
    (hUx : 0 < logRatio U x) (hVs : 0 < logRatio V (ε - x)) :
    HasDerivAt (analyticLogCost p κ U V ε)
      (-p / (x * logRatio U x) + κ / ((ε - x) * logRatio V (ε - x))) x := by
  have hc := (hasDerivAt_const x ε).sub (hasDerivAt_id x)
  have h1 := ((hasDerivAt_logRatio (U := U) (ne_of_gt hx.1)).log hUx.ne').const_mul p
  have h2 := (((hasDerivAt_logRatio (U := V) (ne_of_gt (sub_pos.mpr hx.2))).comp x hc).log hVs.ne').const_mul κ
  convert! h1.add h2 using 1
  dsimp [analyticLogCost]
  field_simp
  ring

/-- Fermat's theorem applied to the actual analytic cost's logarithm gives the
approved angular/scalar stationarity equation. -/
theorem analytic_stationarity_of_localMin {p κ U V ε x : ℝ}
    (hx : x ∈ Ioo 0 ε)
    (hUx : 0 < logRatio U x) (hVs : 0 < logRatio V (ε - x))
    (hmin : IsLocalMin (analyticLogCost p κ U V ε) x) :
    κ * x * logRatio U x = p * (ε - x) * logRatio V (ε - x) := by
  have heq := hmin.hasDerivAt_eq_zero (hasDerivAt_analyticLogCost hx hUx hVs)
  have hx0 := ne_of_gt hx.1
  have hs0 := ne_of_gt (sub_pos.mpr hx.2)
  have hU0 := ne_of_gt hUx
  have hV0 := ne_of_gt hVs
  field_simp at heq
  nlinarith [heq]


/-- Derivative sign equals the balance-function sign because its denominator is
strictly positive on the admissible interval. -/
theorem deriv_analyticLogCost_eq_balance {p κ U V ε x : ℝ}
    (hx : x ∈ Ioo 0 ε)
    (hUx : 0 < logRatio U x) (hVs : 0 < logRatio V (ε - x)) :
    deriv (analyticLogCost p κ U V ε) x =
      analyticBalance p κ U V ε x /
        ((x * logRatio U x) * ((ε - x) * logRatio V (ε - x))) := by
  rw [(hasDerivAt_analyticLogCost hx hUx hVs).deriv, analyticBalance_eq]
  have hx0 := ne_of_gt hx.1
  have hs0 := ne_of_gt (sub_pos.mpr hx.2)
  have hU0 := ne_of_gt hUx
  have hV0 := ne_of_gt hVs
  field_simp
  ring

/-- The unique zero of the balance is the strict global optimum of the positive
analytic width objective. -/
theorem analytic_root_strict_min {p κ U V ε c : ℝ}
    (hp : 0 < p) (hκ : 0 < κ)
    (hU : 1 < logRatio U ε) (hV : 1 < logRatio V ε)
    (hc : c ∈ Ioo 0 ε) (hc0 : analyticBalance p κ U V ε c = 0) :
    ∀ x ∈ Ioo 0 ε, x ≠ c →
      analyticCost p κ U V ε c < analyticCost p κ U V ε x := by
  have hm := analyticBalance_strictMonoOn hp hκ hU hV
  have hdom : ∀ x ∈ Ioo 0 ε,
      0 < logRatio U x ∧ 0 < logRatio V (ε - x) := by
    intro x hx
    have hs : ε - x ∈ Ioo 0 ε := ⟨sub_pos.mpr hx.2, by linarith [hx.1]⟩
    exact ⟨lt_trans zero_lt_one (logRatio_gt_one hx hU),
      lt_trans zero_lt_one (logRatio_gt_one hs hV)⟩
  have hmin : ∀ x ∈ Ioo 0 ε, x ≠ c →
      analyticLogCost p κ U V ε c < analyticLogCost p κ U V ε x := by
    apply strict_min_of_deriv_sign hc
    · intro x hx
      exact (hasDerivAt_analyticLogCost hx (hdom x hx).1 (hdom x hx).2).differentiableAt
    · intro x hx
      have hxe : x < ε := lt_trans hx.2 hc.2
      have hd := hdom x ⟨hx.1, hxe⟩
      rw [deriv_analyticLogCost_eq_balance ⟨hx.1, hxe⟩ hd.1 hd.2]
      apply div_neg_of_neg_of_pos
      · have hlt := hm ⟨hx.1.le, hxe.le⟩ ⟨hc.1.le, hc.2.le⟩ hx.2
        simpa [hc0] using hlt
      · exact mul_pos (mul_pos hx.1 hd.1) (mul_pos (sub_pos.mpr hxe) hd.2)
    · intro x hx
      have hx0 : 0 < x := lt_trans hc.1 hx.1
      have hd := hdom x ⟨hx0, hx.2⟩
      rw [deriv_analyticLogCost_eq_balance ⟨hx0, hx.2⟩ hd.1 hd.2]
      apply div_pos
      · have hlt := hm ⟨hc.1.le, hc.2.le⟩ ⟨hx0.le, hx.2.le⟩ hx.1
        simpa [hc0] using hlt
      · exact mul_pos (mul_pos hx0 hd.1) (mul_pos (sub_pos.mpr hx.2) hd.2)
  intro x hx hne
  exact Real.exp_lt_exp.mpr (hmin x hx hne)

/-- Main analytic allocation theorem: a unique strictly globally minimizing
angular error budget exists, and satisfies the claimed balance equation. -/
theorem existsUnique_analytic_optimizer {p κ U V ε : ℝ}
    (hp : 0 < p) (hκ : 0 < κ) (hε : 0 < ε)
    (hU : 1 < logRatio U ε) (hV : 1 < logRatio V ε) :
    ∃! c, c ∈ Ioo 0 ε ∧
      κ * c * logRatio U c = p * (ε - c) * logRatio V (ε - c) ∧
      ∀ x ∈ Ioo 0 ε, x ≠ c →
        analyticCost p κ U V ε c < analyticCost p κ U V ε x := by
  obtain ⟨c, hc, huniq⟩ := existsUnique_analyticBalance_zero hp hκ hε hU hV
  refine ⟨c, ⟨hc.1, ?_, analytic_root_strict_min hp hκ hU hV hc.1 hc.2⟩, ?_⟩
  · have heq := hc.2
    rw [analyticBalance_eq, sub_eq_zero] at heq
    exact heq
  · intro y hy
    apply huniq y
    exact ⟨hy.1, by rw [analyticBalance_eq, hy.2.1, sub_self]⟩

/-- A fixed positive multiplier in the width does not change its minimizer. -/
theorem mul_cost_preserves_strict_min {f : ℝ → ℝ} {S : Set ℝ} {c C : ℝ}
    (hC : 0 < C) (hmin : ∀ x ∈ S, x ≠ c → f c < f x) :
    ∀ x ∈ S, x ≠ c → C * f c < C * f x := by
  intro x hx hne
  exact mul_lt_mul_of_pos_left (hmin x hx hne) hC


/-- The log-coordinate algebraic objective is exactly the power-law cost. -/
theorem algebraicCost_eq_rpow {A B ε x : ℝ} (hx : x ∈ Ioo 0 ε) :
    algebraicCost A B ε x = x ^ (-A) * (ε - x) ^ (-B) := by
  unfold algebraicCost
  rw [Real.rpow_def_of_pos hx.1, Real.rpow_def_of_pos (sub_pos.mpr hx.2),
    ← Real.exp_add]
  congr 1
  unfold algebraicLogCost
  ring

/-- Likewise the analytic objective is the product of powers of log resolutions. -/
theorem analyticCost_eq_rpow {p κ U V ε x : ℝ}
    (hUx : 0 < logRatio U x) (hVs : 0 < logRatio V (ε - x)) :
    analyticCost p κ U V ε x = (logRatio U x) ^ p * (logRatio V (ε - x)) ^ κ := by
  unfold analyticCost
  rw [Real.rpow_def_of_pos hUx, Real.rpow_def_of_pos hVs, ← Real.exp_add]
  congr 1
  unfold analyticLogCost
  ring

/-- Upward integer rounding can only decrease an exponential error certificate. -/
theorem exponential_error_rounding {A a n : ℝ} (hA : 0 ≤ A) (ha : 0 ≤ a) :
    A * Real.exp (-a * (⌈n⌉₊ : ℝ)) ≤ A * Real.exp (-a * n) := by
  apply mul_le_mul_of_nonneg_left _ hA
  apply Real.exp_le_exp.mpr
  exact mul_le_mul_of_nonpos_left (Nat.le_ceil n) (neg_nonpos.mpr ha)

/-- Upward rounding of finitely many nonnegative resolutions adds less than one
unit per coordinate, with the true linear halo cost multiplier retained. -/
theorem linear_cost_rounding {ι : Type*} [Fintype ι] [Nonempty ι]
    (c n : ι → ℝ) (hc : ∀ i, 0 < c i) (hn : ∀ i, 0 ≤ n i) :
    (∑ i, c i * (⌈n i⌉₊ : ℝ)) < (∑ i, c i * n i) + ∑ i, c i := by
  calc
    (∑ i, c i * (⌈n i⌉₊ : ℝ)) < ∑ i, c i * (n i + 1) := by
      apply Finset.sum_lt_sum_of_nonempty Finset.univ_nonempty
      intro i _
      exact mul_lt_mul_of_pos_left (Nat.ceil_lt_add_one (hn i)) (hc i)
    _ = (∑ i, c i * n i) + ∑ i, c i := by
      simp_rw [mul_add, mul_one, Finset.sum_add_distrib]

/-- A scalar exponential lies above its tangent at an arbitrary reference point. -/
theorem exponential_supporting_line (D a y z : ℝ) (hD : 0 ≤ D) :
    D * Real.exp (-a * z) +
      (a * D * Real.exp (-a * z)) * (z - y) ≤ D * Real.exp (-a * y) := by
  have ht := Real.add_one_le_exp (a * (z - y))
  have hm := mul_le_mul_of_nonneg_left ht (mul_nonneg hD (Real.exp_pos (-a * z)).le)
  have he : Real.exp (-a * z) * Real.exp (a * (z - y)) = Real.exp (-a * y) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc
    D * Real.exp (-a * z) + (a * D * Real.exp (-a * z)) * (z - y)
        = (D * Real.exp (-a * z)) * (a * (z - y) + 1) := by ring
    _ ≤ (D * Real.exp (-a * z)) * Real.exp (a * (z - y)) := hm
    _ = D * Real.exp (-a * y) := by rw [mul_assoc, he]

/-- Total inverse-rate-weighted linear cost. -/
noncomputable def allocationMass {ι : Type*} [Fintype ι] (c a : ι → ℝ) : ℝ :=
  ∑ i, c i / a i

/-- Explicit optimal resolution on each spoke when no baseline constraint is
active. Angular quadrature weights are included in `D i`. -/
noncomputable def spokeSplit {ι : Type*} [Fintype ι]
    (D a c : ι → ℝ) (ε : ℝ) (i : ι) : ℝ :=
  Real.log (D i * a i * allocationMass c a / (c i * ε)) / a i

theorem allocationMass_pos {ι : Type*} [Fintype ι] [Nonempty ι]
    {c a : ι → ℝ} (hc : ∀ i, 0 < c i) (ha : ∀ i, 0 < a i) :
    0 < allocationMass c a := by
  apply Finset.sum_pos
  · intro i _
    exact div_pos (hc i) (ha i)
  · exact Finset.univ_nonempty

/-- The explicit allocation realizes the predicted per-spoke error shares. -/
theorem spokeSplit_error {ι : Type*} [Fintype ι] [Nonempty ι]
    {D a c : ι → ℝ} {ε : ℝ}
    (hD : ∀ i, 0 < D i) (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i)
    (hε : 0 < ε) (i : ι) :
    D i * Real.exp (-a i * spokeSplit D a c ε i) =
      ε * (c i / a i) / allocationMass c a := by
  have hS := allocationMass_pos hc ha
  have hr : 0 < D i * a i * allocationMass c a / (c i * ε) :=
    div_pos (mul_pos (mul_pos (hD i) (ha i)) hS) (mul_pos (hc i) hε)
  unfold spokeSplit
  rw [show -a i * (Real.log (D i * a i * allocationMass c a / (c i * ε)) / a i) =
    -Real.log (D i * a i * allocationMass c a / (c i * ε)) by field_simp [ne_of_gt (ha i)]]
  rw [Real.exp_neg, Real.exp_log hr]
  field_simp [ne_of_gt (ha i), ne_of_gt (hD i), ne_of_gt (hc i), ne_of_gt hε, ne_of_gt hS]

theorem spokeSplit_error_sum {ι : Type*} [Fintype ι] [Nonempty ι]
    {D a c : ι → ℝ} {ε : ℝ}
    (hD : ∀ i, 0 < D i) (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i)
    (hε : 0 < ε) :
    (∑ i, D i * Real.exp (-a i * spokeSplit D a c ε i)) = ε := by
  simp_rw [spokeSplit_error hD ha hc hε]
  simp only [div_eq_mul_inv]
  rw [← Finset.sum_mul, ← Finset.mul_sum]
  change ε * allocationMass c a * (allocationMass c a)⁻¹ = ε
  exact mul_inv_cancel_right₀ (ne_of_gt (allocationMass_pos hc ha)) ε

/-- The explicit per-spoke allocation globally minimizes linear neuron cost
among ALL allocations meeting the exponential error budget. -/
theorem spokeSplit_global_min {ι : Type*} [Fintype ι] [Nonempty ι]
    {D a c : ι → ℝ} {ε : ℝ}
    (hD : ∀ i, 0 < D i) (ha : ∀ i, 0 < a i) (hc : ∀ i, 0 < c i)
    (hε : 0 < ε) (n : ι → ℝ)
    (hn : (∑ i, D i * Real.exp (-a i * n i)) ≤ ε) :
    (∑ i, c i * spokeSplit D a c ε i) ≤ ∑ i, c i * n i := by
  have hS := allocationMass_pos hc ha
  have ht : ∀ i,
      D i * Real.exp (-a i * spokeSplit D a c ε i) +
        (ε / allocationMass c a) *
          (c i * spokeSplit D a c ε i - c i * n i) ≤
        D i * Real.exp (-a i * n i) := by
    intro i
    have hb := exponential_supporting_line (D i) (a i) (n i)
      (spokeSplit D a c ε i) (hD i).le
    have hm : a i * D i * Real.exp (-a i * spokeSplit D a c ε i) =
        ε / allocationMass c a * c i := by
      rw [mul_assoc, spokeSplit_error hD ha hc hε]
      field_simp [ne_of_gt (ha i)]
    convert! hb using 1
    rw [hm]
    ring
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun i _ => ht i)
  rw [Finset.sum_add_distrib, spokeSplit_error_sum hD ha hc hε,
    ← Finset.mul_sum, Finset.sum_sub_distrib] at hs
  have he : 0 < ε / allocationMass c a := div_pos hε hS
  nlinarith [hs.trans hn]


/-- An explicit finite-accuracy estimate behind the asymptotic split. The
constant does not depend on epsilon or the minimizing fraction. -/
theorem analytic_fraction_balance_bound {p κ U V ε θ : ℝ}
    (hp : 0 < p) (hκ : 0 < κ) (hε : 0 < ε)
    (hθ : θ ∈ Ioo 0 1)
    (hb : κ * θ * logRatio U (ε * θ) =
      p * (1 - θ) * logRatio V (ε * (1 - θ))) :
    |((p + κ) * θ - p) * Real.log ε| ≤
      κ * |Real.log U| + p * |Real.log V| + κ + p := by
  have hu : 0 < 1 - θ := sub_pos.mpr hθ.2
  have hu1 : 1 - θ < 1 := by linarith [hθ.1]
  have hθlog : |θ * Real.log θ| ≤ 1 := by
    simpa [mul_comm] using (Real.abs_log_mul_self_lt θ hθ.1 hθ.2.le).le
  have hulog : |(1 - θ) * Real.log (1 - θ)| ≤ 1 := by
    simpa [mul_comm] using (Real.abs_log_mul_self_lt (1 - θ) hu hu1.le).le
  have hθU : |θ * Real.log U| ≤ |Real.log U| := by
    rw [abs_mul, abs_of_pos hθ.1]
    exact mul_le_of_le_one_left (abs_nonneg _) hθ.2.le
  have huV : |(1 - θ) * Real.log V| ≤ |Real.log V| := by
    rw [abs_mul, abs_of_pos hu]
    exact mul_le_of_le_one_left (abs_nonneg _) hu1.le
  have hid : ((p + κ) * θ - p) * Real.log ε =
      κ * (θ * Real.log U) - p * ((1 - θ) * Real.log V) -
      κ * (θ * Real.log θ) + p * ((1 - θ) * Real.log (1 - θ)) := by
    unfold logRatio at hb
    rw [Real.log_mul hε.ne' hθ.1.ne', Real.log_mul hε.ne' hu.ne'] at hb
    nlinarith [hb]
  rw [hid]
  have htri (a b : ℝ) : |a - b| ≤ |a| + |b| := by
    simpa using abs_sub_le a 0 b
  calc
    |κ * (θ * Real.log U) - p * ((1 - θ) * Real.log V) -
        κ * (θ * Real.log θ) + p * ((1 - θ) * Real.log (1 - θ))|
        ≤ |κ * (θ * Real.log U) - p * ((1 - θ) * Real.log V) -
            κ * (θ * Real.log θ)| + |p * ((1 - θ) * Real.log (1 - θ))| :=
      abs_add_le _ _
    _ ≤ (|κ * (θ * Real.log U) - p * ((1 - θ) * Real.log V)| +
          |κ * (θ * Real.log θ)|) + |p * ((1 - θ) * Real.log (1 - θ))| := by
      linarith [htri (κ * (θ * Real.log U) - p * ((1 - θ) * Real.log V))
        (κ * (θ * Real.log θ))]
    _ ≤ ((|κ * (θ * Real.log U)| + |p * ((1 - θ) * Real.log V)|) +
          |κ * (θ * Real.log θ)|) + |p * ((1 - θ) * Real.log (1 - θ))| := by
      linarith [htri (κ * (θ * Real.log U)) (p * ((1 - θ) * Real.log V))]
    _ = κ * |θ * Real.log U| + p * |(1 - θ) * Real.log V| +
        κ * |θ * Real.log θ| + p * |(1 - θ) * Real.log (1 - θ)| := by
      simp only [abs_mul, abs_of_pos hp, abs_of_pos hκ]
    _ ≤ κ * |Real.log U| + p * |Real.log V| + κ + p := by
      have h1 := mul_le_mul_of_nonneg_left hθU hκ.le
      have h2 := mul_le_mul_of_nonneg_left huV hp.le
      have h3 := mul_le_mul_of_nonneg_left hθlog hκ.le
      have h4 := mul_le_mul_of_nonneg_left hulog hp.le
      linarith

/-- Any analytic optimizer obeys an explicit O(1/|log epsilon|) fraction error.
In particular its angular share approaches p/(p+κ), without an assumption
that the two logarithms already have the same leading term. -/
theorem analytic_optimizer_fraction_bound {p κ U V ε x : ℝ}
    (hp : 0 < p) (hκ : 0 < κ) (hε : 0 < ε) (hε1 : ε < 1)
    (hx : x ∈ Ioo 0 ε)
    (hb : analyticBalance p κ U V ε x = 0) :
    |x / ε - p / (p + κ)| ≤
      (κ * |Real.log U| + p * |Real.log V| + κ + p) /
        ((p + κ) * |Real.log ε|) := by
  have hs : 0 < p + κ := add_pos hp hκ
  have hlog : 0 < |Real.log ε| := abs_pos.mpr (ne_of_lt (Real.log_neg hε hε1))
  have hθ : x / ε ∈ Ioo 0 1 :=
    ⟨div_pos hx.1 hε, (div_lt_one hε).mpr hx.2⟩
  have hid1 : ε * (x / ε) = x := mul_div_cancel₀ x hε.ne'
  have hid2 : ε * (1 - x / ε) = ε - x := by rw [mul_sub, mul_one, hid1]
  have hb' : κ * (x / ε) * logRatio U (ε * (x / ε)) =
      p * (1 - x / ε) * logRatio V (ε * (1 - x / ε)) := by
    rw [hid1, hid2]
    rw [analyticBalance_eq, sub_eq_zero] at hb
    apply (mul_left_cancel₀ hε.ne')
    calc
      ε * (κ * (x / ε) * logRatio U x) = κ * x * logRatio U x := by
        field_simp
      _ = p * (ε - x) * logRatio V (ε - x) := hb
      _ = ε * (p * (1 - x / ε) * logRatio V (ε - x)) := by
        field_simp
  have he := analytic_fraction_balance_bound hp hκ hε hθ hb'
  have hid : ((p + κ) * (x / ε) - p) * Real.log ε =
      (p + κ) * (x / ε - p / (p + κ)) * Real.log ε := by
    field_simp
  rw [hid, abs_mul, abs_mul, abs_of_pos hs] at he
  apply (le_div_iff₀ (mul_pos hs hlog)).mpr
  nlinarith [he]


/-- Active-set allocation in error coordinates. A baseline resolution places
an upper cap on the remaining certified error of each spoke. -/
noncomputable def waterfillError {ι : Type*} (cap weight : ι → ℝ) (t : ℝ) (i : ι) : ℝ :=
  min (cap i) (t * weight i)

/-- The scalar water-level equation has a positive finite solution whenever
the desired budget is below the sum of baseline errors. -/
theorem exists_waterfill_level {ι : Type*} [Fintype ι] [Nonempty ι]
    {cap weight : ι → ℝ} {ε : ℝ}
    (hc : ∀ i, 0 < cap i) (hw : ∀ i, 0 < weight i)
    (hε : 0 < ε) (hbudget : ε < ∑ i, cap i) :
    ∃ t, 0 < t ∧ (∑ i, waterfillError cap weight t i) = ε := by
  classical
  let T : ℝ := ∑ i, cap i / weight i
  have hT : 0 < T := by
    exact Finset.sum_pos (fun i _ => div_pos (hc i) (hw i)) Finset.univ_nonempty
  have hcont : Continuous (fun t : ℝ => ∑ i, waterfillError cap weight t i) := by
    apply continuous_finsetSum
    intro i _
    exact continuous_const.min (continuous_id.mul_const (weight i))
  have hzero : (∑ i, waterfillError cap weight 0 i) = 0 := by
    apply Finset.sum_eq_zero
    intro i _
    simp [waterfillError, min_eq_right (hc i).le]
  have hupper : (∑ i, waterfillError cap weight T i) = ∑ i, cap i := by
    apply Finset.sum_congr rfl
    intro i _
    unfold waterfillError
    apply min_eq_left
    have hi : cap i / weight i ≤ T := by
      exact Finset.single_le_sum (fun j _ => (div_pos (hc j) (hw j)).le)
        (Finset.mem_univ i)
    exact (div_le_iff₀ (hw i)).mp hi
  have hl : (∑ i, waterfillError cap weight 0 i) < ε := by rwa [hzero]
  have hr : ε < ∑ i, waterfillError cap weight T i := by rwa [hupper]
  obtain ⟨t, ht, he⟩ := intermediate_value_Ioo hT.le hcont.continuousOn ⟨hl, hr⟩
  exact ⟨t, ht.1, he⟩

/-- Explicit scalar comparison underlying optimal clipped allocation. -/
theorem waterfill_log_comparison {cap w t D e : ℝ}
    (hc : 0 < cap) (hw : 0 < w) (ht : 0 < t) (hD : 0 < D)
    (he : 0 < e) (hec : e ≤ cap) :
    (min cap (t * w) - e) / t ≤
      w * Real.log (D / e) - w * Real.log (D / min cap (t * w)) := by
  let z := min cap (t * w)
  have hz : 0 < z := lt_min hc (mul_pos ht hw)
  have hlog := Real.log_le_sub_one_of_pos (div_pos he hz)
  rw [Real.log_div he.ne' hz.ne'] at hlog
  have hlog' : w * (1 - e / z) ≤
      w * Real.log (D / e) - w * Real.log (D / z) := by
    rw [Real.log_div hD.ne' he.ne', Real.log_div hD.ne' hz.ne']
    have hm := mul_le_mul_of_nonneg_left hlog hw.le
    nlinarith
  have hlin : (z - e) / t ≤ w * (1 - e / z) := by
    rcases le_total cap (t * w) with hct | htc
    · have hzc : z = cap := min_eq_left hct
      rw [hzc]
      have hgap : 0 ≤ cap - e := sub_nonneg.mpr hec
      have hratio : 1 / t ≤ w / cap := by
        apply (div_le_div_iff₀ ht hc).mpr
        nlinarith
      have hm := mul_le_mul_of_nonneg_right hratio hgap
      convert! hm using 1 <;> field_simp
    · have hzt : z = t * w := min_eq_right htc
      rw [hzt]
      apply le_of_eq
      field_simp
  exact hlin.trans hlog'

/-- The water-filling allocation is a GLOBAL optimizer, including active
baseline constraints. The objective is the exact linear resolution cost after
the substitution `n_i = log(D_i/e_i) / a_i`, with `weight_i = c_i/a_i`. -/
theorem waterfill_global_min {ι : Type*} [Fintype ι]
    {cap weight D : ι → ℝ} {t ε : ℝ}
    (hc : ∀ i, 0 < cap i) (hw : ∀ i, 0 < weight i) (hD : ∀ i, 0 < D i)
    (ht : 0 < t) (hsum : (∑ i, waterfillError cap weight t i) = ε)
    (e : ι → ℝ) (he : ∀ i, 0 < e i) (hec : ∀ i, e i ≤ cap i)
    (hbudget : (∑ i, e i) ≤ ε) :
    (∑ i, weight i * Real.log (D i / waterfillError cap weight t i)) ≤
      ∑ i, weight i * Real.log (D i / e i) := by
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
    waterfill_log_comparison (hc i) (hw i) ht (hD i) (he i) (hec i))
  change (∑ i, (waterfillError cap weight t i - e i) / t) ≤ _ at hs
  have hleft : (∑ i, (waterfillError cap weight t i - e i) / t) =
      (ε - ∑ i, e i) / t := by
    simp only [div_eq_mul_inv]
    rw [← Finset.sum_mul, Finset.sum_sub_distrib, hsum]
  rw [hleft, Finset.sum_sub_distrib] at hs
  have hnonneg : 0 ≤ (ε - ∑ i, e i) / t := div_nonneg (sub_nonneg.mpr hbudget) ht.le
  simpa only [waterfillError] using (sub_nonneg.mp (hnonneg.trans hs))

/-- Error coordinates translate back to the claimed log-resolution formula and
respect every baseline resolution. -/
theorem waterfill_resolution_ge_baseline {D a c μ n₀ : ℝ}
    (hD : 0 < D) (ha : 0 < a) (hc : 0 < c) (hμ : 0 < μ) :
    Real.log (D / min (D * Real.exp (-a * n₀)) (c / (μ * a))) / a =
      max n₀ (Real.log (μ * a * D / c) / a) := by
  have hbase : 0 < D * Real.exp (-a * n₀) := mul_pos hD (Real.exp_pos _)
  have hfree : 0 < c / (μ * a) := div_pos hc (mul_pos hμ ha)
  have hlogbase : Real.log (D / (D * Real.exp (-a * n₀))) / a = n₀ := by
    rw [Real.log_div hD.ne' hbase.ne', Real.log_mul hD.ne' (Real.exp_ne_zero _), Real.log_exp]
    field_simp
    ring
  have hlogfree : Real.log (D / (c / (μ * a))) / a = Real.log (μ * a * D / c) / a := by
    congr 2
    field_simp
  rcases le_total (D * Real.exp (-a * n₀)) (c / (μ * a)) with hbf | hfb
  · rw [min_eq_left hbf, hlogbase]
    apply (max_eq_left _).symm
    rw [← hlogbase, ← hlogfree]
    apply (div_le_div_iff_of_pos_right ha).mpr
    apply Real.log_le_log (div_pos hD hfree)
    exact div_le_div_of_nonneg_left hD.le hbase hbf
  · rw [min_eq_right hfb, hlogfree]
    apply (max_eq_right _).symm
    rw [← hlogbase, ← hlogfree]
    apply (div_le_div_iff_of_pos_right ha).mpr
    apply Real.log_le_log (div_pos hD hbase)
    exact div_le_div_of_nonneg_left hD.le hfree hfb


/-- The unconstrained minimum clips to any closed feasible subinterval. This
handles lower resolution constraints after translating them to budget bounds. -/
theorem clipped_min_of_deriv_sign {f : ℝ → ℝ} {a b c l u : ℝ}
    (hc : c ∈ Ioo a b) (hl : a < l) (hu : u < b) (hlu : l ≤ u)
    (hd : ∀ x ∈ Ioo a b, DifferentiableAt ℝ f x)
    (hneg : ∀ x ∈ Ioo a c, deriv f x < 0)
    (hpos : ∀ x ∈ Ioo c b, 0 < deriv f x) :
    ∀ y ∈ Icc l u, f (max l (min c u)) ≤ f y := by
  intro y hy
  by_cases hcl : c ≤ l
  · rw [min_eq_left (hcl.trans hlu), max_eq_left hcl]
    rcases eq_or_lt_of_le hy.1 with hly | hly
    · rw [hly]
    · have hmono : StrictMonoOn f (Icc l y) := by
        apply strictMonoOn_of_deriv_pos (convex_Icc l y)
        · intro x hx
          exact (hd x ⟨lt_of_lt_of_le hl hx.1,
            lt_of_le_of_lt (hx.2.trans hy.2) hu⟩).continuousAt.continuousWithinAt
        · intro x hx
          rw [interior_Icc] at hx
          exact hpos x ⟨lt_of_le_of_lt hcl hx.1, lt_of_lt_of_le hx.2 (hy.2.trans hu.le)⟩
      exact (hmono ⟨le_rfl, hly.le⟩ ⟨hly.le, le_rfl⟩ hly).le
  · by_cases huc : u ≤ c
    · rw [min_eq_right huc, max_eq_right hlu]
      rcases eq_or_lt_of_le hy.2 with hyu | hyu
      · rw [hyu]
      · have hanti : StrictAntiOn f (Icc y u) := by
          apply strictAntiOn_of_deriv_neg (convex_Icc y u)
          · intro x hx
            exact (hd x ⟨lt_of_lt_of_le hl (hy.1.trans hx.1),
              lt_of_le_of_lt hx.2 hu⟩).continuousAt.continuousWithinAt
          · intro x hx
            rw [interior_Icc] at hx
            exact hneg x ⟨lt_of_lt_of_le hl (hy.1.trans hx.1.le), lt_of_lt_of_le hx.2 huc⟩
        exact (hanti ⟨le_rfl, hyu.le⟩ ⟨hyu.le, le_rfl⟩ hyu).le
    · have hlc : l < c := lt_of_not_ge hcl
      have hcu : c < u := lt_of_not_ge huc
      rw [min_eq_left hcu.le, max_eq_right hlc.le]
      by_cases hyc : y = c
      · rw [hyc]
      · exact (strict_min_of_deriv_sign hc hd hneg hpos y
          ⟨lt_of_lt_of_le hl hy.1, lt_of_le_of_lt hy.2 hu⟩ hyc).le

/-- The analytic optimum subject to a closed admissible budget interval is
obtained by clipping the unique unconstrained optimum to that interval. -/
theorem analytic_clipped_min {p κ U V ε c l u : ℝ}
    (hp : 0 < p) (hκ : 0 < κ)
    (hU : 1 < logRatio U ε) (hV : 1 < logRatio V ε)
    (hc : c ∈ Ioo 0 ε) (hc0 : analyticBalance p κ U V ε c = 0)
    (hl : 0 < l) (hu : u < ε) (hlu : l ≤ u) :
    ∀ y ∈ Icc l u,
      analyticCost p κ U V ε (max l (min c u)) ≤ analyticCost p κ U V ε y := by
  have hm := analyticBalance_strictMonoOn hp hκ hU hV
  have hdom : ∀ x ∈ Ioo 0 ε,
      0 < logRatio U x ∧ 0 < logRatio V (ε - x) := by
    intro x hx
    have hs : ε - x ∈ Ioo 0 ε := ⟨sub_pos.mpr hx.2, by linarith [hx.1]⟩
    exact ⟨lt_trans zero_lt_one (logRatio_gt_one hx hU),
      lt_trans zero_lt_one (logRatio_gt_one hs hV)⟩
  have hd : ∀ x ∈ Ioo 0 ε, DifferentiableAt ℝ (analyticLogCost p κ U V ε) x := by
    intro x hx
    exact (hasDerivAt_analyticLogCost hx (hdom x hx).1 (hdom x hx).2).differentiableAt
  have hn : ∀ x ∈ Ioo 0 c, deriv (analyticLogCost p κ U V ε) x < 0 := by
    intro x hx
    have hxe : x < ε := lt_trans hx.2 hc.2
    have hdx := hdom x ⟨hx.1, hxe⟩
    rw [deriv_analyticLogCost_eq_balance ⟨hx.1, hxe⟩ hdx.1 hdx.2]
    apply div_neg_of_neg_of_pos
    · have ht := hm ⟨hx.1.le, hxe.le⟩ ⟨hc.1.le, hc.2.le⟩ hx.2
      simpa [hc0] using ht
    · exact mul_pos (mul_pos hx.1 hdx.1) (mul_pos (sub_pos.mpr hxe) hdx.2)
  have hp' : ∀ x ∈ Ioo c ε, 0 < deriv (analyticLogCost p κ U V ε) x := by
    intro x hx
    have hx0 : 0 < x := lt_trans hc.1 hx.1
    have hdx := hdom x ⟨hx0, hx.2⟩
    rw [deriv_analyticLogCost_eq_balance ⟨hx0, hx.2⟩ hdx.1 hdx.2]
    apply div_pos
    · have ht := hm ⟨hc.1.le, hc.2.le⟩ ⟨hx0.le, hx.2.le⟩ hx.1
      simpa [hc0] using ht
    · exact mul_pos (mul_pos hx0 hdx.1) (mul_pos (sub_pos.mpr hx.2) hdx.2)
  intro y hy
  exact Real.exp_le_exp.mpr (clipped_min_of_deriv_sign hc hl hu hlu hd hn hp' y hy)

/-- Rounding a stretched-exponential resolution preserves feasibility. -/
theorem stretched_exponential_error_rounding {A a n ν : ℝ}
    (hA : 0 ≤ A) (ha : 0 ≤ a) (hn : 0 ≤ n) (hν : 0 ≤ ν) :
    A * Real.exp (-a * (⌈n⌉₊ : ℝ) ^ ν) ≤ A * Real.exp (-a * n ^ ν) := by
  apply mul_le_mul_of_nonneg_left _ hA
  apply Real.exp_le_exp.mpr
  exact mul_le_mul_of_nonpos_left (Real.rpow_le_rpow hn (Nat.le_ceil n) hν)
    (neg_nonpos.mpr ha)

/-- Counted integer tensor width after upward rounding, including any halo
already incorporated into H. This is an additive-resolution guarantee, not a
claim that componentwise rounding is the exact discrete optimum. -/
theorem rounded_product_width {q H : ℝ} (p : ℕ) (hq : 0 ≤ q) (hH : 0 ≤ H) :
    ((⌈q⌉₊ : ℝ) ^ p) * (⌈H⌉₊ : ℝ) ≤ (q + 1) ^ p * (H + 1) := by
  have hq' := (Nat.ceil_lt_add_one hq).le
  have hH' := (Nat.ceil_lt_add_one hH).le
  exact mul_le_mul (pow_le_pow_left₀ (Nat.cast_nonneg _) hq' p) hH'
    (Nat.cast_nonneg _) (pow_nonneg (by linarith) p)


/-- Formal asymptotic fraction for any selection of the proved analytic
optimizer (uniqueness makes the choice immaterial). -/
theorem analytic_optimizer_fraction_tendsto {p κ U V ε₀ : ℝ} {x : ℝ → ℝ}
    (hp : 0 < p) (hκ : 0 < κ) (hε₀ : 0 < ε₀)
    (hroot : ∀ ε ∈ Ioo 0 ε₀,
      x ε ∈ Ioo 0 ε ∧ analyticBalance p κ U V ε (x ε) = 0) :
    Filter.Tendsto (fun ε => x ε / ε) (nhdsWithin 0 (Ioi 0)) (nhds (p / (p + κ))) := by
  have hden : Filter.Tendsto (fun ε : ℝ => (p + κ) * |Real.log ε|)
      (nhdsWithin 0 (Ioi 0)) Filter.atTop := by
    exact (tendsto_abs_atBot_atTop.comp Real.tendsto_log_nhdsGT_zero).const_mul_atTop
      (add_pos hp hκ)
  have hzero := hden.const_div_atTop (κ * |Real.log U| + p * |Real.log V| + κ + p)
  apply tendsto_iff_dist_tendsto_zero.mpr
  apply squeeze_zero' (Filter.Eventually.of_forall (fun ε => dist_nonneg)) _ hzero
  filter_upwards [Ioo_mem_nhdsGT hε₀, Ioo_mem_nhdsGT (zero_lt_one : (0 : ℝ) < 1)] with ε hε hε1
  obtain ⟨hx, hb⟩ := hroot ε hε
  simpa only [Real.dist_eq] using analytic_optimizer_fraction_bound hp hκ hε.1 hε1.2 hx hb

/-- The asymptotic ANGULAR/SCALAR error-budget ratio is p/κ, with p=d-1. -/
theorem analytic_optimizer_ratio_tendsto {p κ U V ε₀ : ℝ} {x : ℝ → ℝ}
    (hp : 0 < p) (hκ : 0 < κ) (hε₀ : 0 < ε₀)
    (hroot : ∀ ε ∈ Ioo 0 ε₀,
      x ε ∈ Ioo 0 ε ∧ analyticBalance p κ U V ε (x ε) = 0) :
    Filter.Tendsto (fun ε => x ε / (ε - x ε)) (nhdsWithin 0 (Ioi 0)) (nhds (p / κ)) := by
  have hs := add_pos hp hκ
  have hf := analytic_optimizer_fraction_tendsto hp hκ hε₀ hroot
  have hlim : 1 - p / (p + κ) ≠ 0 := by
    apply ne_of_gt
    apply sub_pos.mpr
    apply (div_lt_one hs).mpr
    linarith
  have hd := hf.div (tendsto_const_nhds.sub hf) hlim
  have hid : (p / (p + κ)) / (1 - p / (p + κ)) = p / κ := by
    field_simp [hκ.ne', hs.ne', hlim]
    ring
  rw [hid] at hd
  apply hd.congr'
  filter_upwards [Ioo_mem_nhdsGT hε₀] with ε hε
  have he := hε.1.ne'
  have hx := (hroot ε hε).1
  have hsub : ε - x ε ≠ 0 := ne_of_gt (sub_pos.mpr hx.2)
  dsimp
  field_simp [he, hsub]

/-- The usual algebraic split after substituting dimension and regularity
exponents: p=d-1, angular order t, scalar order s. -/
theorem algebraic_geometry_ratio {p s t ε : ℝ}
    (hp : 0 < p) (hs : 0 < s) (ht : 0 < t) (hε : 0 < ε) :
    algebraicSplit (p / t) (1 / s) ε /
      (ε - algebraicSplit (p / t) (1 / s) ε) = p * s / t := by
  rw [algebraicSplit_ratio (div_pos hp ht) (div_pos zero_lt_one hs) hε]
  field_simp

end QIRadon.Allocation
