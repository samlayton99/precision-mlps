import Mathlib

/-! The actual radial ball-mixture cutoff: exact domain agreement, finite support,
and magnitude bounds. Smoothness/holomorphic profile estimates are not assumed here. -/

noncomputable section
open MeasureTheory Set Metric
open scoped ContDiff

namespace QIRadon.Radon

/-- A specified mixture of ball indicators, exactly as in the boundary construction. -/
def radialCutoff (w : ℝ → ℝ) (R₀ R r : ℝ) : ℝ :=
  ∫ a in Set.Icc R₀ R, if r ≤ a then w a else 0

/-- Inside the reserved radius, every ball in the mixture includes the point. -/
theorem radialCutoff_eq_one {w : ℝ → ℝ} {R₀ R r : ℝ}
    (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1) (hr : r ≤ R₀) :
    radialCutoff w R₀ R r = 1 := by
  unfold radialCutoff
  rw [← hnormalized]
  apply setIntegral_congr_fun measurableSet_Icc
  intro a ha
  simp only [if_pos (hr.trans ha.1)]

/-- Outside the largest ball the cutoff is identically zero. -/
theorem radialCutoff_eq_zero {w : ℝ → ℝ} {R₀ R r : ℝ} (hr : R < r) :
    radialCutoff w R₀ R r = 0 := by
  unfold radialCutoff
  have hz : (∫ a in Set.Icc R₀ R, if r ≤ a then w a else 0) =
      ∫ a in Set.Icc R₀ R, (0 : ℝ) := by
    apply setIntegral_congr_fun measurableSet_Icc
    intro a ha
    simp only [if_neg (not_le.mpr (ha.2.trans_lt hr))]
  rw [hz, integral_zero]

/-- A nonnegative radius density produces a nonnegative cutoff. -/
theorem radialCutoff_nonneg {w : ℝ → ℝ} {R₀ R r : ℝ}
    (hw : ∀ a ∈ Set.Icc R₀ R, 0 ≤ w a) :
    0 ≤ radialCutoff w R₀ R r := by
  apply integral_nonneg_of_ae
  filter_upwards [ae_restrict_mem measurableSet_Icc] with a ha
  split_ifs <;> first | exact hw a ha | exact le_refl _

/-- Normalization ensures the cutoff never amplifies the target. -/
theorem radialCutoff_le_one {w : ℝ → ℝ} {R₀ R r : ℝ}
    (hi : IntegrableOn w (Set.Icc R₀ R))
    (hw : ∀ a ∈ Set.Icc R₀ R, 0 ≤ w a)
    (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1) :
    radialCutoff w R₀ R r ≤ 1 := by
  rw [← hnormalized]
  apply integral_mono_of_nonneg
  · filter_upwards [ae_restrict_mem measurableSet_Icc] with a ha
    split_ifs <;> first | exact hw a ha | exact le_refl _
  · exact hi
  · filter_upwards [ae_restrict_mem measurableSet_Icc] with a ha
    split_ifs <;> first | exact hw a ha | exact le_refl _

/-- The ball-mixture cutoff is an ordinary tail integral when the supplied radius
density vanishes outside its specified finite interval. -/
theorem radialCutoff_eq_tail {w : ℝ → ℝ} {R₀ R : ℝ}
    (hi : Integrable w) (hsupp : Function.support w ⊆ Set.Icc R₀ R) (r : ℝ) :
    radialCutoff w R₀ R r = ∫ a in r..R, w a := by
  have hwhole : radialCutoff w R₀ R r = ∫ a in Set.Ioi r, w a := by
    unfold radialCutoff
    rw [setIntegral_eq_integral_of_forall_compl_eq_zero]
    · change (∫ a : ℝ, (Set.Ici r).indicator w a) = _
      rw [integral_indicator measurableSet_Ici, integral_Ici_eq_integral_Ioi]
    · intro a ha
      have hwzero : w a = 0 := by
        by_contra hne
        exact ha (hsupp hne)
      simp [hwzero]
  have houter : (∫ a in Set.Ioi R, w a) = 0 := by
    apply integral_eq_zero_of_ae
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with a ha
    by_contra hne
    have hm := hsupp hne
    exact not_le.mpr ha hm.2
  have hsplit := intervalIntegral.integral_Ioi_sub_Ioi' hi.integrableOn hi.integrableOn
    (a := r) (b := R)
  rw [houter, sub_zero] at hsplit
  exact hwhole.trans hsplit

/-- Smoothness of the specified cutoff follows from FTC, not from assuming an
analytic or smooth profile oracle. -/
theorem radialCutoff_contDiff {w : ℝ → ℝ} {R₀ R : ℝ}
    (hi : Integrable w) (hsupp : Function.support w ⊆ Set.Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) : ContDiff ℝ ∞ (radialCutoff w R₀ R) := by
  have hd : ∀ r : ℝ, HasDerivAt (fun r => ∫ a in r..R, w a) (-w r) r := by
    intro r
    exact intervalIntegral.integral_hasDerivAt_left (hw.continuous.intervalIntegrable r R)
      hw.continuous.aestronglyMeasurable.stronglyMeasurableAtFilter hw.continuous.continuousAt
  have htail : ContDiff ℝ ∞ (fun r => ∫ a in r..R, w a) := by
    rw [contDiff_infty_iff_deriv]
    refine ⟨fun r => (hd r).differentiableAt, ?_⟩
    have heq : deriv (fun r => ∫ a in r..R, w a) = fun r => -w r := by
      funext r
      exact (hd r).deriv
    rw [heq]
    exact hw.neg
  have heq : radialCutoff w R₀ R = fun r => ∫ a in r..R, w a := by
    funext r
    exact radialCutoff_eq_tail hi hsupp r
  rw [heq]
  exact htail

variable {V : Type*} [NormedAddCommGroup V]

/-- The extension is specified by finite radius integration, not an arbitrary
whole-space analytic continuation. -/
def radialExtension (f : V → ℝ) (w : ℝ → ℝ) (R₀ R : ℝ) (x : V) : ℝ :=
  f x * radialCutoff w R₀ R ‖x‖

/-- The extension agrees exactly with the supplied target on the inner ball. -/
theorem radialExtension_eq {f : V → ℝ} {w : ℝ → ℝ} {R₀ R : ℝ}
    (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1) {x : V} (hx : ‖x‖ ≤ R₀) :
    radialExtension f w R₀ R x = f x := by
  simp [radialExtension, radialCutoff_eq_one hnormalized hx]

/-- No target values outside the finite outer ball affect the extension. -/
theorem radialExtension_eq_zero {f : V → ℝ} {w : ℝ → ℝ} {R₀ R : ℝ}
    {x : V} (hx : R < ‖x‖) : radialExtension f w R₀ R x = 0 := by
  simp [radialExtension, radialCutoff_eq_zero hx]

/-- The actual support is contained in the explicitly chosen outer ball. -/
theorem radialExtension_support_subset (f : V → ℝ) (w : ℝ → ℝ) (R₀ R : ℝ) :
    Function.support (radialExtension f w R₀ R) ⊆ closedBall (0 : V) R := by
  intro x hx
  by_contra hout
  have hn : R < ‖x‖ := by simpa only [mem_closedBall_zero_iff, not_le] using hout
  exact hx (radialExtension_eq_zero hn)

/-- The extension has compact support on finite-dimensional normed spaces. -/
theorem radialExtension_hasCompactSupport [ProperSpace V]
    (f : V → ℝ) (w : ℝ → ℝ) (R₀ R : ℝ) :
    HasCompactSupport (radialExtension f w R₀ R) := by
  apply HasCompactSupport.of_support_subset_isCompact (isCompact_closedBall (0 : V) R)
  exact radialExtension_support_subset f w R₀ R

/-- The extension retains the real-domain magnitude bound supplied for the target. -/
theorem radialExtension_abs_le {f : V → ℝ} {w : ℝ → ℝ} {R₀ R B : ℝ}
    (hB : 0 ≤ B) (hf : ∀ x, ‖x‖ ≤ R → |f x| ≤ B)
    (hi : IntegrableOn w (Set.Icc R₀ R))
    (hw : ∀ a ∈ Set.Icc R₀ R, 0 ≤ w a)
    (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1) (x : V) :
    |radialExtension f w R₀ R x| ≤ B := by
  by_cases hx : ‖x‖ ≤ R
  · rw [radialExtension, abs_mul, abs_of_nonneg (radialCutoff_nonneg hw)]
    calc
      |f x| * radialCutoff w R₀ R ‖x‖ ≤ |f x| * 1 :=
        mul_le_mul_of_nonneg_left (radialCutoff_le_one hi hw hnormalized) (abs_nonneg _)
      _ ≤ B := by simpa using hf x hx
  · rw [radialExtension_eq_zero (lt_of_not_ge hx), abs_zero]
    exact hB

/-- The radial cutoff is smooth even at the origin: it is locally constant there. -/
theorem radialCutoff_norm_contDiff [InnerProductSpace ℝ V]
    {w : ℝ → ℝ} {R₀ R : ℝ} (hR₀ : 0 < R₀)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Set.Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1) :
    ContDiff ℝ ∞ (fun x : V => radialCutoff w R₀ R ‖x‖) := by
  apply contDiff_iff_contDiffAt.mpr
  intro x
  by_cases hx : x = 0
  · subst x
    apply (contDiffAt_const : ContDiffAt ℝ ∞ (fun _ : V => (1 : ℝ)) 0).congr_of_eventuallyEq
    filter_upwards [Metric.ball_mem_nhds (0 : V) hR₀] with y hy
    have hnorm : ‖y‖ < R₀ := by simpa only [mem_ball_zero_iff] using hy
    exact radialCutoff_eq_one hnormalized hnorm.le
  · exact (radialCutoff_contDiff hi hsupp hw).contDiffAt.comp x
      (contDiffAt_id.norm ℝ hx)

/-- Local smoothness of the target around the closed outer ball is sufficient;
outside that neighborhood the extension is identically zero. -/
theorem radialExtension_contDiff [InnerProductSpace ℝ V]
    {f : V → ℝ} {w : ℝ → ℝ} {R₀ R : ℝ} {U : Set V}
    (hR₀ : 0 < R₀) (hU : IsOpen U) (hcover : closedBall (0 : V) R ⊆ U)
    (hf : ContDiffOn ℝ ∞ f U)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Set.Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1) :
    ContDiff ℝ ∞ (radialExtension f w R₀ R) := by
  have hcut := radialCutoff_norm_contDiff (V := V) hR₀ hi hsupp hw hnormalized
  apply contDiff_iff_contDiffAt.mpr
  intro x
  by_cases hx : x ∈ U
  · exact (hf.contDiffAt (hU.mem_nhds hx)).mul hcut.contDiffAt
  · have hout : x ∉ closedBall (0 : V) R := fun h => hx (hcover h)
    have hn : R < ‖x‖ := by simpa only [mem_closedBall_zero_iff, not_le] using hout
    apply (contDiffAt_const : ContDiffAt ℝ ∞ (fun _ : V => (0 : ℝ)) x).congr_of_eventuallyEq
    have hnear : {y : V | R < ‖y‖} ∈ nhds x :=
      (isOpen_lt continuous_const continuous_norm).mem_nhds hn
    filter_upwards [hnear] with y hy
    exact radialExtension_eq_zero hy

/-- The prescribed radius-mixture extension is an actual Schwartz function under
local target smoothness and a smooth finite radius density. -/
def radialExtensionSchwartz [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
    {f : V → ℝ} {w : ℝ → ℝ} {R₀ R : ℝ} {U : Set V}
    (hR₀ : 0 < R₀) (hU : IsOpen U) (hcover : closedBall (0 : V) R ⊆ U)
    (hf : ContDiffOn ℝ ∞ f U)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Set.Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1) :
    SchwartzMap V ℝ :=
  (radialExtension_hasCompactSupport f w R₀ R).toSchwartzMap
    (radialExtension_contDiff hR₀ hU hcover hf hi hsupp hw hnormalized)

end QIRadon.Radon
