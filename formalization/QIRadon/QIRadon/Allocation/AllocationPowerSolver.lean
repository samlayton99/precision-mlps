import QIRadon.Allocation.AllocationPower
import Mathlib

/-! Construction of the general-power per-spoke optimizer by nested scalar
 root selection. The common multiplier is proved to exist from the requested
 total error budget; it is not supplied as a KKT hypothesis. -/

noncomputable section
open Set
open scoped BigOperators Real

namespace QIRadon.AllocationPower

/-- The unique clipped spoke root. Its defining equation uses a maximum only
 in log coordinates; the activation threshold is the minimum resolution. -/
def spokeResolution (κ a base : ℝ) (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base)
    (b : ℝ) : ℝ :=
  Classical.choose (existsUnique_spoke_root hκ ha hbase
    (le_max_left (logStationarity κ a base) b))

theorem spokeResolution_spec {κ a base : ℝ} (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base)
    (b : ℝ) :
    spokeResolution κ a base hκ ha hbase b ∈
      Icc base (max base ((max (logStationarity κ a base) b)/a)) ∧
    logStationarity κ a (spokeResolution κ a base hκ ha hbase b) =
      max (logStationarity κ a base) b :=
  (Classical.choose_spec (existsUnique_spoke_root hκ ha hbase
    (le_max_left (logStationarity κ a base) b))).1

theorem spokeResolution_ge_base {κ a base : ℝ} (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base)
    (b : ℝ) : base ≤ spokeResolution κ a base hκ ha hbase b :=
  (spokeResolution_spec hκ ha hbase b).1.1

theorem spokeResolution_eq_base {κ a base b : ℝ} (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base)
    (hb : b ≤ logStationarity κ a base) : spokeResolution κ a base hκ ha hbase b = base := by
  apply (logStationarity_strictMonoOn hκ ha).injOn
    (hbase.trans (spokeResolution_ge_base hκ ha hbase b)) hbase
  rw [(spokeResolution_spec hκ ha hbase b).2, max_eq_left hb]

theorem spokeResolution_mono {κ a base : ℝ} (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base) :
    Monotone (spokeResolution κ a base hκ ha hbase) := by
  intro b c hbc
  by_contra h
  have hn : spokeResolution κ a base hκ ha hbase c < spokeResolution κ a base hκ ha hbase b :=
    lt_of_not_ge h
  have hs := logStationarity_strictMonoOn hκ ha
    (hbase.trans (spokeResolution_ge_base hκ ha hbase c))
    (hbase.trans (spokeResolution_ge_base hκ ha hbase b)) hn
  rw [(spokeResolution_spec hκ ha hbase c).2, (spokeResolution_spec hκ ha hbase b).2] at hs
  exact (not_lt_of_ge (max_le_max le_rfl hbc)) hs

theorem logStationarity_strong_mono {κ a x y : ℝ} (hκ : 1 ≤ κ) (hx : 1 ≤ x) (hxy : x ≤ y) :
    a*(y-x) ≤ logStationarity κ a y-logStationarity κ a x := by
  have hx0 : 0 < x := zero_lt_one.trans_le hx
  have hlog : Real.log x ≤ Real.log y := Real.log_le_log hx0 hxy
  have h := mul_le_mul_of_nonneg_left hlog (sub_nonneg.mpr hκ)
  unfold logStationarity
  linarith

theorem spokeResolution_difference {κ a base b c : ℝ}
    (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base) (hbc : b ≤ c) :
    a*(spokeResolution κ a base hκ ha hbase c-spokeResolution κ a base hκ ha hbase b) ≤ c-b := by
  have h := logStationarity_strong_mono (a := a) hκ
    (hbase.trans (spokeResolution_ge_base hκ ha hbase b)) (spokeResolution_mono hκ ha hbase hbc)
  rw [(spokeResolution_spec hκ ha hbase c).2, (spokeResolution_spec hκ ha hbase b).2] at h
  have hmax : max (logStationarity κ a base) c-max (logStationarity κ a base) b ≤ c-b := by
    by_cases hb : logStationarity κ a base ≤ b
    · rw [max_eq_right hb, max_eq_right (hb.trans hbc)]
    · rw [max_eq_left (le_of_not_ge hb)]
      by_cases hc : logStationarity κ a base ≤ c
      · rw [max_eq_right hc]; linarith
      · rw [max_eq_left (le_of_not_ge hc)]; linarith
  exact h.trans hmax

theorem spokeResolution_abs_bound {κ a base : ℝ}
    (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base) (b c : ℝ) :
    |spokeResolution κ a base hκ ha hbase b-spokeResolution κ a base hκ ha hbase c| ≤
      (1/a)*|b-c| := by
  rcases le_total b c with hbc | hcb
  · have hm := spokeResolution_mono hκ ha hbase hbc
    rw [abs_of_nonpos (sub_nonpos.mpr hm), abs_of_nonpos (sub_nonpos.mpr hbc)]
    have h := spokeResolution_difference hκ ha hbase hbc
    have hd := (le_div_iff₀ ha).mpr (show
      (spokeResolution κ a base hκ ha hbase c-spokeResolution κ a base hκ ha hbase b)*a ≤ c-b by
      simpa only [mul_comm] using h)
    convert hd using 1 <;> first | rfl | ring
  · have hm := spokeResolution_mono hκ ha hbase hcb
    rw [abs_of_nonneg (sub_nonneg.mpr hm), abs_of_nonneg (sub_nonneg.mpr hcb)]
    have h := spokeResolution_difference hκ ha hbase hcb
    have hd := (le_div_iff₀ ha).mpr (show
      (spokeResolution κ a base hκ ha hbase b-spokeResolution κ a base hκ ha hbase c)*a ≤ b-c by
      simpa only [mul_comm] using h)
    convert hd using 1 <;> first | rfl | ring

theorem spokeResolution_continuous {κ a base : ℝ}
    (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base) :
    Continuous (spokeResolution κ a base hκ ha hbase) := by
  have hlip : LipschitzWith ⟨1/a, by positivity⟩ (spokeResolution κ a base hκ ha hbase) := by
    apply LipschitzWith.of_dist_le_mul
    intro b c
    exact spokeResolution_abs_bound hκ ha hbase b c
  exact hlip.continuous

theorem le_spokeResolution_of_log_le {κ a base b y : ℝ}
    (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base) (hy : base ≤ y)
    (hb : logStationarity κ a y ≤ b) : y ≤ spokeResolution κ a base hκ ha hbase b := by
  by_contra h
  have hn := lt_of_not_ge h
  have hs := logStationarity_strictMonoOn hκ ha
    (hbase.trans (spokeResolution_ge_base hκ ha hbase b)) (hbase.trans hy) hn
  rw [(spokeResolution_spec hκ ha hbase b).2] at hs
  exact (not_lt_of_ge (hb.trans (le_max_right _ _))) hs


/-- Positive data of the actual general-power cost and exponential error
 certificate. These fields are inputs, not optimization conclusions. -/
structure SpokeData (ι : Type*) where
  κ : ℝ
  c : ι → ℝ
  a : ι → ℝ
  D : ι → ℝ
  base : ι → ℝ
  power_ge_one : 1 ≤ κ
  cost_pos : ∀ i, 0 < c i
  rate_pos : ∀ i, 0 < a i
  error_pos : ∀ i, 0 < D i
  base_ge_one : ∀ i, 1 ≤ base i

namespace SpokeData

variable {ι : Type*} [Fintype ι] (p : SpokeData ι)

def logOffset (i : ι) : ℝ := Real.log (p.a i*p.D i/(p.κ*p.c i))

def resolution (t : ℝ) (i : ι) : ℝ :=
  spokeResolution p.κ (p.a i) (p.base i) p.power_ge_one (p.rate_pos i) (p.base_ge_one i)
    (t+p.logOffset i)

def baselineError : ℝ := ∑ i, p.D i*Real.exp (-p.a i*p.base i)

def totalError (t : ℝ) : ℝ := ∑ i, p.D i*Real.exp (-p.a i*p.resolution t i)

omit [Fintype ι] in
theorem resolution_ge_base (t : ℝ) (i : ι) : p.base i ≤ p.resolution t i :=
  spokeResolution_ge_base p.power_ge_one (p.rate_pos i) (p.base_ge_one i) _

omit [Fintype ι] in
theorem resolution_log (t : ℝ) (i : ι) :
    logStationarity p.κ (p.a i) (p.resolution t i) =
      max (logStationarity p.κ (p.a i) (p.base i)) (t+p.logOffset i) :=
  (spokeResolution_spec p.power_ge_one (p.rate_pos i) (p.base_ge_one i) _).2

omit [Fintype ι] in
theorem resolution_continuous (i : ι) : Continuous (fun t ↦ p.resolution t i) :=
  (spokeResolution_continuous p.power_ge_one (p.rate_pos i) (p.base_ge_one i)).comp
    (continuous_id.add_const (p.logOffset i))

omit [Fintype ι] in
theorem resolution_monotone (i : ι) : Monotone (fun t ↦ p.resolution t i) := by
  intro s t hst
  exact spokeResolution_mono p.power_ge_one (p.rate_pos i) (p.base_ge_one i)
    (add_le_add hst le_rfl)

theorem totalError_continuous : Continuous p.totalError := by
  apply continuous_finsetSum
  intro i _
  exact ((Real.continuous_exp.comp ((p.resolution_continuous i).const_mul (-p.a i))).const_mul (p.D i))

theorem totalError_antitone : Antitone p.totalError := by
  intro s t hst
  apply Finset.sum_le_sum
  intro i _
  apply mul_le_mul_of_nonneg_left ?_ (p.error_pos i).le
  apply Real.exp_le_exp.mpr
  exact mul_le_mul_of_nonpos_left (p.resolution_monotone i hst) (by linarith [p.rate_pos i])

def activationThreshold (i : ι) : ℝ := logStationarity p.κ (p.a i) (p.base i)-p.logOffset i

def lowerLog : ℝ := -1-∑ i, |p.activationThreshold i|

theorem lowerLog_lt_threshold (i : ι) : p.lowerLog < p.activationThreshold i := by
  have hi : |p.activationThreshold i| ≤ ∑ j, |p.activationThreshold j| :=
    Finset.single_le_sum (fun j _ ↦ abs_nonneg (p.activationThreshold j)) (Finset.mem_univ i)
  unfold lowerLog
  linarith [neg_abs_le (p.activationThreshold i)]

theorem resolution_lowerLog (i : ι) : p.resolution p.lowerLog i = p.base i := by
  apply spokeResolution_eq_base
  have h := p.lowerLog_lt_threshold i
  unfold activationThreshold at h
  linarith

theorem totalError_lowerLog : p.totalError p.lowerLog = p.baselineError := by
  unfold totalError baselineError
  simp_rw [p.resolution_lowerLog]

def targetResolution (ε : ℝ) (i : ι) : ℝ :=
  max (p.base i) (Real.log (2*(Fintype.card ι:ℝ)*p.D i/ε)/p.a i)

def upperThreshold (ε : ℝ) (i : ι) : ℝ :=
  logStationarity p.κ (p.a i) (p.targetResolution ε i)-p.logOffset i

def upperLog (ε : ℝ) : ℝ := 1+∑ i, |p.upperThreshold ε i|

theorem upperThreshold_lt_upperLog (ε : ℝ) (i : ι) : p.upperThreshold ε i < p.upperLog ε := by
  have hi : |p.upperThreshold ε i| ≤ ∑ j, |p.upperThreshold ε j| :=
    Finset.single_le_sum (fun j _ ↦ abs_nonneg (p.upperThreshold ε j)) (Finset.mem_univ i)
  unfold upperLog
  linarith [le_abs_self (p.upperThreshold ε i)]

theorem lowerLog_lt_upperLog (ε : ℝ) : p.lowerLog < p.upperLog ε := by
  have hl : 0 ≤ ∑ i, |p.activationThreshold i| := Finset.sum_nonneg (fun _ _ ↦ abs_nonneg _)
  have hu : 0 ≤ ∑ i, |p.upperThreshold ε i| := Finset.sum_nonneg (fun _ _ ↦ abs_nonneg _)
  unfold lowerLog upperLog
  linarith

theorem targetResolution_le_upper_resolution (ε : ℝ) (i : ι) :
    p.targetResolution ε i ≤ p.resolution (p.upperLog ε) i := by
  apply le_spokeResolution_of_log_le p.power_ge_one (p.rate_pos i) (p.base_ge_one i)
    (le_max_left _ _)
  have h := p.upperThreshold_lt_upperLog ε i
  unfold upperThreshold at h
  change logStationarity p.κ (p.a i) (p.targetResolution ε i) ≤ p.upperLog ε+p.logOffset i
  linarith

theorem targetResolution_error_le [Nonempty ι] {ε : ℝ} (hε : 0 < ε) (i : ι) :
    p.D i*Real.exp (-p.a i*p.targetResolution ε i) ≤ ε/(2*(Fintype.card ι:ℝ)) := by
  have hcard : 0 < (Fintype.card ι:ℝ) := Nat.cast_pos.mpr Fintype.card_pos
  have hD := p.error_pos i
  have hlog : Real.log (2*(Fintype.card ι:ℝ)*p.D i/ε) ≤ p.a i*p.targetResolution ε i := by
    have h := (div_le_iff₀ (p.rate_pos i)).mp (le_max_right (p.base i)
      (Real.log (2*(Fintype.card ι:ℝ)*p.D i/ε)/p.a i))
    simpa only [targetResolution, mul_comm] using h
  calc
    _ ≤ p.D i*Real.exp (-Real.log (2*(Fintype.card ι:ℝ)*p.D i/ε)) := by
      apply mul_le_mul_of_nonneg_left ?_ (p.error_pos i).le
      apply Real.exp_le_exp.mpr
      linarith
    _ = _ := by
      rw [Real.exp_neg, Real.exp_log (by positivity)]
      field_simp [hD.ne']

theorem totalError_upperLog_le [Nonempty ι] {ε : ℝ} (hε : 0 < ε) :
    p.totalError (p.upperLog ε) ≤ ε/2 := by
  calc
    _ ≤ ∑ i, p.D i*Real.exp (-p.a i*p.targetResolution ε i) := by
      apply Finset.sum_le_sum
      intro i _
      apply mul_le_mul_of_nonneg_left ?_ (p.error_pos i).le
      apply Real.exp_le_exp.mpr
      exact mul_le_mul_of_nonpos_left (p.targetResolution_le_upper_resolution ε i)
        (by linarith [p.rate_pos i])
    _ ≤ ∑ _i : ι, ε/(2*(Fintype.card ι:ℝ)) :=
      Finset.sum_le_sum (fun i _ ↦ p.targetResolution_error_le hε i)
    _ = ε/2 := by
      have hcard : (Fintype.card ι:ℝ) ≠ 0 := (Nat.cast_pos.mpr Fintype.card_pos).ne'
      simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      field_simp

theorem exists_logMultiplier [Nonempty ι] {ε : ℝ} (hε : 0 < ε) (hbudget : ε < p.baselineError) :
    ∃ t ∈ Ioo p.lowerLog (p.upperLog ε), p.totalError t = ε := by
  have hl : ε < p.totalError p.lowerLog := by rwa [p.totalError_lowerLog]
  have hu : p.totalError (p.upperLog ε) < ε := (p.totalError_upperLog_le hε).trans_lt (by linarith)
  exact intermediate_value_Ioo' (p.lowerLog_lt_upperLog ε).le p.totalError_continuous.continuousOn ⟨hu,hl⟩


omit [Fintype ι] in
theorem power_pos : 0 < p.κ := zero_lt_one.trans_le p.power_ge_one

omit [Fintype ι] in
theorem exp_logOffset (i : ι) : Real.exp (p.logOffset i) = p.a i*p.D i/(p.κ*p.c i) := by
  have ha := p.rate_pos i
  have hD := p.error_pos i
  have hc := p.cost_pos i
  have hk := p.power_pos
  exact Real.exp_log (by positivity)

omit [Fintype ι] in
theorem marginal_identity (i : ι) (y : ℝ) (hy : 0 < y) :
    p.κ*p.c i*y^(p.κ-1) =
      (p.κ*p.c i*Real.exp (-p.a i*y))*Real.exp (logStationarity p.κ (p.a i) y) := by
  rw [logStationarity, Real.exp_add, Real.rpow_def_of_pos hy]
  have he : Real.exp (-p.a i*y)*Real.exp (p.a i*y)=1 := by
    rw [← Real.exp_add]
    simp
  have hswap : (p.κ-1)*Real.log y = Real.log y*(p.κ-1) := by ring
  rw [hswap]
  calc
    _ = (p.κ*p.c i*Real.exp (Real.log y*(p.κ-1)))*1 := by ring
    _ = _ := by rw [← he]; ring

omit [Fintype ι] in
theorem lagrange_identity (t : ℝ) (i : ι) (y : ℝ) :
    Real.exp t*p.a i*p.D i*Real.exp (-p.a i*y) =
      (p.κ*p.c i*Real.exp (-p.a i*y))*Real.exp (t+p.logOffset i) := by
  have hc : p.κ*p.c i ≠ 0 := (mul_pos p.power_pos (p.cost_pos i)).ne'
  rw [Real.exp_add, p.exp_logOffset]
  field_simp [p.power_pos.ne', (p.cost_pos i).ne']

def slack (t : ℝ) (i : ι) : ℝ :=
  p.κ*p.c i*(p.resolution t i)^(p.κ-1) -
    Real.exp t*p.a i*p.D i*Real.exp (-p.a i*p.resolution t i)

omit [Fintype ι] in
theorem slack_nonneg (t : ℝ) (i : ι) : 0 ≤ p.slack t i := by
  have hy : 0 < p.resolution t i := zero_lt_one.trans_le
    ((p.base_ge_one i).trans (p.resolution_ge_base t i))
  have hlog : t+p.logOffset i ≤ logStationarity p.κ (p.a i) (p.resolution t i) := by
    rw [p.resolution_log]
    exact le_max_right _ _
  unfold slack
  rw [p.marginal_identity i _ hy, p.lagrange_identity]
  apply sub_nonneg.mpr
  exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hlog)
    (mul_nonneg (mul_pos p.power_pos (p.cost_pos i)).le (Real.exp_pos _).le)

omit [Fintype ι] in
theorem slack_complementarity (t : ℝ) (i : ι) :
    p.slack t i*(p.resolution t i-p.base i)=0 := by
  by_cases hb : t+p.logOffset i ≤ logStationarity p.κ (p.a i) (p.base i)
  · have heq : p.resolution t i = p.base i :=
      spokeResolution_eq_base p.power_ge_one (p.rate_pos i) (p.base_ge_one i) hb
    rw [heq, sub_self, mul_zero]
  · have hy : 0 < p.resolution t i := zero_lt_one.trans_le
      ((p.base_ge_one i).trans (p.resolution_ge_base t i))
    have hlog : logStationarity p.κ (p.a i) (p.resolution t i) = t+p.logOffset i := by
      rw [p.resolution_log,
        max_eq_right (le_of_not_ge hb)]
    unfold slack
    rw [p.marginal_identity i _ hy, p.lagrange_identity, hlog, sub_self, zero_mul]

omit [Fintype ι] in
theorem resolution_stationary (t : ℝ) (i : ι) :
    p.κ*p.c i*(p.resolution t i)^(p.κ-1) =
      Real.exp t*p.a i*p.D i*Real.exp (-p.a i*p.resolution t i)+p.slack t i := by
  unfold slack
  ring

/-- The common multiplier is chosen from the proved nonempty finite bracket. -/
def solveLogMultiplier [Nonempty ι] (ε : ℝ) (hε : 0 < ε) (hbudget : ε < p.baselineError) : ℝ :=
  Classical.choose (p.exists_logMultiplier hε hbudget)

theorem solveLogMultiplier_spec [Nonempty ι] {ε : ℝ} (hε : 0 < ε) (hbudget : ε < p.baselineError) :
    p.solveLogMultiplier ε hε hbudget ∈ Ioo p.lowerLog (p.upperLog ε) ∧
      p.totalError (p.solveLogMultiplier ε hε hbudget) = ε :=
  Classical.choose_spec (p.exists_logMultiplier hε hbudget)

/-- The constructed general-power allocation is globally optimal among all
 feasible continuous resolutions. The KKT conditions have all been derived
 from the one-dimensional solver, rather than assumed. -/
theorem solver_global_min [Nonempty ι] {ε : ℝ} (hε : 0 < ε) (hbudget : ε < p.baselineError)
    (z : ι → ℝ) (hz : ∀ i, p.base i ≤ z i)
    (herr : (∑ i, p.D i*Real.exp (-p.a i*z i)) ≤ ε) :
    (∑ i, p.c i*(p.resolution (p.solveLogMultiplier ε hε hbudget) i)^p.κ) ≤
      ∑ i, p.c i*(z i)^p.κ := by
  apply constrained_per_spoke_global_min p.power_ge_one
    (fun i ↦ (p.cost_pos i).le) (fun i ↦ (p.error_pos i).le)
    (fun i ↦ zero_lt_one.trans_le (p.base_ge_one i))
    (p.resolution_ge_base _) hz (Real.exp_pos _).le
    (p.slack_nonneg _) (p.slack_complementarity _) (p.resolution_stationary _)
  change (∑ i, p.D i*Real.exp (-p.a i*z i)) ≤ p.totalError (p.solveLogMultiplier ε hε hbudget)
  rw [(p.solveLogMultiplier_spec hε hbudget).2]
  exact herr


theorem exists_active_of_error_lt_baseline (t : ℝ) (ht : p.totalError t < p.baselineError) :
    ∃ i, p.base i < p.resolution t i := by
  by_contra h
  push Not at h
  have heq : ∀ i, p.resolution t i = p.base i :=
    fun i ↦ le_antisymm (h i) (p.resolution_ge_base t i)
  have herr : p.totalError t = p.baselineError := by
    unfold totalError baselineError
    simp_rw [heq]
  linarith

omit [Fintype ι] in
theorem resolution_strict_of_active (s t : ℝ) (hst : s < t) (i : ι)
    (hactive : p.base i < p.resolution s i) : p.resolution s i < p.resolution t i := by
  have hb : logStationarity p.κ (p.a i) (p.base i) < s+p.logOffset i := by
    by_contra h
    have he := spokeResolution_eq_base p.power_ge_one (p.rate_pos i) (p.base_ge_one i)
      (le_of_not_gt h)
    change p.resolution s i = p.base i at he
    linarith
  apply lt_of_le_of_ne (p.resolution_monotone i hst.le)
  intro heq
  change p.resolution s i = p.resolution t i at heq
  have hs := p.resolution_log s i
  have ht := p.resolution_log t i
  rw [max_eq_right hb.le] at hs
  rw [max_eq_right (show logStationarity p.κ (p.a i) (p.base i) ≤ t+p.logOffset i by linarith)] at ht
  rw [heq] at hs
  linarith

/-- After any spoke becomes active the total error decreases strictly. Its
 initial plateau is exactly the baseline error and cannot contain an interior
 requested budget. -/
theorem totalError_strict_of_below_baseline (s t : ℝ) (hst : s < t)
    (hs : p.totalError s < p.baselineError) : p.totalError t < p.totalError s := by
  obtain ⟨i, hi⟩ := p.exists_active_of_error_lt_baseline s hs
  apply Finset.sum_lt_sum
  · intro j _
    apply mul_le_mul_of_nonneg_left ?_ (p.error_pos j).le
    apply Real.exp_le_exp.mpr
    exact mul_le_mul_of_nonpos_left (p.resolution_monotone j hst.le) (by linarith [p.rate_pos j])
  · refine ⟨i, Finset.mem_univ i, ?_⟩
    apply mul_lt_mul_of_pos_left ?_ (p.error_pos i)
    apply Real.exp_lt_exp.mpr
    have hm := mul_lt_mul_of_pos_left (p.resolution_strict_of_active s t hst i hi) (p.rate_pos i)
    linarith

theorem logMultiplier_unique {ε s t : ℝ} (hbudget : ε < p.baselineError)
    (hs : p.totalError s = ε) (ht : p.totalError t = ε) : s = t := by
  rcases lt_trichotomy s t with hst | heq | hts
  · have h := p.totalError_strict_of_below_baseline s t hst (by rwa [hs])
    rw [hs,ht] at h
    exact False.elim (lt_irrefl _ h)
  · exact heq
  · have h := p.totalError_strict_of_below_baseline t s hts (by rwa [ht])
    rw [hs,ht] at h
    exact False.elim (lt_irrefl _ h)

theorem existsUnique_logMultiplier [Nonempty ι] {ε : ℝ}
    (hε : 0 < ε) (hbudget : ε < p.baselineError) : ∃! t : ℝ, p.totalError t = ε := by
  obtain ⟨t, _, ht⟩ := p.exists_logMultiplier hε hbudget
  exact ⟨t, ht, fun s hs ↦ p.logMultiplier_unique hbudget hs ht⟩

/-- The positive common multiplier has an explicit finite logarithmic bracket. -/
theorem positive_multiplier_certificate [Nonempty ι] {ε : ℝ}
    (hε : 0 < ε) (hbudget : ε < p.baselineError) :
    0 < Real.exp (p.solveLogMultiplier ε hε hbudget) ∧
    Real.exp p.lowerLog < Real.exp (p.solveLogMultiplier ε hε hbudget) ∧
    Real.exp (p.solveLogMultiplier ε hε hbudget) < Real.exp (p.upperLog ε) ∧
    p.totalError (p.solveLogMultiplier ε hε hbudget) = ε := by
  have h := p.solveLogMultiplier_spec hε hbudget
  exact ⟨Real.exp_pos _, Real.exp_lt_exp.mpr h.1.1, Real.exp_lt_exp.mpr h.1.2, h.2⟩

/-- The actual integer output of upward rounding remains within budget and
 costs at most 2^κ times the globally optimal continuous certificate. -/
theorem rounded_solver_certificate [Nonempty ι] {ε : ℝ}
    (hε : 0 < ε) (hbudget : ε < p.baselineError) :
    (∑ i, p.D i*Real.exp (-p.a i*(⌈p.resolution (p.solveLogMultiplier ε hε hbudget) i⌉₊:ℝ))) ≤ ε ∧
    (∑ i, p.c i*(⌈p.resolution (p.solveLogMultiplier ε hε hbudget) i⌉₊:ℝ)^p.κ) ≤
      (2:ℝ)^p.κ*(∑ i, p.c i*(p.resolution (p.solveLogMultiplier ε hε hbudget) i)^p.κ) := by
  constructor
  · apply le_trans (Finset.sum_le_sum (fun i _ ↦ QIRadon.Allocation.exponential_error_rounding
      (p.error_pos i).le (p.rate_pos i).le))
    exact (p.solveLogMultiplier_spec hε hbudget).2.le
  · exact power_cost_rounding p.power_pos.le (fun i ↦ (p.cost_pos i).le)
      (fun i ↦ (p.base_ge_one i).trans (p.resolution_ge_base _ i))


/-- If the minimum resolutions already meet the budget, they are optimal and
 no multiplier solve is needed. -/
theorem baseline_global_min (z : ι → ℝ) (hz : ∀ i, p.base i ≤ z i) :
    (∑ i, p.c i*(p.base i)^p.κ) ≤ ∑ i, p.c i*(z i)^p.κ := by
  apply Finset.sum_le_sum
  intro i _
  apply mul_le_mul_of_nonneg_left ?_ (p.cost_pos i).le
  exact Real.rpow_le_rpow (zero_le_one.trans (p.base_ge_one i)) (hz i) p.power_pos.le


omit [Fintype ι] in
theorem logMultiplier_rhs (t : ℝ) (i : ι) :
    Real.log (Real.exp t*p.a i*p.D i/(p.κ*p.c i)) = t+p.logOffset i := by
  have ha := p.rate_pos i
  have hD := p.error_pos i
  have hc := p.cost_pos i
  have hk := p.power_pos
  have hr : 0 < p.a i*p.D i/(p.κ*p.c i) := by positivity
  rw [show Real.exp t*p.a i*p.D i/(p.κ*p.c i) =
    Real.exp t*(p.a i*p.D i/(p.κ*p.c i)) by ring,
    Real.log_mul (Real.exp_pos _).ne' hr.ne', Real.log_exp]
  rfl

omit [Fintype ι] in
theorem resolution_active_bracket (t : ℝ) (i : ι)
    (ht : logStationarity p.κ (p.a i) (p.base i) < t+p.logOffset i) :
    p.resolution t i ∈ Icc (p.base i) (max (p.base i) ((t+p.logOffset i)/p.a i)) ∧
    logStationarity p.κ (p.a i) (p.resolution t i) = t+p.logOffset i := by
  have h := spokeResolution_spec p.power_ge_one (p.rate_pos i) (p.base_ge_one i) (t+p.logOffset i)
  simpa only [resolution, max_eq_right ht.le] using h

/-- Any finite outer bracket supplies a feasible upper-end allocation and
 an explicit per-spoke distance to the exact optimum. Thus bisection need not
 evaluate the limiting multiplier exactly to produce certified allocations. -/
theorem upper_bracket_certificate [Nonempty ι] {ε l u : ℝ}
    (hε : 0 < ε) (hbudget : ε < p.baselineError)
    (hl : l ≤ p.solveLogMultiplier ε hε hbudget)
    (hu : p.solveLogMultiplier ε hε hbudget ≤ u) :
    p.totalError u ≤ ε ∧
    ∀ i, p.resolution (p.solveLogMultiplier ε hε hbudget) i ≤ p.resolution u i ∧
      p.resolution u i ≤ p.resolution (p.solveLogMultiplier ε hε hbudget) i+(u-l)/p.a i := by
  constructor
  · exact (p.totalError_antitone hu).trans_eq (p.solveLogMultiplier_spec hε hbudget).2
  · intro i
    refine ⟨p.resolution_monotone i hu, ?_⟩
    have h := spokeResolution_difference p.power_ge_one (p.rate_pos i) (p.base_ge_one i)
      (add_le_add hu (le_refl (p.logOffset i)))
    change p.a i*(p.resolution u i-p.resolution (p.solveLogMultiplier ε hε hbudget) i) ≤
      (u+p.logOffset i)-(p.solveLogMultiplier ε hε hbudget+p.logOffset i) at h
    have hh : p.a i*(p.resolution u i-p.resolution (p.solveLogMultiplier ε hε hbudget) i) ≤ u-l := by
      linarith
    have hd := (le_div_iff₀ (p.rate_pos i)).mpr
      (show (p.resolution u i-p.resolution (p.solveLogMultiplier ε hε hbudget) i)*p.a i ≤ u-l by
        simpa only [mul_comm] using hh)
    linarith


omit [Fintype ι] in
theorem logOffset_separated (i : ι) :
    p.logOffset i = Real.log (p.a i)+Real.log (p.D i)-Real.log p.κ-Real.log (p.c i) := by
  rw [logOffset, Real.log_div (mul_pos (p.rate_pos i) (p.error_pos i)).ne'
    (mul_pos p.power_pos (p.cost_pos i)).ne',
    Real.log_mul (p.rate_pos i).ne' (p.error_pos i).ne',
    Real.log_mul p.power_pos.ne' (p.cost_pos i).ne']
  ring

end SpokeData

end QIRadon.AllocationPower
