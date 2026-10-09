import QIRadon.Allocation.Uniform
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-! Per-spoke allocation for the actual nonlinear sufficient row cost c n^κ.
Unlike the linear-cost water-filling formula, these optimality certificates
apply to κ≥1. They certify a candidate satisfying the displayed multiplier
conditions. `AllocationPowerSolver.lean` supplies existence and root selection. -/

noncomputable section
open scoped BigOperators Real
namespace QIRadon.AllocationPower

theorem power_supporting_line {κ y z : ℝ} (hκ : 1 ≤ κ) (hy : 0 < y) (hz : 0 ≤ z) :
    y^κ + κ*y^(κ-1)*(z-y) ≤ z^κ := by
  have hs : -1 ≤ z/y-1 := by linarith [div_nonneg hz hy.le]
  have hh := one_add_mul_self_le_rpow_one_add hs hκ
  rw [show 1+(z/y-1)=z/y by ring, Real.div_rpow hz hy.le] at hh
  have hm := mul_le_mul_of_nonneg_right hh (Real.rpow_nonneg hy.le κ)
  have hp : y^κ ≠ 0 := (Real.rpow_pos_of_pos hy κ).ne'
  rw [div_mul_cancel₀ _ hp] at hm
  have he : (1+κ*(z/y-1))*y^κ = y^κ + κ*y^(κ-1)*(z-y) := by
    rw [Real.rpow_sub hy, Real.rpow_one]
    field_simp
  rwa [he] at hm

/-- Convexity of the exponential error: moving to a feasible error budget
has nonnegative first-order displacement at the reference resolutions. This
single inequality is shared by the unconstrained and constrained cost proofs. -/
theorem exponential_budget_tangent_nonneg {ι : Type*} [Fintype ι]
    {a D y z : ι → ℝ} (hD : ∀ i, 0 ≤ D i)
    (hbudget : (∑ i, D i*Real.exp (-a i*z i)) ≤ ∑ i, D i*Real.exp (-a i*y i)) :
    0 ≤ ∑ i, (a i*D i*Real.exp (-a i*y i))*(z i-y i) := by
  have he := Finset.sum_le_sum (s := Finset.univ)
    (fun i _ => QIRadon.Allocation.exponential_supporting_line (D i) (a i) (z i) (y i) (hD i))
  rw [Finset.sum_add_distrib] at he
  have hn : (∑ i, (a i*D i*Real.exp (-a i*y i))*(y i-z i)) =
      -(∑ i, (a i*D i*Real.exp (-a i*y i))*(z i-y i)) := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hn] at he
  linarith

/-- Global minimization of the continuous per-spoke width certificate.
D includes each positive angular quadrature weight. The multiplier conditions
are the correct general-κ equations, not the κ=1 water-filling formula. -/
theorem per_spoke_global_min {ι : Type*} [Fintype ι]
    {c a D y z : ι → ℝ} {κ τ : ℝ}
    (hκ : 1 ≤ κ) (hc : ∀ i, 0 ≤ c i) (hD : ∀ i, 0 ≤ D i)
    (hy : ∀ i, 0 < y i) (hz : ∀ i, 0 ≤ z i) (hτ : 0 ≤ τ)
    (hstationary : ∀ i, κ*c i*(y i)^(κ-1) = τ*a i*D i*Real.exp (-a i*y i))
    (hbudget : (∑ i, D i*Real.exp (-a i*z i)) ≤ ∑ i, D i*Real.exp (-a i*y i)) :
    (∑ i, c i*(y i)^κ) ≤ ∑ i, c i*(z i)^κ := by
  have he0 := exponential_budget_tangent_nonneg hD hbudget
  have hp := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
    mul_le_mul_of_nonneg_left (power_supporting_line hκ (hy i) (hz i)) (hc i))
  have hterm (i : ι) : c i*((y i)^κ+κ*(y i)^(κ-1)*(z i-y i)) =
      c i*(y i)^κ+τ*((a i*D i*Real.exp (-a i*y i))*(z i-y i)) := by
    calc
      _ = c i*(y i)^κ + (κ*c i*(y i)^(κ-1))*(z i-y i) := by ring
      _ = _ := by rw [hstationary i]; ring
  simp_rw [hterm] at hp
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hp
  exact (le_add_of_nonneg_right (mul_nonneg hτ he0)).trans hp

/-- Baseline-constrained general-κ allocation. Slack multiplier ν is positive
only on a spoke at its minimum resolution; the explicit complementarity
equation is what makes the same global supporting-line proof go through. -/
theorem constrained_per_spoke_global_min {ι : Type*} [Fintype ι]
    {c a D base y z ν : ι → ℝ} {κ τ : ℝ}
    (hκ : 1 ≤ κ) (hc : ∀ i, 0 ≤ c i) (hD : ∀ i, 0 ≤ D i)
    (hbase : ∀ i, 0 < base i) (hy : ∀ i, base i ≤ y i) (hz : ∀ i, base i ≤ z i)
    (hτ : 0 ≤ τ) (hν : ∀ i, 0 ≤ ν i)
    (hcomp : ∀ i, ν i*(y i-base i)=0)
    (hstat : ∀ i, κ*c i*(y i)^(κ-1) = τ*a i*D i*Real.exp (-a i*y i)+ν i)
    (hbudget : (∑ i, D i*Real.exp (-a i*z i)) ≤ ∑ i, D i*Real.exp (-a i*y i)) :
    (∑ i, c i*(y i)^κ) ≤ ∑ i, c i*(z i)^κ := by
  have he0 := exponential_budget_tangent_nonneg hD hbudget
  have hνz (i : ι) : 0 ≤ ν i*(z i-y i) := by
    have hh := mul_nonneg (hν i) (sub_nonneg.mpr (hz i))
    nlinarith [hcomp i]
  have hp := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
    mul_le_mul_of_nonneg_left
      (power_supporting_line hκ ((hbase i).trans_le (hy i)) ((hbase i).le.trans (hz i))) (hc i))
  have hterm (i : ι) : c i*((y i)^κ+κ*(y i)^(κ-1)*(z i-y i)) =
      c i*(y i)^κ+τ*((a i*D i*Real.exp (-a i*y i))*(z i-y i))+ν i*(z i-y i) := by
    calc
      _ = c i*(y i)^κ+(κ*c i*(y i)^(κ-1))*(z i-y i) := by ring
      _ = _ := by rw [hstat i]; ring
  simp_rw [hterm] at hp
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum] at hp
  have hνsum := Finset.sum_nonneg (s := Finset.univ) (fun i _ => hνz i)
  have hτsum := mul_nonneg hτ he0
  linarith

/-- Rounding resolutions upward preserves every exponential error bound and
increases a general-κ cost by at most 2^κ once each resolution is at least one. -/
theorem power_cost_rounding {ι : Type*} [Fintype ι]
    {c y : ι → ℝ} {κ : ℝ} (hκ : 0 ≤ κ)
    (hc : ∀ i, 0 ≤ c i) (hy : ∀ i, 1 ≤ y i) :
    (∑ i, c i*(⌈y i⌉₊:ℝ)^κ) ≤ (2:ℝ)^κ*(∑ i, c i*(y i)^κ) := by
  have hpoint (i : ι) : c i*(⌈y i⌉₊:ℝ)^κ ≤ (2:ℝ)^κ*(c i*(y i)^κ) := by
    have hy0 : 0 ≤ y i := le_trans zero_le_one (hy i)
    have hround : (⌈y i⌉₊:ℝ) ≤ 2*y i := by
      have hh := Nat.ceil_lt_add_one hy0
      linarith [hy i]
    have hp := Real.rpow_le_rpow (Nat.cast_nonneg _) hround hκ
    rw [Real.mul_rpow (by norm_num : (0:ℝ) ≤ 2) hy0] at hp
    exact (mul_le_mul_of_nonneg_left hp (hc i)).trans_eq (by ring)
  simpa only [Finset.mul_sum] using Finset.sum_le_sum (s := Finset.univ) (fun i _ => hpoint i)

/-- Logarithmic form of one spoke's multiplier equation. It is numerically
preferable to balancing potentially enormous exponentials directly. -/
def logStationarity (κ a n : ℝ) : ℝ := (κ-1)*Real.log n+a*n

theorem logStationarity_strictMonoOn {κ a : ℝ} (hκ : 1 ≤ κ) (ha : 0 < a) :
    StrictMonoOn (logStationarity κ a) (Set.Ici 1) := by
  intro y hy z _hz hyz
  have hy0 : 0 < y := lt_of_lt_of_le zero_lt_one hy
  have hlog := Real.strictMonoOn_log hy0 (hy0.trans hyz) hyz
  exact add_lt_add_of_le_of_lt (mul_le_mul_of_nonneg_left hlog.le (sub_nonneg.mpr hκ))
    (mul_lt_mul_of_pos_left hyz ha)

theorem logStationarity_continuousOn {κ a : ℝ} :
    ContinuousOn (logStationarity κ a) (Set.Ici 1) := by
  intro x hx
  have hx0 : x ≠ 0 := (lt_of_lt_of_le zero_lt_one hx).ne'
  exact ((Real.continuousAt_log hx0).const_mul (κ-1) |>.add
    (continuousAt_id.const_mul a)).continuousWithinAt

/-- Each active spoke equation has a unique root in an explicit finite
bracket. A scalar bisection can therefore select its resolution. This statement
is composed with the common-multiplier solver in `AllocationPowerSolver.lean`. -/
theorem existsUnique_spoke_root {κ a base b : ℝ}
    (hκ : 1 ≤ κ) (ha : 0 < a) (hbase : 1 ≤ base)
    (hb : logStationarity κ a base ≤ b) :
    ∃! n : ℝ, n ∈ Set.Icc base (max base (b/a)) ∧ logStationarity κ a n = b := by
  let hi := max base (b/a)
  have hhi : 1 ≤ hi := hbase.trans (le_max_left _ _)
  have hlog : 0 ≤ Real.log hi := Real.log_nonneg hhi
  have hahi : b ≤ a*hi := by
    have hh := (div_le_iff₀ ha).mp (le_max_right base (b/a))
    simpa only [mul_comm] using hh
  have hupper : b ≤ logStationarity κ a hi := by
    have hh := mul_nonneg (sub_nonneg.mpr hκ) hlog
    unfold logStationarity
    linarith
  have hcont : ContinuousOn (logStationarity κ a) (Set.Icc base hi) :=
    logStationarity_continuousOn.mono (fun _ hn => hbase.trans hn.1)
  obtain ⟨n, hn, heq⟩ := intermediate_value_Icc (le_max_left base (b/a)) hcont ⟨hb,hupper⟩
  refine ⟨n,⟨hn,heq⟩,?_⟩
  intro z hz
  apply (logStationarity_strictMonoOn hκ ha).injOn
    (hbase.trans hz.1.1) (hbase.trans hn.1)
  exact hz.2.trans heq.symm

end QIRadon.AllocationPower
