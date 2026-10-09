import Mathlib

/-! Explicit Gaussian-polynomial window used in local-profile preparation.
This module proves actual finite-series inequalities, not an assumed approximation rate. -/

namespace QIRadon

noncomputable section

def flatWindow (n : ℕ) (u : ℝ) : ℝ :=
  Real.exp (-(n : ℝ) * u) *
    ∑ k ∈ Finset.range (n + 1), ((n : ℝ) * u) ^ k / (k.factorial : ℝ)

lemma real_exp_hasSum (x : ℝ) :
    HasSum (fun k : ℕ => x ^ k / (k.factorial : ℝ)) (Real.exp x) := by
  simpa only [Real.exp_eq_exp_ℝ] using NormedSpace.expSeries_div_hasSum_exp x

lemma flatWindow_nonneg (n : ℕ) {u : ℝ} (hu : 0 ≤ u) : 0 ≤ flatWindow n u := by
  unfold flatWindow
  positivity

lemma flatWindow_le_one (n : ℕ) {u : ℝ} (hu : 0 ≤ u) : flatWindow n u ≤ 1 := by
  have hsum := Real.sum_le_exp_of_nonneg (show 0 ≤ (n : ℝ) * u by positivity) (n + 1)
  have hmul := mul_le_mul_of_nonneg_left hsum (Real.exp_pos (-(n : ℝ) * u)).le
  have he : Real.exp (-(n : ℝ) * u) * Real.exp ((n : ℝ) * u) = 1 := by
    rw [← Real.exp_add]
    have hz : -(n : ℝ) * u + (n : ℝ) * u = 0 := by ring
    rw [hz, Real.exp_zero]
  simpa only [flatWindow, he] using hmul

theorem flatWindow_error_bound (n : ℕ) {u : ℝ} (hu : 0 < u) (hu1 : u ≤ 1) :
    1 - flatWindow n u ≤ Real.exp (-(n : ℝ) * (u - 1 - Real.log u)) := by
  let x : ℝ := (n : ℝ) * u
  let tx : ℕ → ℝ := fun k => x ^ k / (k.factorial : ℝ)
  let tn : ℕ → ℝ := fun k => (n : ℝ) ^ k / (k.factorial : ℝ)
  have sx : Summable tx := (real_exp_hasSum x).summable
  have sn : Summable tn := (real_exp_hasSum (n : ℝ)).summable
  have sxshift : Summable (fun j => tx (j + (n + 1))) :=
    (summable_nat_add_iff (n + 1)).mpr sx
  have snshift : Summable (fun j => tn (j + (n + 1))) :=
    (summable_nat_add_iff (n + 1)).mpr sn
  have hterm (j : ℕ) : tx (j + (n + 1)) ≤ u ^ n * tn (j + (n + 1)) := by
    have hp : u ^ (j + (n + 1)) ≤ u ^ n :=
      pow_le_pow_of_le_one hu.le hu1 (by omega)
    dsimp [tx, tn, x]
    calc
      ((n : ℝ) * u) ^ (j + (n + 1)) / ↑(j + (n + 1)).factorial =
          u ^ (j + (n + 1)) * ((n : ℝ) ^ (j + (n + 1)) / ↑(j + (n + 1)).factorial) := by
        rw [mul_pow]
        ring
      _ ≤ u ^ n * ((n : ℝ) ^ (j + (n + 1)) / ↑(j + (n + 1)).factorial) :=
        mul_le_mul_of_nonneg_right hp (by positivity)
  have hcomp := Summable.tsum_le_tsum hterm sxshift (snshift.mul_left (u ^ n))
  rw [tsum_mul_left] at hcomp
  have hsx := sx.sum_add_tsum_nat_add (n + 1)
  have hsn := sn.sum_add_tsum_nat_add (n + 1)
  have hxt : (∑' k, tx k) = Real.exp x := (real_exp_hasSum x).tsum_eq
  have hnt : (∑' k, tn k) = Real.exp (n : ℝ) := (real_exp_hasSum (n : ℝ)).tsum_eq
  rw [hxt] at hsx
  rw [hnt] at hsn
  have hsum_nonneg : 0 ≤ ∑ k ∈ Finset.range (n + 1), tn k := by
    apply Finset.sum_nonneg
    intro k hk
    dsimp [tn]
    positivity
  have htail : (∑' j, tx (j + (n + 1))) ≤ u ^ n * Real.exp (n : ℝ) := by
    calc
      _ ≤ u ^ n * (∑' j, tn (j + (n + 1))) := hcomp
      _ ≤ u ^ n * Real.exp (n : ℝ) := mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have hcancel : Real.exp (-x) * Real.exp x = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  have hidentity : 1 - flatWindow n u = Real.exp (-x) * (∑' j, tx (j + (n + 1))) := by
    unfold flatWindow
    have hexp : Real.exp (-(n : ℝ) * u) = Real.exp (-x) := by
      congr 1
      dsimp [x]
      ring
    rw [hexp]
    change 1 - Real.exp (-x) * (∑ k ∈ Finset.range (n + 1), tx k) = _
    nlinarith [congrArg (fun z : ℝ => Real.exp (-x) * z) hsx]
  rw [hidentity]
  calc
    _ ≤ Real.exp (-x) * (u ^ n * Real.exp (n : ℝ)) :=
      mul_le_mul_of_nonneg_left htail (Real.exp_pos _).le
    _ = Real.exp (-(n : ℝ) * (u - 1 - Real.log u)) := by
      have hp : u ^ n = Real.exp ((n : ℝ) * Real.log u) := by
        rw [Real.exp_nat_mul, Real.exp_log hu]
      rw [hp, ← Real.exp_add, ← Real.exp_add]
      congr 1
      dsimp [x]
      ring

lemma flatWindow_rate_pos {u : ℝ} (hu : 0 < u) (hu1 : u < 1) :
    0 < u - 1 - Real.log u := by
  have h := Real.log_lt_sub_one_of_pos hu (ne_of_lt hu1)
  linarith

lemma flatWindow_rate_antitone {u v : ℝ} (hu : 0 < u) (huv : u ≤ v) (hv1 : v ≤ 1) :
    v - 1 - Real.log v ≤ u - 1 - Real.log u := by
  have hv : 0 < v := hu.trans_le huv
  have hlog := Real.log_le_sub_one_of_pos (div_pos hu hv)
  rw [Real.log_div hu.ne' hv.ne'] at hlog
  have hdiv : u / v * v = u := div_mul_cancel₀ u hv.ne'
  have hp := mul_nonneg (sub_nonneg.mpr hv1) (sub_nonneg.mpr huv)
  have hfrac : u / v - 1 ≤ u - v := by
    apply (mul_le_mul_iff_left₀ hv).mp
    nlinarith
  linarith

lemma flatWindow_zero (n : ℕ) : flatWindow n 0 = 1 := by
  simp only [flatWindow, mul_zero, Real.exp_zero, one_mul]
  rw [Finset.sum_eq_single 0]
  · norm_num
  · intro k hk hne
    simp [zero_pow hne]
  · simp

/-- A uniform real-domain bound for the actual finite Gaussian-polynomial window. -/
theorem flatWindow_uniform_error (n : ℕ) {u v : ℝ}
    (hu : 0 ≤ u) (huv : u ≤ v) (_hv : 0 < v) (hv1 : v < 1) :
    |1 - flatWindow n u| ≤ Real.exp (-(n : ℝ) * (v - 1 - Real.log v)) := by
  rw [abs_of_nonneg (sub_nonneg.mpr (flatWindow_le_one n hu))]
  by_cases hz : u = 0
  · subst u
    rw [flatWindow_zero, sub_self]
    exact (Real.exp_pos _).le
  have hup : 0 < u := lt_of_le_of_ne hu (Ne.symm hz)
  calc
    _ ≤ Real.exp (-(n : ℝ) * (u - 1 - Real.log u)) :=
      flatWindow_error_bound n hup (huv.trans hv1.le)
    _ ≤ Real.exp (-(n : ℝ) * (v - 1 - Real.log v)) := by
      apply Real.exp_le_exp.mpr
      exact mul_le_mul_of_nonpos_left (flatWindow_rate_antitone hup huv hv1.le)
        (neg_nonpos.mpr (Nat.cast_nonneg n))

theorem flatWindow_preparation_error (n : ℕ) {g p u v E M : ℝ}
    (happrox : |g - p| ≤ E) (hp : |p| ≤ M)
    (hu : 0 ≤ u) (huv : u ≤ v) (hv : 0 < v) (hv1 : v < 1) :
    |g - p * flatWindow n u| ≤
      E + M * Real.exp (-(n : ℝ) * (v - 1 - Real.log v)) := by
  have herr := flatWindow_uniform_error n hu huv hv hv1
  calc
    |g - p * flatWindow n u| = |(g - p) + p * (1 - flatWindow n u)| := by ring_nf
    _ ≤ |g - p| + |p * (1 - flatWindow n u)| := abs_add_le _ _
    _ = |g - p| + |p| * |1 - flatWindow n u| := by rw [abs_mul]
    _ ≤ E + M * Real.exp (-(n : ℝ) * (v - 1 - Real.log v)) := by
      apply add_le_add happrox
      exact mul_le_mul hp herr (abs_nonneg _) (le_trans (abs_nonneg _) hp)

end
end QIRadon
