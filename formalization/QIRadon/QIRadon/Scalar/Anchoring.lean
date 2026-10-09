import Mathlib

/-! Sharp r-fold anchoring bound on a finite real interval. The assumptions are
local to the interval; global differentiability is not imposed on the target. -/

namespace QIRadon

open Set
open scoped Interval

noncomputable section

theorem anchored_iteratedDeriv_pointwise (r : ℕ) {f : ℝ → ℝ} {T E x : ℝ}
    (hT : 0 < T) (hf : ContDiffOn ℝ r f (Icc (-T) T))
    (hjet : ∀ k < r, iteratedDeriv k f 0 = 0)
    (hbound : ∀ y ∈ Icc (-T) T, |iteratedDeriv r f y| ≤ E)
    (hx : x ∈ Icc (-T) T) :
    |f x| ≤ E * |x| ^ r / (r.factorial : ℝ) := by
  cases r with
  | zero => simpa using hbound x hx
  | succ n =>
    have hE : 0 ≤ E := (abs_nonneg _).trans (hbound 0 ⟨by linarith, hT.le⟩)
    by_cases hx0 : x = 0
    · subst x
      have hz := hjet 0 (by omega)
      simp only [iteratedDeriv_zero] at hz
      simp [hz]
    have hne : (0 : ℝ) ≠ x := Ne.symm hx0
    have hseg : uIcc (0 : ℝ) x ⊆ Icc (-T) T := by
      intro y hy
      exact ⟨(le_min (by linarith) hx.1).trans hy.1,
        hy.2.trans (max_le hT.le hx.2)⟩
    have hlocal : ContDiffOn ℝ (n + 1 : ℕ) f (uIcc (0 : ℝ) x) := hf.mono hseg
    have hzero : taylorWithinEval f n (uIcc (0 : ℝ) x) 0 x = 0 := by
      rw [taylor_within_apply]
      apply Finset.sum_eq_zero
      intro k hk
      have hkn : k < n + 1 := Finset.mem_range.mp hk
      have hc : ContDiffAt ℝ k f 0 := by
        apply (hf.contDiffAt (Icc_mem_nhds (by linarith) hT)).of_le
        exact_mod_cast (Nat.le_of_lt hkn)
      rw [iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hne) hc (by simp), hjet k hkn]
      simp
    obtain ⟨y, hy, heq⟩ := taylor_mean_remainder_lagrange_iteratedDeriv hne hlocal
    rw [hzero, sub_zero, sub_zero] at heq
    have hyD : y ∈ Icc (-T) T := hseg ⟨hy.1.le, hy.2.le⟩
    rw [heq, abs_div, abs_mul, abs_pow, abs_of_pos (by positivity : (0 : ℝ) < ↑(n + 1).factorial)]
    exact div_le_div_of_nonneg_right
      (mul_le_mul_of_nonneg_right (hbound y hyD) (by positivity)) (by positivity)

theorem anchored_iteratedDeriv_uniform (r : ℕ) {f : ℝ → ℝ} {T E : ℝ}
    (hT : 0 < T) (hf : ContDiffOn ℝ r f (Icc (-T) T))
    (hjet : ∀ k < r, iteratedDeriv k f 0 = 0)
    (hbound : ∀ y ∈ Icc (-T) T, |iteratedDeriv r f y| ≤ E) :
    ∀ x ∈ Icc (-T) T, |f x| ≤ E * T ^ r / (r.factorial : ℝ) := by
  intro x hx
  have hE : 0 ≤ E := (abs_nonneg _).trans (hbound 0 ⟨by linarith, hT.le⟩)
  calc
    _ ≤ E * |x| ^ r / (r.factorial : ℝ) :=
      anchored_iteratedDeriv_pointwise r hT hf hjet hbound hx
    _ ≤ E * T ^ r / (r.factorial : ℝ) := by
      gcongr
      exact abs_le.mpr hx

/-- For positive derivative order Taylor samples only the interior. This version
does not impose spurious differentiability across the interval endpoints. -/
theorem anchored_succ_iteratedDeriv_uniform (n : ℕ) {f : ℝ → ℝ} {T E : ℝ}
    (hT : 0 < T) (hf : ContDiffOn ℝ (n + 1 : ℕ) f (Icc (-T) T))
    (hjet : ∀ k < n + 1, iteratedDeriv k f 0 = 0)
    (hbound : ∀ y ∈ Ioo (-T) T, |iteratedDeriv (n + 1) f y| ≤ E) :
    ∀ x ∈ Icc (-T) T, |f x| ≤ E * T ^ (n + 1) / ((n + 1).factorial : ℝ) := by
  intro x hx
  have hE : 0 ≤ E := (abs_nonneg _).trans (hbound 0 ⟨by linarith, hT⟩)
  by_cases hx0 : x = 0
  · subst x
    have hz := hjet 0 (by omega)
    simp only [iteratedDeriv_zero] at hz
    rw [hz, abs_zero]
    positivity
  have hne : (0 : ℝ) ≠ x := Ne.symm hx0
  have hseg : uIcc (0 : ℝ) x ⊆ Icc (-T) T := by
    intro y hy
    exact ⟨(le_min (by linarith) hx.1).trans hy.1,
      hy.2.trans (max_le hT.le hx.2)⟩
  have hlocal : ContDiffOn ℝ (n + 1 : ℕ) f (uIcc (0 : ℝ) x) := hf.mono hseg
  have hzero : taylorWithinEval f n (uIcc (0 : ℝ) x) 0 x = 0 := by
    rw [taylor_within_apply]
    apply Finset.sum_eq_zero
    intro k hk
    have hkn : k < n + 1 := Finset.mem_range.mp hk
    have hc : ContDiffAt ℝ k f 0 := by
      apply (hf.contDiffAt (Icc_mem_nhds (by linarith) hT)).of_le
      exact_mod_cast (Nat.le_of_lt hkn)
    rw [iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc hne) hc (by simp), hjet k hkn]
    simp
  obtain ⟨y, hy, heq⟩ := taylor_mean_remainder_lagrange_iteratedDeriv hne hlocal
  rw [hzero, sub_zero, sub_zero] at heq
  have hyD : y ∈ Ioo (-T) T :=
    ⟨lt_of_le_of_lt (le_min (by linarith) hx.1) hy.1,
      lt_of_lt_of_le hy.2 (max_le hT.le hx.2)⟩
  rw [heq, abs_div, abs_mul, abs_pow,
    abs_of_pos (by positivity : (0 : ℝ) < ↑(n + 1).factorial)]
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact mul_le_mul (hbound y hyD) (pow_le_pow_left₀ (abs_nonneg _) (abs_le.mpr hx) _)
    (by positivity) hE

end
end QIRadon
