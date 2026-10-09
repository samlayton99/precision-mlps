import Mathlib.Analysis.Complex.Liouville
import Mathlib.Analysis.Calculus.FDeriv.Analytic
import Mathlib.Analysis.Calculus.ContDiff.FiniteDimension
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.Tactic

/-! Mixed Cauchy bounds on the actual ℓ¹ domain of complex rotation
parameters. Successive first-order Cauchy estimates avoid the spurious extra
factorial that would destroy geometric angular convergence. -/

noncomputable section
open Set Metric
open scoped BigOperators ContDiff

namespace QIRadon.MixedCauchy

/-- Total complex displacement, matching the domain available for a sequence
of independently complexified rotations. -/
def l1Size {n : ℕ} (z : Fin n → ℂ) : ℝ := ∑ i, ‖z i‖

def l1Ball (n : ℕ) (δ : ℝ) : Set (Fin n → ℂ) := {z | l1Size z < δ}

@[simp] theorem l1Size_zero (n : ℕ) : l1Size (0 : Fin n → ℂ) = 0 := by simp [l1Size]

theorem l1Size_nonneg {n : ℕ} (z : Fin n → ℂ) : 0 ≤ l1Size z :=
  Finset.sum_nonneg (fun _ _ => norm_nonneg _)

theorem l1Size_coordinate_shift {n : ℕ} (z : Fin n → ℂ) (i : Fin n) (t : ℂ) :
    l1Size (z + t • Pi.single i 1) ≤ l1Size z + ‖t‖ := by
  calc
    _ ≤ ∑ j : Fin n, (‖z j‖ + ‖(t • Pi.single i (1 : ℂ)) j‖) := by
      apply Finset.sum_le_sum
      intro j _
      exact norm_add_le _ _
    _ = _ := by
      rw [Finset.sum_add_distrib]
      congr 1
      simp [Pi.single_apply, apply_ite norm]

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℂ V] [CompleteSpace V]

/-- Actual successive coordinate derivatives. A list records their order;
indices need not be distinct. -/
def mixedPartial {n : ℕ} : List (Fin n) → ((Fin n → ℂ) → V) → (Fin n → ℂ) → V
  | [], F => F
  | i :: w, F => fun z => fderiv ℂ (mixedPartial w F) z (Pi.single i 1)

theorem mixedPartial_analytic {n : ℕ} {F : (Fin n → ℂ) → V} {S : Set (Fin n → ℂ)}
    (hF : AnalyticOnNhd ℂ F S) (w : List (Fin n)) :
    AnalyticOnNhd ℂ (mixedPartial w F) S := by
  induction w with
  | nil => exact hF
  | cons i w ih =>
      intro z hz
      exact ((ContinuousLinearMap.apply ℂ V (Pi.single i 1)).analyticAt _).comp ((ih z hz).fderiv)

/-- Main local estimate. Each derivative spends a radius r of the available
ℓ¹ domain. The target bound alone is assumed; derivative bounds are derived. -/
theorem mixedPartial_bound {n : ℕ} {F : (Fin n → ℂ) → V} {δ M r : ℝ}
    (hF : AnalyticOnNhd ℂ F (l1Ball n δ))
    (hbound : ∀ z ∈ l1Ball n δ, ‖F z‖ ≤ M)
    (hr : 0 < r) (w : List (Fin n)) :
    ∀ z, l1Size z + (w.length : ℝ) * r < δ →
      ‖mixedPartial w F z‖ ≤ M / r ^ w.length := by
  induction w with
  | nil =>
      intro z hz
      simpa only [mixedPartial, List.length_nil, Nat.cast_zero, zero_mul, add_zero, pow_zero, div_one]
        using hbound z (by simpa only [l1Ball, mem_ofPred_eq, List.length_nil, Nat.cast_zero, zero_mul, add_zero] using hz)
  | cons i w ih =>
      intro z hz
      have hz0 : z ∈ l1Ball n δ := by
        have hn : 0 ≤ (List.length w : ℝ) := Nat.cast_nonneg _
        simp only [List.length_cons, Nat.cast_add, Nat.cast_one] at hz
        change l1Size z < δ
        nlinarith
      let P := mixedPartial w F
      let line : ℂ → Fin n → ℂ := fun t => z + t • Pi.single i 1
      have hline0 : line 0 = z := by simp [line]
      have hinside : ∀ t ∈ closedBall (0 : ℂ) r,
          l1Size (line t) + (w.length : ℝ) * r < δ := by
        intro t ht
        have ht' : ‖t‖ ≤ r := by simpa only [mem_closedBall, dist_zero_right] using ht
        have hs := l1Size_coordinate_shift z i t
        simp only [List.length_cons, Nat.cast_add, Nat.cast_one] at hz
        dsimp only [line]
        linarith
      have ha : AnalyticOnNhd ℂ (fun t => P (line t)) (closedBall (0 : ℂ) r) := by
        intro t ht
        have ht' : line t ∈ l1Ball n δ := by
          change l1Size (line t) < δ
          have hh := hinside t ht
          have hn : 0 ≤ (w.length : ℝ) * r := by positivity
          linarith
        exact ((mixedPartial_analytic hF w) (line t) ht').comp
          (by dsimp [line]; fun_prop)
      have hdc : DiffContOnCl ℂ (fun t => P (line t)) (ball (0 : ℂ) r) := by
        apply DifferentiableOn.diffContOnCl
        rw [closure_ball _ hr.ne']
        exact ha.differentiableOn
      have hsphere : ∀ t ∈ sphere (0 : ℂ) r, ‖P (line t)‖ ≤ M / r ^ w.length := by
        intro t ht
        exact ih (line t) (hinside t (sphere_subset_closedBall ht))
      have hc := Complex.norm_deriv_le_of_forall_mem_sphere_norm_le hr hdc hsphere
      have hdline : HasDerivAt line (Pi.single i (1 : ℂ)) 0 := by
        convert! ((hasDerivAt_id (0 : ℂ)).smul_const (Pi.single i (1 : ℂ))).const_add z using 1
        simp only [one_smul]
      have hP := ((mixedPartial_analytic hF w) z hz0).differentiableAt.hasFDerivAt
      have hP' : HasFDerivAt P (fderiv ℂ P z) (line 0) := by simpa only [hline0] using hP
      have hder := (hP'.comp_hasDerivAt 0 hdline).deriv
      change deriv (fun t => P (line t)) 0 = mixedPartial (i :: w) F z at hder
      rw [hder] at hc
      apply hc.trans_eq
      simp only [List.length_cons, pow_succ, div_div]

/-- At the origin, a half-radius budget gives the mixed derivative bound
M(2l/δ)^l. No additional factorial appears, even for repeated indices. -/
theorem mixedPartial_zero_bound {n : ℕ} {F : (Fin n → ℂ) → V} {δ M : ℝ}
    (hδ : 0 < δ) (hF : AnalyticOnNhd ℂ F (l1Ball n δ))
    (hbound : ∀ z ∈ l1Ball n δ, ‖F z‖ ≤ M)
    (w : List (Fin n)) (hw : 0 < w.length) :
    ‖mixedPartial w F 0‖ ≤ M * ((2 * (w.length : ℝ)) / δ) ^ w.length := by
  have hwR : 0 < (w.length : ℝ) := Nat.cast_pos.mpr hw
  have hr : 0 < δ / (2 * (w.length : ℝ)) := by positivity
  have hbudget : l1Size (0 : Fin n → ℂ) + (w.length : ℝ) * (δ / (2 * (w.length : ℝ))) < δ := by
    rw [l1Size_zero, zero_add]
    have he : (w.length : ℝ) * (δ / (2 * (w.length : ℝ))) = δ / 2 := by field_simp
    rw [he]
    linarith
  have hh := mixedPartial_bound hF hbound hr w 0 hbudget
  apply hh.trans_eq
  rw [div_eq_mul_inv, ← inv_pow, inv_div]


/-- Exactly one derivative in every complex parameter, in index order. -/
def allCoordinatePartial {n : ℕ} (F : (Fin n → ℂ) → V) (z : Fin n → ℂ) : V :=
  mixedPartial (List.ofFn (fun i : Fin n => i)) F z

theorem allCoordinatePartial_zero_bound {n : ℕ} (hn : 0 < n)
    {F : (Fin n → ℂ) → V} {δ M : ℝ}
    (hδ : 0 < δ) (hF : AnalyticOnNhd ℂ F (l1Ball n δ))
    (hbound : ∀ z ∈ l1Ball n δ, ‖F z‖ ≤ M) :
    ‖allCoordinatePartial F 0‖ ≤ M * ((2 * (n : ℝ)) / δ)^n := by
  simpa only [allCoordinatePartial, List.length_ofFn] using
    mixedPartial_zero_bound hδ hF hbound (List.ofFn (fun i : Fin n => i))
      (by simpa only [List.length_ofFn] using hn)

theorem continuous_l1Size (n : ℕ) : Continuous (l1Size (n := n)) := by
  unfold l1Size
  fun_prop

theorem isOpen_l1Ball (n : ℕ) (δ : ℝ) : IsOpen (l1Ball n δ) :=
  isOpen_lt (continuous_l1Size n) continuous_const


omit [CompleteSpace V]

/-- The same actual coordinate partials stay jointly C∞ on an open set.
This supports integrated profiles without requiring a joint analytic-integral
library theorem. -/
theorem mixedPartial_smooth {n : ℕ} {F : (Fin n → ℂ) → V} {S : Set (Fin n → ℂ)}
    (hS : IsOpen S) (hF : ContDiffOn ℂ ∞ F S) (w : List (Fin n)) :
    ContDiffOn ℂ ∞ (mixedPartial w F) S := by
  induction w with
  | nil => exact hF
  | cons i w ih =>
      have hd : ContDiffOn ℂ ∞ (fderiv ℂ (mixedPartial w F)) S :=
        ih.fderiv_of_isOpen hS (by simp)
      exact (contDiffOn_clm_apply.mp hd) (Pi.single i 1)

/-- Sharp local mixed Cauchy estimate from actual joint C∞ complex regularity.
Only one-dimensional Cauchy estimates are used, so the constant is identical
to the jointly analytic version and has no extra factorial. -/
theorem mixedPartial_bound_of_smooth {n : ℕ} {F : (Fin n → ℂ) → V} {δ M r : ℝ}
    (hF : ContDiffOn ℂ ∞ F (l1Ball n δ))
    (hbound : ∀ z ∈ l1Ball n δ, ‖F z‖ ≤ M)
    (hr : 0 < r) (w : List (Fin n)) :
    ∀ z, l1Size z + (w.length : ℝ) * r < δ →
      ‖mixedPartial w F z‖ ≤ M / r ^ w.length := by
  induction w with
  | nil =>
      intro z hz
      simpa only [mixedPartial, List.length_nil, Nat.cast_zero, zero_mul, add_zero, pow_zero, div_one]
        using hbound z (by simpa only [l1Ball, mem_ofPred_eq, List.length_nil, Nat.cast_zero, zero_mul, add_zero] using hz)
  | cons i w ih =>
      intro z hz
      have hz0 : z ∈ l1Ball n δ := by
        have hn : 0 ≤ (List.length w : ℝ) := Nat.cast_nonneg _
        simp only [List.length_cons, Nat.cast_add, Nat.cast_one] at hz
        change l1Size z < δ
        nlinarith
      let P := mixedPartial w F
      let line : ℂ → Fin n → ℂ := fun t => z + t • Pi.single i 1
      have hline0 : line 0 = z := by simp [line]
      have hinside : ∀ t ∈ closedBall (0 : ℂ) r,
          l1Size (line t) + (w.length : ℝ) * r < δ := by
        intro t ht
        have ht' : ‖t‖ ≤ r := by simpa only [mem_closedBall, dist_zero_right] using ht
        have hs := l1Size_coordinate_shift z i t
        simp only [List.length_cons, Nat.cast_add, Nat.cast_one] at hz
        dsimp only [line]
        linarith
      have ha : DifferentiableOn ℂ (fun t => P (line t)) (closedBall (0 : ℂ) r) := by
        intro t ht
        have ht' : line t ∈ l1Ball n δ := by
          change l1Size (line t) < δ
          have hh := hinside t ht
          have hn : 0 ≤ (w.length : ℝ) * r := by positivity
          linarith
        have hp : DifferentiableAt ℂ P (line t) :=
          ((mixedPartial_smooth (isOpen_l1Ball n δ) hF w).differentiableOn (by simp)).differentiableAt
            ((isOpen_l1Ball n δ).mem_nhds ht')
        exact (hp.comp t (by dsimp [line]; fun_prop)).differentiableWithinAt
      have hdc : DiffContOnCl ℂ (fun t => P (line t)) (ball (0 : ℂ) r) := by
        apply DifferentiableOn.diffContOnCl
        rw [closure_ball _ hr.ne']
        exact ha
      have hsphere : ∀ t ∈ sphere (0 : ℂ) r, ‖P (line t)‖ ≤ M / r ^ w.length := by
        intro t ht
        exact ih (line t) (hinside t (sphere_subset_closedBall ht))
      have hc := Complex.norm_deriv_le_of_forall_mem_sphere_norm_le hr hdc hsphere
      have hdline : HasDerivAt line (Pi.single i (1 : ℂ)) 0 := by
        convert! ((hasDerivAt_id (0 : ℂ)).smul_const (Pi.single i (1 : ℂ))).const_add z using 1
        simp only [one_smul]
      have hP := (((mixedPartial_smooth (isOpen_l1Ball n δ) hF w).differentiableOn (by simp)).differentiableAt
        ((isOpen_l1Ball n δ).mem_nhds hz0)).hasFDerivAt
      have hP' : HasFDerivAt P (fderiv ℂ P z) (line 0) := by simpa only [hline0] using hP
      have hder := (hP'.comp_hasDerivAt 0 hdline).deriv
      change deriv (fun t => P (line t)) 0 = mixedPartial (i :: w) F z at hder
      rw [hder] at hc
      apply hc.trans_eq
      simp only [List.length_cons, pow_succ, div_div]

theorem mixedPartial_zero_bound_of_smooth {n : ℕ} {F : (Fin n → ℂ) → V} {δ M : ℝ}
    (hδ : 0 < δ) (hF : ContDiffOn ℂ ∞ F (l1Ball n δ))
    (hbound : ∀ z ∈ l1Ball n δ, ‖F z‖ ≤ M)
    (w : List (Fin n)) (hw : 0 < w.length) :
    ‖mixedPartial w F 0‖ ≤ M * ((2 * (w.length : ℝ)) / δ) ^ w.length := by
  have hwR : 0 < (w.length : ℝ) := Nat.cast_pos.mpr hw
  have hr : 0 < δ / (2 * (w.length : ℝ)) := by positivity
  have hbudget : l1Size (0 : Fin n → ℂ) + (w.length : ℝ) * (δ / (2 * (w.length : ℝ))) < δ := by
    rw [l1Size_zero, zero_add]
    have he : (w.length : ℝ) * (δ / (2 * (w.length : ℝ))) = δ / 2 := by field_simp
    rw [he]
    linarith
  have hh := mixedPartial_bound_of_smooth hF hbound hr w 0 hbudget
  apply hh.trans_eq
  rw [div_eq_mul_inv, ← inv_pow, inv_div]


theorem allCoordinatePartial_zero_bound_of_smooth {n : ℕ} (hn : 0 < n)
    {F : (Fin n → ℂ) → V} {δ M : ℝ}
    (hδ : 0 < δ) (hF : ContDiffOn ℂ ∞ F (l1Ball n δ))
    (hbound : ∀ z ∈ l1Ball n δ, ‖F z‖ ≤ M) :
    ‖allCoordinatePartial F 0‖ ≤ M * ((2 * (n : ℝ)) / δ)^n := by
  simpa only [allCoordinatePartial, List.length_ofFn] using
    mixedPartial_zero_bound_of_smooth hδ hF hbound (List.ofFn (fun i : Fin n => i))
      (by simpa only [List.length_ofFn] using hn)


end QIRadon.MixedCauchy
