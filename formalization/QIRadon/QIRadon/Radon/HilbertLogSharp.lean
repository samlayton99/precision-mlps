import QIRadon.Radon.HilbertAnalytic

/-! The sharp endpoint-log estimate used in the approved Hilbert certificate.
The proof integrates the derivative along a radial segment. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped Topology
namespace QIRadon.Radon

private theorem log_path_den_pos {L Q t : ℝ} (hQ0 : 0 < Q) (hQ : Q < L)
    (ht : t ∈ Icc (0:ℝ) 1) : 0 < L^2-t^2*Q^2 := by
  have hL : 0 < L := hQ0.trans hQ
  have htQ : 0 ≤ t*Q := mul_nonneg ht.1 hQ0.le
  have htQL : t*Q < L := lt_of_le_of_lt (by nlinarith [ht.2] : t*Q ≤ Q) hQ
  nlinarith

private theorem complex_log_path_den {L Q t : ℝ} (hQ0 : 0 < Q) (hQ : Q < L)
    (ht : t ∈ Icc (0:ℝ) 1) {z : ℂ} (hz : ‖z‖ ≤ Q) :
    L^2-t^2*Q^2 ≤ ‖(L:ℂ)^2-(t:ℂ)^2*z^2‖ := by
  have hL : 0 < L := hQ0.trans hQ
  have hn := norm_sub_norm_le ((L:ℂ)^2) ((t:ℂ)^2*z^2)
  simp only [norm_pow, norm_mul, Complex.norm_real, Real.norm_eq_abs,
    abs_of_pos hL, abs_of_nonneg ht.1] at hn
  have hnorm : ‖z‖^2 ≤ Q^2 := sq_le_sq₀ (norm_nonneg z) hQ0.le |>.mpr hz
  nlinarith [mul_le_mul_of_nonneg_left hnorm (sq_nonneg t)]

private theorem log_path_deriv_norm_le {L Q t : ℝ} (hQ0 : 0 < Q) (hQ : Q < L)
    (ht : t ∈ Icc (0:ℝ) 1) {z : ℂ} (hz : ‖z‖ ≤ Q) :
    ‖(((L:ℂ)+(t:ℂ)*z)⁻¹+((L:ℂ)-(t:ℂ)*z)⁻¹)*z‖ ≤
      2*L*Q/(L^2-t^2*Q^2) := by
  have hL : 0 < L := hQ0.trans hQ
  have htz : ‖(t:ℂ)*z‖ < L := by
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg ht.1]
    have hm := mul_le_mul ht.2 hz (norm_nonneg z) (by norm_num : (0:ℝ) ≤ 1)
    exact lt_of_le_of_lt (by simpa using hm) hQ
  have hp : (L:ℂ)+(t:ℂ)*z ≠ 0 := by
    intro heq
    have he : (t:ℂ)*z=-(L:ℂ) := by linear_combination heq
    rw [he, norm_neg, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hL] at htz
    exact lt_irrefl _ htz
  have hm : (L:ℂ)-(t:ℂ)*z ≠ 0 := by
    intro heq
    have he := sub_eq_zero.mp heq
    have hn := congrArg norm he
    simp only [Complex.norm_real, Real.norm_eq_abs, abs_of_pos hL] at hn
    linarith
  have hd := complex_log_path_den hQ0 hQ ht hz
  have hdpos := log_path_den_pos hQ0 hQ ht
  have hden : (L:ℂ)^2-(t:ℂ)^2*z^2 ≠ 0 := by
    intro heq
    rw [heq, norm_zero] at hd
    linarith
  have heq : (((L:ℂ)+(t:ℂ)*z)⁻¹+((L:ℂ)-(t:ℂ)*z)⁻¹)*z =
      (2*(L:ℂ)*z)/((L:ℂ)^2-(t:ℂ)^2*z^2) := by
    field_simp
    ring
  rw [heq, norm_div]
  have hnum : ‖2*(L:ℂ)*z‖ ≤ 2*L*Q := by
    simp only [norm_mul, Complex.norm_ofNat, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hL]
    exact mul_le_mul_of_nonneg_left hz (by positivity)
  exact (div_le_div_of_nonneg_right hnum (norm_nonneg _)).trans
    (div_le_div_of_nonneg_left (by positivity) hdpos hd)

/-- Exact approved logarithmic majorant for the complex endpoint correction. -/
theorem hilbertLog_norm_le_sharp {L Q : ℝ} (hQ0 : 0 < Q) (hQ : Q < L)
    {z : ℂ} (hz : ‖z‖ ≤ Q) :
    ‖hilbertLog L z‖ ≤ Real.log ((L+Q)/(L-Q)) := by
  let D : ℝ → ℂ := fun t => (((L:ℂ)+(t:ℂ)*z)⁻¹+((L:ℂ)-(t:ℂ)*z)⁻¹)*z
  let B : ℝ → ℝ := fun t => 2*L*Q/(L^2-t^2*Q^2)
  have hderiv : ∀ t ∈ Icc (0:ℝ) 1,
      HasDerivAt (fun t : ℝ => hilbertLog L ((t:ℂ)*z)) (D t) t := by
    intro t ht
    have htz : ‖(t:ℂ)*z‖ < L := by
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg ht.1]
      exact lt_of_le_of_lt ((mul_le_mul ht.2 hz (norm_nonneg z) (by norm_num)).trans_eq (one_mul Q)) hQ
    simpa only [Function.comp_def, id_eq, one_mul, D] using
      ((hasDerivAt_hilbertLog htz).comp (t:ℂ) ((hasDerivAt_id (t:ℂ)).mul_const z)).comp_ofReal
  have hDcont : ContinuousOn D (Icc (0:ℝ) 1) := by
    intro t ht
    have htz : ‖(t:ℂ)*z‖ < L := by
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg ht.1]
      exact lt_of_le_of_lt ((mul_le_mul ht.2 hz (norm_nonneg z) (by norm_num)).trans_eq (one_mul Q)) hQ
    have hp : (L:ℂ)+(t:ℂ)*z ≠ 0 := by
      intro heq
      have he : (t:ℂ)*z=-(L:ℂ) := by linear_combination heq
      rw [he, norm_neg, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (hQ0.trans hQ)] at htz
      exact lt_irrefl _ htz
    have hm : (L:ℂ)-(t:ℂ)*z ≠ 0 := by
      intro heq
      have he := sub_eq_zero.mp heq
      rw [← he, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (hQ0.trans hQ)] at htz
      exact lt_irrefl _ htz
    exact (((continuous_const.add (Complex.continuous_ofReal.mul_const z)).continuousAt.inv₀ hp).add
      ((continuous_const.sub (Complex.continuous_ofReal.mul_const z)).continuousAt.inv₀ hm)).mul
        continuousAt_const |>.continuousWithinAt
  have hBcont : ContinuousOn B (Icc (0:ℝ) 1) := by
    apply continuousOn_const.div
      (continuousOn_const.sub ((continuousOn_id.pow 2).mul_const (Q^2)))
    intro t ht
    exact (log_path_den_pos hQ0 hQ ht).ne'
  have hintB : IntervalIntegrable B volume 0 1 := by
    apply ContinuousOn.intervalIntegrable
    simpa only [uIcc_of_le (by norm_num : (0:ℝ) ≤ 1)] using hBcont
  have hintD : IntervalIntegrable D volume 0 1 := by
    apply ContinuousOn.intervalIntegrable
    simpa only [uIcc_of_le (by norm_num : (0:ℝ) ≤ 1)] using hDcont
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun t ht => hderiv t (by simpa using ht)) hintD
  have hbound : ‖∫ t in (0:ℝ)..1, D t‖ ≤ ∫ t in (0:ℝ)..1, B t := by
    apply intervalIntegral.norm_integral_le_of_norm_le (by norm_num) _ hintB
    filter_upwards with t
    intro ht
    exact log_path_deriv_norm_le hQ0 hQ ⟨ht.1.le, ht.2⟩ hz
  have hrealderiv : ∀ t ∈ Icc (0:ℝ) 1,
      HasDerivAt (fun t : ℝ => Real.log (L+t*Q)-Real.log (L-t*Q)) (B t) t := by
    intro t ht
    have hL : 0 < L := hQ0.trans hQ
    have hm : 0 < L-t*Q := by nlinarith [ht.2]
    have hp : 0 < L+t*Q := by nlinarith [ht.1]
    have hd := (((hasDerivAt_const t L).add ((hasDerivAt_id t).mul_const Q)).log hp.ne').sub
      (((hasDerivAt_const t L).sub ((hasDerivAt_id t).mul_const Q)).log hm.ne')
    have hcoef : B t = Q/(L+t*Q)+Q/(L-t*Q) := by
      dsimp [B]
      rw [div_add_div _ _ hp.ne' hm.ne']
      congr 1 <;> ring
    rw [hcoef]
    simpa only [Pi.add_apply, Pi.sub_apply, id_eq, zero_add, one_mul, zero_sub, neg_div, sub_neg_eq_add,
      Pi.sub_def] using hd
  have hFTCReal := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun t ht => hrealderiv t (by simpa using ht)) hintB
  rw [hFTC, hFTCReal] at hbound
  simpa [D, hilbertLog, Real.log_div (by linarith : L+Q ≠ 0) (by linarith : L-Q ≠ 0)] using hbound

/-- The sharper approved endpoint logarithm replaces the preliminary derivative-
supremum bound. At δ=4η the inner term is already better than R8. -/
theorem hilbertContinuation_norm_le_sharp {p : ℝ → ℂ} (hp : Integrable p)
    {P : ℂ → ℂ} {L δ B Q : ℝ} (hδ : 0 < δ) (hB : 0 ≤ B)
    (hQ0 : 0 < Q) (hQ : Q < L)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    {z : ℂ} (hz : z ∈ scalarTube L δ ∩ ball 0 Q) :
    ‖hilbertContinuation p P L z‖ ≤ Real.pi⁻¹ *
      (4*L*B/δ + Real.log ((L+Q)/(L-Q))*B + (∫ t : ℝ, ‖p t‖)/(L-Q)) := by
  have hzQ : ‖z‖ < Q := by simpa using hz.2
  have hin := hilbertInner_norm_le (hQ0.trans hQ).le hδ hB hP hbound hz.1
  have hout := hilbertOuter_norm_le hp hQ hzQ
  have hlog := hilbertLog_norm_le_sharp hQ0 hQ hzQ.le
  have hprod : ‖hilbertLog L z * P z‖ ≤ Real.log ((L+Q)/(L-Q))*B := by
    rw [norm_mul]
    exact mul_le_mul hlog (hbound z hz.1) (norm_nonneg _) ((norm_nonneg _).trans hlog)
  unfold hilbertContinuation
  rw [norm_mul, norm_inv, Complex.norm_real, Real.norm_eq_abs, abs_of_pos Real.pi_pos]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  exact (norm_add_le _ _).trans (add_le_add
    ((norm_add_le _ _).trans (add_le_add hin hprod)) hout)

end QIRadon.Radon
