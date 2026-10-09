import QIRadon.Radon.Analytic
import QIRadon.Radon.HilbertLocal

/-! Local holomorphy of the canonical Hilbert filter, obtained from its actual
finite singularity-subtracted formula. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform Topology
namespace QIRadon.Radon

theorem scalarTube_mono {S L δ η : ℝ} (hSL : S ≤ L) (hδη : δ ≤ η) :
    scalarTube S δ ⊆ scalarTube L η := by
  intro z hz
  simp only [scalarTube, mem_iUnion] at hz ⊢
  obtain ⟨s, hs, hz⟩ := hz
  exact ⟨s, ⟨by linarith [hs.1], by linarith [hs.2]⟩,
    lt_of_lt_of_le hz hδη⟩

theorem ofReal_mem_scalarTube {L δ t : ℝ} (hδ : 0 < δ) (ht : t ∈ Icc (-L) L) :
    (t:ℂ) ∈ scalarTube L δ := by
  simp only [scalarTube, mem_iUnion]
  exact ⟨t, ht, mem_ball_self hδ⟩

/-- A bounded holomorphic function has uniformly controlled divided differences
from every real point in its tube, including the removable diagonal. -/
theorem norm_dslope_on_scalarTube {P : ℂ → ℂ} {L δ B : ℝ}
    (hδ : 0 < δ) (hB : 0 ≤ B)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    {t : ℝ} (ht : t ∈ Icc (-L) L) {z : ℂ} (hz : z ∈ scalarTube L δ) :
    ‖dslope P (t:ℂ) z‖ ≤ 2*B/δ := by
  have hball : ball (t:ℂ) δ ⊆ scalarTube L δ := by
    intro y hy
    simp only [scalarTube, mem_iUnion]
    exact ⟨t, ht, hy⟩
  have htV := ofReal_mem_scalarTube hδ ht
  by_cases hnear : z ∈ ball (t:ℂ) δ
  · apply Complex.norm_dslope_le_div_of_mapsTo_ball (hP.mono hball) _ hnear
    intro y hy
    rw [mem_closedBall_iff_norm]
    exact (norm_sub_le _ _).trans (by linarith [hbound y (hball hy), hbound (t:ℂ) htV])
  · have hdist : δ ≤ ‖z-(t:ℂ)‖ := by simpa only [mem_ball_iff_norm, not_lt] using hnear
    have hne : z ≠ (t:ℂ) := by intro h; simp [h] at hdist; linarith
    rw [dslope_of_ne P hne, slope_def_field, norm_div]
    exact (div_le_div_of_nonneg_left (by positivity : 0 ≤ 2*B) hδ hdist).trans'
      (div_le_div_of_nonneg_right ((norm_sub_le _ _).trans (by
        linarith [hbound z hz, hbound (t:ℂ) htV])) (norm_nonneg _))

private theorem ae_real_ne_complex (z : ℂ) (μ : Measure ℝ) [NullSingletonClass μ] :
    ∀ᵐ (t : ℝ) ∂μ, z ≠ (t:ℂ) := by
  filter_upwards [μ.ae_ne z.re] with t ht
  intro heq
  have h := congrArg Complex.re heq
  simp only [Complex.ofReal_re] at h
  exact ht h.symm

private theorem dslope_parameter_measurable {P : ℂ → ℂ} {L δ : ℝ}
    (hδ : 0 < δ) (hP : DifferentiableOn ℂ P (scalarTube L δ)) (z : ℂ) :
    AEStronglyMeasurable (fun t : ℝ => dslope P (t:ℂ) z) (volume.restrict (Icc (-L) L)) := by
  have hp : ContinuousOn (fun t : ℝ => P (t:ℂ)) (Icc (-L) L) :=
    hP.continuousOn.comp Complex.continuous_ofReal.continuousOn
      (fun t ht => ofReal_mem_scalarTube hδ ht)
  have hm : AEStronglyMeasurable (fun t : ℝ => P (t:ℂ)) (volume.restrict (Icc (-L) L)) :=
    hp.aestronglyMeasurable measurableSet_Icc
  have hratio : AEStronglyMeasurable
      (fun t : ℝ => (P z-P (t:ℂ))/(z-(t:ℂ))) (volume.restrict (Icc (-L) L)) :=
    (aestronglyMeasurable_const.sub hm).div₀
      ((continuous_const.sub Complex.continuous_ofReal).aestronglyMeasurable)
  apply hratio.congr
  filter_upwards [ae_real_ne_complex z (volume.restrict (Icc (-L) L))] with t ht
  rw [dslope_of_ne P ht, slope_def_field]

/-- The inner finite integral is holomorphic; diagonal singularities are removed
by `dslope`, whose analyticity is proved by the removable-singularity theorem. -/
theorem hilbertInner_analytic {P : ℂ → ℂ} {L δ B : ℝ}
    (hδ : 0 < δ) (hB : 0 ≤ B)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B) :
    AnalyticOnNhd ℂ (fun z => -(∫ t in Icc (-L) L, dslope P (t:ℂ) z))
      (scalarTube L δ) := by
  have h := analyticOnNhd_integral_of_dominated (isOpen_scalarTube L δ)
    (fun z _ => dslope_parameter_measurable hδ hP z)
    (M := fun _ : ℝ => 2*B/δ) ?_ ?_ (integrable_const _)
  · exact h.neg
  · filter_upwards [ae_restrict_mem measurableSet_Icc] with t ht
    exact (Complex.differentiableOn_dslope
      ((isOpen_scalarTube L δ).mem_nhds (ofReal_mem_scalarTube hδ ht))).mpr hP
  · filter_upwards [ae_restrict_mem measurableSet_Icc] with t ht
    intro z hz
    exact norm_dslope_on_scalarTube hδ hB hP hbound ht hz

/-- Uniform inner-term bound, including the actual interval length. -/
theorem hilbertInner_norm_le {P : ℂ → ℂ} {L δ B : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hB : 0 ≤ B)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    {z : ℂ} (hz : z ∈ scalarTube L δ) :
    ‖-(∫ t in Icc (-L) L, dslope P (t:ℂ) z)‖ ≤ 4*L*B/δ := by
  rw [norm_neg]
  have h := norm_integral_le_of_norm_le_const (μ := volume.restrict (Icc (-L) L)) (C := 2*B/δ)
    (ae_restrict_mem measurableSet_Icc |>.mono (fun t ht =>
      norm_dslope_on_scalarTube hδ hB hP hbound ht hz))
  have hmass : (volume.restrict (Icc (-L) L)).real univ = 2*L := by
    simp only [Measure.real, Measure.restrict_apply_univ, Real.volume_Icc]
    rw [ENNReal.toReal_ofReal (by linarith)]
    ring
  rw [hmass] at h
  convert h using 1
  ring

theorem outside_complex_distance {L Q t : ℝ} (_hQ : Q < L)
    {z : ℂ} (hz : ‖z‖ < Q) (ht : t ∉ Icc (-L) L) :
    L-Q ≤ ‖z-(t:ℂ)‖ := by
  have habs : L ≤ |t| := by
    have ht' : t < -L ∨ L < t := by simpa only [mem_Icc, not_and_or, not_le] using ht
    rcases ht' with ht | ht
    · exact le_trans (by linarith) (neg_le_abs t)
    · exact le_trans ht.le (le_abs_self t)
  have hn := norm_sub_norm_le (t:ℂ) z
  rw [Complex.norm_real, Real.norm_eq_abs, norm_sub_rev] at hn
  linarith

/-- The outer contribution is holomorphic on any disk that stays strictly inside
its excluded real interval. -/
theorem hilbertOuter_analytic {p : ℝ → ℂ} (hp : Integrable p)
    {L Q : ℝ} (hQ : Q < L) :
    AnalyticOnNhd ℂ (fun z => ∫ t in (Icc (-L) L)ᶜ, p t/(z-(t:ℂ)))
      (ball 0 Q) := by
  apply analyticOnNhd_integral_of_dominated isOpen_ball
    (M := fun t : ℝ => ‖p t‖/(L-Q))
  · intro z _
    exact hp.aestronglyMeasurable.restrict.div₀
      ((continuous_const.sub Complex.continuous_ofReal).aestronglyMeasurable)
  · filter_upwards [ae_restrict_mem measurableSet_Icc.compl] with t ht
    intro z hz
    have hz' : ‖z‖ < Q := by simpa only [mem_ball_zero_iff] using hz
    have hd := outside_complex_distance hQ hz' ht
    have hne : z-(t:ℂ) ≠ 0 := by intro heq; simp [heq] at hd; linarith
    exact (differentiableAt_const (p t)).div
      ((differentiableAt_id).sub_const (t:ℂ)) hne |>.differentiableWithinAt
  · filter_upwards [ae_restrict_mem measurableSet_Icc.compl] with t ht
    intro z hz
    rw [norm_div]
    have hz' : ‖z‖ < Q := by simpa only [mem_ball_zero_iff] using hz
    exact div_le_div_of_nonneg_left (norm_nonneg (p t)) (by linarith)
      (outside_complex_distance hQ hz' ht)
  · exact (hp.norm.div_const (L-Q)).restrict

/-- The outer term is controlled by the real profile's actual L1 norm. -/
theorem hilbertOuter_norm_le {p : ℝ → ℂ} (hp : Integrable p)
    {L Q : ℝ} (hQ : Q < L) {z : ℂ} (hz : ‖z‖ < Q) :
    ‖∫ t in (Icc (-L) L)ᶜ, p t/(z-(t:ℂ))‖ ≤
      (∫ t : ℝ, ‖p t‖)/(L-Q) := by
  have hb : ∀ᵐ t ∂volume.restrict (Icc (-L) L)ᶜ,
      ‖p t/(z-(t:ℂ))‖ ≤ ‖p t‖/(L-Q) := by
    filter_upwards [ae_restrict_mem measurableSet_Icc.compl] with t ht
    rw [norm_div]
    exact div_le_div_of_nonneg_left (norm_nonneg (p t)) (by linarith)
      (outside_complex_distance hQ hz ht)
  have h := norm_integral_le_of_norm_le ((hp.norm.div_const (L-Q)).restrict) hb
  rw [integral_div] at h
  exact h.trans (div_le_div_of_nonneg_right
    (setIntegral_le_integral hp.norm (Filter.Eventually.of_forall (fun t => norm_nonneg (p t))))
    (by linarith))

/-- Holomorphic endpoint correction in the finite Hilbert formula. -/
def hilbertLog (L : ℝ) (z : ℂ) : ℂ :=
  Complex.log ((L:ℂ)+z)-Complex.log ((L:ℂ)-z)

private theorem log_arguments_in_slit {L : ℝ} {z : ℂ} (hz : ‖z‖ < L) :
    (L:ℂ)+z ∈ Complex.slitPlane ∧ (L:ℂ)-z ∈ Complex.slitPlane := by
  have hre := Complex.abs_re_le_norm z
  constructor
  · apply Complex.mem_slitPlane_iff.mpr
    left
    simp only [Complex.add_re, Complex.ofReal_re]
    have := neg_abs_le z.re
    linarith
  · apply Complex.mem_slitPlane_iff.mpr
    left
    simp only [Complex.sub_re, Complex.ofReal_re]
    have := le_abs_self z.re
    linarith

theorem hasDerivAt_hilbertLog {L : ℝ} {z : ℂ} (hz : ‖z‖ < L) :
    HasDerivAt (hilbertLog L) (((L:ℂ)+z)⁻¹+((L:ℂ)-z)⁻¹) z := by
  obtain ⟨hp, hm⟩ := log_arguments_in_slit hz
  have hd := (((hasDerivAt_id z).const_add (L:ℂ)).clog hp).sub
    (((hasDerivAt_const z (L:ℂ)).sub (hasDerivAt_id z)).clog hm)
  unfold hilbertLog
  convert hd using 1 <;> first | rfl | simp [div_eq_mul_inv]

theorem hilbertLog_analytic {L Q : ℝ} (hQ : Q ≤ L) :
    AnalyticOnNhd ℂ (hilbertLog L) (ball 0 Q) := by
  apply DifferentiableOn.analyticOnNhd _ isOpen_ball
  intro z hz
  exact (hasDerivAt_hilbertLog (lt_of_lt_of_le (by simpa using hz) hQ)).differentiableAt.differentiableWithinAt

/-- A direct derivative bound gives a simple explicit endpoint-term estimate. -/
theorem hilbertLog_norm_le {L Q : ℝ} (hQ0 : 0 < Q) (hQ : Q < L)
    {z : ℂ} (hz : ‖z‖ < Q) :
    ‖hilbertLog L z‖ ≤ 2*Q/(L-Q) := by
  have hL : 0 < L := hQ0.trans hQ
  have hderiv : ∀ w ∈ ball (0:ℂ) Q,
      HasDerivWithinAt (hilbertLog L) (((L:ℂ)+w)⁻¹+((L:ℂ)-w)⁻¹) (ball 0 Q) w := by
    intro w hw
    exact (hasDerivAt_hilbertLog (lt_trans (by simpa using hw) hQ)).hasDerivWithinAt
  have hb : ∀ w ∈ ball (0:ℂ) Q,
      ‖((L:ℂ)+w)⁻¹+((L:ℂ)-w)⁻¹‖ ≤ 2/(L-Q) := by
    intro w hw
    have hw' : ‖w‖ < Q := by simpa using hw
    have hminus : L-Q ≤ ‖(L:ℂ)-w‖ := by
      have hn := norm_sub_norm_le (L:ℂ) w
      simp only [Complex.norm_real, Real.norm_eq_abs, abs_of_pos hL] at hn
      linarith
    have hplus : L-Q ≤ ‖(L:ℂ)+w‖ := by
      have hn := norm_sub_norm_le (L:ℂ) (-w)
      simp only [Complex.norm_real, Real.norm_eq_abs, abs_of_pos hL, norm_neg, sub_neg_eq_add] at hn
      linarith
    calc
      _ ≤ ‖((L:ℂ)+w)⁻¹‖+‖((L:ℂ)-w)⁻¹‖ := norm_add_le _ _
      _ = ‖(L:ℂ)+w‖⁻¹+‖(L:ℂ)-w‖⁻¹ := by rw [norm_inv, norm_inv]
      _ ≤ (L-Q)⁻¹+(L-Q)⁻¹ := add_le_add
        (inv_anti₀ (by linarith) hplus) (inv_anti₀ (by linarith) hminus)
      _ = 2/(L-Q) := by ring
  have hm := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le hderiv hb
    (convex_ball (0:ℂ) Q) (mem_ball_self hQ0) (show z ∈ ball (0:ℂ) Q by simpa using hz)
  simp only [hilbertLog, add_zero, sub_zero, sub_self] at hm
  have hpos : 0 ≤ 2/(L-Q) := by positivity
  change ‖Complex.log ((L:ℂ)+z)-Complex.log ((L:ℂ)-z)‖ ≤ _
  calc
    _ ≤ 2/(L-Q)*‖z‖ := hm
    _ ≤ 2/(L-Q)*Q := mul_le_mul_of_nonneg_left hz.le hpos
    _ = 2*Q/(L-Q) := by ring

/-- Explicit locally holomorphic expression for the Hilbert transform. The outer
integral uses the actual real profile; it does not require globally analytic data. -/
def hilbertContinuation (p : ℝ → ℂ) (P : ℂ → ℂ) (L : ℝ) (z : ℂ) : ℂ :=
  (Real.pi:ℂ)⁻¹ * (-(∫ t in Icc (-L) L, dslope P (t:ℂ) z) +
    hilbertLog L z * P z + ∫ t in (Icc (-L) L)ᶜ, p t/(z-(t:ℂ)))

/-- The finite expression is holomorphic wherever the original local complex
extension exists and the outer denominator remains separated. -/
theorem hilbertContinuation_analytic {p : ℝ → ℂ} (hp : Integrable p)
    {P : ℂ → ℂ} {L δ B Q : ℝ} (hδ : 0 < δ) (hB : 0 ≤ B) (hQ : Q < L)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B) :
    AnalyticOnNhd ℂ (hilbertContinuation p P L)
      (scalarTube L δ ∩ ball 0 Q) := by
  have hin := (hilbertInner_analytic hδ hB hP hbound).mono
    (show scalarTube L δ ∩ ball (0:ℂ) Q ⊆ scalarTube L δ from inter_subset_left)
  have hout := (hilbertOuter_analytic hp hQ).mono
    (show scalarTube L δ ∩ ball (0:ℂ) Q ⊆ ball (0:ℂ) Q from inter_subset_right)
  have hlog := (hilbertLog_analytic hQ.le).mono
    (show scalarTube L δ ∩ ball (0:ℂ) Q ⊆ ball (0:ℂ) Q from inter_subset_right)
  have hPc := (hP.analyticOnNhd (isOpen_scalarTube L δ)).mono
    (show scalarTube L δ ∩ ball (0:ℂ) Q ⊆ scalarTube L δ from inter_subset_left)
  exact analyticOnNhd_const.mul ((hin.add (hlog.mul hPc)).add hout)

/-- The continuation has an explicit bound in terms of local analytic data and
an actual real-profile L1 tail; there is no analyticity assumption on that tail. -/
theorem hilbertContinuation_norm_le {p : ℝ → ℂ} (hp : Integrable p)
    {P : ℂ → ℂ} {L δ B Q : ℝ} (hδ : 0 < δ) (hB : 0 ≤ B)
    (hQ0 : 0 < Q) (hQ : Q < L)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    {z : ℂ} (hz : z ∈ scalarTube L δ ∩ ball 0 Q) :
    ‖hilbertContinuation p P L z‖ ≤ Real.pi⁻¹ *
      (4*L*B/δ + (2*Q/(L-Q))*B + (∫ t : ℝ, ‖p t‖)/(L-Q)) := by
  have hzQ : ‖z‖ < Q := by simpa using hz.2
  have hin := hilbertInner_norm_le (hQ0.trans hQ).le hδ hB hP hbound hz.1
  have hout := hilbertOuter_norm_le hp hQ hzQ
  have hlog := hilbertLog_norm_le hQ0 hQ hzQ
  have hprod : ‖hilbertLog L z * P z‖ ≤ (2*Q/(L-Q))*B := by
    rw [norm_mul]
    exact mul_le_mul hlog (hbound z hz.1) (norm_nonneg _) (by positivity)
  unfold hilbertContinuation
  rw [norm_mul, norm_inv, Complex.norm_real, Real.norm_eq_abs, abs_of_pos Real.pi_pos]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  exact (norm_add_le _ _).trans (add_le_add
    ((norm_add_le _ _).trans (add_le_add hin hprod)) hout)

/-- Identification on the real interval closes the analytic bridge: this is a
continuation of the canonical Fourier Hilbert transform, not a new profile. -/
theorem hilbertContinuation_eq_hilbertFourier {p : ℝ → ℂ}
    (hp : Integrable p) (hphat : Integrable (𝓕 p))
    {P : ℂ → ℂ} {L δ B : ℝ} (hδ : 0 < δ) (hB : 0 ≤ B)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    (hagree : ∀ t ∈ Icc (-L) L, p t=P (t:ℂ))
    {s : ℝ} (hs : |s| < L) :
    hilbertContinuation p P L (s:ℂ) = hilbertFourier p s := by
  have hsI : s ∈ Icc (-L) L := abs_le.mp hs.le
  have hsV := ofReal_mem_scalarTube hδ hsI
  have hLip : ∀ t ∈ Icc (-L) L, ‖p t-p s‖ ≤ (2*B/δ)*|s-t| := by
    intro t ht
    rw [hagree t ht, hagree s hsI, ← sub_smul_dslope P (s:ℂ) (t:ℂ), norm_smul]
    have hn := norm_dslope_on_scalarTube hδ hB hP hbound hsI (ofReal_mem_scalarTube hδ ht)
    calc
      _ ≤ ‖(t:ℂ)-(s:ℂ)‖*(2*B/δ) := mul_le_mul_of_nonneg_left hn (norm_nonneg _)
      _ = (2*B/δ)*|s-t| := by
        rw [← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs, abs_sub_comm]
        ring
  rw [hilbertFourier_local_formula hp hphat hs (by positivity) hLip]
  have hinner : -(∫ t in Icc (-L) L, dslope P (t:ℂ) (s:ℂ)) =
      ∫ t in Icc (-L) L, (p t-p s)/((s-t:ℝ):ℂ) := by
    rw [← integral_neg]
    apply integral_congr_ae
    filter_upwards [ae_restrict_mem measurableSet_Icc,
      ae_real_ne_complex (s:ℂ) (volume.restrict (Icc (-L) L))] with t ht hne
    rw [dslope_of_ne P hne, slope_def_field, hagree t ht, hagree s hsI,
      Complex.ofReal_sub]
    ring
  have hplus : 0 ≤ L+s := by have := neg_abs_le s; linarith
  have hminus : 0 ≤ L-s := by have := le_abs_self s; linarith
  have hlog : hilbertLog L (s:ℂ) = ((Real.log (L+s)-Real.log (L-s):ℝ):ℂ) := by
    simp only [hilbertLog, ← Complex.ofReal_add, ← Complex.ofReal_sub,
      ← Complex.ofReal_log hplus, ← Complex.ofReal_log hminus]
  simp only [hilbertContinuation, hinner, hlog, hagree s hsI, Complex.ofReal_sub]

end QIRadon.Radon
