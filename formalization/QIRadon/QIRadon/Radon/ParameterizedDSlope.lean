import QIRadon.Radon.HilbertAnalytic
import QIRadon.Radon.ParameterIntegral
import Mathlib.Analysis.Complex.RealDeriv

/-! The actual removable divided difference under additional complex parameters.
Its segment-integral identity proves joint smoothness across the diagonal. -/

noncomputable section
open MeasureTheory Set Metric Filter
open scoped Topology ContDiff
namespace QIRadon.Radon

theorem scalarTube_segment {L δ t u : ℝ} (_hδ : 0 < δ)
    (ht : t ∈ Icc (-L) L) (hu : u ∈ Icc (0:ℝ) 1)
    {z : ℂ} (hz : z ∈ scalarTube L δ) :
    (t:ℂ)+(u:ℂ)*(z-(t:ℂ)) ∈ scalarTube L δ := by
  simp only [scalarTube, mem_iUnion] at hz ⊢
  obtain ⟨s, hs, hz⟩ := hz
  have hb : t+u*(s-t) ∈ Icc (-L) L := by
    constructor <;> nlinarith [mul_nonneg hu.1 (sub_nonneg.mpr hs.1),
      mul_nonneg (sub_nonneg.mpr hu.2) (sub_nonneg.mpr ht.1),
      mul_nonneg hu.1 (sub_nonneg.mpr hs.2),
      mul_nonneg (sub_nonneg.mpr hu.2) (sub_nonneg.mpr ht.2)]
  refine ⟨t+u*(s-t), hb, ?_⟩
  rw [mem_ball_iff_norm]
  have he : (t:ℂ)+(u:ℂ)*(z-(t:ℂ))-((t+u*(s-t):ℝ):ℂ) = (u:ℂ)*(z-(s:ℂ)) := by
    push_cast
    ring
  rw [he, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hu.1]
  have hz' : ‖z-(s:ℂ)‖ < δ := mem_ball_iff_norm.mp hz
  exact (mul_le_of_le_one_left (norm_nonneg _) hu.2).trans_lt hz'

/-- The divided difference is the average derivative on the actual segment,
including z=t where the integral equals the derivative itself. -/
theorem dslope_segment_integral {P : ℂ → ℂ} {L δ : ℝ}
    (hδ : 0 < δ) (hP : DifferentiableOn ℂ P (scalarTube L δ))
    {t : ℝ} (ht : t ∈ Icc (-L) L) {z : ℂ} (hz : z ∈ scalarTube L δ) :
    dslope P (t:ℂ) z =
      ∫ u in (0:ℝ)..1, deriv P ((t:ℂ)+(u:ℂ)*(z-(t:ℂ))) := by
  by_cases heq : z = (t:ℂ)
  · subst z
    simp
  let line : ℝ → ℂ := fun u => (t:ℂ)+(u:ℂ)*(z-(t:ℂ))
  have hline : ∀ u ∈ Icc (0:ℝ) 1, line u ∈ scalarTube L δ :=
    fun u hu => scalarTube_segment hδ ht hu hz
  have hp (u : ℝ) (hu : u ∈ Icc (0:ℝ) 1) : AnalyticAt ℂ P (line u) :=
    hP.analyticAt ((isOpen_scalarTube L δ).mem_nhds (hline u hu))
  have hd : ∀ u ∈ Set.uIcc (0:ℝ) 1,
      HasDerivAt (fun v : ℝ => P (line v)) (deriv P (line u)*(z-(t:ℂ))) u := by
    intro u hu
    have hu' : u ∈ Icc (0:ℝ) 1 := by simpa using hu
    have hcomplex := (hp u hu').differentiableAt.hasDerivAt.comp (u:ℂ)
      (((hasDerivAt_id (u:ℂ)).mul_const (z-(t:ℂ))).const_add (t:ℂ))
    simpa only [line, Function.comp_def, id_eq, one_mul] using! hcomplex.comp_ofReal
  have hc : ContinuousOn (fun u : ℝ => deriv P (line u)) (Icc (0:ℝ) 1) := by
    intro u hu
    exact ((hp u hu).deriv.continuousAt.comp
      (show ContinuousAt line u by fun_prop)).continuousWithinAt
  have hi := hc.intervalIntegrable_of_Icc (μ := volume) (by norm_num : (0:ℝ) ≤ 1)
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt hd (hi.mul_const (z-(t:ℂ)))
  rw [intervalIntegral.integral_mul_const] at hFTC
  simp only [line, Complex.ofReal_one, one_mul, add_sub_cancel, Complex.ofReal_zero,
    zero_mul, add_zero] at hFTC
  rw [dslope_of_ne P heq, slope_def_field]
  exact (eq_div_iff (sub_ne_zero.mpr heq)).mpr hFTC |>.symm

variable {A : Type*} [NormedAddCommGroup A] [NormedSpace ℂ A] [FiniteDimensional ℂ A]

def scalarPartial (P : A × ℂ → ℂ) (p : A × ℂ) : ℂ :=
  (fderiv ℂ P p) (0,1)

omit [FiniteDimensional ℂ A] in
theorem scalarPartial_eq_deriv {P : A × ℂ → ℂ} {U : Set A} {L δ : ℝ}
    (hU : IsOpen U) (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ))
    {p : A × ℂ} (hp : p ∈ U ×ˢ scalarTube L δ) :
    scalarPartial P p = deriv (fun z => P (p.1,z)) p.2 := by
  have h := (hP.differentiableOn (by simp)).differentiableAt
    ((hU.prod (isOpen_scalarTube L δ)).mem_nhds hp)
  have hd : HasDerivAt (fun z : ℂ => (p.1,z)) (0,1) p.2 := by
    simpa using (hasDerivAt_const p.2 p.1).prodMk (hasDerivAt_id p.2)
  exact (h.hasFDerivAt.comp_hasDerivAt p.2 hd).deriv.symm

theorem scalarPartial_smooth {P : A × ℂ → ℂ} {U : Set A} {L δ : ℝ}
    (hU : IsOpen U) (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ)) :
    ContDiffOn ℂ ∞ (scalarPartial P) (U ×ˢ scalarTube L δ) := by
  exact (contDiffOn_clm_apply.mp
    (hP.fderiv_of_isOpen (hU.prod (isOpen_scalarTube L δ)) (by simp))) (0,1)

theorem scalarTube_closedBall_half {L δ : ℝ} {z : ℂ}
    (hz : z ∈ scalarTube L (δ/2)) : closedBall z (δ/2) ⊆ scalarTube L δ := by
  simp only [scalarTube, mem_iUnion] at hz
  obtain ⟨s, hs, hz⟩ := hz
  intro w hw
  simp only [scalarTube, mem_iUnion]
  refine ⟨s, hs, ?_⟩
  have h1 : dist z (s:ℂ) < δ/2 := hz
  have h2 : dist w z ≤ δ/2 := hw
  have h3 := dist_triangle w z (s:ℂ)
  change dist w (s:ℂ) < δ
  linarith

omit [FiniteDimensional ℂ A] in
theorem scalarPartial_bound {P : A × ℂ → ℂ} {U : Set A} {L δ B : ℝ}
    (hU : IsOpen U) (hδ : 0 < δ)
    (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ))
    (hb : ∀ p ∈ U ×ˢ scalarTube L δ, ‖P p‖ ≤ B)
    {p : A × ℂ} (hp : p ∈ U ×ˢ scalarTube L (δ/2)) :
    ‖scalarPartial P p‖ ≤ 2*B/δ := by
  have hsub := scalarTube_closedBall_half hp.2
  have hps : p ∈ U ×ˢ scalarTube L δ :=
    ⟨hp.1, scalarTube_mono le_rfl (by linarith) hp.2⟩
  rw [scalarPartial_eq_deriv hU hP hps]
  have hdiff : DifferentiableOn ℂ (fun z => P (p.1,z)) (scalarTube L δ) := by
    intro z hz
    have hd : DifferentiableAt ℂ P (p.1,z) :=
      (hP.differentiableOn (by simp)).differentiableAt
        ((hU.prod (isOpen_scalarTube L δ)).mem_nhds ⟨hp.1,hz⟩)
    exact (hd.comp z (show DifferentiableAt ℂ (fun y : ℂ => (p.1,y)) z by fun_prop)).differentiableWithinAt
  have hdc : DiffContOnCl ℂ (fun z => P (p.1,z)) (ball p.2 (δ/2)) := by
    apply DifferentiableOn.diffContOnCl
    rw [closure_ball _ (by positivity : δ/2 ≠ 0)]
    exact hdiff.mono hsub
  have hh := Complex.norm_deriv_le_of_forall_mem_sphere_norm_le
    (by positivity : 0 < δ/2) hdc (fun z hz => hb (p.1,z) ⟨hp.1,hsub (sphere_subset_closedBall hz)⟩)
  convert hh using 1
  ring

/-- Joint smoothness of the actual divided difference at one real base point. -/
theorem parameterized_dslope_smooth {P : A × ℂ → ℂ} {U : Set A} {L δ B t : ℝ}
    (hU : IsOpen U) (hδ : 0 < δ)
    (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ))
    (hb : ∀ p ∈ U ×ˢ scalarTube L δ, ‖P p‖ ≤ B)
    (ht : t ∈ Icc (-L) L) :
    ContDiffOn ℂ ∞ (fun p : A × ℂ => dslope (fun z => P (p.1,z)) (t:ℂ) p.2)
      (U ×ˢ scalarTube L (δ/2)) := by
  let D := U ×ˢ scalarTube L (δ/2)
  let V := U ×ˢ scalarTube L δ
  let f : (A × ℂ) → ℝ → ℂ := fun p u =>
    scalarPartial P (p.1,(t:ℂ)+(u:ℂ)*(p.2-(t:ℂ)))
  have hmap (u : ℝ) (hu : u ∈ Icc (0:ℝ) 1) (p : A × ℂ) (hp : p ∈ D) :
      (p.1,(t:ℂ)+(u:ℂ)*(p.2-(t:ℂ))) ∈ D :=
    ⟨hp.1,scalarTube_segment (half_pos hδ) ht hu hp.2⟩
  have hDV : D ⊆ V := fun p hp => ⟨hp.1,scalarTube_mono le_rfl (by linarith) hp.2⟩
  have hpart := scalarPartial_smooth hU hP
  have hmeas (p : A × ℂ) (hp : p ∈ D) :
      AEStronglyMeasurable (f p) (volume.restrict (Icc (0:ℝ) 1)) := by
    apply ContinuousOn.aestronglyMeasurable _ measurableSet_Icc
    intro u hu
    have hcont := hpart.continuousOn.continuousAt
      ((hU.prod (isOpen_scalarTube L δ)).mem_nhds (hDV (hmap u hu p hp)))
    exact (hcont.comp (f := fun v : ℝ => (p.1,(t:ℂ)+(v:ℂ)*(p.2-(t:ℂ))))
      (by fun_prop)).continuousWithinAt
  have hsmooth : ∀ᵐ u ∂(volume.restrict (Icc (0:ℝ) 1)),
      ContDiffOn ℂ ∞ (fun p => f p u) D := by
    filter_upwards [ae_restrict_mem measurableSet_Icc] with u hu
    apply hpart.comp (f := fun p : A × ℂ => (p.1,(t:ℂ)+(u:ℂ)*(p.2-(t:ℂ)))) (by fun_prop)
    exact fun p hp => hDV (hmap u hu p hp)
  have hbound : ∀ᵐ u ∂(volume.restrict (Icc (0:ℝ) 1)), ∀ p ∈ D, ‖f p u‖ ≤ 2*B/δ := by
    filter_upwards [ae_restrict_mem measurableSet_Icc] with u hu
    intro p hp
    exact scalarPartial_bound hU hδ hP hb (hmap u hu p hp)
  have hi := ParameterIntegral.smoothOn_integral_of_dominated
    (hU.prod (isOpen_scalarTube L (δ/2))) hmeas hsmooth hbound (integrable_const (2*B/δ))
  apply hi.congr
  intro p hp
  have hpV := hDV hp
  have hpDiff : DifferentiableOn ℂ (fun z => P (p.1,z)) (scalarTube L δ) := by
    intro z hz
    have hd : DifferentiableAt ℂ P (p.1,z) :=
      (hP.differentiableOn (by simp)).differentiableAt
        ((hU.prod (isOpen_scalarTube L δ)).mem_nhds ⟨hp.1,hz⟩)
    exact (hd.comp z (show DifferentiableAt ℂ (fun y : ℂ => (p.1,y)) z by fun_prop)).differentiableWithinAt
  rw [dslope_segment_integral hδ hpDiff ht hpV.2,
    intervalIntegral.integral_of_le (by norm_num : (0:ℝ) ≤ 1), ← integral_Icc_eq_integral_Ioc]
  apply integral_congr_ae
  filter_upwards [ae_restrict_mem measurableSet_Icc] with u hu
  exact (scalarPartial_eq_deriv hU hP (hDV (hmap u hu p hp))).symm

/-- The complete finite inner Hilbert term is jointly complex smooth in its
parameters and offset, including every removed diagonal singularity. -/
theorem parameterized_hilbertInner_smooth
    {P : A × ℂ → ℂ} {U : Set A} {L δ B : ℝ}
    (hU : IsOpen U) (hδ : 0 < δ) (hB : 0 ≤ B)
    (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ))
    (hb : ∀ p ∈ U ×ˢ scalarTube L δ, ‖P p‖ ≤ B) :
    ContDiffOn ℂ ∞
      (fun p : A × ℂ => -(∫ t in Icc (-L) L, dslope (fun z => P (p.1,z)) (t:ℂ) p.2))
      (U ×ˢ scalarTube L (δ/2)) := by
  have hslice (a : A) (ha : a ∈ U) :
      DifferentiableOn ℂ (fun z => P (a,z)) (scalarTube L δ) := by
    intro z hz
    have hd : DifferentiableAt ℂ P (a,z) :=
      (hP.differentiableOn (by simp)).differentiableAt
        ((hU.prod (isOpen_scalarTube L δ)).mem_nhds ⟨ha,hz⟩)
    exact (hd.comp z (show DifferentiableAt ℂ (fun y : ℂ => (a,y)) z by fun_prop)).differentiableWithinAt
  have hmeas (p : A × ℂ) (hp : p ∈ U ×ˢ scalarTube L (δ/2)) :
      AEStronglyMeasurable (fun t : ℝ => dslope (fun z => P (p.1,z)) (t:ℂ) p.2)
        (volume.restrict (Icc (-L) L)) := by
    have hcont : ContinuousOn (fun t : ℝ => P (p.1,(t:ℂ))) (Icc (-L) L) :=
      (hslice p.1 hp.1).continuousOn.comp Complex.continuous_ofReal.continuousOn
        (fun t ht => ofReal_mem_scalarTube hδ ht)
    have hh : AEStronglyMeasurable
        (fun t : ℝ => (P p-P (p.1,(t:ℂ)))/(p.2-(t:ℂ)))
        (volume.restrict (Icc (-L) L)) :=
      (aestronglyMeasurable_const.sub (hcont.aestronglyMeasurable measurableSet_Icc)).div₀
      ((continuous_const.sub Complex.continuous_ofReal).aestronglyMeasurable)
    apply hh.congr
    filter_upwards [(volume.restrict (Icc (-L) L)).ae_ne p.2.re] with t ht
    have hne : p.2 ≠ (t:ℂ) := by
      intro h
      exact ht (congrArg Complex.re h).symm
    rw [dslope_of_ne _ hne, slope_def_field]
  have hsmooth : ∀ᵐ (t : ℝ) ∂(volume.restrict (Icc (-L) L)),
      ContDiffOn ℂ ∞ (fun p : A × ℂ => dslope (fun z => P (p.1,z)) (t:ℂ) p.2)
        (U ×ˢ scalarTube L (δ/2)) := by
    filter_upwards [ae_restrict_mem measurableSet_Icc] with t ht
    exact parameterized_dslope_smooth hU hδ hP hb ht
  have hbound : ∀ᵐ (t : ℝ) ∂(volume.restrict (Icc (-L) L)), ∀ p ∈ U ×ˢ scalarTube L (δ/2),
      ‖dslope (fun z => P (p.1,z)) (t:ℂ) p.2‖ ≤ 2*B/δ := by
    filter_upwards [ae_restrict_mem measurableSet_Icc] with t ht
    intro p hp
    exact norm_dslope_on_scalarTube hδ hB (hslice p.1 hp.1)
      (fun z hz => hb (p.1,z) ⟨hp.1,hz⟩) ht
      (scalarTube_mono le_rfl (by linarith) hp.2)
  exact (ParameterIntegral.smoothOn_integral_of_dominated
    (hU.prod (isOpen_scalarTube L (δ/2))) hmeas hsmooth hbound (integrable_const (2*B/δ))).neg

omit [FiniteDimensional ℂ A] in
theorem parameterized_hilbertInner_bound
    {P : A × ℂ → ℂ} {U : Set A} {L δ B : ℝ}
    (hU : IsOpen U) (hL : 0 ≤ L) (hδ : 0 < δ) (hB : 0 ≤ B)
    (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ))
    (hb : ∀ p ∈ U ×ˢ scalarTube L δ, ‖P p‖ ≤ B)
    {p : A × ℂ} (hp : p ∈ U ×ˢ scalarTube L δ) :
    ‖-(∫ t in Icc (-L) L, dslope (fun z => P (p.1,z)) (t:ℂ) p.2)‖ ≤ 4*L*B/δ := by
  apply hilbertInner_norm_le hL hδ hB _ (fun z hz => hb (p.1,z) ⟨hp.1,hz⟩) hp.2
  intro z hz
  have hd : DifferentiableAt ℂ P (p.1,z) :=
    (hP.differentiableOn (by simp)).differentiableAt
      ((hU.prod (isOpen_scalarTube L δ)).mem_nhds ⟨hp.1,hz⟩)
  exact (hd.comp z (show DifferentiableAt ℂ (fun y : ℂ => (p.1,y)) z by fun_prop)).differentiableWithinAt

/-- Actual successive offset derivatives while retaining every other parameter. -/
def iteratedScalarPartial : ℕ → (A × ℂ → ℂ) → (A × ℂ → ℂ)
  | 0, P => P
  | n+1, P => scalarPartial (iteratedScalarPartial n P)

theorem iteratedScalarPartial_smooth {P : A × ℂ → ℂ} {U : Set A} {L δ : ℝ}
    (hU : IsOpen U) (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ)) (n : ℕ) :
    ContDiffOn ℂ ∞ (iteratedScalarPartial n P) (U ×ˢ scalarTube L δ) := by
  induction n with
  | zero => exact hP
  | succ n ih => exact scalarPartial_smooth hU ih

theorem iteratedScalarPartial_eq_iteratedDeriv
    {P : A × ℂ → ℂ} {U : Set A} {L δ : ℝ}
    (hU : IsOpen U) (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ)) (n : ℕ)
    {p : A × ℂ} (hp : p ∈ U ×ˢ scalarTube L δ) :
    iteratedScalarPartial n P p = iteratedDeriv n (fun z => P (p.1,z)) p.2 := by
  induction n generalizing p with
  | zero => rfl
  | succ n ih =>
    rw [iteratedScalarPartial,
      scalarPartial_eq_deriv hU (iteratedScalarPartial_smooth hU hP n) hp,
      iteratedDeriv_succ]
    apply Filter.EventuallyEq.deriv_eq
    filter_upwards [(isOpen_scalarTube L δ).mem_nhds hp.2] with z hz
    exact ih ⟨hp.1,hz⟩

end QIRadon.Radon
