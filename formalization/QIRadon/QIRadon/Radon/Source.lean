import QIRadon.Radon.FilteredContinuation
import QIRadon.Radon.ProjectionBounds

/-! The actual local analytic target, finite radial extension, canonical Fourier
projection, and finite complex section are assembled without a profile oracle. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology
namespace QIRadon.Radon
variable {V E : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]
  [NormedAddCommGroup E] [NormedSpace ℂ E]

/-- Complex-valued version of the prescribed real radial extension. -/
def complexRadialExtension (F : E → ℂ) (ι : V →ₗᵢ[ℝ] E) (w : ℝ → ℝ)
    (R₀ R : ℝ) (x : V) : ℂ :=
  F (ι x) * (radialCutoff w R₀ R ‖x‖:ℂ)

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
theorem complexRadialExtension_eq_zero (F : E → ℂ) (ι : V →ₗᵢ[ℝ] E) (w : ℝ → ℝ)
    {R₀ R : ℝ} {x : V} (hx : R < ‖x‖) : complexRadialExtension F ι w R₀ R x=0 := by
  simp [complexRadialExtension, radialCutoff_eq_zero hx]

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
theorem complexRadialExtension_contDiff {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E)
    {w : ℝ → ℝ} {R₀ R σ : ℝ} (hR₀ : 0 < R₀) (hσ : 0 < σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube ι R σ))
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hnormalized : (∫ a in Icc R₀ R, w a)=1) :
    ContDiff ℝ ∞ (complexRadialExtension F ι w R₀ R) := by
  have hcut : ContDiff ℝ ∞ (fun x : V => (radialCutoff w R₀ R ‖x‖:ℂ)) :=
    Complex.ofRealCLM.contDiff.comp (radialCutoff_norm_contDiff hR₀ hi hsupp hw hnormalized)
  apply contDiff_iff_contDiffAt.mpr
  intro x
  by_cases hx : ι x ∈ realBallTube ι R σ
  · have hreal : ContDiffAt ℝ ∞ F (ι x) := (hF (ι x) hx).restrictScalars.contDiffAt
    exact (hreal.comp x ι.contDiff.contDiffAt).mul hcut.contDiffAt
  · have hout : R < ‖x‖ := by
      by_contra hh
      have hxR : x ∈ closedBall (0:V) R := mem_closedBall_zero_iff.mpr (le_of_not_gt hh)
      apply hx
      exact mem_iUnion.mpr ⟨x, mem_iUnion.mpr ⟨hxR, mem_ball_self hσ⟩⟩
    apply (contDiffAt_const : ContDiffAt ℝ ∞ (fun _ : V => (0:ℂ)) x).congr_of_eventuallyEq
    filter_upwards [(isOpen_lt continuous_const continuous_norm).mem_nhds hout] with y hy
    exact complexRadialExtension_eq_zero F ι w hy

omit [MeasurableSpace V] [BorelSpace V] in
theorem complexRadialExtension_hasCompactSupport (F : E → ℂ) (ι : V →ₗᵢ[ℝ] E)
    (w : ℝ → ℝ) (R₀ R : ℝ) : HasCompactSupport (complexRadialExtension F ι w R₀ R) := by
  apply HasCompactSupport.of_support_subset_isCompact (isCompact_closedBall (0:V) R)
  intro x hx
  by_contra hn
  have hnorm : R < ‖x‖ := by simpa only [mem_closedBall_zero_iff, not_le] using hn
  exact hx (complexRadialExtension_eq_zero F ι w hnorm)

/-- The canonical extension is an actual Schwartz map from local target
analyticity and the smooth finite radius mixture. -/
def complexRadialExtensionSchwartz {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E)
    {w : ℝ → ℝ} {R₀ R σ : ℝ} (hR₀ : 0 < R₀) (hσ : 0 < σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube ι R σ))
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hnormalized : (∫ a in Icc R₀ R, w a)=1) : 𝓢(V, ℂ) :=
  (complexRadialExtension_hasCompactSupport F ι w R₀ R).toSchwartzMap
    (complexRadialExtension_contDiff ι hR₀ hσ hF hi hsupp hw hnormalized)

omit [MeasurableSpace V] [BorelSpace V] in
@[simp] theorem complexRadialExtensionSchwartz_apply {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E)
    {w : ℝ → ℝ} {R₀ R σ : ℝ} (hR₀ : 0 < R₀) (hσ : 0 < σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube ι R σ))
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hnormalized : (∫ a in Icc R₀ R, w a)=1) (x : V) :
    complexRadialExtensionSchwartz ι hR₀ hσ hF hi hsupp hw hnormalized x =
      complexRadialExtension F ι w R₀ R x := rfl

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
theorem complexRadialExtension_norm_le {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E)
    {w : ℝ → ℝ} {R₀ R σ B : ℝ} (hσ : 0 < σ) (hB : 0 ≤ B)
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B)
    (hi : IntegrableOn w (Icc R₀ R)) (hw : ∀ a ∈ Icc R₀ R, 0 ≤ w a)
    (hnormalized : (∫ a in Icc R₀ R, w a)=1) (x : V) :
    ‖complexRadialExtension F ι w R₀ R x‖ ≤ B := by
  by_cases hx : ‖x‖ ≤ R
  · have hxU : ι x ∈ realBallTube ι R σ :=
      mem_iUnion.mpr ⟨x, mem_iUnion.mpr ⟨mem_closedBall_zero_iff.mpr hx, mem_ball_self hσ⟩⟩
    rw [complexRadialExtension, norm_mul, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (radialCutoff_nonneg hw)]
    exact (mul_le_mul_of_nonneg_left (radialCutoff_le_one hi hw hnormalized) (norm_nonneg _)).trans
      (by simpa using hFBound _ hxU)
  · rw [complexRadialExtension_eq_zero F ι w (lt_of_not_ge hx), norm_zero]
    exact hB

/-- A supported bounded target has the explicit volume-times-magnitude L1 bound. -/
theorem integral_norm_le_ball_volume {f : V → ℂ} {R B : ℝ}
    (hbound : ∀ x, ‖f x‖ ≤ B) (hsupp : ∀ x, R < ‖x‖ → f x=0) :
    (∫ x : V, ‖f x‖) ≤ B * volume.real (closedBall (0:V) R) := by
  have heq : (fun x : V => ‖f x‖) = (closedBall (0:V) R).indicator (fun x => ‖f x‖) := by
    funext x
    by_cases hx : x ∈ closedBall (0:V) R
    · simp [hx]
    · have hnorm : R < ‖x‖ := by simpa only [mem_closedBall_zero_iff, not_le] using hx
      simp [hx, hsupp x hnorm]
  rw [heq, integral_indicator measurableSet_closedBall]
  let : IsFiniteMeasure (volume.restrict (closedBall (0:V) R)) :=
    ⟨by rw [Measure.restrict_apply_univ]; exact (isCompact_closedBall (0:V) R).measure_lt_top⟩
  have h := norm_integral_le_of_norm_le_const (μ := volume.restrict (closedBall (0:V) R))
    (Filter.Eventually.of_forall (fun x : V => show ‖‖f x‖‖ ≤ B by simpa using hbound x))
  rw [Real.norm_eq_abs, abs_of_nonneg (integral_nonneg (fun x => norm_nonneg (f x)))] at h
  simpa only [Measure.real, Measure.restrict_apply_univ] using h

/-- The actual scalar canonical projection gets its analytic continuation,
pointwise real identification, and L1 mass bound from the local target data. -/
theorem canonicalProjection_certificate {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E)
    (θ : sphere (0:V) 1) {w : ℝ → ℝ} {L δ Q R₀ R σ B : ℝ}
    (hR₀ : 0 < R₀) (hσ : 0 < σ) (hB : 0 ≤ B) (hL : 0 ≤ L) (hδ : 0 < δ)
    (hgap : L+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube ι R σ))
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a)
    (hnormalized : (∫ a in Icc R₀ R, w a)=1) :
    let f := complexRadialExtensionSchwartz ι hR₀ hσ hF hi hsupp hw hnormalized
    let p := projectionSchwartz f θ
    let P := complexCrossSection F ι θ w R₀ R
    AnalyticOnNhd ℂ P (scalarTube L δ) ∧
      (∀ z ∈ scalarTube L δ, ‖P z‖ ≤
        volume.real (closedBall (0:Hyperplane θ) 1) * Real.sqrt (R^2+Q^2)^(Module.finrank ℝ V-1)*B) ∧
      (∀ s ∈ Icc (-L) L, p s=P (s:ℂ)) ∧
      (∫ s : ℝ, ‖p s‖) ≤ B*volume.real (closedBall (0:V) R) := by
  dsimp only
  have hbd : ∀ x, ‖complexRadialExtensionSchwartz ι hR₀ hσ hF hi hsupp hw hnormalized x‖ ≤ B :=
    complexRadialExtension_norm_le ι hσ hB hFBound hi.integrableOn hwpos hnormalized
  have hsup : ∀ x, R < ‖x‖ → complexRadialExtensionSchwartz ι hR₀ hσ hF hi hsupp hw hnormalized x=0 :=
    fun x hx => complexRadialExtension_eq_zero F ι w hx
  refine ⟨complexCrossSection_analytic ι θ hw.continuous hi.integrableOn hB hL hδ hgap hQR
    hmargin hF.differentiableOn hFBound, ?_, ?_, ?_⟩
  · intro z hz
    exact complexCrossSection_norm_le ι θ hw.continuous hi.integrableOn hwpos hnormalized hB hL hδ hgap hQR
      hmargin hF.differentiableOn hFBound hz
  · intro s hs
    rw [projectionSchwartz_eq_radon_supported _ hbd hsup θ]
    exact (complexCrossSection_eq_radon_real ι θ hw.continuous hi.integrableOn hB hL hδ hgap hQR
      hmargin hF.differentiableOn hFBound (abs_le.mpr hs)).symm
  · exact (projectionSchwartz_integral_norm_le _ hbd hsup θ).trans
      (integral_norm_le_ball_volume hbd hsup)

end QIRadon.Radon
