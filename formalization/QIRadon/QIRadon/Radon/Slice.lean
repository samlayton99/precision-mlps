import Mathlib

/-! The Fourier-slice identity from actual hyperplane integration and an explicitly
constructed orthogonal coordinate frame. -/

noncomputable section
open MeasureTheory Set Metric
open scoped RealInnerProductSpace FourierTransform

namespace QIRadon.Radon

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]

/-- Actual orthogonal hyperplane to the selected unit direction. -/
abbrev Hyperplane (θ : sphere (0 : V) 1) := (ℝ ∙ θ.val)ᗮ

/-- Explicit orthogonal splitting into the scalar direction and its hyperplane. -/
def frameLp (θ : sphere (0 : V) 1) : WithLp 2 (ℝ × Hyperplane θ) ≃ₗᵢ[ℝ] V :=
  ((LinearIsometryEquiv.toSpanUnitSingleton θ.val
    (mem_sphere_zero_iff_norm.mp θ.property)).withLpProdCongr 2
      (LinearIsometryEquiv.refl ℝ (Hyperplane θ))).trans
        (Submodule.orthogonalDecomposition (ℝ ∙ θ.val)).symm

/-- The same frame on ordinary product coordinates, whose measure is product volume. -/
def frame (θ : sphere (0 : V) 1) (z : ℝ × Hyperplane θ) : V :=
  frameLp θ (WithLp.toLp 2 z)

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
@[simp] theorem frame_apply (θ : sphere (0 : V) 1) (z : ℝ × Hyperplane θ) :
    frame θ z = z.1 • θ.val + z.2.val := by
  simp [frame, frameLp]

/-- This specific coordinate change preserves actual Euclidean volume. -/
theorem frame_measurePreserving (θ : sphere (0 : V) 1) :
    MeasurePreserving (frame θ) (volume.prod volume) volume := by
  exact (frameLp θ).measurePreserving.comp
    (WithLp.volume_preserving_toLp ℝ (Hyperplane θ))

/-- The frame is a measurable embedding, so integration and integrability can
be transferred without assuming a change-of-variables oracle. -/
theorem frame_measurableEmbedding (θ : sphere (0 : V) 1) :
    MeasurableEmbedding (frame θ) :=
  (frameLp θ).toHomeomorph.measurableEmbedding.comp
    (MeasurableEquiv.toLp 2 (ℝ × Hyperplane θ)).measurableEmbedding

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
@[simp] theorem inner_frame_direction (θ : sphere (0 : V) 1)
    (z : ℝ × Hyperplane θ) : inner ℝ (frame θ z) θ.val = z.1 := by
  rw [frame_apply, inner_add_left, real_inner_smul_left,
    real_inner_self_eq_norm_sq,
    (Submodule.mem_orthogonal_singleton_iff_inner_left.mp z.2.property)]
  have hθ : ‖θ.val‖ = 1 := mem_sphere_zero_iff_norm.mp θ.property
  simp [hθ]

/-- The hyperplane Radon projection as an actual integral, not a Fourier definition. -/
def hyperplaneRadon (f : V → ℂ) (θ : sphere (0 : V) 1) (s : ℝ) : ℂ :=
  ∫ y : Hyperplane θ, f (s • θ.val + y.val)

/-- Integrability of hyperplane projections follows from the actual coordinate
frame and Fubini's theorem. -/
theorem hyperplaneRadon_integrable {f : V → ℂ} (hf : Integrable f)
    (θ : sphere (0 : V) 1) : Integrable (hyperplaneRadon f θ) := by
  have hc : Integrable (fun z : ℝ × Hyperplane θ => f (frame θ z))
      (volume.prod volume) :=
    ((frame_measurePreserving θ).integrable_comp_emb (frame_measurableEmbedding θ)).mpr hf
  unfold hyperplaneRadon
  simpa only [frame_apply] using hc.integral_prod_left

/-- The Fourier transform of the actual hyperplane Radon projection is the
actual multidimensional Fourier transform restricted to the direction. -/
theorem fourier_slice {f : V → ℂ} (hf : Integrable f)
    (θ : sphere (0 : V) 1) (κ : ℝ) :
    𝓕 (hyperplaneRadon f θ) κ = 𝓕 f (κ • θ.val) := by
  have hc : Integrable (fun z : ℝ × Hyperplane θ => f (frame θ z))
      (volume.prod volume) :=
    ((frame_measurePreserving θ).integrable_comp_emb (frame_measurableEmbedding θ)).mpr hf
  have hweighted : Integrable (fun z : ℝ × Hyperplane θ =>
      Real.fourierChar (-(z.1 * κ)) • f (frame θ z)) (volume.prod volume) := by
    apply hc.norm.mono'
    · exact (Real.continuous_fourierChar.comp
        (continuous_fst.mul continuous_const).neg).aestronglyMeasurable.smul
          hc.aestronglyMeasurable
    · filter_upwards with z
      simp only [Circle.norm_smul, le_refl]
  calc
    𝓕 (hyperplaneRadon f θ) κ =
        ∫ s : ℝ, Real.fourierChar (-(s * κ)) •
          ∫ y : Hyperplane θ, f (frame θ (s, y)) := by
      simp only [Real.fourier_real_eq, hyperplaneRadon, frame_apply]
    _ = ∫ s : ℝ, ∫ y : Hyperplane θ,
        Real.fourierChar (-(s * κ)) • f (frame θ (s, y)) := by
      simp only [Circle.smul_def, integral_smul]
    _ = ∫ z : ℝ × Hyperplane θ,
        Real.fourierChar (-(z.1 * κ)) • f (frame θ z) :=
      (integral_prod _ hweighted).symm
    _ = ∫ z : ℝ × Hyperplane θ,
        Real.fourierChar (-inner ℝ (frame θ z) (κ • θ.val)) • f (frame θ z) := by
      apply integral_congr_ae
      filter_upwards with z
      simp only [real_inner_smul_right, inner_frame_direction, mul_comm]
    _ = 𝓕 f (κ • θ.val) := by
      rw [Real.fourier_eq]
      exact (frame_measurePreserving θ).integral_comp (frame_measurableEmbedding θ)
        (fun x : V => Real.fourierChar (-inner ℝ x (κ • θ.val)) • f x)

end QIRadon.Radon
