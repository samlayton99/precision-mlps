import QIRadon.Radon.Filter

/-! Real projection bounds and pointwise identification for finite supported
targets, connecting analytic finite sections to the canonical Fourier slices. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap Topology
namespace QIRadon.Radon
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]

/-- A continuous target supported in a finite ball has continuous actual Radon
projections. Domination uses one fixed transverse ball for every offset. -/
theorem hyperplaneRadon_continuous_supported {f : V → ℂ} (hf : Continuous f)
    {R B : ℝ} (hbound : ∀ x, ‖f x‖ ≤ B)
    (hsupp : ∀ x, R < ‖x‖ → f x=0) (θ : sphere (0:V) 1) :
    Continuous (hyperplaneRadon f θ) := by
  let majorant : Hyperplane θ → ℝ := (closedBall 0 R).indicator (fun _ => B)
  have hm : Integrable majorant := by
    have hi : IntegrableOn (fun _ : Hyperplane θ => B) (closedBall 0 R) :=
      continuousOn_const.integrableOn_compact (isCompact_closedBall (0 : Hyperplane θ) R)
    exact (integrable_indicator_iff measurableSet_closedBall).mpr hi
  unfold hyperplaneRadon
  apply continuous_of_dominated (bound := majorant)
  · intro s
    exact (hf.comp (by fun_prop)).aestronglyMeasurable
  · intro s
    filter_upwards with y
    by_cases hy : y ∈ closedBall (0 : Hyperplane θ) R
    · simpa only [majorant, indicator_of_mem hy] using hbound (s • θ.val+y.val)
    · have hRy : R < ‖y‖ := by simpa only [mem_closedBall_zero_iff, not_le] using hy
      rw [hsupp _ (lt_of_lt_of_le hRy (norm_hyperplane_le_point θ s y))]
      simp [majorant, hy]
  · exact hm
  · filter_upwards with y
    exact hf.comp (by fun_prop)

/-- A projection's L1 norm is no larger than the target's L1 norm, derived using
the actual volume-preserving coordinate frame and Fubini. -/
theorem hyperplaneRadon_integral_norm_le {f : V → ℂ} (hf : Integrable f)
    (θ : sphere (0:V) 1) :
    (∫ s : ℝ, ‖hyperplaneRadon f θ s‖) ≤ ∫ x : V, ‖f x‖ := by
  have hc : Integrable (fun z : ℝ × Hyperplane θ => f (frame θ z)) (volume.prod volume) :=
    ((frame_measurePreserving θ).integrable_comp_emb (frame_measurableEmbedding θ)).mpr hf
  have hbound : ∀ s : ℝ, ‖hyperplaneRadon f θ s‖ ≤
      ∫ y : Hyperplane θ, ‖f (frame θ (s,y))‖ := by
    intro s
    simpa only [hyperplaneRadon, frame_apply] using
      norm_integral_le_integral_norm (fun y : Hyperplane θ => f (s • θ.val+y.val))
  calc
    _ ≤ ∫ s : ℝ, ∫ y : Hyperplane θ, ‖f (frame θ (s,y))‖ :=
      integral_mono (hyperplaneRadon_integrable hf θ).norm hc.norm.integral_prod_left hbound
    _ = ∫ z : ℝ × Hyperplane θ, ‖f (frame θ z)‖ ∂(volume.prod volume) :=
      (integral_prod _ hc.norm).symm
    _ = ∫ x : V, ‖f x‖ :=
      (frame_measurePreserving θ).integral_comp (frame_measurableEmbedding θ) (fun x => ‖f x‖)

/-- The Fourier-slice Schwartz profile agrees pointwise with the actual finite
projection for a supported target, including its outer real tail. -/
theorem projectionSchwartz_eq_radon_supported (f : 𝓢(V, ℂ))
    {R B : ℝ} (hbound : ∀ x, ‖f x‖ ≤ B)
    (hsupp : ∀ x, R < ‖x‖ → f x=0) (θ : sphere (0:V) 1) :
    (projectionSchwartz f θ : ℝ → ℂ) = hyperplaneRadon f θ := by
  funext s
  exact projectionSchwartz_eq_radon f θ
    ((hyperplaneRadon_continuous_supported f.continuous hbound hsupp θ).continuousAt)

/-- The canonical inverse slice therefore inherits the usable real L1 bound. -/
theorem projectionSchwartz_integral_norm_le (f : 𝓢(V, ℂ))
    {R B : ℝ} (hbound : ∀ x, ‖f x‖ ≤ B)
    (hsupp : ∀ x, R < ‖x‖ → f x=0) (θ : sphere (0:V) 1) :
    (∫ s : ℝ, ‖projectionSchwartz f θ s‖) ≤ ∫ x : V, ‖f x‖ := by
  have heq := projectionSchwartz_eq_radon_supported f hbound hsupp θ
  simp_rw [show ∀ s, projectionSchwartz f θ s=hyperplaneRadon f θ s from congrFun heq]
  exact hyperplaneRadon_integral_norm_le f.integrable θ

end QIRadon.Radon
