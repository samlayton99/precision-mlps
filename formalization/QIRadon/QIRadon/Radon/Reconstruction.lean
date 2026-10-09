import QIRadon.Radon.Polar
import QIRadon.Radon.Slice
import QIRadon.Radon.Boundary

/-!
# Canonical normalized ridge reconstruction

Mathlib uses frequency in cycles: its Fourier character is exp(2π i t).
Consequently the normalized full-frequency profile has coefficient |S|/2,
not |S|/(2(2π)^d). Changing variable ω=2πξ gives the paper convention.
-/

noncomputable section
open MeasureTheory Set Metric
open scoped RealInnerProductSpace FourierTransform SchwartzMap ContDiff

namespace QIRadon.Radon

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V]

/-- Raw surface measure inherited from actual Euclidean volume. -/
def surfaceMeasure (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] :
    Measure (sphere (0 : V) 1) := (volume : Measure V).toSphere

/-- Actual total surface area, retaining the dimension normalization. -/
def surfaceArea (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] : ℝ :=
  (surfaceMeasure V).real Set.univ

/-- Normalized directional probability measure. -/
def normalizedSphere (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] :
    Measure (sphere (0 : V) 1) :=
  ((surfaceMeasure V) Set.univ)⁻¹ • surfaceMeasure V

theorem surface_mass_ne_zero : surfaceMeasure V Set.univ ≠ 0 := by
  have hm : (volume : Measure V).toSphere ≠ 0 := Measure.toSphere_ne_zero volume
  exact Measure.measure_univ_ne_zero.mpr hm

omit [Nontrivial V] in
theorem surface_mass_ne_top : surfaceMeasure V Set.univ ≠ ⊤ := by
  exact measure_ne_top (volume : Measure V).toSphere Set.univ

theorem surfaceArea_pos : 0 < surfaceArea V := by
  exact ENNReal.toReal_pos surface_mass_ne_zero surface_mass_ne_top

instance normalizedSphere_probability : IsProbabilityMeasure (normalizedSphere V) where
  measure_univ := by
    simp only [normalizedSphere, Measure.smul_apply, smul_eq_mul]
    exact ENNReal.inv_mul_cancel surface_mass_ne_zero surface_mass_ne_top

/-- The actual area factor cancels normalization of sphere measure exactly. -/
theorem integral_surfaceArea_mul_normalized (H : sphere (0 : V) 1 → ℝ) :
    (∫ θ, surfaceArea V * H θ ∂normalizedSphere V) = ∫ θ, H θ ∂surfaceMeasure V := by
  rw [normalizedSphere, integral_smul_measure, integral_const_mul]
  simp only [surfaceArea, measureReal_def, ENNReal.toReal_inv, smul_eq_mul]
  have hs : ((surfaceMeasure V) Set.univ).toReal ≠ 0 :=
    ne_of_gt (surfaceArea_pos (V := V))
  rw [← mul_assoc, inv_mul_cancel₀ hs, one_mul]

omit [Nontrivial V] in
/-- Positive-frequency profile integrability follows from the actual polar
change of coordinates, rather than being assumed of the ridge representation. -/
theorem integrable_positive_ridge {f : V → ℂ} (hF : Integrable (𝓕 f)) (x : V) :
    Integrable (fun θ : sphere (0 : V) 1 =>
      QIRadon.positiveRidgeProfile f θ (inner ℝ θ.val x)) (surfaceMeasure V) := by
  have hi : Integrable (fun ξ : V => Real.fourierChar (inner ℝ ξ x) • (𝓕 f) ξ) := by
    simpa only [inner_neg_right, neg_neg] using
      (Real.fourierIntegral_convergent_iff (-x)).mpr hF
  have hp := QIRadon.integrable_polar_product hi
  simpa only [QIRadon.positiveRidgeProfile, real_inner_smul_left, surfaceMeasure] using
    hp.integral_prod_left

/-- Canonical real profiles for a real target. The full signed-frequency
interpretation will be proved below; no target fit is used. -/
def realRidgeProfile (f : V → ℝ) (θ : sphere (0 : V) 1) (s : ℝ) : ℝ :=
  surfaceArea V * (QIRadon.positiveRidgeProfile (fun x => (f x : ℂ)) θ s).re

/-- Actual normalized reconstruction of a real target from its Fourier-defined
profiles. Neither a ridge identity nor an angular approximation is a hypothesis. -/
theorem normalized_real_ridge_reconstruction {f : V → ℝ}
    (hf : Integrable (fun x => (f x : ℂ)))
    (hF : Integrable (𝓕 (fun x => (f x : ℂ))))
    {x : V} (hx : ContinuousAt f x) :
    f x = ∫ θ : sphere (0 : V) 1,
      realRidgeProfile f θ (inner ℝ θ.val x) ∂normalizedSphere V := by
  have hrec := QIRadon.positive_ridge_reconstruction hf hF
    (Complex.continuous_ofReal.continuousAt.comp hx)
  have hint := integrable_positive_ridge hF x
  have hre := congrArg Complex.re hrec
  simp only [Complex.ofReal_re] at hre
  rw [show (fun θ : sphere (0 : V) 1 => realRidgeProfile f θ (inner ℝ θ.val x)) =
    (fun θ => surfaceArea V *
      (QIRadon.positiveRidgeProfile (fun y => (f y : ℂ)) θ (inner ℝ θ.val x)).re) by rfl,
    integral_surfaceArea_mul_normalized]
  exact hre.trans (Complex.reCLM.integral_comp_comm hint).symm

/-- Antipodal map on the actual unit sphere. -/
def antipode (θ : sphere (0 : V) 1) : sphere (0 : V) 1 :=
  ⟨-θ.val, by simpa only [mem_sphere_zero_iff_norm, norm_neg] using θ.property⟩

omit [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V] in
@[simp] theorem antipode_val (θ : sphere (0 : V) 1) : (antipode θ).val = -θ.val := rfl

omit [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V] in
@[simp] theorem antipode_antipode (θ : sphere (0 : V) 1) : antipode (antipode θ) = θ := by
  apply Subtype.ext
  simp

private theorem conj_fourierChar (t : ℝ) :
    starRingEnd ℂ (Real.fourierChar t : ℂ) = (Real.fourierChar (-t) : ℂ) := by
  rw [AddChar.map_neg_eq_inv, Circle.coe_inv_eq_conj]

omit [Nontrivial V] in
/-- Conjugate symmetry of the actual Fourier transform of a real function. -/
theorem fourier_real_neg (f : V → ℝ) (ξ : V) :
    𝓕 (fun x => (f x : ℂ)) (-ξ) = starRingEnd ℂ (𝓕 (fun x => (f x : ℂ)) ξ) := by
  simp only [Real.fourier_eq, inner_neg_right, neg_neg]
  rw [← integral_conj]
  apply integral_congr_ae
  filter_upwards with x
  simp only [Circle.smul_def, smul_eq_mul, map_mul, conj_fourierChar,
    Complex.conj_ofReal, neg_neg]

omit [Nontrivial V] in
/-- Opposite positive-frequency profiles are complex conjugates for real targets. -/
theorem positive_ridge_antipodal (f : V → ℝ) (θ : sphere (0 : V) 1) (s : ℝ) :
    QIRadon.positiveRidgeProfile (fun x => (f x : ℂ)) (antipode θ) (-s) =
      starRingEnd ℂ (QIRadon.positiveRidgeProfile (fun x => (f x : ℂ)) θ s) := by
  unfold QIRadon.positiveRidgeProfile
  rw [← integral_conj]
  apply integral_congr_ae
  filter_upwards with r
  simp only [antipode_val, smul_neg, fourier_real_neg, Circle.smul_def,
    smul_eq_mul, map_mul, conj_fourierChar, mul_neg]

omit [Nontrivial V] in
/-- The real canonical profile is exactly antipodally paired. -/
theorem real_ridge_antipodal (f : V → ℝ) (θ : sphere (0 : V) 1) (s : ℝ) :
    realRidgeProfile f (antipode θ) (-s) = realRidgeProfile f θ s := by
  unfold realRidgeProfile
  rw [positive_ridge_antipodal]
  rfl

omit [Nontrivial V] in
/-- The real profile equals the normalized average of the two actual signed
frequency rays. This establishes its canonical full-frequency interpretation. -/
theorem real_ridge_eq_paired_rays (f : V → ℝ) (θ : sphere (0 : V) 1) (s : ℝ) :
    (realRidgeProfile f θ s : ℂ) =
      (surfaceArea V / 2 : ℂ) *
        (QIRadon.positiveRidgeProfile (fun x => (f x : ℂ)) θ s +
         QIRadon.positiveRidgeProfile (fun x => (f x : ℂ)) (antipode θ) (-s)) := by
  rw [positive_ridge_antipodal]
  unfold realRidgeProfile
  apply Complex.ext
  · simp
    ring
  · simp

omit [Nontrivial V] in
/-- Expose the radial Jacobian as the ordinary power r^(d-1) in a real integral. -/
theorem positive_ridge_eq_positive_integral (f : V → ℂ)
    (θ : sphere (0 : V) 1) (s : ℝ) :
    QIRadon.positiveRidgeProfile f θ s =
      ∫ r in Set.Ioi (0 : ℝ), r^(Module.finrank ℝ V - 1) •
        (Real.fourierChar (r*s) • 𝓕 f (r • θ.val)) := by
  unfold QIRadon.positiveRidgeProfile
  simp only [Measure.volumeIoiPow, ENNReal.ofReal]
  rw [integral_withDensity_eq_integral_smul]
  · rw [integral_subtype_comap measurableSet_Ioi
      (fun r : ℝ => Real.toNNReal (r^(Module.finrank ℝ V - 1)) •
        (Real.fourierChar (r*s) • 𝓕 f (r • θ.val)))]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro r hr
    dsimp only
    rw [NNReal.smul_def, Real.coe_toNNReal _ (pow_nonneg (show 0 ≤ r from le_of_lt hr) _)]
  · exact (measurable_subtype_coe.pow_const _).real_toNNReal

/-- The full signed-frequency Radon integrand in the cycles-frequency convention. -/
def fullFrequencyIntegrand (f : V → ℂ) (θ : sphere (0 : V) 1)
    (s r : ℝ) : ℂ :=
  |r|^(Module.finrank ℝ V - 1) • (Real.fourierChar (r*s) • 𝓕 f (r • θ.val))

omit [Nontrivial V] in
/-- Splitting the actual full signed-frequency integral into its two rays gives
exactly the antipodal profile pair. -/
theorem full_frequency_eq_paired_rays (f : V → ℂ) (θ : sphere (0 : V) 1) (s : ℝ)
    (hray : Integrable (fullFrequencyIntegrand f θ s)) :
    (∫ r : ℝ, fullFrequencyIntegrand f θ s r) =
      QIRadon.positiveRidgeProfile f θ s +
      QIRadon.positiveRidgeProfile f (antipode θ) (-s) := by
  have hsplit := integral_add_compl (s := Set.Ioi (0 : ℝ)) measurableSet_Ioi hray
  rw [Set.compl_Ioi] at hsplit
  rw [← hsplit, positive_ridge_eq_positive_integral,
    positive_ridge_eq_positive_integral]
  congr 1
  · apply setIntegral_congr_fun measurableSet_Ioi
    intro r hr
    simp only [fullFrequencyIntegrand, abs_of_pos (show 0 < r from hr)]
  · have hneg := integral_comp_neg_Ioi (0 : ℝ) (fullFrequencyIntegrand f θ s)
    simp only [neg_zero] at hneg
    rw [← hneg]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro r hr
    simp only [fullFrequencyIntegrand, abs_neg, abs_of_pos (show 0 < r from hr), antipode_val,
      neg_smul, smul_neg, mul_neg, neg_mul]

omit [Nontrivial V] in
/-- The canonical normalized full-frequency Radon profile, including its exact
surface-area factor. The integrability premise will be derived for Schwartz targets. -/
theorem real_ridge_eq_full_frequency (f : V → ℝ) (θ : sphere (0 : V) 1) (s : ℝ)
    (hray : Integrable (fullFrequencyIntegrand (fun x => (f x : ℂ)) θ s)) :
    (realRidgeProfile f θ s : ℂ) =
      (surfaceArea V / 2 : ℂ) *
        ∫ r : ℝ, |r|^(Module.finrank ℝ V - 1) •
          (Real.fourierChar (r*s) • 𝓕 (fun x => (f x : ℂ)) (r • θ.val)) := by
  rw [real_ridge_eq_paired_rays, ← full_frequency_eq_paired_rays _ θ s hray]
  rfl

/-- Restriction of a Schwartz function to a unit radial line, constructed via
its actual linear embedding. -/
def restrictDirection (F : 𝓢(V, ℂ)) (θ : sphere (0 : V) 1) : 𝓢(ℝ, ℂ) := by
  let l : ℝ →L[ℝ] V := (ContinuousLinearMap.id ℝ ℝ).smulRight θ.val
  have hθ : ‖θ.val‖ = 1 := mem_sphere_zero_iff_norm.mp θ.property
  exact SchwartzMap.compCLM ℂ l.hasTemperateGrowth ⟨1, 1, by
    intro r
    simp only [l, ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.id_apply,
      norm_smul, hθ, mul_one, one_mul, pow_one]
    linarith⟩ F

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V] in
@[simp] theorem restrictDirection_apply (F : 𝓢(V, ℂ)) (θ : sphere (0 : V) 1) (r : ℝ) :
    restrictDirection F θ r = F (r • θ.val) := rfl

omit [Nontrivial V] in
/-- Every full frequency ray is integrable for a Schwartz target. Thus the
full-frequency formula has no exceptional-direction assumption in this class. -/
theorem integrable_fullFrequency_schwartz (f : 𝓢(V, ℂ))
    (θ : sphere (0 : V) 1) (s : ℝ) :
    Integrable (fullFrequencyIntegrand f θ s) := by
  have hint := (restrictDirection (𝓕 f) θ).integrable_pow_mul volume
    (Module.finrank ℝ V - 1)
  apply hint.mono'
  · unfold fullFrequencyIntegrand
    have hFourier : Continuous (fun r : ℝ => 𝓕 (f : V → ℂ) (r • θ.val)) := by
      rw [← SchwartzMap.fourier_coe]
      exact (𝓕 f).continuous.comp (continuous_id.smul continuous_const)
    exact ((continuous_abs.pow _).smul
      ((Real.continuous_fourierChar.comp (continuous_id.mul continuous_const)).smul
        hFourier)).aestronglyMeasurable
  · filter_upwards with r
    simp only [fullFrequencyIntegrand, norm_smul, Circle.norm_smul,
      Real.norm_eq_abs, abs_pow, abs_abs, restrictDirection_apply, SchwartzMap.fourier_coe, le_refl]

/-- A real Schwartz function explicitly embedded into complex Schwartz space. -/
def complexifySchwartz (f : 𝓢(V, ℝ)) : 𝓢(V, ℂ) :=
  SchwartzMap.postcompCLM (𝕜 := ℝ) Complex.ofRealCLM f

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V] in
@[simp] theorem complexifySchwartz_apply (f : 𝓢(V, ℝ)) (x : V) :
    complexifySchwartz f x = (f x : ℂ) := rfl

omit [Nontrivial V] in
/-- Full signed-frequency formula for every scalar argument and every direction
of a real Schwartz target, with all ray integrability proved. -/
theorem real_ridge_full_frequency_schwartz (f : 𝓢(V, ℝ))
    (θ : sphere (0 : V) 1) (s : ℝ) :
    (realRidgeProfile f θ s : ℂ) =
      (surfaceArea V / 2 : ℂ) *
        ∫ r : ℝ, |r|^(Module.finrank ℝ V - 1) •
          (Real.fourierChar (r*s) • 𝓕 (fun x => (f x : ℂ)) (r • θ.val)) := by
  exact real_ridge_eq_full_frequency f θ s
    (integrable_fullFrequency_schwartz (complexifySchwartz f) θ s)

/-- Exact normalized canonical reconstruction for every real Schwartz target. -/
theorem normalized_real_schwartz_reconstruction (f : 𝓢(V, ℝ)) (x : V) :
    f x = ∫ θ : sphere (0 : V) 1,
      realRidgeProfile f θ (inner ℝ θ.val x) ∂normalizedSphere V := by
  apply normalized_real_ridge_reconstruction (complexifySchwartz f).integrable
  · change Integrable (𝓕 ((complexifySchwartz f) : V → ℂ))
    rw [← SchwartzMap.fourier_coe]
    exact (𝓕 (complexifySchwartz f)).integrable
  · exact f.continuous.continuousAt

/-- Explicit surface-area normalization in every finite positive dimension. -/
theorem surfaceArea_gamma :
    surfaceArea V =
      2 * Real.sqrt Real.pi ^ Module.finrank ℝ V /
        Real.Gamma ((Module.finrank ℝ V : ℝ) / 2) := by
  have hd : (0 : ℝ) < Module.finrank ℝ V := by exact_mod_cast Module.finrank_pos
  have hhalf : 0 < (Module.finrank ℝ V : ℝ) / 2 := by positivity
  have hg := Real.Gamma_pos_of_pos (show 0 < (Module.finrank ℝ V : ℝ)/2+1 by positivity)
  have hv : 0 ≤ Real.sqrt Real.pi ^ Module.finrank ℝ V /
      Real.Gamma ((Module.finrank ℝ V : ℝ)/2+1) := by positivity
  unfold surfaceArea surfaceMeasure
  rw [Measure.toSphere_real_apply_univ]
  simp only [measureReal_def, InnerProductSpace.volume_ball,
    ENNReal.ofReal_one, one_pow, one_mul, ENNReal.toReal_ofReal hv]
  rw [Real.Gamma_add_one hhalf.ne']
  field_simp

/-- The profile written as the full canonical filter of the actual hyperplane
Radon projection. In cycles-frequency units the filter is |ξ|^(d-1). -/
def filteredRadonProfile (f : V → ℂ) (θ : sphere (0 : V) 1) (s : ℝ) : ℂ :=
  (surfaceArea V / 2 : ℂ) * ∫ r : ℝ,
    |r|^(Module.finrank ℝ V - 1) •
      (Real.fourierChar (r*s) • 𝓕 (hyperplaneRadon f θ) r)

omit [Nontrivial V] in
/-- The Fourier-defined real profile is the full Radon-filtered projection,
proved via the actual hyperplane Fourier-slice theorem. -/
theorem real_ridge_eq_filtered_radon (f : 𝓢(V, ℝ))
    (θ : sphere (0 : V) 1) (s : ℝ) :
    (realRidgeProfile f θ s : ℂ) =
      filteredRadonProfile (fun x => (f x : ℂ)) θ s := by
  rw [real_ridge_full_frequency_schwartz]
  unfold filteredRadonProfile
  congr 1
  apply integral_congr_ae
  filter_upwards with r
  have hs := fourier_slice (f := fun x => (f x : ℂ))
    (complexifySchwartz f).integrable θ r
  rw [hs]

/-- Exact reconstruction from the canonical filtered hyperplane Radon transform,
with normalized spherical measure and explicit dimension-dependent constant. -/
theorem normalized_filtered_radon_reconstruction (f : 𝓢(V, ℝ)) (x : V) :
    f x = ∫ θ : sphere (0 : V) 1,
      (filteredRadonProfile (fun y => (f y : ℂ)) θ (inner ℝ θ.val x)).re
        ∂normalizedSphere V := by
  rw [normalized_real_schwartz_reconstruction f x]
  apply integral_congr_ae
  filter_upwards with θ
  have h := congrArg Complex.re (real_ridge_eq_filtered_radon f θ (inner ℝ θ.val x))
  simpa only [Complex.ofReal_re] using h

/-- The canonical Radon reconstruction for a target known only near the finite
outer ball. The explicit ball-mixture extension supplies Schwartz regularity;
no arbitrary whole-space extension or ridge identity is assumed. -/
theorem local_radial_filtered_reconstruction
    {f : V → ℝ} {w : ℝ → ℝ} {R₀ R : ℝ} {U : Set V}
    (hR₀ : 0 < R₀) (hU : IsOpen U) (hcover : closedBall (0 : V) R ⊆ U)
    (hf : ContDiffOn ℝ ∞ f U)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Set.Icc R₀ R)
    (hw : ContDiff ℝ ∞ w) (hnormalized : (∫ a in Set.Icc R₀ R, w a) = 1)
    {x : V} (hx : ‖x‖ ≤ R₀) :
    f x = ∫ θ : sphere (0 : V) 1,
      (filteredRadonProfile (fun y => (radialExtension f w R₀ R y : ℂ))
        θ (inner ℝ θ.val x)).re ∂normalizedSphere V := by
  let F := radialExtensionSchwartz hR₀ hU hcover hf hi hsupp hw hnormalized
  have hr := normalized_filtered_radon_reconstruction F x
  have hval : F x = f x := radialExtension_eq hnormalized hx
  rw [hval] at hr
  exact hr

end QIRadon.Radon
