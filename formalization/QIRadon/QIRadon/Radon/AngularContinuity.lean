import QIRadon.Radon.Reconstruction
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals

/-! Continuity of the actual canonical angular integrand, derived from the
Fourier-ray formula. The dominating function is independent of the unit direction
and scalar coordinate, and follows from two Schwartz seminorms. -/
set_option maxHeartbeats 800000
noncomputable section
open MeasureTheory Set Metric
open scoped FourierTransform SchwartzMap
namespace QIRadon.Radon
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
/-- One explicit integrable ray majorant, uniform over every unit direction.
Its numerator is a pair of actual Schwartz seminorms, not an assumed bound. -/
theorem schwartz_ray_majorant (F : 𝓢(V, ℂ)) (k : ℕ)
    (θ : sphere (0:V) 1) (r : ℝ) :
    |r|^k * ‖F (r • θ.val)‖ ≤
      (SchwartzMap.seminorm ℝ k 0 F + SchwartzMap.seminorm ℝ (k+2) 0 F)/(1+r^2) := by
  have hθ : ‖θ.val‖ = 1 := mem_sphere_zero_iff_norm.mp θ.property
  have hnorm : ‖r • θ.val‖ = |r| := by simp only [norm_smul, Real.norm_eq_abs, hθ, mul_one]
  have h₁ := SchwartzMap.norm_pow_mul_le_seminorm ℝ F k (r • θ.val)
  have h₂ := SchwartzMap.norm_pow_mul_le_seminorm ℝ F (k+2) (r • θ.val)
  rw [hnorm] at h₁ h₂
  rw [pow_add, sq_abs] at h₂
  apply (le_div_iff₀ (by positivity : 0 < 1+r^2)).mpr
  nlinarith

/-- The full signed Fourier-ray integral varies continuously with both the
unit direction and scalar position. All domination comes from the actual
Schwartz transform, so there is no angular-continuity assumption. -/
theorem fullFrequencyIntegral_continuous (f : 𝓢(V, ℂ)) :
    Continuous (fun p : sphere (0:V) 1 × ℝ =>
      ∫ r : ℝ, fullFrequencyIntegrand f p.1 p.2 r) := by
  let k := Module.finrank ℝ V-1
  let C := SchwartzMap.seminorm ℝ k 0 (𝓕 f) + SchwartzMap.seminorm ℝ (k+2) 0 (𝓕 f)
  apply continuous_of_dominated (bound := fun r : ℝ => C/(1+r^2))
  · intro p
    exact (integrable_fullFrequency_schwartz f p.1 p.2).aestronglyMeasurable
  · intro p
    filter_upwards with r
    simp only [fullFrequencyIntegrand, norm_smul, Circle.norm_smul,
      Real.norm_eq_abs, abs_pow, abs_abs, ← SchwartzMap.fourier_coe]
    exact schwartz_ray_majorant (𝓕 f) k p.1 r
  · simpa only [div_eq_mul_inv] using integrable_inv_one_add_sq.const_mul C
  · filter_upwards with r
    have harg : Continuous (fun p : sphere (0:V) 1 × ℝ => r • p.1.val) :=
      continuous_const.smul (continuous_subtype_val.comp continuous_fst)
    have hf : Continuous (fun p : sphere (0:V) 1 × ℝ => 𝓕 (f:V→ℂ) (r • p.1.val)) := by
      rw [← SchwartzMap.fourier_coe]
      exact (𝓕 f).continuous.comp harg
    have hp : Continuous (fun p : sphere (0:V) 1 × ℝ => Real.fourierChar (r*p.2)) :=
      Real.continuous_fourierChar.comp (continuous_const.mul continuous_snd)
    exact (continuous_const : Continuous (fun _ : sphere (0:V) 1 × ℝ => |r|^k)).smul
      (hp.smul hf)

/-- Joint continuity of the actual real canonical ridge profile. -/
theorem realRidgeProfile_joint_continuous (f : 𝓢(V, ℝ)) :
    Continuous (fun p : sphere (0:V) 1 × ℝ => realRidgeProfile f p.1 p.2) := by
  have hc : Continuous (fun p : sphere (0:V) 1 × ℝ =>
      (surfaceArea V/2 : ℂ) * ∫ r : ℝ, fullFrequencyIntegrand (complexifySchwartz f) p.1 p.2 r) :=
    continuous_const.mul (fullFrequencyIntegral_continuous (complexifySchwartz f))
  apply (Complex.continuous_re.comp hc).congr
  intro p
  have hh := congrArg Complex.re (real_ridge_full_frequency_schwartz f p.1 p.2)
  simpa only [Complex.ofReal_re, Function.comp_apply, fullFrequencyIntegrand,
    show ((complexifySchwartz f):V→ℂ)=(fun y => (f y:ℂ)) from rfl] using hh.symm

/-- Continuity at the actual input-dependent coordinate theta·x. -/
theorem realRidgeProfile_angular_continuous (f : 𝓢(V, ℝ)) (x : V) :
    Continuous (fun θ : sphere (0:V) 1 => realRidgeProfile f θ (inner ℝ θ.val x)) := by
  exact (realRidgeProfile_joint_continuous f).comp
    (continuous_id.prodMk (continuous_subtype_val.inner continuous_const))

/-- The approved centered canonical angular object, as an actual continuous
function on the sphere. No regularity or approximation conclusion is assumed. -/
def canonicalAngular (f : 𝓢(V, ℝ)) (x : V) : C(sphere (0:V) 1, ℝ) where
  toFun θ := realRidgeProfile f θ (inner ℝ θ.val x)-realRidgeProfile f θ 0
  continuous_toFun := by
    have hz : Continuous (fun θ : sphere (0:V) 1 => realRidgeProfile f θ 0) :=
      (realRidgeProfile_joint_continuous f).comp
        (show Continuous (fun θ : sphere (0:V) 1 => (θ, (0:ℝ))) from
          continuous_id.prodMk continuous_const)
    exact (realRidgeProfile_angular_continuous f x).sub hz

@[simp] theorem canonicalAngular_apply (f : 𝓢(V, ℝ)) (x : V) (θ : sphere (0:V) 1) :
    canonicalAngular f x θ = realRidgeProfile f θ (inner ℝ θ.val x)-realRidgeProfile f θ 0 := rfl

/-- Exact complexification matches the source's canonical filtered profiles. -/
theorem canonicalAngular_eq_filtered (f : 𝓢(V, ℝ)) (x : V) (θ : sphere (0:V) 1) :
    (canonicalAngular f x θ : ℂ) =
      filteredRadonProfile (complexifySchwartz f) θ (inner ℝ θ.val x)-
        filteredRadonProfile (complexifySchwartz f) θ 0 := by
  rw [canonicalAngular_apply, Complex.ofReal_sub, real_ridge_eq_filtered_radon,
    real_ridge_eq_filtered_radon]
  rfl

@[simp] theorem canonicalAngular_zero (f : 𝓢(V, ℝ)) : canonicalAngular f 0 = 0 := by
  ext θ
  simp [canonicalAngular_apply]

/-- The actual centered canonical angular object has the antipodal symmetry
needed by the folded quadrature rule. -/
theorem canonicalAngular_antipodal (f : 𝓢(V, ℝ)) (x : V) (θ : sphere (0:V) 1) :
    canonicalAngular f x (antipode θ) = canonicalAngular f x θ := by
  simp only [canonicalAngular_apply, antipode_val, inner_neg_left]
  rw [real_ridge_antipodal]
  have hz : realRidgeProfile f (antipode θ) 0 = realRidgeProfile f θ 0 := by
    simpa only [neg_zero] using real_ridge_antipodal f θ 0
  rw [hz]

/-- Continuity directly in the source assembly's complex filtered-profile
notation, derived from the same actual canonical angular function. -/
theorem filteredRadonProfile_centered_continuous (f : 𝓢(V, ℝ)) (x : V) :
    Continuous (fun θ : sphere (0:V) 1 =>
      filteredRadonProfile (complexifySchwartz f) θ (inner ℝ θ.val x)-
        filteredRadonProfile (complexifySchwartz f) θ 0) := by
  have hc := Complex.continuous_ofReal.comp (canonicalAngular f x).continuous
  apply hc.congr
  intro θ
  exact canonicalAngular_eq_filtered f x θ

end QIRadon.Radon
