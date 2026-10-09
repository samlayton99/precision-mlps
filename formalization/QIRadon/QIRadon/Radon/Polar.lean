import Mathlib

/-! Actual polar-coordinate integration underlying the Radon ridge decomposition.
The radial power is carried by Mathlib's explicit `volumeIoiPow` measure. -/

namespace QIRadon

open MeasureTheory Set Metric
open scoped RealInnerProductSpace FourierTransform

noncomputable section

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem integral_polar_product (f : V → E) :
    (∫ x, f x) =
      ∫ z : sphere (0 : V) 1 × Ioi (0 : ℝ), f (z.2.val • z.1.val)
        ∂((volume : Measure V).toSphere.prod
          (Measure.volumeIoiPow (Module.finrank ℝ V - 1))) := by
  let e := homeomorphUnitSphereProd V
  let g : sphere (0 : V) 1 × Ioi (0 : ℝ) → E := fun z => f (z.2.val • z.1.val)
  have hpres := (volume : Measure V).measurePreserving_homeomorphUnitSphereProd
  calc
    (∫ x, f x) =
        ∫ x : ({(0 : V)}ᶜ : Set V), f x.val ∂((volume : Measure V).comap Subtype.val) := by
      rw [integral_subtype_comap (measurableSet_singleton _).compl]
      rw [restrict_compl_singleton]
    _ = ∫ z, g z ∂((volume : Measure V).toSphere.prod
          (Measure.volumeIoiPow (Module.finrank ℝ V - 1))) := by
      have h := hpres.integral_comp (Homeomorph.measurableEmbedding _) g
      convert h using 1
      apply integral_congr_ae
      filter_upwards [] with x
      change f x.val = f ((e x).2.val • (e x).1.val)
      congr 1
      have he := congrArg Subtype.val (e.symm_apply_apply x)
      simpa only [e, homeomorphUnitSphereProd_symm_apply_coe] using he.symm
    _ = _ := rfl

omit [Nontrivial V] [NormedSpace ℝ E] in
theorem integrable_polar_product {f : V → E} (hf : Integrable f) :
    Integrable (fun z : sphere (0 : V) 1 × Ioi (0 : ℝ) => f (z.2.val • z.1.val))
      ((volume : Measure V).toSphere.prod
        (Measure.volumeIoiPow (Module.finrank ℝ V - 1))) := by
  let e := homeomorphUnitSphereProd V
  let g : sphere (0 : V) 1 × Ioi (0 : ℝ) → E := fun z => f (z.2.val • z.1.val)
  have hs : Integrable (fun x : ({(0 : V)}ᶜ : Set V) => f x.val)
      ((volume : Measure V).comap Subtype.val) :=
    (integrableOn_iff_comap_subtypeVal (measurableSet_singleton _).compl).mp hf.restrict
  have heq : g ∘ e = fun x : ({(0 : V)}ᶜ : Set V) => f x.val := by
    funext x
    change f ((e x).2.val • (e x).1.val) = f x.val
    congr 1
    have he := congrArg Subtype.val (e.symm_apply_apply x)
    simpa only [e, homeomorphUnitSphereProd_symm_apply_coe] using he
  have hc : Integrable (g ∘ e) ((volume : Measure V).comap Subtype.val) := by
    rw [heq]
    exact hs
  exact ((volume : Measure V).measurePreserving_homeomorphUnitSphereProd.integrable_comp_emb
    (Homeomorph.measurableEmbedding _)).mp hc

theorem integral_polar_iterated {f : V → E} (hf : Integrable f) :
    (∫ x, f x) =
      ∫ θ : sphere (0 : V) 1,
        ∫ r : Ioi (0 : ℝ), f (r.val • θ.val)
          ∂(Measure.volumeIoiPow (Module.finrank ℝ V - 1))
        ∂((volume : Measure V).toSphere) := by
  rw [integral_polar_product]
  exact integral_prod _ (integrable_polar_product hf)

/-- Positive-frequency profiles, with unnormalized surface measure. The radial
Jacobian r^(d-1) is carried explicitly by `volumeIoiPow`. -/
def positiveRidgeProfile (f : V → ℂ) (θ : sphere (0 : V) 1) (s : ℝ) : ℂ :=
  ∫ r : Ioi (0 : ℝ), Real.fourierChar (r.val * s) • (𝓕 f) (r.val • θ.val)
    ∂(Measure.volumeIoiPow (Module.finrank ℝ V - 1))

/-- Actual directional Fourier reconstruction. This is a precursor to the
normalized antipodally folded real Radon formula, not an assumption of it. -/
theorem positive_ridge_reconstruction {f : V → ℂ}
    (hf : Integrable f) (hF : Integrable (𝓕 f)) {x : V} (hx : ContinuousAt f x) :
    f x = ∫ θ : sphere (0 : V) 1,
      positiveRidgeProfile f θ (inner ℝ θ.val x) ∂((volume : Measure V).toSphere) := by
  have hi : Integrable (fun ξ : V => Real.fourierChar (inner ℝ ξ x) • (𝓕 f) ξ) := by
    simpa only [inner_neg_right, neg_neg] using
      (Real.fourierIntegral_convergent_iff (-x)).mpr hF
  rw [← hf.fourierInv_fourier_eq hF hx, Real.fourierInv_eq]
  rw [integral_polar_iterated hi]
  apply integral_congr_ae
  filter_upwards [] with θ
  unfold positiveRidgeProfile
  apply integral_congr_ae
  filter_upwards [] with r
  simp only [real_inner_smul_left]

end
end QIRadon
