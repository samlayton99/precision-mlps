import QIRadon.Radon.RotationLift

/-! The angular Hilbert tail as an actual fixed real spatial integral. This
avoids assuming scalar analyticity outside the certified inner interval. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology RealInnerProductSpace
namespace QIRadon.Radon

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]

private def tailWeight (L : ℝ) (z : ℂ) (t : ℝ) : ℂ :=
  ((Icc (-L) L)ᶜ).indicator (fun t : ℝ => (z-(t:ℂ))⁻¹) t

private theorem tailWeight_measurable (L : ℝ) (z : ℂ) : Measurable (tailWeight L z) := by
  exact ((measurable_const.sub Complex.continuous_ofReal.measurable).inv).indicator measurableSet_Icc.compl

private theorem tailWeight_norm_le {L Q : ℝ} (hQ : Q < L) {z : ℂ} (hz : ‖z‖ < Q) (t : ℝ) :
    ‖tailWeight L z t‖ ≤ (L-Q)⁻¹ := by
  by_cases ht : t ∈ (Icc (-L) L)ᶜ
  · rw [tailWeight, indicator_of_mem ht, norm_inv]
    exact inv_anti₀ (by linarith) (outside_complex_distance hQ hz ht)
  · rw [tailWeight, indicator_of_notMem ht, norm_zero]
    positivity

/-- The nonlocal tail is exactly a spatial integral in the actual volume-
preserving frame, so angular continuation needs only rotated target evaluations. -/
theorem hilbertOuter_eq_spatial {f : V → ℂ} (hf : Integrable f)
    (θ : sphere (0:V) 1) {L Q : ℝ} (hQ : Q < L) {z : ℂ} (hz : ‖z‖ < Q) :
    (∫ t in (Icc (-L) L)ᶜ, hyperplaneRadon f θ t/(z-(t:ℂ))) =
      ∫ x : V, tailWeight L z (inner ℝ x θ.val)*f x := by
  have hweightMeas : Measurable (fun x : V => tailWeight L z (inner ℝ x θ.val)) :=
    (tailWeight_measurable L z).comp (by fun_prop)
  have hi : Integrable (fun x : V => tailWeight L z (inner ℝ x θ.val)*f x) :=
    hf.bdd_mul (c := (L-Q)⁻¹) hweightMeas.aestronglyMeasurable
      (Filter.Eventually.of_forall (fun x => tailWeight_norm_le hQ hz (inner ℝ x θ.val)))
  have hframe := ((frame_measurePreserving θ).integrable_comp_emb (frame_measurableEmbedding θ)).mpr hi
  have heq : (fun q : ℝ × Hyperplane θ => tailWeight L z (inner ℝ (frame θ q) θ.val)*f (frame θ q)) =
      fun q => tailWeight L z q.1*f (frame θ q) := by
    funext q
    rw [inner_frame_direction]
  simp only [Function.comp_def] at hframe
  rw [heq] at hframe
  calc
    _ = ∫ t : ℝ, tailWeight L z t*hyperplaneRadon f θ t := by
      rw [← integral_indicator measurableSet_Icc.compl]
      apply integral_congr_ae
      filter_upwards with t
      by_cases ht : t ∈ (Icc (-L) L)ᶜ
      · simp [tailWeight, ht, div_eq_mul_inv, mul_comm]
      · simp [tailWeight, ht]
    _ = ∫ t : ℝ, ∫ y : Hyperplane θ, tailWeight L z t*f (frame θ (t,y)) := by
      simp only [integral_const_mul, frame_apply, hyperplaneRadon]
    _ = ∫ q : ℝ × Hyperplane θ, tailWeight L z q.1*f (frame θ q) ∂(volume.prod volume) :=
      (integral_prod _ hframe).symm
    _ = ∫ x : V, tailWeight L z (inner ℝ x θ.val)*f x := by
      rw [← heq]
      exact (frame_measurePreserving θ).integral_comp (frame_measurableEmbedding θ)
        (fun x : V => tailWeight L z (inner ℝ x θ.val)*f x)

/-- The actual finite real region used to compute the tail. -/
def tailSpatialDomain (θ : sphere (0:V) 1) (L R : ℝ) : Set V :=
  closedBall 0 R ∩ {x | inner ℝ x θ.val ∉ Icc (-L) L}

theorem measurableSet_tailSpatialDomain (θ : sphere (0:V) 1) (L R : ℝ) :
    MeasurableSet (tailSpatialDomain θ L R) :=
  measurableSet_closedBall.inter ((measurableSet_Icc.compl).preimage (by fun_prop))

/-- For the finite radial extension, the entire outer tail is a finite spatial
integral; no values beyond the known target's outer ball are needed. -/
theorem hilbertOuter_eq_spatial_finite {f : V → ℂ} (hf : Integrable f)
    (θ : sphere (0:V) 1) {L Q R : ℝ} (hQ : Q < L) {z : ℂ} (hz : ‖z‖ < Q)
    (hsupp : ∀ x, R < ‖x‖ → f x=0) :
    (∫ t in (Icc (-L) L)ᶜ, hyperplaneRadon f θ t/(z-(t:ℂ))) =
      ∫ x in tailSpatialDomain θ L R, f x/(z-(inner ℝ x θ.val:ℂ)) := by
  rw [hilbertOuter_eq_spatial hf θ hQ hz, ← integral_indicator (measurableSet_tailSpatialDomain θ L R)]
  apply integral_congr_ae
  filter_upwards with x
  by_cases hR : x ∈ closedBall (0:V) R
  · by_cases ht : inner ℝ x θ.val ∈ (Icc (-L) L)ᶜ
    · have hdom : x ∈ tailSpatialDomain θ L R := ⟨hR, ht⟩
      rw [indicator_of_mem hdom, tailWeight, indicator_of_mem ht]
      simp only [div_eq_mul_inv, mul_comm]
    · have hdom : x ∉ tailSpatialDomain θ L R := fun h => ht h.2
      rw [indicator_of_notMem hdom, tailWeight, indicator_of_notMem ht, zero_mul]
  · have hx : R < ‖x‖ := by simpa only [mem_closedBall_zero_iff, not_le] using hR
    have hdom : x ∉ tailSpatialDomain θ L R := fun h => hR h.1
    rw [indicator_of_notMem hdom, hsupp x hx, mul_zero]

section RotationTail
open QIRadon.ComplexRotation
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Joint angles and scalar values with explicit separation from the outer tail. -/
def rotationTailDomain (n : ℕ) (Q η : ℝ) : Set ((Fin n → ℂ) × ℂ) :=
  {p | (∑ i, ‖p.1 i‖) < η ∧ ‖p.2‖ < Q}

theorem isOpen_rotationTailDomain (n : ℕ) (Q η : ℝ) :
    IsOpen (rotationTailDomain n Q η) := by
  have hc : Continuous (fun p : (Fin n → ℂ) × ℂ => ∑ i, ‖p.1 i‖) := by fun_prop
  exact (isOpen_lt hc continuous_const).inter
    (isOpen_lt (by fun_prop : Continuous (fun p : (Fin n → ℂ) × ℂ => ‖p.2‖)) continuous_const)

/-- The actually evaluated finite spatial tail under complex rotation words. -/
def rotatedHilbertTail {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ) (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ) (R₀ R L : ℝ)
    (p : (Fin n → ℂ) × ℂ) : ℂ :=
  ∫ x in tailSpatialDomain θ L R,
    rotationTarget F pairs p.1 (realEmbedding x) * (radialCutoff w R₀ R ‖x‖:ℂ) /
      (p.2-(inner ℝ x θ.val:ℂ))

/-- Joint complex smoothness of the nonlocal tail follows from bounded analytic
target evaluations on one fixed finite real region. -/
theorem rotatedHilbertTail_smooth {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {R₀ R L Q η σ₀ σ B : ℝ}
    (hR₀ : 0 < R₀) (hQ : Q < L) (hη : 0 ≤ η) (hσ₀ : 0 < σ₀) (hB : 0 ≤ B)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R) (hw : ContDiff ℝ ∞ w)
    (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a) (hnormalized : (∫ a in Icc R₀ R, w a)=1) :
    ContDiffOn ℂ ∞ (rotatedHilbertTail F pairs θ w R₀ R L) (rotationTailDomain n Q η) := by
  let D := tailSpatialDomain θ L R
  have hDmeas : MeasurableSet D := measurableSet_tailSpatialDomain θ L R
  have hDfin : volume D < ⊤ := lt_of_le_of_lt (measure_mono inter_subset_left)
    (isCompact_closedBall (0:EuclideanSpace ℝ ι) R).measure_lt_top
  let : IsFiniteMeasure (volume.restrict D) := ⟨by rw [Measure.restrict_apply_univ]; exact hDfin⟩
  have hcut : Continuous (fun x : EuclideanSpace ℝ ι => (radialCutoff w R₀ R ‖x‖:ℂ)) :=
    Complex.continuous_ofReal.comp (radialCutoff_norm_contDiff hR₀ hi hsupp hw hnormalized).continuous
  have hxin : ∀ x ∈ D, realEmbedding x ∈ realBallTube realEmbedding R σ₀ := by
    intro x hx
    exact mem_iUnion.mpr ⟨x, mem_iUnion.mpr ⟨hx.1, mem_ball_self hσ₀⟩⟩
  have hden : ∀ p ∈ rotationTailDomain n Q η, ∀ x ∈ D, p.2-(inner ℝ x θ.val:ℂ) ≠ 0 := by
    intro p hp x hx
    have hd := outside_complex_distance hQ hp.2 hx.2
    intro heq
    rw [heq, norm_zero] at hd
    linarith
  unfold rotatedHilbertTail
  apply ParameterIntegral.smoothOn_integral_of_analytic_fibers (isOpen_rotationTailDomain n Q η)
    (M := fun _ : EuclideanSpace ℝ ι => B/(L-Q))
  · intro p hp
    have ht := rotationTarget_analytic pairs hη hp.1.le hrotation hF
    have hcF : ContinuousOn (fun x : EuclideanSpace ℝ ι => rotationTarget F pairs p.1 (realEmbedding x)) D :=
      ht.continuousOn.comp realEmbedding.continuous.continuousOn hxin
    have hcont : ContinuousOn (fun x : EuclideanSpace ℝ ι =>
        rotationTarget F pairs p.1 (realEmbedding x)*(radialCutoff w R₀ R ‖x‖:ℂ)/(p.2-(inner ℝ x θ.val:ℂ))) D :=
      (hcF.mul hcut.continuousOn).div (by fun_prop) (hden p hp)
    exact hcont.aestronglyMeasurable hDmeas
  · filter_upwards [ae_restrict_mem hDmeas] with x hx
    intro p hp
    have hrot := ((rotationWord_entire pairs (realEmbedding x)) p.1 (mem_univ _)).comp
      ((ContinuousLinearMap.fst ℂ (Fin n → ℂ) ℂ).analyticAt p)
    have hm := rotationWord_mem_realBallTube pairs hη hp.1.le hrotation (hxin x hx)
    have hFc : AnalyticAt ℂ (fun q : (Fin n → ℂ) × ℂ =>
        F (rotationWord pairs q.1 (realEmbedding x))) p :=
      (hF _ hm).comp (f := fun q : (Fin n → ℂ) × ℂ => rotationWord pairs q.1 (realEmbedding x)) (x := p) hrot
    have hnum : AnalyticAt ℂ (fun q : (Fin n → ℂ) × ℂ =>
        F (rotationWord pairs q.1 (realEmbedding x))*(radialCutoff w R₀ R ‖x‖:ℂ)) p :=
      hFc.mul analyticAt_const
    have hd : AnalyticAt ℂ (fun q : (Fin n → ℂ) × ℂ => q.2-(inner ℝ x θ.val:ℂ)) p :=
      ((ContinuousLinearMap.snd ℂ (Fin n → ℂ) ℂ).analyticAt p).sub analyticAt_const
    exact hnum.div hd (hden p hp x hx)
  · filter_upwards [ae_restrict_mem hDmeas] with x hx
    intro p hp
    have htarget := rotationTarget_norm_le pairs hη hp.1.le hrotation hFBound _ (hxin x hx)
    have hnum : ‖rotationTarget F pairs p.1 (realEmbedding x)*(radialCutoff w R₀ R ‖x‖:ℂ)‖ ≤ B := by
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (radialCutoff_nonneg hwpos)]
      exact (mul_le_mul_of_nonneg_left (radialCutoff_le_one hi.integrableOn hwpos hnormalized)
        (norm_nonneg _)).trans (by simpa using htarget)
    rw [norm_div]
    exact (div_le_div_of_nonneg_right hnum (norm_nonneg _)).trans
      (div_le_div_of_nonneg_left hB (by linarith) (outside_complex_distance hQ hp.2 hx.2))
  · exact integrable_const _

/-- The finite spatial evaluator is exactly the outer piece of the true Radon
projection under each complex rotation parameter. -/
theorem rotatedHilbertTail_eq_outer {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {R₀ R L Q η σ₀ σ : ℝ}
    (hR₀ : 0 < R₀) (hQ : Q < L) (hη : 0 ≤ η) (hσ₀ : 0 < σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R) (hw : ContDiff ℝ ∞ w)
    (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationTailDomain n Q η) :
    rotatedHilbertTail F pairs θ w R₀ R L p =
      ∫ t in (Icc (-L) L)ᶜ,
        hyperplaneRadon (complexRadialExtension (rotationTarget F pairs p.1) realEmbedding w R₀ R) θ t /
          (p.2-(t:ℂ)) := by
  have ht := rotationTarget_analytic pairs hη hp.1.le hrotation hF
  let f := complexRadialExtensionSchwartz (realEmbedding (ι := ι)) hR₀ hσ₀ ht hi hsupp hw hnormalized
  have hf : Integrable (complexRadialExtension (rotationTarget F pairs p.1) realEmbedding w R₀ R) :=
    f.integrable
  exact (hilbertOuter_eq_spatial_finite hf θ hQ hp.2
    (fun x hx => complexRadialExtension_eq_zero _ realEmbedding w hx)).symm

/-- The outer real-tail mass produces the same finite volume bound under every
complex rotation word, with no scalar analyticity outside the inner interval. -/
theorem rotatedHilbertTail_norm_le {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {R₀ R L Q η σ₀ σ B : ℝ}
    (hQ : Q < L) (hη : 0 ≤ η) (hσ₀ : 0 < σ₀) (hB : 0 ≤ B)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hi : IntegrableOn w (Icc R₀ R)) (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a)
    (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationTailDomain n Q η) :
    ‖rotatedHilbertTail F pairs θ w R₀ R L p‖ ≤
      B*volume.real (closedBall (0:EuclideanSpace ℝ ι) R)/(L-Q) := by
  let D := tailSpatialDomain θ L R
  have hDmeas : MeasurableSet D := measurableSet_tailSpatialDomain θ L R
  have hballfin := (isCompact_closedBall (0:EuclideanSpace ℝ ι) R).measure_lt_top (μ := volume)
  have hDfin : volume D < ⊤ := lt_of_le_of_lt (measure_mono inter_subset_left) hballfin
  let : IsFiniteMeasure (volume.restrict D) := ⟨by rw [Measure.restrict_apply_univ]; exact hDfin⟩
  have hb : ∀ᵐ x ∂volume.restrict D,
      ‖rotationTarget F pairs p.1 (realEmbedding x)*(radialCutoff w R₀ R ‖x‖:ℂ)/(p.2-(inner ℝ x θ.val:ℂ))‖ ≤ B/(L-Q) := by
    filter_upwards [ae_restrict_mem hDmeas] with x hx
    have hxin : realEmbedding x ∈ realBallTube realEmbedding R σ₀ :=
      mem_iUnion.mpr ⟨x, mem_iUnion.mpr ⟨hx.1, mem_ball_self hσ₀⟩⟩
    have ht := rotationTarget_norm_le pairs hη hp.1.le hrotation hFBound _ hxin
    have hnum : ‖rotationTarget F pairs p.1 (realEmbedding x)*(radialCutoff w R₀ R ‖x‖:ℂ)‖ ≤ B := by
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (radialCutoff_nonneg hwpos)]
      exact (mul_le_mul_of_nonneg_left (radialCutoff_le_one hi hwpos hnormalized) (norm_nonneg _)).trans
        (by simpa using ht)
    rw [norm_div]
    exact (div_le_div_of_nonneg_right hnum (norm_nonneg _)).trans
      (div_le_div_of_nonneg_left hB (by linarith) (outside_complex_distance hQ hp.2 hx.2))
  have h := norm_integral_le_of_norm_le_const hb
  have hvol : (volume.restrict D).real univ ≤ volume.real (closedBall (0:EuclideanSpace ℝ ι) R) := by
    simp only [Measure.real, Measure.restrict_apply_univ]
    exact ENNReal.toReal_mono hballfin.ne (measure_mono inter_subset_left)
  exact h.trans (by
    calc
      B/(L-Q)*(volume.restrict D).real univ ≤
          B/(L-Q)*volume.real (closedBall (0:EuclideanSpace ℝ ι) R) :=
        mul_le_mul_of_nonneg_left hvol (by positivity)
      _ = _ := by ring)

end RotationTail
end QIRadon.Radon
