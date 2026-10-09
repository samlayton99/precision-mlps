import QIRadon.Radon.RotationTail
import QIRadon.Radon.ParameterizedDSlope

/-! Joint complex rotation/scalar continuation of the exact finite Hilbert
formula, followed by the canonical dimension-dependent scalar filter. -/
set_option maxHeartbeats 800000
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology
namespace QIRadon.Radon
open QIRadon.ComplexRotation
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

def rotationBudget (n : ℕ) (η : ℝ) : Set (Fin n → ℂ) := {a | (∑ i, ‖a i‖)<η}

theorem isOpen_rotationBudget (n : ℕ) (η : ℝ) : IsOpen (rotationBudget n η) :=
  isOpen_lt (by fun_prop : Continuous (fun a : Fin n → ℂ => ∑ i, ‖a i‖)) continuous_const

/-- The complete finite Hilbert continuation, with an actual spatial tail. -/
def rotatedHilbert {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ) (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ) (R₀ R L : ℝ)
    (p : (Fin n → ℂ) × ℂ) : ℂ :=
  (Real.pi:ℂ)⁻¹ * (-(∫ t in Icc (-L) L,
    dslope (fun z => rotatedCrossSection F pairs θ w R₀ R (p.1,z)) (t:ℂ) p.2) +
    hilbertLog L p.2 * rotatedCrossSection F pairs θ w R₀ R p +
    rotatedHilbertTail F pairs θ w R₀ R L p)

/-- Joint smoothness of the actual finite Hilbert continuation on an explicitly
smaller scalar tube, under the same target and radius-density hypotheses. -/
theorem rotatedHilbert_smooth {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {L δ Q R₀ R σ₀ σ η B S ε Qh : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hgap : L+δ ≤ Q) (hQR : Q < R₀)
    (hη : 0 ≤ η) (hB : 0 ≤ B) (hσ₀ : 0 < σ₀)
    (hS : S ≤ L) (hεδ : ε ≤ δ/2) (hSQh : S+ε ≤ Qh) (hQhL : Qh < L)
    (hscalar : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R) (hw : ContDiff ℝ ∞ w)
    (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a) (hnormalized : (∫ a in Icc R₀ R, w a)=1) :
    ContDiffOn ℂ ∞ (rotatedHilbert F pairs θ w R₀ R L) (rotationSectionDomain n S ε η) := by
  let P := rotatedCrossSection F pairs θ w R₀ R
  let PB := volume.real (closedBall (0:Hyperplane θ) 1)*Real.sqrt (R^2+Q^2)^(Module.finrank ℝ (EuclideanSpace ℝ ι)-1)*B
  have hPB : 0 ≤ PB := by dsimp [PB]; positivity
  have hP : ContDiffOn ℂ ∞ P (rotationBudget n η ×ˢ scalarTube L δ) :=
    rotatedCrossSection_smooth pairs θ hL hδ hgap hQR hη hB hscalar hrotation hF hFBound hw.continuous hi.integrableOn
  have hPb : ∀ p ∈ rotationBudget n η ×ˢ scalarTube L δ, ‖P p‖ ≤ PB := by
    intro p hp
    exact rotatedCrossSection_norm_le pairs θ hL hδ hgap hQR hη hB hscalar hrotation
      hF hFBound hw.continuous hi.integrableOn hwpos hnormalized hp
  have hsub : rotationSectionDomain n S ε η ⊆ rotationBudget n η ×ˢ scalarTube L (δ/2) :=
    fun p hp => ⟨hp.1, scalarTube_mono hS hεδ hp.2⟩
  have hsubP : rotationSectionDomain n S ε η ⊆ rotationBudget n η ×ˢ scalarTube L δ :=
    fun p hp => ⟨hp.1, scalarTube_mono hS (by linarith) hp.2⟩
  have hin := (parameterized_hilbertInner_smooth (isOpen_rotationBudget n η) hδ hPB hP hPb).mono hsub
  have htail := (rotatedHilbertTail_smooth pairs θ (by linarith) hQhL hη hσ₀ hB
    hrotation hF hFBound hi hsupp hw hwpos hnormalized).mono
      (show rotationSectionDomain n S ε η ⊆ rotationTailDomain n Qh η from
        fun p hp => ⟨hp.1, scalarTube_norm_lt hSQh hp.2⟩)
  have hlog : ContDiffOn ℂ ∞ (fun p : (Fin n → ℂ) × ℂ => hilbertLog L p.2)
      (rotationSectionDomain n S ε η) := by
    have hsmooth : ContDiffOn ℂ ∞ (hilbertLog L) (ball (0:ℂ) Qh) :=
      (hilbertLog_analytic (Q := Qh) hQhL.le).contDiffOn isOpen_ball.uniqueDiffOn
    exact hsmooth.comp (by fun_prop) (fun p hp =>
      mem_ball_zero_iff.mpr (scalarTube_norm_lt hSQh hp.2))
  exact contDiffOn_const.mul ((hin.add (hlog.mul (hP.mono hsubP))).add htail)

/-- Uniform joint bound with the exact approved logarithmic endpoint term. -/
theorem rotatedHilbert_norm_le {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {L δ Q R₀ R σ₀ σ η B S ε Qh : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hgap : L+δ ≤ Q) (hQR : Q < R₀)
    (hη : 0 ≤ η) (hB : 0 ≤ B) (hσ₀ : 0 < σ₀)
    (hS : S ≤ L) (hεδ : ε ≤ δ) (hSQh : S+ε ≤ Qh) (hQh0 : 0 < Qh) (hQhL : Qh < L)
    (hscalar : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hi : IntegrableOn w (Icc R₀ R)) (hw : Continuous w)
    (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a) (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain n S ε η) :
    let PB := volume.real (closedBall (0:Hyperplane θ) 1)*Real.sqrt (R^2+Q^2)^(Module.finrank ℝ (EuclideanSpace ℝ ι)-1)*B
    ‖rotatedHilbert F pairs θ w R₀ R L p‖ ≤ Real.pi⁻¹ *
      (4*L*PB/δ + Real.log ((L+Qh)/(L-Qh))*PB + B*volume.real (closedBall (0:EuclideanSpace ℝ ι) R)/(L-Qh)) := by
  dsimp only
  let PB := volume.real (closedBall (0:Hyperplane θ) 1)*Real.sqrt (R^2+Q^2)^(Module.finrank ℝ (EuclideanSpace ℝ ι)-1)*B
  have hPB : 0 ≤ PB := by dsimp [PB]; positivity
  have htarget := rotationTarget_analytic pairs hη hp.1.le hrotation hF
  have htargetB := rotationTarget_norm_le pairs hη hp.1.le hrotation hFBound
  have hPs := complexCrossSection_analytic (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
    (realEmbedding (ι := ι)) θ hw hi hB hL hδ hgap hQR hscalar htarget.differentiableOn htargetB
  have hPb : ∀ z ∈ scalarTube L δ, ‖rotatedCrossSection F pairs θ w R₀ R (p.1,z)‖ ≤ PB := by
    intro z hz
    exact rotatedCrossSection_norm_le pairs θ hL hδ hgap hQR hη hB hscalar hrotation hF hFBound
      hw hi hwpos hnormalized ⟨hp.1,hz⟩
  have hz := scalarTube_mono hS hεδ hp.2
  have hin := hilbertInner_norm_le hL hδ hPB hPs.differentiableOn hPb hz
  have hzQ : ‖p.2‖ < Qh := scalarTube_norm_lt hSQh hp.2
  have hlog := hilbertLog_norm_le_sharp hQh0 hQhL hzQ.le
  have htail := rotatedHilbertTail_norm_le pairs θ hQhL hη hσ₀ hB hrotation hFBound hi hwpos hnormalized
    (show p ∈ rotationTailDomain n Qh η from ⟨hp.1,hzQ⟩)
  have hprod : ‖hilbertLog L p.2*rotatedCrossSection F pairs θ w R₀ R p‖ ≤
      Real.log ((L+Qh)/(L-Qh))*PB := by
    rw [norm_mul]
    exact mul_le_mul hlog (hPb p.2 hz) (norm_nonneg _) ((norm_nonneg _).trans hlog)
  unfold rotatedHilbert
  rw [norm_mul, norm_inv, Complex.norm_real, Real.norm_eq_abs, abs_of_pos Real.pi_pos]
  exact mul_le_mul_of_nonneg_left
    ((norm_add_le _ _).trans (add_le_add ((norm_add_le _ _).trans (add_le_add hin hprod)) htail))
    (by positivity)

/-- Exact identification with the scalar finite Hilbert continuation of the
actual rotated target projection. -/
theorem rotatedHilbert_eq_continuation {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {R₀ R L Qh η σ₀ σ : ℝ}
    (hR₀ : 0 < R₀) (hQhL : Qh < L) (hη : 0 ≤ η) (hσ₀ : 0 < σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R) (hw : ContDiff ℝ ∞ w)
    (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationTailDomain n Qh η) :
    rotatedHilbert F pairs θ w R₀ R L p =
      hilbertContinuation
        (hyperplaneRadon (complexRadialExtension (rotationTarget F pairs p.1) realEmbedding w R₀ R) θ)
        (fun z => rotatedCrossSection F pairs θ w R₀ R (p.1,z)) L p.2 := by
  unfold rotatedHilbert hilbertContinuation
  rw [rotatedHilbertTail_eq_outer pairs θ hR₀ hQhL hη hσ₀ hrotation hF hi hsupp hw hnormalized hp]

/-- Uniform Cauchy bounds for the actual iterated joint scalar partials. -/
theorem iteratedScalarPartial_bound {A : Type*} [NormedAddCommGroup A]
    [NormedSpace ℂ A] [FiniteDimensional ℂ A]
    {P : A × ℂ → ℂ} {U : Set A} {L δ B : ℝ}
    (hU : IsOpen U) (hδ : 0 < δ) (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ))
    (hb : ∀ p ∈ U ×ˢ scalarTube L δ, ‖P p‖ ≤ B)
    (r : ℕ) {p : A × ℂ} (hp : p ∈ U ×ˢ scalarTube L (δ/2)) :
    ‖iteratedScalarPartial r P p‖ ≤ r.factorial*B/(δ/2)^r := by
  have hpfull : p ∈ U ×ˢ scalarTube L δ := ⟨hp.1, scalarTube_mono le_rfl (by linarith) hp.2⟩
  rw [iteratedScalarPartial_eq_iteratedDeriv hU hP r hpfull]
  apply iteratedDeriv_bound_on_smaller_tube hδ _ (fun z hz => hb (p.1,z) ⟨hp.1,hz⟩) r hp.2
  intro z hz
  have hd : DifferentiableAt ℂ P (p.1,z) := (hP.differentiableOn (by simp)).differentiableAt
    ((hU.prod (isOpen_scalarTube L δ)).mem_nhds ⟨hp.1,hz⟩)
  exact (hd.comp z (by fun_prop : DifferentiableAt ℂ (fun z : ℂ => (p.1,z)) z)).differentiableWithinAt

/-- Actual odd-dimensional filtered lift; the area/2 and inverse Fourier
frequency factors are explicit. -/
def rotatedEvenProfile {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ) (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ) (R₀ R : ℝ) (k : ℕ)
    (p : (Fin n → ℂ) × ℂ) : ℂ :=
  (surfaceArea (EuclideanSpace ℝ ι)/2:ℂ) * ((2*Real.pi*Complex.I)^(2*k))⁻¹ *
    iteratedScalarPartial (2*k) (rotatedCrossSection F pairs θ w R₀ R) p

/-- Actual even-dimensional filtered lift from the finite joint Hilbert formula. -/
def rotatedOddProfile {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ) (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ) (R₀ R L : ℝ) (k : ℕ)
    (p : (Fin n → ℂ) × ℂ) : ℂ :=
  (surfaceArea (EuclideanSpace ℝ ι)/2:ℂ) * (Complex.I*((2*Real.pi*Complex.I)^(2*k+1))⁻¹) *
    iteratedScalarPartial (2*k+1) (rotatedHilbert F pairs θ w R₀ R L) p

/-- The filters retain joint complex smoothness on their actual domains. -/
theorem rotatedEvenProfile_smooth {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ)
    (R₀ R L δ η : ℝ) (k : ℕ)
    (hP : ContDiffOn ℂ ∞ (rotatedCrossSection F pairs θ w R₀ R) (rotationSectionDomain n L δ η)) :
    ContDiffOn ℂ ∞ (rotatedEvenProfile F pairs θ w R₀ R k) (rotationSectionDomain n L δ η) :=
  contDiffOn_const.mul (iteratedScalarPartial_smooth (isOpen_rotationBudget n η) hP (2*k))

theorem rotatedOddProfile_smooth {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ)
    (R₀ R L S ε η : ℝ) (k : ℕ)
    (hH : ContDiffOn ℂ ∞ (rotatedHilbert F pairs θ w R₀ R L) (rotationSectionDomain n S ε η)) :
    ContDiffOn ℂ ∞ (rotatedOddProfile F pairs θ w R₀ R L k) (rotationSectionDomain n S ε η) :=
  contDiffOn_const.mul (iteratedScalarPartial_smooth (isOpen_rotationBudget n η) hH (2*k+1))

/-- The actual odd-dimensional filtered lift inherits the explicit Cauchy bound. -/
theorem rotatedEvenProfile_norm_le_of_bound {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ)
    (R₀ R : ℝ) {L δ η PB : ℝ} (hδ : 0 < δ) (k : ℕ)
    (hP : ContDiffOn ℂ ∞ (rotatedCrossSection F pairs θ w R₀ R) (rotationSectionDomain n L δ η))
    (hb : ∀ p ∈ rotationSectionDomain n L δ η, ‖rotatedCrossSection F pairs θ w R₀ R p‖ ≤ PB)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain n L (δ/2) η) :
    ‖rotatedEvenProfile F pairs θ w R₀ R k p‖ ≤
      ‖(surfaceArea (EuclideanSpace ℝ ι)/2:ℂ)*((2*Real.pi*Complex.I)^(2*k))⁻¹‖ *
        ((2*k).factorial*PB/(δ/2)^(2*k)) := by
  unfold rotatedEvenProfile
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_left
    (iteratedScalarPartial_bound (isOpen_rotationBudget n η) hδ hP hb (2*k) hp) (norm_nonneg _)

theorem rotatedOddProfile_norm_le_of_bound {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) (w : ℝ → ℝ)
    (R₀ R L : ℝ) {S ε η HB : ℝ} (hε : 0 < ε) (k : ℕ)
    (hH : ContDiffOn ℂ ∞ (rotatedHilbert F pairs θ w R₀ R L) (rotationSectionDomain n S ε η))
    (hb : ∀ p ∈ rotationSectionDomain n S ε η, ‖rotatedHilbert F pairs θ w R₀ R L p‖ ≤ HB)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain n S (ε/2) η) :
    ‖rotatedOddProfile F pairs θ w R₀ R L k p‖ ≤
      ‖(surfaceArea (EuclideanSpace ℝ ι)/2:ℂ)*(Complex.I*((2*Real.pi*Complex.I)^(2*k+1))⁻¹)‖ *
        ((2*k+1).factorial*HB/(ε/2)^(2*k+1)) := by
  unfold rotatedOddProfile
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_left
    (iteratedScalarPartial_bound (isOpen_rotationBudget n η) hε hH hb (2*k+1) hp) (norm_nonneg _)

section CanonicalAgreement
variable [Nonempty ι]

/-- At every real scalar offset in the certified interval, the odd-dimensional
joint lift is the original canonical Radon profile of the actual rotated target. -/
theorem rotatedEvenProfile_eq_filtered {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {L δ Q R₀ R σ₀ σ η B : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hgap : L+δ ≤ Q) (hQR : Q < R₀)
    (hη : 0 ≤ η) (hB : 0 ≤ B) (hσ₀ : 0 < σ₀)
    (hscalar : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R) (hw : ContDiff ℝ ∞ w)
    (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a) (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    (k : ℕ) (hdim : Module.finrank ℝ (EuclideanSpace ℝ ι)-1=2*k)
    {angles : Fin n → ℂ} (ha : angles ∈ rotationBudget n η) {s : ℝ} (hs : |s| < L) :
    rotatedEvenProfile F pairs θ w R₀ R k (angles,(s:ℂ)) =
      filteredRadonProfile (complexRadialExtension (rotationTarget F pairs angles) realEmbedding w R₀ R) θ s := by
  have hR₀ : 0 < R₀ := by linarith
  have ht := rotationTarget_analytic pairs hη ha.le hrotation hF
  have hb := rotationTarget_norm_le pairs hη ha.le hrotation hFBound
  let f := complexRadialExtensionSchwartz (F := rotationTarget F pairs angles) (realEmbedding (ι := ι)) hR₀ hσ₀ ht hi hsupp hw hnormalized
  let p := projectionSchwartz f θ
  let P : ℂ → ℂ := fun z => rotatedCrossSection F pairs θ w R₀ R (angles,z)
  have hcert := canonicalProjection_certificate (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
    (realEmbedding (ι := ι)) θ hR₀ hσ₀ hB hL hδ hgap hQR hscalar ht hb hi hsupp hw hwpos hnormalized
  have hPs : DifferentiableOn ℂ P (scalarTube L δ) := hcert.1.differentiableOn
  have hag : ∀ t ∈ Icc (-L) L, p t=P (t:ℂ) := hcert.2.2.1
  have hPjoint := rotatedCrossSection_smooth pairs θ hL hδ hgap hQR hη hB hscalar hrotation hF hFBound
    hw.continuous hi.integrableOn
  have hpoint : (angles,(s:ℂ)) ∈ rotationBudget n η ×ˢ scalarTube L δ :=
    ⟨ha, ofReal_mem_scalarTube hδ (abs_le.mp hs.le)⟩
  have hd := iteratedScalarPartial_eq_iteratedDeriv (isOpen_rotationBudget n η) hPjoint (2*k) hpoint
  calc
    _ = (surfaceArea (EuclideanSpace ℝ ι)/2:ℂ)*evenFilterContinuation P k (s:ℂ) := by
      unfold rotatedEvenProfile evenFilterContinuation
      rw [hd]
      ring
    _ = (surfaceArea (EuclideanSpace ℝ ι)/2:ℂ)*radialFilter p (2*k) s := by
      rw [evenFilterContinuation_eq_radialFilter p hδ hPs hag k hs]
    _ = filteredRadonProfile f θ s := by
      rw [filteredRadonProfile_eq_radialFilter f θ s, hdim]
    _ = _ := rfl

omit [Nonempty ι] in
/-- The actual joint Hilbert slice equals the finite scalar continuation of the
canonical inverse Fourier projection, on the whole safe scalar disk. -/
theorem rotatedHilbert_slice_eq_canonical {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {R₀ R L Qh η σ₀ σ B : ℝ}
    (hR₀ : 0 < R₀) (hQhL : Qh < L) (hη : 0 ≤ η) (hσ₀ : 0 < σ₀) (hB : 0 ≤ B)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R) (hw : ContDiff ℝ ∞ w)
    (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a) (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    {angles : Fin n → ℂ} (ha : angles ∈ rotationBudget n η) {z : ℂ} (hz : ‖z‖ < Qh)
    (f : 𝓢(EuclideanSpace ℝ ι, ℂ))
    (hf : ∀ x, f x=complexRadialExtension (rotationTarget F pairs angles) realEmbedding w R₀ R x) :
    rotatedHilbert F pairs θ w R₀ R L (angles,z) =
      hilbertContinuation (projectionSchwartz f θ)
        (fun v => rotatedCrossSection F pairs θ w R₀ R (angles,v)) L z := by
  rw [rotatedHilbert_eq_continuation pairs θ hR₀ hQhL hη hσ₀ hrotation hF hi hsupp hw hnormalized
    (show (angles,z) ∈ rotationTailDomain n Qh η from ⟨ha,hz⟩)]
  have hb := rotationTarget_norm_le pairs hη ha.le hrotation hFBound
  have hbd : ∀ x, ‖f x‖ ≤ B := by
    intro x
    rw [hf x]
    exact complexRadialExtension_norm_le (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
      (F := rotationTarget F pairs angles) (realEmbedding (ι := ι)) hσ₀ hB hb hi.integrableOn hwpos hnormalized x
  have hsup : ∀ x, R < ‖x‖ → f x=0 := by
    intro x hx
    rw [hf x]
    exact complexRadialExtension_eq_zero (rotationTarget F pairs angles) (realEmbedding (ι := ι)) w hx
  have hproj := projectionSchwartz_eq_radon_supported f hbd hsup θ
  rw [hproj, funext hf]

/-- Even-dimensional joint filtering agrees with the canonical Radon profile;
the Hilbert formula is an exact analytic continuation, not a substituted solver. -/
theorem rotatedOddProfile_eq_filtered {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {L δ Q R₀ R σ₀ σ η B S ε Qh : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hgap : L+δ ≤ Q) (hQR : Q < R₀)
    (hη : 0 ≤ η) (hB : 0 ≤ B) (hσ₀ : 0 < σ₀) (hε : 0 < ε)
    (hS : S ≤ L) (hεδ : ε ≤ δ/2) (hSQh : S+ε ≤ Qh) (hQhL : Qh < L)
    (hscalar : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hi : Integrable w) (hsupp : Function.support w ⊆ Icc R₀ R) (hw : ContDiff ℝ ∞ w)
    (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a) (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    (k : ℕ) (hdim : Module.finrank ℝ (EuclideanSpace ℝ ι)-1=2*k+1)
    {angles : Fin n → ℂ} (ha : angles ∈ rotationBudget n η) {s : ℝ} (hs : |s| ≤ S) :
    rotatedOddProfile F pairs θ w R₀ R L k (angles,(s:ℂ)) =
      filteredRadonProfile (complexRadialExtension (rotationTarget F pairs angles) realEmbedding w R₀ R) θ s := by
  have hR₀ : 0 < R₀ := by linarith
  have ht := rotationTarget_analytic pairs hη ha.le hrotation hF
  have hb := rotationTarget_norm_le pairs hη ha.le hrotation hFBound
  let f := complexRadialExtensionSchwartz (F := rotationTarget F pairs angles) (realEmbedding (ι := ι)) hR₀ hσ₀ ht hi hsupp hw hnormalized
  let p := projectionSchwartz f θ
  let P : ℂ → ℂ := fun z => rotatedCrossSection F pairs θ w R₀ R (angles,z)
  have hcert := canonicalProjection_certificate (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
    (realEmbedding (ι := ι)) θ hR₀ hσ₀ hB hL hδ hgap hQR hscalar ht hb hi hsupp hw hwpos hnormalized
  have hPs : DifferentiableOn ℂ P (scalarTube L δ) := hcert.1.differentiableOn
  have hPB : 0 ≤ volume.real (closedBall (0:Hyperplane θ) 1)*Real.sqrt (R^2+Q^2)^(Module.finrank ℝ (EuclideanSpace ℝ ι)-1)*B := by positivity
  have hag : ∀ t ∈ Icc (-L) L, p t=P (t:ℂ) := hcert.2.2.1
  have hHjoint := rotatedHilbert_smooth pairs θ hL hδ hgap hQR hη hB hσ₀ hS hεδ hSQh hQhL
    hscalar hrotation hF hFBound hi hsupp hw hwpos hnormalized
  have hpoint : (angles,(s:ℂ)) ∈ rotationBudget n η ×ˢ scalarTube S ε :=
    ⟨ha, ofReal_mem_scalarTube hε (abs_le.mp hs)⟩
  have hd := iteratedScalarPartial_eq_iteratedDeriv (isOpen_rotationBudget n η) hHjoint (2*k+1) hpoint
  have hsQ : |s| < Qh := by linarith
  have heq : (fun z => rotatedHilbert F pairs θ w R₀ R L (angles,z)) =ᶠ[𝓝 (s:ℂ)] hilbertContinuation p P L := by
    filter_upwards [isOpen_ball.mem_nhds (show (s:ℂ) ∈ ball (0:ℂ) Qh by
      simpa only [mem_ball_zero_iff, Complex.norm_real, Real.norm_eq_abs] using hsQ)] with z hz
    exact rotatedHilbert_slice_eq_canonical pairs θ hR₀ hQhL hη hσ₀ hB hrotation hF hFBound
      hi hsupp hw hwpos hnormalized ha (by simpa using hz) f (fun _ => rfl)
  have hderiv := heq.iteratedDeriv_eq (2*k+1)
  calc
    _ = (surfaceArea (EuclideanSpace ℝ ι)/2:ℂ)*oddFilterContinuation p P L k (s:ℂ) := by
      unfold rotatedOddProfile oddFilterContinuation
      rw [hd, hderiv]
      ring
    _ = (surfaceArea (EuclideanSpace ℝ ι)/2:ℂ)*radialFilter p (2*k+1) s := by
      rw [oddFilterContinuation_eq_radialFilter p hδ hPB hQhL hPs hcert.2.1 hag k hsQ]
    _ = filteredRadonProfile f θ s := by
      rw [filteredRadonProfile_eq_radialFilter f θ s, hdim]
    _ = _ := rfl

end CanonicalAgreement
end QIRadon.Radon
