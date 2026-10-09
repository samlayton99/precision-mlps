import QIRadon.Radon.Source
import QIRadon.Radon.ParameterIntegral
import QIRadon.Quadrature.ComplexPlaneRotation

/-! Actual complex rotation-word lifts of the finite Radon cross-section formula.
The construction is based at a real direction and a fixed real transverse ball;
it does not assume a global ambient angular analytic extension. -/
set_option maxHeartbeats 800000
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology
namespace QIRadon.Radon
open QIRadon.ComplexRotation
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Joint angle/offset parameter domain: the angle budget is ℓ¹, independent of
word length, and the scalar offset stays in its actual certified tube. -/
def rotationSectionDomain (n : ℕ) (L δ η : ℝ) : Set ((Fin n → ℂ) × ℂ) :=
  {p | (∑ i, ‖p.1 i‖) < η ∧ p.2 ∈ scalarTube L δ}

theorem isOpen_rotationSectionDomain (n : ℕ) (L δ η : ℝ) :
    IsOpen (rotationSectionDomain n L δ η) := by
  have hc : Continuous (fun p : (Fin n → ℂ) × ℂ => ∑ i, ‖p.1 i‖) := by fun_prop
  exact (isOpen_lt hc continuous_const).inter
    ((isOpen_scalarTube L δ).preimage continuous_snd)

/-- Actual complex rotations stay within the target's real-ball tube under an
explicit combined scalar/rotation displacement margin. -/
theorem rotationWord_mem_realBallTube {n : ℕ} (pairs : Fin n → ι × ι)
    {angles : Fin n → ℂ} {η R σ₀ σ : ℝ} (hη : 0 ≤ η)
    (hangles : (∑ i, ‖angles i‖) ≤ η)
    (hmargin : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    {x : EuclideanSpace ℂ ι} (hx : x ∈ realBallTube realEmbedding R σ₀) :
    rotationWord pairs angles x ∈ realBallTube realEmbedding R σ := by
  simp only [realBallTube, mem_iUnion] at hx ⊢
  obtain ⟨v, hv, hxv⟩ := hx
  refine ⟨v, hv, ?_⟩
  have hvR : ‖v‖ ≤ R := mem_closedBall_zero_iff.mp hv
  have hxv' : ‖x-realEmbedding v‖ < σ₀ := by simpa only [mem_ball_iff_norm] using hxv
  have hxnorm : ‖x‖ < R+σ₀ := by
    have ht := norm_le_norm_sub_add x (realEmbedding v)
    rw [realEmbedding.norm_map] at ht
    linarith
  have hfactor : 0 ≤ Real.exp (2*η)-1 := by
    have := Real.one_le_exp_iff.mpr (by linarith : 0 ≤ 2*η)
    linarith
  have hrot : ‖rotationWord pairs angles x-x‖ ≤ (Real.exp (2*η)-1)*(R+σ₀) := by
    exact (rotationWord_displacement pairs angles x).trans
      (mul_le_mul (by gcongr) hxnorm.le (norm_nonneg x) hfactor)
  have htri := norm_add_le (rotationWord pairs angles x-x) (x-realEmbedding v)
  rw [sub_add_sub_cancel] at htri
  rw [mem_ball_iff_norm]
  linarith

/-- Analytic rotation words may act on a vector that itself depends analytically
on the same parameters. -/
theorem indexedWord_analytic_comp {A : Type*} [NormedAddCommGroup A] [NormedSpace ℂ A]
    {n : ℕ} (pairs : Fin n → ι × ι) (word : List (Fin n))
    {angles : A → Fin n → ℂ} {v : A → EuclideanSpace ℂ ι} {a : A}
    (ha : ∀ i, AnalyticAt ℂ (fun z => angles z i) a) (hv : AnalyticAt ℂ v a) :
    AnalyticAt ℂ (fun z => indexedWord pairs word (angles z) (v z)) a := by
  induction word with
  | nil => exact hv
  | cons i word ih =>
    exact complexPlaneRotate_analytic_comp (ha i) ih (pairs i).1 (pairs i).2

/-- The actual complexified point inserted into the local target. -/
def rotationSectionPoint {n : ℕ} (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (u : Hyperplane θ) (a : ℝ)
    (p : (Fin n → ℂ) × ℂ) : EuclideanSpace ℂ ι :=
  rotationWord pairs p.1
    (complexSectionPoint (realEmbedding θ.val) (realEmbedding u.val) a p.2)

/-- Domain safety is derived from the actual scalar square-root estimate and
actual rotation-word displacement, not supplied as a maps-to oracle. -/
theorem rotationSectionPoint_mem_targetTube {n : ℕ} (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (u : Hyperplane θ) (hu : ‖u‖ ≤ 1)
    {L δ Q R₀ R σ₀ σ η a : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hgap : L+δ ≤ Q) (hQR : Q < R₀)
    (ha : a ∈ Icc R₀ R) (hη : 0 ≤ η)
    (hscalar : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain n L δ η) :
    rotationSectionPoint pairs θ u a p ∈ realBallTube realEmbedding R σ := by
  have hsection : complexSectionPoint (realEmbedding θ.val) (realEmbedding u.val) a p.2 ∈
      realBallTube realEmbedding R σ₀ := by
    exact complexSectionPoint_mem_targetTube (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
      (realEmbedding (ι := ι)) θ u hu hL hδ hgap hQR ha hscalar hp.2
  exact rotationWord_mem_realBallTube pairs hη hp.1.le hrotation hsection

/-- A fixed complex rotation word acts analytically on its vector input. -/
theorem rotationWord_analytic_vector {n : ℕ} (pairs : Fin n → ι × ι)
    (angles : Fin n → ℂ) :
    AnalyticOnNhd ℂ (rotationWord pairs angles) univ := by
  intro x _
  exact indexedWord_analytic_comp pairs (List.ofFn (fun i : Fin n => i))
    (fun _ => analyticAt_const) analyticAt_id

/-- Joint analyticity of the actual rotated section point. -/
theorem rotationSectionPoint_analytic {n : ℕ} (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (u : Hyperplane θ)
    {a L δ Q R₀ R η : ℝ} (hgap : L+δ ≤ Q) (hQR : Q < R₀) (ha : a ∈ Icc R₀ R) :
    AnalyticOnNhd ℂ (rotationSectionPoint pairs θ u a) (rotationSectionDomain n L δ η) := by
  intro p hp
  have hs : p.2 ∈ ball (0:ℂ) a := mem_ball_zero_iff.mpr
    ((scalarTube_norm_lt hgap hp.2).trans_le (hQR.le.trans ha.1))
  have hv := ((differentiableOn_complexSectionPoint
    (realEmbedding θ.val) (realEmbedding u.val) a).analyticOnNhd isOpen_ball p.2 hs).comp
      ((ContinuousLinearMap.snd ℂ (Fin n → ℂ) ℂ).analyticAt p)
  apply indexedWord_analytic_comp pairs (List.ofFn (fun i : Fin n => i)) _ hv
  intro i
  exact ((ContinuousLinearMap.proj (R := ℂ) (φ := fun _ : Fin n => ℂ) i).analyticAt p.1).comp
    ((ContinuousLinearMap.fst ℂ (Fin n → ℂ) ℂ).analyticAt p)

/-- Fixed-angle pullback of the local target. -/
def rotationTarget {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ)
    (pairs : Fin n → ι × ι) (angles : Fin n → ℂ) (x : EuclideanSpace ℂ ι) : ℂ :=
  F (rotationWord pairs angles x)

theorem rotationTarget_analytic {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) {angles : Fin n → ℂ} {η R σ₀ σ : ℝ}
    (hη : 0 ≤ η) (hangles : (∑ i, ‖angles i‖) ≤ η)
    (hmargin : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ)) :
    AnalyticOnNhd ℂ (rotationTarget F pairs angles) (realBallTube realEmbedding R σ₀) := by
  intro x hx
  exact (hF _ (rotationWord_mem_realBallTube pairs hη hangles hmargin hx)).comp
    (rotationWord_analytic_vector pairs angles x (mem_univ _))

theorem rotationTarget_norm_le {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) {angles : Fin n → ℂ} {η R σ₀ σ B : ℝ}
    (hη : 0 ≤ η) (hangles : (∑ i, ‖angles i‖) ≤ η)
    (hmargin : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B) :
    ∀ x ∈ realBallTube realEmbedding R σ₀, ‖rotationTarget F pairs angles x‖ ≤ B := by
  intro x hx
  exact hFBound _ (rotationWord_mem_realBallTube pairs hη hangles hmargin hx)

/-- The actual finite radius/transverse-ball integral under a complex rotation
word. Both angles and scalar position are live parameters. -/
def rotatedCrossSection {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ)
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1)
    (w : ℝ → ℝ) (R₀ R : ℝ) (p : (Fin n → ℂ) × ℂ) : ℂ :=
  complexCrossSection (rotationTarget F pairs p.1) realEmbedding θ w R₀ R p.2

/-- Actual integrated rotation families are jointly complex smooth on their
explicit domain. No global ambient angular extension is postulated. -/
theorem rotatedCrossSection_smooth {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {L δ Q R₀ R σ₀ σ η B : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hgap : L+δ ≤ Q) (hQR : Q < R₀) (hη : 0 ≤ η) (hB : 0 ≤ B)
    (hscalar : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hw : Continuous w) (hwi : IntegrableOn w (Icc R₀ R)) :
    ContDiffOn ℂ ∞ (rotatedCrossSection F pairs θ w R₀ R) (rotationSectionDomain n L δ η) := by
  unfold rotatedCrossSection complexCrossSection
  apply ParameterIntegral.smoothOn_integral_of_analytic_fibers (isOpen_rotationSectionDomain n L δ η)
    (M := fun v : ℝ × Hyperplane θ => |w v.1| * Real.sqrt (R^2+Q^2)^(Module.finrank ℝ (EuclideanSpace ℝ ι)-1)*B)
  · intro p hp
    have htarget := rotationTarget_analytic pairs hη hp.1.le hrotation hF
    have hm := complexCrossSection_integrand_measurable
      (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
      (F := rotationTarget F pairs p.1) (S := L) (δ := δ) (Q := Q) (R₀ := R₀) (R := R) (σ := σ₀)
      (realEmbedding (ι := ι)) θ hw hL hδ hgap hQR hscalar htarget.differentiableOn hp.2
    exact hm
  · filter_upwards [ae_restrict_mem (measurableSet_Icc.prod measurableSet_closedBall)] with v hv
    intro p hp
    have hs : p.2 ∈ ball (0:ℂ) v.1 := mem_ball_zero_iff.mpr
      ((scalarTube_norm_lt hgap hp.2).trans_le (hQR.le.trans hv.1.1))
    have hroot := ((differentiableOn_section_sqrt v.1).analyticOnNhd isOpen_ball p.2 hs).comp
      ((ContinuousLinearMap.snd ℂ (Fin n → ℂ) ℂ).analyticAt p)
    have harg := rotationSectionPoint_analytic pairs θ v.2 hgap hQR hv.1 p hp
    have hmaps := rotationSectionPoint_mem_targetTube pairs θ v.2
      (mem_closedBall_zero_iff.mp hv.2) hL hδ hgap hQR hv.1 hη hscalar hrotation hp
    exact (analyticAt_const.mul (hroot.pow _)).mul ((hF _ hmaps).comp harg)
  · filter_upwards [ae_restrict_mem (measurableSet_Icc.prod measurableSet_closedBall)] with v hv
    intro p hp
    have htarget := rotationTarget_norm_le pairs hη hp.1.le hrotation hFBound
    exact complexCrossSection_integrand_bound (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
        (F := rotationTarget F pairs p.1) (realEmbedding (ι := ι)) θ w hB hL hδ hgap hQR hscalar htarget hv hp.2
  · exact complexCrossSection_majorant_integrable (Q := Q) (B := B) θ hwi

/-- Rotation words do not change the local projection magnitude bound: the
actual complex target remains in the same certified tube. -/
theorem rotatedCrossSection_norm_le {n : ℕ} {F : EuclideanSpace ℂ ι → ℂ}
    (pairs : Fin n → ι × ι) (θ : sphere (0:EuclideanSpace ℝ ι) 1) {w : ℝ → ℝ}
    {L δ Q R₀ R σ₀ σ η B : ℝ}
    (hL : 0 ≤ L) (hδ : 0 < δ) (hgap : L+δ ≤ Q) (hQR : Q < R₀) (hη : 0 ≤ η) (hB : 0 ≤ B)
    (hscalar : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ₀)
    (hrotation : σ₀+(Real.exp (2*η)-1)*(R+σ₀) ≤ σ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hFBound : ∀ x ∈ realBallTube realEmbedding R σ, ‖F x‖ ≤ B)
    (hw : Continuous w) (hwi : IntegrableOn w (Icc R₀ R))
    (hwpos : ∀ a ∈ Icc R₀ R, 0 ≤ w a) (hnormalized : (∫ a in Icc R₀ R, w a)=1)
    {p : (Fin n → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain n L δ η) :
    ‖rotatedCrossSection F pairs θ w R₀ R p‖ ≤
      volume.real (closedBall (0:Hyperplane θ) 1)*Real.sqrt (R^2+Q^2)^(Module.finrank ℝ (EuclideanSpace ℝ ι)-1)*B := by
  have ht := rotationTarget_analytic pairs hη hp.1.le hrotation hF
  have hb := rotationTarget_norm_le pairs hη hp.1.le hrotation hFBound
  exact complexCrossSection_norm_le (V := EuclideanSpace ℝ ι) (E := EuclideanSpace ℂ ι)
    (realEmbedding (ι := ι)) θ hw hwi hwpos hnormalized hB hL hδ hgap hQR hscalar
      ht.differentiableOn hb hp.2

/-- The real isometry corresponding to the same outermost-first rotation word. -/
def realIndexedRotation {n : ℕ} (pairs : Fin n → ι × ι)
    (hdistinct : ∀ i, (pairs i).1 ≠ (pairs i).2) :
    List (Fin n) → (Fin n → ℝ) → EuclideanSpace ℝ ι ≃ₗᵢ[ℝ] EuclideanSpace ℝ ι
  | [], _ => LinearIsometryEquiv.refl ℝ _
  | i::word, angles => (realIndexedRotation pairs hdistinct word angles).trans
      (QIRadon.Gaussian.planeRotation (pairs i).1 (pairs i).2 (hdistinct i) (angles i))

/-- Actual complex and real words agree under the coordinatewise real embedding. -/
theorem indexedWord_ofReal {n : ℕ} (pairs : Fin n → ι × ι)
    (hdistinct : ∀ i, (pairs i).1 ≠ (pairs i).2) (word : List (Fin n))
    (angles : Fin n → ℝ) (x : EuclideanSpace ℝ ι) :
    indexedWord pairs word (fun i => (angles i:ℂ)) (realEmbedding x) =
      realEmbedding (realIndexedRotation pairs hdistinct word angles x) := by
  induction word with
  | nil => rfl
  | cons i word ih =>
    rw [indexedWord, ih, complexPlaneRotate_real _ _ (hdistinct i)]
    rfl

def realRotationWord {n : ℕ} (pairs : Fin n → ι × ι)
    (hdistinct : ∀ i, (pairs i).1 ≠ (pairs i).2) (angles : Fin n → ℝ) :
    EuclideanSpace ℝ ι ≃ₗᵢ[ℝ] EuclideanSpace ℝ ι :=
  realIndexedRotation pairs hdistinct (List.ofFn (fun i : Fin n => i)) angles

theorem rotationWord_ofReal {n : ℕ} (pairs : Fin n → ι × ι)
    (hdistinct : ∀ i, (pairs i).1 ≠ (pairs i).2) (angles : Fin n → ℝ)
    (x : EuclideanSpace ℝ ι) :
    rotationWord pairs (fun i => (angles i:ℂ)) (realEmbedding x) =
      realEmbedding (realRotationWord pairs hdistinct angles x) :=
  indexedWord_ofReal pairs hdistinct _ angles x

/-- A real isometry acts on the actual unit sphere. -/
def isometryDirection {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (U : V ≃ₗᵢ[ℝ] V) (θ : sphere (0:V) 1) : sphere (0:V) 1 :=
  ⟨U θ.val, mem_sphere_zero_iff_norm.mpr (by rw [U.norm_map]; exact mem_sphere_zero_iff_norm.mp θ.property)⟩

/-- Canonical filtered profiles intertwine with actual real rotations, derived
from the Fourier-slice and isometry change-of-variables theorems. -/
theorem filteredRadonProfile_comp_isometry {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V]
    {f : V → ℂ} (hf : Integrable f) (U : V ≃ₗᵢ[ℝ] V) (θ : sphere (0:V) 1) (s : ℝ) :
    filteredRadonProfile (fun x => f (U x)) θ s =
      filteredRadonProfile f (isometryDirection U θ) s := by
  have hfc : Integrable (fun x => f (U x)) :=
    (U.measurePreserving.integrable_comp_emb U.toHomeomorph.measurableEmbedding).mpr hf
  unfold filteredRadonProfile
  congr 1
  apply integral_congr_ae
  filter_upwards with ξ
  rw [fourier_slice hfc θ ξ, fourier_slice hf (isometryDirection U θ) ξ]
  have hFourier := Real.fourier_comp_linearIsometry U f (ξ • θ.val)
  rw [show (fun x => f (U x))=f ∘ U from rfl, hFourier]
  simp only [map_smul, isometryDirection]

/-- At real angle parameters, the actual radial extension is exactly the pullback
of the original extension; the cutoff is radial and therefore unchanged. -/
theorem complexRadialExtension_rotation_real {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ)
    (pairs : Fin n → ι × ι) (hdistinct : ∀ i, (pairs i).1 ≠ (pairs i).2)
    (angles : Fin n → ℝ) (w : ℝ → ℝ) (R₀ R : ℝ) (x : EuclideanSpace ℝ ι) :
    complexRadialExtension (rotationTarget F pairs (fun i => (angles i:ℂ))) realEmbedding w R₀ R x =
      complexRadialExtension F realEmbedding w R₀ R (realRotationWord pairs hdistinct angles x) := by
  unfold complexRadialExtension rotationTarget
  rw [rotationWord_ofReal pairs hdistinct angles, (realRotationWord pairs hdistinct angles).norm_map]

/-- Total real plane isometry; diagonal generator pairs are exactly identity,
matching the complex word used in Casimir sums. -/
def totalRealPlaneRotation (i j : ι) (t : ℝ) :
    EuclideanSpace ℝ ι ≃ₗᵢ[ℝ] EuclideanSpace ℝ ι :=
  if h : i=j then LinearIsometryEquiv.refl ℝ _ else QIRadon.Gaussian.planeRotation i j h t

theorem complexPlaneRotate_ofReal_total (i j : ι) (t : ℝ) (x : EuclideanSpace ℝ ι) :
    complexPlaneRotate i j (t:ℂ) (realEmbedding x) = realEmbedding (totalRealPlaneRotation i j t x) := by
  by_cases h : i=j
  · simp [totalRealPlaneRotation, complexPlaneRotate, h]
  · rw [totalRealPlaneRotation, dif_neg h]
    exact complexPlaneRotate_real i j h t x

def totalRealIndexedRotation {n : ℕ} (pairs : Fin n → ι × ι) :
    List (Fin n) → (Fin n → ℝ) → EuclideanSpace ℝ ι ≃ₗᵢ[ℝ] EuclideanSpace ℝ ι
  | [], _ => LinearIsometryEquiv.refl ℝ _
  | i::word, angles => (totalRealIndexedRotation pairs word angles).trans
      (totalRealPlaneRotation (pairs i).1 (pairs i).2 (angles i))

theorem indexedWord_ofReal_total {n : ℕ} (pairs : Fin n → ι × ι) (word : List (Fin n))
    (angles : Fin n → ℝ) (x : EuclideanSpace ℝ ι) :
    indexedWord pairs word (fun i => (angles i:ℂ)) (realEmbedding x) =
      realEmbedding (totalRealIndexedRotation pairs word angles x) := by
  induction word with
  | nil => rfl
  | cons i word ih =>
    rw [indexedWord, ih, complexPlaneRotate_ofReal_total]
    rfl

def totalRealRotationWord {n : ℕ} (pairs : Fin n → ι × ι) (angles : Fin n → ℝ) :
    EuclideanSpace ℝ ι ≃ₗᵢ[ℝ] EuclideanSpace ℝ ι :=
  totalRealIndexedRotation pairs (List.ofFn (fun i : Fin n => i)) angles

theorem rotationWord_ofReal_total {n : ℕ} (pairs : Fin n → ι × ι) (angles : Fin n → ℝ)
    (x : EuclideanSpace ℝ ι) :
    rotationWord pairs (fun i => (angles i:ℂ)) (realEmbedding x) =
      realEmbedding (totalRealRotationWord pairs angles x) :=
  indexedWord_ofReal_total pairs _ angles x

theorem complexRadialExtension_rotation_real_total {n : ℕ} (F : EuclideanSpace ℂ ι → ℂ)
    (pairs : Fin n → ι × ι) (angles : Fin n → ℝ) (w : ℝ → ℝ) (R₀ R : ℝ) (x : EuclideanSpace ℝ ι) :
    complexRadialExtension (rotationTarget F pairs (fun i => (angles i:ℂ))) realEmbedding w R₀ R x =
      complexRadialExtension F realEmbedding w R₀ R (totalRealRotationWord pairs angles x) := by
  unfold complexRadialExtension rotationTarget
  rw [rotationWord_ofReal_total, (totalRealRotationWord pairs angles).norm_map]

end QIRadon.Radon
