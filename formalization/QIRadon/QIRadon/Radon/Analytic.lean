import QIRadon.Radon.CrossSection
import Mathlib

/-! Analytic continuation infrastructure for the explicit finite cross sections.
Analyticity of an integrated profile is derived from its actual integrand, not
introduced as an oracle assumption. -/

noncomputable section
open MeasureTheory Set Metric Filter
open scoped Topology

namespace QIRadon.Radon

/-- Measurability of parameter derivatives from measurable evaluations and actual
pointwise complex differentiability. Only a local disk of evaluations is used. -/
theorem derivative_parameter_aestronglyMeasurable
    {A : Type*} [MeasurableSpace A] {μ : Measure A}
    {F : ℂ → A → ℂ} {z₀ : ℂ} {δ : ℝ} (hδ : 0 < δ)
    (hMeas : ∀ z ∈ ball z₀ δ, AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableAt ℂ (fun z => F z a) z₀) :
    AEStronglyMeasurable (fun a => deriv (fun z => F z a) z₀) μ := by
  let t : ℕ → ℝ := fun n => (δ / 2) / ((n : ℝ) + 1)
  have htpos : ∀ n, 0 < t n := by intro n; dsimp [t]; positivity
  have htlt : ∀ n, t n < δ := by
    intro n
    have hle : t n ≤ δ / 2 := by
      dsimp [t]
      apply (div_le_iff₀ (by positivity : (0 : ℝ) < (n : ℝ)+1)).mpr
      nlinarith [Nat.cast_nonneg (α := ℝ) n]
    linarith
  have htlim : Tendsto t atTop (nhds (0 : ℝ)) := by
    have hl := (tendsto_const_nhds (x := δ/2)).mul
      (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ))
    simpa only [mul_zero, div_eq_mul_inv, one_mul, t] using hl
  have hctlim : Tendsto (fun n => (t n : ℂ)) atTop (nhdsWithin 0 {0}ᶜ) := by
    apply tendsto_nhdsWithin_iff.mpr
    refine ⟨?_, ?_⟩
    · simpa only [Complex.ofReal_zero, Function.comp_def] using Complex.continuous_ofReal.continuousAt.tendsto.comp htlim
    · filter_upwards with n
      exact Complex.ofReal_ne_zero.mpr (htpos n).ne'
  have hnear : ∀ n, z₀ + (t n : ℂ) ∈ ball z₀ δ := by
    intro n
    rw [mem_ball_iff_norm, add_sub_cancel_left]
    simpa only [Complex.norm_real, Real.norm_eq_abs, abs_of_pos (htpos n)] using htlt n
  let Q : ℕ → A → ℂ := fun n a => (t n : ℂ)⁻¹ • (F (z₀ + t n) a - F z₀ a)
  have hQ : ∀ n, AEStronglyMeasurable (Q n) μ := by
    intro n
    simpa only [Q, Pi.smul_def, Pi.sub_def] using
      ((hMeas _ (hnear n)).sub (hMeas z₀ (mem_ball_self hδ))).const_smul ((t n : ℂ)⁻¹)
  apply aestronglyMeasurable_of_tendsto_ae atTop hQ
  filter_upwards [hDiff] with a ha
  exact ha.hasDerivAt.tendsto_slope_zero.comp hctlim

/-- Holomorphic integrands uniformly dominated on a disk have a holomorphic
integral at its center. The derivative majorant is derived by Cauchy's estimate. -/
theorem differentiableAt_integral_of_disk_bound
    {A : Type*} [MeasurableSpace A] {μ : Measure A}
    {F : ℂ → A → ℂ} {z₀ : ℂ} {δ : ℝ} {M : A → ℝ} (hδ : 0 < δ)
    (hMeas : ∀ z ∈ ball z₀ (2*δ), AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableOn ℂ (fun z => F z a) (ball z₀ (2*δ)))
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ ball z₀ (2*δ), ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) :
    DifferentiableAt ℂ (fun z => ∫ a, F z a ∂μ) z₀ := by
  have h2δ : 0 < 2*δ := by positivity
  have hcenter : z₀ ∈ ball z₀ (2*δ) := mem_ball_self h2δ
  have hFint : Integrable (F z₀) μ :=
    hM.mono' (hMeas z₀ hcenter) (hBound.mono (fun a ha => ha z₀ hcenter))
  have hF' : AEStronglyMeasurable (fun a => deriv (fun z => F z a) z₀) μ :=
    derivative_parameter_aestronglyMeasurable h2δ hMeas
      (hDiff.mono (fun a ha => ha.differentiableAt (isOpen_ball.mem_nhds hcenter)))
  have hderivBound : ∀ᵐ a ∂μ, ∀ z ∈ ball z₀ δ,
      ‖deriv (fun w => F w a) z‖ ≤ M a / δ := by
    filter_upwards [hDiff, hBound] with a ha hba
    intro z hz
    have hsub : closedBall z δ ⊆ ball z₀ (2*δ) := by
      intro u hu
      have hu' : dist u z ≤ δ := hu
      have hz' : dist z z₀ < δ := hz
      have htri := dist_triangle u z z₀
      change dist u z₀ < 2*δ
      linarith
    apply Complex.norm_deriv_le_of_forall_mem_sphere_norm_le hδ
    · apply DifferentiableOn.diffContOnCl
      rw [closure_ball z hδ.ne']
      exact ha.mono hsub
    · intro u hu
      exact hba u (hsub hu.le)
  have hlocMeas : ∀ᶠ z in nhds z₀, AEStronglyMeasurable (F z) μ := by
    filter_upwards [ball_mem_nhds z₀ h2δ] with z hz
    exact hMeas z hz
  have hlocDiff : ∀ᵐ a ∂μ, ∀ z ∈ ball z₀ δ,
      HasDerivAt (fun w => F w a) (deriv (fun w => F w a) z) z := by
    filter_upwards [hDiff] with a ha
    intro z hz
    have hzbig : z ∈ ball z₀ (2*δ) := by
      have hz' : dist z z₀ < δ := hz
      change dist z z₀ < 2*δ
      linarith
    exact (ha.differentiableAt (isOpen_ball.mem_nhds hzbig)).hasDerivAt
  exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (ball_mem_nhds z₀ hδ) hlocMeas hFint hF'
    hderivBound (hM.div_const δ) hlocDiff).2.differentiableAt

/-- A parameter integral is analytic on an open complex set when its actual
integrands are holomorphic and share an integrable magnitude bound there. -/
theorem analyticOnNhd_integral_of_dominated
    {A : Type*} [MeasurableSpace A] {μ : Measure A}
    {F : ℂ → A → ℂ} {U : Set ℂ} {M : A → ℝ} (hU : IsOpen U)
    (hMeas : ∀ z ∈ U, AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableOn ℂ (fun z => F z a) U)
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ U, ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) :
    AnalyticOnNhd ℂ (fun z => ∫ a, F z a ∂μ) U := by
  apply DifferentiableOn.analyticOnNhd _ hU
  intro z hz
  obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp (hU.mem_nhds hz)
  have hδ : 0 < ε / 3 := by positivity
  have hsub : ball z (2*(ε/3)) ⊆ U := by
    intro w hw
    apply hball
    have hd : dist w z < 2*(ε/3) := hw
    change dist w z < ε
    linarith
  apply DifferentiableAt.differentiableWithinAt
  exact differentiableAt_integral_of_disk_bound hδ
    (fun w hw => hMeas w (hsub hw))
    (hDiff.mono (fun a ha => ha.mono hsub))
    (hBound.mono (fun a ha w hw => ha w (hsub hw))) hM

/-- The same domination provides the uniform bound used by Cauchy estimates. -/
theorem norm_parameter_integral_le {A : Type*} [MeasurableSpace A] {μ : Measure A}
    {F : ℂ → A → ℂ} {M : A → ℝ} (hM : Integrable M μ) {z : ℂ}
    (hbound : ∀ᵐ a ∂μ, ‖F z a‖ ≤ M a) :
    ‖∫ a, F z a ∂μ‖ ≤ ∫ a, M a ∂μ :=
  norm_integral_le_of_norm_le hM hbound

/-- Complex square roots used by the finite cross section avoid their branch cut
on every disk strictly smaller than the radius a. -/
theorem section_radicand_re_pos {a : ℝ} {z : ℂ} (ha : ‖z‖ < a) :
    0 < ((a : ℂ)^2-z^2).re := by
  have ha0 : 0 < a := lt_of_le_of_lt (norm_nonneg z) ha
  have hnorm := Complex.sq_norm z
  rw [Complex.normSq_apply] at hnorm
  simp only [Complex.sub_re, pow_two, Complex.mul_re, Complex.ofReal_re,
    Complex.ofReal_im, mul_zero, sub_zero]
  nlinarith [norm_nonneg z, sq_nonneg z.im]

/-- The finite-cross-section square root is genuinely holomorphic on |z|<a. -/
theorem differentiableOn_section_sqrt (a : ℝ) :
    DifferentiableOn ℂ (fun z : ℂ => Complex.sqrt ((a : ℂ)^2-z^2)) (ball 0 a) := by
  intro z hz
  have hr : ((a : ℂ)^2-z^2) ∈ Complex.slitPlane :=
    Complex.mem_slitPlane_iff.mpr (Or.inl (section_radicand_re_pos (by simpa using hz)))
  exact ((Complex.differentiableAt_sqrt hr).comp z
    (by fun_prop : DifferentiableAt ℂ (fun z : ℂ => (a : ℂ)^2-z^2) z)).differentiableWithinAt

/-- The complex cross-section argument itself is holomorphic; target holomorphy
can therefore be composed with this explicit map. -/
def complexSectionPoint {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
    (θ u : E) (a : ℝ) (z : ℂ) : E :=
  z • θ + Complex.sqrt ((a : ℂ)^2-z^2) • u

theorem differentiableOn_complexSectionPoint {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℂ E] (θ u : E) (a : ℝ) :
    DifferentiableOn ℂ (complexSectionPoint θ u a) (ball 0 a) := by
  exact (differentiableOn_id.smul_const θ).add
    ((differentiableOn_section_sqrt a).smul_const u)

/-- Derivative of the exact square-root factor occurring in the cross section. -/
theorem hasDerivAt_section_sqrt {a : ℝ} {z : ℂ} (ha : ‖z‖ < a) :
    HasDerivAt (fun w : ℂ => Complex.sqrt ((a : ℂ)^2-w^2))
      (-z * ((a : ℂ)^2-z^2)^(-1/2 : ℂ)) z := by
  have hr : ((a : ℂ)^2-z^2) ∈ Complex.slitPlane :=
    Complex.mem_slitPlane_iff.mpr (Or.inl (section_radicand_re_pos ha))
  have hrad : HasDerivAt (fun w : ℂ => (a : ℂ)^2-w^2) (-2*z) z := by
    simpa using ((hasDerivAt_id z).fun_pow 2).const_sub ((a : ℂ)^2)
  have hcomp := (Complex.hasDerivAt_sqrt hr).comp z hrad
  have hcomp' : HasDerivAt (Complex.sqrt ∘ (fun w : ℂ => (a : ℂ)^2-w^2))
      (-z * ((a : ℂ)^2-z^2)^(-1/2 : ℂ)) z := hcomp.congr_deriv (by ring)
  simpa only [Function.comp_def] using hcomp' 

/-- A uniform derivative bound on the actual section square root. The constant
is the one used by the approved scalar complex-neighborhood argument. -/
theorem section_sqrt_derivative_norm_le {R₀ Q a : ℝ} (hQ : 0 ≤ Q)
    (hQR : Q < R₀) (ha : R₀ ≤ a) {z : ℂ} (hz : ‖z‖ ≤ Q) :
    ‖-z * ((a : ℂ)^2-z^2)^(-1/2 : ℂ)‖ ≤ Q / Real.sqrt (R₀^2-Q^2) := by
  have hR : 0 < R₀ := lt_of_le_of_lt hQ hQR
  have hdelta : 0 < R₀^2-Q^2 := by nlinarith
  have hn : ‖z‖^2 = z.re*z.re+z.im*z.im := by
    rw [Complex.sq_norm, Complex.normSq_apply]
  have hre := Complex.re_le_norm ((a : ℂ)^2-z^2)
  simp only [Complex.sub_re, pow_two, Complex.mul_re, Complex.ofReal_re,
    Complex.ofReal_im, mul_zero, sub_zero] at hre
  simp only [← pow_two] at hre
  have hlow : R₀^2-Q^2 ≤ ‖(a : ℂ)^2-z^2‖ := by
    nlinarith [norm_nonneg z, sq_nonneg z.im]
  have hp := Real.rpow_le_rpow_of_nonpos hdelta hlow (by norm_num : -(1/2 : ℝ) ≤ 0)
  have hpow : ‖((a : ℂ)^2-z^2)^(-1/2 : ℂ)‖ =
      ‖(a : ℂ)^2-z^2‖ ^ (-(1/2 : ℝ)) := by
    convert Complex.norm_cpow_real ((a : ℂ)^2-z^2) (-(1/2 : ℝ)) using 1; norm_num
  rw [norm_mul, norm_neg, hpow]
  calc
    ‖z‖ * ‖(a : ℂ)^2-z^2‖ ^ (-(1/2 : ℝ)) ≤
        Q * (R₀^2-Q^2)^(-(1/2 : ℝ)) :=
      mul_le_mul hz hp (by positivity) hQ
    _ = Q / Real.sqrt (R₀^2-Q^2) := by
      rw [Real.rpow_neg hdelta.le, ← Real.sqrt_eq_rpow, div_eq_mul_inv]

/-- Lipschitz control of the complex square-root cross section on a safe disk. -/
theorem section_sqrt_lipschitz {R₀ Q a : ℝ} (hQ : 0 ≤ Q)
    (hQR : Q < R₀) (ha : R₀ ≤ a) {z w : ℂ} (hz : ‖z‖ ≤ Q) (hw : ‖w‖ ≤ Q) :
    ‖Complex.sqrt ((a : ℂ)^2-z^2) - Complex.sqrt ((a : ℂ)^2-w^2)‖ ≤
      (Q / Real.sqrt (R₀^2-Q^2)) * ‖z-w‖ := by
  apply Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := fun u : ℂ => Complex.sqrt ((a : ℂ)^2-u^2))
    (x := w) (y := z) (C := Q / Real.sqrt (R₀^2-Q^2))
    (f' := fun u : ℂ => -u * ((a : ℂ)^2-u^2)^(-1/2 : ℂ))
    (s := closedBall (0 : ℂ) Q)
  · intro u hu
    have hun : ‖u‖ ≤ Q := mem_closedBall_zero_iff.mp hu
    exact (hasDerivAt_section_sqrt (hun.trans_lt (hQR.trans_le ha))).hasDerivWithinAt
  · intro u hu
    exact section_sqrt_derivative_norm_le hQ hQR ha (mem_closedBall_zero_iff.mp hu)
  · exact convex_closedBall (0 : ℂ) Q
  · exact mem_closedBall_zero_iff.mpr hw
  · exact mem_closedBall_zero_iff.mpr hz

/-- Explicit complex tube around the real scalar interval [-S,S]. -/
def scalarTube (S δ : ℝ) : Set ℂ :=
  ⋃ s ∈ Set.Icc (-S) S, ball (s : ℂ) δ

/-- Explicit complex neighborhood of the embedded real outer ball. -/
def realBallTube {V E : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (ι : V →ₗᵢ[ℝ] E) (R σ : ℝ) : Set E :=
  ⋃ x ∈ closedBall (0 : V) R, ball (ι x) σ

theorem isOpen_scalarTube (S δ : ℝ) : IsOpen (scalarTube S δ) := by
  unfold scalarTube
  exact isOpen_iUnion (fun _ => isOpen_iUnion (fun _ => isOpen_ball))

theorem isOpen_realBallTube {V E : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (ι : V →ₗᵢ[ℝ] E) (R σ : ℝ) : IsOpen (realBallTube ι R σ) := by
  unfold realBallTube
  exact isOpen_iUnion (fun _ => isOpen_iUnion (fun _ => isOpen_ball))

/-- A scalar tube stays in a safe complex disk under the explicit gap condition. -/
theorem scalarTube_norm_lt {S δ Q : ℝ} (hgap : S + δ ≤ Q)
    {z : ℂ} (hz : z ∈ scalarTube S δ) : ‖z‖ < Q := by
  simp only [scalarTube, Set.mem_iUnion] at hz
  obtain ⟨s, hs, hzs⟩ := hz
  have habs : |s| ≤ S := abs_le.mpr hs
  have hd : ‖z-(s : ℂ)‖ < δ := by simpa only [mem_ball_iff_norm] using hzs
  have hn := norm_le_norm_sub_add z (s : ℂ)
  simp only [Complex.norm_real, Real.norm_eq_abs] at hn
  linarith

/-- The explicit complex section point moves by at most L times the movement in
its scalar coordinate, with the approved L=1+Q/sqrt(R0²-Q²). -/
theorem complexSectionPoint_displacement {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℂ E] {θ u : E}
    (hθ : ‖θ‖ ≤ 1) (hu : ‖u‖ ≤ 1)
    {R₀ Q a : ℝ} (hQ : 0 ≤ Q) (hQR : Q < R₀) (ha : R₀ ≤ a)
    {z w : ℂ} (hz : ‖z‖ ≤ Q) (hw : ‖w‖ ≤ Q) :
    ‖complexSectionPoint θ u a z - complexSectionPoint θ u a w‖ ≤
      (1 + Q / Real.sqrt (R₀^2-Q^2)) * ‖z-w‖ := by
  have heq : complexSectionPoint θ u a z - complexSectionPoint θ u a w =
      (z-w) • θ +
        (Complex.sqrt ((a : ℂ)^2-z^2)-Complex.sqrt ((a : ℂ)^2-w^2)) • u := by
    simp only [complexSectionPoint, sub_smul]
    abel
  rw [heq]
  calc
    _ ≤ ‖(z-w) • θ‖ +
        ‖(Complex.sqrt ((a : ℂ)^2-z^2)-Complex.sqrt ((a : ℂ)^2-w^2)) • u‖ :=
      norm_add_le _ _
    _ ≤ ‖z-w‖ +
        ‖Complex.sqrt ((a : ℂ)^2-z^2)-Complex.sqrt ((a : ℂ)^2-w^2)‖ := by
      simp only [norm_smul]
      apply add_le_add
      · exact (mul_le_mul_of_nonneg_left hθ (norm_nonneg _)).trans_eq (mul_one _)
      · exact (mul_le_mul_of_nonneg_left hu (norm_nonneg _)).trans_eq (mul_one _)
    _ ≤ ‖z-w‖ + (Q / Real.sqrt (R₀^2-Q^2)) * ‖z-w‖ :=
      add_le_add le_rfl (section_sqrt_lipschitz hQ hQR ha hz hw)
    _ = _ := by ring

/-- On real scalar coordinates the complex square-root branch is the ordinary
nonnegative real cross-section radius. -/
theorem section_sqrt_ofReal (a s : ℝ) (hsq : 0 ≤ a^2-s^2) :
    Complex.sqrt ((a : ℂ)^2-(s : ℂ)^2) = (Real.sqrt (a^2-s^2) : ℂ) := by
  have heq : (a : ℂ)^2-(s : ℂ)^2 = ((a^2-s^2 : ℝ) : ℂ) := by push_cast; rfl
  rw [heq, Complex.sqrt, Real.sqrt_eq_rpow]
  convert (Complex.ofReal_cpow hsq (1/2 : ℝ)).symm using 1; norm_num

/-- Complex cross-section arguments restrict to the actual embedded real points. -/
theorem complexSectionPoint_ofReal {V E : Type*}
    [NormedAddCommGroup V] [NormedSpace ℝ V]
    [NormedAddCommGroup E] [NormedSpace ℂ E] [NormedSpace ℝ E] [IsScalarTower ℝ ℂ E]
    (ι : V →ₗᵢ[ℝ] E) (θ u : V) (a s : ℝ) (hsq : 0 ≤ a^2-s^2) :
    complexSectionPoint (ι θ) (ι u) a (s : ℂ) =
      ι (s • θ + Real.sqrt (a^2-s^2) • u) := by
  simp only [complexSectionPoint, section_sqrt_ofReal a s hsq, map_add, map_smul]
  rw [← IsScalarTower.algebraMap_smul ℂ s (ι θ),
    ← IsScalarTower.algebraMap_smul ℂ (Real.sqrt (a^2-s^2)) (ι u)]
  rfl

/-- Every real unit-ball cross-section point stays inside the radius-a ball. -/
theorem real_sectionPoint_norm_le {V : Type*}
    [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (θ : sphere (0 : V) 1) (u : Hyperplane θ) (hu : ‖u‖ ≤ 1)
    {a s : ℝ} (ha : |s| < a) :
    ‖s • θ.val + Real.sqrt (a^2-s^2) • u.val‖ ≤ a := by
  have ha0 : 0 < a := lt_of_le_of_lt (abs_nonneg s) ha
  have hsq : 0 ≤ a^2-s^2 := by nlinarith [abs_nonneg s, sq_abs s]
  have hr := Real.sqrt_nonneg (a^2-s^2)
  have heq := norm_hyperplane_point_sq θ s (Real.sqrt (a^2-s^2) • u)
  have hyn : ‖Real.sqrt (a^2-s^2) • u‖ ≤ Real.sqrt (a^2-s^2) := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hr]
    exact (mul_le_mul_of_nonneg_left hu hr).trans_eq (mul_one _)
  have hrsq := Real.sq_sqrt hsq
  change ‖s • θ.val + Real.sqrt (a^2-s^2) • u.val‖^2 =
    s^2 + ‖Real.sqrt (a^2-s^2) • u‖^2 at heq
  nlinarith [norm_nonneg (Real.sqrt (a^2-s^2) • u),
    norm_nonneg (s • θ.val + Real.sqrt (a^2-s^2) • u.val)]

/-- Quantitative domain safety for the actual complex cross-section map. The
hypotheses are explicit scalar margin inequalities, not a profile-analyticity
assumption or a maps-to oracle. -/
theorem complexSectionPoint_mem_targetTube {V E : Type*}
    [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    [NormedAddCommGroup E] [NormedSpace ℂ E] [NormedSpace ℝ E] [IsScalarTower ℝ ℂ E]
    (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) (u : Hyperplane θ) (hu : ‖u‖ ≤ 1)
    {S δ Q R₀ R σ a : ℝ}
    (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (ha : a ∈ Set.Icc R₀ R)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    {z : ℂ} (hz : z ∈ scalarTube S δ) :
    complexSectionPoint (ι θ.val) (ι u.val) a z ∈ realBallTube ι R σ := by
  have hzQ : ‖z‖ < Q := scalarTube_norm_lt hgap hz
  simp only [scalarTube, Set.mem_iUnion] at hz
  obtain ⟨s, hs, hzs⟩ := hz
  have hsabs : |s| ≤ S := abs_le.mpr hs
  have hQ : 0 ≤ Q := by linarith
  have hsQ : ‖(s : ℂ)‖ ≤ Q := by
    simp only [Complex.norm_real, Real.norm_eq_abs]
    linarith
  have hsa : |s| < a := by linarith [ha.1]
  have hsq : 0 ≤ a^2-s^2 := by nlinarith [abs_nonneg s, sq_abs s]
  let x := s • θ.val + Real.sqrt (a^2-s^2) • u.val
  have hx : x ∈ closedBall (0 : V) R :=
    mem_closedBall_zero_iff.mpr ((real_sectionPoint_norm_le θ u hu hsa).trans ha.2)
  have hθ : ‖ι θ.val‖ ≤ 1 := by
    rw [ι.norm_map]
    exact (mem_sphere_zero_iff_norm.mp θ.property).le
  have hu' : ‖ι u.val‖ ≤ 1 := by simpa only [ι.norm_map, Submodule.coe_norm] using hu
  have hmove := complexSectionPoint_displacement hθ hu' hQ hQR ha.1 hzQ.le hsQ
  have hL : 0 < 1+Q/Real.sqrt (R₀^2-Q^2) := by positivity
  have hdist : ‖z-(s : ℂ)‖ < δ := by simpa only [mem_ball_iff_norm] using hzs
  have hstrict : ‖complexSectionPoint (ι θ.val) (ι u.val) a z -
      complexSectionPoint (ι θ.val) (ι u.val) a (s : ℂ)‖ < σ :=
    hmove.trans_lt ((mul_lt_mul_of_pos_left hdist hL).trans_le hmargin)
  unfold realBallTube
  apply Set.mem_iUnion.mpr
  refine ⟨x, Set.mem_iUnion.mpr ⟨hx, ?_⟩⟩
  rw [mem_ball_iff_norm]
  change ‖complexSectionPoint (ι θ.val) (ι u.val) a z - ι x‖ < σ
  rw [complexSectionPoint_ofReal ι θ.val u.val a s hsq] at hstrict
  exact hstrict

/-- Norm of the principal complex square root. -/
theorem norm_complex_sqrt (z : ℂ) : ‖Complex.sqrt z‖ = Real.sqrt ‖z‖ := by
  simpa [Complex.sqrt, Real.sqrt_eq_rpow, one_div] using Complex.norm_cpow_inv_nat z 2

/-- Uniform magnitude bound for the complex cross-section radius. -/
theorem section_sqrt_norm_le {a R Q : ℝ} (ha : 0 ≤ a) (haR : a ≤ R)
    (_hQ : 0 ≤ Q) {z : ℂ} (hz : ‖z‖ ≤ Q) :
    ‖Complex.sqrt ((a : ℂ)^2-z^2)‖ ≤ Real.sqrt (R^2+Q^2) := by
  rw [norm_complex_sqrt]
  apply Real.sqrt_le_sqrt
  calc
    ‖(a : ℂ)^2-z^2‖ ≤ ‖(a : ℂ)^2‖ + ‖z^2‖ := norm_sub_le _ _
    _ ≤ R^2+Q^2 := by
      simp only [norm_pow, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg ha]
      gcongr

section CrossSectionAnalytic

variable {V E : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]
  [NormedAddCommGroup E] [NormedSpace ℂ E] [NormedSpace ℝ E] [IsScalarTower ℝ ℂ E]

/-- The finite real parameter region is fixed while the scalar input is complex. -/
def sectionParameterSet (θ : sphere (0 : V) 1) (R₀ R : ℝ) : Set (ℝ × Hyperplane θ) :=
  Set.Icc R₀ R ×ˢ closedBall (0 : Hyperplane θ) 1

/-- Explicit holomorphic candidate obtained from the actual finite cross-section
formula. The cutoff radius is never evaluated at complex arguments. -/
def complexCrossSection (F : E → ℂ) (ι : V →ₗᵢ[ℝ] E)
    (θ : sphere (0 : V) 1) (w : ℝ → ℝ) (R₀ R : ℝ) (z : ℂ) : ℂ :=
  ∫ p : ℝ × Hyperplane θ in sectionParameterSet θ R₀ R,
    (w p.1 : ℂ) * (Complex.sqrt ((p.1 : ℂ)^2-z^2))^(Module.finrank ℝ V-1) *
      F (complexSectionPoint (ι θ.val) (ι p.2.val) p.1 z)

/-- Measurability of the finite complex cross-section integrand follows from local
target holomorphy and the proved domain-safety inequalities. -/
theorem complexCrossSection_integrand_measurable
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) {w : ℝ → ℝ}
    {S δ Q R₀ R σ : ℝ} (hw : Continuous w)
    (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hF : DifferentiableOn ℂ F (realBallTube ι R σ))
    {z : ℂ} (hz : z ∈ scalarTube S δ) :
    AEStronglyMeasurable (fun p : ℝ × Hyperplane θ =>
      (w p.1 : ℂ) * (Complex.sqrt ((p.1 : ℂ)^2-z^2))^(Module.finrank ℝ V-1) *
      F (complexSectionPoint (ι θ.val) (ι p.2.val) p.1 z))
      ((volume.prod volume).restrict (sectionParameterSet θ R₀ R)) := by
  have hroot : ContinuousOn (fun p : ℝ × Hyperplane θ =>
      Complex.sqrt ((p.1 : ℂ)^2-z^2)) (sectionParameterSet θ R₀ R) := by
    intro p hp
    have hzpa : ‖z‖ < p.1 :=
      (scalarTube_norm_lt hgap hz).trans_le (hQR.le.trans hp.1.1)
    have hslit := Complex.mem_slitPlane_iff.mpr (Or.inl (section_radicand_re_pos hzpa))
    have hpoly : Continuous (fun p : ℝ × Hyperplane θ => (p.1 : ℂ)^2-z^2) := by fun_prop
    have hc : ContinuousAt Complex.sqrt ((p.1 : ℂ)^2-z^2) :=
      (Complex.differentiableAt_sqrt hslit).continuousAt
    exact (hc.comp (f := fun q : ℝ × Hyperplane θ => (q.1 : ℂ)^2-z^2)
      (x := p) hpoly.continuousAt).continuousWithinAt
  have harg : ContinuousOn (fun p : ℝ × Hyperplane θ =>
      complexSectionPoint (ι θ.val) (ι p.2.val) p.1 z) (sectionParameterSet θ R₀ R) := by
    unfold complexSectionPoint
    exact continuousOn_const.add (hroot.smul (by fun_prop))
  have hmaps : Set.MapsTo (fun p : ℝ × Hyperplane θ =>
      complexSectionPoint (ι θ.val) (ι p.2.val) p.1 z)
      (sectionParameterSet θ R₀ R) (realBallTube ι R σ) := by
    intro p hp
    exact complexSectionPoint_mem_targetTube ι θ p.2
      (mem_closedBall_zero_iff.mp hp.2) hS hδ hgap hQR hp.1 hmargin hz
  have hcont := ((Complex.continuous_ofReal.comp (hw.comp continuous_fst)).continuousOn.mul
    (hroot.pow (Module.finrank ℝ V-1))).mul (hF.continuousOn.comp harg hmaps)
  exact hcont.aestronglyMeasurable (measurableSet_Icc.prod measurableSet_closedBall)

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
/-- The actual finite integrand is holomorphic in z at every real parameter in
the prescribed radius and unit-ball region. -/
theorem complexCrossSection_integrand_differentiable
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) (w : ℝ → ℝ)
    {S δ Q R₀ R σ : ℝ}
    (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hF : DifferentiableOn ℂ F (realBallTube ι R σ))
    {p : ℝ × Hyperplane θ} (hp : p ∈ sectionParameterSet θ R₀ R) :
    DifferentiableOn ℂ (fun z : ℂ =>
      (w p.1 : ℂ) * (Complex.sqrt ((p.1 : ℂ)^2-z^2))^(Module.finrank ℝ V-1) *
      F (complexSectionPoint (ι θ.val) (ι p.2.val) p.1 z)) (scalarTube S δ) := by
  have hsub : scalarTube S δ ⊆ ball (0 : ℂ) p.1 := by
    intro z hz
    exact mem_ball_zero_iff.mpr
      ((scalarTube_norm_lt hgap hz).trans_le (hQR.le.trans hp.1.1))
  have hroot := (differentiableOn_section_sqrt p.1).mono hsub
  have harg := (differentiableOn_complexSectionPoint (ι θ.val) (ι p.2.val) p.1).mono hsub
  have hmaps : Set.MapsTo (complexSectionPoint (ι θ.val) (ι p.2.val) p.1)
      (scalarTube S δ) (realBallTube ι R σ) := by
    intro z hz
    exact complexSectionPoint_mem_targetTube ι θ p.2 (mem_closedBall_zero_iff.mp hp.2)
      hS hδ hgap hQR hp.1 hmargin hz
  exact ((hroot.pow (Module.finrank ℝ V-1)).const_mul (w p.1 : ℂ)).mul (hF.comp harg hmaps)

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
/-- Explicit uniform magnitude bound for the actual finite cross-section integrand. -/
theorem complexCrossSection_integrand_bound
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) (w : ℝ → ℝ)
    {S δ Q R₀ R σ B : ℝ} (_hB : 0 ≤ B)
    (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B)
    {p : ℝ × Hyperplane θ} (hp : p ∈ sectionParameterSet θ R₀ R)
    {z : ℂ} (hz : z ∈ scalarTube S δ) :
    ‖(w p.1 : ℂ) * (Complex.sqrt ((p.1 : ℂ)^2-z^2))^(Module.finrank ℝ V-1) *
      F (complexSectionPoint (ι θ.val) (ι p.2.val) p.1 z)‖ ≤
      |w p.1| * (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B := by
  have hQ : 0 ≤ Q := by linarith
  have ha0 : 0 ≤ p.1 := hQ.trans (hQR.le.trans hp.1.1)
  have hroot := section_sqrt_norm_le ha0 hp.1.2 hQ (scalarTube_norm_lt hgap hz).le
  have htarget := hFBound _ (complexSectionPoint_mem_targetTube ι θ p.2
    (mem_closedBall_zero_iff.mp hp.2) hS hδ hgap hQR hp.1 hmargin hz)
  simp only [norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs]
  exact mul_le_mul
    (mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _) hroot _) (abs_nonneg _))
    htarget (norm_nonneg _) (by positivity)

/-- The displayed majorant is integrable over the actual finite parameter region. -/
theorem complexCrossSection_majorant_integrable (θ : sphere (0 : V) 1)
    {w : ℝ → ℝ} {R₀ R Q B : ℝ} (hw : IntegrableOn w (Set.Icc R₀ R)) :
    Integrable (fun p : ℝ × Hyperplane θ =>
      |w p.1| * (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B)
      ((volume.prod volume).restrict (sectionParameterSet θ R₀ R)) := by
  have hc : IntegrableOn (fun _ : Hyperplane θ =>
      (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B) (closedBall 0 1) :=
    continuousOn_const.integrableOn_compact (isCompact_closedBall (0 : Hyperplane θ) 1)
  have hprod := hw.norm.mul_prod hc
  simpa only [Measure.prod_restrict, Real.norm_eq_abs, sectionParameterSet, mul_assoc] using hprod

/-- Holomorphic continuation of the actual finite cross-section construction
from target holomorphy on the supplied complex neighborhood. -/
theorem complexCrossSection_analytic
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) {w : ℝ → ℝ}
    {S δ Q R₀ R σ B : ℝ} (hw : Continuous w) (hwi : IntegrableOn w (Set.Icc R₀ R))
    (hB : 0 ≤ B) (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hF : DifferentiableOn ℂ F (realBallTube ι R σ))
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B) :
    AnalyticOnNhd ℂ (complexCrossSection F ι θ w R₀ R) (scalarTube S δ) := by
  unfold complexCrossSection
  apply analyticOnNhd_integral_of_dominated (isOpen_scalarTube S δ)
  · intro z hz
    exact complexCrossSection_integrand_measurable ι θ hw hS hδ hgap hQR hmargin hF hz
  · filter_upwards [ae_restrict_mem (measurableSet_Icc.prod measurableSet_closedBall)] with p hp
    exact complexCrossSection_integrand_differentiable ι θ w hS hδ hgap hQR hmargin hF hp
  · filter_upwards [ae_restrict_mem (measurableSet_Icc.prod measurableSet_closedBall)] with p hp
    intro z hz
    exact complexCrossSection_integrand_bound ι θ w hB hS hδ hgap hQR hmargin hFBound hp hz
  · exact complexCrossSection_majorant_integrable θ hwi

/-- Integrability of the actual complex cross-section integrand at every point
in the certified scalar tube. -/
theorem complexCrossSection_integrand_integrable
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) {w : ℝ → ℝ}
    {S δ Q R₀ R σ B : ℝ} (hw : Continuous w) (hwi : IntegrableOn w (Set.Icc R₀ R))
    (hB : 0 ≤ B) (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hF : DifferentiableOn ℂ F (realBallTube ι R σ))
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B)
    {z : ℂ} (hz : z ∈ scalarTube S δ) :
    Integrable (fun p : ℝ × Hyperplane θ =>
      (w p.1 : ℂ) * (Complex.sqrt ((p.1 : ℂ)^2-z^2))^(Module.finrank ℝ V-1) *
      F (complexSectionPoint (ι θ.val) (ι p.2.val) p.1 z))
      ((volume.prod volume).restrict (sectionParameterSet θ R₀ R)) := by
  apply (complexCrossSection_majorant_integrable θ hwi).mono'
    (complexCrossSection_integrand_measurable ι θ hw hS hδ hgap hQR hmargin hF hz)
  filter_upwards [ae_restrict_mem (measurableSet_Icc.prod measurableSet_closedBall)] with p hp
  exact complexCrossSection_integrand_bound ι θ w hB hS hδ hgap hQR hmargin hFBound hp hz

/-- The integral majorant evaluates to finite unit-ball volume times radius mass. -/
theorem complexCrossSection_majorant_integral (θ : sphere (0 : V) 1)
    (w : ℝ → ℝ) (R₀ R Q B : ℝ) :
    (∫ p : ℝ × Hyperplane θ in sectionParameterSet θ R₀ R,
      |w p.1| * (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B) =
      (∫ a in Set.Icc R₀ R, |w a|) *
        (volume : Measure (Hyperplane θ)).real (closedBall 0 1) *
        (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B := by
  unfold sectionParameterSet
  rw [Measure.volume_eq_prod, ← Measure.prod_restrict]
  simp_rw [mul_assoc]
  rw [integral_prod_mul (fun a : ℝ => |w a|)
    (fun _ : Hyperplane θ => (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B), integral_const]
  simp only [measureReal_def, Measure.restrict_apply MeasurableSet.univ,
    Set.univ_inter, smul_eq_mul]

/-- Explicit magnitude bound for the holomorphic projection continuation,
including the actual transverse unit-ball volume. -/
theorem complexCrossSection_norm_le
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) {w : ℝ → ℝ}
    {S δ Q R₀ R σ B : ℝ} (_hw : Continuous w) (hwi : IntegrableOn w (Set.Icc R₀ R))
    (hwnonneg : ∀ a ∈ Set.Icc R₀ R, 0 ≤ w a)
    (hwnormalized : (∫ a in Set.Icc R₀ R, w a) = 1)
    (hB : 0 ≤ B) (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (_hF : DifferentiableOn ℂ F (realBallTube ι R σ))
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B)
    {z : ℂ} (hz : z ∈ scalarTube S δ) :
    ‖complexCrossSection F ι θ w R₀ R z‖ ≤
      (volume : Measure (Hyperplane θ)).real (closedBall 0 1) *
        (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B := by
  have hnorm : ‖complexCrossSection F ι θ w R₀ R z‖ ≤
      ∫ p : ℝ × Hyperplane θ in sectionParameterSet θ R₀ R,
        |w p.1| * (Real.sqrt (R^2+Q^2))^(Module.finrank ℝ V-1) * B := by
    apply norm_integral_le_of_norm_le (complexCrossSection_majorant_integrable θ hwi)
    filter_upwards [ae_restrict_mem (measurableSet_Icc.prod measurableSet_closedBall)] with p hp
    exact complexCrossSection_integrand_bound ι θ w hB hS hδ hgap hQR hmargin hFBound hp hz
  rw [complexCrossSection_majorant_integral] at hnorm
  have hwabs : (∫ a in Set.Icc R₀ R, |w a|) = 1 := by
    rw [← hwnormalized]
    apply setIntegral_congr_fun measurableSet_Icc
    intro a ha
    exact abs_of_nonneg (hwnonneg a ha)
  simpa only [hwabs, one_mul] using hnorm

/-- At points of the certified tube, the complex continuation is exactly a
nested pair of finite integrals. -/
theorem complexCrossSection_eq_iterated
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) {w : ℝ → ℝ}
    {S δ Q R₀ R σ B : ℝ} (hw : Continuous w) (hwi : IntegrableOn w (Set.Icc R₀ R))
    (hB : 0 ≤ B) (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hF : DifferentiableOn ℂ F (realBallTube ι R σ))
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B)
    {z : ℂ} (hz : z ∈ scalarTube S δ) :
    complexCrossSection F ι θ w R₀ R z =
      ∫ a in Set.Icc R₀ R, (w a : ℂ) * (Complex.sqrt ((a : ℂ)^2-z^2))^(Module.finrank ℝ V-1) *
        ∫ u in closedBall (0 : Hyperplane θ) 1,
          F (complexSectionPoint (ι θ.val) (ι u.val) a z) := by
  have hint := complexCrossSection_integrand_integrable ι θ hw hwi hB hS hδ hgap hQR hmargin hF hFBound hz
  unfold complexCrossSection sectionParameterSet
  rw [Measure.volume_eq_prod, ← Measure.prod_restrict]
  rw [sectionParameterSet, ← Measure.prod_restrict] at hint
  rw [integral_prod _ hint]
  apply integral_congr_ae
  filter_upwards with a
  rw [integral_const_mul]

/-- The complex continuation agrees with the actual hyperplane projection on
its real interval. This is the bridge from an analytic candidate to the
canonical Radon construction. -/
theorem complexCrossSection_eq_radon_real
    {F : E → ℂ} (ι : V →ₗᵢ[ℝ] E) (θ : sphere (0 : V) 1) {w : ℝ → ℝ}
    {S δ Q R₀ R σ B s : ℝ} (hw : Continuous w) (hwi : IntegrableOn w (Set.Icc R₀ R))
    (hB : 0 ≤ B) (hS : 0 ≤ S) (hδ : 0 < δ) (hgap : S+δ ≤ Q) (hQR : Q < R₀)
    (hmargin : (1+Q/Real.sqrt (R₀^2-Q^2))*δ ≤ σ)
    (hF : DifferentiableOn ℂ F (realBallTube ι R σ))
    (hFBound : ∀ v ∈ realBallTube ι R σ, ‖F v‖ ≤ B)
    (hs : |s| ≤ S) :
    complexCrossSection F ι θ w R₀ R (s : ℂ) =
      hyperplaneRadon (fun x => F (ι x) * (radialCutoff w R₀ R ‖x‖ : ℂ)) θ s := by
  have hQ : 0 ≤ Q := by linarith
  have hσ : 0 < σ := lt_of_lt_of_le (mul_pos (by positivity) hδ) hmargin
  have hmaps : Set.MapsTo ι (closedBall (0 : V) R) (realBallTube ι R σ) := by
    intro x hx
    exact Set.mem_iUnion.mpr ⟨x, Set.mem_iUnion.mpr ⟨hx, mem_ball_self hσ⟩⟩
  have hreal : ContinuousOn (fun x => F (ι x)) (closedBall (0 : V) R) :=
    hF.continuousOn.comp ι.continuous.continuousOn hmaps
  have hrealB : ∀ x ∈ closedBall (0 : V) R, ‖F (ι x)‖ ≤ B :=
    fun x hx => hFBound _ (hmaps hx)
  have hsR₀ : |s| < R₀ := by linarith
  have hsTube : (s : ℂ) ∈ scalarTube S δ :=
    Set.mem_iUnion.mpr ⟨s, Set.mem_iUnion.mpr ⟨abs_le.mp hs, mem_ball_self hδ⟩⟩
  rw [complexCrossSection_eq_iterated ι θ hw hwi hB hS hδ hgap hQR hmargin hF hFBound hsTube,
    ballMixture_unitBall_formula hreal hwi hB hrealB θ hsR₀]
  apply setIntegral_congr_fun measurableSet_Icc
  intro a ha
  dsimp only
  have hsa : |s| < a := hsR₀.trans_le ha.1
  have hsq : 0 ≤ a^2-s^2 := by nlinarith [abs_nonneg s, sq_abs s]
  rw [section_sqrt_ofReal a s hsq]
  have hinner : (∫ u in closedBall (0 : Hyperplane θ) 1,
      F (complexSectionPoint (ι θ.val) (ι u.val) a (s : ℂ))) =
      ∫ u in closedBall (0 : Hyperplane θ) 1,
        F (ι (s • θ.val + Real.sqrt (a^2-s^2) • u.val)) := by
    apply integral_congr_ae
    filter_upwards with u
    rw [complexSectionPoint_ofReal ι θ.val u.val a s hsq]
  rw [hinner, Complex.real_smul, Complex.ofReal_pow]
  ring

end CrossSectionAnalytic

end QIRadon.Radon
