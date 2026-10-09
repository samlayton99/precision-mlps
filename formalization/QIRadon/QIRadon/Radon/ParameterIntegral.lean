import QIRadon.Radon.Analytic
import Mathlib.Analysis.Complex.Schwarz
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.Normed.Module.FiniteDimension

/-! Finite-dimensional complex parameter integration. Joint derivative
measurability is derived from scalar line difference quotients and a finite
basis; the derivative majorant is derived from actual local value bounds. -/

noncomputable section
open MeasureTheory Set Metric Filter
open scoped Topology BigOperators ContDiff
namespace QIRadon.Radon.ParameterIntegral

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
variable {A : Type*} [MeasurableSpace A] {μ : Measure A}

/-- A derivative evaluated on any fixed vector is measurable because it is the
limit of measurable scalar-line difference quotients. -/
theorem fderiv_apply_aestronglyMeasurable
    {F : E → A → ℂ} {z₀ : E} {δ : ℝ} (hδ : 0 < δ)
    (hMeas : ∀ z ∈ ball z₀ δ, AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableAt ℂ (fun z => F z a) z₀) (v : E) :
    AEStronglyMeasurable (fun a => fderiv ℂ (fun z => F z a) z₀ v) μ := by
  let τ := δ/(1+‖v‖)
  have hτ : 0 < τ := by dsimp [τ]; positivity
  let G : ℂ → A → ℂ := fun t a => F (z₀+t • v) a
  have hnear : ∀ t ∈ ball (0 : ℂ) τ, z₀+t • v ∈ ball z₀ δ := by
    intro t ht
    have ht' : ‖t‖ < δ/(1+‖v‖) := by simpa only [mem_ball, dist_zero_right, τ] using ht
    have hh := (lt_div_iff₀ (show 0 < 1+‖v‖ by positivity)).mp ht'
    rw [mem_ball_iff_norm, add_sub_cancel_left, norm_smul]
    nlinarith [norm_nonneg t]
  have hline : HasDerivAt (fun t : ℂ => z₀+t • v) v 0 := by
    convert! ((hasDerivAt_id (0 : ℂ)).smul_const v).const_add z₀ using 1
    simp only [one_smul]
  have hcomp : ∀ᵐ a ∂μ, HasDerivAt (fun t => G t a) (fderiv ℂ (fun z => F z a) z₀ v) 0 := by
    filter_upwards [hDiff] with a ha
    have hf : HasFDerivAt (fun z => F z a) (fderiv ℂ (fun z => F z a) z₀) (z₀+(0 : ℂ) • v) := by
      simpa only [zero_smul, add_zero] using ha.hasFDerivAt
    exact hf.comp_hasDerivAt 0 hline
  have hm := QIRadon.Radon.derivative_parameter_aestronglyMeasurable hτ
    (fun t ht => hMeas _ (hnear t ht)) (hcomp.mono (fun _ ha => ha.differentiableAt))
  exact hm.congr (hcomp.mono (fun _ ha => ha.deriv))

/-- In finite dimension, finitely many measurable derivative evaluations
reconstruct the actual bounded linear derivative. -/
theorem fderiv_aestronglyMeasurable [FiniteDimensional ℂ E]
    {F : E → A → ℂ} {z₀ : E} {δ : ℝ} (hδ : 0 < δ)
    (hMeas : ∀ z ∈ ball z₀ δ, AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableAt ℂ (fun z => F z a) z₀) :
    AEStronglyMeasurable (fun a => fderiv ℂ (fun z => F z a) z₀) μ := by
  let e := Module.finBasis ℂ E
  let coord := fun i : Fin (Module.finrank ℂ E) => (e.coord i).toContinuousLinearMap
  have hmeas : AEStronglyMeasurable (fun a => ∑ i : Fin (Module.finrank ℂ E),
      (fderiv ℂ (fun z => F z a) z₀ (e i)) • coord i) μ := by
    have hh := Finset.aestronglyMeasurable_sum Finset.univ (fun i _ =>
      (fderiv_apply_aestronglyMeasurable hδ hMeas hDiff (e i)).smul_const (coord i))
    convert! hh using 1
    ext a
    simp only [Finset.sum_apply]
  apply hmeas.congr
  filter_upwards with a
  ext x
  rw [sum_apply]
  have hh := congrArg (fderiv ℂ (fun z => F z a) z₀) (e.sum_equivFun x)
  rw [map_sum] at hh
  calc
    _ = ∑ i : Fin (Module.finrank ℂ E),
        fderiv ℂ (fun z => F z a) z₀ (e.equivFun x i • e i) := by
      apply Finset.sum_congr rfl
      intro i _
      simp only [smul_apply, map_smul, smul_eq_mul]
      change (fderiv ℂ (fun z => F z a) z₀ (e i)) * e.repr x i =
        e.equivFun x i * fderiv ℂ (fun z => F z a) z₀ (e i)
      rw [Module.Basis.equivFun_apply]
      ring
    _ = _ := hh

/-- A local value envelope gives an operator-derivative envelope. The factor
two comes from centering the image ball at the actual function value. -/
theorem local_fderiv_bound {f : E → ℂ} {z₀ : E} {δ M : ℝ}
    (hδ : 0 < δ) (hDiff : DifferentiableOn ℂ f (ball z₀ (2*δ)))
    (hBound : ∀ z ∈ ball z₀ (2*δ), ‖f z‖ ≤ M) :
    ∀ z ∈ ball z₀ δ, ‖fderiv ℂ f z‖ ≤ (2*M)/δ := by
  intro z hz
  have hsub : ball z δ ⊆ ball z₀ (2*δ) := by
    intro u hu
    have htri := dist_triangle u z z₀
    have hu' : dist u z < δ := hu
    have hz' : dist z z₀ < δ := hz
    change dist u z₀ < 2*δ
    linarith
  have hzbig := hsub (mem_ball_self hδ)
  apply Complex.norm_fderiv_le_div_of_mapsTo_ball (hDiff.mono hsub) _ hδ
  intro u hu
  change dist (f u) (f z) ≤ 2*M
  rw [dist_eq_norm]
  exact (norm_sub_le _ _).trans (by linarith [hBound u (hsub hu), hBound z hzbig])

/-- Genuine joint Fréchet differentiability of the actual integral under only
measurability, fiber holomorphy, and an integrable local value envelope. -/
theorem hasFDerivAt_integral_of_ball_bound [FiniteDimensional ℂ E]
    {F : E → A → ℂ} {z₀ : E} {δ : ℝ} {M : A → ℝ} (hδ : 0 < δ)
    (hMeas : ∀ z ∈ ball z₀ (2*δ), AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableOn ℂ (fun z => F z a) (ball z₀ (2*δ)))
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ ball z₀ (2*δ), ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) :
    HasFDerivAt (fun z => ∫ a, F z a ∂μ)
      (∫ a, fderiv ℂ (fun z => F z a) z₀ ∂μ) z₀ := by
  have h2δ : 0 < 2*δ := by positivity
  have hcenter := mem_ball_self (x := z₀) h2δ
  have hFint : Integrable (F z₀) μ :=
    hM.mono' (hMeas z₀ hcenter) (hBound.mono (fun a ha => ha z₀ hcenter))
  have hF' := fderiv_aestronglyMeasurable h2δ hMeas
    (hDiff.mono (fun a ha => ha.differentiableAt (isOpen_ball.mem_nhds hcenter)))
  have hderiv : ∀ᵐ a ∂μ, ∀ z ∈ ball z₀ δ,
      ‖fderiv ℂ (fun w => F w a) z‖ ≤ (2*M a)/δ := by
    filter_upwards [hDiff, hBound] with a ha hba
    exact local_fderiv_bound hδ ha hba
  have hlocMeas : ∀ᶠ z in nhds z₀, AEStronglyMeasurable (F z) μ := by
    filter_upwards [ball_mem_nhds z₀ h2δ] with z hz
    exact hMeas z hz
  have hlocDiff : ∀ᵐ a ∂μ, ∀ z ∈ ball z₀ δ,
      HasFDerivAt (fun w => F w a) (fderiv ℂ (fun w => F w a) z) z := by
    filter_upwards [hDiff] with a ha
    intro z hz
    have hzbig : z ∈ ball z₀ (2*δ) := by
      have hh : dist z z₀ < δ := hz
      change dist z z₀ < 2*δ
      linarith
    exact (ha.differentiableAt (isOpen_ball.mem_nhds hzbig)).hasFDerivAt
  exact (hasFDerivAt_integral_of_dominated_of_fderiv_le
    (ball_mem_nhds z₀ hδ) hlocMeas hFint hF' hderiv
    ((hM.const_mul 2).div_const δ) hlocDiff)

theorem differentiableOn_integral_of_dominated [FiniteDimensional ℂ E]
    {F : E → A → ℂ} {U : Set E} {M : A → ℝ} (hU : IsOpen U)
    (hMeas : ∀ z ∈ U, AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableOn ℂ (fun z => F z a) U)
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ U, ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) :
    DifferentiableOn ℂ (fun z => ∫ a, F z a ∂μ) U := by
  intro z hz
  obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp (hU.mem_nhds hz)
  have hδ : 0 < ε/3 := by positivity
  have hsub : ball z (2*(ε/3)) ⊆ U := by
    intro w hw
    apply hball
    have hd : dist w z < 2*(ε/3) := hw
    change dist w z < ε
    linarith
  exact (hasFDerivAt_integral_of_ball_bound hδ
    (fun w hw => hMeas w (hsub hw))
    (hDiff.mono (fun a ha => ha.mono hsub))
    (hBound.mono (fun a ha w hw => ha w (hsub hw))) hM).differentiableAt.differentiableWithinAt


/-- The derivative formula at any point of an arbitrary open parameter set. -/
theorem hasFDerivAt_integral_of_dominated [FiniteDimensional ℂ E]
    {F : E → A → ℂ} {U : Set E} {M : A → ℝ} (hU : IsOpen U)
    (hMeas : ∀ z ∈ U, AEStronglyMeasurable (F z) μ)
    (hDiff : ∀ᵐ a ∂μ, DifferentiableOn ℂ (fun z => F z a) U)
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ U, ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) {z : E} (hz : z ∈ U) :
    HasFDerivAt (fun z => ∫ a, F z a ∂μ) (∫ a, fderiv ℂ (fun z => F z a) z ∂μ) z := by
  obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp (hU.mem_nhds hz)
  have hδ : 0 < ε/3 := by positivity
  have hsub : ball z (2*(ε/3)) ⊆ U := by
    intro w hw
    apply hball
    have hd : dist w z < 2*(ε/3) := hw
    change dist w z < ε
    linarith
  exact hasFDerivAt_integral_of_ball_bound hδ
    (fun w hw => hMeas w (hsub hw))
    (hDiff.mono (fun a ha => ha.mono hsub))
    (hBound.mono (fun a ha w hw => ha w (hsub hw))) hM

/-- Every finite derivative order of the actual integral exists. The proof
recursively differentiates measurable scalar evaluations of its Fréchet
operator; Cauchy/Schwarz supplies each required local integrable envelope. -/
theorem contDiffOn_integral_of_dominated [FiniteDimensional ℂ E]
    (n : ℕ) {F : E → A → ℂ} {U : Set E} {M : A → ℝ} (hU : IsOpen U)
    (hMeas : ∀ z ∈ U, AEStronglyMeasurable (F z) μ)
    (hSmooth : ∀ᵐ a ∂μ, ContDiffOn ℂ ∞ (fun z => F z a) U)
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ U, ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) :
    ContDiffOn ℂ n (fun z => ∫ a, F z a ∂μ) U := by
  induction n generalizing F U M with
  | zero =>
      rw [Nat.cast_zero, contDiffOn_zero]
      exact (differentiableOn_integral_of_dominated hU hMeas
        (hSmooth.mono (fun _ ha => ha.differentiableOn (by simp))) hBound hM).continuousOn
  | succ n ih =>
      have hDiff : ∀ᵐ a ∂μ, DifferentiableOn ℂ (fun z => F z a) U :=
        hSmooth.mono (fun _ ha => ha.differentiableOn (by simp))
      intro z hz
      obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp (hU.mem_nhds hz)
      let δ := ε/3
      have hδ : 0 < δ := by dsimp [δ]; positivity
      have hsub : ball z (2*δ) ⊆ U := by
        intro w hw
        apply hball
        have hd : dist w z < 2*δ := hw
        change dist w z < ε
        dsimp [δ] at hd
        linarith
      have hsmall : ball z δ ⊆ U := by
        intro w hw
        apply hsub
        have hd : dist w z < δ := hw
        change dist w z < 2*δ
        linarith
      let D : E → A → E →L[ℂ] ℂ := fun w a => fderiv ℂ (fun t => F t a) w
      have hDbound : ∀ᵐ a ∂μ, ∀ w ∈ ball z δ, ‖D w a‖ ≤ (2*M a)/δ := by
        filter_upwards [hDiff, hBound] with a ha hb
        exact local_fderiv_bound hδ (ha.mono hsub) (fun w hw => hb w (hsub hw))
      have hDmeas : ∀ w ∈ ball z δ, AEStronglyMeasurable (D w) μ := by
        intro w hw
        obtain ⟨ρ, hρ, hρball⟩ := Metric.mem_nhds_iff.mp (hU.mem_nhds (hsmall hw))
        exact fderiv_aestronglyMeasurable hρ (fun t ht => hMeas t (hρball ht))
          (hDiff.mono (fun _ ha => ha.differentiableAt (hU.mem_nhds (hsmall hw))))
      have hDint : ∀ w ∈ ball z δ, Integrable (D w) μ := by
        intro w hw
        exact ((hM.const_mul 2).div_const δ).mono' (hDmeas w hw)
          (hDbound.mono (fun _ ha => ha w hw))
      have hDscalar : ∀ v : E, ContDiffOn ℂ n
          (fun w => (fderiv ℂ (fun t => ∫ a, F t a ∂μ) w) v) (ball z δ) := by
        intro v
        have hScalarMeas : ∀ w ∈ ball z δ, AEStronglyMeasurable (fun a => D w a v) μ :=
          fun w hw => (hDmeas w hw).apply_continuousLinearMap v
        have hScalarSmooth : ∀ᵐ a ∂μ,
            ContDiffOn ℂ ∞ (fun w => D w a v) (ball z δ) := by
          filter_upwards [hSmooth] with a ha
          have hd : ContDiffOn ℂ ∞ (fderiv ℂ (fun w => F w a)) U :=
            ha.fderiv_of_isOpen hU (by simp)
          exact ((contDiffOn_clm_apply.mp hd) v).mono hsmall
        have hScalarBound : ∀ᵐ a ∂μ, ∀ w ∈ ball z δ,
            ‖D w a v‖ ≤ ((2*M a)/δ)*‖v‖ := by
          filter_upwards [hDbound] with a ha
          intro w hw
          exact (D w a).le_opNorm v |>.trans (mul_le_mul_of_nonneg_right (ha w hw) (norm_nonneg v))
        have hh := ih isOpen_ball hScalarMeas hScalarSmooth hScalarBound
          (((hM.const_mul 2).div_const δ).mul_const ‖v‖)
        apply hh.congr
        intro w hw
        rw [(hasFDerivAt_integral_of_dominated hU hMeas hDiff hBound hM (hsmall hw)).fderiv]
        exact ContinuousLinearMap.integral_apply (hDint w hw) v
      have hDC : ContDiffOn ℂ n
          (fderiv ℂ (fun t => ∫ a, F t a ∂μ)) (ball z δ) := contDiffOn_clm_apply.mpr hDscalar
      have hpoint : ContDiffAt ℂ (n+1) (fun t => ∫ a, F t a ∂μ) z := by
        rw [contDiffAt_succ_iff_hasFDerivAt]
        refine ⟨fderiv ℂ (fun t => ∫ a, F t a ∂μ), ?_, hDC.contDiffAt (ball_mem_nhds z hδ)⟩
        refine ⟨ball z δ, ball_mem_nhds z hδ, ?_⟩
        intro w hw
        exact (hasFDerivAt_integral_of_dominated hU hMeas hDiff hBound hM (hsmall hw)).differentiableAt.hasFDerivAt
      exact hpoint.contDiffWithinAt

/-- Actual joint complex C∞ regularity; no joint analyticity theorem for the
integral is assumed. This suffices for successive one-variable Cauchy bounds. -/
theorem smoothOn_integral_of_dominated [FiniteDimensional ℂ E]
    {F : E → A → ℂ} {U : Set E} {M : A → ℝ} (hU : IsOpen U)
    (hMeas : ∀ z ∈ U, AEStronglyMeasurable (F z) μ)
    (hSmooth : ∀ᵐ a ∂μ, ContDiffOn ℂ ∞ (fun z => F z a) U)
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ U, ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) :
    ContDiffOn ℂ ∞ (fun z => ∫ a, F z a ∂μ) U := by
  rw [contDiffOn_infty]
  intro n
  exact contDiffOn_integral_of_dominated n hU hMeas hSmooth hBound hM

/-- Direct input from actual jointly analytic fibers, preserving all mixed
derivatives of the integrated function without an analytic-integral oracle. -/
theorem smoothOn_integral_of_analytic_fibers [FiniteDimensional ℂ E]
    {F : E → A → ℂ} {U : Set E} {M : A → ℝ} (hU : IsOpen U)
    (hMeas : ∀ z ∈ U, AEStronglyMeasurable (F z) μ)
    (hAnalytic : ∀ᵐ a ∂μ, AnalyticOnNhd ℂ (fun z => F z a) U)
    (hBound : ∀ᵐ a ∂μ, ∀ z ∈ U, ‖F z a‖ ≤ M a)
    (hM : Integrable M μ) :
    ContDiffOn ℂ ∞ (fun z => ∫ a, F z a ∂μ) U :=
  smoothOn_integral_of_dominated hU hMeas
    (hAnalytic.mono (fun _ ha => ha.contDiffOn hU.uniqueDiffOn)) hBound hM

end QIRadon.Radon.ParameterIntegral
