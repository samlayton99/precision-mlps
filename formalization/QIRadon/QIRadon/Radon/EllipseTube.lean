import QIRadon.Radon.Analytic
import QIRadon.Scalar.Preparation.PeriodicStrip

/-! Quantitative containment of the actual cosine-defined Bernstein ellipse
 in the scalar tube supplied by the Radon construction. -/

open Set Metric
open scoped ContDiff Topology

namespace QIRadon.Radon

noncomputable section
set_option backward.isDefEq.respectTransparency false

theorem cos_real_add_displacement (x : ℝ) (y : ℂ) :
    ‖Complex.cos ((x:ℂ)+y)-(Real.cos x:ℂ)‖ ≤ Real.exp ‖y‖-1 := by
  have hid : Complex.cos ((x:ℂ)+y)-Complex.cos (x:ℂ) =
      (Complex.exp ((x:ℂ)*Complex.I)*(Complex.exp (y*Complex.I)-1) +
        Complex.exp (-(x:ℂ)*Complex.I)*(Complex.exp (-y*Complex.I)-1))/2 := by
    simp only [Complex.cos, add_mul, neg_add, Complex.exp_add]
    ring
  have h1 : ‖Complex.exp (y*Complex.I)-1‖ ≤ Real.exp ‖y‖-1 := by
    simpa using Complex.norm_exp_sub_sum_le_exp_norm_sub_sum (y*Complex.I) 1
  have h2 : ‖Complex.exp (-y*Complex.I)-1‖ ≤ Real.exp ‖y‖-1 := by
    simpa using Complex.norm_exp_sub_sum_le_exp_norm_sub_sum (-y*Complex.I) 1
  rw [Complex.ofReal_cos, hid, norm_div, Complex.norm_two]
  have h := norm_add_le
    (Complex.exp ((x:ℂ)*Complex.I)*(Complex.exp (y*Complex.I)-1))
    (Complex.exp (-(x:ℂ)*Complex.I)*(Complex.exp (-y*Complex.I)-1))
  simp only [norm_mul, Complex.norm_exp, Complex.mul_re, Complex.ofReal_re,
    Complex.ofReal_im, Complex.I_re, Complex.I_im, mul_zero, zero_mul,
    sub_zero, Complex.neg_re, Complex.neg_im, neg_zero, Real.exp_zero, one_mul] at h
  linarith

theorem cos_near_real_segment (z : ℂ) :
    ‖Complex.cos z-(Real.cos z.re:ℂ)‖ ≤ Real.exp |z.im|-1 := by
  have hz : z = (z.re:ℂ)+(z.im:ℂ)*Complex.I := z.re_add_im.symm
  have h := cos_real_add_displacement z.re ((z.im:ℂ)*Complex.I)
  simpa only [← hz, norm_mul, Complex.norm_I, mul_one, Complex.norm_real, Real.norm_eq_abs] using h

/-- General scaled ellipse containment. The nearest real point is explicitly
 A*cos(Re z), and the imaginary-strip displacement controls its tube radius. -/
theorem scaled_bernsteinEllipse_subset_scalarTube {A L δ η : ℝ}
    (hA : 0 ≤ A) (hAL : A ≤ L) (hmargin : A*(Real.exp η-1) < δ) :
    (fun z : ℂ ↦ (A:ℂ)*z) '' ProfilePreparation.bernsteinEllipse η ⊆ scalarTube L δ := by
  rintro z ⟨y, ⟨w, hw, rfl⟩, rfl⟩
  have hs : |A*Real.cos w.re| ≤ L := by
    rw [abs_mul, abs_of_nonneg hA]
    exact (mul_le_mul_of_nonneg_left (Real.abs_cos_le_one _) hA).trans (by simpa using hAL)
  simp only [scalarTube, mem_iUnion]
  refine ⟨A*Real.cos w.re, abs_le.mp hs, ?_⟩
  rw [mem_ball, dist_eq_norm]
  have heq : (A:ℂ)*Complex.cos w-((A*Real.cos w.re:ℝ):ℂ) =
      (A:ℂ)*(Complex.cos w-(Real.cos w.re:ℂ)) := by push_cast; ring
  rw [heq, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hA]
  apply lt_of_le_of_lt (mul_le_mul_of_nonneg_left (cos_near_real_segment w) hA) ?_
  have hw' : |w.im| ≤ η := hw
  exact (mul_le_mul_of_nonneg_left
    (sub_le_sub_right (Real.exp_le_exp.mpr hw') 1) hA).trans_lt hmargin

def ellipseWidth (A δ : ℝ) : ℝ := Real.log (1+δ/(2*A))

theorem ellipseWidth_pos {A δ : ℝ} (hA : 0 < A) (hδ : 0 < δ) : 0 < ellipseWidth A δ := by
  apply Real.log_pos
  have h : 0 < δ/(2*A) := by positivity
  linarith

theorem ellipseWidth_margin {A δ : ℝ} (hA : 0 < A) (hδ : 0 < δ) :
    A*(Real.exp (ellipseWidth A δ)-1) = δ/2 := by
  rw [ellipseWidth, Real.exp_log (by positivity)]
  field_simp
  ring

/-- The explicit logarithmic strip width reserves half the tube radius, so
 the closed Bernstein ellipse remains strictly inside the open source tube. -/
theorem scaled_bernsteinEllipse_log_subset {A L δ : ℝ}
    (hA : 0 < A) (hAL : A ≤ L) (hδ : 0 < δ) :
    (fun z : ℂ ↦ (A:ℂ)*z) '' ProfilePreparation.bernsteinEllipse (ellipseWidth A δ) ⊆ scalarTube L δ := by
  apply scaled_bernsteinEllipse_subset_scalarTube hA.le hAL
  rw [ellipseWidth_margin hA hδ]
  linarith

theorem bernsteinEllipse_log_subset {L δ : ℝ} (hL : 1 ≤ L) (hδ : 0 < δ) :
    ProfilePreparation.bernsteinEllipse (Real.log (1+δ/2)) ⊆ scalarTube L δ := by
  simpa [ellipseWidth] using scaled_bernsteinEllipse_log_subset (A := 1) (by norm_num) hL hδ


/-- In one complex variable, actual differentiability on the open tube gives
 the local analyticity needed by Chebyshev preparation. -/
theorem scalarTube_analyticOnNhd {G : ℂ → ℂ} {L δ : ℝ}
    (hG : DifferentiableOn ℂ G (scalarTube L δ)) :
    AnalyticOnNhd ℂ G (scalarTube L δ) := by
  intro z hz
  exact hG.analyticAt ((isOpen_scalarTube L δ).mem_nhds hz)

theorem scalarTube_smooth_analyticOnNhd {G : ℂ → ℂ} {L δ : ℝ}
    (hG : ContDiffOn ℂ ∞ G (scalarTube L δ)) :
    AnalyticOnNhd ℂ G (scalarTube L δ) :=
  scalarTube_analyticOnNhd (hG.differentiableOn (by simp))

/-- Analyticity and the existing source magnitude bound transfer together to
 the explicit scaled Bernstein ellipse; no larger enclosing-disk assumption
 is introduced. -/
theorem scaled_ellipse_transfer {G : ℂ → ℂ} {A L δ B : ℝ}
    (hA : 0 < A) (hAL : A ≤ L) (hδ : 0 < δ)
    (hG : DifferentiableOn ℂ G (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖G z‖ ≤ B) :
    AnalyticOnNhd ℂ (fun z ↦ G ((A:ℂ)*z))
      (ProfilePreparation.bernsteinEllipse (ellipseWidth A δ)) ∧
    ∀ z ∈ ProfilePreparation.bernsteinEllipse (ellipseWidth A δ), ‖G ((A:ℂ)*z)‖ ≤ B := by
  have hmaps : MapsTo (fun z : ℂ ↦ (A:ℂ)*z)
      (ProfilePreparation.bernsteinEllipse (ellipseWidth A δ)) (scalarTube L δ) := by
    intro z hz
    exact scaled_bernsteinEllipse_log_subset hA hAL hδ ⟨z,hz,rfl⟩
  constructor
  · intro z hz
    exact ((scalarTube_analyticOnNhd hG) _ (hmaps hz)).comp (by fun_prop)
  · intro z hz
    exact hbound _ (hmaps hz)

theorem ellipse_transfer {G : ℂ → ℂ} {L δ B : ℝ}
    (hL : 1 ≤ L) (hδ : 0 < δ)
    (hG : DifferentiableOn ℂ G (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖G z‖ ≤ B) :
    AnalyticOnNhd ℂ G (ProfilePreparation.bernsteinEllipse (Real.log (1+δ/2))) ∧
    ∀ z ∈ ProfilePreparation.bernsteinEllipse (Real.log (1+δ/2)), ‖G z‖ ≤ B := by
  simpa [ellipseWidth] using scaled_ellipse_transfer (A := 1) (by norm_num) hL hδ hG hbound

/-- Joint complex smoothness of an actual parameterized source profile is
 sufficient; only its one-dimensional scalar restriction is made analytic. -/
theorem scalar_slice_ellipse_transfer {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
    {U : Set E} {P : E × ℂ → ℂ} {L δ B : ℝ}
    (hL : 1 ≤ L) (hδ : 0 < δ)
    (hP : ContDiffOn ℂ ∞ P (U ×ˢ scalarTube L δ))
    (hbound : ∀ p ∈ U ×ˢ scalarTube L δ, ‖P p‖ ≤ B)
    (u : E) (hu : u ∈ U) :
    AnalyticOnNhd ℂ (fun z ↦ P (u,z))
      (ProfilePreparation.bernsteinEllipse (Real.log (1+δ/2))) ∧
    ∀ z ∈ ProfilePreparation.bernsteinEllipse (Real.log (1+δ/2)), ‖P (u,z)‖ ≤ B := by
  have hs : ContDiffOn ℂ ∞ (fun z ↦ P (u,z)) (scalarTube L δ) :=
    hP.comp (contDiffOn_const.prodMk contDiffOn_id) (fun _ hz ↦ ⟨hu,hz⟩)
  exact ellipse_transfer hL hδ (hs.differentiableOn (by simp)) (fun z hz ↦ hbound _ ⟨hu,hz⟩)

end
end QIRadon.Radon
