import Mathlib.Analysis.Complex.Liouville
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Topology.UniformSpace.UniformApproximation

/-! Continuity of derivatives of uniformly bounded holomorphic orbit families.
Only real-parameter slices are assumed continuous in the external parameter.
No continuity of a chosen complex extension in that parameter is assumed. -/

noncomputable section
open Set Metric Filter
open scoped Topology

namespace QIRadon.OrbitContinuity

theorem second_derivative_bound {f : ℂ → ℂ} {r B : ℝ} (hr : 0 < r)
    (hf : DifferentiableOn ℂ f (ball 0 r))
    (hb : ∀ z ∈ ball (0:ℂ) r, ‖f z‖ ≤ B)
    {z : ℂ} (hz : ‖z‖ ≤ r/4) :
    ‖deriv (deriv f) z‖ ≤ 8*B/r^2 := by
  have hs : closedBall z (r/2) ⊆ ball (0:ℂ) r := by
    intro w hw
    have hw' : ‖w-z‖ ≤ r/2 := by simpa [mem_closedBall, dist_eq_norm] using hw
    have hn : ‖w‖ ≤ ‖w-z‖+‖z‖ := by simpa using norm_add_le (w-z) z
    rw [mem_ball, dist_zero_right]
    linarith
  have hdc : DiffContOnCl ℂ f (ball z (r/2)) := by
    apply DifferentiableOn.diffContOnCl
    rw [closure_ball _ (by positivity : r/2 ≠ 0)]
    exact hf.mono hs
  have h := Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le 2
    (by positivity : 0 < r/2) hdc (fun w hw => hb w (hs (sphere_subset_closedBall hw)))
  simpa only [iteratedDeriv_succ, iteratedDeriv_zero, Nat.factorial_two, Nat.cast_ofNat,
    show (2:ℝ)*B/(r/2)^2 = 8*B/r^2 by ring] using h

/-- Uniform real (indeed complex) finite-difference approximation of the derivative. -/
theorem slope_error {f : ℂ → ℂ} {r B : ℝ} (hr : 0 < r)
    (hf : DifferentiableOn ℂ f (ball 0 r))
    (hb : ∀ z ∈ ball (0:ℂ) r, ‖f z‖ ≤ B)
    {t : ℂ} (ht : t ≠ 0) (htR : ‖t‖ ≤ r/4) :
    ‖(f t-f 0)/t - deriv f 0‖ ≤ (8*B/r^2)*‖t‖ := by
  let S := closedBall (0:ℂ) ‖t‖
  have hsub : S ⊆ ball (0:ℂ) r := by
    intro z hz
    have hz' : ‖z‖ ≤ ‖t‖ := by simpa [S] using hz
    simpa only [mem_ball, dist_zero_right] using (show ‖z‖ < r by linarith)
  have ha (z : ℂ) (hz : z ∈ S) : AnalyticAt ℂ f z :=
    hf.analyticAt (isOpen_ball.mem_nhds (hsub hz))
  have h0 : (0:ℂ) ∈ S := by simp [S]
  have htS : t ∈ S := by simp [S]
  have hdb (z : ℂ) (hz : z ∈ S) : ‖deriv (deriv f) z‖ ≤ 8*B/r^2 := by
    exact second_derivative_bound hr hf hb
      ((by simpa [S] using hz : ‖z‖ ≤ ‖t‖).trans htR)
  have hd (z : ℂ) (hz : z ∈ S) : ‖deriv f z-deriv f 0‖ ≤ (8*B/r^2)*‖t‖ := by
    have h := (convex_closedBall (0:ℂ) ‖t‖).norm_image_sub_le_of_norm_deriv_le
      (fun w hw => (ha w hw).deriv.differentiableAt) hdb h0 hz
    rw [sub_zero] at h
    apply h.trans
    have hB : 0 ≤ B := (norm_nonneg _).trans (hb 0 (mem_ball_self hr))
    gcongr
    simpa [S] using hz
  let g : ℂ → ℂ := fun z => f z-f 0-z*deriv f 0
  have hg (z : ℂ) (hz : z ∈ S) : HasDerivAt g (deriv f z-deriv f 0) z := by
    convert! (((ha z hz).differentiableAt.hasDerivAt.sub_const (f 0)).sub
      ((hasDerivAt_id z).mul_const (deriv f 0))) using 1
    simp
  have hb' (z : ℂ) (hz : z ∈ S) : ‖deriv g z‖ ≤ (8*B/r^2)*‖t‖ := by
    rw [(hg z hz).deriv]
    exact hd z hz
  have hg' := (convex_closedBall (0:ℂ) ‖t‖).norm_image_sub_le_of_norm_deriv_le
    (fun z hz => (hg z hz).differentiableAt) hb' h0 htS
  simp only [g, sub_zero, zero_mul, sub_self] at hg'
  have hid : (f t-f 0)/t-deriv f 0 = (f t-f 0-t*deriv f 0)/t := by field_simp
  rw [hid, norm_div]
  exact (div_le_iff₀ (norm_pos_iff.mpr ht)).mpr hg'

/-- Uniformly bounded holomorphic extensions make the derivative continuous in
an external parameter, even when only real slices are known continuous. -/
theorem continuous_deriv_zero {X : Type*} [TopologicalSpace X]
    (F : X → ℂ → ℂ) {r B : ℝ} (hr : 0 < r)
    (hf : ∀ x, DifferentiableOn ℂ (F x) (ball 0 r))
    (hb : ∀ x z, z ∈ ball (0:ℂ) r → ‖F x z‖ ≤ B)
    (hc : ∀ t : ℝ, |t| < r → Continuous (fun x => F x (t:ℂ))) :
    Continuous (fun x => deriv (F x) 0) := by
  apply continuous_of_uniform_approx_of_continuous
  intro u hu
  obtain ⟨ε, hε, hεu⟩ := Metric.uniformity_basis_dist.mem_iff.mp hu
  let C := 8*|B|/r^2
  have hC : 0 ≤ C := by dsimp [C]; positivity
  let t := min (r/8) (ε/(2*(C+1)))
  have ht : 0 < t := lt_min (by positivity) (by positivity)
  have htR : t ≤ r/4 := (min_le_left _ _).trans (by linarith)
  have htε : C*t < ε := by
    have ht1 : t ≤ ε/(2*(C+1)) := min_le_right _ _
    have hh := (le_div_iff₀ (show 0 < 2*(C+1) by positivity)).mp ht1
    nlinarith
  let Q : X → ℂ := fun x => (F x (t:ℂ)-F x 0)/(t:ℂ)
  refine ⟨Q, ?_, ?_⟩
  · exact ((hc t (by rw [abs_of_pos ht]; linarith)).sub
      (by simpa using hc 0 (by simpa using hr))).div_const (t:ℂ)
  · intro x
    apply hεu
    change dist (deriv (F x) 0) (Q x) < ε
    rw [dist_eq_norm, norm_sub_rev]
    have hh := slope_error hr (hf x)
      (fun z hz => (hb x z hz).trans (le_abs_self B))
      (show (t:ℂ) ≠ 0 by exact_mod_cast ht.ne')
      (show ‖(t:ℂ)‖ ≤ r/4 by simpa [abs_of_pos ht] using htR)
    have hh' : ‖Q x-deriv (F x) 0‖ ≤ C*t := by
      simpa only [Q, C, Complex.norm_real, Real.norm_eq_abs, abs_of_pos ht] using hh
    exact hh'.trans_lt htε

end QIRadon.OrbitContinuity
