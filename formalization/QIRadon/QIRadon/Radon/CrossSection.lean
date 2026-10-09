import QIRadon.Radon.Slice
import QIRadon.Radon.Boundary

/-! Actual finite ball-mixture cross sections of the prescribed radial extension. -/

noncomputable section
open MeasureTheory Set Metric
open scoped RealInnerProductSpace Pointwise

namespace QIRadon.Radon

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
/-- Orthogonal hyperplane coordinates give the exact cross-section radius. -/
theorem norm_hyperplane_point_sq (θ : sphere (0 : V) 1) (s : ℝ) (y : Hyperplane θ) :
    ‖s • θ.val + y.val‖^2 = s^2 + ‖y‖^2 := by
  have hy : inner ℝ θ.val y.val = 0 :=
    Submodule.mem_orthogonal_singleton_iff_inner_right.mp y.property
  have hθ : ‖θ.val‖ = 1 := mem_sphere_zero_iff_norm.mp θ.property
  rw [norm_add_sq_real, real_inner_smul_left, hy, mul_zero, mul_zero, add_zero,
    norm_smul, hθ, mul_one, Real.norm_eq_abs, sq_abs]
  rfl

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
theorem norm_hyperplane_le_point (θ : sphere (0 : V) 1) (s : ℝ) (y : Hyperplane θ) :
    ‖y‖ ≤ ‖s • θ.val + y.val‖ := by
  have heq := norm_hyperplane_point_sq θ s y
  nlinarith [sq_nonneg s, norm_nonneg y, norm_nonneg (s • θ.val + y.val)]

omit [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] in
/-- The slice of a radius-a ball is exactly the hyperplane ball of radius sqrt(a²-s²). -/
theorem hyperplane_ball_iff (θ : sphere (0 : V) 1) {s a : ℝ}
    (ha : |s| < a) (y : Hyperplane θ) :
    ‖s • θ.val + y.val‖ ≤ a ↔ ‖y‖ ≤ Real.sqrt (a^2-s^2) := by
  have ha0 : 0 < a := lt_of_le_of_lt (abs_nonneg s) ha
  have hsq : 0 ≤ a^2 - s^2 := by nlinarith [sq_abs s, abs_nonneg s]
  have hsqrt := Real.sq_sqrt hsq
  have heq := norm_hyperplane_point_sq θ s y
  have hn := Real.sqrt_nonneg (a^2-s^2)
  constructor <;> intro h <;>
    nlinarith [norm_nonneg y, norm_nonneg (s • θ.val + y.val)]

/-- The ball-mixture kernel with the radius density already restricted to its
finite support interval. -/
def mixtureIntegrand (f : V → ℂ) (b : ℝ → ℝ) (θ : sphere (0 : V) 1)
    (s : ℝ) (a : ℝ) (y : Hyperplane θ) : ℂ :=
  (b a : ℂ) * if ‖s • θ.val + y.val‖ ≤ a then f (s • θ.val + y.val) else 0

/-- Dominated Fubini is justified by an actual finite hyperplane ball and the
supplied target bound. No interchange of potentially divergent integrals is assumed. -/
theorem mixtureIntegrand_integrable {f : V → ℂ} (hf : Measurable f)
    {b : ℝ → ℝ} (hb : Integrable b) {R₀ R B : ℝ}
    (hsupp : Function.support b ⊆ Set.Icc R₀ R) (hB : 0 ≤ B)
    (hfB : ∀ x, ‖f x‖ ≤ B) (θ : sphere (0 : V) 1) (s : ℝ) :
    Integrable (Function.uncurry (mixtureIntegrand f b θ s)) (volume.prod volume) := by
  let g : Hyperplane θ → ℝ := (closedBall 0 R).indicator (fun _ => B)
  have hg : Integrable g :=
    (continuousOn_const.integrableOn_compact (isCompact_closedBall (0 : Hyperplane θ) R)).integrable_indicator
      measurableSet_closedBall
  apply (hb.norm.mul_prod hg).mono'
  · have hp : Measurable (fun z : ℝ × Hyperplane θ => f (s • θ.val + z.2.val)) :=
      hf.comp (by fun_prop)
    have hm : MeasurableSet {z : ℝ × Hyperplane θ | ‖s • θ.val + z.2.val‖ ≤ z.1} := by
      exact measurableSet_le (by fun_prop) measurable_fst
    have hindicator := (hp.aestronglyMeasurable (μ := volume.prod volume)).indicator hm
    exact (Complex.continuous_ofReal.comp_aestronglyMeasurable
      hb.aestronglyMeasurable.comp_fst).mul hindicator
  · filter_upwards with z
    rcases z with ⟨a, y⟩
    by_cases hba : b a = 0
    · simp [mixtureIntegrand, hba]
    · have ha := hsupp hba
      by_cases hy : ‖s • θ.val + y.val‖ ≤ a
      · have hyr : ‖y‖ ≤ R := (norm_hyperplane_le_point θ s y).trans (hy.trans ha.2)
        have hym : y ∈ closedBall (0 : Hyperplane θ) R := mem_closedBall_zero_iff.mpr hyr
        simp only [Function.uncurry_apply_pair, mixtureIntegrand, if_pos hy, norm_mul,
          Complex.norm_real, Real.norm_eq_abs]
        dsimp only [g]
        rw [Set.indicator_of_mem hym]
        exact mul_le_mul_of_nonneg_left (hfB _) (abs_nonneg _)
      · simp only [Function.uncurry_apply_pair, mixtureIntegrand, if_neg hy,
          mul_zero, norm_zero]
        apply mul_nonneg (norm_nonneg _)
        dsimp only [g]
        by_cases hym : y ∈ closedBall (0 : Hyperplane θ) R <;> simp [hym, hB]

/-- Multiplying the actual radial cutoff by a target value puts that value inside
the finite radius integral. -/
private theorem mul_radialCutoff (z : ℂ) (w : ℝ → ℝ) (R₀ R r : ℝ) :
    z * (radialCutoff w R₀ R r : ℂ) =
      ∫ a in Set.Icc R₀ R, (w a : ℂ) * (if r ≤ a then z else 0) := by
  unfold radialCutoff
  have ho : (∫ a in Set.Icc R₀ R, ((if r ≤ a then w a else 0 : ℝ) : ℂ)) =
      ((∫ a in Set.Icc R₀ R, if r ≤ a then w a else 0 : ℝ) : ℂ) :=
    integral_ofReal
  rw [← ho, ← integral_const_mul]
  apply integral_congr_ae
  filter_upwards with a
  split_ifs
  · ring
  · simp

/-- The unscaled finite ball-mixture cross-section formula, with dominated Fubini
proved above. This internal version takes a measurable bounded target; the local
version below obtains such a target by an explicit finite-ball restriction. -/
theorem ballMixture_crossSection_bounded {f : V → ℂ} (hf : Measurable f)
    {w : ℝ → ℝ} {R₀ R B : ℝ} (hw : IntegrableOn w (Set.Icc R₀ R))
    (hB : 0 ≤ B) (hfB : ∀ x, ‖f x‖ ≤ B) (θ : sphere (0 : V) 1) (s : ℝ) :
    hyperplaneRadon (fun x => f x * (radialCutoff w R₀ R ‖x‖ : ℂ)) θ s =
      ∫ a in Set.Icc R₀ R, (w a : ℂ) *
        ∫ y : Hyperplane θ, if ‖s • θ.val + y.val‖ ≤ a
          then f (s • θ.val + y.val) else 0 := by
  let b := (Set.Icc R₀ R).indicator w
  have hb : Integrable b := hw.integrable_indicator measurableSet_Icc
  have hsupp : Function.support b ⊆ Set.Icc R₀ R := by
    intro a ha
    by_contra hnot
    exact ha (by simp [b, hnot])
  have hi := mixtureIntegrand_integrable hf hb hsupp hB hfB θ s
  have hrow : ∀ y : Hyperplane θ,
      (∫ a in Set.Icc R₀ R, (w a : ℂ) *
        (if ‖s • θ.val + y.val‖ ≤ a then f (s • θ.val + y.val) else 0)) =
      ∫ a : ℝ, mixtureIntegrand f b θ s a y := by
    intro y
    rw [← integral_indicator measurableSet_Icc]
    apply integral_congr_ae
    filter_upwards with a
    by_cases ha : a ∈ Set.Icc R₀ R <;> simp [mixtureIntegrand, b, ha]
  unfold hyperplaneRadon
  simp_rw [mul_radialCutoff, hrow]
  rw [← integral_integral_swap hi]
  calc
    (∫ a : ℝ, ∫ y : Hyperplane θ, mixtureIntegrand f b θ s a y) =
        ∫ a : ℝ, (b a : ℂ) *
          ∫ y : Hyperplane θ, if ‖s • θ.val + y.val‖ ≤ a then f (s • θ.val + y.val) else 0 := by
      simp only [mixtureIntegrand, integral_const_mul]
    _ = _ := by
      rw [← integral_indicator measurableSet_Icc]
      apply integral_congr_ae
      filter_upwards with a
      by_cases ha : a ∈ Set.Icc R₀ R <;> simp [b, ha]

omit [InnerProductSpace ℝ V] [FiniteDimensional ℝ V] in
/-- A locally continuous target can be restricted to the outer ball to obtain a
measurable bounded function, with no condition on its arbitrary values outside. -/
private theorem measurable_restrict_ball {f : V → ℂ} {R : ℝ}
    (hf : ContinuousOn f (closedBall (0 : V) R)) :
    Measurable ((closedBall (0 : V) R).indicator f) := by
  classical
  exact hf.measurable_piecewise continuousOn_const measurableSet_closedBall

/-- Actual local-data cross-section identity. Only target continuity and a bound
on the finite outer ball are used. -/
theorem ballMixture_crossSection {f : V → ℂ} {w : ℝ → ℝ} {R₀ R B : ℝ}
    (hf : ContinuousOn f (closedBall (0 : V) R))
    (hw : IntegrableOn w (Set.Icc R₀ R)) (hB : 0 ≤ B)
    (hfB : ∀ x ∈ closedBall (0 : V) R, ‖f x‖ ≤ B)
    (θ : sphere (0 : V) 1) (s : ℝ) :
    hyperplaneRadon (fun x => f x * (radialCutoff w R₀ R ‖x‖ : ℂ)) θ s =
      ∫ a in Set.Icc R₀ R, (w a : ℂ) *
        ∫ y : Hyperplane θ, if ‖s • θ.val + y.val‖ ≤ a
          then f (s • θ.val + y.val) else 0 := by
  let f₀ := (closedBall (0 : V) R).indicator f
  have hf₀ : Measurable f₀ := measurable_restrict_ball hf
  have hbound : ∀ x, ‖f₀ x‖ ≤ B := by
    intro x
    by_cases hx : x ∈ closedBall (0 : V) R
    · simpa [f₀, hx] using hfB x hx
    · simpa [f₀, hx] using hB
  have hbase := ballMixture_crossSection_bounded hf₀ hw hB hbound θ s
  have hext : (fun x => f₀ x * (radialCutoff w R₀ R ‖x‖ : ℂ)) =
      (fun x => f x * (radialCutoff w R₀ R ‖x‖ : ℂ)) := by
    funext x
    by_cases hx : x ∈ closedBall (0 : V) R
    · simp [f₀, hx]
    · have hn : R < ‖x‖ := by simpa only [mem_closedBall_zero_iff, not_le] using hx
      simp [radialCutoff_eq_zero hn]
  rw [hext] at hbase
  rw [hbase]
  apply setIntegral_congr_fun measurableSet_Icc
  intro a ha
  dsimp only
  congr 1
  apply integral_congr_ae
  filter_upwards with y
  by_cases hy : ‖s • θ.val + y.val‖ ≤ a
  · have hp : s • θ.val + y.val ∈ closedBall (0 : V) R :=
      mem_closedBall_zero_iff.mpr (hy.trans ha.2)
    simp [f₀, hy, hp]
  · simp [hy]

omit [MeasurableSpace V] [BorelSpace V] in
/-- The transverse dimension is derived from the orthogonal decomposition. -/
theorem finrank_hyperplane (θ : sphere (0 : V) 1) :
    Module.finrank ℝ (Hyperplane θ) = Module.finrank ℝ V - 1 := by
  have hθ : θ.val ≠ 0 := by
    intro hz
    have hn := mem_sphere_zero_iff_norm.mp θ.property
    simp [hz] at hn
  have hd := Submodule.finrank_add_finrank_orthogonal (ℝ ∙ θ.val)
  rw [finrank_span_singleton hθ] at hd
  change 1 + Module.finrank ℝ (Hyperplane θ) = Module.finrank ℝ V at hd
  change Module.finrank ℝ (Hyperplane θ) = _
  omega

private theorem pos_smul_unit_closedBall {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] {r : ℝ} (hr : 0 < r) :
    r • closedBall (0 : E) 1 = closedBall (0 : E) r := by
  ext y
  constructor
  · rintro ⟨u, hu, rfl⟩
    rw [mem_closedBall_zero_iff, norm_smul, Real.norm_eq_abs, abs_of_pos hr]
    have hun : ‖u‖ ≤ 1 := mem_closedBall_zero_iff.mp hu
    nlinarith
  · intro hy
    refine ⟨r⁻¹ • y, ?_, ?_⟩
    · rw [mem_closedBall_zero_iff, norm_smul, Real.norm_eq_abs,
        abs_of_pos (inv_pos.mpr hr)]
      have hyn : ‖y‖ ≤ r := mem_closedBall_zero_iff.mp hy
      calc
        r⁻¹ * ‖y‖ ≤ r⁻¹ * r := mul_le_mul_of_nonneg_left hyn (inv_nonneg.mpr hr.le)
        _ = 1 := inv_mul_cancel₀ hr.ne'
    · dsimp only
      rw [smul_smul, mul_inv_cancel₀ hr.ne', one_smul]

/-- Actual finite hyperplane cross-section scaled to the unit ball, including
the derived `(d-1)`-dimensional Jacobian. -/
theorem ball_crossSection_scaled (f : V → ℂ) (θ : sphere (0 : V) 1)
    {s a : ℝ} (ha : |s| < a) :
    (∫ y : Hyperplane θ, if ‖s • θ.val + y.val‖ ≤ a
      then f (s • θ.val + y.val) else 0) =
      (Real.sqrt (a^2-s^2))^(Module.finrank ℝ V - 1) •
        ∫ u in closedBall (0 : Hyperplane θ) 1,
          f (s • θ.val + (Real.sqrt (a^2-s^2)) • u.val) := by
  classical
  let r := Real.sqrt (a^2-s^2)
  have hr : 0 < r := by
    apply Real.sqrt_pos_of_pos
    have ha0 : 0 < a := lt_of_le_of_lt (abs_nonneg s) ha
    nlinarith [sq_abs s, abs_nonneg s]
  have hind :
      (fun y : Hyperplane θ => if ‖s • θ.val + y.val‖ ≤ a then f (s • θ.val + y.val) else 0) =
      (closedBall (0 : Hyperplane θ) r).indicator (fun y => f (s • θ.val + y.val)) := by
    funext y
    rw [Set.indicator_apply]
    congr 1
    exact propext ((hyperplane_ball_iff θ ha y).trans mem_closedBall_zero_iff.symm)
  rw [hind, integral_indicator measurableSet_closedBall]
  have hscale := Measure.setIntegral_comp_smul_of_pos (volume : Measure (Hyperplane θ))
    (fun y : Hyperplane θ => f (s • θ.val + y.val)) (closedBall 0 1) hr
  rw [pos_smul_unit_closedBall hr] at hscale
  have heq := congrArg (fun z : ℂ => r^(Module.finrank ℝ (Hyperplane θ)) • z) hscale
  rw [smul_smul, mul_inv_cancel₀ (pow_ne_zero _ hr.ne'), one_smul] at heq
  rw [← heq, finrank_hyperplane]
  rfl

/-- The finite ball-mixture formula used for analytic continuation. It is derived
from actual hyperplane integration and local target data, not assumed. -/
theorem ballMixture_unitBall_formula {f : V → ℂ} {w : ℝ → ℝ} {R₀ R B s : ℝ}
    (hf : ContinuousOn f (closedBall (0 : V) R))
    (hw : IntegrableOn w (Set.Icc R₀ R)) (hB : 0 ≤ B)
    (hfB : ∀ x ∈ closedBall (0 : V) R, ‖f x‖ ≤ B)
    (θ : sphere (0 : V) 1) (hs : |s| < R₀) :
    hyperplaneRadon (fun x => f x * (radialCutoff w R₀ R ‖x‖ : ℂ)) θ s =
      ∫ a in Set.Icc R₀ R, (w a : ℂ) *
        ((Real.sqrt (a^2-s^2))^(Module.finrank ℝ V - 1) •
          ∫ u in closedBall (0 : Hyperplane θ) 1,
            f (s • θ.val + (Real.sqrt (a^2-s^2)) • u.val)) := by
  rw [ballMixture_crossSection hf hw hB hfB θ s]
  apply setIntegral_congr_fun measurableSet_Icc
  intro a ha
  dsimp only
  rw [ball_crossSection_scaled f θ (hs.trans_le ha.1)]

end QIRadon.Radon
