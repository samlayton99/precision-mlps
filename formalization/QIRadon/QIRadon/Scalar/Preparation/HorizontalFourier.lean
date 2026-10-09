import Mathlib.Analysis.Complex.CauchyIntegral
import Mathlib.Analysis.Fourier.FourierTransform
import Mathlib.MeasureTheory.Integral.Gamma
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.MeasureTheory.Integral.IntegralEqImproper
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.LinearCombination

/-! Full-line Fourier decay from an entire Gaussian envelope on horizontal
lines. Contour translation is proved by finite rectangles and vanishing
vertical sides. -/

namespace QIRadon.ProfilePreparation

open Set MeasureTheory Filter
open scoped Interval

/-- The angular-frequency Fourier transform used in the paper. -/
noncomputable def angularFourier (G : ℂ → ℂ) (ω : ℝ) : ℂ :=
  ∫ x : ℝ, G (x : ℂ) * Complex.exp (-(ω : ℂ) * Complex.I * (x : ℂ))

/-- Continuous frequencies have the same shifted-line modulus identity. -/
theorem norm_fourierPhase (ω x y : ℝ) :
    ‖Complex.exp (-(ω : ℂ) * Complex.I * ((x : ℂ) + y * Complex.I))‖ =
      Real.exp (ω * y) := by
  rw [Complex.norm_exp]
  congr 1
  simp [Complex.mul_re, Complex.mul_im]

/-- Every horizontal restriction is absolutely integrable under its actual
Gaussian envelope. -/
theorem gaussian_horizontal_integrable {F : ℂ → ℂ} {C a δ y : ℝ}
    (hF : Differentiable ℂ F) (_hC : 0 ≤ C) (ha : 0 < a) (hy : |y| ≤ δ)
    (hbound : ∀ x t : ℝ, |t| ≤ δ → ‖F ((x : ℂ) + t * Complex.I)‖ ≤ C * Real.exp (-a * x ^ 2)) :
    Integrable (fun x : ℝ => F ((x : ℂ) + y * Complex.I)) := by
  apply ((integrable_exp_neg_mul_sq ha).const_mul C).mono'
  · exact (hF.continuous.comp (Complex.continuous_ofReal.add continuous_const)).aestronglyMeasurable
  · exact Eventually.of_forall (fun x => hbound x y hy)

/-- The finite vertical side has exponentially small norm as its real
coordinate moves to infinity. -/
theorem gaussian_vertical_integral_bound {F : ℂ → ℂ} {C a δ y : ℝ}
    (hδ : 0 ≤ δ) (hy : |y| ≤ δ)
    (hbound : ∀ x t : ℝ, |t| ≤ δ → ‖F ((x : ℂ) + t * Complex.I)‖ ≤ C * Real.exp (-a * x ^ 2))
    (R : ℝ) :
    ‖∫ t in (0 : ℝ)..y, F ((R : ℂ) + t * Complex.I)‖ ≤
      C * Real.exp (-a * R ^ 2) * |y| := by
  have ht : ∀ t ∈ Ι (0 : ℝ) y, |t| ≤ δ := by
    intro t ht
    have hy' := abs_le.mp hy
    apply abs_le.mpr
    constructor
    · exact (le_min (by linarith) hy'.1).trans ht.1.le
    · exact ht.2.trans (max_le hδ hy'.2)
  simpa only [sub_zero] using intervalIntegral.norm_integral_le_of_norm_le_const
    (fun t ht' => hbound R t (ht t ht'))

/-- Full-line horizontal contour shift justified by an actual Gaussian decay
bound, with no unproved contour-at-infinity premise. -/
theorem gaussian_horizontal_integral_shift {F : ℂ → ℂ} {C a δ y : ℝ}
    (hF : Differentiable ℂ F) (hC : 0 ≤ C) (ha : 0 < a)
    (hδ : 0 ≤ δ) (hy : |y| ≤ δ)
    (hbound : ∀ x t : ℝ, |t| ≤ δ → ‖F ((x : ℂ) + t * Complex.I)‖ ≤ C * Real.exp (-a * x ^ 2)) :
    (∫ x : ℝ, F (x : ℂ)) = ∫ x : ℝ, F ((x : ℂ) + y * Complex.I) := by
  have hfinite (R : ℝ) :
      ‖(∫ x in -R..R, F (x : ℂ)) -
        (∫ x in -R..R, F ((x : ℂ) + y * Complex.I))‖ ≤
        (2 * C * |y|) * Real.exp (-a * R ^ 2) := by
    have hh := Complex.integral_boundary_rect_eq_zero_of_differentiableOn F
      (Complex.mk (-R) 0) (Complex.mk R y) hF.differentiableOn
    dsimp only at hh
    simp only [Complex.ofReal_zero, zero_mul, add_zero] at hh
    have he : (∫ x in -R..R, F (x : ℂ)) -
        (∫ x in -R..R, F ((x : ℂ) + y * Complex.I)) =
        Complex.I * ((∫ t in (0 : ℝ)..y, F (((-R : ℝ) : ℂ) + t * Complex.I)) -
          (∫ t in (0 : ℝ)..y, F ((R : ℂ) + t * Complex.I))) := by
      simpa only [smul_eq_mul] using (show _ = _ by linear_combination hh)
    rw [he, norm_mul, Complex.norm_I, one_mul]
    have hleft := gaussian_vertical_integral_bound hδ hy hbound (-R)
    have hright := gaussian_vertical_integral_bound hδ hy hbound R
    rw [neg_sq] at hleft
    have htri := norm_sub_le
      (∫ t in (0 : ℝ)..y, F (((-R : ℝ) : ℂ) + t * Complex.I))
      (∫ t in (0 : ℝ)..y, F ((R : ℂ) + t * Complex.I))
    nlinarith
  have hi0 : Integrable (fun x : ℝ => F (x : ℂ)) := by
    simpa only [Complex.ofReal_zero, zero_mul, add_zero] using
      gaussian_horizontal_integrable hF hC ha (show |(0 : ℝ)| ≤ δ by simpa) hbound
  have hiy := gaussian_horizontal_integrable hF hC ha hy hbound
  have h0lim := intervalIntegral_tendsto_integral hi0
    (tendsto_neg_atTop_atBot : Tendsto (fun R : ℝ => -R) atTop atBot)
    (tendsto_id : Tendsto (fun R : ℝ => R) atTop atTop)
  have hylim := intervalIntegral_tendsto_integral hiy
    (tendsto_neg_atTop_atBot : Tendsto (fun R : ℝ => -R) atTop atBot)
    (tendsto_id : Tendsto (fun R : ℝ => R) atTop atTop)
  have hpow : Tendsto (fun R : ℝ => R ^ 2) atTop atTop := by
    exact tendsto_pow_atTop (by decide)
  have hexp := Real.tendsto_exp_atBot.comp (hpow.const_mul_atTop_of_neg (neg_neg_of_pos ha))
  have hzero : Tendsto (fun R : ℝ => (2 * C * |y|) * Real.exp (-a * R ^ 2)) atTop (nhds 0) := by
    simpa using hexp.const_mul (2 * C * |y|)
  have hnormzero := squeeze_zero (fun R => norm_nonneg _) hfinite hzero
  have hid := tendsto_nhds_unique (h0lim.sub hylim).norm hnormzero
  exact sub_eq_zero.mp (norm_eq_zero.mp hid)

/-- Exponential decay of the actual angular-frequency Fourier transform. -/
theorem angularFourier_gaussian_strip_bound {G : ℂ → ℂ} {C a δ : ℝ}
    (hG : Differentiable ℂ G) (hC : 0 ≤ C) (ha : 0 < a) (hδ : 0 ≤ δ)
    (hbound : ∀ x y : ℝ, |y| ≤ δ → ‖G ((x : ℂ) + y * Complex.I)‖ ≤ C * Real.exp (-a * x ^ 2))
    (ω : ℝ) :
    ‖angularFourier G ω‖ ≤ C * Real.sqrt (Real.pi / a) * Real.exp (-δ * |ω|) := by
  let y : ℝ := if 0 ≤ ω then -δ else δ
  have hy : |y| ≤ δ := by dsimp [y]; split_ifs <;> simp [abs_of_nonneg hδ]
  have hωy : ω * y = -δ * |ω| := by
    dsimp [y]
    split_ifs with hω
    · rw [abs_of_nonneg hω]; ring
    · rw [abs_of_neg (lt_of_not_ge hω)]; ring
  let F : ℂ → ℂ := fun z => G z * Complex.exp (-(ω : ℂ) * Complex.I * z)
  have hF : Differentiable ℂ F := by dsimp [F]; fun_prop
  have hFb : ∀ x t : ℝ, |t| ≤ δ →
      ‖F ((x : ℂ) + t * Complex.I)‖ ≤
        (C * Real.exp (|ω| * δ)) * Real.exp (-a * x ^ 2) := by
    intro x t ht
    dsimp [F]
    rw [norm_mul, norm_fourierPhase]
    have he : Real.exp (ω * t) ≤ Real.exp (|ω| * δ) := by
      apply Real.exp_le_exp.mpr
      calc
        ω * t ≤ |ω * t| := le_abs_self _
        _ = |ω| * |t| := abs_mul _ _
        _ ≤ |ω| * δ := mul_le_mul_of_nonneg_left ht (abs_nonneg _)
    have hm := mul_le_mul (hbound x t ht) he (Real.exp_pos _).le (by positivity)
    convert! hm using 1
    ring
  have hshift := gaussian_horizontal_integral_shift hF (by positivity) ha hδ hy hFb
  unfold angularFourier
  change ‖∫ x : ℝ, F (x : ℂ)‖ ≤ _
  rw [hshift]
  have hL1 : ‖∫ x : ℝ, F ((x : ℂ) + y * Complex.I)‖ ≤
      ∫ x : ℝ, (C * Real.exp (ω * y)) * Real.exp (-a * x ^ 2) := by
    apply norm_integral_le_of_norm_le ((integrable_exp_neg_mul_sq ha).const_mul _)
    apply Eventually.of_forall
    intro x
    dsimp [F]
    rw [norm_mul, norm_fourierPhase]
    have hm := mul_le_mul_of_nonneg_right (hbound x y hy) (Real.exp_pos (ω * y)).le
    convert! hm using 1
    ring
  apply hL1.trans_eq
  rw [integral_const_mul, integral_gaussian, hωy]
  ring


open scoped FourierTransform

/-- Explicit connector: angular frequency omega equals cycles frequency
omega/(2*pi). This prevents mixing the two normalizations downstream. -/
theorem angularFourier_eq_cycles (G : ℂ → ℂ) (ω : ℝ) :
    angularFourier G ω = 𝓕 (fun x : ℝ => G (x : ℂ)) (ω / (2 * Real.pi)) := by
  rw [Real.fourier_real_eq_integral_exp_smul]
  unfold angularFourier
  apply integral_congr_ae
  apply Eventually.of_forall
  intro x
  have he : (((-2 * Real.pi * x * (ω / (2 * Real.pi))) : ℝ) : ℂ) * Complex.I =
      -(ω : ℂ) * Complex.I * (x : ℂ) := by
    push_cast
    field_simp
  dsimp only
  rw [he]
  simp only [smul_eq_mul, mul_comm]

theorem angularFourier_continuous {G : ℂ → ℂ}
    (hG : Integrable (fun x : ℝ => G (x : ℂ))) : Continuous (angularFourier G) := by
  have hc : Continuous (𝓕 (fun x : ℝ => G (x : ℂ))) := by
    change Continuous (VectorFourier.fourierIntegral Real.fourierChar volume (innerₗ ℝ)
      (fun x : ℝ => G (x : ℂ)))
    exact VectorFourier.fourierIntegral_continuous Real.continuous_fourierChar (by fun_prop) hG
  have he : angularFourier G = fun ω => 𝓕 (fun x : ℝ => G (x : ℂ)) (ω / (2 * Real.pi)) :=
    funext (angularFourier_eq_cycles G)
  rw [he]
  exact hc.comp (continuous_id.div_const _)

/-- The explicit polynomial-exponential integrable majorant needed for every
fixed derivative order r. -/
theorem absolute_exponential_moment (r : ℕ) {c : ℝ} (hc : 0 < c) :
    Integrable (fun x : ℝ => |x| ^ r * Real.exp (-c * |x|)) ∧
      (∫ x : ℝ, |x| ^ r * Real.exp (-c * |x|)) =
        2 * (r.factorial : ℝ) / c ^ (r + 1) := by
  let f : ℝ → ℝ := fun x => |x| ^ r * Real.exp (-c * |x|)
  have hp : IntegrableOn f (Ioi 0) := by
    have hh := integrableOn_rpow_mul_exp_neg_mul_rpow (p := 1) (s := (r : ℝ))
      (lt_of_lt_of_le (by norm_num : (-1 : ℝ) < 0) (Nat.cast_nonneg r)) (by norm_num) hc
    apply hh.congr_fun _ measurableSet_Ioi
    intro x hx
    simp only [f, abs_of_pos (show 0 < x from hx), Real.rpow_one, Real.rpow_natCast]
  have hn : IntegrableOn f (Iio 0) := by
    rw [← (Measure.measurePreserving_neg (volume : Measure ℝ)).integrableOn_comp_preimage
      (Homeomorph.neg ℝ).measurableEmbedding]
    simpa only [Function.comp_def, neg_preimage, neg_Iio, neg_zero, f, abs_neg] using hp
  have hf : Integrable f := by
    rw [← integrableOn_univ, ← Iio_union_Ici (a := (0 : ℝ)), integrableOn_union]
    exact ⟨hn, (integrableOn_Ici_iff_integrableOn_Ioi).mpr hp⟩
  have hposval : (∫ x in Ioi (0 : ℝ), f x) = (r.factorial : ℝ) / c ^ (r + 1) := by
    have he := integral_rpow_mul_exp_neg_mul_rpow (p := 1) (q := (r : ℝ))
      (by norm_num) (lt_of_lt_of_le (by norm_num : (-1 : ℝ) < 0) (Nat.cast_nonneg r)) hc
    have hleft : (∫ x in Ioi (0 : ℝ), f x) =
        ∫ x in Ioi (0 : ℝ), x ^ (r : ℝ) * Real.exp (-c * x ^ (1 : ℝ)) := by
      apply setIntegral_congr_fun measurableSet_Ioi
      intro x hx
      simp only [f, abs_of_pos (show 0 < x from hx), Real.rpow_one, Real.rpow_natCast]
    rw [hleft, he]
    simp only [div_one, mul_one, Real.Gamma_nat_eq_factorial]
    rw [show -((r : ℝ) + 1) = -((r + 1 : ℕ) : ℝ) by push_cast; rfl,
      Real.rpow_neg hc.le, Real.rpow_natCast]
    ring
  have hnegval : (∫ x in Iio (0 : ℝ), f x) = ∫ x in Ioi (0 : ℝ), f x := by
    have he := (Measure.measurePreserving_neg (volume : Measure ℝ)).setIntegral_preimage_emb
      (Homeomorph.neg ℝ).measurableEmbedding f (Ioi 0)
    simpa only [neg_preimage, neg_Ioi, neg_zero, f, abs_neg] using he
  refine ⟨hf, ?_⟩
  have hsplit := integral_add_compl (s := Ioi (0 : ℝ)) measurableSet_Ioi hf
  rw [compl_Ioi, integral_Iic_eq_integral_Iio, hnegval, hposval] at hsplit
  dsimp [f] at hsplit
  calc
    _ = (r.factorial : ℝ) / c ^ (r + 1) + (r.factorial : ℝ) / c ^ (r + 1) := hsplit.symm
    _ = _ := by ring

/-- Every polynomially weighted exponential Fourier moment is finite and
quantitatively bounded, derived from the actual strip Fourier-decay theorem. -/
theorem angularFourier_weighted_moment {G : ℂ → ℂ} {C a δ σ : ℝ}
    (hG : Differentiable ℂ G) (hC : 0 ≤ C) (ha : 0 < a)
    (hδ : 0 ≤ δ) (hσ : σ < δ)
    (hbound : ∀ x y : ℝ, |y| ≤ δ → ‖G ((x : ℂ) + y * Complex.I)‖ ≤ C * Real.exp (-a * x ^ 2))
    (r : ℕ) :
    Integrable (fun ω : ℝ => Real.exp (σ * |ω|) * |ω| ^ r * ‖angularFourier G ω‖) ∧
    (∫ ω : ℝ, Real.exp (σ * |ω|) * |ω| ^ r * ‖angularFourier G ω‖) ≤
      (C * Real.sqrt (Real.pi / a)) *
        (2 * (r.factorial : ℝ) / (δ - σ) ^ (r + 1)) := by
  have hc : 0 < δ - σ := sub_pos.mpr hσ
  obtain ⟨hmajor, hmajorval⟩ := absolute_exponential_moment r hc
  let D : ℝ := C * Real.sqrt (Real.pi / a)
  have hD : 0 ≤ D := by dsimp [D]; positivity
  have hb : ∀ ω : ℝ,
      Real.exp (σ * |ω|) * |ω| ^ r * ‖angularFourier G ω‖ ≤
        D * (|ω| ^ r * Real.exp (-(δ - σ) * |ω|)) := by
    intro ω
    have hh := mul_le_mul_of_nonneg_left (angularFourier_gaussian_strip_bound hG hC ha hδ hbound ω)
      (show 0 ≤ Real.exp (σ * |ω|) * |ω| ^ r by positivity)
    have hexp : Real.exp (σ * |ω|) * Real.exp (-δ * |ω|) =
        Real.exp (-(δ - σ) * |ω|) := by
      rw [← Real.exp_add]
      congr 1
      ring
    convert! hh using 1
    calc
      D * (|ω| ^ r * Real.exp (-(δ - σ) * |ω|)) =
          D * |ω| ^ r * (Real.exp (σ * |ω|) * Real.exp (-δ * |ω|)) := by rw [hexp]; ring
      _ = _ := by dsimp [D]; ring
  have hreal : Integrable (fun x : ℝ => G (x : ℂ)) := by
    simpa only [Complex.ofReal_zero, zero_mul, add_zero] using gaussian_horizontal_integrable hG hC ha
      (show |(0 : ℝ)| ≤ δ by simpa) hbound
  have hcont : Continuous (fun ω : ℝ => Real.exp (σ * |ω|) * |ω| ^ r * ‖angularFourier G ω‖) := by
    have hft := angularFourier_continuous hreal
    fun_prop
  have hmaj := hmajor.const_mul D
  have hint : Integrable (fun ω : ℝ => Real.exp (σ * |ω|) * |ω| ^ r * ‖angularFourier G ω‖) := by
    apply hmaj.mono' hcont.aestronglyMeasurable
    apply Eventually.of_forall
    intro ω
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    exact hb ω
  refine ⟨hint, ?_⟩
  calc
    _ ≤ ∫ ω : ℝ, D * (|ω| ^ r * Real.exp (-(δ - σ) * |ω|)) := integral_mono hint hmaj hb
    _ = _ := by rw [integral_const_mul, hmajorval]

end QIRadon.ProfilePreparation
