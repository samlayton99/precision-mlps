import QIRadon.Kernel.Certificate
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals

/-! The Abel-regularized Hilbert multiplier, evaluated by actual half-line
exponential integrals. Frequency is in cycles; the angular scale is 2πx. -/

noncomputable section
open MeasureTheory Set Filter
open scoped Topology
open scoped FourierTransform

namespace QIRadon.Radon

/-- Abel damping of the Hilbert Fourier multiplier; the value at zero is zero. -/
def hilbertAbelMultiplier (a ξ : ℝ) : ℂ :=
  -Complex.I * (Real.sign ξ : ℂ) * Complex.exp (-((a * |ξ| : ℝ) : ℂ))

private theorem hilbertAbelMultiplier_pos (a : ℝ) {ξ : ℝ} (hξ : 0 < ξ) :
    hilbertAbelMultiplier a ξ = -Complex.I * Complex.exp ((-a : ℂ) * ξ) := by
  simp [hilbertAbelMultiplier, Real.sign_of_pos hξ, abs_of_pos hξ]

private theorem hilbertAbelMultiplier_neg (a : ℝ) {ξ : ℝ} (hξ : ξ < 0) :
    hilbertAbelMultiplier a ξ = Complex.I * Complex.exp ((a : ℂ) * ξ) := by
  simp [hilbertAbelMultiplier, Real.sign_of_neg hξ, abs_of_neg hξ]

/-- The damped sign multiplier is genuinely integrable for every positive damping. -/
theorem hilbertAbelMultiplier_integrable {a : ℝ} (ha : 0 < a) :
    Integrable (hilbertAbelMultiplier a) := by
  have hp : IntegrableOn (hilbertAbelMultiplier a) (Set.Ioi 0) := by
    have he : IntegrableOn (fun ξ : ℝ => -Complex.I * Complex.exp ((-a : ℂ)*ξ)) (Set.Ioi 0) := (integrableOn_exp_mul_complex_Ioi (a := (-a : ℂ)) (by simpa using neg_neg_of_pos ha) 0).const_mul
      (-Complex.I)
    apply he.congr_fun _ measurableSet_Ioi
    intro ξ hξ
    exact (hilbertAbelMultiplier_pos a hξ).symm
  have hn : IntegrableOn (hilbertAbelMultiplier a) (Set.Iio 0) := by
    have he : IntegrableOn (fun ξ : ℝ => Complex.I * Complex.exp ((a : ℂ)*ξ)) (Set.Iic 0) := (integrableOn_exp_mul_complex_Iic (a := (a : ℂ)) (by simpa using ha) 0).const_mul
      Complex.I
    have hei : IntegrableOn (fun ξ : ℝ => Complex.I * Complex.exp ((a : ℂ)*ξ)) (Set.Iio 0) :=
      he.mono_set Set.Iio_subset_Iic_self
    apply hei.congr_fun _ measurableSet_Iio
    intro ξ hξ
    exact (hilbertAbelMultiplier_neg a hξ).symm
  have hnc : IntegrableOn (hilbertAbelMultiplier a) (Set.Iic 0) :=
    (integrableOn_Iic_iff_integrableOn_Iio).mpr hn
  have hall := hnc.union hp
  simpa only [Set.Iic_union_Ioi, integrableOn_univ] using hall

private theorem phase_abel_pos (a x : ℝ) {ξ : ℝ} (hξ : 0 < ξ) :
    QIRadon.Kernel.phase (ξ*x) * hilbertAbelMultiplier a ξ =
      -Complex.I * Complex.exp (((-a : ℂ)+(2*Real.pi*x : ℂ)*Complex.I) * ξ) := by
  rw [hilbertAbelMultiplier_pos a hξ]
  simp only [QIRadon.Kernel.phase, Real.fourierChar_apply]
  calc
    _ = -Complex.I * (Complex.exp ((2*Real.pi*(ξ*x) : ℝ)*Complex.I) *
        Complex.exp ((-a : ℂ)*ξ)) := by ring
    _ = -Complex.I * Complex.exp (((2*Real.pi*(ξ*x) : ℝ) : ℂ)*Complex.I + (-a : ℂ)*ξ) := by
      rw [Complex.exp_add]
    _ = _ := by
      congr 1
      push_cast
      ring_nf

private theorem phase_abel_neg (a x : ℝ) {ξ : ℝ} (hξ : ξ < 0) :
    QIRadon.Kernel.phase (ξ*x) * hilbertAbelMultiplier a ξ =
      Complex.I * Complex.exp (((a : ℂ)+(2*Real.pi*x : ℂ)*Complex.I) * ξ) := by
  rw [hilbertAbelMultiplier_neg a hξ]
  simp only [QIRadon.Kernel.phase, Real.fourierChar_apply]
  calc
    _ = Complex.I * (Complex.exp ((2*Real.pi*(ξ*x) : ℝ)*Complex.I) *
        Complex.exp ((a : ℂ)*ξ)) := by ring
    _ = Complex.I * Complex.exp (((2*Real.pi*(ξ*x) : ℝ) : ℂ)*Complex.I + (a : ℂ)*ξ) := by
      rw [Complex.exp_add]
    _ = _ := by
      congr 1
      push_cast
      ring_nf

/-- Exact inverse Fourier transform of the Abel-damped Hilbert multiplier.
The identity is derived from two convergent half-line exponential integrals. -/
theorem synthesize_hilbertAbelMultiplier {a : ℝ} (ha : 0 < a) (x : ℝ) :
    QIRadon.Kernel.synthesize (hilbertAbelMultiplier a) x =
      ((4*Real.pi*x) / (a^2 + (2*Real.pi*x)^2) : ℂ) := by
  let cp : ℂ := (-a : ℂ)+(2*Real.pi*x : ℂ)*Complex.I
  let cm : ℂ := (a : ℂ)+(2*Real.pi*x : ℂ)*Complex.I
  have hcp : cp.re < 0 := by dsimp [cp]; simpa using neg_neg_of_pos ha
  have hcm : 0 < cm.re := by dsimp [cm]; simpa using ha
  have hp : (∫ ξ in Set.Ioi (0 : ℝ),
      QIRadon.Kernel.phase (ξ*x) * hilbertAbelMultiplier a ξ) = Complex.I / cp := by
    calc
      _ = ∫ ξ in Set.Ioi (0 : ℝ), -Complex.I * Complex.exp (cp*ξ) := by
        apply setIntegral_congr_fun measurableSet_Ioi
        intro ξ hξ
        exact phase_abel_pos a x hξ
      _ = _ := by
        rw [integral_const_mul, integral_exp_mul_complex_Ioi hcp]
        simp [div_eq_mul_inv]
  have hn : (∫ ξ in Set.Iic (0 : ℝ),
      QIRadon.Kernel.phase (ξ*x) * hilbertAbelMultiplier a ξ) = Complex.I / cm := by
    calc
      _ = ∫ ξ in Set.Iic (0 : ℝ), Complex.I * Complex.exp (cm*ξ) := by
        apply integral_congr_ae
        filter_upwards [ae_restrict_mem measurableSet_Iic,
          (volume.restrict (Set.Iic (0 : ℝ))).ae_ne 0] with ξ hξ hne
        exact phase_abel_neg a x (lt_of_le_of_ne hξ hne)
      _ = _ := by
        rw [integral_const_mul, integral_exp_mul_complex_Iic hcm]
        simp [div_eq_mul_inv]
  have hi : Integrable (fun ξ : ℝ =>
      QIRadon.Kernel.phase (ξ*x) * hilbertAbelMultiplier a ξ) := by
    have hb := hilbertAbelMultiplier_integrable ha
    apply hb.norm.mono'
    · exact (by unfold QIRadon.Kernel.phase; fun_prop : Continuous (fun ξ : ℝ => QIRadon.Kernel.phase (ξ*x))).aestronglyMeasurable.mul
        hb.aestronglyMeasurable
    · filter_upwards with ξ
      simp only [norm_mul, QIRadon.Kernel.norm_phase, one_mul, le_refl]
  rw [QIRadon.Kernel.synthesize_eq_integral,
    ← integral_add_compl (s := Set.Ioi (0 : ℝ)) measurableSet_Ioi hi, Set.compl_Ioi,
    hp, hn]
  have hcp : cp ≠ 0 := by intro h; simp [h] at hcp
  have hcm : cm ≠ 0 := by intro h; simp [h] at hcm
  have hden : (a^2 + (2*Real.pi*x)^2 : ℂ) ≠ 0 := by
    have hreal : (0 : ℝ) < a^2+(2*Real.pi*x)^2 := by nlinarith [sq_nonneg (2*Real.pi*x)]
    exact_mod_cast hreal.ne'
  apply (eq_div_iff hden).mpr
  field_simp [hcp, hcm]
  dsimp [cp, cm]
  ring_nf
  simp [Complex.I_sq]
  ring

/-- Abel regularization acts by an explicit physical convolution, with no PDE or
auxiliary inverse problem hidden in the definition. -/
theorem hilbertAbel_convolution {a : ℝ} (ha : 0 < a)
    {p : ℝ → ℂ} (hp : Integrable p) (s : ℝ) :
    QIRadon.Kernel.synthesize (fun ξ => hilbertAbelMultiplier a ξ * 𝓕 p ξ) s =
      ∫ t : ℝ, ((4*Real.pi*(s-t)) / (a^2 + (2*Real.pi*(s-t))^2) : ℂ) * p t := by
  rw [← QIRadon.Kernel.synthesize_convolve (hilbertAbelMultiplier_integrable ha) hp]
  calc
    _ = ∫ t : ℝ, QIRadon.Kernel.synthesize (hilbertAbelMultiplier a) (s-t) * p t := by
      simpa only [sub_sub_cancel] using
        (integral_sub_left_eq_self (fun t : ℝ =>
          QIRadon.Kernel.synthesize (hilbertAbelMultiplier a) (s-t) * p t) volume s)
    _ = _ := by simp only [synthesize_hilbertAbelMultiplier ha, Complex.ofReal_sub]

/-- Pointwise multiplier domination needed for removing the Abel damping. -/
theorem norm_hilbertAbelMultiplier_le_one {a : ℝ} (ha : 0 ≤ a) (ξ : ℝ) :
    ‖hilbertAbelMultiplier a ξ‖ ≤ 1 := by
  rcases lt_trichotomy ξ 0 with hξ | hξ | hξ
  · rw [hilbertAbelMultiplier_neg a hξ]
    simp only [norm_mul, Complex.norm_I, one_mul, Complex.norm_exp]
    simp only [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, mul_zero, sub_zero]
    exact Real.exp_le_one_iff.mpr (mul_nonpos_of_nonneg_of_nonpos ha hξ.le)
  · simp [hξ, hilbertAbelMultiplier]
  · rw [hilbertAbelMultiplier_pos a hξ]
    simp only [norm_mul, norm_neg, Complex.norm_I, one_mul, Complex.norm_exp]
    simp only [Complex.mul_re, Complex.neg_re, Complex.ofReal_re,
      Complex.neg_im, Complex.ofReal_im, neg_zero, mul_zero, sub_zero]
    exact Real.exp_le_one_iff.mpr (mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr ha) hξ.le)

/-- The genuine Fourier Hilbert transform for an integrable spectrum. -/
def hilbertFourier (p : ℝ → ℂ) (s : ℝ) : ℂ :=
  QIRadon.Kernel.synthesize (fun ξ => -Complex.I*(Real.sign ξ:ℂ)*𝓕 p ξ) s

/-- Removing positive Abel damping converges to the canonical sign multiplier.
Only integrability of the target spectrum is needed for this spectral limit. -/
theorem hilbertAbel_tendsto {p : ℝ → ℂ} (hp : Integrable (𝓕 p))
    {a : ℕ → ℝ} (ha : ∀ n, 0 < a n) (ha0 : Tendsto a atTop (𝓝 0)) (s : ℝ) :
    Tendsto (fun n => QIRadon.Kernel.synthesize
      (fun ξ => hilbertAbelMultiplier (a n) ξ * 𝓕 p ξ) s)
      atTop (𝓝 (hilbertFourier p s)) := by
  simp only [QIRadon.Kernel.synthesize_eq_integral, hilbertFourier]
  apply tendsto_integral_of_dominated_convergence (fun ξ => ‖𝓕 p ξ‖)
  · intro n
    exact ((by unfold QIRadon.Kernel.phase; fun_prop :
        Continuous (fun ξ : ℝ => QIRadon.Kernel.phase (ξ*s))).aestronglyMeasurable.mul
      ((hilbertAbelMultiplier_integrable (ha n)).aestronglyMeasurable.mul hp.aestronglyMeasurable))
  · exact hp.norm
  · intro n
    filter_upwards with ξ
    simp only [norm_mul, QIRadon.Kernel.norm_phase, one_mul]
    exact mul_le_of_le_one_left (norm_nonneg _) (norm_hilbertAbelMultiplier_le_one (ha n).le ξ)
  · filter_upwards with ξ
    have he : Tendsto (fun n => Complex.exp (-((a n*|ξ|:ℝ):ℂ))) atTop (𝓝 1) := by
      have hc : Tendsto (fun n => -((a n*|ξ|:ℝ):ℂ)) atTop (𝓝 (0:ℂ)) := by
        have hr : Tendsto (fun n => a n*|ξ|) atTop (𝓝 0) := by simpa using ha0.mul_const |ξ|
        simpa [Function.comp_def] using ((Complex.continuous_ofReal.tendsto 0).comp hr).neg
      simpa [Function.comp_def] using (Complex.continuous_exp.tendsto (0:ℂ)).comp hc
    have ht := he.const_mul (-Complex.I*(Real.sign ξ:ℂ))
    have ht2 := ht.mul_const (𝓕 p ξ)
    have ht3 := ht2.const_mul (QIRadon.Kernel.phase (ξ*s))
    simpa [hilbertAbelMultiplier] using ht3

end QIRadon.Radon
