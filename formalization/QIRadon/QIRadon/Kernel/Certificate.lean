import QIRadon.Scalar.Anchoring
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Fourier.Convolution
import Mathlib.Analysis.Fourier.PoissonSummation
import Mathlib.Analysis.Fourier.Inversion
import Mathlib.MeasureTheory.Measure.Haar.NormedSpace

/-!
# General-kernel analytic components

Frequency `ξ` is measured in cycles, following mathlib's transform
`𝓕 f ξ = ∫ x, exp (-2π i x ξ) f x`. The paper's angular frequency is
`ω = 2π ξ`. No special activation formula is used here.

These are proved analytic components of the approved finite-kernel theorem.
The full sharp-cutoff density / Poisson / halo construction is not asserted
until its hypotheses are derived for that construction.
-/

noncomputable section

open MeasureTheory Set FourierTransform
open scoped Real SchwartzMap

namespace QIRadon.Kernel

/-- Inverse spectral synthesis in the cycles-frequency normalization. -/
def synthesize (b : ℝ → ℂ) (x : ℝ) : ℂ := 𝓕⁻ b x

/-- Retain precisely the closed frequency band `[-B,B]`. -/
def bandSpectrum (b : ℝ → ℂ) (B : ℝ) : ℝ → ℂ :=
  (Set.Icc (-B) B).indicator b

/-- The finite-band inverse-kernel density. `Khat` is the kernel transform;
its nonvanishing on the band is an obligation of the construction. -/
def cutoffDensity (pHat Khat : ℝ → ℂ) (B : ℝ) (x : ℝ) : ℂ :=
  synthesize (bandSpectrum (fun ξ => pHat ξ / Khat ξ) B) x

/-- Fourier synthesis is uniformly bounded by the spectral L¹ norm. -/
theorem norm_synthesize_le (b : ℝ → ℂ) (x : ℝ) :
    ‖synthesize b x‖ ≤ ∫ ξ : ℝ, ‖b ξ‖ := by
  unfold synthesize
  rw [Real.fourierInv_eq]
  calc
    ‖∫ ξ : ℝ, 𝐞 (inner ℝ ξ x) • b ξ‖ ≤
        ∫ ξ : ℝ, ‖𝐞 (inner ℝ ξ x) • b ξ‖ := norm_integral_le_integral_norm _
    _ = ∫ ξ : ℝ, ‖b ξ‖ := by simp only [Circle.norm_smul]

/-- Spectral L¹ perturbations give pointwise synthesis perturbations. -/
theorem norm_synthesize_sub_le {b₁ b₂ : ℝ → ℂ}
    (hb₁ : Integrable b₁) (hb₂ : Integrable b₂) (x : ℝ) :
    ‖synthesize b₁ x - synthesize b₂ x‖ ≤ ∫ ξ : ℝ, ‖b₁ ξ - b₂ ξ‖ := by
  have hi₁ : Integrable (fun ξ : ℝ => 𝐞 (inner ℝ ξ x) • b₁ ξ) := by
    simpa only [inner_neg_right, neg_neg] using
      (Real.fourierIntegral_convergent_iff (-x)).mpr hb₁
  have hi₂ : Integrable (fun ξ : ℝ => 𝐞 (inner ℝ ξ x) • b₂ ξ) := by
    simpa only [inner_neg_right, neg_neg] using
      (Real.fourierIntegral_convergent_iff (-x)).mpr hb₂
  have heq : synthesize b₁ x - synthesize b₂ x =
      synthesize (fun ξ => b₁ ξ - b₂ ξ) x := by
    simp only [synthesize, Real.fourierInv_eq, smul_sub]
    exact (integral_sub hi₁ hi₂).symm
  rw [heq]
  exact norm_synthesize_le _ _

/-- A sharp frequency cutoff incurs at most the omitted spectral mass.
No smooth cutoff, least-squares solve, or target-error assumption is used. -/
theorem norm_synthesize_sub_band_le {b : ℝ → ℂ} (hb : Integrable b)
    (B x : ℝ) :
    ‖synthesize b x - synthesize (bandSpectrum b B) x‖ ≤
      ∫ ξ in (Set.Icc (-B) B)ᶜ, ‖b ξ‖ := by
  have hband : Integrable (bandSpectrum b B) := hb.indicator measurableSet_Icc
  calc
    ‖synthesize b x - synthesize (bandSpectrum b B) x‖ ≤
        ∫ ξ : ℝ, ‖b ξ - bandSpectrum b B ξ‖ :=
      norm_synthesize_sub_le hb hband x
    _ = ∫ ξ in (Set.Icc (-B) B)ᶜ, ‖b ξ‖ := by
      rw [← integral_indicator measurableSet_Icc.compl]
      apply integral_congr_ae
      filter_upwards with ξ
      by_cases hξ : ξ ∈ Set.Icc (-B) B <;>
        simp [bandSpectrum, hξ]

/-- The density bound used in the halo certificate is an actual integral bound. -/
theorem norm_cutoffDensity_le (pHat Khat : ℝ → ℂ) (B x : ℝ) :
    ‖cutoffDensity pHat Khat B x‖ ≤
      ∫ ξ in Set.Icc (-B) B, ‖pHat ξ‖ / ‖Khat ξ‖ := by
  calc
    ‖cutoffDensity pHat Khat B x‖ ≤
        ∫ ξ : ℝ, ‖bandSpectrum (fun ξ => pHat ξ / Khat ξ) B ξ‖ :=
      norm_synthesize_le _ _
    _ = ∫ ξ in Set.Icc (-B) B, ‖pHat ξ‖ / ‖Khat ξ‖ := by
      rw [← integral_indicator measurableSet_Icc]
      apply integral_congr_ae
      filter_upwards with ξ
      by_cases hξ : ξ ∈ Set.Icc (-B) B <;>
        simp [bandSpectrum, hξ]

/-- Cancelling the inverse kernel multiplier recovers the retained target spectrum.
The needed nonvanishing assumption is stated on exactly the retained band. -/
theorem band_quotient_mul (pHat Khat : ℝ → ℂ) (B : ℝ)
    (hK : ∀ ξ ∈ Set.Icc (-B) B, Khat ξ ≠ 0) :
    (fun ξ => bandSpectrum (fun ξ => pHat ξ / Khat ξ) B ξ * Khat ξ) =
      bandSpectrum pHat B := by
  funext ξ
  by_cases hξ : ξ ∈ Set.Icc (-B) B
  · simp [bandSpectrum, hξ, div_mul_cancel₀ _ (hK ξ hξ)]
  · simp [bandSpectrum, hξ]

/-- Applying the frequency-tail estimate to the actual Fourier transform of a target.
Fourier inversion is proved by mathlib, not supplied as a reconstruction axiom. -/
theorem cutoff_target_error {p : ℝ → ℂ} (hp : Continuous p)
    (hip : Integrable p) (hfp : Integrable (𝓕 p)) (B x : ℝ) :
    ‖p x - synthesize (bandSpectrum (𝓕 p) B) x‖ ≤
      ∫ ξ in (Set.Icc (-B) B)ᶜ, ‖𝓕 p ξ‖ := by
  have hinv : synthesize (𝓕 p) x = p x :=
    congrFun (hp.fourierInv_fourier_eq hip hfp) x
  rw [← hinv]
  exact norm_synthesize_sub_band_le hfp B x

/-- Every Schwartz function is absolutely summable on the integer lattice. -/
theorem schwartz_summable_lattice (F : 𝓢(ℝ, ℂ)) :
    Summable (fun j : ℤ => F (j : ℝ)) := by
  exact summable_of_isBigO (Real.summable_abs_int_rpow (by norm_num : (1 : ℝ) < 2))
    ((F.isBigO_cocompact_rpow (-2)).comp_tendsto Int.tendsto_coe_cofinite)

/-- Exact Poisson quadrature remainder: nonzero dual-lattice frequencies are
precisely the difference between the sampled sum and the integral. -/
theorem schwartz_poisson_remainder (F : 𝓢(ℝ, ℂ)) :
    (∑' j : ℤ, F (j : ℝ)) - (∫ x : ℝ, F x) =
      ∑' k : ℤ, if k = 0 then 0 else 𝓕 F (k : ℝ) := by
  have hP := F.tsum_eq_tsum_fourier 0
  have hs := schwartz_summable_lattice (𝓕 F)
  have hsplit := hs.tsum_eq_add_tsum_ite 0
  have hzero : 𝓕 F (0 : ℝ) = ∫ x : ℝ, F x := by
    rw [SchwartzMap.fourier_coe, Real.fourier_eq]
    simp
  have heq : (∑' j : ℤ, F (j : ℝ)) = ∑' k : ℤ, 𝓕 F (k : ℝ) := by
    simpa using hP
  rw [heq, hsplit]
  simp only [Int.cast_zero, hzero, add_sub_cancel_left]

/-- The Poisson error is bounded by all nonzero Fourier lattice values;
no nearest-alias-only estimate is substituted. -/
theorem norm_schwartz_poisson_remainder_le (F : 𝓢(ℝ, ℂ)) :
    ‖(∑' j : ℤ, F (j : ℝ)) - (∫ x : ℝ, F x)‖ ≤
      ∑' k : ℤ, if k = 0 then 0 else ‖𝓕 F (k : ℝ)‖ := by
  rw [schwartz_poisson_remainder]
  have hs : Summable (fun k : ℤ => if k = 0 then (0 : ℂ) else 𝓕 F (k : ℝ)) := by
    exact (schwartz_summable_lattice (𝓕 F)).congr_cofinite (by
      filter_upwards [Filter.eventually_cofinite_ne (0 : ℤ)] with k hk
      simp [hk])
  simpa only [apply_ite norm, norm_zero] using norm_tsum_le_tsum_norm hs.norm

/-- Exact finite-grid truncation identity, valid for any finite retained set. -/
theorem lattice_tail_identity {f : ℤ → ℂ} (hf : Summable f) (J : Finset ℤ) :
    (∑' j : ℤ, f j) - ∑ j ∈ J, f j =
      ∑' j : ↑(J : Set ℤ)ᶜ, f j := by
  have hs := hf.sum_add_tsum_compl (s := J)
  rw [← hs]
  simp

/-- The finite halo error is the actual omitted spatial tail. -/
theorem lattice_tail_bound {f : ℤ → ℂ} (hf : Summable f) (J : Finset ℤ) :
    ‖(∑' j : ℤ, f j) - ∑ j ∈ J, f j‖ ≤
      ∑' j : ↑(J : Set ℤ)ᶜ, ‖f j‖ := by
  rw [lattice_tail_identity hf J]
  exact norm_tsum_le_tsum_norm (hf.subtype _).norm

/-- A finite quadrature certificate from the genuine Poisson identity and
omitted lattice points. `F` is the full quadrature integrand, not an abstract error. -/
theorem finite_schwartz_quadrature_error (F : 𝓢(ℝ, ℂ)) (J : Finset ℤ) :
    ‖(∫ x : ℝ, F x) - ∑ j ∈ J, F (j : ℝ)‖ ≤
      (∑' k : ℤ, if k = 0 then 0 else ‖𝓕 F (k : ℝ)‖) +
      ∑' j : ↑(J : Set ℤ)ᶜ, ‖F (j : ℝ)‖ := by
  calc
    ‖(∫ x : ℝ, F x) - ∑ j ∈ J, F (j : ℝ)‖ ≤
        ‖(∫ x : ℝ, F x) - ∑' j : ℤ, F (j : ℝ)‖ +
        ‖(∑' j : ℤ, F (j : ℝ)) - ∑ j ∈ J, F (j : ℝ)‖ :=
      norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ ≤ _ := by
      rw [norm_sub_rev (∫ x : ℝ, F x)]
      exact add_le_add (norm_schwartz_poisson_remainder_le F)
        (lattice_tail_bound (schwartz_summable_lattice F) J)

/-- The unit Fourier character, with cycles rather than angular frequency. -/
def phase (u : ℝ) : ℂ := (𝐞 u : ℂ)

@[simp] theorem norm_phase (u : ℝ) : ‖phase u‖ = 1 := by
  exact (𝐞 u).norm_coe

@[simp] theorem phase_add (u v : ℝ) : phase (u + v) = phase u * phase v := by
  simp [phase, AddChar.map_add_eq_mul]

/-- Write inverse synthesis in explicit one-dimensional coordinates. -/
theorem synthesize_eq_integral (b : ℝ → ℂ) (x : ℝ) :
    synthesize b x = ∫ ξ : ℝ, phase (ξ * x) * b ξ := by
  simp [synthesize, Real.fourierInv_eq, phase, Circle.smul_def, mul_comm]

/-- Fourier transform of a reflected translate, as an actual integral identity. -/
theorem integral_phase_reflect (K : ℝ → ℂ) (s ξ : ℝ) :
    (∫ c : ℝ, phase (ξ * c) * K (s - c)) = phase (ξ * s) * 𝓕 K ξ := by
  calc
    (∫ c : ℝ, phase (ξ * c) * K (s - c)) =
        ∫ u : ℝ, phase (ξ * (s - u)) * K u := by
      simpa only [sub_sub_cancel] using
        (integral_sub_left_eq_self
          (fun u : ℝ => phase (ξ * (s - u)) * K u) volume s)
    _ = ∫ u : ℝ, phase (ξ * s) * (phase (-(u * ξ)) * K u) := by
      apply integral_congr_ae
      filter_upwards with u
      rw [show ξ * (s - u) = ξ * s + -(u * ξ) by ring, phase_add, mul_assoc]
    _ = phase (ξ * s) * 𝓕 K ξ := by
      rw [integral_const_mul, Real.fourier_real_eq]
      simp only [phase, Circle.smul_def, smul_eq_mul]

/-- Exact inverse-multiplier reconstruction for an integrable spectrum and kernel.
The density itself need not be integrable: this includes sharp-cutoff spectra. -/
theorem synthesize_convolve {b K : ℝ → ℂ}
    (hb : Integrable b) (hK : Integrable K) (s : ℝ) :
    (∫ c : ℝ, synthesize b c * K (s - c)) =
      synthesize (fun ξ => b ξ * 𝓕 K ξ) s := by
  have hKs : Integrable (fun c : ℝ => K (s - c)) := hK.comp_sub_left s
  have hprod : Integrable
      (fun z : ℝ × ℝ => phase (z.2 * z.1) * b z.2 * K (s - z.1))
      (volume.prod volume) := by
    apply (hKs.norm.mul_prod hb.norm).mono'
    · have hp : Continuous (fun z : ℝ × ℝ => phase (z.2 * z.1)) := by
        unfold phase
        fun_prop
      exact (hp.aestronglyMeasurable.mul hb.aestronglyMeasurable.comp_snd).mul
        hKs.aestronglyMeasurable.comp_fst
    · filter_upwards with z
      simp only [norm_mul, norm_phase, one_mul]
      exact le_of_eq (mul_comm _ _)
  calc
    (∫ c : ℝ, synthesize b c * K (s - c)) =
        ∫ c : ℝ, ∫ ξ : ℝ, phase (ξ * c) * b ξ * K (s - c) := by
      simp only [synthesize_eq_integral, integral_mul_const]
    _ = ∫ ξ : ℝ, ∫ c : ℝ, phase (ξ * c) * b ξ * K (s - c) :=
      integral_integral_swap hprod
    _ = ∫ ξ : ℝ, b ξ * (phase (ξ * s) * 𝓕 K ξ) := by
      apply integral_congr_ae
      filter_upwards with ξ
      calc
        (∫ c : ℝ, phase (ξ * c) * b ξ * K (s - c)) =
            b ξ * ∫ c : ℝ, phase (ξ * c) * K (s - c) := by
          rw [← integral_const_mul]
          apply integral_congr_ae
          filter_upwards with c
          ring
        _ = _ := by rw [integral_phase_reflect]
    _ = synthesize (fun ξ => b ξ * 𝓕 K ξ) s := by
      rw [synthesize_eq_integral]
      apply integral_congr_ae
      filter_upwards with ξ
      ring

/-- The specified cutoff density reconstructs the target's retained frequency band.
This is the actual deconvolution identity; no reconstruction property is assumed. -/
theorem cutoffDensity_convolve {pHat K : ℝ → ℂ} (B : ℝ)
    (hb : Integrable (bandSpectrum (fun ξ => pHat ξ / 𝓕 K ξ) B))
    (hK : Integrable K)
    (hnz : ∀ ξ ∈ Set.Icc (-B) B, 𝓕 K ξ ≠ 0) (s : ℝ) :
    (∫ c : ℝ, cutoffDensity pHat (𝓕 K) B c * K (s - c)) =
      synthesize (bandSpectrum pHat B) s := by
  unfold cutoffDensity
  rw [synthesize_convolve hb hK s, band_quotient_mul pHat (𝓕 K) B hnz]

/-- A compact-band inverse multiplier is integrable if the target spectrum is
integrable and the kernel transform is continuous and nonzero on that band. -/
theorem integrable_band_quotient {pHat Khat : ℝ → ℂ}
    (hp : Integrable pHat) (hc : Continuous Khat) (B : ℝ)
    (hnz : ∀ ξ ∈ Set.Icc (-B) B, Khat ξ ≠ 0) :
    Integrable (bandSpectrum (fun ξ => pHat ξ / Khat ξ) B) := by
  have hinv : ContinuousOn (fun ξ => (Khat ξ)⁻¹) (Set.Icc (-B) B) :=
    hc.continuousOn.inv₀ hnz
  have hprod : IntegrableOn (fun ξ => pHat ξ * (Khat ξ)⁻¹) (Set.Icc (-B) B) :=
    hp.integrableOn.mul_continuousOn hinv isCompact_Icc
  simpa [bandSpectrum, div_eq_mul_inv] using
    (integrable_indicator_iff measurableSet_Icc).mpr hprod

/-- The Fourier transform of the actual density-times-translated-kernel integrand.
This is the frequency-shift identity needed to turn Poisson into the explicit alias sum. -/
theorem fourier_synthesize_mul_reflect {b K : ℝ → ℂ}
    (hb : Integrable b) (hK : Integrable K) (s ξ : ℝ) :
    𝓕 (fun c => synthesize b c * K (s - c)) ξ =
      ∫ η : ℝ, b η * (phase ((η - ξ) * s) * 𝓕 K (η - ξ)) := by
  have hKs : Integrable (fun c : ℝ => K (s - c)) := hK.comp_sub_left s
  have hprod : Integrable
      (fun z : ℝ × ℝ => phase ((z.2 - ξ) * z.1) * b z.2 * K (s - z.1))
      (volume.prod volume) := by
    apply (hKs.norm.mul_prod hb.norm).mono'
    · have hp : Continuous (fun z : ℝ × ℝ => phase ((z.2 - ξ) * z.1)) := by
        unfold phase
        fun_prop
      exact (hp.aestronglyMeasurable.mul hb.aestronglyMeasurable.comp_snd).mul
        hKs.aestronglyMeasurable.comp_fst
    · filter_upwards with z
      simp only [norm_mul, norm_phase, one_mul]
      exact le_of_eq (mul_comm _ _)
  calc
    𝓕 (fun c => synthesize b c * K (s - c)) ξ =
        ∫ c : ℝ, phase (-(c * ξ)) * (synthesize b c * K (s - c)) := by
      rw [Real.fourier_real_eq]
      simp only [phase, Circle.smul_def, smul_eq_mul]
    _ = ∫ c : ℝ, ∫ η : ℝ, phase ((η - ξ) * c) * b η * K (s - c) := by
      apply integral_congr_ae
      filter_upwards with c
      rw [synthesize_eq_integral, ← integral_mul_const, ← integral_const_mul]
      apply integral_congr_ae
      filter_upwards with η
      rw [show (η - ξ) * c = -(c * ξ) + η * c by ring, phase_add]
      ring
    _ = ∫ η : ℝ, ∫ c : ℝ, phase ((η - ξ) * c) * b η * K (s - c) :=
      integral_integral_swap hprod
    _ = _ := by
      apply integral_congr_ae
      filter_upwards with η
      calc
        (∫ c : ℝ, phase ((η - ξ) * c) * b η * K (s - c)) =
            b η * ∫ c : ℝ, phase ((η - ξ) * c) * K (s - c) := by
          rw [← integral_const_mul]
          apply integral_congr_ae
          filter_upwards with c
          ring
        _ = _ := by rw [integral_phase_reflect]

/-- Inverse synthesis of an integrable spectrum is continuous. -/
theorem continuous_synthesize {b : ℝ → ℂ} (hb : Integrable b) :
    Continuous (synthesize b) := by
  change Continuous (VectorFourier.fourierIntegral 𝐞 volume (-innerₗ ℝ) b)
  exact VectorFourier.fourierIntegral_continuous Real.continuous_fourierChar
    (by fun_prop) hb

/-- A Schwartz kernel has a uniform quadratic envelope. -/
theorem schwartz_quadratic_envelope (K : 𝓢(ℝ, ℂ)) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ u : ℝ, (1 + |u|)^2 * ‖K u‖ ≤ C := by
  let C := (4 : ℝ) * ((Finset.Iic (2, 0)).sup
    (fun m : ℕ × ℕ => SchwartzMap.seminorm ℝ m.1 m.2)) K
  refine ⟨C, by positivity, ?_⟩
  intro u
  convert
    (SchwartzMap.one_add_le_sup_seminorm_apply (𝕜 := ℝ)
      (m := (2, 0)) (k := 2) (n := 0) le_rfl le_rfl K u) using 1 <;> norm_num [C]

/-- Translating a quadratically decaying function over a bounded set preserves a
uniform quadratic envelope. -/
theorem quadratic_envelope_translate {K : ℝ → ℂ} {C B : ℝ}
    (hB : 0 ≤ B) (hK : ∀ u, (1 + |u|)^2 * ‖K u‖ ≤ C)
    {η : ℝ} (hη : |η| ≤ B) (x : ℝ) :
    (1 + |x|)^2 * ‖K (η - x)‖ ≤ (1 + B)^2 * C := by
  have htri : |x| ≤ B + |η - x| := by
    calc
      |x| = |η - (η - x)| := by congr 1; ring
      _ ≤ |η| + |η - x| := abs_sub _ _
      _ ≤ B + |η - x| := by gcongr
  have hgrow : 1 + |x| ≤ (1 + B) * (1 + |η - x|) := by
    nlinarith [abs_nonneg (η - x)]
  calc
    (1 + |x|)^2 * ‖K (η - x)‖ ≤
        ((1 + B) * (1 + |η - x|))^2 * ‖K (η - x)‖ := by
      gcongr
    _ = (1 + B)^2 * ((1 + |η - x|)^2 * ‖K (η - x)‖) := by ring
    _ ≤ (1 + B)^2 * C :=
      mul_le_mul_of_nonneg_left (hK _) (sq_nonneg _)

/-- A quadratic envelope supplies the exact decay hypothesis in mathlib's
Poisson summation theorem. -/
theorem isBigO_of_quadratic_envelope {E : Type*} [NormedAddCommGroup E] {F : ℝ → E} {C : ℝ}
    (_hC : 0 ≤ C) (hF : ∀ x, (1 + |x|)^2 * ‖F x‖ ≤ C) :
    F =O[Filter.cocompact ℝ] (fun x : ℝ => |x| ^ (-(2 : ℝ))) := by
  simp only [Asymptotics.IsBigO, Asymptotics.IsBigOWith]
  refine ⟨C, Filter.Eventually.filter_mono Filter.cocompact_le_cofinite ?_⟩
  filter_upwards [Filter.eventually_cofinite_ne (0 : ℝ)] with x hx
  have hp : 0 < |x|^2 := sq_pos_of_pos (abs_pos.mpr hx)
  rw [Real.norm_of_nonneg (by positivity), Real.rpow_neg (abs_nonneg x),
    show (2 : ℝ) = (2 : ℕ) by norm_num, Real.rpow_natCast, ← div_eq_mul_inv,
    le_div_iff₀ hp]
  calc
    ‖F x‖ * |x|^2 ≤ ‖F x‖ * (1 + |x|)^2 := by gcongr; linarith [abs_nonneg x]
    _ ≤ C := by simpa [mul_comm] using hF x

/-- The actual density/kernel product has quadratic spatial decay, even when the
sharp-cutoff density itself has slowly decaying tails. -/
theorem synthesize_mul_reflect_spatial_decay {b : ℝ → ℂ}
    (K : 𝓢(ℝ, ℂ)) (s : ℝ) :
    (fun c => synthesize b c * K (s - c)) =O[Filter.cocompact ℝ]
      (fun c : ℝ => |c| ^ (-(2 : ℝ))) := by
  obtain ⟨C, hC, henv⟩ := schwartz_quadratic_envelope K
  let A := ∫ ξ : ℝ, ‖b ξ‖
  have hA : 0 ≤ A := integral_nonneg (fun _ => norm_nonneg _)
  apply isBigO_of_quadratic_envelope (C := A * (1 + |s|)^2 * C) (by positivity)
  intro c
  rw [norm_mul]
  calc
    (1 + |c|)^2 * (‖synthesize b c‖ * ‖K (s - c)‖) ≤
        A * ((1 + |c|)^2 * ‖K (s - c)‖) := by
      calc
        _ = ‖synthesize b c‖ * ((1 + |c|)^2 * ‖K (s - c)‖) := by ring
        _ ≤ _ := mul_le_mul_of_nonneg_right (norm_synthesize_le b c) (by positivity)
    _ ≤ A * ((1 + |s|)^2 * C) :=
      mul_le_mul_of_nonneg_left
        (quadratic_envelope_translate (abs_nonneg s) henv le_rfl c) hA
    _ = _ := by ring

/-- The Fourier transform of the actual cutoff-density/kernel product also has
quadratic decay. Compact spectral support and the Schwartz kernel imply this;
no decay of the sharp-cutoff density is assumed. -/
theorem synthesize_mul_reflect_fourier_decay {b : ℝ → ℂ}
    (hb : Integrable b) {B : ℝ} (hB : 0 ≤ B)
    (hsupp : ∀ η, B < |η| → b η = 0)
    (K : 𝓢(ℝ, ℂ)) (s : ℝ) :
    (𝓕 (fun c => synthesize b c * K (s - c))) =O[Filter.cocompact ℝ]
      (fun ξ : ℝ => |ξ| ^ (-(2 : ℝ))) := by
  obtain ⟨C, hC, henv⟩ := schwartz_quadratic_envelope (𝓕 K)
  let A := ∫ η : ℝ, ‖b η‖
  have hA : 0 ≤ A := integral_nonneg (fun _ => norm_nonneg _)
  let D := (1 + B)^2 * C
  have hD : 0 ≤ D := by dsimp [D]; positivity
  apply isBigO_of_quadratic_envelope (C := A * D) (by positivity)
  intro ξ
  rw [fourier_synthesize_mul_reflect hb K.integrable]
  have hd : 0 < (1 + |ξ|)^2 := by positivity
  have hpoint : ∀ η : ℝ,
      ‖b η * (phase ((η - ξ) * s) * 𝓕 (K : ℝ → ℂ) (η - ξ))‖ ≤
        ‖b η‖ * (D / (1 + |ξ|)^2) := by
    intro η
    by_cases hη : |η| ≤ B
    · have hbound := quadratic_envelope_translate hB henv hη ξ
      have hbound' : ‖𝓕 K (η - ξ)‖ ≤ D / (1 + |ξ|)^2 := by
        apply (le_div_iff₀ hd).mpr
        simpa only [D, mul_comm] using hbound
      simp only [norm_mul, norm_phase, one_mul]
      rw [← SchwartzMap.fourier_coe]
      exact mul_le_mul_of_nonneg_left hbound' (norm_nonneg _)
    · rw [hsupp η (lt_of_not_ge hη)]
      simp
  have hint := norm_integral_le_of_norm_le
    (hb.norm.mul_const (D / (1 + |ξ|)^2)) (Filter.Eventually.of_forall hpoint)
  rw [integral_mul_const] at hint
  calc
    (1 + |ξ|)^2 * ‖∫ η : ℝ, b η *
        (phase ((η - ξ) * s) * 𝓕 (K : ℝ → ℂ) (η - ξ))‖ ≤
        (1 + |ξ|)^2 * (A * (D / (1 + |ξ|)^2)) :=
      mul_le_mul_of_nonneg_left hint (le_of_lt hd)
    _ = A * D := by field_simp

/-- Poisson summation applies to the actual compact-spectrum density/kernel
integrand under the original Schwartz-kernel assumptions. -/
theorem poisson_synthesize_mul_reflect {b : ℝ → ℂ}
    (hb : Integrable b) {B : ℝ} (hB : 0 ≤ B)
    (hsupp : ∀ η, B < |η| → b η = 0)
    (K : 𝓢(ℝ, ℂ)) (s : ℝ) :
    (∑' j : ℤ, synthesize b (j : ℝ) * K (s - j)) =
      ∑' k : ℤ, ∫ η : ℝ, b η *
        (phase ((η - (k : ℝ)) * s) * 𝓕 (K : ℝ → ℂ) (η - k)) := by
  have hc : Continuous (fun c => synthesize b c * K (s - c)) :=
    (continuous_synthesize hb).mul (K.continuous.comp (continuous_const.sub continuous_id))
  have hp := Real.tsum_eq_tsum_fourier_of_rpow_decay hc
    (by norm_num : (1 : ℝ) < 2)
    (synthesize_mul_reflect_spatial_decay K s)
    (synthesize_mul_reflect_fourier_decay hb hB hsupp K s) 0
  simp only [zero_add] at hp
  simpa [fourier_synthesize_mul_reflect hb K.integrable] using hp

/-- Fourier scaling with every normalization factor explicit. -/
theorem fourier_comp_pos_mul (F : ℝ → ℂ) {h : ℝ} (hh : 0 < h) (ξ : ℝ) :
    𝓕 (fun x => F (h * x)) ξ = (h⁻¹ : ℂ) * 𝓕 F (ξ / h) := by
  rw [Real.fourier_real_eq]
  calc
    (∫ x : ℝ, 𝐞 (-(x * ξ)) • F (h * x)) =
        ∫ x : ℝ, 𝐞 (-((h * x) * (ξ / h))) • F (h * x) := by
      apply integral_congr_ae
      filter_upwards with x
      congr 2
      field_simp
    _ = |h⁻¹| • 𝓕 F (ξ / h) := by
      simpa only [Real.fourier_real_eq] using
        (Measure.integral_comp_mul_left
          (fun u : ℝ => 𝐞 (-(u * (ξ / h))) • F u) h)
    _ = _ := by
      rw [abs_of_pos (inv_pos.mpr hh)]
      simp only [Complex.real_smul, Complex.ofReal_inv]

/-- Quadratic decay is preserved under a nonzero scaling of the argument. -/
theorem quadratic_decay_comp_pos_mul {E : Type*} [NormedAddCommGroup E] {F : ℝ → E} {h : ℝ} (hh : 0 < h)
    (hF : F =O[Filter.cocompact ℝ] (fun x : ℝ => |x| ^ (-(2 : ℝ)))) :
    (fun x => F (h * x)) =O[Filter.cocompact ℝ]
      (fun x : ℝ => |x| ^ (-(2 : ℝ))) := by
  have hc := hF.comp_tendsto (Filter.tendsto_cocompact_mul_left₀ hh.ne')
  have hnonzero : h ^ (-(2 : ℝ)) ≠ 0 := (Real.rpow_pos_of_pos hh _).ne'
  apply (Asymptotics.isBigO_const_mul_right_iff hnonzero).mp
  exact hc.congr_right (fun x => by
    simp only [Function.comp_apply, abs_mul, abs_of_pos hh,
      Real.mul_rpow hh.le (abs_nonneg x)])

/-- Scaled Poisson summation under the exact decay hypotheses proved above. -/
theorem poisson_scaled_of_decay {F : ℝ → ℂ} (hc : Continuous F)
    (hF : F =O[Filter.cocompact ℝ] (fun x : ℝ => |x| ^ (-(2 : ℝ))))
    (hhat : (𝓕 F) =O[Filter.cocompact ℝ] (fun x : ℝ => |x| ^ (-(2 : ℝ))))
    {h : ℝ} (hh : 0 < h) :
    (h : ℂ) * (∑' j : ℤ, F (h * j)) = ∑' k : ℤ, 𝓕 F ((k : ℝ) / h) := by
  have hcscale : Continuous (fun x => F (h * x)) :=
    hc.comp (continuous_const.mul continuous_id)
  have hhatScale :
      (𝓕 (fun x => F (h * x))) =O[Filter.cocompact ℝ]
        (fun ξ : ℝ => |ξ| ^ (-(2 : ℝ))) := by
    have hmul := (quadratic_decay_comp_pos_mul (inv_pos.mpr hh) hhat).const_mul_left
      (h⁻¹ : ℂ)
    refine hmul.congr_left ?_
    intro ξ
    rw [fourier_comp_pos_mul F hh]
    simp only [div_eq_mul_inv, mul_comm]
  have hp := Real.tsum_eq_tsum_fourier_of_rpow_decay hcscale
    (by norm_num : (1 : ℝ) < 2) (quadratic_decay_comp_pos_mul hh hF) hhatScale 0
  have hp' : (∑' j : ℤ, F (h * j)) =
      ∑' k : ℤ, (h⁻¹ : ℂ) * 𝓕 F ((k : ℝ) / h) := by
    simpa [fourier_comp_pos_mul F hh] using hp
  rw [hp', tsum_mul_left, ← mul_assoc]
  norm_cast
  simp [hh.ne']

/-- Poisson summation for a general positive center spacing `h`, on the actual
finite-band coefficient density times translated Schwartz kernel. -/
theorem poisson_scaled_synthesize_mul_reflect {b : ℝ → ℂ}
    (hb : Integrable b) {B : ℝ} (hB : 0 ≤ B)
    (hsupp : ∀ η, B < |η| → b η = 0)
    (K : 𝓢(ℝ, ℂ)) (s : ℝ) {h : ℝ} (hh : 0 < h) :
    (h : ℂ) * (∑' j : ℤ, synthesize b (h * j) * K (s - h * j)) =
      ∑' k : ℤ, ∫ η : ℝ, b η *
        (phase ((η - (k : ℝ) / h) * s) * 𝓕 (K : ℝ → ℂ) (η - k / h)) := by
  let F : ℝ → ℂ := fun c => synthesize b c * K (s - c)
  have hc : Continuous F :=
    (continuous_synthesize hb).mul (K.continuous.comp (continuous_const.sub continuous_id))
  calc
    _ = ∑' k : ℤ, 𝓕 F ((k : ℝ) / h) :=
      poisson_scaled_of_decay hc (synthesize_mul_reflect_spatial_decay (b := b) K s)
        (synthesize_mul_reflect_fourier_decay hb hB hsupp K s) hh
    _ = _ := tsum_congr fun k =>
      fourier_synthesize_mul_reflect hb K.integrable s ((k : ℝ) / h)

/-- The grid samples of a quadratically decaying function are absolutely summable. -/
theorem summable_lattice_of_quadratic_decay {E : Type*} [NormedAddCommGroup E]
    [CompleteSpace E] {F : ℝ → E}
    (hF : F =O[Filter.cocompact ℝ] (fun x : ℝ => |x| ^ (-(2 : ℝ)))) :
    Summable (fun j : ℤ => F (j : ℝ)) :=
  summable_of_isBigO (Real.summable_abs_int_rpow (by norm_num : (1 : ℝ) < 2))
    (hF.comp_tendsto Int.tendsto_coe_cofinite)

/-- All off-zero Fourier modes, plus the finite spatial tail, bound the finite
quadrature error for any positive spacing. -/
theorem finite_scaled_quadrature_error {F : ℝ → ℂ} (hc : Continuous F)
    (hF : F =O[Filter.cocompact ℝ] (fun x : ℝ => |x| ^ (-(2 : ℝ))))
    (hhat : (𝓕 F) =O[Filter.cocompact ℝ] (fun x : ℝ => |x| ^ (-(2 : ℝ))))
    {h : ℝ} (hh : 0 < h) (J : Finset ℤ) :
    ‖(∫ x : ℝ, F x) - (h : ℂ) * ∑ j ∈ J, F (h * j)‖ ≤
      (∑' k : ℤ, if k = 0 then 0 else ‖𝓕 F ((k : ℝ) / h)‖) +
      h * ∑' j : ↑(J : Set ℤ)ᶜ, ‖F (h * (j : ℤ))‖ := by
  have hgrid := summable_lattice_of_quadratic_decay (quadratic_decay_comp_pos_mul hh hF)
  have hfreq : Summable (fun k : ℤ => 𝓕 F ((k : ℝ) / h)) := by
    simpa only [div_eq_mul_inv, mul_comm] using
      summable_lattice_of_quadratic_decay
        (quadratic_decay_comp_pos_mul (inv_pos.mpr hh) hhat)
  have hrest : Summable (fun k : ℤ => if k = 0 then (0 : ℂ) else 𝓕 F ((k : ℝ) / h)) :=
    hfreq.congr_cofinite (by
      filter_upwards [Filter.eventually_cofinite_ne (0 : ℤ)] with k hk
      simp [hk])
  have hz : 𝓕 F 0 = ∫ x : ℝ, F x := by simp [Real.fourier_eq]
  have hp := poisson_scaled_of_decay hc hF hhat hh
  have hsplit := hfreq.tsum_eq_add_tsum_ite 0
  have herr : (h : ℂ) * (∑' j : ℤ, F (h * j)) - (∫ x : ℝ, F x) =
      ∑' k : ℤ, if k = 0 then 0 else 𝓕 F ((k : ℝ) / h) := by
    rw [hp, hsplit]
    simp only [Int.cast_zero, zero_div, hz, add_sub_cancel_left]
  have hnorm : ‖(∫ x : ℝ, F x) - (h : ℂ) * (∑' j : ℤ, F (h * j))‖ ≤
      ∑' k : ℤ, if k = 0 then 0 else ‖𝓕 F ((k : ℝ) / h)‖ := by
    rw [norm_sub_rev, herr]
    simpa only [apply_ite norm, norm_zero] using norm_tsum_le_tsum_norm hrest.norm
  have htail : ‖(h : ℂ) * (∑' j : ℤ, F (h * j)) -
      (h : ℂ) * ∑ j ∈ J, F (h * j)‖ ≤
      h * ∑' j : ↑(J : Set ℤ)ᶜ, ‖F (h * (j : ℤ))‖ := by
    rw [← mul_sub, norm_mul]
    simp only [Complex.norm_real, Real.norm_eq_abs, abs_of_pos hh]
    exact mul_le_mul_of_nonneg_left (lattice_tail_bound hgrid J) hh.le
  exact (norm_sub_le_norm_sub_add_norm_sub _ _ _).trans (add_le_add hnorm htail)

/-- Actual finite-band deconvolution followed by actual finite sampling: the
pointwise derivative error is bounded by spectral cutoff, every Fourier alias,
and every omitted center. The kernel is a Schwartz function; no approximation
or Poisson conclusion is assumed. This theorem concerns the kernel sum before
integration into the chosen activation. -/
theorem finite_cutoff_kernel_error {p : ℝ → ℂ} (hp : Continuous p)
    (hip : Integrable p) (hfp : Integrable (𝓕 p))
    (K : 𝓢(ℝ, ℂ)) {B h : ℝ} (hB : 0 ≤ B) (hh : 0 < h)
    (hnz : ∀ ξ ∈ Set.Icc (-B) B, 𝓕 (K : ℝ → ℂ) ξ ≠ 0)
    (s : ℝ) (J : Finset ℤ) :
    let ρ := cutoffDensity (𝓕 p) (𝓕 (K : ℝ → ℂ)) B
    let F := fun c => ρ c * K (s - c)
    ‖p s - (h : ℂ) * ∑ j ∈ J, ρ (h * j) * K (s - h * j)‖ ≤
      (∫ ξ in (Set.Icc (-B) B)ᶜ, ‖𝓕 p ξ‖) +
      (∑' k : ℤ, if k = 0 then 0 else ‖𝓕 F ((k : ℝ) / h)‖) +
      h * ∑' j : ↑(J : Set ℤ)ᶜ, ‖F (h * (j : ℤ))‖ := by
  let b := bandSpectrum (fun ξ => 𝓕 p ξ / 𝓕 (K : ℝ → ℂ) ξ) B
  have hKcont : Continuous (𝓕 (K : ℝ → ℂ)) := by
    rw [← SchwartzMap.fourier_coe]
    exact (𝓕 K).continuous
  have hb : Integrable b := integrable_band_quotient hfp hKcont B hnz
  have hsupp : ∀ η, B < |η| → b η = 0 := by
    intro η hη
    have hout : η ∉ Set.Icc (-B) B := by
      intro hin
      have habs : |η| ≤ B := abs_le.mpr hin
      linarith
    simp [b, bandSpectrum, hout]
  let F := fun c => synthesize b c * K (s - c)
  have hc : Continuous F :=
    (continuous_synthesize hb).mul (K.continuous.comp (continuous_const.sub continuous_id))
  have hquad := finite_scaled_quadrature_error hc
    (synthesize_mul_reflect_spatial_decay (b := b) K s)
    (synthesize_mul_reflect_fourier_decay hb hB hsupp K s) hh J
  have hconv : (∫ c : ℝ, F c) = synthesize (bandSpectrum (𝓕 p) B) s :=
    cutoffDensity_convolve B hb K.integrable hnz s
  have hcut : ‖p s - ∫ c : ℝ, F c‖ ≤
      ∫ ξ in (Set.Icc (-B) B)ᶜ, ‖𝓕 p ξ‖ := by
    rw [hconv]
    exact cutoff_target_error hp hip hfp B s
  dsimp only
  change ‖p s - (h : ℂ) * ∑ j ∈ J, F (h * j)‖ ≤
    (∫ ξ in (Set.Icc (-B) B)ᶜ, ‖𝓕 p ξ‖) +
      (∑' k : ℤ, if k = 0 then 0 else ‖𝓕 F ((k : ℝ) / h)‖) +
      h * ∑' j : ↑(J : Set ℤ)ᶜ, ‖F (h * (j : ℤ))‖
  exact (norm_sub_le_norm_sub_add_norm_sub _ _ _).trans
    ((add_le_add hcut hquad).trans_eq (add_assoc _ _ _).symm)

/-- Absolute spectral overlap at an alias shift, independent of evaluation point. -/
def aliasMass (b Khat : ℝ → ℂ) (ξ : ℝ) : ℝ :=
  ∫ η : ℝ, ‖b η‖ * ‖Khat (η - ξ)‖

@[simp] theorem aliasMass_nonneg (b Khat : ℝ → ℂ) (ξ : ℝ) :
    0 ≤ aliasMass b Khat ξ :=
  integral_nonneg (fun _ => mul_nonneg (norm_nonneg _) (norm_nonneg _))

/-- The entire spectral-overlap mass, not only the modulus of its oscillatory
integral, has a quadratic envelope. This justifies summing every alias. -/
theorem aliasMass_decay {b : ℝ → ℂ} (hb : Integrable b)
    {B : ℝ} (hB : 0 ≤ B) (hsupp : ∀ η, B < |η| → b η = 0)
    (Khat : 𝓢(ℝ, ℂ)) :
    aliasMass b Khat =O[Filter.cocompact ℝ]
      (fun ξ : ℝ => |ξ| ^ (-(2 : ℝ))) := by
  obtain ⟨C, hC, henv⟩ := schwartz_quadratic_envelope Khat
  let A := ∫ η : ℝ, ‖b η‖
  have hA : 0 ≤ A := integral_nonneg (fun _ => norm_nonneg _)
  let D := (1 + B)^2 * C
  have hD : 0 ≤ D := by dsimp [D]; positivity
  apply isBigO_of_quadratic_envelope (C := A * D) (by positivity)
  intro ξ
  rw [Real.norm_eq_abs, abs_of_nonneg (aliasMass_nonneg _ _ _)]
  have hd : 0 < (1 + |ξ|)^2 := by positivity
  have hpoint : ∀ η : ℝ, ‖b η‖ * ‖Khat (η - ξ)‖ ≤
        ‖b η‖ * (D / (1 + |ξ|)^2) := by
    intro η
    by_cases hη : |η| ≤ B
    · have hbound := quadratic_envelope_translate hB henv hη ξ
      have hbound' : ‖Khat (η - ξ)‖ ≤ D / (1 + |ξ|)^2 := by
        apply (le_div_iff₀ hd).mpr
        simpa only [D, mul_comm] using hbound
      exact mul_le_mul_of_nonneg_left hbound' (norm_nonneg _)
    · rw [hsupp η (lt_of_not_ge hη)]
      simp
  have hint := integral_mono_of_nonneg
    (Filter.Eventually.of_forall (fun η => mul_nonneg (norm_nonneg (b η))
      (norm_nonneg (Khat (η - ξ)))))
    (hb.norm.mul_const (D / (1 + |ξ|)^2)) (Filter.Eventually.of_forall hpoint)
  rw [integral_mul_const] at hint
  calc
    (1 + |ξ|)^2 * aliasMass b Khat ξ ≤
        (1 + |ξ|)^2 * (A * (D / (1 + |ξ|)^2)) :=
      mul_le_mul_of_nonneg_left hint (le_of_lt hd)
    _ = A * D := by field_simp

/-- A single alias integral is bounded by its absolute spectral overlap. -/
theorem norm_alias_integral_le {b : ℝ → ℂ} (hb : Integrable b)
    (Khat : 𝓢(ℝ, ℂ)) (s ξ : ℝ) :
    ‖∫ η : ℝ, b η * (phase ((η - ξ) * s) * Khat (η - ξ))‖ ≤
      aliasMass b Khat ξ := by
  have hint : Integrable (fun η : ℝ => ‖b η‖ * ‖Khat (η - ξ)‖) := by
    apply hb.norm.mul_bdd
      ((Khat.continuous.comp (continuous_id.sub continuous_const)).norm.aestronglyMeasurable)
    filter_upwards with η
    simpa only [norm_norm, Function.comp_apply, Pi.sub_apply, id_eq] using SchwartzMap.norm_le_seminorm ℝ Khat (η - ξ)
  apply norm_integral_le_of_norm_le hint
  filter_upwards with η
  simp only [norm_mul, norm_phase, one_mul, le_refl]

/-- The sum of all aliases has a phase-independent, summable majorant. -/
theorem sum_fourier_alias_le_mass {b : ℝ → ℂ} (hb : Integrable b)
    {B h : ℝ} (hB : 0 ≤ B) (hh : 0 < h)
    (hsupp : ∀ η, B < |η| → b η = 0) (K : 𝓢(ℝ, ℂ)) (s : ℝ) :
    (∑' k : ℤ, if k = 0 then 0 else
      ‖𝓕 (fun c => synthesize b c * K (s - c)) ((k : ℝ) / h)‖) ≤
    ∑' k : ℤ, if k = 0 then 0 else aliasMass b (𝓕 (K : ℝ → ℂ)) ((k : ℝ) / h) := by
  have hfreq : Summable (fun k : ℤ =>
      𝓕 (fun c => synthesize b c * K (s - c)) ((k : ℝ) / h)) := by
    simpa only [div_eq_mul_inv, mul_comm] using
      summable_lattice_of_quadratic_decay (quadratic_decay_comp_pos_mul (inv_pos.mpr hh)
        (synthesize_mul_reflect_fourier_decay hb hB hsupp K s))
  have hmass : Summable (fun k : ℤ => aliasMass b (𝓕 (K : ℝ → ℂ)) ((k : ℝ) / h)) := by
    simpa only [div_eq_mul_inv, mul_comm, SchwartzMap.fourier_coe] using
      summable_lattice_of_quadratic_decay (quadratic_decay_comp_pos_mul (inv_pos.mpr hh)
        (aliasMass_decay hb hB hsupp (𝓕 K)))
  have hfreq' : Summable (fun k : ℤ => if k = 0 then 0 else
      ‖𝓕 (fun c => synthesize b c * K (s - c)) ((k : ℝ) / h)‖) :=
    hfreq.norm.congr_cofinite (by
      filter_upwards [Filter.eventually_cofinite_ne (0 : ℤ)] with k hk
      simp [hk])
  have hmass' : Summable (fun k : ℤ => if k = 0 then 0 else
      aliasMass b (𝓕 (K : ℝ → ℂ)) ((k : ℝ) / h)) :=
    hmass.congr_cofinite (by
      filter_upwards [Filter.eventually_cofinite_ne (0 : ℤ)] with k hk
      simp [hk])
  apply Summable.tsum_le_tsum _ hfreq' hmass'
  intro k
  by_cases hk : k = 0
  · simp [hk]
  · simp only [hk, if_false]
    rw [fourier_synthesize_mul_reflect hb K.integrable]
    simpa only [SchwartzMap.fourier_coe] using norm_alias_integral_le hb (𝓕 K) s ((k : ℝ) / h)

/-- A translated Schwartz kernel has absolutely summable samples for every
positive grid spacing. -/
theorem summable_translated_kernel (K : 𝓢(ℝ, ℂ)) (s : ℝ)
    {h : ℝ} (hh : 0 < h) :
    Summable (fun j : ℤ => K (s - h * j)) := by
  obtain ⟨C, hC, henv⟩ := schwartz_quadratic_envelope K
  have hd : (fun c : ℝ => K (s - c)) =O[Filter.cocompact ℝ]
      (fun c : ℝ => |c| ^ (-(2 : ℝ))) := by
    apply isBigO_of_quadratic_envelope (C := (1 + |s|)^2 * C) (by positivity)
    exact quadratic_envelope_translate (abs_nonneg s) henv le_rfl
  exact summable_lattice_of_quadratic_decay (quadratic_decay_comp_pos_mul hh hd)

/-- The finite-center tail is controlled by the density's spectral L¹ bound times
the explicit omitted-kernel tail. -/
theorem sum_spatial_tail_le_density_bound (b : ℝ → ℂ)
    (K : 𝓢(ℝ, ℂ)) (s : ℝ) {h : ℝ} (hh : 0 < h) (J : Finset ℤ) :
    (∑' j : ↑(J : Set ℤ)ᶜ, ‖synthesize b (h * (j : ℤ)) *
      K (s - h * (j : ℤ))‖) ≤
      (∫ ξ : ℝ, ‖b ξ‖) *
        ∑' j : ↑(J : Set ℤ)ᶜ, ‖K (s - h * (j : ℤ))‖ := by
  have hFsum := summable_lattice_of_quadratic_decay
    (quadratic_decay_comp_pos_mul hh (synthesize_mul_reflect_spatial_decay (b := b) K s))
  have hKsum := summable_translated_kernel K s hh
  rw [← tsum_mul_left]
  apply Summable.tsum_le_tsum _ (hFsum.subtype _).norm
    ((hKsum.subtype _).norm.mul_left _)
  intro j
  simp only [Function.comp_apply, norm_mul]
  exact mul_le_mul_of_nonneg_right (norm_synthesize_le b _) (norm_nonneg _)

/-- The finite, general-kernel certificate in the approved spectral-input form.
All three error terms are explicit integrals or absolutely convergent sums.
`pHat` is a supplied integrable target spectrum; the target itself is its inverse
Fourier transform, so it need not be integrable in space. The kernel may already
include any positive bandwidth scaling. No fitted coefficients or pre-assumed
approximation inequality occur in the hypotheses. -/
theorem finite_spectral_kernel_certificate {pHat : ℝ → ℂ}
    (hpHat : Integrable pHat) (K : 𝓢(ℝ, ℂ)) {B h : ℝ}
    (hB : 0 ≤ B) (hh : 0 < h)
    (hnz : ∀ ξ ∈ Set.Icc (-B) B, 𝓕 (K : ℝ → ℂ) ξ ≠ 0)
    (s : ℝ) (J : Finset ℤ) :
    let b := bandSpectrum (fun ξ => pHat ξ / 𝓕 (K : ℝ → ℂ) ξ) B
    let ρ := synthesize b
    ‖synthesize pHat s - (h : ℂ) * ∑ j ∈ J, ρ (h * j) * K (s - h * j)‖ ≤
      (∫ ξ in (Set.Icc (-B) B)ᶜ, ‖pHat ξ‖) +
      (∑' k : ℤ, if k = 0 then 0 else
        aliasMass b (𝓕 (K : ℝ → ℂ)) ((k : ℝ) / h)) +
      h * (∫ ξ : ℝ, ‖b ξ‖) *
        ∑' j : ↑(J : Set ℤ)ᶜ, ‖K (s - h * (j : ℤ))‖ := by
  let b := bandSpectrum (fun ξ => pHat ξ / 𝓕 (K : ℝ → ℂ) ξ) B
  have hKcont : Continuous (𝓕 (K : ℝ → ℂ)) := by
    rw [← SchwartzMap.fourier_coe]
    exact (𝓕 K).continuous
  have hb : Integrable b := integrable_band_quotient hpHat hKcont B hnz
  have hsupp : ∀ η, B < |η| → b η = 0 := by
    intro η hη
    have hout : η ∉ Set.Icc (-B) B := by
      intro hin
      have habs : |η| ≤ B := abs_le.mpr hin
      linarith
    simp [b, bandSpectrum, hout]
  let F := fun c => synthesize b c * K (s - c)
  have hc : Continuous F :=
    (continuous_synthesize hb).mul (K.continuous.comp (continuous_const.sub continuous_id))
  have hquad := finite_scaled_quadrature_error hc
    (synthesize_mul_reflect_spatial_decay (b := b) K s)
    (synthesize_mul_reflect_fourier_decay hb hB hsupp K s) hh J
  have hconv : (∫ c : ℝ, F c) = synthesize (bandSpectrum pHat B) s :=
    cutoffDensity_convolve B hb K.integrable hnz s
  have hcut : ‖synthesize pHat s - ∫ c : ℝ, F c‖ ≤
      ∫ ξ in (Set.Icc (-B) B)ᶜ, ‖pHat ξ‖ := by
    rw [hconv]
    exact norm_synthesize_sub_band_le hpHat B s
  have halias := sum_fourier_alias_le_mass hb hB hh hsupp K s
  have htail := mul_le_mul_of_nonneg_left
    (sum_spatial_tail_le_density_bound b K s hh J) hh.le
  dsimp only
  change ‖synthesize pHat s - (h : ℂ) * ∑ j ∈ J, F (h * j)‖ ≤ _
  have hbound := (norm_sub_le_norm_sub_add_norm_sub
    (synthesize pHat s) (∫ c : ℝ, F c) ((h : ℂ) * ∑ j ∈ J, F (h * j))).trans
      (add_le_add hcut (hquad.trans (add_le_add halias htail)))
  simpa only [add_assoc, mul_assoc] using hbound

/-- The bandwidth-scaled real kernel has mass-preserving normalization. -/
def scaledRealKernel (K : ℝ → ℝ) (γ : ℝ) (u : ℝ) : ℝ := γ * K (γ * u)

/-- The r-th derivative of a scaled and translated activation has the required
kernel normalization. This includes r=0. -/
theorem iteratedDeriv_scaled_activation {φ K : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) {m : ℝ}
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (γ c s : ℝ) :
    iteratedDeriv r (fun x => φ (γ * (x - c))) s =
      γ^r * m * K (γ * (s - c)) := by
  have hshift := congrFun (iteratedDeriv_comp_sub_const r (fun u => φ (γ * u)) c) s
  rw [hshift, iteratedDeriv_comp_const_mul hφ γ]
  simp only [hkernel]
  ring

/-- The finite raw activation network. The coefficient factor is written as
h*γ/(m*γ^r), avoiding an erroneous natural-number r-1 exponent when r=0. -/
def rawActivationNetwork (φ ρ : ℝ → ℝ) (r : ℕ) (m γ h : ℝ)
    (J : Finset ℤ) (s : ℝ) : ℝ :=
  (h * γ / (m * γ^r)) * ∑ j ∈ J, ρ (h * j) * φ (γ * (s - h * j))

/-- Differentiating the actual finite network gives the finite kernel sum exactly. -/
theorem rawActivationNetwork_iteratedDeriv {φ K : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) {m γ : ℝ} (hm : m ≠ 0) (hγ : γ ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (ρ : ℝ → ℝ) (h : ℝ) (J : Finset ℤ) (s : ℝ) :
    iteratedDeriv r (rawActivationNetwork φ ρ r m γ h J) s =
      h * ∑ j ∈ J, ρ (h * j) * scaledRealKernel K γ (s - h * j) := by
  unfold rawActivationNetwork
  rw [iteratedDeriv_const_mul_field]
  have hsmooth : ∀ j ∈ J,
      ContDiffAt ℝ r (fun x => ρ (h * j) * φ (γ * (x - h * j))) s := by
    intro j hj
    exact ((contDiff_const).mul (hφ.comp (by fun_prop))).contDiffAt
  rw [iteratedDeriv_fun_sum hsmooth]
  simp only [iteratedDeriv_const_mul_field,
    iteratedDeriv_scaled_activation hφ hkernel]
  rw [Finset.mul_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  dsimp [scaledRealKernel]
  field_simp

/-- The raw finite activation network has the regularity required for anchoring. -/
theorem rawActivationNetwork_contDiff {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (ρ : ℝ → ℝ) (m γ h : ℝ) (J : Finset ℤ) :
    ContDiff ℝ r (rawActivationNetwork φ ρ r m γ h J) := by
  unfold rawActivationNetwork
  apply contDiff_const.mul
  apply ContDiff.sum
  intro j hj
  exact contDiff_const.mul (hφ.comp (by fun_prop))

/-- A real Schwartz kernel with positive bandwidth, bundled as a complex-valued
Schwartz function for the Fourier construction. This is an explicit operation. -/
def complexScaledKernel (K : 𝓢(ℝ, ℝ)) (γ : ℝ) (hγ : γ ≠ 0) : 𝓢(ℝ, ℂ) :=
  γ • SchwartzMap.postcompCLM (𝕜 := ℝ) Complex.ofRealCLM
    (SchwartzMap.compCLMOfContinuousLinearEquiv ℝ
      (ContinuousLinearEquiv.unitsEquivAut ℝ (Units.mk0 γ hγ)) K)

@[simp] theorem complexScaledKernel_apply (K : 𝓢(ℝ, ℝ)) (γ : ℝ) (hγ : γ ≠ 0) (u : ℝ) :
    complexScaledKernel K γ hγ u = (scaledRealKernel K γ u : ℂ) := by
  simp [complexScaledKernel, scaledRealKernel, Complex.real_smul, mul_comm]

/-- Real-part conversion preserves the proved complex error bound and gives
ordinary real kernel readouts. -/
theorem finite_spectral_real_kernel_certificate {pHat : ℝ → ℂ}
    (hpHat : Integrable pHat) (K : 𝓢(ℝ, ℝ)) {γ B h : ℝ}
    (hγ : γ ≠ 0) (hB : 0 ≤ B) (hh : 0 < h)
    (hnz : ∀ ξ ∈ Set.Icc (-B) B,
      𝓕 ((complexScaledKernel K γ hγ) : ℝ → ℂ) ξ ≠ 0)
    (s : ℝ) (J : Finset ℤ) :
    let Kγ := complexScaledKernel K γ hγ
    let b := bandSpectrum (fun ξ => pHat ξ / 𝓕 (Kγ : ℝ → ℂ) ξ) B
    let ρ := fun c => (synthesize b c).re
    |(synthesize pHat s).re - h * ∑ j ∈ J, ρ (h * j) *
      scaledRealKernel K γ (s - h * j)| ≤
      (∫ ξ in (Set.Icc (-B) B)ᶜ, ‖pHat ξ‖) +
      (∑' k : ℤ, if k = 0 then 0 else
        aliasMass b (𝓕 (Kγ : ℝ → ℂ)) ((k : ℝ) / h)) +
      h * (∫ ξ : ℝ, ‖b ξ‖) *
        ∑' j : ↑(J : Set ℤ)ᶜ, ‖Kγ (s - h * (j : ℤ))‖ := by
  have hc := finite_spectral_kernel_certificate hpHat (complexScaledKernel K γ hγ)
    hB hh hnz s J
  dsimp only at hc ⊢
  have hre := Complex.abs_re_le_norm
    (synthesize pHat s - (h : ℂ) * ∑ j ∈ J,
      synthesize (bandSpectrum (fun ξ => pHat ξ /
        𝓕 ((complexScaledKernel K γ hγ) : ℝ → ℂ) ξ) B) (h * j) *
          complexScaledKernel K γ hγ (s - h * j))
  have hresult := hre.trans hc
  simpa only [Complex.sub_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
    zero_mul, sub_zero, Complex.re_sum, complexScaledKernel_apply,
    mul_zero] using hresult

/-- The fully explicit pointwise cutoff--alias--halo certificate. -/
def kernelErrorCertificate (pHat K : ℝ → ℂ) (B h s : ℝ) (J : Finset ℤ) : ℝ :=
  let b := bandSpectrum (fun ξ => pHat ξ / 𝓕 K ξ) B
  (∫ ξ in (Set.Icc (-B) B)ᶜ, ‖pHat ξ‖) +
    (∑' k : ℤ, if k = 0 then 0 else aliasMass b (𝓕 K) ((k : ℝ) / h)) +
    h * (∫ ξ : ℝ, ‖b ξ‖) * ∑' j : ↑(J : Set ℤ)ᶜ, ‖K (s - h * (j : ℤ))‖

/-- The readout density for the real activation network, defined by a finite
spectral integral followed by taking its real part. -/
def realReadoutDensity (pHat : ℝ → ℂ) (K : 𝓢(ℝ, ℝ)) (γ : ℝ) (hγ : γ ≠ 0)
    (B : ℝ) (c : ℝ) : ℝ :=
  (cutoffDensity pHat (𝓕 ((complexScaledKernel K γ hγ) : ℝ → ℂ)) B c).re

/-- The actual finite activation network obeys the general-kernel derivative
certificate. The target derivative is supplied through its integrable spectrum.
The remaining Taylor anchoring converts this to a function-value error. -/
theorem rawActivationNetwork_derivative_certificate {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m γ B h : ℝ}
    (hm : m ≠ 0) (hγ : γ ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    {pHat : ℝ → ℂ} (hpHat : Integrable pHat)
    (hB : 0 ≤ B) (hh : 0 < h)
    (hnz : ∀ ξ ∈ Set.Icc (-B) B,
      𝓕 ((complexScaledKernel K γ hγ) : ℝ → ℂ) ξ ≠ 0)
    (s : ℝ) (J : Finset ℤ) :
    |(synthesize pHat s).re -
      iteratedDeriv r
        (rawActivationNetwork φ (realReadoutDensity pHat K γ hγ B) r m γ h J) s| ≤
      kernelErrorCertificate pHat (complexScaledKernel K γ hγ) B h s J := by
  rw [rawActivationNetwork_iteratedDeriv hφ hm hγ hkernel]
  exact finite_spectral_real_kernel_certificate hpHat K hγ hB hh hnz s J

/-- The mass-preserving bandwidth scaling divides frequency by gamma, with no
additional Fourier multiplier. -/
theorem fourier_scaled_kernel (K : ℝ → ℂ) {γ : ℝ} (hγ : 0 < γ) (ξ : ℝ) :
    𝓕 (fun u => (γ : ℂ) * K (γ * u)) ξ = 𝓕 K (ξ / γ) := by
  have hlinear : 𝓕 (fun u => (γ : ℂ) * K (γ * u)) ξ =
      (γ : ℂ) * 𝓕 (fun u => K (γ * u)) ξ := by
    simp only [Real.fourier_real_eq, Circle.smul_def, smul_eq_mul]
    rw [← integral_const_mul]
    apply integral_congr_ae
    filter_upwards with u
    ring
  rw [hlinear, fourier_comp_pos_mul K hγ, ← mul_assoc]
  norm_cast
  simp [hγ.ne']

/-- The bundled real scaled kernel has exactly the expected Fourier transform. -/
theorem complexScaledKernel_fourier (K : 𝓢(ℝ, ℝ)) {γ : ℝ} (hγ : 0 < γ) (ξ : ℝ) :
    𝓕 ((complexScaledKernel K γ hγ.ne') : ℝ → ℂ) ξ =
      𝓕 (fun u : ℝ => (K u : ℂ)) (ξ / γ) := by
  have heq : ((complexScaledKernel K γ hγ.ne') : ℝ → ℂ) =
      (fun u : ℝ => (γ : ℂ) * (K (γ * u) : ℂ)) := by
    funext u
    simp [scaledRealKernel]
  rw [heq]
  exact fourier_scaled_kernel (fun u => (K u : ℂ)) hγ ξ

/-- The explicit degree-below-r polynomial restoring prescribed derivative jets.
Its realization by the selected activation remains an architecture obligation. -/
def anchorPolynomial (r : ℕ) (a : ℕ → ℝ) (s : ℝ) : ℝ :=
  ∑ k ∈ Finset.range r, (a k / (k.factorial : ℝ)) * s^k

/-- The anchor polynomial is smooth at every order. -/
theorem anchorPolynomial_contDiff (r : ℕ) (a : ℕ → ℝ) (n : WithTop ℕ∞) :
    ContDiff ℝ n (anchorPolynomial r a) := by
  unfold anchorPolynomial
  fun_prop

/-- Every requested derivative at the anchor is exactly the supplied jet. -/
theorem anchorPolynomial_jet (r : ℕ) (a : ℕ → ℝ) {k : ℕ} (hk : k < r) :
    iteratedDeriv k (anchorPolynomial r a) 0 = a k := by
  unfold anchorPolynomial
  rw [iteratedDeriv_fun_sum (fun j hj => by fun_prop)]
  simp only [iteratedDeriv_const_mul_field, iteratedDeriv_fun_pow_zero]
  rw [Finset.sum_eq_single k]
  · simp only [if_true]
    exact div_mul_cancel₀ _ (by exact_mod_cast Nat.factorial_ne_zero k)
  · intro j hj hjk
    simp [hjk.symm]
  · intro hnot
    exact False.elim (hnot (Finset.mem_range.mpr hk))

/-- The polynomial cannot change the r-th derivative of the network. -/
theorem anchorPolynomial_top_derivative (r : ℕ) (a : ℕ → ℝ) (s : ℝ) :
    iteratedDeriv r (anchorPolynomial r a) s = 0 := by
  unfold anchorPolynomial
  rw [iteratedDeriv_fun_sum (fun j hj => by fun_prop)]
  apply Finset.sum_eq_zero
  intro j hj
  have hjr : j < r := Finset.mem_range.mp hj
  simp [iteratedDeriv_const_mul_field, iteratedDeriv_pow,
    Nat.descFactorial_eq_zero_iff_lt.mpr hjr]

/-- Add the exact polynomial needed to match the target's first r jets at zero.
For r=0 it is absent; for r=1 it is an ordinary output bias. -/
def anchoredActivationNetwork (φ ρ : ℝ → ℝ) (r : ℕ) (m γ h : ℝ)
    (J : Finset ℤ) (g : ℝ → ℝ) (s : ℝ) : ℝ :=
  let S := rawActivationNetwork φ ρ r m γ h J
  S s + anchorPolynomial r (fun k => iteratedDeriv k g 0 - iteratedDeriv k S 0) s

/-- The explicit anchor cancels all low-order jets of the actual approximation error. -/
theorem anchoredActivationNetwork_zeroJets {φ g : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (hg : ContDiffAt ℝ r g 0)
    (ρ : ℝ → ℝ) (m γ h : ℝ) (J : Finset ℤ) :
    ∀ k < r, iteratedDeriv k
      (fun s => anchoredActivationNetwork φ ρ r m γ h J g s - g s) 0 = 0 := by
  intro k hk
  let S := rawActivationNetwork φ ρ r m γ h J
  let a := fun j => iteratedDeriv j g 0 - iteratedDeriv j S 0
  have hS : ContDiffAt ℝ k S 0 :=
    ((rawActivationNetwork_contDiff hφ ρ m γ h J).of_le (by exact_mod_cast hk.le)).contDiffAt
  have hP : ContDiffAt ℝ k (anchorPolynomial r a) 0 :=
    (anchorPolynomial_contDiff r a k).contDiffAt
  have hgk : ContDiffAt ℝ k g 0 := hg.of_le (by exact_mod_cast hk.le)
  change iteratedDeriv k (fun s => (S s + anchorPolynomial r a s) - g s) 0 = 0
  rw [iteratedDeriv_fun_sub (hS.add hP) hgk, iteratedDeriv_fun_add hS hP,
    anchorPolynomial_jet r a hk]
  dsimp [a]
  ring

/-- The explicit anchor polynomial does not alter the constructed top derivative. -/
theorem anchoredActivationNetwork_top_derivative {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (ρ g : ℝ → ℝ) (m γ h : ℝ) (J : Finset ℤ) (s : ℝ) :
    iteratedDeriv r (anchoredActivationNetwork φ ρ r m γ h J g) s =
      iteratedDeriv r (rawActivationNetwork φ ρ r m γ h J) s := by
  let S := rawActivationNetwork φ ρ r m γ h J
  let a := fun j => iteratedDeriv j g 0 - iteratedDeriv j S 0
  change iteratedDeriv r (fun u => S u + anchorPolynomial r a u) s = _
  rw [iteratedDeriv_fun_add (rawActivationNetwork_contDiff hφ ρ m γ h J).contDiffAt
    (anchorPolynomial_contDiff r a r).contDiffAt, anchorPolynomial_top_derivative, add_zero]

/-- The anchored approximant is globally C^r independently of target regularity;
the target supplies only a finite list of real jet values in this definition. -/
theorem anchoredActivationNetwork_contDiff {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (ρ g : ℝ → ℝ) (m γ h : ℝ) (J : Finset ℤ) :
    ContDiff ℝ r (anchoredActivationNetwork φ ρ r m γ h J g) :=
  (rawActivationNetwork_contDiff hφ ρ m γ h J).add
    (anchorPolynomial_contDiff r _ r)

/-- For first-order derivative kernels, the anchor is literally one output bias. -/
theorem anchoredActivationNetwork_one (φ ρ g : ℝ → ℝ) (m γ h : ℝ)
    (J : Finset ℤ) (s : ℝ) :
    anchoredActivationNetwork φ ρ 1 m γ h J g s =
      rawActivationNetwork φ ρ 1 m γ h J s +
        (g 0 - rawActivationNetwork φ ρ 1 m γ h J 0) := by
  simp [anchoredActivationNetwork, anchorPolynomial]

/-- Raw kernel activations need no integration constant. -/
theorem anchoredActivationNetwork_zero (φ ρ g : ℝ → ℝ) (m γ h : ℝ)
    (J : Finset ℤ) (s : ℝ) :
    anchoredActivationNetwork φ ρ 0 m γ h J g s =
      rawActivationNetwork φ ρ 0 m γ h J s := by
  simp [anchoredActivationNetwork, anchorPolynomial]

/-- General finite activation construction with the sharp r-fold function-error
bound. The only uniform-bound premise concerns the explicit spectral cutoff,
alias and halo quantities, not the network's approximation error. Target
regularity is local to the closed target interval; no differentiability across
its endpoints is added. For r≥2, realizing the displayed anchor polynomial in
the activation architecture remains a separate obligation. -/
theorem finite_activation_function_certificate {φ g : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m γ B h T E : ℝ}
    (hm : m ≠ 0) (hγ : γ ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    {pHat : ℝ → ℂ} (hpHat : Integrable pHat)
    (hB : 0 ≤ B) (hh : 0 < h) (hT : 0 < T)
    (hnz : ∀ ξ ∈ Set.Icc (-B) B,
      𝓕 ((complexScaledKernel K γ hγ) : ℝ → ℂ) ξ ≠ 0)
    (J : Finset ℤ) (hg : ContDiffOn ℝ r g (Set.Icc (-T) T))
    (hprofile : ∀ s ∈ Set.Icc (-T) T,
      iteratedDeriv r g s = (synthesize pHat s).re)
    (hcertificate : ∀ s ∈ Set.Icc (-T) T,
      kernelErrorCertificate pHat (complexScaledKernel K γ hγ) B h s J ≤ E) :
    ∀ x ∈ Set.Icc (-T) T,
      |anchoredActivationNetwork φ (realReadoutDensity pHat K γ hγ B) r m γ h J g x - g x| ≤
        E * T^r / (r.factorial : ℝ) := by
  let ρ := realReadoutDensity pHat K γ hγ B
  have hderiv : ∀ y ∈ Set.Icc (-T) T,
      |iteratedDeriv r (rawActivationNetwork φ ρ r m γ h J) y - iteratedDeriv r g y| ≤ E := by
    intro y hy
    rw [hprofile y hy, abs_sub_comm]
    exact (rawActivationNetwork_derivative_certificate hφ K hm hγ hkernel hpHat
      hB hh hnz y J).trans (hcertificate y hy)
  cases r with
  | zero =>
    intro x hx
    simpa only [anchoredActivationNetwork_zero, iteratedDeriv_zero, pow_zero,
      Nat.factorial_zero, Nat.cast_one, mul_one, div_one, ρ] using hderiv x hx
  | succ n =>
    let q := anchoredActivationNetwork φ ρ (n + 1) m γ h J g
    have hq : ContDiff ℝ (n + 1 : ℕ) q :=
      anchoredActivationNetwork_contDiff hφ ρ g m γ h J
    have herror : ContDiffOn ℝ (n + 1 : ℕ) (fun s => q s - g s) (Set.Icc (-T) T) :=
      hq.contDiffOn.sub hg
    have hzero : ContDiffAt ℝ (n + 1 : ℕ) g 0 :=
      hg.contDiffAt (Icc_mem_nhds (by linarith) hT)
    have hjets : ∀ k < n + 1, iteratedDeriv k (fun s => q s - g s) 0 = 0 :=
      anchoredActivationNetwork_zeroJets hφ hzero ρ m γ h J
    have htop : ∀ y ∈ Set.Ioo (-T) T,
        |iteratedDeriv (n + 1) (fun s => q s - g s) y| ≤ E := by
      intro y hy
      have hgy : ContDiffAt ℝ (n + 1 : ℕ) g y :=
        hg.contDiffAt (Icc_mem_nhds hy.1 hy.2)
      rw [iteratedDeriv_fun_sub hq.contDiffAt hgy]
      change |iteratedDeriv (n + 1)
        (anchoredActivationNetwork φ ρ (n + 1) m γ h J g) y - iteratedDeriv (n + 1) g y| ≤ E
      rw [anchoredActivationNetwork_top_derivative hφ]
      exact hderiv y ⟨hy.1.le, hy.2.le⟩
    exact QIRadon.anchored_succ_iteratedDeriv_uniform n hT herror hjets htop

end QIRadon.Kernel
