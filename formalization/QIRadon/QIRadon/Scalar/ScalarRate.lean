import QIRadon.Kernel.Certificate
import QIRadon.Kernel.KernelRates

/-!
# Connecting kernel envelopes to the actual construction

All frequencies in this file use mathlib's cycles convention. Thus the sampling
spacing is `h = 1 / (H * B)`, and the aliases are at integer multiples of `H*B`.
No error bound for the constructed network is assumed here.
-/

noncomputable section
open MeasureTheory Set FourierTransform
open scoped Real SchwartzMap

namespace QIRadon.ScalarRate

open QIRadon.Kernel QIRadon.KernelRates

/-- The L1 size of the actual finite-band inverse-kernel spectrum. -/
theorem band_density_integral_bound {pHat Khat : ℝ → ℂ} {ν B γ L : ℝ}
    (E : SpectralEnvelope Khat ν) (hp : Integrable pHat)
    (hB : 0 < B) (hγ : 0 < γ) (hscale : (B / γ) ^ ν = L) :
    (∫ ξ : ℝ, ‖bandSpectrum (fun ξ => pHat ξ / Khat (ξ / γ)) B ξ‖) ≤
      (∫ ξ : ℝ, ‖pHat ξ‖) * ((1 / E.c) * Real.exp (E.A * L)) := by
  have hc := E.c_pos
  have hm := hp.norm.mul_const ((1 / E.c) * Real.exp (E.A * L))
  have hpoint : ∀ ξ : ℝ,
      ‖bandSpectrum (fun ξ => pHat ξ / Khat (ξ / γ)) B ξ‖ ≤
        ‖pHat ξ‖ * ((1 / E.c) * Real.exp (E.A * L)) := by
    intro ξ
    by_cases hx : ξ ∈ Icc (-B) B
    · simp only [bandSpectrum, indicator_of_mem hx]
      exact E.band_density_amplification _ hB hγ hscale (abs_le.mpr hx)
    · simp only [bandSpectrum, indicator_of_notMem hx, norm_zero]
      positivity
  have hb := integral_mono_of_nonneg
    (Filter.Eventually.of_forall (fun ξ => norm_nonneg _)) hm
    (Filter.Eventually.of_forall hpoint)
  simpa only [integral_mul_const] using hb

/-- Bounding one actual alias overlap integral from the spectral envelope.
The minus sign in Poisson summation is handled by the integer `-k`. -/
theorem alias_mass_bound {pHat Khat : ℝ → ℂ} {ν B γ L H : ℝ}
    (E : SpectralEnvelope Khat ν) (hp : Integrable pHat)
    (hB : 0 < B) (hγ : 0 < γ) (hL : 0 < L)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν) (k : ℤ) (hk : k ≠ 0) :
    aliasMass (bandSpectrum (fun ξ => pHat ξ / Khat (ξ / γ)) B)
      (fun ξ => Khat (ξ / γ)) ((k : ℝ) * (H * B)) ≤
      (∫ ξ : ℝ, ‖pHat ξ‖) *
        ((E.C / E.c) * Real.exp (-4 * L * |(k : ℝ)| ^ ν)) := by
  have hc := E.c_pos
  have hC := E.C_pos
  let C := (E.C / E.c) * Real.exp (-4 * L * |(k : ℝ)| ^ ν)
  have hpoint : ∀ ξ : ℝ,
      ‖bandSpectrum (fun ξ => pHat ξ / Khat (ξ / γ)) B ξ‖ *
        ‖Khat ((ξ - (k : ℝ) * (H * B)) / γ)‖ ≤ ‖pHat ξ‖ * C := by
    intro ξ
    by_cases hx : ξ ∈ Icc (-B) B
    · simp only [bandSpectrum, indicator_of_mem hx, norm_div]
      have hratio := E.integer_alias_bound (-k) hB hγ hL hscale hH hsep
        (abs_le.mpr hx) (neg_ne_zero.mpr hk)
      simp only [Int.cast_neg, abs_neg, neg_mul, ← sub_eq_add_neg] at hratio
      rw [norm_div] at hratio
      calc
        _ = ‖pHat ξ‖ * (‖Khat ((ξ - (k : ℝ) * (H * B)) / γ)‖ /
            ‖Khat (ξ / γ)‖) := by ring
        _ ≤ _ := by simpa only [C, neg_mul] using
          mul_le_mul_of_nonneg_left hratio (norm_nonneg (pHat ξ))
    · simp only [bandSpectrum, indicator_of_notMem hx, norm_zero, zero_mul]
      dsimp [C]
      positivity
  have hb := integral_mono_of_nonneg
    (Filter.Eventually.of_forall (fun ξ => mul_nonneg (norm_nonneg _) (norm_nonneg _)))
    (hp.norm.mul_const C) (Filter.Eventually.of_forall hpoint)
  simpa only [aliasMass, integral_mul_const, C] using hb

/-- All aliases of the actual finite-band construction, with an explicit
constant. Integrating the target spectrum does not introduce a factor B. -/
theorem alias_mass_series_bound {pHat Khat : ℝ → ℂ} {ν B γ L H : ℝ}
    (E : SpectralEnvelope Khat ν) (hp : Integrable pHat)
    (hB : 0 < B) (hγ : 0 < γ) (hL : 1 ≤ L)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν) :
    let aliases : ℤ → ℝ := fun k => if k = 0 then 0 else
      aliasMass (bandSpectrum (fun ξ => pHat ξ / Khat (ξ / γ)) B)
        (fun ξ => Khat (ξ / γ)) ((k : ℝ) * (H * B))
    Summable aliases ∧ (∑' k, aliases k) ≤
      (∫ ξ : ℝ, ‖pHat ξ‖) * (E.C / E.c) * Real.exp (-2 * L) *
        (1 + 2 * tailConstant 2 ν) := by
  dsimp only
  have hc := E.c_pos
  have hC := E.C_pos
  let A := (∫ ξ : ℝ, ‖pHat ξ‖) * (E.C / E.c) * Real.exp (-2 * L)
  have hA : 0 ≤ A := by dsimp [A]; positivity
  obtain ⟨hs, hb⟩ := integer_rpow_exponential_sum
    (by norm_num : (0 : ℝ) < 2) E.exponent_pos
  rw [← tailConstant_eq_integral (by norm_num : (0 : ℝ) < 2) E.exponent_pos] at hb
  have hmajor := hs.mul_left A
  have hle : ∀ k : ℤ, (if k = 0 then 0 else
      aliasMass (bandSpectrum (fun ξ => pHat ξ / Khat (ξ / γ)) B)
        (fun ξ => Khat (ξ / γ)) ((k : ℝ) * (H * B))) ≤
      A * Real.exp (-2 * |(k : ℝ)| ^ ν) := by
    intro k
    split_ifs with hk
    · positivity
    · have hmass := alias_mass_bound E hp hB hγ (by linarith) hscale hH hsep k hk
      have hk1 : 1 ≤ |(k : ℝ)| := by exact_mod_cast Int.one_le_abs hk
      have hfac := alias_exponential_factor hL
        (Real.one_le_rpow hk1 E.exponent_pos.le)
      apply hmass.trans
      have hm := mul_le_mul_of_nonneg_left hfac
        (mul_nonneg (integral_nonneg (μ := volume) (fun ξ => norm_nonneg (pHat ξ)))
          (div_pos E.C_pos E.c_pos).le)
      simpa only [A, mul_assoc] using hm
  have ha := hmajor.of_nonneg_of_le (fun k => by
    split_ifs
    · exact le_refl 0
    · exact aliasMass_nonneg _ _ _) hle
  refine ⟨ha, ?_⟩
  calc
    _ ≤ ∑' k : ℤ, A * Real.exp (-2 * |(k : ℝ)| ^ ν) := ha.tsum_le_tsum hle hmajor
    _ = A * ∑' k : ℤ, Real.exp (-2 * |(k : ℝ)| ^ ν) := tsum_mul_left
    _ ≤ A * (1 + 2 * tailConstant 2 ν) := mul_le_mul_of_nonneg_left hb hA

/-- Enumerate precisely the two half-grids omitted by a symmetric finite grid. -/
def outsideIndex (J : ℕ) : ℕ ⊕ ℕ → ↑((Finset.Icc (-(J : ℤ)) (J : ℤ) : Finset ℤ) : Set ℤ)ᶜ
  | Sum.inl n => ⟨(J : ℤ) + 1 + n, by simp only [Set.mem_compl_iff, Finset.mem_coe, Finset.mem_Icc]; omega⟩
  | Sum.inr n => ⟨-((J : ℤ) + 1 + n), by simp only [Set.mem_compl_iff, Finset.mem_coe, Finset.mem_Icc]; omega⟩

theorem outsideIndex_bijective (J : ℕ) : Function.Bijective (outsideIndex J) := by
  constructor
  · intro a b hab
    have hv := congrArg Subtype.val hab
    cases a with
    | inl n =>
      cases b with
      | inl m => simp only [outsideIndex] at hv; congr 1; omega
      | inr m => simp only [outsideIndex] at hv; omega
    | inr n =>
      cases b with
      | inl m => simp only [outsideIndex] at hv; omega
      | inr m => simp only [outsideIndex] at hv; congr 1; omega
  · intro j
    have hj := j.property
    simp only [Set.mem_compl_iff, Finset.mem_coe, Finset.mem_Icc] at hj
    by_cases hz : (J : ℤ) < j.val
    · refine ⟨Sum.inl (j.val - J - 1).toNat, ?_⟩
      apply Subtype.ext
      simp only [outsideIndex]
      omega
    · refine ⟨Sum.inr (-j.val - J - 1).toNat, ?_⟩
      apply Subtype.ext
      simp only [outsideIndex]
      omega

/-- Exact reindexing of the halo. No center is discarded from the certificate. -/
theorem omitted_grid_sum {F : ℤ → ℝ} (hF : Summable F) (J : ℕ) :
    (∑' j : ↑((Finset.Icc (-(J : ℤ)) (J : ℤ) : Finset ℤ) : Set ℤ)ᶜ, F j) =
      ∑' n : ℕ, (F ((J : ℤ) + 1 + n) + F (-((J : ℤ) + 1 + n))) := by
  let e := Equiv.ofBijective (outsideIndex J) (outsideIndex_bijective J)
  have he := e.tsum_eq (fun j => F j.val)
  have hp : Summable (fun n : ℕ => F ((J : ℤ) + 1 + n)) :=
    hF.comp_injective (i := fun n : ℕ => (J : ℤ) + 1 + n) (by intro a b hab; dsimp only at hab; omega)
  have hn : Summable (fun n : ℕ => F (-((J : ℤ) + 1 + n))) :=
    hF.comp_injective (i := fun n : ℕ => -((J : ℤ) + 1 + n)) (by intro a b hab; dsimp only at hab; omega)
  rw [hp.tsum_add hn]
  rw [← he]
  exact hp.tsum_sum hn

/-- Spatial envelope bound for exactly the subtype tail in the master certificate. -/
theorem finite_halo_tail_subtype (K : 𝓢(ℝ, ℝ)) {μ : ℝ}
    (S : SpatialEnvelope K μ) {T D h γ s : ℝ} (J : ℕ)
    (hh : 0 < h) (hγ : 0 < γ) (hD : 0 ≤ D)
    (hs : |s| ≤ T) (hcut : T + D ≤ ((J : ℝ) + 1) * h) :
    h * (∑' j : ↑((Finset.Icc (-(J : ℤ)) (J : ℤ) : Finset ℤ) : Set ℤ)ᶜ,
      ‖complexScaledKernel K γ hγ.ne' (s - h * (j : ℤ))‖) ≤
        2 * S.C * Real.exp (-(S.b / 2) * (γ * D) ^ μ) *
          (γ * h + tailConstant (S.b / 2) μ) := by
  have hsum := (summable_translated_kernel (complexScaledKernel K γ hγ.ne') s hh).norm
  rw [omitted_grid_sum hsum J]
  convert (S.finite_halo_tail J hh hγ hD hs hcut).2 using 1
  congr 1
  apply tsum_congr
  intro n
  simp only [complexScaledKernel_apply, scaledRealKernel, Complex.norm_real,
    Real.norm_eq_abs, Int.cast_add, Int.cast_neg, Int.cast_natCast, Int.cast_one]
  congr 2 <;> ring_nf

/-- Three explicit finite-construction errors after substituting the actual
spectral and spatial envelopes. The only target term not yet specialized is its
Fourier tail, allowing the same theorem to serve analytic and algebraic data. -/
theorem finite_kernel_envelope_certificate
    (K : 𝓢(ℝ, ℝ)) {ν μ : ℝ}
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {B γ L H T D s : ℝ} (J : ℕ)
    (hB : 0 < B) (hγ : 0 < γ) (hL : 1 ≤ L)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hD : 0 ≤ D) (hs : |s| ≤ T)
    (hcut : T + D ≤ ((J : ℝ) + 1) * (1 / (H * B))) :
    kernelErrorCertificate pHat (complexScaledKernel K γ hγ.ne') B
      (1 / (H * B)) s (Finset.Icc (-(J : ℤ)) (J : ℤ)) ≤
      (∫ ξ in (Icc (-B) B)ᶜ, ‖pHat ξ‖) +
      (∫ ξ : ℝ, ‖pHat ξ‖) * (E.C / E.c) * Real.exp (-2 * L) *
        (1 + 2 * tailConstant 2 ν) +
      (∫ ξ : ℝ, ‖pHat ξ‖) * ((1 / E.c) * Real.exp (E.A * L)) *
        (2 * S.C * Real.exp (-(S.b / 2) * (γ * D) ^ μ) *
          (γ * (1 / (H * B)) + tailConstant (S.b / 2) μ)) := by
  have hH0 : 0 < H := by linarith
  have hh : 0 < 1 / (H * B) := one_div_pos.mpr (mul_pos hH0 hB)
  have hden := band_density_integral_bound E hp hB hγ hscale
  have halias := (alias_mass_series_bound E hp hB hγ hL hscale hH hsep).2
  have htail := finite_halo_tail_subtype K S J hh hγ hD hs hcut
  have hf : 𝓕 ((complexScaledKernel K γ hγ.ne') : ℝ → ℂ) =
      fun ξ => 𝓕 (fun u : ℝ => (K u : ℂ)) (ξ / γ) :=
    funext (complexScaledKernel_fourier K hγ)
  unfold kernelErrorCertificate
  rw [hf]
  dsimp only
  have ha : (∑' k : ℤ, if k = 0 then 0 else
      aliasMass (bandSpectrum (fun ξ => pHat ξ /
        𝓕 (fun u : ℝ => (K u : ℂ)) (ξ / γ)) B)
        (fun ξ => 𝓕 (fun u : ℝ => (K u : ℂ)) (ξ / γ))
        ((k : ℝ) / (1 / (H * B)))) ≤
      (∫ ξ : ℝ, ‖pHat ξ‖) * (E.C / E.c) * Real.exp (-2 * L) *
        (1 + 2 * tailConstant 2 ν) := by
    simpa only [div_eq_mul_inv, one_div, inv_inv, one_mul] using halias
  apply add_le_add (add_le_add le_rfl ha) ?_
  have hn : 0 ≤ (∫ ξ : ℝ, ‖bandSpectrum (fun ξ => pHat ξ /
      𝓕 (fun u : ℝ => (K u : ℂ)) (ξ / γ)) B ξ‖) :=
    integral_nonneg (fun _ => norm_nonneg _)
  have htn : 0 ≤ 2 * S.C * Real.exp (-(S.b / 2) * (γ * D) ^ μ) *
      (γ * (1 / (H * B)) + tailConstant (S.b / 2) μ) := by
    have hc := S.C_pos
    have ht := tailConstant_nonneg (half_pos S.b_pos) S.exponent_pos
    positivity
  calc
    _ = (∫ ξ : ℝ, ‖bandSpectrum (fun ξ => pHat ξ /
        𝓕 (fun u : ℝ => (K u : ℂ)) (ξ / γ)) B ξ‖) *
        ((1 / (H * B)) * ∑' j : ↑((Finset.Icc (-(J : ℤ)) (J : ℤ) : Finset ℤ) : Set ℤ)ᶜ,
          ‖complexScaledKernel K γ hγ.ne' (s - (1 / (H * B)) * (j : ℤ))‖) := by ring
    _ ≤ _ := (mul_le_mul_of_nonneg_left htail hn).trans
      (mul_le_mul_of_nonneg_right hden htn)

/-- The prescribed bandwidth, spacing, and halo convert the explicit certificate
into a Fourier tail plus exponentially small sampling and halo terms. -/
theorem scheduled_kernel_certificate
    (K : 𝓢(ℝ, ℝ)) {ν μ : ℝ}
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {B L H T Z s : ℝ}
    (hB : 0 < B) (hL : 1 ≤ L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hs : |s| ≤ T) :
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := ⌊(T + D) / h⌋₊
    kernelErrorCertificate pHat (complexScaledKernel K γ (by positivity)) B h s
      (Finset.Icc (-(J : ℤ)) (J : ℤ)) ≤
      (∫ ξ in (Icc (-B) B)ᶜ, ‖pHat ξ‖) +
      (∫ ξ : ℝ, ‖pHat ξ‖) * (E.C / E.c) * Real.exp (-2 * L) *
        (1 + 2 * tailConstant 2 ν) +
      (∫ ξ : ℝ, ‖pHat ξ‖) * (2 * S.C / E.c) *
        (1 + tailConstant (S.b / 2) μ) * Real.exp (-3 * L) := by
  have hL0 : 0 < L := by linarith
  have hH0 : 0 < H := by linarith
  let γ := B / L ^ (1 / ν)
  let h := 1 / (H * B)
  let D := Z * L ^ (1 / μ) / γ
  let J := ⌊(T + D) / h⌋₊
  have hγ : 0 < γ := div_pos hB (Real.rpow_pos_of_pos hL0 _)
  have hh : 0 < h := one_div_pos.mpr (mul_pos hH0 hB)
  have hD : 0 ≤ D := div_nonneg (mul_nonneg hZ (Real.rpow_nonneg hL0.le _)) hγ.le
  have hc := E.c_pos
  have hCs := S.C_pos
  have hSconst := tailConstant_nonneg (half_pos S.b_pos) S.exponent_pos
  have hbound := finite_kernel_envelope_certificate K E S hp J hB hγ hL
    (normalized_bandwidth_power hB hL0 E.exponent_pos) hH hsep hD hs
    (floor_halo_cutoff hh)
  dsimp only at hbound ⊢
  apply hbound.trans
  apply add_le_add le_rfl
  have hspacing : γ * h ≤ 1 := by
    have hpow : 1 ≤ L ^ (1 / ν) :=
      Real.one_le_rpow hL (one_div_pos.mpr E.exponent_pos).le
    have hid : γ * h = 1 / (H * L ^ (1 / ν)) := by dsimp [γ, h]; field_simp
    rw [hid]
    apply (div_le_iff₀ (mul_pos hH0 (Real.rpow_pos_of_pos hL0 _))).mpr
    nlinarith
  have hcancel : Real.exp (E.A * L) * Real.exp (-(S.b / 2) * (γ * D) ^ μ) ≤
      Real.exp (-3 * L) := by
    have hid : γ * D = Z * L ^ (1 / μ) := by dsimp [D]; field_simp
    rw [hid]
    exact inverse_times_halo_exponential S.b_pos S.exponent_pos hZ hL0 hhalo
  let A := ∫ ξ : ℝ, ‖pHat ξ‖
  have hA : 0 ≤ A := integral_nonneg (fun _ => norm_nonneg _)
  change A * (1 / E.c * Real.exp (E.A * L)) *
      (2 * S.C * Real.exp (-(S.b / 2) * (γ * D) ^ μ) *
        (γ * h + tailConstant (S.b / 2) μ)) ≤ _
  calc
    _ = A * (2 * S.C / E.c) *
        (Real.exp (E.A * L) * Real.exp (-(S.b / 2) * (γ * D) ^ μ)) *
        (γ * h + tailConstant (S.b / 2) μ) := by ring
    _ ≤ A * (2 * S.C / E.c) * Real.exp (-3 * L) *
        (1 + tailConstant (S.b / 2) μ) :=
      mul_le_mul
        (mul_le_mul_of_nonneg_left hcancel (by positivity))
        (add_le_add hspacing le_rfl) (by positivity) (by positivity)
    _ = _ := by dsimp [A]; ring

/-- Fixed kernel-dependent factor; it is independent of accuracy and B. -/
def envelopeConstant {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ) : ℝ :=
  1 + (E.C / E.c) * (1 + 2 * tailConstant 2 ν) +
    (2 * S.C / E.c) * (1 + tailConstant (S.b / 2) μ)

theorem envelopeConstant_pos {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ) :
    0 < envelopeConstant E S := by
  have hc := E.c_pos
  have hC := E.C_pos
  have hCs := S.C_pos
  have ht := tailConstant_nonneg (by norm_num : (0 : ℝ) < 2) E.exponent_pos
  have hts := tailConstant_nonneg (half_pos S.b_pos) S.exponent_pos
  unfold envelopeConstant
  positivity

/-- The global exponential Fourier moment controls the ordinary spectral mass. -/
theorem spectral_mass_le_weighted {pHat : ℝ → ℂ} {σ : ℝ}
    (hp : Integrable pHat) (hσ : 0 ≤ σ)
    (hW : Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) * ‖pHat ξ‖)) :
    (∫ ξ : ℝ, ‖pHat ξ‖) ≤ ∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat ξ‖ := by
  apply integral_mono hp.norm hW
  intro ξ
  have he := Real.one_le_exp_iff.mpr (mul_nonneg hσ (abs_nonneg ξ))
  simpa only [one_mul] using mul_le_mul_of_nonneg_right he (norm_nonneg (pHat ξ))

/-- Actual geometric scalar certificate for a global exponential Fourier moment.
This theorem has no assumed quadrature or deconvolution approximation bound. -/
theorem analytic_kernel_certificate
    (K : 𝓢(ℝ, ℝ)) {ν μ : ℝ}
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {σ L H T Z s : ℝ} (hσ : 0 < σ)
    (hW : Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) * ‖pHat ξ‖))
    (hL : 1 ≤ L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hs : |s| ≤ T) :
    let B := L / σ
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := ⌊(T + D) / h⌋₊
    kernelErrorCertificate pHat (complexScaledKernel K γ (by positivity)) B h s
      (Finset.Icc (-(J : ℤ)) (J : ℤ)) ≤
      (∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat ξ‖) *
        envelopeConstant E S * Real.exp (-L) := by
  have hL0 : 0 < L := by linarith
  have hB : 0 < L / σ := div_pos hL0 hσ
  have hb := scheduled_kernel_certificate K E S hp hB hL hH hsep hZ hhalo hs
  dsimp only at hb ⊢
  apply hb.trans
  have hcut := exponential_spectral_cutoff hσ.le hp hW (B := L / σ)
  have hset : (Icc (-(L / σ)) (L / σ))ᶜ = {ξ : ℝ | L / σ < |ξ|} := by
    ext ξ
    simp only [mem_compl_iff, mem_Icc, mem_ofPred_eq, ← abs_le, not_le]
  rw [← hset] at hcut
  have hexp : -σ * (L / σ) = -L := by field_simp
  rw [hexp] at hcut
  let A := ∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat ξ‖
  let a := ∫ ξ : ℝ, ‖pHat ξ‖
  have hA : 0 ≤ A := integral_nonneg (fun _ => by positivity)
  have ha : 0 ≤ a := integral_nonneg (fun _ => norm_nonneg _)
  have haA : a ≤ A := spectral_mass_le_weighted hp hσ.le hW
  have he2 : Real.exp (-2 * L) ≤ Real.exp (-L) := Real.exp_le_exp.mpr (by linarith)
  have he3 : Real.exp (-3 * L) ≤ Real.exp (-L) := Real.exp_le_exp.mpr (by linarith)
  have hc := E.c_pos
  have hC := E.C_pos
  have hCs := S.C_pos
  have ht := tailConstant_nonneg (by norm_num : (0 : ℝ) < 2) E.exponent_pos
  have hts := tailConstant_nonneg (half_pos S.b_pos) S.exponent_pos
  have halias : a * (E.C / E.c) * Real.exp (-2 * L) * (1 + 2 * tailConstant 2 ν) ≤
      A * (E.C / E.c) * Real.exp (-L) * (1 + 2 * tailConstant 2 ν) := by
    gcongr
  have htail : a * (2 * S.C / E.c) * (1 + tailConstant (S.b / 2) μ) * Real.exp (-3 * L) ≤
      A * (2 * S.C / E.c) * (1 + tailConstant (S.b / 2) μ) * Real.exp (-L) := by
    gcongr
  calc
    _ ≤ Real.exp (-L) * A +
        A * (E.C / E.c) * Real.exp (-L) * (1 + 2 * tailConstant 2 ν) +
        A * (2 * S.C / E.c) * (1 + tailConstant (S.b / 2) μ) * Real.exp (-L) :=
      add_le_add (add_le_add hcut halias) htail
    _ = _ := by dsimp [A, envelopeConstant]; ring

/-- Explicit accuracy selection, including the zero-target case. -/
theorem log_accuracy_choice {C ε : ℝ} (hC : 0 ≤ C) (hε : 0 < ε) :
    C * Real.exp (-max 1 (Real.log (C / ε))) ≤ ε := by
  rcases hC.eq_or_lt with hzero | hpos
  · simp only [← hzero, zero_mul]
    exact hε.le
  · have hlog := le_max_right (1 : ℝ) (Real.log (C / ε))
    have he := Real.exp_le_exp.mpr (neg_le_neg hlog)
    have hratio : 0 < C / ε := div_pos hpos hε
    rw [Real.exp_neg (Real.log (C / ε)), Real.exp_log hratio] at he
    have hm := mul_le_mul_of_nonneg_left he hC
    have hid : C * (C / ε)⁻¹ = ε := by field_simp
    simpa only [hid] using hm

/-- The actual centered grid has this exact neuron count, including both halos. -/
theorem symmetric_grid_card (J : ℕ) :
    (Finset.Icc (-(J : ℤ)) (J : ℤ)).card = 2 * J + 1 := by
  rw [Int.card_Icc]
  omega

/-- Width bound in the cycles convention used by the actual Fourier theorem.
This explicitly converts the angular-frequency count, including its 2π. -/
theorem analytic_grid_width_cycles {T σ L μ ν Z H : ℝ}
    (hT : 0 ≤ T) (hσ : 0 < σ) (hL : 1 ≤ L)
    (hZ : 0 ≤ Z) (hH : 0 < H) :
    let B := L / σ
    let γ := B / L ^ (1 / ν)
    let D := Z * L ^ (1 / μ) / γ
    let h := 1 / (H * B)
    let J := ⌊(T + D) / h⌋₊
    ((Finset.Icc (-(J : ℤ)) (J : ℤ)).card : ℝ) ≤
      (2 * H * T / σ + 2 * H * Z + 1) * L ^ max 1 (1 / μ + 1 / ν) := by
  dsimp only
  rw [symmetric_grid_card]
  push_cast
  have hb := analytic_grid_width_power_bound (μ := μ) (ν := ν)
    (H := 2 * Real.pi * H) (b₀ := 1 / σ)
    hT (one_div_pos.mpr hσ) hL hZ (mul_pos (mul_pos (by norm_num) Real.pi_pos) hH)
  dsimp only at hb
  have hB : (1 / σ) * L = L / σ := by ring
  have hh : 2 * Real.pi / (2 * Real.pi * H * (L / σ)) = 1 / (H * (L / σ)) := by
    field_simp
  have hc : (2 * Real.pi * H * T * (1 / σ) / Real.pi +
      2 * Real.pi * H * Z / Real.pi + 1) = 2 * H * T / σ + 2 * H * Z + 1 := by
    field_simp
  simpa only [hB, hh, hc] using hb

/-- Actual activation network, with the explicit integration-jet polynomial.
For r=0 this is exactly the raw MLP; for r=1 the anchor is a single output bias.
For larger r its implementation cost is still a separate declared obligation. -/
theorem analytic_activation_error
    {φ g : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {σ L H T Z : ℝ} (hσ : 0 < σ)
    (hW : Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) * ‖pHat ξ‖))
    (hL : 1 ≤ L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hT : 0 < T) (hg : ContDiffOn ℝ r g (Icc (-T) T))
    (hprofile : ∀ s ∈ Icc (-T) T, iteratedDeriv r g s = (synthesize pHat s).re) :
    let B := L / σ
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := Finset.Icc (-(⌊(T + D) / h⌋₊ : ℤ)) (⌊(T + D) / h⌋₊ : ℤ)
    ∀ x ∈ Icc (-T) T,
      |anchoredActivationNetwork φ (realReadoutDensity pHat K γ (by positivity) B)
        r m γ h J g x - g x| ≤
      ((∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat ξ‖) * envelopeConstant E S *
        Real.exp (-L)) * T^r / (r.factorial : ℝ) := by
  have hL0 : 0 < L := by linarith
  have hH0 : 0 < H := by linarith
  have hB : 0 < L / σ := div_pos hL0 hσ
  have hγ : 0 < (L / σ) / L ^ (1 / ν) := div_pos hB (Real.rpow_pos_of_pos hL0 _)
  apply finite_activation_function_certificate hφ K hm hγ.ne' hkernel hp hB.le
    (one_div_pos.mpr (mul_pos hH0 hB)) hT
  · intro ξ _
    rw [complexScaledKernel_fourier K hγ]
    exact E.nonzero _
  · exact hg
  · exact hprofile
  · intro s hs
    exact analytic_kernel_certificate K E S hp hσ hW hL hH hsep hZ hhalo
      (abs_le.mpr hs)

/-- A single explicit accuracy choice simultaneously certifies the actual
activation network and counts every grid neuron. The polynomial anchor overhead
for r≥2 is deliberately not erased from the architecture discussion. -/
theorem analytic_accuracy_and_width
    {φ g : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {σ H T Z ε : ℝ} (hσ : 0 < σ)
    (hW : Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) * ‖pHat ξ‖))
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hT : 0 < T) (hε : 0 < ε) (hg : ContDiffOn ℝ r g (Icc (-T) T))
    (hprofile : ∀ s ∈ Icc (-T) T, iteratedDeriv r g s = (synthesize pHat s).re) :
    let C := (∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat ξ‖) *
      envelopeConstant E S * T^r / (r.factorial : ℝ)
    let L := max 1 (Real.log (C / ε))
    let B := L / σ
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := Finset.Icc (-(⌊(T + D) / h⌋₊ : ℤ)) (⌊(T + D) / h⌋₊ : ℤ)
    (∀ x ∈ Icc (-T) T,
      |anchoredActivationNetwork φ (realReadoutDensity pHat K γ (by positivity) B)
        r m γ h J g x - g x| ≤ ε) ∧
    (J.card : ℝ) ≤ (2 * H * T / σ + 2 * H * Z + 1) * L ^ max 1 (1 / μ + 1 / ν) := by
  dsimp only
  let C := (∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat ξ‖) *
    envelopeConstant E S * T^r / (r.factorial : ℝ)
  let L := max 1 (Real.log (C / ε))
  have hL : 1 ≤ L := le_max_left _ _
  have hC : 0 ≤ C := by
    have he := (envelopeConstant_pos E S).le
    dsimp [C]
    positivity
  have heps := log_accuracy_choice hC hε
  constructor
  · intro x hx
    have hb := analytic_activation_error hφ K hm hkernel E S hp hσ hW hL hH hsep
      hZ hhalo hT hg hprofile x hx
    apply hb.trans
    calc
      _ = C * Real.exp (-L) := by dsimp [C]; ring
      _ ≤ ε := heps
  · exact analytic_grid_width_cycles hT.le hσ hL hZ (by linarith)

/-- Algebraic regularity controls the ordinary spectral mass as well. -/
theorem spectral_mass_le_polynomial {pHat : ℝ → ℂ} {s : ℝ}
    (hp : Integrable pHat) (hs : 0 ≤ s)
    (hW : Integrable (fun ξ : ℝ => (1 + |ξ|) ^ s * ‖pHat ξ‖)) :
    (∫ ξ : ℝ, ‖pHat ξ‖) ≤ ∫ ξ : ℝ, (1 + |ξ|) ^ s * ‖pHat ξ‖ := by
  apply integral_mono hp.norm hW
  intro ξ
  have he := Real.one_le_rpow (show 1 ≤ 1 + |ξ| by linarith [abs_nonneg ξ]) hs
  simpa only [one_mul] using mul_le_mul_of_nonneg_right he (norm_nonneg (pHat ξ))

/-- Actual scalar algebraic certificate, keeping B and L independent so the
frequency resolution and exponentially small aliases can be budgeted separately. -/
theorem algebraic_kernel_certificate
    (K : 𝓢(ℝ, ℝ)) {ν μ : ℝ}
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {s B L H T Z x : ℝ} (hs : 0 ≤ s) (hB : 0 < B)
    (hW : Integrable (fun ξ : ℝ => (1 + |ξ|) ^ s * ‖pHat ξ‖))
    (hL : 1 ≤ L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hx : |x| ≤ T) :
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := ⌊(T + D) / h⌋₊
    kernelErrorCertificate pHat (complexScaledKernel K γ (by positivity)) B h x
      (Finset.Icc (-(J : ℤ)) (J : ℤ)) ≤
      (∫ ξ : ℝ, (1 + |ξ|) ^ s * ‖pHat ξ‖) *
        ((1 + B)^(-s) + (envelopeConstant E S - 1) * Real.exp (-2 * L)) := by
  have hb := scheduled_kernel_certificate K E S hp hB hL hH hsep hZ hhalo hx
  dsimp only at hb ⊢
  apply hb.trans
  have hcut := polynomial_spectral_cutoff hs hB.le hp hW
  have hset : (Icc (-B) B)ᶜ = {ξ : ℝ | B < |ξ|} := by
    ext ξ
    simp only [mem_compl_iff, mem_Icc, mem_ofPred_eq, ← abs_le, not_le]
  rw [← hset] at hcut
  let A := ∫ ξ : ℝ, (1 + |ξ|) ^ s * ‖pHat ξ‖
  let a := ∫ ξ : ℝ, ‖pHat ξ‖
  have hA : 0 ≤ A := integral_nonneg (fun _ => by positivity)
  have ha : 0 ≤ a := integral_nonneg (fun _ => norm_nonneg _)
  have haA : a ≤ A := spectral_mass_le_polynomial hp hs hW
  have he : Real.exp (-3 * L) ≤ Real.exp (-2 * L) := Real.exp_le_exp.mpr (by linarith)
  have hc := E.c_pos
  have hC := E.C_pos
  have hCs := S.C_pos
  have ht := tailConstant_nonneg (by norm_num : (0 : ℝ) < 2) E.exponent_pos
  have hts := tailConstant_nonneg (half_pos S.b_pos) S.exponent_pos
  have halias : a * (E.C / E.c) * Real.exp (-2 * L) * (1 + 2 * tailConstant 2 ν) ≤
      A * (E.C / E.c) * Real.exp (-2 * L) * (1 + 2 * tailConstant 2 ν) := by
    gcongr
  have htail : a * (2 * S.C / E.c) * (1 + tailConstant (S.b / 2) μ) * Real.exp (-3 * L) ≤
      A * (2 * S.C / E.c) * (1 + tailConstant (S.b / 2) μ) * Real.exp (-2 * L) := by
    gcongr
  calc
    _ ≤ (1 + B)^(-s) * A +
        A * (E.C / E.c) * Real.exp (-2 * L) * (1 + 2 * tailConstant 2 ν) +
        A * (2 * S.C / E.c) * (1 + tailConstant (S.b / 2) μ) * Real.exp (-2 * L) :=
      add_le_add (add_le_add hcut halias) htail
    _ = _ := by dsimp [A, envelopeConstant]; ring

/-- Select only logarithmically many alias-suppression units for an algebraic
cutoff B. This avoids paying a polynomial price for suppressing the aliases. -/
theorem algebraic_alias_schedule {s B : ℝ} (hB : 0 < B) :
    Real.exp (-2 * max 1 ((s / 2) * Real.log (1 + B))) ≤ (1 + B)^(-s) := by
  rw [Real.rpow_def_of_pos (by linarith : 0 < 1 + B)]
  apply Real.exp_le_exp.mpr
  have hl := le_max_right (1 : ℝ) ((s / 2) * Real.log (1 + B))
  nlinarith

/-- Fully scheduled algebraic certificate for the actual finite kernel sum. -/
theorem algebraic_scheduled_certificate
    (K : 𝓢(ℝ, ℝ)) {ν μ : ℝ}
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {s B H T Z x : ℝ} (hs : 0 ≤ s) (hB : 0 < B)
    (hW : Integrable (fun ξ : ℝ => (1 + |ξ|) ^ s * ‖pHat ξ‖))
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hx : |x| ≤ T) :
    let L := max 1 ((s / 2) * Real.log (1 + B))
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := ⌊(T + D) / h⌋₊
    kernelErrorCertificate pHat (complexScaledKernel K γ (by positivity)) B h x
      (Finset.Icc (-(J : ℤ)) (J : ℤ)) ≤
      (∫ ξ : ℝ, (1 + |ξ|) ^ s * ‖pHat ξ‖) * envelopeConstant E S * (1 + B)^(-s) := by
  have hb := algebraic_kernel_certificate K E S hp hs hB hW
    (le_max_left 1 ((s / 2) * Real.log (1 + B)))
    hH hsep hZ hhalo hx
  dsimp only at hb ⊢
  apply hb.trans
  have hA : 0 ≤ ∫ ξ : ℝ, (1 + |ξ|) ^ s * ‖pHat ξ‖ :=
    integral_nonneg (fun _ => by positivity)
  have hc := E.c_pos
  have hC := E.C_pos
  have hCs := S.C_pos
  have ht := tailConstant_nonneg (by norm_num : (0 : ℝ) < 2) E.exponent_pos
  have hts := tailConstant_nonneg (half_pos S.b_pos) S.exponent_pos
  have hconst : 0 ≤ envelopeConstant E S - 1 := by
    unfold envelopeConstant
    linarith [mul_nonneg (div_pos hC hc).le (show 0 ≤ 1 + 2 * tailConstant 2 ν by positivity),
      mul_nonneg (show 0 ≤ 2 * S.C / E.c by positivity)
        (show 0 ≤ 1 + tailConstant (S.b / 2) μ by positivity)]
  have hm := mul_le_mul_of_nonneg_left (algebraic_alias_schedule (s := s) hB) hconst
  calc
    _ ≤ (∫ ξ : ℝ, (1 + |ξ|) ^ s * ‖pHat ξ‖) *
        ((1 + B)^(-s) + (envelopeConstant E S - 1) * (1 + B)^(-s)) := by
      exact mul_le_mul_of_nonneg_left (add_le_add le_rfl hm) hA
    _ = _ := by ring

/-- General finite width count, with B and the suppression order L independent. -/
theorem grid_width_cycles {T B L μ ν Z H : ℝ}
    (hT : 0 ≤ T) (hB : 0 < B) (hL : 0 < L)
    (hZ : 0 ≤ Z) (hH : 0 < H) :
    let γ := B / L ^ (1 / ν)
    let D := Z * L ^ (1 / μ) / γ
    let h := 1 / (H * B)
    let J := ⌊(T + D) / h⌋₊
    ((Finset.Icc (-(J : ℤ)) (J : ℤ)).card : ℝ) ≤
      2 * H * T * B + 2 * H * Z * L ^ (1 / μ + 1 / ν) + 1 := by
  dsimp only
  rw [symmetric_grid_card]
  push_cast
  have hb := constructed_grid_width_bound (μ := μ) (ν := ν)
    (H := 2 * Real.pi * H) hT hB hL hZ
    (mul_pos (mul_pos (by norm_num) Real.pi_pos) hH)
  dsimp only at hb
  have hh : 2 * Real.pi / (2 * Real.pi * H * B) = 1 / (H * B) := by field_simp
  have hc1 : 2 * Real.pi * H * T / Real.pi = 2 * H * T := by field_simp
  have hc2 : 2 * Real.pi * H * Z / Real.pi = 2 * H * Z := by field_simp
  simpa only [hh, hc1, hc2] using hb

/-- Algebraic error for the actual activation network with its prescribed jets. -/
theorem algebraic_activation_error
    {φ g : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {s B H T Z : ℝ} (hs : 0 ≤ s) (hB : 0 < B)
    (hW : Integrable (fun ξ : ℝ => (1 + |ξ|)^s * ‖pHat ξ‖))
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hT : 0 < T) (hg : ContDiffOn ℝ r g (Icc (-T) T))
    (hprofile : ∀ x ∈ Icc (-T) T, iteratedDeriv r g x = (synthesize pHat x).re) :
    let L := max 1 ((s / 2) * Real.log (1 + B))
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := Finset.Icc (-(⌊(T + D) / h⌋₊ : ℤ)) (⌊(T + D) / h⌋₊ : ℤ)
    ∀ x ∈ Icc (-T) T,
      |anchoredActivationNetwork φ (realReadoutDensity pHat K γ (by positivity) B)
        r m γ h J g x - g x| ≤
      ((∫ ξ : ℝ, (1 + |ξ|)^s * ‖pHat ξ‖) * envelopeConstant E S *
        (1 + B)^(-s)) * T^r / (r.factorial : ℝ) := by
  have hL : 0 < max 1 ((s / 2) * Real.log (1 + B)) := lt_of_lt_of_le zero_lt_one (le_max_left _ _)
  have hH0 : 0 < H := by linarith
  have hγ : 0 < B / (max 1 ((s / 2) * Real.log (1 + B))) ^ (1 / ν) :=
    div_pos hB (Real.rpow_pos_of_pos hL _)
  apply finite_activation_function_certificate hφ K hm hγ.ne' hkernel hp hB.le
    (one_div_pos.mpr (mul_pos hH0 hB)) hT
  · intro ξ _
    rw [complexScaledKernel_fourier K hγ]
    exact E.nonzero _
  · exact hg
  · exact hprofile
  · intro x hx
    exact algebraic_scheduled_certificate K E S hp hs hB hW hH hsep hZ hhalo
      (abs_le.mpr hx)

/-- Explicit cutoff selection for algebraic accuracy; no failure at zero target. -/
theorem algebraic_accuracy_choice {C ε s : ℝ}
    (hC : 0 ≤ C) (hε : 0 < ε) (hs : 0 < s) :
    C * (1 + Real.exp (max 1 (Real.log (C / ε)) / s)) ^ (-s) ≤ ε := by
  let L := max 1 (Real.log (C / ε))
  have he := log_accuracy_choice hC hε
  have hpow : (1 + Real.exp (L / s)) ^ (-s) ≤ (Real.exp (L / s)) ^ (-s) :=
    Real.rpow_le_rpow_of_nonpos (Real.exp_pos _) (by linarith [Real.exp_pos (L / s)])
      (neg_nonpos.mpr hs.le)
  have hid : (Real.exp (L / s)) ^ (-s) = Real.exp (-L) := by
    rw [Real.rpow_def_of_pos (Real.exp_pos _), Real.log_exp]
    congr 1
    field_simp
  rw [hid] at hpow
  exact (mul_le_mul_of_nonneg_left hpow hC).trans he

/-- Explicit algebraic accuracy and the actual halo-inclusive finite width.
The bound is O(B + log(B)^(1/μ+1/ν)); ScalarScaling proves this is O(B). -/
theorem algebraic_accuracy_and_width
    {φ g : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    {pHat : ℝ → ℂ} (hp : Integrable pHat)
    {s H T Z ε : ℝ} (hs : 0 < s)
    (hW : Integrable (fun ξ : ℝ => (1 + |ξ|)^s * ‖pHat ξ‖))
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hT : 0 < T) (hε : 0 < ε) (hg : ContDiffOn ℝ r g (Icc (-T) T))
    (hprofile : ∀ x ∈ Icc (-T) T, iteratedDeriv r g x = (synthesize pHat x).re) :
    let C := (∫ ξ : ℝ, (1 + |ξ|)^s * ‖pHat ξ‖) *
      envelopeConstant E S * T^r / (r.factorial : ℝ)
    let B := Real.exp (max 1 (Real.log (C / ε)) / s)
    let L := max 1 ((s / 2) * Real.log (1 + B))
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := Finset.Icc (-(⌊(T + D) / h⌋₊ : ℤ)) (⌊(T + D) / h⌋₊ : ℤ)
    (∀ x ∈ Icc (-T) T,
      |anchoredActivationNetwork φ (realReadoutDensity pHat K γ (by positivity) B)
        r m γ h J g x - g x| ≤ ε) ∧
    (J.card : ℝ) ≤ 2 * H * T * B + 2 * H * Z * L ^ (1 / μ + 1 / ν) + 1 := by
  dsimp only
  let C := (∫ ξ : ℝ, (1 + |ξ|)^s * ‖pHat ξ‖) *
    envelopeConstant E S * T^r / (r.factorial : ℝ)
  let B := Real.exp (max 1 (Real.log (C / ε)) / s)
  let L := max 1 ((s / 2) * Real.log (1 + B))
  have hB : 0 < B := Real.exp_pos _
  have hL : 1 ≤ L := le_max_left _ _
  have hC : 0 ≤ C := by
    have he := (envelopeConstant_pos E S).le
    dsimp [C]
    positivity
  have heps := algebraic_accuracy_choice hC hε hs
  constructor
  · intro x hx
    have hb := algebraic_activation_error hφ K hm hkernel E S hp hs.le hB hW hH hsep
      hZ hhalo hT hg hprofile x hx
    apply hb.trans
    calc
      _ = C * (1 + B)^(-s) := by dsimp [C]; ring
      _ ≤ ε := heps
  · exact grid_width_cycles hT.le hB (lt_of_lt_of_le zero_lt_one hL) hZ (by linarith)

end QIRadon.ScalarRate
