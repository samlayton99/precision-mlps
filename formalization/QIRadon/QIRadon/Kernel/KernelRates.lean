import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Complex.Norm
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Analysis.SumIntegralComparisons
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.MeasureTheory.Integral.Gamma
import Mathlib.MeasureTheory.Function.LocallyIntegrable
import Mathlib.Topology.Algebra.InfiniteSum.NatInt
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-!
# Pointwise kernel-envelope estimates for the constructive rate theorem

The input is a two-sided property of the Fourier transform of the actual
kernel. No alias or density bound is assumed. These lemmas derive inverse
amplification and all individual aliases from that property.
-/

namespace QIRadon.KernelRates

/-- Quantitative two-sided Fourier decay for a specified kernel transform.
`A` is permitted to be zero; the lower envelope must stay strictly positive. -/
structure SpectralEnvelope (Khat : ℝ → ℂ) (ν : ℝ) where
  c : ℝ
  C : ℝ
  A : ℝ
  a : ℝ
  c_pos : 0 < c
  C_pos : 0 < C
  A_nonneg : 0 ≤ A
  a_pos : 0 < a
  exponent_pos : 0 < ν
  lower : ∀ ξ, c * Real.exp (-A * |ξ| ^ ν) ≤ ‖Khat ξ‖
  upper : ∀ ξ, ‖Khat ξ‖ ≤ C * Real.exp (-a * |ξ| ^ ν)

variable {Khat : ℝ → ℂ} {ν : ℝ}

/-- The nonvanishing used in deconvolution is a consequence of the envelope. -/
theorem SpectralEnvelope.norm_pos (E : SpectralEnvelope Khat ν) (ξ : ℝ) :
    0 < ‖Khat ξ‖ :=
  lt_of_lt_of_le (mul_pos E.c_pos (Real.exp_pos _)) (E.lower ξ)

theorem SpectralEnvelope.nonzero (E : SpectralEnvelope Khat ν) (ξ : ℝ) :
    Khat ξ ≠ 0 := norm_pos_iff.mp (E.norm_pos ξ)

/-- Inverse amplification follows directly from the lower spectral envelope. -/
theorem SpectralEnvelope.inverse_bound (E : SpectralEnvelope Khat ν) (ξ : ℝ) :
    1 / ‖Khat ξ‖ ≤ (1 / E.c) * Real.exp (E.A * |ξ| ^ ν) := by
  calc
    1 / ‖Khat ξ‖ ≤ 1 / (E.c * Real.exp (-E.A * |ξ| ^ ν)) :=
      div_le_div_of_nonneg_left zero_le_one
        (mul_pos E.c_pos (Real.exp_pos _)) (E.lower ξ)
    _ = (1 / E.c) * Real.exp (E.A * |ξ| ^ ν) := by
      rw [show -E.A * |ξ| ^ ν = -(E.A * |ξ| ^ ν) by ring, Real.exp_neg]
      field_simp

/-- General alias-ratio estimate before inserting any parameter schedule. -/
theorem SpectralEnvelope.ratio_bound (E : SpectralEnvelope Khat ν) (u v : ℝ) :
    ‖Khat u / Khat v‖ ≤
      (E.C / E.c) * Real.exp (E.A * |v| ^ ν - E.a * |u| ^ ν) := by
  rw [norm_div]
  calc
    ‖Khat u‖ / ‖Khat v‖ ≤
        (E.C * Real.exp (-E.a * |u| ^ ν)) /
          (E.c * Real.exp (-E.A * |v| ^ ν)) :=
      div_le_div₀ (mul_pos E.C_pos (Real.exp_pos _)).le (E.upper u)
        (mul_pos E.c_pos (Real.exp_pos _)) (E.lower v)
    _ = (E.C / E.c) * Real.exp (E.A * |v| ^ ν - E.a * |u| ^ ν) := by
      rw [show E.A * |v| ^ ν - E.a * |u| ^ ν =
        (-E.a * |u| ^ ν) - (-E.A * |v| ^ ν) by ring, Real.exp_sub]
      field_simp

/-- A retained target band and a separated alias lattice give this geometric
frequency separation, uniformly over all nonzero integer aliases. -/
theorem alias_frequency_separation {B H ω k : ℝ}
    (hB : 0 < B) (hH : 2 ≤ H) (hω : |ω| ≤ B) (hk : 1 ≤ |k|) :
    (H / 2) * |k| * B ≤ |ω + k * (H * B)| := by
  have hH0 : 0 ≤ H := by linarith
  have habs : |k * (H * B)| = |k| * H * B := by
    rw [abs_mul, abs_mul, abs_of_nonneg hH0, abs_of_pos hB]
    ring
  have ht : |k * (H * B)| ≤ |ω + k * (H * B)| + |ω| := by
    calc
      |k * (H * B)| = |(ω + k * (H * B)) + (-ω)| := by congr 1; ring
      _ ≤ |ω + k * (H * B)| + |-ω| := abs_add_le _ _
      _ = |ω + k * (H * B)| + |ω| := by rw [abs_neg]
  rw [habs] at ht
  have hprod : 2 ≤ H * |k| := by
    have hm := mul_le_mul hH hk (by norm_num : (0 : ℝ) ≤ 1) hH0
    simpa using hm
  have hh := mul_le_mul_of_nonneg_right hprod hB.le
  nlinarith

/-- The bandwidth schedule has exactly the required normalized passband scale. -/
theorem normalized_bandwidth_power {B L ν : ℝ}
    (hB : 0 < B) (hL : 0 < L) (hν : 0 < ν) :
    (B / (B / L ^ (1 / ν))) ^ ν = L := by
  have ht := Real.rpow_pos_of_pos hL (1 / ν)
  have hid : B / (B / L ^ (1 / ν)) = L ^ (1 / ν) := by field_simp
  rw [hid, ← Real.rpow_mul hL.le]
  rw [one_div_mul_cancel hν.ne', Real.rpow_one]

/-- On the retained band, the normalized frequency power is bounded by L. -/
theorem normalized_band_power_le {B γ L ν ω : ℝ}
    (_hB : 0 < B) (hγ : 0 < γ) (hν : 0 < ν)
    (hscale : (B / γ) ^ ν = L) (hω : |ω| ≤ B) :
    |ω / γ| ^ ν ≤ L := by
  rw [abs_div, abs_of_pos hγ]
  rw [← hscale]
  exact Real.rpow_le_rpow (div_nonneg (abs_nonneg _) hγ.le)
    ((div_le_div_iff_of_pos_right hγ).mpr hω) hν.le

/-- The alias's normalized frequency retains the full integer-alias factor. -/
theorem normalized_alias_power_ge {B γ L ν H ω k : ℝ}
    (hB : 0 < B) (hγ : 0 < γ) (hν : 0 < ν)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hω : |ω| ≤ B) (hk : 1 ≤ |k|) :
    (H / 2) ^ ν * |k| ^ ν * L ≤ |(ω + k * (H * B)) / γ| ^ ν := by
  have hsep := alias_frequency_separation hB hH hω hk
  have hH0 : 0 ≤ H / 2 := by linarith
  have hk0 : 0 ≤ |k| := abs_nonneg _
  have hbase : (H / 2) * |k| * (B / γ) ≤ |(ω + k * (H * B)) / γ| := by
    rw [abs_div, abs_of_pos hγ]
    convert! (div_le_div_iff_of_pos_right hγ).mpr hsep using 1
    ring
  have hp := Real.rpow_le_rpow
    (mul_nonneg (mul_nonneg hH0 hk0) (div_pos hB hγ).le) hbase hν.le
  rw [Real.mul_rpow (mul_nonneg hH0 hk0) (div_pos hB hγ).le,
    Real.mul_rpow hH0 hk0, hscale] at hp
  exact hp

/-- Each alias is exponentially suppressed by the chosen separated frequency
lattice. The denominator's nonzero property was deduced from the envelope. -/
theorem SpectralEnvelope.alias_bound (E : SpectralEnvelope Khat ν)
    {B γ L H ω k : ℝ}
    (hB : 0 < B) (hγ : 0 < γ) (hL : 0 < L)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hω : |ω| ≤ B) (hk : 1 ≤ |k|) :
    ‖Khat ((ω + k * (H * B)) / γ) / Khat (ω / γ)‖ ≤
      (E.C / E.c) * Real.exp (-4 * L * |k| ^ ν) := by
  apply (E.ratio_bound _ _).trans
  apply mul_le_mul_of_nonneg_left _ (div_pos E.C_pos E.c_pos).le
  apply Real.exp_le_exp.mpr
  have hpass := normalized_band_power_le hB hγ E.exponent_pos hscale hω
  have halias := normalized_alias_power_ge hB hγ E.exponent_pos hscale hH hω hk
  have hkpow : 1 ≤ |k| ^ ν := Real.one_le_rpow hk E.exponent_pos.le
  have h1 := mul_le_mul_of_nonneg_left hpass E.A_nonneg
  have h2 := mul_le_mul_of_nonneg_left halias E.a_pos.le
  have h3 := mul_le_mul_of_nonneg_right hsep
    (mul_nonneg (Real.rpow_nonneg (abs_nonneg k) ν) hL.le)
  have h4 := mul_le_mul_of_nonneg_left hkpow (mul_nonneg E.A_nonneg hL.le)
  nlinarith [h1, h2, h3, h4]

/-- Integer-indexed version: all k != 0, not only the nearest alias. -/
theorem SpectralEnvelope.integer_alias_bound (E : SpectralEnvelope Khat ν)
    {B γ L H ω : ℝ} (k : ℤ)
    (hB : 0 < B) (hγ : 0 < γ) (hL : 0 < L)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hω : |ω| ≤ B) (hk : k ≠ 0) :
    ‖Khat ((ω + (k : ℝ) * (H * B)) / γ) / Khat (ω / γ)‖ ≤
      (E.C / E.c) * Real.exp (-4 * L * |(k : ℝ)| ^ ν) := by
  apply E.alias_bound hB hγ hL hscale hH hsep hω
  exact_mod_cast Int.one_le_abs hk

/-- Pointwise deconvolution amplification on the passband, used before taking
the finite Fourier integral that defines the coefficient density. -/
theorem SpectralEnvelope.band_density_amplification (E : SpectralEnvelope Khat ν)
    {B γ L ω : ℝ} (z : ℂ)
    (hB : 0 < B) (hγ : 0 < γ)
    (hscale : (B / γ) ^ ν = L) (hω : |ω| ≤ B) :
    ‖z / Khat (ω / γ)‖ ≤ ‖z‖ * ((1 / E.c) * Real.exp (E.A * L)) := by
  rw [norm_div, div_eq_mul_inv]
  apply mul_le_mul_of_nonneg_left _ (norm_nonneg _)
  have hinv := E.inverse_bound (ω / γ)
  simp only [one_div] at hinv ⊢
  apply hinv.trans
  apply mul_le_mul_of_nonneg_left _ (inv_pos.mpr E.c_pos).le
  apply Real.exp_le_exp.mpr
  exact mul_le_mul_of_nonneg_left
    (normalized_band_power_le hB hγ E.exponent_pos hscale hω) E.A_nonneg


open Set MeasureTheory Filter

/-- The finite coefficient integral used by the actual construction. -/
noncomputable def cutoffDensity (pHat Khat : ℝ → ℂ) (B γ c : ℝ) : ℂ :=
  (1 / (2 * Real.pi) : ℝ) •
    ∫ ω in Icc (-B) B,
      (pHat ω / Khat (ω / γ)) * Complex.exp (((ω * c : ℝ) : ℂ) * Complex.I)

/-- Finite density bound obtained from the target L1 spectrum and the proved
passband amplification; no density-size assumption is inserted. -/
theorem SpectralEnvelope.cutoffDensity_bound (E : SpectralEnvelope Khat ν)
    {pHat : ℝ → ℂ} {B γ L c : ℝ}
    (hB : 0 < B) (hγ : 0 < γ) (hscale : (B / γ) ^ ν = L)
    (hP : IntegrableOn pHat (Icc (-B) B)) :
    ‖cutoffDensity pHat Khat B γ c‖ ≤
      ((1 / E.c) * Real.exp (E.A * L)) *
        ((1 / (2 * Real.pi)) * ∫ ω in Icc (-B) B, ‖pHat ω‖) := by
  let C : ℝ := (1 / E.c) * Real.exp (E.A * L)
  have hInt : ‖∫ ω in Icc (-B) B,
      (pHat ω / Khat (ω / γ)) * Complex.exp (((ω * c : ℝ) : ℂ) * Complex.I)‖ ≤
      ∫ ω in Icc (-B) B, ‖pHat ω‖ * C := by
    apply norm_integral_le_of_norm_le (hP.norm.mul_const C)
    filter_upwards [ae_restrict_mem measurableSet_Icc] with ω hω
    rw [norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one]
    exact E.band_density_amplification (pHat ω) hB hγ hscale
      (abs_le.mpr hω)
  unfold cutoffDensity
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : (0 : ℝ) < 1 / (2 * Real.pi))]
  have hm := mul_le_mul_of_nonneg_left hInt (by positivity : (0 : ℝ) ≤ 1 / (2 * Real.pi))
  rw [integral_mul_const] at hm
  convert! hm using 1
  dsimp [C]
  ring

/-- Regularity sufficient for the displayed coefficient integral to be an
ordinary integrable finite integral. The Fourier transform of a Schwartz
kernel satisfies the continuity assumption. -/
theorem SpectralEnvelope.cutoffDensity_integrable (E : SpectralEnvelope Khat ν)
    {pHat : ℝ → ℂ} {B γ c : ℝ}
    (hK : Continuous Khat) (hP : IntegrableOn pHat (Icc (-B) B)) :
    IntegrableOn (fun ω => (pHat ω / Khat (ω / γ)) *
      Complex.exp (((ω * c : ℝ) : ℂ) * Complex.I)) (Icc (-B) B) := by
  have hk : Continuous (fun ω : ℝ => Khat (ω / γ)) :=
    hK.comp (continuous_id.div_const γ)
  have he : Continuous (fun ω : ℝ => Complex.exp (((ω * c : ℝ) : ℂ) * Complex.I)) :=
    Complex.continuous_exp.comp ((Complex.continuous_ofReal.comp
      (continuous_id.mul_const c)).mul_const Complex.I)
  have hm : Continuous (fun ω : ℝ => (Khat (ω / γ))⁻¹ *
      Complex.exp (((ω * c : ℝ) : ℂ) * Complex.I)) :=
    (hk.inv₀ (fun ω => E.nonzero (ω / γ))).mul he
  have hi := hP.mul_continuousOn hm.continuousOn isCompact_Icc
  simpa only [div_eq_mul_inv, mul_assoc] using hi

/-- Monotonicity of the positive exponential envelope. -/
theorem antitone_rpow_exponential {b ν : ℝ} (hb : 0 < b) (hν : 0 < ν) :
    AntitoneOn (fun x : ℝ => Real.exp (-b * x ^ ν)) (Ici 0) := by
  intro x hx y hy hxy
  apply Real.exp_le_exp.mpr
  exact mul_le_mul_of_nonpos_left (Real.rpow_le_rpow hx hxy hν.le) (neg_nonpos.mpr hb.le)

/-- The envelope integral is finite for every positive spectral exponent. -/
theorem integrable_rpow_exponential {b ν : ℝ} (hb : 0 < b) (hν : 0 < ν) :
    IntegrableOn (fun x : ℝ => Real.exp (-b * x ^ ν)) (Ioi 0) := by
  simpa using integrableOn_rpow_mul_exp_neg_mul_rpow (s := 0)
    (by norm_num) hν hb

/-- Every positive exponent gives an absolutely summable alias envelope. -/
theorem summable_rpow_exponential {b ν : ℝ} (hb : 0 < b) (hν : 0 < ν) :
    Summable (fun n : ℕ => Real.exp (-b * (n : ℝ) ^ ν)) := by
  exact (antitone_rpow_exponential hb hν).summable_of_integrableOn_Ioi_zero
    (integrable_rpow_exponential hb hν) (fun x _ => (Real.exp_pos _).le)

set_option maxHeartbeats 600000 in
/-- Two-sided integer envelope with an explicit one-dimensional integral
constant. This avoids leaving an unspecified infinite-series constant. -/
theorem integer_rpow_exponential_sum {b ν : ℝ} (hb : 0 < b) (hν : 0 < ν) :
    Summable (fun k : ℤ => Real.exp (-b * |(k : ℝ)| ^ ν)) ∧
    (∑' k : ℤ, Real.exp (-b * |(k : ℝ)| ^ ν)) ≤
      1 + 2 * ∫ x in Ioi (0 : ℝ), Real.exp (-b * x ^ ν) := by
  have hn : Summable (fun n : ℕ => Real.exp (-b * |((n : ℤ) : ℝ)| ^ ν)) := by
    simpa only [Int.cast_natCast, Int.cast_neg, abs_neg, Nat.abs_cast] using summable_rpow_exponential hb hν
  have hneg : Summable (fun n : ℕ => Real.exp (-b * |((-n : ℤ) : ℝ)| ^ ν)) := by
    simpa only [Int.cast_natCast, Int.cast_neg, abs_neg, Nat.abs_cast] using summable_rpow_exponential hb hν
  have hsum : Summable (fun k : ℤ => Real.exp (-b * |(k : ℝ)| ^ ν)) :=
    Summable.of_nat_of_neg (f := fun k : ℤ => Real.exp (-b * |(k : ℝ)| ^ ν)) hn hneg
  refine ⟨hsum, ?_⟩
  have hnat := (antitone_rpow_exponential hb hν).tsum_le_integral
    (integrable_rpow_exponential hb hν) (fun x _ => (Real.exp_pos _).le)
  have hid := Summable.tsum_of_nat_of_neg
    (f := fun k : ℤ => Real.exp (-b * |(k : ℝ)| ^ ν)) hn hneg
  simp only [Int.cast_natCast, Int.cast_neg, abs_neg, Nat.abs_cast,
    Int.cast_zero, abs_zero, Real.zero_rpow hν.ne', mul_zero, Real.exp_zero] at hid hnat
  rw [hid]
  linarith

/-- A uniform separation of the L-dependent and alias-index-dependent decay. -/
theorem alias_exponential_factor {L z : ℝ} (hL : 1 ≤ L) (hz : 1 ≤ z) :
    Real.exp (-4 * L * z) ≤ Real.exp (-2 * L) * Real.exp (-2 * z) := by
  rw [← Real.exp_add]
  apply Real.exp_le_exp.mpr
  have hp : 0 ≤ (L - 1) * (z - 1) := mul_nonneg (by linarith) (by linarith)
  nlinarith

/-- Whole nonzero-alias series, with summability proved rather than assumed. -/
theorem SpectralEnvelope.alias_series_bound (E : SpectralEnvelope Khat ν)
    {B γ L H ω : ℝ}
    (hB : 0 < B) (hγ : 0 < γ) (hL : 1 ≤ L)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν) (hω : |ω| ≤ B) :
    let aliases : ℤ → ℝ := fun k => if k = 0 then 0 else
      ‖Khat ((ω + (k : ℝ) * (H * B)) / γ) / Khat (ω / γ)‖
    Summable aliases ∧
      (∑' k, aliases k) ≤
        (E.C / E.c) * Real.exp (-2 * L) *
          (1 + 2 * ∫ x in Ioi (0 : ℝ), Real.exp (-2 * x ^ ν)) := by
  dsimp only
  let C : ℝ := (E.C / E.c) * Real.exp (-2 * L)
  have hC : 0 ≤ C := (mul_pos (div_pos E.C_pos E.c_pos) (Real.exp_pos _)).le
  obtain ⟨hs, hbound⟩ := integer_rpow_exponential_sum (by norm_num : (0 : ℝ) < 2) E.exponent_pos
  have hmajor := hs.mul_left C
  have hle : ∀ k : ℤ, (if k = 0 then 0 else
      ‖Khat ((ω + (k : ℝ) * (H * B)) / γ) / Khat (ω / γ)‖) ≤
        C * Real.exp (-2 * |(k : ℝ)| ^ ν) := by
    intro k
    split_ifs with hk
    · exact mul_nonneg hC (Real.exp_pos _).le
    · have hk1 : 1 ≤ |(k : ℝ)| := by exact_mod_cast Int.one_le_abs hk
      have hp := alias_exponential_factor hL (Real.one_le_rpow hk1 E.exponent_pos.le)
      have he := E.integer_alias_bound k hB hγ (by linarith) hscale hH hsep hω hk
      apply he.trans
      have hm := mul_le_mul_of_nonneg_left hp (div_pos E.C_pos E.c_pos).le
      simpa only [C, mul_assoc] using hm
  have hnonneg : ∀ k : ℤ, 0 ≤ (if k = 0 then 0 else
      ‖Khat ((ω + (k : ℝ) * (H * B)) / γ) / Khat (ω / γ)‖) := by
    intro k
    split_ifs <;> positivity
  have ha := hmajor.of_nonneg_of_le hnonneg hle
  refine ⟨ha, ?_⟩
  calc
    _ ≤ ∑' k : ℤ, C * Real.exp (-2 * |(k : ℝ)| ^ ν) :=
      ha.tsum_le_tsum hle hmajor
    _ = C * ∑' k : ℤ, Real.exp (-2 * |(k : ℝ)| ^ ν) := tsum_mul_left
    _ ≤ C * (1 + 2 * ∫ x in Ioi (0 : ℝ), Real.exp (-2 * x ^ ν)) :=
      mul_le_mul_of_nonneg_left hbound hC


/-- The alias-series constant is an ordinary Gamma-function value. -/
theorem SpectralEnvelope.alias_series_bound_gamma (E : SpectralEnvelope Khat ν)
    {B γ L H ω : ℝ}
    (hB : 0 < B) (hγ : 0 < γ) (hL : 1 ≤ L)
    (hscale : (B / γ) ^ ν = L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν) (hω : |ω| ≤ B) :
    (∑' k : ℤ, if k = 0 then 0 else
      ‖Khat ((ω + (k : ℝ) * (H * B)) / γ) / Khat (ω / γ)‖) ≤
      (E.C / E.c) * Real.exp (-2 * L) *
        (1 + 2 * (2 : ℝ) ^ (-1 / ν) * Real.Gamma (1 / ν + 1)) := by
  have hb := (E.alias_series_bound hB hγ hL hscale hH hsep hω).2
  rw [integral_exp_neg_mul_rpow E.exponent_pos (by norm_num : (0 : ℝ) < 2)] at hb
  convert! hb using 1
  ring

/-- The prescribed bandwidth supplies the scale condition used in the alias
lemma, so the final statement does not assume that condition independently. -/
theorem SpectralEnvelope.scheduled_alias_bound (E : SpectralEnvelope Khat ν)
    {B L H ω : ℝ} (k : ℤ)
    (hB : 0 < B) (hL : 0 < L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hω : |ω| ≤ B) (hk : k ≠ 0) :
    ‖Khat ((ω + (k : ℝ) * (H * B)) / (B / L ^ (1 / ν))) /
      Khat (ω / (B / L ^ (1 / ν)))‖ ≤
      (E.C / E.c) * Real.exp (-4 * L * |(k : ℝ)| ^ ν) := by
  exact E.integer_alias_bound k hB
    (div_pos hB (Real.rpow_pos_of_pos hL _)) hL
    (normalized_bandwidth_power hB hL E.exponent_pos) hH hsep hω hk

/-- Explicit integral constant for a one-sided spatial envelope. -/
noncomputable def tailConstant (b μ : ℝ) : ℝ :=
  b ^ (-1 / μ) * Real.Gamma (1 / μ + 1)

theorem tailConstant_eq_integral {b μ : ℝ} (hb : 0 < b) (hμ : 0 < μ) :
    tailConstant b μ = ∫ x in Ioi (0 : ℝ), Real.exp (-b * x ^ μ) :=
  (integral_exp_neg_mul_rpow hμ hb).symm

theorem tailConstant_nonneg {b μ : ℝ} (hb : 0 < b) (hμ : 0 < μ) :
    0 ≤ tailConstant b μ := by
  rw [tailConstant_eq_integral hb hμ]
  exact integral_nonneg (fun _ => (Real.exp_pos _).le)

/-- Rescaling the spatial integration constant is exact. -/
theorem tailConstant_scale {b μ spacing : ℝ}
    (hb : 0 < b) (hμ : 0 < μ) (hspacing : 0 < spacing) :
    spacing * tailConstant (b * spacing ^ μ) μ = tailConstant b μ := by
  unfold tailConstant
  rw [Real.mul_rpow hb.le (Real.rpow_pos_of_pos hspacing μ).le,
    ← Real.rpow_mul hspacing.le]
  rw [show μ * (-1 / μ) = -1 by field_simp, Real.rpow_neg_one]
  field_simp

/-- The summable shifted grid tail used for a finite halo. No spatial-tail
summation oracle is assumed. -/
theorem shifted_grid_tail {b μ δ spacing : ℝ}
    (hb : 0 < b) (hμ : 0 < μ) (hδ : 0 ≤ δ) (hspacing : 0 < spacing) :
    Summable (fun n : ℕ => Real.exp (-b * (δ + spacing * n) ^ μ)) ∧
    spacing * (∑' n : ℕ, Real.exp (-b * (δ + spacing * n) ^ μ)) ≤
      Real.exp (-(b / 2) * δ ^ μ) * (spacing + tailConstant (b / 2) μ) := by
  have hb2 : 0 < b / 2 := half_pos hb
  have hbp : 0 < (b / 2) * spacing ^ μ := mul_pos hb2 (Real.rpow_pos_of_pos hspacing μ)
  let C : ℝ := Real.exp (-(b / 2) * δ ^ μ)
  have hC : 0 ≤ C := (Real.exp_pos _).le
  have hsum := summable_rpow_exponential hbp hμ
  have hmajor := hsum.mul_left C
  have hle : ∀ n : ℕ, Real.exp (-b * (δ + spacing * n) ^ μ) ≤
      C * Real.exp (-((b / 2) * spacing ^ μ) * (n : ℝ) ^ μ) := by
    intro n
    have hn : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
    have hln : 0 ≤ spacing * n := mul_nonneg hspacing.le hn
    have hd1 := Real.rpow_le_rpow hδ (le_add_of_nonneg_right hln) hμ.le
    have hd2 := Real.rpow_le_rpow hln (le_add_of_nonneg_left hδ) hμ.le
    rw [Real.mul_rpow hspacing.le hn] at hd2
    dsimp [C]
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    have h1 := mul_le_mul_of_nonneg_left hd1 hb.le
    have h2 := mul_le_mul_of_nonneg_left hd2 hb.le
    nlinarith
  have hactual := hmajor.of_nonneg_of_le (fun n => (Real.exp_pos _).le) hle
  refine ⟨hactual, ?_⟩
  have hnat := (antitone_rpow_exponential hbp hμ).tsum_le_integral
    (integrable_rpow_exponential hbp hμ) (fun x _ => (Real.exp_pos _).le)
  simp only [Real.zero_rpow hμ.ne', mul_zero, Real.exp_zero] at hnat
  rw [← tailConstant_eq_integral hbp hμ] at hnat
  have ht : (∑' n : ℕ, Real.exp (-b * (δ + spacing * n) ^ μ)) ≤
      C * (1 + tailConstant ((b / 2) * spacing ^ μ) μ) := by
    calc
      _ ≤ ∑' n : ℕ, C * Real.exp (-((b / 2) * spacing ^ μ) * (n : ℝ) ^ μ) :=
        hactual.tsum_le_tsum hle hmajor
      _ = C * ∑' n : ℕ, Real.exp (-((b / 2) * spacing ^ μ) * (n : ℝ) ^ μ) := tsum_mul_left
      _ ≤ _ := mul_le_mul_of_nonneg_left hnat hC
  have hm := mul_le_mul_of_nonneg_left ht hspacing.le
  have hid : spacing * (C * (1 + tailConstant ((b / 2) * spacing ^ μ) μ)) =
      C * (spacing + tailConstant (b / 2) μ) := by
    calc
      _ = C * (spacing + spacing * tailConstant ((b / 2) * spacing ^ μ) μ) := by ring
      _ = _ := by rw [tailConstant_scale hb2 hμ hspacing]
  simpa only [hid, C] using hm

/-- Explicit tail envelope of a real spatial kernel. -/
structure SpatialEnvelope (K : ℝ → ℝ) (μ : ℝ) where
  C : ℝ
  b : ℝ
  C_pos : 0 < C
  b_pos : 0 < b
  exponent_pos : 0 < μ
  upper : ∀ u, |K u| ≤ C * Real.exp (-b * |u| ^ μ)

/-- Each omitted center is bounded using its actual separation from the target
interval; this is the elementary step needed on both sides of a halo. -/
theorem SpatialEnvelope.scaled_tail_term {K : ℝ → ℝ} {μ : ℝ}
    (E : SpatialEnvelope K μ) {γ δ spacing u : ℝ} (n : ℕ)
    (hγ : 0 < γ) (hδ : 0 ≤ δ) (hspacing : 0 < spacing)
    (hd : δ + spacing * n ≤ |γ * u|) :
    |γ * K (γ * u)| ≤
      γ * E.C * Real.exp (-E.b * (δ + spacing * n) ^ μ) := by
  rw [abs_mul, abs_of_pos hγ]
  have hn : 0 ≤ δ + spacing * n := add_nonneg hδ (mul_nonneg hspacing.le (Nat.cast_nonneg _))
  calc
    γ * |K (γ * u)| ≤ γ * (E.C * Real.exp (-E.b * |γ * u| ^ μ)) :=
      mul_le_mul_of_nonneg_left (E.upper _) hγ.le
    _ ≤ γ * (E.C * Real.exp (-E.b * (δ + spacing * n) ^ μ)) := by
      apply mul_le_mul_of_nonneg_left _ hγ.le
      apply mul_le_mul_of_nonneg_left _ E.C_pos.le
      apply Real.exp_le_exp.mpr
      exact mul_le_mul_of_nonpos_left (Real.rpow_le_rpow hn hd E.exponent_pos.le)
        (neg_nonpos.mpr E.b_pos.le)
    _ = _ := by ring


/-- Geometry of the two omitted half-grids, retaining exactly indices -J..J. -/
theorem omitted_centers_separated {T D h γ s : ℝ} (J n : ℕ)
    (hh : 0 < h) (hγ : 0 < γ) (hD : 0 ≤ D)
    (hs : |s| ≤ T) (hcut : T + D ≤ ((J : ℝ) + 1) * h) :
    γ * D + (γ * h) * n ≤ |γ * (s - (((J : ℝ) + 1) + n) * h)| ∧
    γ * D + (γ * h) * n ≤ |γ * (s + (((J : ℝ) + 1) + n) * h)| := by
  have hs' := abs_le.mp hs
  have hn : 0 ≤ (n : ℝ) := Nat.cast_nonneg n
  have hnh : 0 ≤ (n : ℝ) * h := mul_nonneg hn hh.le
  have hr : s - (((J : ℝ) + 1) + n) * h ≤ 0 := by nlinarith
  have hl : 0 ≤ s + (((J : ℝ) + 1) + n) * h := by nlinarith
  constructor
  · rw [abs_mul, abs_of_pos hγ, abs_of_nonpos hr]
    have hmul := mul_le_mul_of_nonneg_left hcut hγ.le
    have hsmul := mul_le_mul_of_nonneg_left hs'.2 hγ.le
    nlinarith
  · rw [abs_mul, abs_of_pos hγ, abs_of_nonneg hl]
    have hmul := mul_le_mul_of_nonneg_left hcut hγ.le
    have hsmul := mul_le_mul_of_nonneg_left hs'.1 hγ.le
    nlinarith

/-- The actual two-sided finite-halo tail. Every summand is an omitted grid
center; its convergence and quantitative bound are proved from K's envelope. -/
theorem SpatialEnvelope.finite_halo_tail {K : ℝ → ℝ} {μ : ℝ}
    (E : SpatialEnvelope K μ) {T D h γ s : ℝ} (J : ℕ)
    (hh : 0 < h) (hγ : 0 < γ) (hD : 0 ≤ D)
    (hs : |s| ≤ T) (hcut : T + D ≤ ((J : ℝ) + 1) * h) :
    let tail : ℕ → ℝ := fun n =>
      |γ * K (γ * (s - (((J : ℝ) + 1) + n) * h))| +
      |γ * K (γ * (s + (((J : ℝ) + 1) + n) * h))|
    Summable tail ∧
    h * (∑' n, tail n) ≤
      2 * E.C * Real.exp (-(E.b / 2) * (γ * D) ^ μ) *
        (γ * h + tailConstant (E.b / 2) μ) := by
  dsimp only
  obtain ⟨hseries, hbound⟩ := shifted_grid_tail E.b_pos E.exponent_pos
    (mul_nonneg hγ.le hD) (mul_pos hγ hh)
  have hmajor := hseries.mul_left (2 * γ * E.C)
  have hle : ∀ n : ℕ,
      |γ * K (γ * (s - (((J : ℝ) + 1) + n) * h))| +
      |γ * K (γ * (s + (((J : ℝ) + 1) + n) * h))| ≤
      (2 * γ * E.C) * Real.exp (-E.b * (γ * D + (γ * h) * n) ^ μ) := by
    intro n
    have hdist := omitted_centers_separated J n hh hγ hD hs hcut
    have hr := E.scaled_tail_term n hγ (mul_nonneg hγ.le hD) (mul_pos hγ hh) hdist.1
    have hl := E.scaled_tail_term n hγ (mul_nonneg hγ.le hD) (mul_pos hγ hh) hdist.2
    linarith
  have hnonneg : ∀ n : ℕ,
      0 ≤ |γ * K (γ * (s - (((J : ℝ) + 1) + n) * h))| +
      |γ * K (γ * (s + (((J : ℝ) + 1) + n) * h))| := by
    intro n
    positivity
  have hactual := hmajor.of_nonneg_of_le hnonneg hle
  refine ⟨hactual, ?_⟩
  have hsum := hactual.tsum_le_tsum hle hmajor
  rw [tsum_mul_left] at hsum
  have hmul := mul_le_mul_of_nonneg_left hsum hh.le
  have hbound' := mul_le_mul_of_nonneg_left hbound
    (mul_pos (by norm_num : (0 : ℝ) < 2) E.C_pos).le
  nlinarith [hmul, hbound']

/-- The floor-based cutoff used in the construction satisfies the first-omitted
center condition in the finite-halo theorem. -/
theorem floor_halo_cutoff {T D h : ℝ} (hh : 0 < h) :
    T + D ≤ ((⌊(T + D) / h⌋₊ : ℝ) + 1) * h := by
  have ht := Nat.lt_floor_add_one ((T + D) / h)
  exact ((div_lt_iff₀ hh).mp ht).le

/-- Exact dependence of physical halo extent on the two regularity exponents. -/
theorem bandwidth_halo_identity {B L μ ν Z : ℝ}
    (hB : 0 < B) (hL : 0 < L) :
    B * (Z * L ^ (1 / μ) / (B / L ^ (1 / ν))) =
      Z * L ^ (1 / μ + 1 / ν) := by
  rw [Real.rpow_add hL]
  have ht := Real.rpow_pos_of_pos hL (1 / ν)
  field_simp

/-- Actual number of grid neurons, including halos, before asymptotic notation.
The spacing is 2π/(H B), and the physical halo is the explicit spectral/spatial
scale used in the rate construction. -/
theorem constructed_grid_width_bound {T B L μ ν Z H : ℝ}
    (hT : 0 ≤ T) (hB : 0 < B) (hL : 0 < L) (hZ : 0 ≤ Z) (hH : 0 < H) :
    let D := Z * L ^ (1 / μ) / (B / L ^ (1 / ν))
    let h := 2 * Real.pi / (H * B)
    (2 * (⌊(T + D) / h⌋₊ : ℝ) + 1) ≤
      (H * T / Real.pi) * B +
      (H * Z / Real.pi) * L ^ (1 / μ + 1 / ν) + 1 := by
  dsimp only
  let D := Z * L ^ (1 / μ) / (B / L ^ (1 / ν))
  let h := 2 * Real.pi / (H * B)
  have hh : 0 < h := div_pos (mul_pos (by norm_num) Real.pi_pos) (mul_pos hH hB)
  have hD : 0 ≤ D := div_nonneg
    (mul_nonneg hZ (Real.rpow_pos_of_pos hL _).le)
    (div_pos hB (Real.rpow_pos_of_pos hL _)).le
  have hf := Nat.floor_le (div_nonneg (add_nonneg hT hD) hh.le)
  have hid : 2 * ((T + D) / h) + 1 =
      (H * T / Real.pi) * B + (H * Z / Real.pi) * L ^ (1 / μ + 1 / ν) + 1 := by
    have hp := bandwidth_halo_identity (μ := μ) (ν := ν) (Z := Z) hB hL
    change B * D = Z * L ^ (1 / μ + 1 / ν) at hp
    dsimp [h]
    field_simp
    nlinarith [hp]
  calc
    (2 * (⌊(T + D) / h⌋₊ : ℝ) + 1) ≤ 2 * ((T + D) / h) + 1 := by linarith
    _ = _ := hid


/-- A finite, explicit halo-factor choice cancels the density amplification. -/
theorem halo_factor_choice {A b μ : ℝ}
    (hA : 0 ≤ A) (hb : 0 < b) (hμ : 0 < μ) :
    (b / 2) * (((2 * (A + 3) / b) ^ (1 / μ)) ^ μ) = A + 3 := by
  have hbase : 0 < 2 * (A + 3) / b := by positivity
  rw [← Real.rpow_mul hbase.le, one_div_mul_cancel hμ.ne', Real.rpow_one]
  field_simp

/-- The exponent cancellation is proved from the spatial parameter schedule,
not postulated as a fast halo bound. -/
theorem inverse_times_halo_exponential {A b μ Z L : ℝ}
    (_hb : 0 < b) (hμ : 0 < μ) (hZ : 0 ≤ Z) (hL : 0 < L)
    (hfactor : A + 3 ≤ (b / 2) * Z ^ μ) :
    Real.exp (A * L) * Real.exp (-(b / 2) * (Z * L ^ (1 / μ)) ^ μ) ≤
      Real.exp (-3 * L) := by
  rw [← Real.exp_add]
  apply Real.exp_le_exp.mpr
  rw [Real.mul_rpow hZ (Real.rpow_pos_of_pos hL _).le,
    ← Real.rpow_mul hL.le, one_div_mul_cancel hμ.ne', Real.rpow_one]
  have hm := mul_le_mul_of_nonneg_right hfactor hL.le
  nlinarith

/-- A computable frequency-separation factor satisfying the all-alias bound's
condition exists for every positive decay rate. -/
theorem spectral_separation_choice {A a ν : ℝ}
    (hA : 0 ≤ A) (ha : 0 < a) (hν : 0 < ν) :
    let H := 2 * max 1 (((A + 4) / a) ^ (1 / ν))
    2 ≤ H ∧ A + 4 ≤ a * (H / 2) ^ ν := by
  dsimp only
  constructor
  · nlinarith [le_max_left (1 : ℝ) (((A + 4) / a) ^ (1 / ν))]
  · have hbase : 0 < (A + 4) / a := by positivity
    have hp : (((A + 4) / a) ^ (1 / ν)) ^ ν = (A + 4) / a := by
      rw [← Real.rpow_mul hbase.le, one_div_mul_cancel hν.ne', Real.rpow_one]
    have hm := Real.rpow_le_rpow (Real.rpow_pos_of_pos hbase _).le
      (le_max_right (1 : ℝ) (((A + 4) / a) ^ (1 / ν))) hν.le
    rw [hp] at hm
    have hm' := mul_le_mul_of_nonneg_left hm ha.le
    have hd : a * ((A + 4) / a) = A + 4 := by field_simp
    rw [hd] at hm'
    convert! hm' using 1
    congr 2
    ring

/-- Exponential Fourier-moment control gives an explicit cutoff error. -/
theorem exponential_spectral_cutoff {pHat : ℝ → ℂ} {σ B : ℝ}
    (hσ : 0 ≤ σ)
    (hP : Integrable pHat)
    (hW : Integrable (fun ω : ℝ => Real.exp (σ * |ω|) * ‖pHat ω‖)) :
    (∫ ω in {ω : ℝ | B < |ω|}, ‖pHat ω‖) ≤
      Real.exp (-σ * B) * ∫ ω : ℝ, Real.exp (σ * |ω|) * ‖pHat ω‖ := by
  let S : Set ℝ := {ω : ℝ | B < |ω|}
  have hS : MeasurableSet S := measurableSet_lt measurable_const continuous_abs.measurable
  have hupper := hW.const_mul (Real.exp (-σ * B))
  have hpoint : ∀ ω ∈ S, ‖pHat ω‖ ≤
      Real.exp (-σ * B) * (Real.exp (σ * |ω|) * ‖pHat ω‖) := by
    intro ω hω
    have he : 1 ≤ Real.exp (σ * (|ω| - B)) :=
      Real.one_le_exp_iff.mpr (mul_nonneg hσ (sub_nonneg.mpr hω.le))
    have hm := mul_le_mul_of_nonneg_right he (norm_nonneg (pHat ω))
    rw [one_mul] at hm
    have hid : Real.exp (-σ * B) * (Real.exp (σ * |ω|) * ‖pHat ω‖) =
        Real.exp (σ * (|ω| - B)) * ‖pHat ω‖ := by
      rw [← mul_assoc, ← Real.exp_add]
      congr 2
      ring
    rwa [hid]
  calc
    _ ≤ ∫ ω in S, Real.exp (-σ * B) * (Real.exp (σ * |ω|) * ‖pHat ω‖) :=
      setIntegral_mono_on hP.norm.integrableOn hupper.integrableOn hS hpoint
    _ ≤ ∫ ω : ℝ, Real.exp (-σ * B) * (Real.exp (σ * |ω|) * ‖pHat ω‖) :=
      setIntegral_le_integral hupper
        (Filter.Eventually.of_forall (fun ω => by positivity))
    _ = _ := integral_const_mul _ _


/-- Algebraic Fourier-moment control yields an explicit cutoff rate, for the
finite-regularity construction using only real-frequency information. -/
theorem polynomial_spectral_cutoff {pHat : ℝ → ℂ} {s B : ℝ}
    (hs : 0 ≤ s) (hB : 0 ≤ B)
    (hP : Integrable pHat)
    (hW : Integrable (fun ω : ℝ => (1 + |ω|) ^ s * ‖pHat ω‖)) :
    (∫ ω in {ω : ℝ | B < |ω|}, ‖pHat ω‖) ≤
      (1 + B) ^ (-s) * ∫ ω : ℝ, (1 + |ω|) ^ s * ‖pHat ω‖ := by
  let S : Set ℝ := {ω : ℝ | B < |ω|}
  have hS : MeasurableSet S := measurableSet_lt measurable_const continuous_abs.measurable
  have hbase : 0 < 1 + B := by linarith
  have hpow : 0 < (1 + B) ^ s := Real.rpow_pos_of_pos hbase s
  have hinv : (1 + B) ^ (-s) = ((1 + B) ^ s)⁻¹ := Real.rpow_neg hbase.le s
  have hupper := hW.const_mul ((1 + B) ^ (-s))
  have hpoint : ∀ ω ∈ S, ‖pHat ω‖ ≤
      (1 + B) ^ (-s) * ((1 + |ω|) ^ s * ‖pHat ω‖) := by
    intro ω hω
    have how : B < |ω| := hω
    have hh := Real.rpow_le_rpow hbase.le (show 1 + B ≤ 1 + |ω| by linarith) hs
    have hm := mul_le_mul_of_nonneg_right hh (norm_nonneg (pHat ω))
    rw [hinv]
    apply (mul_le_mul_iff_right₀ hpow).mp
    calc
      (1 + B) ^ s * ‖pHat ω‖ ≤ (1 + |ω|) ^ s * ‖pHat ω‖ := hm
      _ = (1 + B) ^ s * (((1 + B) ^ s)⁻¹ * ((1 + |ω|) ^ s * ‖pHat ω‖)) := by
        field_simp
  calc
    _ ≤ ∫ ω in S, (1 + B) ^ (-s) * ((1 + |ω|) ^ s * ‖pHat ω‖) :=
      setIntegral_mono_on hP.norm.integrableOn hupper.integrableOn hS hpoint
    _ ≤ ∫ ω : ℝ, (1 + B) ^ (-s) * ((1 + |ω|) ^ s * ‖pHat ω‖) :=
      setIntegral_le_integral hupper
        (Filter.Eventually.of_forall (fun ω => by positivity))
    _ = _ := integral_const_mul _ _

/-- The max-exponent in the promised scalar width law is obtained from the
actual floor-counted grid, including the entire halo. -/
theorem analytic_grid_width_power_bound {T b₀ L μ ν Z H : ℝ}
    (hT : 0 ≤ T) (hb₀ : 0 < b₀) (hL : 1 ≤ L)
    (hZ : 0 ≤ Z) (hH : 0 < H) :
    let B := b₀ * L
    let D := Z * L ^ (1 / μ) / (B / L ^ (1 / ν))
    let h := 2 * Real.pi / (H * B)
    2 * (⌊(T + D) / h⌋₊ : ℝ) + 1 ≤
      (H * T * b₀ / Real.pi + H * Z / Real.pi + 1) *
        L ^ (max 1 (1 / μ + 1 / ν)) := by
  have hL0 : 0 < L := lt_of_lt_of_le zero_lt_one hL
  have hB : 0 < b₀ * L := mul_pos hb₀ hL0
  have hw := constructed_grid_width_bound (μ := μ) (ν := ν)
    hT hB hL0 hZ hH
  dsimp only at hw ⊢
  apply hw.trans
  let κ : ℝ := max 1 (1 / μ + 1 / ν)
  have hκ1 : 1 ≤ κ := le_max_left _ _
  have hκs : 1 / μ + 1 / ν ≤ κ := le_max_right _ _
  have hfirst : L ≤ L ^ κ := by
    simpa only [Real.rpow_one] using Real.rpow_le_rpow_of_exponent_le hL hκ1
  have hsecond : L ^ (1 / μ + 1 / ν) ≤ L ^ κ :=
    Real.rpow_le_rpow_of_exponent_le hL hκs
  have hone : 1 ≤ L ^ κ := Real.one_le_rpow hL (by linarith)
  have hc1 : 0 ≤ H * T * b₀ / Real.pi := by positivity
  have hc2 : 0 ≤ H * Z / Real.pi := by positivity
  have hm1 := mul_le_mul_of_nonneg_left hfirst hc1
  have hm2 := mul_le_mul_of_nonneg_left hsecond hc2
  calc
    (H * T / Real.pi) * (b₀ * L) + (H * Z / Real.pi) * L ^ (1 / μ + 1 / ν) + 1
        = (H * T * b₀ / Real.pi) * L + (H * Z / Real.pi) * L ^ (1 / μ + 1 / ν) + 1 := by ring
    _ ≤ (H * T * b₀ / Real.pi) * L ^ κ + (H * Z / Real.pi) * L ^ κ + L ^ κ :=
      add_le_add (add_le_add hm1 hm2) hone
    _ = _ := by dsimp [κ]; ring

end QIRadon.KernelRates
