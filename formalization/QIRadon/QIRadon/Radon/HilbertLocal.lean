import QIRadon.Radon.HilbertAbel
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

/-! A finite, singularity-subtracted formula for the canonical Hilbert filter. -/
noncomputable section
open MeasureTheory Set Filter
open scoped FourierTransform Topology
namespace QIRadon.Radon

/-- The real conjugate-Poisson kernel without its factor `1/π`. -/
def cauchyAbel (ε u : ℝ) : ℝ := u / (ε^2+u^2)

private theorem abel_den_pos {ε : ℝ} (hε : 0 < ε) (u : ℝ) :
    0 < ε^2+u^2 := by positivity

theorem cauchyAbel_continuous {ε : ℝ} (hε : 0 < ε) :
    Continuous (cauchyAbel ε) := by
  unfold cauchyAbel
  exact continuous_id.div (continuous_const.add (continuous_id.pow 2))
    (fun u => (abel_den_pos hε u).ne')

theorem abs_cauchyAbel_le {ε : ℝ} (hε : 0 < ε) (u : ℝ) :
    |cauchyAbel ε u| ≤ 1 / (2*ε) := by
  rw [cauchyAbel, abs_div, abs_of_pos (abel_den_pos hε u), div_le_div_iff₀
    (abel_den_pos hε u) (by positivity)]
  nlinarith [sq_nonneg (|u|-ε), sq_abs u]

theorem abs_cauchyAbel_mul_abs_le_one {ε : ℝ} (hε : 0 < ε) (u : ℝ) :
    |cauchyAbel ε u| * |u| ≤ 1 := by
  rw [cauchyAbel, abs_div, abs_of_pos (abel_den_pos hε u), div_mul_eq_mul_div,
    div_le_one (abel_den_pos hε u), ← sq, sq_abs]
  nlinarith [sq_nonneg ε]

theorem cauchyAbel_tendsto {a : ℕ → ℝ} (ha0 : Tendsto a atTop (𝓝 0))
    {u : ℝ} (hu : u ≠ 0) :
    Tendsto (fun n => cauchyAbel (a n) u) atTop (𝓝 u⁻¹) := by
  have hd : Tendsto (fun n => (a n)^2+u^2) atTop (𝓝 (u^2)) := by
    simpa using (ha0.pow 2).add_const (u^2)
  have ht := (tendsto_const_nhds (x := u) (f := atTop)).div hd (pow_ne_zero 2 hu)
  simpa [cauchyAbel, div_pow, div_eq_mul_inv, ← inv_pow, pow_two, hu, Pi.mul_def, Pi.inv_def] using ht

/-- The elementary endpoint term, proved by the fundamental theorem of calculus. -/
theorem integral_cauchyAbel_Icc {ε L : ℝ} (hε : 0 < ε) (hL : 0 ≤ L) (s : ℝ) :
    (∫ t in Icc (-L) L, cauchyAbel ε (s-t)) =
      (Real.log (ε^2+(s+L)^2)-Real.log (ε^2+(s-L)^2))/2 := by
  have hd : ∀ t : ℝ, HasDerivAt
      (fun t : ℝ => -Real.log (ε^2+(s-t)^2)/2) (cauchyAbel ε (s-t)) t := by
    intro t
    have hv := (hasDerivAt_const t (ε^2)).add
      (((hasDerivAt_const t s).sub (hasDerivAt_id t)).pow 2)
    have hl := (hv.log (abel_den_pos hε (s-t)).ne').neg.div_const 2
    convert hl using 1 <;> try dsimp [cauchyAbel]
    all_goals first | rfl | ring
  have hi : IntervalIntegrable (fun t : ℝ => cauchyAbel ε (s-t)) volume (-L) L :=
    ((cauchyAbel_continuous hε).comp (continuous_const.sub continuous_id)).intervalIntegrable _ _
  have h := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hd t) hi
  rw [intervalIntegral.integral_of_le (by linarith), ← integral_Icc_eq_integral_Ioc] at h
  rw [h]
  simp only [sub_neg_eq_add]
  ring

/-- Explicit regularized physical formula with the usual `1/π` normalization. -/
theorem hilbertAbel_convolution_normalized {ε : ℝ} (hε : 0 < ε)
    {p : ℝ → ℂ} (hp : Integrable p) (s : ℝ) :
    QIRadon.Kernel.synthesize
      (fun ξ => hilbertAbelMultiplier (2*Real.pi*ε) ξ * 𝓕 p ξ) s =
      (Real.pi:ℂ)⁻¹ * ∫ t : ℝ, (cauchyAbel ε (s-t):ℂ) * p t := by
  rw [hilbertAbel_convolution (by positivity) hp, ← integral_const_mul]
  apply integral_congr_ae
  filter_upwards with t
  push_cast
  have hpi : (Real.pi:ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
  have hd : ((ε:ℂ)^2+((s:ℂ)-t)^2) ≠ 0 := by
    have hr := (abel_den_pos hε (s-t)).ne'
    exact_mod_cast hr
  unfold cauchyAbel
  push_cast
  field_simp
  ring

private theorem cauchyAbel_mul_integrable {ε : ℝ} (hε : 0 < ε)
    {p : ℝ → ℂ} (hp : Integrable p) (s : ℝ) :
    Integrable (fun t : ℝ => (cauchyAbel ε (s-t):ℂ)*p t) := by
  apply hp.bdd_mul (c := 1/(2*ε))
    ((Complex.continuous_ofReal.comp ((cauchyAbel_continuous hε).comp
      (continuous_const.sub continuous_id))).aestronglyMeasurable)
  filter_upwards with t
  simpa only [Function.comp_def, Pi.sub_apply, id_eq, Complex.norm_real, Real.norm_eq_abs] using abs_cauchyAbel_le hε (s-t)

/-- The inner singularity disappears after subtracting the value at its center. -/
theorem hilbertAbel_inner_tendsto {p : ℝ → ℂ} (hp : Integrable p)
    {s L K : ℝ} (hK : 0 ≤ K)
    (hLip : ∀ t ∈ Icc (-L) L, ‖p t-p s‖ ≤ K*|s-t|)
    {ε : ℕ → ℝ} (hε : ∀ n, 0 < ε n) (hε0 : Tendsto ε atTop (𝓝 0)) :
    Tendsto (fun n => ∫ t in Icc (-L) L,
      (cauchyAbel (ε n) (s-t):ℂ)*(p t-p s)) atTop
      (𝓝 (∫ t in Icc (-L) L, (p t-p s)/((s-t:ℝ):ℂ))) := by
  apply tendsto_integral_of_dominated_convergence (fun _ : ℝ => K)
  · intro n
    exact ((Complex.continuous_ofReal.comp ((cauchyAbel_continuous (hε n)).comp
      (continuous_const.sub continuous_id))).aestronglyMeasurable.mul
        (hp.aestronglyMeasurable.restrict.sub aestronglyMeasurable_const))
  · exact integrable_const K
  · intro n
    filter_upwards [ae_restrict_mem measurableSet_Icc] with t ht
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
    calc
      |cauchyAbel (ε n) (s-t)| *‖p t-p s‖ ≤
          |cauchyAbel (ε n) (s-t)| *(K*|s-t|) :=
        mul_le_mul_of_nonneg_left (hLip t ht) (abs_nonneg _)
      _ = K*(|cauchyAbel (ε n) (s-t)| *|s-t|) := by ring
      _ ≤ K*1 := mul_le_mul_of_nonneg_left
        (abs_cauchyAbel_mul_abs_le_one (hε n) (s-t)) hK
      _ = K := mul_one _
  · filter_upwards with t
    by_cases hst : s-t=0
    · have ht : t=s := by linarith
      simp [ht]
    · have hc := Complex.continuous_ofReal.continuousAt.tendsto.comp
        (cauchyAbel_tendsto hε0 hst)
      simpa only [Function.comp_def, Complex.ofReal_inv, div_eq_mul_inv, mul_comm] using
        hc.mul_const (p t-p s)

private theorem outside_distance {s L t : ℝ} (_hs : |s| < L)
    (ht : t ∉ Icc (-L) L) : L-|s| ≤ |s-t| := by
  have habs : L ≤ |t| := by
    have ht' : t < -L ∨ L < t := by simpa only [mem_Icc, not_and_or, not_le] using ht
    rcases ht' with ht | ht
    · exact le_trans (by linarith) (neg_le_abs t)
    · exact le_trans ht.le (le_abs_self t)
  have htri : |t| ≤ |s-t|+|s| := by
    calc
      |t| = |-(s-t)+s| := by congr 1; ring
      _ ≤ |-(s-t)|+|s| := abs_add_le _ _
      _ = _ := by rw [abs_neg]
  linarith

/-- Away from the singularity, the Abel limit follows by a uniform distance bound. -/
theorem hilbertAbel_outer_tendsto {p : ℝ → ℂ} (hp : Integrable p)
    {s L : ℝ} (hs : |s| < L)
    {ε : ℕ → ℝ} (hε : ∀ n, 0 < ε n) (hε0 : Tendsto ε atTop (𝓝 0)) :
    Tendsto (fun n => ∫ t in (Icc (-L) L)ᶜ,
      (cauchyAbel (ε n) (s-t):ℂ)*p t) atTop
      (𝓝 (∫ t in (Icc (-L) L)ᶜ, p t/((s-t:ℝ):ℂ))) := by
  apply tendsto_integral_of_dominated_convergence (fun t : ℝ =>
    (L-|s|)⁻¹*‖p t‖)
  · intro n
    exact ((cauchyAbel_mul_integrable (hε n) hp s).aestronglyMeasurable.restrict)
  · exact (hp.norm.const_mul ((L-|s|)⁻¹)).restrict
  · intro n
    filter_upwards [ae_restrict_mem measurableSet_Icc.compl] with t ht
    have hd := outside_distance hs ht
    have hdist : 0 < |s-t| := lt_of_lt_of_le (by linarith : 0 < L-|s|) hd
    have hk : |cauchyAbel (ε n) (s-t)| ≤ (L-|s|)⁻¹ := by
      have hfirst : |cauchyAbel (ε n) (s-t)| ≤ |s-t|⁻¹ := by
        rw [← one_div, le_div_iff₀ hdist]
        exact abs_cauchyAbel_mul_abs_le_one (hε n) (s-t)
      exact hfirst.trans (inv_anti₀ (by linarith) hd)
    simpa only [norm_mul, Complex.norm_real, Real.norm_eq_abs] using
      mul_le_mul_of_nonneg_right hk (norm_nonneg (p t))
  · filter_upwards [ae_restrict_mem measurableSet_Icc.compl] with t ht
    have hd := outside_distance hs ht
    have hst : s-t ≠ 0 := by intro h; simp [h] at hd; linarith
    have hc := Complex.continuous_ofReal.continuousAt.tendsto.comp
      (cauchyAbel_tendsto hε0 hst)
    simpa only [Function.comp_def, Complex.ofReal_inv, div_eq_mul_inv, mul_comm] using
      hc.mul_const (p t)

/-- Exact finite decomposition before taking the Abel limit. -/
theorem hilbertAbel_split {ε L : ℝ} (hε : 0 < ε) (hL : 0 ≤ L)
    {p : ℝ → ℂ} (hp : Integrable p) (s : ℝ) :
    (∫ t : ℝ, (cauchyAbel ε (s-t):ℂ)*p t) =
      (∫ t in Icc (-L) L, (cauchyAbel ε (s-t):ℂ)*(p t-p s)) +
      (((Real.log (ε^2+(s+L)^2)-Real.log (ε^2+(s-L)^2))/2:ℝ):ℂ)*p s +
      (∫ t in (Icc (-L) L)ᶜ, (cauchyAbel ε (s-t):ℂ)*p t) := by
  have hqi : IntegrableOn (fun t : ℝ => (cauchyAbel ε (s-t):ℂ)) (Icc (-L) L) :=
    (Complex.continuous_ofReal.comp ((cauchyAbel_continuous hε).comp
      (continuous_const.sub continuous_id))).integrableOn_Icc
  have hqpi : IntegrableOn (fun t : ℝ => (cauchyAbel ε (s-t):ℂ)*p t) (Icc (-L) L) :=
    (cauchyAbel_mul_integrable hε hp s).integrableOn
  have hlocal : (∫ t in Icc (-L) L, (cauchyAbel ε (s-t):ℂ)*(p t-p s)) =
      (∫ t in Icc (-L) L, (cauchyAbel ε (s-t):ℂ)*p t) -
      (((Real.log (ε^2+(s+L)^2)-Real.log (ε^2+(s-L)^2))/2:ℝ):ℂ)*p s := by
    simp_rw [mul_sub]
    have ho : (∫ t in Icc (-L) L, (cauchyAbel ε (s-t):ℂ)) =
        ((∫ t in Icc (-L) L, cauchyAbel ε (s-t) : ℝ):ℂ) := integral_ofReal
    rw [integral_sub hqpi (hqi.mul_const _), integral_mul_const, ho,
      integral_cauchyAbel_Icc hε hL]
  rw [hlocal]
  simp only [sub_add_cancel]
  exact (integral_add_compl measurableSet_Icc (cauchyAbel_mul_integrable hε hp s)).symm

/-- Canonical Hilbert transform as finite inner subtraction, endpoint logarithms,
and an ordinary integral separated from the evaluation point. -/
theorem hilbertFourier_local_formula {p : ℝ → ℂ}
    (hp : Integrable p) (hphat : Integrable (𝓕 p))
    {s L K : ℝ} (hs : |s| < L) (hK : 0 ≤ K)
    (hLip : ∀ t ∈ Icc (-L) L, ‖p t-p s‖ ≤ K*|s-t|) :
    hilbertFourier p s = (Real.pi:ℂ)⁻¹ *
      ((∫ t in Icc (-L) L, (p t-p s)/((s-t:ℝ):ℂ)) +
        ((Real.log (L+s)-Real.log (L-s):ℝ):ℂ)*p s +
        (∫ t in (Icc (-L) L)ᶜ, p t/((s-t:ℝ):ℂ))) := by
  let ε : ℕ → ℝ := fun n => 1/((n:ℝ)+1)
  have he : ∀ n, 0 < ε n := by intro n; dsimp [ε]; positivity
  have he0 : Tendsto ε atTop (𝓝 0) := tendsto_one_div_add_atTop_nhds_zero_nat
  have hpL : 0 < L+s := by have := neg_abs_le s; linarith
  have hmL : 0 < L-s := by have := le_abs_self s; linarith
  have hLs : s+L ≠ 0 := by linarith
  have hsL : s-L ≠ 0 := by linarith
  have hlogp : Tendsto (fun n => Real.log ((ε n)^2+(s+L)^2)) atTop
      (𝓝 (Real.log ((s+L)^2))) := by
    apply (Real.continuousAt_log (pow_ne_zero 2 hLs)).tendsto.comp
    simpa using (he0.pow 2).add_const ((s+L)^2)
  have hlogm : Tendsto (fun n => Real.log ((ε n)^2+(s-L)^2)) atTop
      (𝓝 (Real.log ((s-L)^2))) := by
    apply (Real.continuousAt_log (pow_ne_zero 2 hsL)).tendsto.comp
    simpa using (he0.pow 2).add_const ((s-L)^2)
  have hlog : Tendsto (fun n =>
      (((Real.log ((ε n)^2+(s+L)^2)-Real.log ((ε n)^2+(s-L)^2))/2:ℝ):ℂ)*p s)
      atTop (𝓝 (((Real.log (L+s)-Real.log (L-s):ℝ):ℂ)*p s)) := by
    have hlim := ((Complex.continuous_ofReal.continuousAt.tendsto.comp
      ((hlogp.sub hlogm).div_const 2)).mul_const (p s))
    have hval : (Real.log ((s+L)^2)-Real.log ((s-L)^2))/2 =
        Real.log (L+s)-Real.log (L-s) := by
      rw [Real.log_pow, Real.log_pow]
      rw [show s+L=L+s by ring, show s-L=-(L-s) by ring, Real.log_neg_eq_log]
      ring
    simpa only [Function.comp_def, hval] using hlim
  have hphysical := ((hilbertAbel_inner_tendsto hp hK hLip he he0).add hlog).add
    (hilbertAbel_outer_tendsto hp hs he he0)
  have hphysical' : Tendsto (fun n => (Real.pi:ℂ)⁻¹ *
      ∫ t : ℝ, (cauchyAbel (ε n) (s-t):ℂ)*p t) atTop
      (𝓝 ((Real.pi:ℂ)⁻¹ *
        ((∫ t in Icc (-L) L, (p t-p s)/((s-t:ℝ):ℂ)) +
          ((Real.log (L+s)-Real.log (L-s):ℝ):ℂ)*p s +
          (∫ t in (Icc (-L) L)ᶜ, p t/((s-t:ℝ):ℂ))))) := by
    simpa only [hilbertAbel_split (he _) (le_of_lt (lt_of_le_of_lt (abs_nonneg s) hs)) hp s]
      using hphysical.const_mul ((Real.pi:ℂ)⁻¹)
  have hspectral := hilbertAbel_tendsto hphat
    (fun n => show 0 < 2*Real.pi*ε n by positivity)
    (by simpa using he0.const_mul (2*Real.pi)) s
  have hspectral' : Tendsto (fun n => (Real.pi:ℂ)⁻¹ *
      ∫ t : ℝ, (cauchyAbel (ε n) (s-t):ℂ)*p t) atTop (𝓝 (hilbertFourier p s)) := by
    simpa only [hilbertAbel_convolution_normalized (he _) hp s] using hspectral
  exact tendsto_nhds_unique hspectral' hphysical'

end QIRadon.Radon
