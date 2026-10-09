import QIRadon.Radon.Filter
import QIRadon.Radon.HilbertLogSharp

/-! Actual local holomorphic continuations of both parities of the canonical
Radon filter. The input complex continuation is subsequently supplied by the
proved finite ball-mixture cross-section construction. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap Topology
namespace QIRadon.Radon

/-- Odd-dimensional radial filter, represented using local complex derivatives. -/
def evenFilterContinuation (P : ℂ → ℂ) (k : ℕ) (z : ℂ) : ℂ :=
  ((2*Real.pi*Complex.I)^(2*k))⁻¹ * iteratedDeriv (2*k) P z

/-- Even-dimensional radial filter, represented using the local finite Hilbert
continuation and its complex derivatives. -/
def oddFilterContinuation (p : ℝ → ℂ) (P : ℂ → ℂ) (L : ℝ) (k : ℕ) (z : ℂ) : ℂ :=
  (Complex.I*((2*Real.pi*Complex.I)^(2*k+1))⁻¹) *
    iteratedDeriv (2*k+1) (hilbertContinuation p P L) z

theorem evenFilterContinuation_analytic {P : ℂ → ℂ} {L δ : ℝ}
    (hP : DifferentiableOn ℂ P (scalarTube L δ)) (k : ℕ) :
    AnalyticOnNhd ℂ (evenFilterContinuation P k) (scalarTube L δ) := by
  have hd : AnalyticOnNhd ℂ (iteratedDeriv (2*k) P) (scalarTube L δ) := by
    simpa only [iteratedDeriv_eq_iterate] using
      (hP.analyticOnNhd (isOpen_scalarTube L δ)).iterated_deriv (2*k)
  exact analyticOnNhd_const.mul hd

theorem evenFilterContinuation_eq_radialFilter (p : 𝓢(ℝ, ℂ))
    {P : ℂ → ℂ} {L δ : ℝ} (hδ : 0 < δ)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hagree : ∀ t ∈ Icc (-L) L, p t=P (t:ℂ))
    (k : ℕ) {s : ℝ} (hs : |s| < L) :
    evenFilterContinuation P k (s:ℂ) = radialFilter p (2*k) s := by
  have hJ : ∀ t ∈ Ioo (-L) L, (t:ℂ) ∈ scalarTube L δ :=
    fun t ht => ofReal_mem_scalarTube hδ (Ioo_subset_Icc_self ht)
  have hd := iteratedDeriv_real_of_complex (p := p) isOpen_Ioo
    (hP.analyticOnNhd (isOpen_scalarTube L δ)) hJ
    (fun t ht => hagree t (Ioo_subset_Icc_self ht)) (2*k) (abs_lt.mp hs)
  rw [radialFilter_even, evenFilterContinuation, hd]

theorem oddFilterContinuation_analytic (p : 𝓢(ℝ, ℂ))
    {P : ℂ → ℂ} {L δ B Q : ℝ} (hδ : 0 < δ) (hB : 0 ≤ B) (hQ : Q < L)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B) (k : ℕ) :
    AnalyticOnNhd ℂ (oddFilterContinuation p P L k)
      (scalarTube L δ ∩ ball 0 Q) := by
  have hh := hilbertContinuation_analytic p.integrable hδ hB hQ hP hbound
  have hd : AnalyticOnNhd ℂ (iteratedDeriv (2*k+1) (hilbertContinuation p P L))
      (scalarTube L δ ∩ ball 0 Q) := by
    simpa only [iteratedDeriv_eq_iterate] using hh.iterated_deriv (2*k+1)
  exact analyticOnNhd_const.mul hd

theorem oddFilterContinuation_eq_radialFilter (p : 𝓢(ℝ, ℂ))
    {P : ℂ → ℂ} {L δ B Q : ℝ} (hδ : 0 < δ) (hB : 0 ≤ B) (hQ : Q < L)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    (hagree : ∀ t ∈ Icc (-L) L, p t=P (t:ℂ))
    (k : ℕ) {s : ℝ} (hs : |s| < Q) :
    oddFilterContinuation p P L k (s:ℂ) = radialFilter p (2*k+1) s := by
  have hJ : ∀ t ∈ Ioo (-Q) Q, (t:ℂ) ∈ scalarTube L δ ∩ ball (0:ℂ) Q := by
    intro t ht
    have htQ : |t| < Q := abs_lt.mpr ht
    exact ⟨ofReal_mem_scalarTube hδ (abs_le.mp (htQ.trans hQ).le),
      by simpa only [mem_ball_zero_iff, Complex.norm_real, Real.norm_eq_abs] using htQ⟩
  have hh := hilbertContinuation_analytic p.integrable hδ hB hQ hP hbound
  have hag : ∀ t ∈ Ioo (-Q) Q, hilbertFourier p t=hilbertContinuation p P L (t:ℂ) := by
    intro t ht
    exact (hilbertContinuation_eq_hilbertFourier p.integrable (𝓕 p).integrable
      hδ hB hP hbound hagree ((abs_lt.mpr ht).trans hQ)).symm
  have hd := iteratedDeriv_real_of_complex isOpen_Ioo hh hJ hag (2*k+1) (abs_lt.mp hs)
  rw [radialFilter_odd_deriv_hilbert, oddFilterContinuation, hd]

/-- Exact magnitudes of the Fourier-normalization coefficients. -/
theorem norm_evenFilterFactor (k : ℕ) :
    ‖((2*Real.pi*Complex.I:ℂ)^(2*k))⁻¹‖ = ((2*Real.pi)^(2*k))⁻¹ := by
  simp [norm_inv, norm_pow, Complex.norm_I, Complex.norm_real,
    abs_of_pos Real.pi_pos]

theorem evenFilterContinuation_norm_le {P : ℂ → ℂ} {L δ B : ℝ}
    (hδ : 0 < δ) (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    (k : ℕ) {z : ℂ} (hz : z ∈ scalarTube L (δ/2)) :
    ‖evenFilterContinuation P k z‖ ≤
      ((2*Real.pi)^(2*k))⁻¹ * ((2*k).factorial*B/(δ/2)^(2*k)) := by
  unfold evenFilterContinuation
  rw [norm_mul, norm_inv, norm_pow]
  have hc : ‖(2*Real.pi*Complex.I:ℂ)‖ = 2*Real.pi := by
    simp [Complex.norm_I, Complex.norm_real, abs_of_pos Real.pi_pos]
  rw [hc]
  exact mul_le_mul_of_nonneg_left (iteratedDeriv_bound_on_smaller_tube hδ hP hbound (2*k) hz)
    (by positivity)

/-- A common inner tube gives the even-dimensional filtered bound using only
local analytic magnitude B and the ordinary real L1 mass of the unfiltered p. -/
theorem oddFilterContinuation_norm_le (p : 𝓢(ℝ, ℂ))
    {P : ℂ → ℂ} {L δ B Q S η : ℝ}
    (hδ : 0 < δ) (hB : 0 ≤ B) (hQ0 : 0 < Q) (hQ : Q < L)
    (hη : 0 < η) (hS : S ≤ L) (hηδ : 2*η ≤ δ) (hSQ : S+2*η ≤ Q)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    (k : ℕ) {z : ℂ} (hz : z ∈ scalarTube S η) :
    ‖oddFilterContinuation p P L k z‖ ≤ ((2*Real.pi)^(2*k+1))⁻¹ *
      ((2*k+1).factorial * (Real.pi⁻¹ *
        (4*L*B/δ + (2*Q/(L-Q))*B + (∫ t : ℝ, ‖p t‖)/(L-Q))) / η^(2*k+1)) := by
  have hsub : scalarTube S (2*η) ⊆ scalarTube L δ ∩ ball (0:ℂ) Q := by
    intro y hy
    exact ⟨scalarTube_mono hS hηδ hy, by
      simpa only [mem_ball_zero_iff] using scalarTube_norm_lt hSQ hy⟩
  have hh := (hilbertContinuation_analytic p.integrable hδ hB hQ hP hbound).differentiableOn.mono hsub
  have hb : ∀ y ∈ scalarTube S (2*η), ‖hilbertContinuation p P L y‖ ≤
      Real.pi⁻¹*(4*L*B/δ+(2*Q/(L-Q))*B+(∫ t : ℝ, ‖p t‖)/(L-Q)) :=
    fun y hy => hilbertContinuation_norm_le p.integrable hδ hB hQ0 hQ hP hbound (hsub hy)
  have hd := iteratedDeriv_bound_on_smaller_tube (show 0 < 2*η by positivity) hh hb (2*k+1)
    (by simpa using hz)
  unfold oddFilterContinuation
  rw [norm_mul]
  have hc : ‖Complex.I*((2*Real.pi*Complex.I:ℂ)^(2*k+1))⁻¹‖ = ((2*Real.pi)^(2*k+1))⁻¹ := by
    simp [norm_inv, norm_pow, Complex.norm_I, Complex.norm_real, abs_of_pos Real.pi_pos]
  rw [hc]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  simpa only [mul_div_cancel_left₀ η (by norm_num : (2:ℝ) ≠ 0)] using hd

/-- Same filtered bound with the approved sharp endpoint logarithm. -/
theorem oddFilterContinuation_norm_le_sharp (p : 𝓢(ℝ, ℂ))
    {P : ℂ → ℂ} {L δ B Q S η : ℝ}
    (hδ : 0 < δ) (hB : 0 ≤ B) (hQ0 : 0 < Q) (hQ : Q < L)
    (hη : 0 < η) (hS : S ≤ L) (hηδ : 2*η ≤ δ) (hSQ : S+2*η ≤ Q)
    (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    (k : ℕ) {z : ℂ} (hz : z ∈ scalarTube S η) :
    ‖oddFilterContinuation p P L k z‖ ≤ ((2*Real.pi)^(2*k+1))⁻¹ *
      ((2*k+1).factorial * (Real.pi⁻¹ *
        (4*L*B/δ + Real.log ((L+Q)/(L-Q))*B + (∫ t : ℝ, ‖p t‖)/(L-Q))) / η^(2*k+1)) := by
  have hsub : scalarTube S (2*η) ⊆ scalarTube L δ ∩ ball (0:ℂ) Q := by
    intro y hy
    exact ⟨scalarTube_mono hS hηδ hy, by
      simpa only [mem_ball_zero_iff] using scalarTube_norm_lt hSQ hy⟩
  have hh := (hilbertContinuation_analytic p.integrable hδ hB hQ hP hbound).differentiableOn.mono hsub
  have hb : ∀ y ∈ scalarTube S (2*η), ‖hilbertContinuation p P L y‖ ≤
      Real.pi⁻¹*(4*L*B/δ+Real.log ((L+Q)/(L-Q))*B+(∫ t : ℝ, ‖p t‖)/(L-Q)) :=
    fun y hy => hilbertContinuation_norm_le_sharp p.integrable hδ hB hQ0 hQ hP hbound (hsub hy)
  have hd := iteratedDeriv_bound_on_smaller_tube (show 0 < 2*η by positivity) hh hb (2*k+1)
    (by simpa using hz)
  unfold oddFilterContinuation
  rw [norm_mul]
  have hc : ‖Complex.I*((2*Real.pi*Complex.I:ℂ)^(2*k+1))⁻¹‖ = ((2*Real.pi)^(2*k+1))⁻¹ := by
    simp [norm_inv, norm_pow, Complex.norm_I, Complex.norm_real, abs_of_pos Real.pi_pos]
  rw [hc]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  simpa only [mul_div_cancel_left₀ η (by norm_num : (2:ℝ) ≠ 0)] using hd

end QIRadon.Radon
