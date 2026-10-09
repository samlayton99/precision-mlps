import QIRadon.Radon.Reconstruction
import QIRadon.Radon.HilbertAnalytic

/-! Actual polynomial/Hilbert decomposition of the canonical Radon frequency
filter, including the cycles-frequency factors. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology
namespace QIRadon.Radon

/-- The actual inverse Fourier slice is Schwartz, for every fixed direction. -/
def projectionSchwartz {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]
    (f : 𝓢(V, ℂ)) (θ : sphere (0:V) 1) : 𝓢(ℝ, ℂ) :=
  𝓕⁻ (restrictDirection (𝓕 f) θ)

@[simp] theorem fourier_projectionSchwartz {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]
    (f : 𝓢(V, ℂ)) (θ : sphere (0:V) 1) (ξ : ℝ) :
    𝓕 (projectionSchwartz f θ : ℝ → ℂ) ξ = 𝓕 (f : V → ℂ) (ξ • θ.val) := by
  simp [projectionSchwartz, ← SchwartzMap.fourier_coe, FourierTransform.fourier_fourierInv_eq]

/-- The Schwartz inverse slice equals the actual hyperplane projection at every
point where the latter is continuous. This is actual Fourier inversion. -/
theorem projectionSchwartz_eq_radon {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]
    (f : 𝓢(V, ℂ)) (θ : sphere (0:V) 1) {s : ℝ}
    (hc : ContinuousAt (hyperplaneRadon f θ) s) :
    projectionSchwartz f θ s = hyperplaneRadon f θ s := by
  have hslice : 𝓕 (hyperplaneRadon f θ) = (restrictDirection (𝓕 f) θ : ℝ → ℂ) := by
    funext ξ
    exact fourier_slice f.integrable θ ξ
  have hi := hyperplaneRadon_integrable f.integrable θ
  have hhi : Integrable (𝓕 (hyperplaneRadon f θ)) := by
    rw [hslice]
    exact (restrictDirection (𝓕 f) θ).integrable
  have hinv := hi.fourierInv_fourier_eq hhi hc
  rw [hslice] at hinv
  simpa only [projectionSchwartz, SchwartzMap.fourierInv_coe] using hinv

/-- Bundle each ordinary real derivative as its actual Schwartz function. -/
def schwartzDerivative (p : 𝓢(ℝ, ℂ)) : ℕ → 𝓢(ℝ, ℂ)
  | 0 => p
  | n+1 => SchwartzMap.derivCLM ℂ ℂ (schwartzDerivative p n)

@[simp] theorem schwartzDerivative_apply (p : 𝓢(ℝ, ℂ)) (n : ℕ) (s : ℝ) :
    schwartzDerivative p n s = iteratedDeriv n p s := by
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    change deriv (schwartzDerivative p n : ℝ → ℂ) s = _
    have heq : (schwartzDerivative p n : ℝ → ℂ)=iteratedDeriv n p := funext ih
    rw [heq, iteratedDeriv_succ]

theorem fourier_schwartzDerivative (p : 𝓢(ℝ, ℂ)) (n : ℕ) (ξ : ℝ) :
    𝓕 (schwartzDerivative p n : ℝ → ℂ) ξ =
      (2*Real.pi*Complex.I*(ξ:ℂ))^n * 𝓕 (p : ℝ → ℂ) ξ := by
  have heq : (schwartzDerivative p n : ℝ → ℂ)=iteratedDeriv n p :=
    funext (schwartzDerivative_apply p n)
  rw [heq]
  have h := Real.fourier_iteratedDeriv (N := (⊤:ℕ∞)) (p.smooth ⊤)
    (fun k _ => show Integrable (iteratedDeriv k p) from
      (schwartzDerivative p k).integrable.congr (Filter.Eventually.of_forall (schwartzDerivative_apply p k)))
    (show (n:ℕ∞) ≤ ⊤ from le_top)
  simpa only [smul_eq_mul] using congrFun h ξ

/-- The full signed-frequency Radon filter before its surface-area multiplier. -/
def radialFilter (p : ℝ → ℂ) (n : ℕ) (s : ℝ) : ℂ :=
  QIRadon.Kernel.synthesize (fun ξ => (|ξ|^n:ℝ) * 𝓕 p ξ) s

private theorem sign_mul_pow_odd (ξ : ℝ) (k : ℕ) :
    Real.sign ξ * ξ^(2*k+1) = |ξ|^(2*k+1) := by
  rcases lt_trichotomy ξ 0 with h | h | h
  · rw [Real.sign_of_neg h, abs_of_neg h, neg_pow]
    simp [pow_add, pow_mul]
  · simp [h]
  · rw [Real.sign_of_pos h, abs_of_pos h, one_mul]

/-- Even filter order is an ordinary even derivative, with its explicit Fourier
normalization. For d odd, n=d−1 is of this form. -/
theorem radialFilter_even (p : 𝓢(ℝ, ℂ)) (k : ℕ) (s : ℝ) :
    radialFilter p (2*k) s =
      ((2*Real.pi*Complex.I)^(2*k))⁻¹ * iteratedDeriv (2*k) p s := by
  have hc : (2*Real.pi*Complex.I:ℂ) ≠ 0 := by
    exact mul_ne_zero (mul_ne_zero (by norm_num) (by exact_mod_cast Real.pi_ne_zero)) Complex.I_ne_zero
  have hspec : (fun ξ : ℝ => (|ξ|^(2*k):ℝ) * 𝓕 (p:ℝ→ℂ) ξ) =
      fun ξ => ((2*Real.pi*Complex.I)^(2*k))⁻¹ * 𝓕 (schwartzDerivative p (2*k):ℝ→ℂ) ξ := by
    funext ξ
    rw [fourier_schwartzDerivative, mul_pow]
    have habs : |ξ|^(2*k)=ξ^(2*k) := by simp [pow_mul, sq_abs]
    rw [habs]
    push_cast
    field_simp
    simp only [mul_pow]
    ring
  unfold radialFilter
  rw [hspec, QIRadon.Kernel.synthesize_eq_integral]
  have hm : (∫ ξ : ℝ, QIRadon.Kernel.phase (ξ*s) *
      (((2*Real.pi*Complex.I)^(2*k))⁻¹ * 𝓕 (schwartzDerivative p (2*k):ℝ→ℂ) ξ)) =
      ((2*Real.pi*Complex.I)^(2*k))⁻¹ *
      QIRadon.Kernel.synthesize (𝓕 (schwartzDerivative p (2*k):ℝ→ℂ)) s := by
    rw [QIRadon.Kernel.synthesize_eq_integral, ← integral_const_mul]
    apply integral_congr_ae
    filter_upwards with ξ
    ring
  rw [hm]
  have hinv := (schwartzDerivative p (2*k)).integrable.fourierInv_fourier_eq (v := s)
    (𝓕 (schwartzDerivative p (2*k))).integrable (schwartzDerivative p (2*k)).continuous.continuousAt
  simpa only [QIRadon.Kernel.synthesize, schwartzDerivative_apply] using congrArg
    (fun z : ℂ => ((2*Real.pi*Complex.I)^(2*k))⁻¹*z) hinv

/-- Odd filter order is the Hilbert transform of an odd derivative. For even d,
this is the canonical even-dimensional Radon filter, not an alternative choice. -/
theorem radialFilter_odd (p : 𝓢(ℝ, ℂ)) (k : ℕ) (s : ℝ) :
    radialFilter p (2*k+1) s =
      (Complex.I*((2*Real.pi*Complex.I)^(2*k+1))⁻¹) *
        hilbertFourier (schwartzDerivative p (2*k+1)) s := by
  have hc : (2*Real.pi*Complex.I:ℂ) ≠ 0 := by
    exact mul_ne_zero (mul_ne_zero (by norm_num) (by exact_mod_cast Real.pi_ne_zero)) Complex.I_ne_zero
  have hspec : (fun ξ : ℝ => (|ξ|^(2*k+1):ℝ) * 𝓕 (p:ℝ→ℂ) ξ) =
      fun ξ => (Complex.I*((2*Real.pi*Complex.I)^(2*k+1))⁻¹) *
        (-Complex.I*(Real.sign ξ:ℂ)*𝓕 (schwartzDerivative p (2*k+1):ℝ→ℂ) ξ) := by
    funext ξ
    rw [fourier_schwartzDerivative, mul_pow, ← sign_mul_pow_odd ξ k]
    push_cast
    field_simp
    simp [Complex.I_sq]
    ring
  unfold radialFilter
  rw [hspec, QIRadon.Kernel.synthesize_eq_integral]
  unfold hilbertFourier
  rw [QIRadon.Kernel.synthesize_eq_integral, ← integral_const_mul]
  apply integral_congr_ae
  filter_upwards with ξ
  ring

/-- The canonical Radon profile is exactly the radial filter of the actual
Schwartz inverse slice, with the sphere-area normalization retained. -/
theorem filteredRadonProfile_eq_radialFilter {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]
    [Nontrivial V] (f : 𝓢(V, ℂ)) (θ : sphere (0:V) 1) (s : ℝ) :
    filteredRadonProfile f θ s = (surfaceArea V/2:ℂ) *
      radialFilter (projectionSchwartz f θ) (Module.finrank ℝ V-1) s := by
  unfold filteredRadonProfile radialFilter
  rw [QIRadon.Kernel.synthesize_eq_integral]
  congr 1
  apply integral_congr_ae
  filter_upwards with ξ
  rw [fourier_slice f.integrable θ ξ, fourier_projectionSchwartz]
  simp only [QIRadon.Kernel.phase, Complex.real_smul, Circle.smul_def, smul_eq_mul]
  ring

/-- Actual complex derivatives restrict to actual real derivatives of any locally
agreeing profile. No derivative agreement is supplied as an assumption. -/
theorem iteratedDeriv_real_of_complex {p : ℝ → ℂ} {P : ℂ → ℂ}
    {U : Set ℂ} {J : Set ℝ} (hJ : IsOpen J)
    (hP : AnalyticOnNhd ℂ P U) (hJU : ∀ s ∈ J, (s:ℂ) ∈ U)
    (hagree : ∀ s ∈ J, p s=P (s:ℂ)) (n : ℕ) {s : ℝ} (hs : s ∈ J) :
    iteratedDeriv n p s = iteratedDeriv n P (s:ℂ) := by
  induction n generalizing s with
  | zero => exact hagree s hs
  | succ n ih =>
    rw [iteratedDeriv_succ, iteratedDeriv_succ]
    have heq : iteratedDeriv n p =ᶠ[𝓝 s] (fun t : ℝ => iteratedDeriv n P (t:ℂ)) := by
      filter_upwards [hJ.mem_nhds hs] with t ht
      exact ih ht
    rw [heq.deriv_eq]
    have ha : AnalyticAt ℂ (iteratedDeriv n P) (s:ℂ) := by
      simpa only [iteratedDeriv_eq_iterate] using (hP (s:ℂ) (hJU s hs)).iterated_deriv n
    exact ha.differentiableAt.hasDerivAt.comp_ofReal.deriv

/-- Cauchy estimates on an explicitly smaller tube give all derivative bounds
needed by either parity of the Radon filter. -/
theorem iteratedDeriv_bound_on_smaller_tube {P : ℂ → ℂ} {L δ B : ℝ}
    (hδ : 0 < δ) (hP : DifferentiableOn ℂ P (scalarTube L δ))
    (hbound : ∀ z ∈ scalarTube L δ, ‖P z‖ ≤ B)
    (n : ℕ) {z : ℂ} (hz : z ∈ scalarTube L (δ/2)) :
    ‖iteratedDeriv n P z‖ ≤ n.factorial*B/(δ/2)^n := by
  obtain ⟨t, ht, hzt⟩ : ∃ (t : ℝ), t ∈ Icc (-L) L ∧ z ∈ ball (t:ℂ) (δ/2) := by
    simpa only [scalarTube, mem_iUnion, exists_prop] using hz
  have hsub : closedBall z (δ/2) ⊆ scalarTube L δ := by
    intro y hy
    simp only [scalarTube, mem_iUnion]
    refine ⟨t, ht, ?_⟩
    have hd := dist_triangle y z (t:ℂ)
    have hy' : dist y z ≤ δ/2 := hy
    have hz' : dist z (t:ℂ) < δ/2 := hzt
    change dist y (t:ℂ) < δ
    linarith
  apply Complex.norm_iteratedDeriv_le_of_forall_mem_sphere_norm_le n (by positivity)
  · apply DifferentiableOn.diffContOnCl
    rw [closure_ball z (by positivity : (δ/2:ℝ) ≠ 0)]
    exact hP.mono hsub
  · intro y hy
    exact hbound y (hsub hy.le)

/-- Differentiating inverse Fourier synthesis inserts the positive-sign frequency
power; integrable moments supply the actual differentiation hypotheses. -/
theorem iteratedDeriv_synthesize {b : ℝ → ℂ}
    (hmom : ∀ k : ℕ, Integrable (fun ξ : ℝ => ξ^k • b ξ)) (n : ℕ) :
    iteratedDeriv n (QIRadon.Kernel.synthesize b) =
      QIRadon.Kernel.synthesize (fun ξ => (2*Real.pi*Complex.I*(ξ:ℂ))^n*b ξ) := by
  have href (k : ℕ) : Integrable (fun ξ : ℝ => ξ^k • b (-ξ)) := by
    have hh := Integrable.smul ((-1:ℝ)^k) ((hmom k).comp_neg)
    apply hh.congr
    filter_upwards with ξ
    simp only [Pi.smul_apply, smul_smul, ← mul_pow, neg_one_mul, neg_neg]
  unfold QIRadon.Kernel.synthesize
  rw [Real.fourierInv_eq_fourier_comp_neg]
  rw [Real.iteratedDeriv_fourier (N := (⊤:ℕ∞)) (fun k _ => href k) (show (n:ℕ∞) ≤ ⊤ from le_top)]
  rw [Real.fourierInv_eq_fourier_comp_neg]
  congr 1
  funext ξ
  simp only [smul_eq_mul, Complex.ofReal_neg]
  congr 2
  ring

private theorem hilbert_sign_measurable : Measurable (fun ξ : ℝ => -Complex.I*(Real.sign ξ:ℂ)) := by
  have hs : Measurable Real.sign := by
    unfold Real.sign
    exact measurable_const.ite measurableSet_Iio (measurable_const.ite measurableSet_Ioi measurable_const)
  exact measurable_const.mul (Complex.continuous_ofReal.measurable.comp hs)

/-- Hilbert synthesis has every frequency moment when its input is Schwartz. -/
theorem hilbert_spectrum_moments (p : 𝓢(ℝ, ℂ)) (k : ℕ) :
    Integrable (fun ξ : ℝ => ξ^k • (-Complex.I*(Real.sign ξ:ℂ)*𝓕 (p:ℝ→ℂ) ξ)) := by
  have hw : Integrable (fun ξ : ℝ => (ξ:ℂ)^k*𝓕 (p:ℝ→ℂ) ξ) := by
    apply ((𝓕 p : 𝓢(ℝ, ℂ)).integrable_pow_mul volume k).mono'
    · exact ((by fun_prop : Continuous (fun ξ : ℝ => (ξ:ℂ)^k)).aestronglyMeasurable.mul
        (𝓕 p).continuous.aestronglyMeasurable)
    · filter_upwards with ξ
      simp [norm_pow, Complex.norm_real, SchwartzMap.fourier_coe]
  have hh := hw.bdd_mul (c := 1) hilbert_sign_measurable.aestronglyMeasurable
    (Filter.Eventually.of_forall (fun ξ => show ‖-Complex.I*(Real.sign ξ:ℂ)‖ ≤ 1 by
      simpa [hilbertAbelMultiplier] using norm_hilbertAbelMultiplier_le_one (le_refl 0) ξ))
  convert hh using 1
  funext ξ
  simp only [Complex.real_smul, Complex.ofReal_pow]
  ring

/-- For the actual Schwartz profile, Hilbert filtering commutes with every real
derivative. This lets the analytic bound use ‖p‖₁ rather than ‖p^(n)‖₁. -/
theorem hilbertFourier_schwartzDerivative (p : 𝓢(ℝ, ℂ)) (n : ℕ) :
    hilbertFourier (schwartzDerivative p n) = iteratedDeriv n (hilbertFourier p) := by
  unfold hilbertFourier
  rw [iteratedDeriv_synthesize (hilbert_spectrum_moments p) n]
  apply congrArg QIRadon.Kernel.synthesize
  funext ξ
  rw [fourier_schwartzDerivative]
  ring

/-- The even-dimensional filter can be formed by differentiating one locally
holomorphic Hilbert continuation, as in the approved finite construction. -/
theorem radialFilter_odd_deriv_hilbert (p : 𝓢(ℝ, ℂ)) (k : ℕ) (s : ℝ) :
    radialFilter p (2*k+1) s =
      (Complex.I*((2*Real.pi*Complex.I)^(2*k+1))⁻¹) *
        iteratedDeriv (2*k+1) (hilbertFourier p) s := by
  rw [radialFilter_odd, hilbertFourier_schwartzDerivative]

end QIRadon.Radon
