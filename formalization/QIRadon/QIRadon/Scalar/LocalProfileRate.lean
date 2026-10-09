import QIRadon.Scalar.ProfilePreparation
import QIRadon.Scalar.ScalarRate

/-! Actual finite-kernel approximation of a locally analytic profile.
The original profile is assumed analytic only on its Bernstein ellipse. The
sampling/window preparation, its Fourier moments, and its actual derivative
synthesis are all derived rather than inserted as approximation hypotheses. -/

noncomputable section
open Set MeasureTheory FourierTransform
open scoped Real SchwartzMap

namespace QIRadon.LocalProfileRate

open QIRadon.ProfilePreparation QIRadon.Kernel QIRadon.KernelRates QIRadon.ScalarRate

/-- Explicit derivative spectrum in cycles frequency. -/
def derivativeSpectrum (N : ℕ) (f : ℝ → ℝ) (T₀ : ℝ) (r : ℕ) (ξ : ℝ) : ℂ :=
  (2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r *
    𝓕 (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) ξ

/-- The real profile actually passed to the activation construction. -/
def realPrepared (N : ℕ) (f : ℝ → ℝ) (T₀ x : ℝ) : ℝ :=
  (preparedProfile N f T₀ (x : ℂ)).re

/-- Smoothness follows from the actual polynomial-exponential formula. -/
theorem prepared_contDiff (N : ℕ) (f : ℝ → ℝ) (T₀ : ℝ) :
    ContDiff ℝ ⊤ (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) := by
  exact ((preparedProfile_differentiable N f T₀).contDiff.restrict_scalars ℝ).comp
    Complex.ofRealCLM.contDiff

theorem realPrepared_contDiff (N : ℕ) (f : ℝ → ℝ) (T₀ : ℝ) :
    ContDiff ℝ ⊤ (realPrepared N f T₀) := by
  have hh := prepared_contDiff N f T₀
  exact Complex.reCLM.contDiff.comp hh

/-- Real-part projection commutes with every derivative of the actual smooth
prepared profile. -/
theorem iteratedDeriv_realPrepared (N : ℕ) (f : ℝ → ℝ) (T₀ : ℝ) (r : ℕ) (x : ℝ) :
    iteratedDeriv r (realPrepared N f T₀) x =
      (iteratedDeriv r (fun t : ℝ => preparedProfile N f T₀ (t : ℂ)) x).re := by
  have hc : ContDiffAt ℝ r (fun t : ℝ => preparedProfile N f T₀ (t : ℂ)) x :=
    (prepared_contDiff N f T₀).contDiffAt.of_le (by simp)
  have hh := Complex.reCLM.iteratedFDeriv_comp_left hc le_rfl
  have he := congrArg (fun A => A (fun _ => (1 : ℝ))) hh
  exact he

theorem realPrepared_derivative_synthesis {f : ℝ → ℝ} {M T₀ : ℝ}
    (hM : 0 ≤ M) (hT : 0 < T₀) {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (r : ℕ) (x : ℝ) :
    iteratedDeriv r (realPrepared N f T₀) x =
      (synthesize (derivativeSpectrum N f T₀ r) x).re := by
  rw [iteratedDeriv_realPrepared]
  have hh := congrFun (preparedProfile_derivative_synthesis hM hT hN hf r) x
  exact congrArg Complex.re hh

/-- General weighted moment estimate for the actual derivative spectrum of an
exponentially decaying continuous spectrum. -/
theorem derivativeSpectrum_weighted_of_decay {H : ℝ → ℂ} {D c σ : ℝ}
    (hH : Continuous H) (hD : 0 ≤ D) (_hc : 0 < c) (hσ : σ < c)
    (hb : ∀ ξ : ℝ, ‖H ξ‖ ≤ D * Real.exp (-c * |ξ|)) (r : ℕ) :
    Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) *
      ‖(2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * H ξ‖) ∧
    (∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖(2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * H ξ‖) ≤
      (D * (2 * Real.pi) ^ r) * (2 * (r.factorial : ℝ) / (c - σ) ^ (r + 1)) := by
  obtain ⟨hi, hival⟩ := absolute_exponential_moment r (sub_pos.mpr hσ)
  let C := D * (2 * Real.pi) ^ r
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hmajor := hi.const_mul C
  have hpoint : ∀ ξ : ℝ, Real.exp (σ * |ξ|) *
      ‖(2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * H ξ‖ ≤
        C * (|ξ| ^ r * Real.exp (-(c - σ) * |ξ|)) := by
    intro ξ
    have hfac : ‖2 * Real.pi * Complex.I * (ξ : ℂ)‖ = 2 * Real.pi * |ξ| := by
      simp [Complex.norm_real, Real.norm_eq_abs, abs_of_pos Real.pi_pos]
    rw [norm_mul, norm_pow, hfac, mul_pow]
    have hh := mul_le_mul_of_nonneg_left (hb ξ)
      (show 0 ≤ Real.exp (σ * |ξ|) * ((2 * Real.pi) ^ r * |ξ| ^ r) by positivity)
    have he : Real.exp (σ * |ξ|) * Real.exp (-c * |ξ|) =
        Real.exp (-(c - σ) * |ξ|) := by rw [← Real.exp_add]; congr 1; ring_nf
    convert! hh using 1 <;> (try dsimp [C]) <;> (try rw [← he]) <;> ring_nf
  have hcont : Continuous (fun ξ : ℝ => Real.exp (σ * |ξ|) *
      ‖(2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * H ξ‖) := by fun_prop
  have hint : Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) *
      ‖(2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r * H ξ‖) := by
    apply hmajor.mono' hcont.aestronglyMeasurable
    apply Filter.Eventually.of_forall
    intro ξ
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    exact hpoint ξ
  refine ⟨hint, ?_⟩
  calc
    _ ≤ ∫ ξ : ℝ, C * (|ξ| ^ r * Real.exp (-(c - σ) * |ξ|)) := integral_mono hint hmajor hpoint
    _ = _ := by rw [integral_const_mul, hival]

/-- Preparation's actual derivative-spectrum moment with a fixed useful
cycles-frequency weight sigma=pi. -/
theorem prepared_derivative_weighted_moment {f : ℝ → ℝ} {M T₀ : ℝ}
    (hM : 0 ≤ M) (hT : 0 < T₀) {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (r : ℕ) :
    Integrable (fun ξ : ℝ => Real.exp (Real.pi * |ξ|) * ‖derivativeSpectrum N f T₀ r ξ‖) ∧
    (∫ ξ : ℝ, Real.exp (Real.pi * |ξ|) * ‖derivativeSpectrum N f T₀ r ξ‖) ≤
      (preparationEnvelope N M 1 T₀ * Real.sqrt (Real.pi / ((N : ℝ) / (4 * T₀ ^ 2)))) *
        (2 * Real.pi) ^ r * (2 * (r.factorial : ℝ) / Real.pi ^ (r + 1)) := by
  have hi : Integrable (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) := by
    simpa only [Complex.ofReal_zero, zero_mul, add_zero] using
      preparedProfile_horizontal_integrable hM (show (0 : ℝ) ≤ 1 by norm_num) hT
        (show |(0 : ℝ)| ≤ 1 by norm_num) hN hf
  have hc : Continuous (𝓕 (fun x : ℝ => preparedProfile N f T₀ (x : ℂ))) := by
    change Continuous (VectorFourier.fourierIntegral Real.fourierChar volume (innerₗ ℝ)
      (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)))
    exact VectorFourier.fourierIntegral_continuous Real.continuous_fourierChar (by fun_prop) hi
  have hb := derivativeSpectrum_weighted_of_decay hc
    (mul_nonneg (preparationEnvelope_nonneg hM (show (0 : ℝ) ≤ 1 by norm_num) hT) (Real.sqrt_nonneg _))
    (show 0 < 2 * Real.pi by positivity) (show Real.pi < 2 * Real.pi by linarith [Real.pi_pos])
    (fun ξ => by simpa only [mul_one] using
      preparedProfile_cycles_decay hM (show (0 : ℝ) ≤ 1 by norm_num) hT hN hf ξ) r
  simpa only [derivativeSpectrum, show 2 * Real.pi - Real.pi = Real.pi by ring_nf] using hb


/-- Preparation growth exponent for the fixed strip of half-width one. -/
def preparationGrowth (T₀ : ℝ) : ℝ :=
  Real.log (2 * (3 + 4 * T₀)) + 3 / (2 * T₀ ^ 2)

theorem preparationGrowth_pos {T₀ : ℝ} (hT : 0 < T₀) : 0 < preparationGrowth T₀ := by
  unfold preparationGrowth
  have hl : 0 < Real.log (2 * (3 + 4 * T₀)) := Real.log_pos (by linarith)
  positivity

/-- All N-independent factors in the prepared derivative-spectrum moment. -/
def localMomentConstant (M T₀ : ℝ) (r : ℕ) : ℝ :=
  (2 * M * Real.sqrt (4 * Real.pi * T₀ ^ 2)) *
    (2 * Real.pi) ^ r * (2 * (r.factorial : ℝ) / Real.pi ^ (r + 1))

theorem localMomentConstant_nonneg {M T₀ : ℝ} (hM : 0 ≤ M) (r : ℕ) :
    0 ≤ localMomentConstant M T₀ r := by unfold localMomentConstant; positivity

/-- The accuracy-dependent spectral norm is bounded by exp(O(N)), with no
uncontrolled polynomial/exponential constant hidden in the statement. -/
theorem prepared_moment_exp_bound {f : ℝ → ℝ} {M T₀ : ℝ}
    (hM : 0 ≤ M) (hT : 0 < T₀) {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (r : ℕ) :
    (∫ ξ : ℝ, Real.exp (Real.pi * |ξ|) * ‖derivativeSpectrum N f T₀ r ξ‖) ≤
      localMomentConstant M T₀ r * Real.exp ((preparationGrowth T₀ + 1) * N) := by
  apply (prepared_derivative_weighted_moment hM hT hN hf r).2.trans
  have hN0 : 0 < (N : ℝ) := Nat.cast_pos.mpr hN
  have hN1 : 1 ≤ (N : ℝ) := by exact_mod_cast hN
  have hsqrt : Real.sqrt (Real.pi / ((N : ℝ) / (4 * T₀ ^ 2))) ≤
      Real.sqrt (4 * Real.pi * T₀ ^ 2) := by
    apply Real.sqrt_le_sqrt
    have hid : Real.pi / ((N : ℝ) / (4 * T₀ ^ 2)) = (4 * Real.pi * T₀ ^ 2) / N := by field_simp
    rw [hid]
    exact div_le_self (by positivity) hN1
  have hpref : (N : ℝ) + 1 ≤ Real.exp (N : ℝ) := Real.add_one_le_exp _
  have henv : preparationEnvelope N M 1 T₀ ≤
      (2 * M) * Real.exp ((preparationGrowth T₀ + 1) * N) := by
    rw [preparationEnvelope_exp_form N (by norm_num) hT]
    have hid : Real.log (2 * (1 + 2 * (1 : ℝ) + 4 * T₀)) +
        3 * (1 : ℝ) ^ 2 / (2 * T₀ ^ 2) = preparationGrowth T₀ := by
      unfold preparationGrowth
      norm_num
    rw [hid]
    have hm := mul_le_mul_of_nonneg_left hpref (show 0 ≤ 2 * M by positivity)
    have hm' := mul_le_mul_of_nonneg_right hm (Real.exp_pos ((N : ℝ) * preparationGrowth T₀)).le
    calc
      _ ≤ (2 * M * Real.exp (N : ℝ)) * Real.exp ((N : ℝ) * preparationGrowth T₀) := hm'
      _ = _ := by
        rw [mul_assoc, ← Real.exp_add]
        congr 2
        ring_nf
  have hfirst := mul_le_mul henv hsqrt (Real.sqrt_nonneg _)
    (mul_nonneg (show 0 ≤ 2 * M by positivity) (Real.exp_pos _).le)
  have hrest : 0 ≤ (2 * Real.pi) ^ r * (2 * (r.factorial : ℝ) / Real.pi ^ (r + 1)) := by positivity
  have hm := mul_le_mul_of_nonneg_right hfirst hrest
  convert! hm using 1 <;> (try unfold localMomentConstant) <;> ring_nf

/-- Exponential suppression chosen linearly in preparation order, not from an
unknown solution norm. -/
def localSuppression (N : ℕ) (T₀ : ℝ) : ℝ :=
  (preparationGrowth T₀ + 3) * N

theorem localSuppression_ge_one {N : ℕ} {T₀ : ℝ} (hN : 0 < N) (hT : 0 < T₀) :
    1 ≤ localSuppression N T₀ := by
  have hg := preparationGrowth_pos hT
  have hn : 1 ≤ (N : ℝ) := by exact_mod_cast hN
  unfold localSuppression
  nlinarith

/-- The actual activation construction approximates the prepared profile with
an exponentially small error despite the growing preparation norm. -/
theorem prepared_activation_geometric_error
    {φ f : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ M T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (hM : 0 ≤ M) (hT : 0 < T₀) {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M)
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ) :
    let L := localSuppression N T₀
    let B := L / Real.pi
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := Finset.Icc (-(⌊(1 + D) / h⌋₊ : ℤ)) (⌊(1 + D) / h⌋₊ : ℤ)
    ∀ x ∈ Icc (-1 : ℝ) 1,
      |anchoredActivationNetwork φ
          (realReadoutDensity (derivativeSpectrum N f T₀ r) K γ (by
            have hL := localSuppression_ge_one hN hT
            have _hL0 : 0 < L := by linarith
            positivity) B)
          r m γ h J (realPrepared N f T₀) x - realPrepared N f T₀ x| ≤
        (localMomentConstant M T₀ r * envelopeConstant E S / (r.factorial : ℝ)) *
          Real.exp (-2 * N) := by
  have hp := preparedProfile_derivativeSpectrum_integrable hM hT hN hf r
  have hW := (prepared_derivative_weighted_moment hM hT hN hf r).1
  have hL := localSuppression_ge_one hN hT
  have hg : ContDiffOn ℝ r (realPrepared N f T₀) (Icc (-1 : ℝ) 1) :=
    ((realPrepared_contDiff N f T₀).of_le (by simp)).contDiffOn
  have hprofile : ∀ s ∈ Icc (-1 : ℝ) 1,
      iteratedDeriv r (realPrepared N f T₀) s = (synthesize (derivativeSpectrum N f T₀ r) s).re := by
    intro s hs
    exact realPrepared_derivative_synthesis hM hT hN hf r s
  have hb := analytic_activation_error hφ K hm hkernel E S hp Real.pi_pos hW hL hH hsep hZ hhalo
    (show (0 : ℝ) < 1 by norm_num) hg hprofile
  dsimp only at hb ⊢
  intro x hx
  apply (hb x hx).trans
  have hnorm := prepared_moment_exp_bound hM hT hN hf r
  have hfac : 0 ≤ envelopeConstant E S * Real.exp (-localSuppression N T₀) / (r.factorial : ℝ) := by
    have he := envelopeConstant_pos E S
    positivity
  have hmul := mul_le_mul_of_nonneg_right hnorm hfac
  have he : Real.exp ((preparationGrowth T₀ + 1) * N) * Real.exp (-localSuppression N T₀) =
      Real.exp (-2 * N) := by
    rw [← Real.exp_add]
    congr 1
    unfold localSuppression
    ring_nf
  calc
    _ = (∫ ξ : ℝ, Real.exp (Real.pi * |ξ|) * ‖derivativeSpectrum N f T₀ r ξ‖) *
        (envelopeConstant E S * Real.exp (-localSuppression N T₀) / (r.factorial : ℝ)) := by
      change (∫ ξ : ℝ, Real.exp (Real.pi * |ξ|) * ‖derivativeSpectrum N f T₀ r ξ‖) *
        envelopeConstant E S * Real.exp (-localSuppression N T₀) * (1 : ℝ) ^ r /
        (r.factorial : ℝ) = _
      simp only [one_pow, mul_one]
      ring_nf
    _ ≤ _ := hmul
    _ = _ := by
      calc
        _ = (localMomentConstant M T₀ r * envelopeConstant E S / (r.factorial : ℝ)) *
          (Real.exp ((preparationGrowth T₀ + 1) * N) * Real.exp (-localSuppression N T₀)) := by ring_nf
        _ = _ := by rw [he]


/-- The actual finite symmetric grid, including all halo centers. -/
def localGrid (N : ℕ) (T₀ H Z ν μ : ℝ) : Finset ℤ :=
  let L := localSuppression N T₀
  let B := L / Real.pi
  let γ := B / L ^ (1 / ν)
  let h := 1 / (H * B)
  let D := Z * L ^ (1 / μ) / γ
  Finset.Icc (-(⌊(1 + D) / h⌋₊ : ℤ)) (⌊(1 + D) / h⌋₊ : ℤ)

/-- The specific activation construction produced from the local sample data.
For r>=2 the explicit anchor polynomial still requires an activation-specific
network realization; this definition does not erase that obligation. -/
def localActivationNetwork (φ f : ℝ → ℝ) (K : 𝓢(ℝ, ℝ))
    (r N : ℕ) (m T₀ H Z ν μ : ℝ) (hN : 0 < N) (hT : 0 < T₀) : ℝ → ℝ :=
  let L := localSuppression N T₀
  let B := L / Real.pi
  let γ := B / L ^ (1 / ν)
  let h := 1 / (H * B)
  anchoredActivationNetwork φ
    (realReadoutDensity (derivativeSpectrum N f T₀ r) K γ (by
      have hL := localSuppression_ge_one hN hT
      have _hL0 : 0 < L := by linarith
      positivity) B)
    r m γ h (localGrid N T₀ H Z ν μ) (realPrepared N f T₀)

def windowRate (T₀ : ℝ) : ℝ :=
  (1 / T₀) ^ 2 - 1 - Real.log ((1 / T₀) ^ 2)

theorem windowRate_pos {T₀ : ℝ} (hT : 1 < T₀) : 0 < windowRate T₀ := by
  have ht : 0 < T₀ := lt_trans zero_lt_one hT
  apply QIRadon.flatWindow_rate_pos
  · positivity
  · have h1 : 0 < 1 / T₀ := one_div_pos.mpr ht
    have h2 : 1 / T₀ < 1 := (div_lt_one ht).mpr hT
    nlinarith

def localRate (δ T₀ : ℝ) : ℝ :=
  min (δ / 2) (min (windowRate T₀ / 2) 2)

theorem localRate_pos {δ T₀ : ℝ} (hδ : 0 < δ) (hT : 1 < T₀) :
    0 < localRate δ T₀ :=
  lt_min (half_pos hδ) (lt_min (half_pos (windowRate_pos hT)) (by norm_num))

def interpolationConstant (M δ : ℝ) : ℝ :=
  M * (6 + 8 / δ) / (1 - Real.exp (-δ))

def windowConstant (M T₀ : ℝ) : ℝ := 2 * M * (1 + 2 / windowRate T₀)

def localErrorConstant {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ)
    (M δ T₀ : ℝ) (r : ℕ) : ℝ :=
  interpolationConstant M δ + windowConstant M T₀ +
    localMomentConstant M T₀ r * envelopeConstant E S / (r.factorial : ℝ)

theorem ellipse_real_bound {g : ℂ → ℂ} {δ M x : ℝ}
    (hδ : 0 ≤ δ) (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M)
    (hx : |x| ≤ 1) : |(g (x : ℂ)).re| ≤ M := by
  apply (Complex.abs_re_le_norm _).trans
  apply hbound
  refine ⟨(Real.arccos x : ℂ), ?_, ?_⟩
  · simpa [closedStrip] using hδ
  · rw [← Complex.ofReal_cos, Real.cos_arccos (neg_le_of_abs_le hx) (le_of_abs_le hx)]

/-- Completed local-target scalar error theorem: local ellipse assumptions,
actual N+1 samples, actual profile preparation, and actual finite activation
construction. No global Fourier assumption on the original target occurs. -/
theorem local_analytic_activation_error
    {g : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) {N : ℕ} (hN : 0 < N)
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ) :
    ∀ x ∈ Icc (-1 : ℝ) 1,
      |localActivationNetwork φ (fun t : ℝ => (g (t : ℂ)).re) K r N m T₀ H Z ν μ hN
          (lt_trans zero_lt_one hT) x - (g (x : ℂ)).re| ≤
        localErrorConstant E S M δ T₀ r * Real.exp (-localRate δ T₀ * N) := by
  have hT0 : 0 < T₀ := lt_trans zero_lt_one hT
  have hM : 0 ≤ M := (abs_nonneg _).trans (ellipse_real_bound hδ.le hbound (x := 0) (by norm_num))
  let f : ℝ → ℝ := fun t => (g (t : ℂ)).re
  have hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M := by
    intro j hj
    exact ellipse_real_bound hδ.le hbound (sampleNode_abs_le_one (N + 1) j)
  have hnR : 0 ≤ (N : ℝ) := Nat.cast_nonneg N
  have hb := prepared_activation_geometric_error hφ K hm hkernel E S hM hT0 hN hf hH hsep hZ hhalo
  change ∀ x ∈ Icc (-1 : ℝ) 1,
    |localActivationNetwork φ f K r N m T₀ H Z ν μ hN hT0 x - f x| ≤ _
  intro x hx
  have hnet : |localActivationNetwork φ f K r N m T₀ H Z ν μ hN hT0 x - realPrepared N f T₀ x| ≤
      (localMomentConstant M T₀ r * envelopeConstant E S / (r.factorial : ℝ)) * Real.exp (-2 * N) := by
    exact hb x hx
  have hprep := preparedProfile_real_error hδ hg hbound hT N (abs_le.mpr hx)
  have hprep' : |realPrepared N f T₀ x - f x| ≤
      interpolationConstant M δ * Real.exp (-(δ / 2) * N) +
      (2 * (N + 1) * M) * Real.exp (-(N : ℝ) * windowRate T₀) := by
    simpa only [realPrepared, f, interpolationConstant, windowRate, abs_sub_comm] using hprep
  have hwin := linear_prefactor_geometric_bound (A := 1) (B := 1)
    (by norm_num) (by norm_num) (windowRate_pos hT) hnR
  have hwin' : (2 * (N + 1) * M) * Real.exp (-(N : ℝ) * windowRate T₀) ≤
      windowConstant M T₀ * Real.exp (-(windowRate T₀ / 2) * N) := by
    have hh := mul_le_mul_of_nonneg_left hwin (show 0 ≤ 2 * M by positivity)
    convert! hh using 1 <;> (try unfold windowConstant) <;> ring_nf
  have hden : 0 < 1 - Real.exp (-δ) := sub_pos.mpr (Real.exp_lt_one_iff.mpr (neg_neg_of_pos hδ))
  have hiC : 0 ≤ interpolationConstant M δ := by unfold interpolationConstant; positivity
  have hwC : 0 ≤ windowConstant M T₀ := by
    have hb := windowRate_pos hT
    unfold windowConstant
    positivity
  have hnC : 0 ≤ localMomentConstant M T₀ r * envelopeConstant E S / (r.factorial : ℝ) := by
    have h1 := localMomentConstant_nonneg (T₀ := T₀) hM r
    have h2 := envelopeConstant_pos E S
    positivity
  have hrate1 : localRate δ T₀ ≤ δ / 2 := min_le_left _ _
  have hrate2 : localRate δ T₀ ≤ windowRate T₀ / 2 := (min_le_right _ _).trans (min_le_left _ _)
  have hrate3 : localRate δ T₀ ≤ 2 := (min_le_right _ _).trans (min_le_right _ _)
  have he1 : Real.exp (-(δ / 2) * N) ≤ Real.exp (-localRate δ T₀ * N) := by
    apply Real.exp_le_exp.mpr
    exact mul_le_mul_of_nonneg_right (neg_le_neg hrate1) hnR
  have he2 : Real.exp (-(windowRate T₀ / 2) * N) ≤ Real.exp (-localRate δ T₀ * N) := by
    apply Real.exp_le_exp.mpr
    exact mul_le_mul_of_nonneg_right (neg_le_neg hrate2) hnR
  have he3 : Real.exp (-2 * N) ≤ Real.exp (-localRate δ T₀ * N) := by
    apply Real.exp_le_exp.mpr
    exact mul_le_mul_of_nonneg_right (neg_le_neg hrate3) hnR
  have hi := mul_le_mul_of_nonneg_left he1 hiC
  have hw := mul_le_mul_of_nonneg_left he2 hwC
  have hn := mul_le_mul_of_nonneg_left he3 hnC
  have htri := abs_sub_le (localActivationNetwork φ f K r N m T₀ H Z ν μ hN hT0 x)
    (realPrepared N f T₀ x) (f x)
  unfold localErrorConstant
  linarith [htri, hprep', hwin', hnet, hi, hw, hn]

/-- The local analytic construction counts every retained grid neuron,
including the entire physical halo. -/
theorem local_analytic_grid_width {T₀ H Z ν μ : ℝ}
    (hT : 0 < T₀) {N : ℕ} (hN : 0 < N) (hH : 0 < H) (hZ : 0 ≤ Z) :
    ((localGrid N T₀ H Z ν μ).card : ℝ) ≤
      ((2 * H / Real.pi + 2 * H * Z + 1) *
        (preparationGrowth T₀ + 3) ^ max 1 (1 / μ + 1 / ν)) *
        (N : ℝ) ^ max 1 (1 / μ + 1 / ν) := by
  have hw := analytic_grid_width_cycles (T := 1) (σ := Real.pi)
    (L := localSuppression N T₀) (μ := μ) (ν := ν) (Z := Z) (H := H)
    (by norm_num) Real.pi_pos (localSuppression_ge_one hN hT) hZ hH
  change ((localGrid N T₀ H Z ν μ).card : ℝ) ≤ _ at hw
  apply hw.trans_eq
  unfold localSuppression
  rw [Real.mul_rpow (by linarith [preparationGrowth_pos hT]) (Nat.cast_nonneg N)]
  ring_nf


/-- Integer preparation order chosen directly from the certified error constant
and rate. The max with one guarantees that the Gaussian window is nontrivial. -/
def localAccuracyOrder (C b ε : ℝ) : ℕ :=
  ⌈max 1 (max 1 (Real.log (C / ε)) / b)⌉₊

theorem localAccuracyOrder_pos (C b ε : ℝ) : 0 < localAccuracyOrder C b ε := by
  apply Nat.one_le_ceil_iff.mpr
  exact lt_of_lt_of_le zero_lt_one (le_max_left _ _)

theorem localAccuracyOrder_error {C b ε : ℝ}
    (hC : 0 ≤ C) (hb : 0 < b) (hε : 0 < ε) :
    C * Real.exp (-b * localAccuracyOrder C b ε) ≤ ε := by
  have hn : max 1 (Real.log (C / ε)) / b ≤ (localAccuracyOrder C b ε : ℝ) :=
    (le_max_right _ _).trans (Nat.le_ceil _)
  have hn' := (div_le_iff₀ hb).mp hn
  have he : Real.exp (-b * localAccuracyOrder C b ε) ≤
      Real.exp (-max 1 (Real.log (C / ε))) := by
    apply Real.exp_le_exp.mpr
    nlinarith
  exact (mul_le_mul_of_nonneg_left he hC).trans (log_accuracy_choice hC hε)

theorem localErrorConstant_nonneg {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ M δ T₀ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ)
    (hM : 0 ≤ M) (hδ : 0 < δ) (hT : 1 < T₀) (r : ℕ) :
    0 ≤ localErrorConstant E S M δ T₀ r := by
  have hd : 0 < 1 - Real.exp (-δ) := sub_pos.mpr (Real.exp_lt_one_iff.mpr (neg_neg_of_pos hδ))
  have hw := windowRate_pos hT
  have hc := localMomentConstant_nonneg (T₀ := T₀) hM r
  have he := envelopeConstant_pos E S
  unfold localErrorConstant interpolationConstant windowConstant
  positivity

/-- Fully explicit accuracy corollary: the order is an integer calculated from
ε and the displayed constants; the conclusion concerns the actual network. -/
theorem local_analytic_activation_accuracy
    {g : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z ε : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) (hε : 0 < ε)
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ) :
    let N := localAccuracyOrder (localErrorConstant E S M δ T₀ r) (localRate δ T₀) ε
    ∀ x ∈ Icc (-1 : ℝ) 1,
      |localActivationNetwork φ (fun t : ℝ => (g (t : ℂ)).re) K r N m T₀ H Z ν μ
          (localAccuracyOrder_pos _ _ _) (lt_trans zero_lt_one hT) x - (g (x : ℂ)).re| ≤ ε := by
  dsimp only
  intro x hx
  have hM : 0 ≤ M := (abs_nonneg _).trans (ellipse_real_bound hδ.le hbound (x := 0) (by norm_num))
  apply (local_analytic_activation_error hδ hg hbound hφ K hm hkernel E S hT
    (localAccuracyOrder_pos _ _ _) hH hsep hZ hhalo x hx).trans
  exact localAccuracyOrder_error (localErrorConstant_nonneg E S hM hδ hT r)
    (localRate_pos hδ hT) hε

/-- Explicit logarithmic width law, with the same integer order used by the
accuracy theorem and with every halo neuron included. -/
theorem local_analytic_accuracy_width {T₀ H Z ν μ C b ε : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) :
    ((localGrid (localAccuracyOrder C b ε) T₀ H Z ν μ).card : ℝ) ≤
      ((2 * H / Real.pi + 2 * H * Z + 1) *
        (preparationGrowth T₀ + 3) ^ max 1 (1 / μ + 1 / ν)) *
        (max 1 (max 1 (Real.log (C / ε)) / b) + 1) ^ max 1 (1 / μ + 1 / ν) := by
  apply (local_analytic_grid_width hT (localAccuracyOrder_pos C b ε) hH hZ).trans
  have hbase : 0 ≤ max (1 : ℝ) (max 1 (Real.log (C / ε)) / b) :=
    le_trans zero_le_one (le_max_left _ _)
  have hn : (localAccuracyOrder C b ε : ℝ) ≤ max 1 (max 1 (Real.log (C / ε)) / b) + 1 :=
    (Nat.ceil_lt_add_one hbase).le
  have hk : 0 ≤ max (1 : ℝ) (1 / μ + 1 / ν) := le_trans zero_le_one (le_max_left _ _)
  have hp := Real.rpow_le_rpow (Nat.cast_nonneg (localAccuracyOrder C b ε)) hn hk
  apply mul_le_mul_of_nonneg_left hp
  have hg := preparationGrowth_pos hT
  positivity


/-- The ellipse over the physical interval [-T,T], rather than a stronger
complex disk around the origin. -/
def scaledBernsteinEllipse (T δ : ℝ) : Set ℂ :=
  (fun z : ℂ => (T : ℂ) * z) '' bernsteinEllipse δ

/-- Physical-interval version. The actual samples are g(T times the Chebyshev
node), and the resulting network is evaluated at x/T. Thus its physical
centers and halo length are multiplied by T and its slopes divided by T. -/
theorem local_analytic_activation_error_scaled
    {g : ℂ → ℂ} {δ M T : ℝ}
    (hδ : 0 < δ) (hT : 0 < T)
    (hg : AnalyticOnNhd ℂ g (scaledBernsteinEllipse T δ))
    (hbound : ∀ z ∈ scaledBernsteinEllipse T δ, ‖g z‖ ≤ M)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (hT0 : 1 < T₀) {N : ℕ} (hN : 0 < N)
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ) :
    ∀ x ∈ Icc (-T) T,
      |localActivationNetwork φ (fun t : ℝ => (g ((T * t : ℝ) : ℂ)).re) K r N m T₀ H Z ν μ hN
          (lt_trans zero_lt_one hT0) (x / T) - (g (x : ℂ)).re| ≤
        localErrorConstant E S M δ T₀ r * Real.exp (-localRate δ T₀ * N) := by
  let gT : ℂ → ℂ := fun z => g ((T : ℂ) * z)
  have ha : AnalyticOnNhd ℂ gT (bernsteinEllipse δ) := by
    intro z hz
    exact (hg ((T : ℂ) * z) ⟨z, hz, rfl⟩).comp (analyticAt_const.mul analyticAt_id)
  have hb : ∀ z ∈ bernsteinEllipse δ, ‖gT z‖ ≤ M := by
    intro z hz
    exact hbound ((T : ℂ) * z) ⟨z, hz, rfl⟩
  have hnet := local_analytic_activation_error hδ ha hb hφ K hm hkernel E S hT0 hN hH hsep hZ hhalo
  intro x hx
  have hxT : x / T ∈ Icc (-1 : ℝ) 1 := by
    constructor
    · apply (le_div_iff₀ hT).mpr
      nlinarith [hx.1]
    · exact (div_le_one hT).mpr hx.2
  have hval : gT ((x / T : ℝ) : ℂ) = g (x : ℂ) := by
    dsimp [gT]
    congr 1
    push_cast
    field_simp [show (T : ℂ) ≠ 0 from Complex.ofReal_ne_zero.mpr hT.ne']
  have hf : (fun t : ℝ => (gT (t : ℂ)).re) =
      (fun t : ℝ => (g ((T * t : ℝ) : ℂ)).re) := by
    funext t
    simp only [gT, Complex.ofReal_mul]
  have hspec := hnet (x / T) hxT
  rw [hf, hval] at hspec
  exact hspec

end QIRadon.LocalProfileRate
