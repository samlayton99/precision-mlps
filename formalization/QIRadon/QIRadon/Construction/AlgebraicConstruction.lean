import QIRadon.Construction.MultidimensionalScaling
import QIRadon.Quadrature.CanonicalConvergence

/-! Actual finite construction in the global polynomial-Fourier/Sobolev class.
The canonical angular norm is a regularity hypothesis on the actual target;
neither an angular approximant nor an approximation rate is assumed. -/

noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap

namespace QIRadon.AlgebraicConstruction
open QIRadon.Gaussian QIRadon.Kernel QIRadon.KernelRates
open QIRadon.ScalarRate QIRadon.ScalarScaling QIRadon.Multidimensional

/-- The explicit angular prefactor; t is the angular regularity after the
uniform-norm loss (d-1)/2. Here the ambient dimension is k+2. -/
def angularConstant (k : ℕ) (A t : ℝ) : ℝ :=
  harmonicGrowthConstant k * A / Real.sqrt (2*t)

theorem angularConstant_nonneg (k : ℕ) {A t : ℝ} (hA : 0 ≤ A) :
    0 ≤ angularConstant k A t := by
  unfold angularConstant
  exact div_nonneg (mul_nonneg (harmonicGrowthConstant_nonneg k) hA) (Real.sqrt_nonneg _)

/-- A simpler power-law form of the proved canonical Sobolev quadrature bound. -/
theorem canonical_sobolev_power_error (q : ℕ) (hq : 0 < q) (k : ℕ)
    (F : C(UnitSphere (k+1), ℝ)) (heven : ∀ θ, F (sphereNeg (k+1) θ) = F θ)
    {s A : ℝ} (hs : ((k:ℝ)+1)/2 < s) (hA : 0 ≤ A)
    (hsum : Summable (fun j : ℕ =>
      (((j:ℝ)+1)^s * ‖continuousHarmonicCoordinates (k+1) F j‖)^2))
    (henergy : (∑' j : ℕ,
      (((j:ℝ)+1)^s * ‖continuousHarmonicCoordinates (k+1) F j‖)^2) ≤ A^2) :
    |(∫ θ, F θ ∂normalizedSphere (k+1)) -
      ∑ i : SphereHalfIndex q (k+1), foldedSphereWeight q hq (k+1) i *
        F (sphereNodeOnSphere q hq (k+1) i.val)| ≤
      angularConstant k A (s-((k:ℝ)+1)/2) * (q:ℝ)^(-(s-((k:ℝ)+1)/2)) := by
  have hqR : 0 < (q:ℝ) := by exact_mod_cast hq
  have hn : q ≤ 2*q-1 := by omega
  have hnR : (q:ℝ) ≤ ((2*q-1:ℕ):ℝ) := by exact_mod_cast hn
  have hn0 : 0 < ((2*q-1:ℕ):ℝ) := hqR.trans_le hnR
  have ht : 0 < s-((k:ℝ)+1)/2 := sub_pos.mpr hs
  have hexp : (k:ℝ)+1-2*s = -2*(s-((k:ℝ)+1)/2) := by ring
  have hden : 2*s-(k:ℝ)-1 = 2*(s-((k:ℝ)+1)/2) := by ring
  have hpow := Real.rpow_le_rpow_of_nonpos hqR hnR
    (show -2*(s-((k:ℝ)+1)/2) ≤ 0 by linarith)
  have hroot : Real.sqrt ((q:ℝ)^(-2*(s-((k:ℝ)+1)/2))) =
      (q:ℝ)^(-(s-((k:ℝ)+1)/2)) := by
    rw [Real.sqrt_eq_rpow, ← Real.rpow_mul hqR.le]
    congr 1
    ring
  have hb := canonical_foldedSphere_sobolev_error q hq k F heven s A hs hA hsum henergy
  rw [hexp, hden] at hb
  apply hb.trans
  have hsq := Real.sqrt_le_sqrt (div_le_div_of_nonneg_right hpow (by positivity :
    0 ≤ 2*(s-((k:ℝ)+1)/2)))
  rw [Real.sqrt_div (Real.rpow_nonneg hqR.le _), hroot] at hsq
  calc
    _ ≤ harmonicGrowthConstant k * A *
        ((q:ℝ)^(-(s-((k:ℝ)+1)/2)) / Real.sqrt (2*(s-((k:ℝ)+1)/2))) := by
      gcongr
      exact mul_nonneg (harmonicGrowthConstant_nonneg k) hA
    _ = _ := by unfold angularConstant; ring

/-- Actual algebraic finite-network error. Angular and radial regularity are
separate, allowing different resolutions rather than imposing isotropy. -/
theorem algebraic_finite_error
    (q : ℕ) (hq : 0 < q) (k : ℕ) {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (g : UnitSphere (k+1) → ℝ → ℝ) (pHat : UnitSphere (k+1) → ℝ → ℂ)
    {s B H T Z A : ℝ} (hs : 0 ≤ s) (hB : 0 < B) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b/2)*Z^μ)
    (hT : 0 < T) (hp : ∀ θ, Integrable (pHat θ))
    (hW : ∀ θ, Integrable (fun ξ : ℝ => (1+|ξ|)^s * ‖pHat θ ξ‖))
    (hWbound : ∀ θ, (∫ ξ : ℝ, (1+|ξ|)^s * ‖pHat θ ξ‖) ≤ A)
    (hg : ∀ θ, ContDiffOn ℝ r (g θ) (Set.Icc (-T) T))
    (hprofile : ∀ θ y, y ∈ Set.Icc (-T) T →
      iteratedDeriv r (g θ) y = (synthesize (pHat θ) y).re)
    (x : EuclideanSpace ℝ (Fin (k+1+1))) (hx : ‖x‖ ≤ T)
    (F : C(UnitSphere (k+1), ℝ)) (hF : ∀ θ, F θ = g θ (inner ℝ θ.val x))
    (heven : ∀ θ, F (sphereNeg (k+1) θ) = F θ)
    {sang Aang : ℝ} (hsang : ((k:ℝ)+1)/2 < sang) (hAang : 0 ≤ Aang)
    (hsum : Summable (fun j : ℕ =>
      (((j:ℝ)+1)^sang * ‖continuousHarmonicCoordinates (k+1) F j‖)^2))
    (henergy : (∑' j : ℕ,
      (((j:ℝ)+1)^sang * ‖continuousHarmonicCoordinates (k+1) F j‖)^2) ≤ Aang^2) :
    let L := max 1 ((s/2)*Real.log (1+B))
    let γ := B/L^(1/ν)
    let h := 1/(H*B)
    let D := Z*L^(1/μ)/γ
    let J := Finset.Icc (-(⌊(T+D)/h⌋₊:ℤ)) (⌊(T+D)/h⌋₊:ℤ)
    |finiteNetwork q hq (k+1) φ g pHat K r m γ B h (by positivity) J x -
      reconstructed (k+1) g x| ≤
      angularConstant k Aang (sang-((k:ℝ)+1)/2) *
        (q:ℝ)^(-(sang-((k:ℝ)+1)/2)) +
      (A * envelopeConstant E S * (1+B)^(-s)) * T^r / (r.factorial:ℝ) := by
  let L := max 1 ((s/2)*Real.log (1+B))
  let γ := B/L^(1/ν)
  let h := 1/(H*B)
  let D := Z*L^(1/μ)/γ
  let J := Finset.Icc (-(⌊(T+D)/h⌋₊:ℤ)) (⌊(T+D)/h⌋₊:ℤ)
  have hγ : γ ≠ 0 := by dsimp [γ, L]; positivity
  let Q := fun θ => anchoredActivationNetwork φ
    (realReadoutDensity (pHat θ) K γ hγ B) r m γ h J (g θ) (inner ℝ θ.val x)
  let e := (A * envelopeConstant E S * (1+B)^(-s)) * T^r / (r.factorial:ℝ)
  have he : ∀ θ, |Q θ - F θ| ≤ e := by
    intro θ
    rw [hF]
    have hb := algebraic_activation_error hφ K hm hkernel E S (hp θ) hs hB (hW θ)
      hH hsep hZ hhalo hT (hg θ) (hprofile θ) _ (abs_le.mp (projection_bound (k+1) θ hx))
    apply hb.trans
    have hc := (envelopeConstant_pos E S).le
    dsimp [e]
    gcongr
    exact hWbound θ
  have hang := canonical_sobolev_power_error q hq k F heven hsang hAang hsum henergy
  have heq : reconstructed (k+1) g x = ∫ θ, F θ ∂Gaussian.normalizedSphere (k+1) := by
    apply integral_congr_ae
    filter_upwards with θ
    exact (hF θ).symm
  dsimp only
  rw [heq]
  change |(∑ i : SphereHalfIndex q (k+1), foldedSphereWeight q hq (k+1) i *
    Q (sphereNodeOnSphere q hq (k+1) i.val)) - ∫ θ, F θ ∂Gaussian.normalizedSphere (k+1)| ≤ _
  exact normalized_approximation_error (foldedSphereWeight q hq (k+1))
    (fun i => Q (sphereNodeOnSphere q hq (k+1) i.val))
    (fun i => F (sphereNodeOnSphere q hq (k+1) i.val)) e
    (fun i => (foldedSphereWeight_positive q hq (k+1) i).le)
    (foldedSphereWeight_normalized q hq (k+1)) (fun i => he _) hang

theorem power_accuracy_choice {C ε t : ℝ} (hC : 0 ≤ C) (hε : 0 < ε) (ht : 0 < t) :
    C * (⌈algebraicCutoff C t ε⌉₊:ℝ)^(-t) ≤ ε := by
  have hB : 0 < algebraicCutoff C t ε := Real.exp_pos _
  have hpow := Real.rpow_le_rpow_of_nonpos hB (Nat.le_ceil _) (neg_nonpos.mpr ht.le)
  have hid : (algebraicCutoff C t ε)^(-t) = Real.exp (-max 1 (Real.log (C/ε))) := by
    unfold algebraicCutoff
    rw [Real.rpow_def_of_pos (Real.exp_pos _), Real.log_exp]
    congr 1
    field_simp
  exact (mul_le_mul_of_nonneg_left hpow hC).trans (by
    rw [hid]
    exact log_accuracy_choice hC hε)

/-- Explicit epsilon-selected construction and its actual integer width.
The angular order uses the sharp Sobolev loss; the scalar cutoff uses its
independent polynomial Fourier moment. No target values are fitted. -/
theorem algebraic_accuracy_and_width
    (k : ℕ) {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (g : UnitSphere (k+1) → ℝ → ℝ) (pHat : UnitSphere (k+1) → ℝ → ℂ)
    {s H T Z A ε : ℝ} (hs : 0 < s) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b/2)*Z^μ)
    (hT : 0 < T) (hA : 0 ≤ A) (hε : 0 < ε)
    (hp : ∀ θ, Integrable (pHat θ))
    (hW : ∀ θ, Integrable (fun ξ : ℝ => (1+|ξ|)^s * ‖pHat θ ξ‖))
    (hWbound : ∀ θ, (∫ ξ : ℝ, (1+|ξ|)^s * ‖pHat θ ξ‖) ≤ A)
    (hg : ∀ θ, ContDiffOn ℝ r (g θ) (Set.Icc (-T) T))
    (hprofile : ∀ θ y, y ∈ Set.Icc (-T) T →
      iteratedDeriv r (g θ) y = (synthesize (pHat θ) y).re)
    (x : EuclideanSpace ℝ (Fin (k+1+1))) (hx : ‖x‖ ≤ T)
    (F : C(UnitSphere (k+1), ℝ)) (hF : ∀ θ, F θ = g θ (inner ℝ θ.val x))
    (heven : ∀ θ, F (sphereNeg (k+1) θ) = F θ)
    {sang Aang : ℝ} (hsang : ((k:ℝ)+1)/2 < sang) (hAang : 0 ≤ Aang)
    (hsum : Summable (fun j : ℕ =>
      (((j:ℝ)+1)^sang * ‖continuousHarmonicCoordinates (k+1) F j‖)^2))
    (henergy : (∑' j : ℕ,
      (((j:ℝ)+1)^sang * ‖continuousHarmonicCoordinates (k+1) F j‖)^2) ≤ Aang^2) :
    let t := sang-((k:ℝ)+1)/2
    let U := angularConstant k Aang t
    let V := A * envelopeConstant E S * T^r / (r.factorial:ℝ)
    let q := ⌈algebraicCutoff U t (ε/2)⌉₊
    let hq : 0 < q := Nat.ceil_pos.mpr (Real.exp_pos _)
    let B := algebraicCutoff V s (ε/2)
    let L := max 1 ((s/2)*Real.log (1+B))
    let γ := B/L^(1/ν)
    let h := 1/(H*B)
    let D := Z*L^(1/μ)/γ
    let J := Finset.Icc (-(⌊(T+D)/h⌋₊:ℤ)) (⌊(T+D)/h⌋₊:ℤ)
    (|finiteNetwork q hq (k+1) φ g pHat K r m γ B h (by
        dsimp [γ, B, L, algebraicCutoff]; positivity) J x -
      reconstructed (k+1) g x| ≤ ε) ∧
      (Fintype.card (SphereHalfIndex q (k+1) × J):ℝ) ≤
        (q:ℝ)^(k+1) * (2*H*T*B+2*H*Z*L^(1/μ+1/ν)+1) := by
  let t := sang-((k:ℝ)+1)/2
  let U := angularConstant k Aang t
  let V := A * envelopeConstant E S * T^r / (r.factorial:ℝ)
  let q := ⌈algebraicCutoff U t (ε/2)⌉₊
  have hq : 0 < q := Nat.ceil_pos.mpr (Real.exp_pos _)
  let B := algebraicCutoff V s (ε/2)
  let L := max 1 ((s/2)*Real.log (1+B))
  have hB : 0 < B := Real.exp_pos _
  have hU : 0 ≤ U := angularConstant_nonneg k hAang
  have hV : 0 ≤ V := by
    have he := (envelopeConstant_pos E S).le
    dsimp [V]
    positivity
  have ht : 0 < t := sub_pos.mpr hsang
  have hpa := power_accuracy_choice hU (half_pos hε) ht
  have hps := ScalarRate.algebraic_accuracy_choice hV (half_pos hε) hs
  constructor
  · have hb := algebraic_finite_error q hq k hφ K hm hkernel E S g pHat hs.le hB hH
      hsep hZ hhalo hT hp hW hWbound hg hprofile x hx F hF heven hsang hAang hsum henergy
    apply hb.trans
    calc
      _ = U*(q:ℝ)^(-t) + V*(1+B)^(-s) := by dsimp [U, V, t]; ring
      _ ≤ ε/2+ε/2 := add_le_add hpa hps
      _ = ε := by ring
  · rw [actual_grid_width q hq (k+1)]
    push_cast
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    exact grid_width_cycles hT.le hB (by positivity) hZ (by linarith)

end QIRadon.AlgebraicConstruction
