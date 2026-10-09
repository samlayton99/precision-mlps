import QIRadon.Scalar.ScalarRate
import QIRadon.Scalar.ScalarScaling
import QIRadon.Architecture.Basic
import QIRadon.Quadrature.HomogeneousQuadrature
import QIRadon.Radon.Reconstruction

/-!
# Actual finite multidimensional construction from certified regularity classes

The angular assumptions below are an actual homogeneous-polynomial L2 expansion
and its exponential weighted energy. Scalar assumptions concern actual Fourier
moments. No quadrature or network approximation bound is a hypothesis.

The local-target theorem must still derive these regularity properties from
the prescribed radial extension. These statements do not replace that task.
-/

noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap

namespace QIRadon.Multidimensional

open QIRadon.Kernel QIRadon.KernelRates QIRadon.ScalarRate QIRadon.Gaussian

/-- The exact continuous reconstruction associated with the supplied ridge family. -/
def reconstructed (n : ℕ) (g : UnitSphere n → ℝ → ℝ)
    (x : EuclideanSpace ℝ (Fin (n + 1))) : ℝ :=
  ∫ θ, g θ (inner ℝ θ.val x) ∂Gaussian.normalizedSphere n

/-- For the canonical Radon profiles, the represented object is exactly the
original target, by the proved Fourier/Radon reconstruction theorem. -/
theorem reconstructed_canonical (n : ℕ)
    (f : 𝓢(EuclideanSpace ℝ (Fin (n+1)), ℝ))
    (x : EuclideanSpace ℝ (Fin (n+1))) :
    reconstructed n (Radon.realRidgeProfile f) x = f x := by
  exact (Radon.normalized_real_schwartz_reconstruction f x).symm

/-- The actual finite network: computed readout densities at actual grid centers,
actual Gauss–Jacobi directions, and explicit integration jets. -/
def finiteNetwork (q : ℕ) (hq : 0 < q) (n : ℕ)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℝ → ℝ)
    (pHat : UnitSphere n → ℝ → ℂ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m γ B h : ℝ) (hγ : γ ≠ 0) (J : Finset ℤ)
    (x : EuclideanSpace ℝ (Fin (n + 1))) : ℝ :=
  ∑ i : SphereHalfIndex q n,
    let θ := sphereNodeOnSphere q hq n i.val
    foldedSphereWeight q hq n i *
      anchoredActivationNetwork φ (realReadoutDensity (pHat θ) K γ hγ B)
        r m γ h J (g θ) (inner ℝ θ.val x)

/-- Unit directions do not enlarge the target interval of a spoke. -/
theorem projection_bound (n : ℕ) (θ : UnitSphere n)
    {x : EuclideanSpace ℝ (Fin (n + 1))} {T : ℝ} (hx : ‖x‖ ≤ T) :
    |inner ℝ θ.val x| ≤ T := by
  have ht : ‖θ.val‖ = 1 := by simpa only [mem_sphere_zero_iff_norm] using θ.property
  have hi := norm_inner_le_norm (𝕜 := ℝ) θ.val x
  simpa only [Real.norm_eq_abs, ht, one_mul] using hi.trans (by simpa [ht] using hx)

/-- Analytic multidimensional error for the actual finite construction.
The two exponential errors are derived from their respective regularity norms.
The first is angular; the second includes scalar cutoff, aliases and both halos. -/
theorem analytic_finite_error
    (q : ℕ) (hq : 0 < q) (n : ℕ) {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (g : UnitSphere n → ℝ → ℝ) (pHat : UnitSphere n → ℝ → ℂ)
    {σ L H T Z A : ℝ} (hσ : 0 < σ) (hL : 1 ≤ L) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hT : 0 < T) (_hA : 0 ≤ A)
    (hp : ∀ θ, Integrable (pHat θ))
    (hW : ∀ θ, Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) * ‖pHat θ ξ‖))
    (hWbound : ∀ θ, (∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat θ ξ‖) ≤ A)
    (hg : ∀ θ, ContDiffOn ℝ r (g θ) (Set.Icc (-T) T))
    (hprofile : ∀ θ s, s ∈ Set.Icc (-T) T →
      iteratedDeriv r (g θ) s = (synthesize (pHat θ) s).re)
    (x : EuclideanSpace ℝ (Fin (n + 1))) (hx : ‖x‖ ≤ T)
    (v : (k : ℕ) → PolynomialL2Space n k)
    (hF : Integrable (fun θ => g θ (inner ℝ θ.val x)) (Gaussian.normalizedSphere n))
    (heven : ∀ θ, g (sphereNeg n θ) (inner ℝ (sphereNeg n θ).val x) = g θ (inner ℝ θ.val x))
    (hrep : ∀ θ, g θ (inner ℝ θ.val x) = ∑' k, polynomialEvaluation n k θ (v k))
    {a Aang : ℝ} (ha : 0 < a) (hAang : 0 ≤ Aang)
    (hsum : Summable (fun k : ℕ => Real.exp (2 * a * k) * ‖v k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * ‖v k‖^2) ≤ Aang^2) :
    let B := L / σ
    let γ := B / L ^ (1 / ν)
    let h := 1 / (H * B)
    let D := Z * L ^ (1 / μ) / γ
    let J := Finset.Icc (-(⌊(T + D) / h⌋₊ : ℤ)) (⌊(T + D) / h⌋₊ : ℤ)
    |finiteNetwork q hq n φ g pHat K r m γ B h (by positivity) J x - reconstructed n g x| ≤
      2 * ((Aang * (((n + 1).factorial : ℝ) / (a / 2)^(n + 1) * Real.exp (a / 2))) *
        Real.exp (-a * q) / (1 - Real.exp (-(a / 2)))) +
      (A * envelopeConstant E S * Real.exp (-L)) * T^r / (r.factorial : ℝ) := by
  let B := L / σ
  let γ := B / L ^ (1 / ν)
  let h := 1 / (H * B)
  let D := Z * L ^ (1 / μ) / γ
  let J := Finset.Icc (-(⌊(T + D) / h⌋₊ : ℤ)) (⌊(T + D) / h⌋₊ : ℤ)
  have hγ : γ ≠ 0 := by dsimp [γ, B]; positivity
  let Q := fun θ => anchoredActivationNetwork φ
    (realReadoutDensity (pHat θ) K γ hγ B) r m γ h J (g θ) (inner ℝ θ.val x)
  let F := fun θ : UnitSphere n => g θ (inner ℝ θ.val x)
  let e := (A * envelopeConstant E S * Real.exp (-L)) * T^r / (r.factorial : ℝ)
  have he : ∀ θ, |Q θ - F θ| ≤ e := by
    intro θ
    have hs := abs_le.mp (projection_bound n θ hx)
    have hb := analytic_activation_error hφ K hm hkernel E S (hp θ) hσ (hW θ)
      hL hH hsep hZ hhalo hT (hg θ) (hprofile θ) (inner ℝ θ.val x) hs
    apply hb.trans
    have hEc := (envelopeConstant_pos E S).le
    dsimp [e]
    gcongr
    exact hWbound θ
  have hang := foldedSphere_analytic_series_error q hq n v F hF heven hrep a Aang ha hAang hsum henergy
  dsimp only
  change |(∑ i : SphereHalfIndex q n, foldedSphereWeight q hq n i * Q (sphereNodeOnSphere q hq n i.val)) -
      ∫ θ, F θ ∂Gaussian.normalizedSphere n| ≤ _
  exact normalized_approximation_error (foldedSphereWeight q hq n)
    (fun i => Q (sphereNodeOnSphere q hq n i.val))
    (fun i => F (sphereNodeOnSphere q hq n i.val)) e
    (fun i => (foldedSphereWeight_positive q hq n i).le)
    (foldedSphereWeight_normalized q hq n) (fun i => he _) hang

/-- Actual tensor index count, with every center in every retained direction. -/
theorem actual_grid_width (q : ℕ) (hq : 0 < q) (n : ℕ) (J : Finset ℤ) :
    Fintype.card (SphereHalfIndex q n × J) = q^n * J.card := by
  simp only [Fintype.card_prod, sphereHalfIndex_card q hq n, Fintype.card_coe]

end QIRadon.Multidimensional
