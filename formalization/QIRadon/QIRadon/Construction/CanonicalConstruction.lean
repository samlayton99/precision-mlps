import QIRadon.Construction.MultidimensionalScaling
import QIRadon.Quadrature.CanonicalConvergence

/-! Actual construction theorems using the canonical angular norm of the target.
No externally supplied expansion, polynomial approximation, or fitted readout
is an assumption. `KnownFunctionConstruction.lean` supplies the raw-target
regularity bridge for the local analytic construction. -/

noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap
namespace QIRadon.CanonicalConstruction
open QIRadon.Gaussian QIRadon.Kernel QIRadon.KernelRates QIRadon.ScalarRate
open QIRadon.Multidimensional QIRadon.MultidimensionalScaling

theorem canonical_analytic_accuracy_and_width
    (n : ℕ) {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (g : UnitSphere n → ℝ → ℝ) (pHat : UnitSphere n → ℝ → ℂ)
    {σ H T Z A ε : ℝ} (hσ : 0 < σ) (hε : 0 < ε) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hT : 0 < T) (hA : 0 ≤ A)
    (hp : ∀ θ, Integrable (pHat θ))
    (hW : ∀ θ, Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) * ‖pHat θ ξ‖))
    (hWbound : ∀ θ, (∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat θ ξ‖) ≤ A)
    (hg : ∀ θ, ContDiffOn ℝ r (g θ) (Set.Icc (-T) T))
    (hprofile : ∀ θ s, s ∈ Set.Icc (-T) T →
      iteratedDeriv r (g θ) s = (synthesize (pHat θ) s).re)
    (x : EuclideanSpace ℝ (Fin (n + 1))) (hx : ‖x‖ ≤ T)
    (F : C(UnitSphere n, ℝ))
    (hF : ∀ θ, F θ = g θ (inner ℝ θ.val x))
    (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    {a Aang : ℝ} (ha : 0 < a) (hAang : 0 ≤ Aang)
    (hsum : Summable (fun k : ℕ => Real.exp (2 * a * k) * ‖continuousHarmonicCoordinates n F k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * ‖continuousHarmonicCoordinates n F k‖^2) ≤ Aang^2) :
    let U := 2 * (Aang * (((n+1).factorial : ℝ) / (a/2)^(n+1) * Real.exp (a/2))) /
      (1 - Real.exp (-(a/2)))
    let V := A * envelopeConstant E S * T^r / (r.factorial : ℝ)
    let L := max 1 (Real.log ((U+V)/ε))
    let q := angularOrder L a
    let B := L / σ
    let γ := B / L^(1/ν)
    let h := 1/(H*B)
    let D := Z*L^(1/μ)/γ
    let J := Finset.Icc (-(⌊(T+D)/h⌋₊ : ℤ)) (⌊(T+D)/h⌋₊ : ℤ)
    (|finiteNetwork q (angularOrder_bounds (le_max_left _ _) ha).1 n φ g pHat K r m γ B h
      (by positivity) J x - reconstructed n g x| ≤ ε) ∧
    (Fintype.card (SphereHalfIndex q n × J) : ℝ) ≤
      ((1/a+1)^n * (2*H*T/σ + 2*H*Z + 1)) * L^((n:ℝ)+max 1 (1/μ+1/ν)) := by
  let v := fun k => (continuousHarmonicCoordinates n F k).val
  have hEq : (fun θ => g θ (inner ℝ θ.val x)) = F := funext (fun θ => (hF θ).symm)
  have hi : Integrable (fun θ => g θ (inner ℝ θ.val x)) (Gaussian.normalizedSphere n) := by
    rw [hEq]
    exact F.continuous.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have he : ∀ θ, g (sphereNeg n θ) (inner ℝ (sphereNeg n θ).val x) = g θ (inner ℝ θ.val x) := by
    intro θ
    rw [← hF, ← hF]
    exact heven θ
  have hrep : ∀ θ, g θ (inner ℝ θ.val x) = ∑' k, polynomialEvaluation n k θ (v k) := by
    intro θ
    rw [← hF]
    exact canonical_harmonic_pointwise_analytic n F a Aang ha hAang hsum henergy θ
  exact analytic_finite_accuracy_and_width n hφ K hm hkernel E S g pHat hσ hε hH
    hsep hZ hhalo hT hA hp hW hWbound hg hprofile x hx v hi he hrep ha hAang hsum henergy

end QIRadon.CanonicalConstruction
