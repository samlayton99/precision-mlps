import QIRadon.Construction.LocalMultidimensional
import QIRadon.Construction.AlgebraicConstruction

/-! Uniform versions of the full-profile construction, with constants and one
finite network fixed before the input x is quantified. CenteredConstruction
provides the separate sharper-scope variant using centered angular norms. -/
noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap
namespace QIRadon.UniformConstruction
open QIRadon.Gaussian QIRadon.Kernel QIRadon.KernelRates
open QIRadon.ProfilePreparation QIRadon.LocalProfileRate QIRadon.LocalMultidimensional
open QIRadon.Multidimensional QIRadon.ScalarRate QIRadon.ScalarScaling QIRadon.AlgebraicConstruction

/-- The actual finite local-profile construction uniformly on the unit ball. -/
theorem local_analytic_uniform_accuracy_and_width
    (n : ℕ)
    (g : UnitSphere n → ℂ → ℂ) {δ M ε : ℝ} (hδ : 0 < δ) (hM : 0 ≤ M) (hε : 0 < ε)
    (hg : ∀ θ, AnalyticOnNhd ℂ (g θ) (bernsteinEllipse δ))
    (hbound : ∀ θ z, z ∈ bernsteinEllipse δ → ‖g θ z‖ ≤ M)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ) (hT : 1 < T₀)
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b/2)*Z^μ)
    (F : EuclideanSpace ℝ (Fin (n+1)) → C(UnitSphere n, ℝ))
    (hF : ∀ x, ‖x‖ ≤ 1 → ∀ θ, F x θ = (g θ (inner ℝ θ.val x : ℝ)).re)
    (heven : ∀ x, ‖x‖ ≤ 1 → ∀ θ, F x (sphereNeg n θ) = F x θ)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : ∀ x, ‖x‖ ≤ 1 → Summable (fun k : ℕ => Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n (F x) k‖^2))
    (henergy : ∀ x, ‖x‖ ≤ 1 → (∑' k : ℕ, Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n (F x) k‖^2) ≤ A^2) :
    let U := A * Real.sqrt (QIRadon.SharpAngular.traceConstant n a)
    let V := localErrorConstant E S M δ T₀ r
    let q := localAccuracyOrder U a (ε/2)
    let N := localAccuracyOrder V (localRate δ T₀) (ε/2)
    (∀ x : EuclideanSpace ℝ (Fin (n+1)), ‖x‖ ≤ 1 →
      |localFiniteNetwork q (localAccuracyOrder_pos _ _ _) n N (localAccuracyOrder_pos _ _ _)
        φ g K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) x -
        reconstructed n (fun θ t => (g θ (t : ℂ)).re) x| ≤ ε) ∧
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) : ℝ) ≤
        (q : ℝ)^n * (((2*H/Real.pi + 2*H*Z + 1) *
          (preparationGrowth T₀ + 3)^max 1 (1/μ+1/ν)) * (N : ℝ)^max 1 (1/μ+1/ν)) := by
  constructor
  · intro x hx
    exact (LocalMultidimensional.local_analytic_accuracy_and_width n g hδ hM hε hg hbound
      hφ K hm hkernel E S hT hH hsep hZ hhalo x hx (F x) (hF x hx) (heven x hx)
      ha hA (hsum x hx) (henergy x hx)).1
  · have hz : ‖(0 : EuclideanSpace ℝ (Fin (n+1)))‖ ≤ 1 := by simp
    exact (LocalMultidimensional.local_analytic_accuracy_and_width n g hδ hM hε hg hbound
      hφ K hm hkernel E S hT hH hsep hZ hhalo 0 hz (F 0) (hF 0 hz) (heven 0 hz)
      ha hA (hsum 0 hz) (henergy 0 hz)).2

/-- The actual finite polynomial-Fourier/Sobolev construction uniformly on its ball. -/
theorem algebraic_uniform_accuracy_and_width
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
    (F : EuclideanSpace ℝ (Fin (k+1+1)) → C(UnitSphere (k+1), ℝ))
    (hF : ∀ x, ‖x‖ ≤ T → ∀ θ, F x θ = g θ (inner ℝ θ.val x))
    (heven : ∀ x, ‖x‖ ≤ T → ∀ θ, F x (sphereNeg (k+1) θ) = F x θ)
    {sang Aang : ℝ} (hsang : ((k:ℝ)+1)/2 < sang) (hAang : 0 ≤ Aang)
    (hsum : ∀ x, ‖x‖ ≤ T → Summable (fun j : ℕ =>
      (((j:ℝ)+1)^sang * ‖continuousHarmonicCoordinates (k+1) (F x) j‖)^2))
    (henergy : ∀ x, ‖x‖ ≤ T → (∑' j : ℕ,
      (((j:ℝ)+1)^sang * ‖continuousHarmonicCoordinates (k+1) (F x) j‖)^2) ≤ Aang^2) :
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
    (∀ x : EuclideanSpace ℝ (Fin (k+1+1)), ‖x‖ ≤ T →
      |finiteNetwork q hq (k+1) φ g pHat K r m γ B h (by
        dsimp [γ, B, L, algebraicCutoff]; positivity) J x -
      reconstructed (k+1) g x| ≤ ε) ∧
      (Fintype.card (SphereHalfIndex q (k+1) × J):ℝ) ≤
        (q:ℝ)^(k+1) * (2*H*T*B+2*H*Z*L^(1/μ+1/ν)+1) := by
  constructor
  · intro x hx
    exact (AlgebraicConstruction.algebraic_accuracy_and_width k hφ K hm hkernel E S g pHat
      hs hH hsep hZ hhalo hT hA hε hp hW hWbound hg hprofile x hx (F x) (hF x hx)
      (heven x hx) hsang hAang (hsum x hx) (henergy x hx)).1
  · have hz : ‖(0 : EuclideanSpace ℝ (Fin (k+1+1)))‖ ≤ T := by simpa using hT.le
    exact (AlgebraicConstruction.algebraic_accuracy_and_width k hφ K hm hkernel E S g pHat
      hs hH hsep hZ hhalo hT hA hε hp hW hWbound hg hprofile 0 hz (F 0) (hF 0 hz)
      (heven 0 hz) hsang hAang (hsum 0 hz) (henergy 0 hz)).2

end QIRadon.UniformConstruction
