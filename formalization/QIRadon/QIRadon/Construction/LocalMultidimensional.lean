import QIRadon.Scalar.LocalProfileRate
import QIRadon.Construction.CanonicalConstruction
import QIRadon.Quadrature.SharpAngular

/-! Finite multidimensional construction from locally analytic spoke profiles.
Unlike the global-Fourier corollary, no global analytic extension or exponential
Fourier moment is assumed of the original profiles. Every row is prepared from
its actual finite Chebyshev samples. Target-side angular regularity is the
canonical weighted harmonic norm of the actual angular function. -/

noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap
namespace QIRadon.LocalMultidimensional
open QIRadon.Gaussian QIRadon.Kernel QIRadon.KernelRates QIRadon.LocalProfileRate
open QIRadon.ProfilePreparation QIRadon.Multidimensional

/-- Actual finite row sum; N counts preparation samples minus one, not neurons. -/
def localFiniteNetwork (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀)
    (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  ∑ i : SphereHalfIndex q n,
    let θ := sphereNodeOnSphere q hq n i.val
    foldedSphereWeight q hq n i *
      localActivationNetwork φ (fun t => (g θ (t : ℂ)).re) K r N m T₀ H Z ν μ hN hT
        (inner ℝ θ.val x)

/-- Derived angular + scalar error of the actual locally prepared construction. -/
theorem local_analytic_finite_error
    (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (g : UnitSphere n → ℂ → ℂ) {δ M : ℝ} (hδ : 0 < δ)
    (hg : ∀ θ, AnalyticOnNhd ℂ (g θ) (bernsteinEllipse δ))
    (hbound : ∀ θ z, z ∈ bernsteinEllipse δ → ‖g θ z‖ ≤ M)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ) (hT : 1 < T₀)
    (hH : 2 ≤ H) (hsep : E.A + 4 ≤ E.a * (H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b/2)*Z^μ)
    (x : EuclideanSpace ℝ (Fin (n+1))) (hx : ‖x‖ ≤ 1)
    (F : C(UnitSphere n, ℝ)) (hF : ∀ θ, F θ = (g θ (inner ℝ θ.val x : ℝ)).re)
    (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ => Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2) ≤ A^2) :
    |localFiniteNetwork q hq n N hN φ g K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) x -
      reconstructed n (fun θ t => (g θ (t : ℂ)).re) x| ≤
      A * Real.sqrt (QIRadon.SharpAngular.traceConstant n a) * Real.exp (-a*q) +
      localErrorConstant E S M δ T₀ r * Real.exp (-localRate δ T₀ * N) := by
  let Q := fun θ => localActivationNetwork φ (fun t => (g θ (t : ℂ)).re) K r N m T₀ H Z ν μ
    hN (lt_trans zero_lt_one hT) (inner ℝ θ.val x)
  let e := localErrorConstant E S M δ T₀ r * Real.exp (-localRate δ T₀ * N)
  have hs : ∀ θ, |Q θ - F θ| ≤ e := by
    intro θ
    rw [hF]
    exact local_analytic_activation_error hδ (hg θ) (hbound θ) hφ K hm hkernel E S hT hN
      hH hsep hZ hhalo _ (abs_le.mp (projection_bound n θ hx))
  have hang := QIRadon.SharpAngular.canonical_quadrature_error q hq n F heven ha hA hsum henergy
  have heq : reconstructed n (fun θ t => (g θ (t : ℂ)).re) x =
      ∫ θ, F θ ∂Gaussian.normalizedSphere n := by
    apply integral_congr_ae
    filter_upwards with θ
    exact (hF θ).symm
  rw [heq]
  change |(∑ i : SphereHalfIndex q n, foldedSphereWeight q hq n i * Q (sphereNodeOnSphere q hq n i.val)) -
    ∫ θ, F θ ∂Gaussian.normalizedSphere n| ≤ _
  exact normalized_approximation_error (foldedSphereWeight q hq n)
    (fun i => Q (sphereNodeOnSphere q hq n i.val))
    (fun i => F (sphereNodeOnSphere q hq n i.val)) e
    (fun i => (foldedSphereWeight_positive q hq n i).le)
    (foldedSphereWeight_normalized q hq n) (fun i => hs _) hang

/-- Exact direction count times the proved local preparation row width. -/
theorem local_total_grid_width (q : ℕ) (hq : 0 < q) (n : ℕ) {N : ℕ}
    {T₀ H Z ν μ : ℝ} (hT : 0 < T₀) (hN : 0 < N) (hH : 0 < H) (hZ : 0 ≤ Z) :
    (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) : ℝ) ≤
      (q : ℝ)^n *
        (((2*H/Real.pi + 2*H*Z + 1) *
          (preparationGrowth T₀ + 3)^max 1 (1/μ+1/ν)) * (N : ℝ)^max 1 (1/μ+1/ν)) := by
  rw [actual_grid_width q hq n]
  push_cast
  exact mul_le_mul_of_nonneg_left (local_analytic_grid_width hT hN hH hZ) (by positivity)

/-- Full actual local-profile accuracy and finite width for a requested epsilon. -/
theorem local_analytic_accuracy_and_width
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
    (x : EuclideanSpace ℝ (Fin (n+1))) (hx : ‖x‖ ≤ 1)
    (F : C(UnitSphere n, ℝ)) (hF : ∀ θ, F θ = (g θ (inner ℝ θ.val x : ℝ)).re)
    (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ => Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2) ≤ A^2) :
    let U := A * Real.sqrt (QIRadon.SharpAngular.traceConstant n a)
    let V := localErrorConstant E S M δ T₀ r
    let q := localAccuracyOrder U a (ε/2)
    let N := localAccuracyOrder V (localRate δ T₀) (ε/2)
    (|localFiniteNetwork q (localAccuracyOrder_pos _ _ _) n N (localAccuracyOrder_pos _ _ _)
        φ g K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) x -
        reconstructed n (fun θ t => (g θ (t : ℂ)).re) x| ≤ ε) ∧
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) : ℝ) ≤
        (q : ℝ)^n * (((2*H/Real.pi + 2*H*Z + 1) *
          (preparationGrowth T₀ + 3)^max 1 (1/μ+1/ν)) * (N : ℝ)^max 1 (1/μ+1/ν)) := by
  let U := A * Real.sqrt (QIRadon.SharpAngular.traceConstant n a)
  let V := localErrorConstant E S M δ T₀ r
  let q := localAccuracyOrder U a (ε/2)
  let N := localAccuracyOrder V (localRate δ T₀) (ε/2)
  have hq : 0 < q := localAccuracyOrder_pos _ _ _
  have hN : 0 < N := localAccuracyOrder_pos _ _ _
  have hU : 0 ≤ U := by
    dsimp [U]
    positivity
  have hV : 0 ≤ V := localErrorConstant_nonneg E S hM hδ hT r
  constructor
  · have hb := local_analytic_finite_error q hq n N hN g hδ hg hbound hφ K hm hkernel E S
      hT hH hsep hZ hhalo x hx F hF heven ha hA hsum henergy
    apply hb.trans
    have hang := localAccuracyOrder_error hU ha (half_pos hε)
    have hscalar := localAccuracyOrder_error hV (localRate_pos hδ hT) (half_pos hε)
    calc
      _ = U*Real.exp (-a*q) + V*Real.exp (-localRate δ T₀*N) := by rfl
      _ ≤ ε/2 + ε/2 := add_le_add hang hscalar
      _ = ε := by ring
  · exact local_total_grid_width q hq n (lt_trans zero_lt_one hT) hN (by linarith) hZ

end QIRadon.LocalMultidimensional
