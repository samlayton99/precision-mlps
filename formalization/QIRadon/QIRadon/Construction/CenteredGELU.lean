import QIRadon.Construction.CenteredConstruction

/-! Uniform ordinary-GELU theorem for the actual canonical Radon target.
This composes local profile regularity and centered canonical angular norms;
it does not claim to derive those norms from bare target analyticity. -/
noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap Topology
namespace QIRadon.CenteredGELU
open QIRadon.Gaussian QIRadon.Activations QIRadon.LocalProfileRate
open QIRadon.ProfilePreparation QIRadon.CenteredConstruction

/-- One fixed standard-GELU MLP approximates the actual target uniformly on the
unit ball, hits f(0) exactly, and includes every halo and its two affine-anchor
neurons in the width. Its coefficients depend only on the prescribed profile
construction, never on the evaluation input x. -/
theorem canonical_uniform_accuracy_and_width
    (n : ℕ) (f : 𝓢(EuclideanSpace ℝ (Fin (n+1)), ℝ))
    (g : UnitSphere n → ℂ → ℂ) {δ M ε T₀ H Z : ℝ}
    (hδ : 0 < δ) (hM : 0 ≤ M) (hε : 0 < ε)
    (hg : ∀ θ, AnalyticOnNhd ℂ (g θ) (bernsteinEllipse δ))
    (hbound : ∀ θ z, z ∈ bernsteinEllipse δ → ‖g θ z‖ ≤ M)
    (hcanonical : ∀ θ (s : ℝ), |s| ≤ 1 →
      (g θ (s:ℂ)).re = Radon.realRidgeProfile f θ s)
    (hT : 1 < T₀) (hH : 2 ≤ H)
    (hsep : geluSpectral.A+4 ≤ geluSpectral.a*(H/2)^(2:ℝ))
    (hZ : 0 ≤ Z) (hhalo : geluSpectral.A+3 ≤ (geluSpatial.b/2)*Z^(2:ℝ))
    (F : EuclideanSpace ℝ (Fin (n+1)) → C(UnitSphere n, ℝ))
    (hF : ∀ x, ‖x‖ ≤ 1 → ∀ θ,
      F x θ = (g θ (inner ℝ θ.val x : ℝ)).re - (g θ 0).re)
    (heven : ∀ x, ‖x‖ ≤ 1 → ∀ θ, F x (sphereNeg n θ) = F x θ)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : ∀ x, ‖x‖ ≤ 1 → Summable (fun k : ℕ =>
      Real.exp (2*a*k)*‖continuousHarmonicCoordinates n (F x) k‖^2))
    (henergy : ∀ x, ‖x‖ ≤ 1 → (∑' k : ℕ,
      Real.exp (2*a*k)*‖continuousHarmonicCoordinates n (F x) k‖^2) ≤ A^2) :
    let U := A*Real.sqrt (SharpAngular.traceConstant n a)
    let V := localErrorConstant geluSpectral geluSpatial M δ T₀ 2
    let q := localAccuracyOrder U a (ε/2)
    let N := localAccuracyOrder V (localRate δ T₀) (ε/4)
    ∃ (w : (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) → Fin (n+1) → ℝ)
      (b v : (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) → ℝ) (bias : ℝ),
      (∀ x : EuclideanSpace ℝ (Fin (n+1)), ‖x‖ ≤ 1 →
        |Architecture.network gelu w b v bias x - f x| ≤ ε) ∧
      Architecture.network gelu w b v bias (0 : Fin (n+1) → ℝ) = f 0 ∧
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) : ℝ) ≤
        (q:ℝ)^n * (((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)) * N) + 2 := by
  let U := A*Real.sqrt (SharpAngular.traceConstant n a)
  let V := localErrorConstant geluSpectral geluSpatial M δ T₀ 2
  let q := localAccuracyOrder U a (ε/2)
  let N := localAccuracyOrder V (localRate δ T₀) (ε/4)
  have hq : 0 < q := localAccuracyOrder_pos _ _ _
  have hN : 0 < N := localAccuracyOrder_pos _ _ _
  obtain ⟨w,b,v,bias,heq⟩ := centered_gelu_is_network (f 0) q hq n N hN
    g geluKernel 1 T₀ H Z 2 2 (lt_trans zero_lt_one hT)
  have hmain := centered_analytic_uniform_accuracy_and_width n (f 0) g hδ hM hε hg hbound
    (gelu_contDiff 2) geluKernel one_ne_zero
    (by intro t; simpa using gelu_second_derivative t) geluSpectral geluSpatial
    hT hH hsep hZ hhalo F hF heven ha hA hsum henergy
  refine ⟨w,b,v,bias,?_,?_,?_⟩
  · intro x hx
    rw [heq, ← centeredReconstructed_canonical_on_ball n f g hcanonical x hx]
    exact hmain.1 x hx
  · have hh := heq (0 : EuclideanSpace ℝ (Fin (n+1)))
    rw [centeredLocalNetwork_zero] at hh
    exact hh
  · have hw := hmain.2
    have hk : max (1:ℝ) (1/2+1/2) = 1 := by norm_num
    rw [hk, Real.rpow_one, Real.rpow_one] at hw
    simp only [Fintype.card_sum, Fintype.card_fin, Nat.cast_add, Nat.cast_ofNat]
    exact add_le_add hw le_rfl

/-- Centering and exact affine-anchor realization preserve the promised
GELU width exponent d=n+1. The two integer orders are those of the theorem above. -/
theorem canonical_total_epsilon_bigO (n : ℕ) {T₀ H Z U V a b : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 < a) (hb : 0 < b) :
    (fun ε : ℝ =>
      let q := localAccuracyOrder U a (ε/2)
      let N := localAccuracyOrder V b (ε/4)
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) : ℝ))
      =O[𝓝[>] (0:ℝ)] (fun ε : ℝ => |Real.log ε|^((n:ℝ)+1)) := by
  have hh := centered_total_grid_epsilon_bigO (ν := 2) (μ := 2) n 2 hT hH hZ hU hV ha hb
  norm_num only [one_div, show (2:ℝ)⁻¹+(2:ℝ)⁻¹=1 by norm_num, max_self] at hh
  exact hh

end QIRadon.CenteredGELU
