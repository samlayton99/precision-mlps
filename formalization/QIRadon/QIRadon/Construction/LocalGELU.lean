import QIRadon.Construction.LocalMultidimensional
import QIRadon.Construction.LocalScaling
import QIRadon.Activations.Anchors

/-! Exact ordinary-MLP realization of the locally prepared GELU construction.
Every formal order-two integration anchor is combined into a single affine
function, represented by two additional GELU units for the whole network. -/
noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap Topology
namespace QIRadon.LocalGELU
open QIRadon.Gaussian QIRadon.Kernel QIRadon.LocalProfileRate
open QIRadon.ProfilePreparation QIRadon.LocalMultidimensional QIRadon.Activations

/-- The raw per-spoke sums are exactly one finite product-indexed MLP. -/
theorem raw_spokes_network {d : ℕ} {ι : Type*} [Fintype ι]
    (J : Finset ℤ) (θ : ι → Fin d → ℝ) (α : ι → ℝ) (ρ : ι → ℝ → ℝ)
    (m γ h : ℝ) (x : Fin d → ℝ) :
    Architecture.network gelu
      (fun p : ι × J => fun k => γ * θ p.1 k)
      (fun p => -γ * (h * (p.2.val : ℝ)))
      (fun p => α p.1 * (h*γ/(m*γ^2)) * ρ p.1 (h*p.2.val)) 0 x =
    ∑ i, α i * rawActivationNetwork gelu (ρ i) 2 m γ h J (∑ k, θ i k*x k) := by
  simp only [Architecture.network, zero_add, Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro i _
  have harg : ∀ j : ℤ, (∑ k, γ*θ i k*x k) + -γ*(h*j) =
      γ*((∑ k, θ i k*x k)-h*j) := by
    intro j
    simp only [mul_assoc, ← Finset.mul_sum]
    ring
  simp_rw [harg]
  rw [Finset.sum_coe_sort J (fun j : ℤ =>
    α i*(h*γ/(m*γ^2))*ρ i (h*j)*gelu (γ*((∑ k, θ i k*x k)-h*j)))]
  unfold rawActivationNetwork
  simp only [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- All anchored spokes, for arbitrary data and coefficients, are exactly one
GELU MLP with grid-count + 2 hidden units. This identity requires no fitting. -/
theorem anchored_spokes_is_network {d : ℕ} {ι : Type*} [Fintype ι]
    (J : Finset ℤ) (θ : ι → Fin d → ℝ) (α : ι → ℝ) (ρ g : ι → ℝ → ℝ)
    (m γ h : ℝ) :
    ∃ (a : (ι × J ⊕ Fin 2) → Fin d → ℝ) (b v : (ι × J ⊕ Fin 2) → ℝ) (bias : ℝ),
      ∀ x : Fin d → ℝ,
        Architecture.network gelu a b v bias x =
        ∑ i, α i * anchoredActivationNetwork gelu (ρ i) 2 m γ h J (g i)
          (∑ k, θ i k*x k) := by
  let S := fun i => rawActivationNetwork gelu (ρ i) 2 m γ h J
  let b₀ := fun i => g i 0 - S i 0
  let a₁ := fun i => deriv (g i) 0 - deriv (S i) 0
  let A := fun k => ∑ i, α i*a₁ i*θ i k
  let ar := fun p : ι × J => fun k => γ*θ p.1 k
  let br := fun p : ι × J => -γ*(h*(p.2.val : ℝ))
  let vr := fun p : ι × J => α p.1*(h*γ/(m*γ^2))*ρ p.1 (h*p.2.val)
  let aa := fun j : Fin 2 => if j = 0 then A else -A
  let ba := fun _ : Fin 2 => (0 : ℝ)
  let va := fun j : Fin 2 => if j = 0 then (1 : ℝ) else -1
  refine ⟨Sum.elim ar aa, Sum.elim br ba, Sum.elim vr va, 0 + ∑ i, α i*b₀ i, ?_⟩
  intro x
  rw [← Architecture.network_sum]
  have hr := raw_spokes_network J θ α ρ m γ h x
  have ha := directional_affine_anchors α b₀ a₁ θ x
  change Architecture.network gelu ar br vr 0 x +
    Architecture.network gelu aa ba va (∑ i, α i*b₀ i) x = _
  rw [hr, ← ha, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  simp only [anchoredActivationNetwork, anchorPolynomial_two_affine,
    iteratedDeriv_zero, iteratedDeriv_one]
  dsimp [b₀, a₁, S]
  ring

/-- Exact GELU realization of the actual finite local Radon construction,
including the Chebyshev preparation and Fourier-defined readout densities. -/
theorem local_gelu_is_network
    (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    ∃ (a : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ ⊕ Fin 2) → Fin (n+1) → ℝ)
      (b v : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ ⊕ Fin 2) → ℝ) (bias : ℝ),
      ∀ x : EuclideanSpace ℝ (Fin (n+1)),
        Architecture.network gelu a b v bias x =
          localFiniteNetwork q hq n N hN gelu g K 2 m T₀ H Z ν μ hT x := by
  let L := localSuppression N T₀
  let B := L / Real.pi
  let γ := B / L^(1/ν)
  let h := 1/(H*B)
  have hγ : γ ≠ 0 := by
    have hL := localSuppression_ge_one hN hT
    have hLp : 0 < L := by dsimp [L]; linarith
    dsimp [γ, B]
    positivity
  let θ := fun i : SphereHalfIndex q n => (sphereNodeOnSphere q hq n i.val).val
  let f := fun i : SphereHalfIndex q n => fun t : ℝ => (g (sphereNodeOnSphere q hq n i.val) (t:ℂ)).re
  let ρ := fun i => realReadoutDensity (derivativeSpectrum N (f i) T₀ 2) K γ hγ B
  obtain ⟨a,b,v,bias,heq⟩ := anchored_spokes_is_network
    (localGrid N T₀ H Z ν μ) (fun i k => θ i k) (foldedSphereWeight q hq n) ρ
    (fun i => realPrepared N (f i) T₀) m γ h
  refine ⟨a,b,v,bias,?_⟩
  intro x
  rw [heq]
  unfold localFiniteNetwork
  apply Finset.sum_congr rfl
  intro i _
  congr 1
  have hd : (∑ k, θ i k*x k) = inner ℝ (θ i) x := by
    simp [PiLp.inner_apply, RCLike.inner_apply, mul_comm]
  rw [hd]
  rfl

/-- The actual GELU architecture adds exactly two units to the complete
halo-inclusive grid count, independently of the number of directions. -/
theorem local_gelu_width (q n N : ℕ) (T₀ H Z ν μ : ℝ) :
    Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ ⊕ Fin 2) =
      Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) + 2 := by simp

/-- Fully finite accuracy theorem for an ordinary, one-hidden-layer standard
GELU MLP. The same parameters reproduce the construction at every input. The
angular regularity hypotheses here remain hypotheses about the actual canonical
harmonic coefficients of the target angular function; their source theorem is
independent of the architectural identity. -/
theorem local_gelu_accuracy_and_width
    (n : ℕ) (g : UnitSphere n → ℂ → ℂ) {δ M ε T₀ H Z : ℝ}
    (hδ : 0 < δ) (hM : 0 ≤ M) (hε : 0 < ε)
    (hg : ∀ θ, AnalyticOnNhd ℂ (g θ) (bernsteinEllipse δ))
    (hbound : ∀ θ z, z ∈ bernsteinEllipse δ → ‖g θ z‖ ≤ M)
    (hT : 1 < T₀) (hH : 2 ≤ H)
    (hsep : geluSpectral.A + 4 ≤ geluSpectral.a * (H/2)^(2:ℝ))
    (hZ : 0 ≤ Z) (hhalo : geluSpectral.A + 3 ≤ (geluSpatial.b/2)*Z^(2:ℝ))
    (x : EuclideanSpace ℝ (Fin (n+1))) (hx : ‖x‖ ≤ 1)
    (F : C(UnitSphere n, ℝ)) (hF : ∀ θ, F θ = (g θ (inner ℝ θ.val x : ℝ)).re)
    (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ => Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2) ≤ A^2) :
    let U := A * Real.sqrt (QIRadon.SharpAngular.traceConstant n a)
    let V := localErrorConstant geluSpectral geluSpatial M δ T₀ 2
    let q := localAccuracyOrder U a (ε/2)
    let N := localAccuracyOrder V (localRate δ T₀) (ε/2)
    ∃ (w : (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) → Fin (n+1) → ℝ)
      (b v : (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) → ℝ) (bias : ℝ),
      (∀ y : EuclideanSpace ℝ (Fin (n+1)), Architecture.network gelu w b v bias y =
        localFiniteNetwork q (localAccuracyOrder_pos _ _ _) n N (localAccuracyOrder_pos _ _ _)
          gelu g geluKernel 2 1 T₀ H Z 2 2 (lt_trans zero_lt_one hT) y) ∧
      |Architecture.network gelu w b v bias x -
        Multidimensional.reconstructed n (fun θ t => (g θ (t : ℂ)).re) x| ≤ ε ∧
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) : ℝ) ≤
        (q : ℝ)^n * (((2*H/Real.pi + 2*H*Z + 1) * (preparationGrowth T₀ + 3)) * N) + 2 := by
  let U := A * Real.sqrt (QIRadon.SharpAngular.traceConstant n a)
  let V := localErrorConstant geluSpectral geluSpatial M δ T₀ 2
  let q := localAccuracyOrder U a (ε/2)
  let N := localAccuracyOrder V (localRate δ T₀) (ε/2)
  obtain ⟨w,b,v,bias,heq⟩ := local_gelu_is_network q (localAccuracyOrder_pos _ _ _) n N
    (localAccuracyOrder_pos _ _ _) g geluKernel 1 T₀ H Z 2 2 (lt_trans zero_lt_one hT)
  refine ⟨w,b,v,bias,heq,?_,?_⟩
  · rw [heq]
    exact (local_analytic_accuracy_and_width n g hδ hM hε hg hbound (gelu_contDiff 2)
      geluKernel one_ne_zero (by intro t; simpa using gelu_second_derivative t)
      geluSpectral geluSpatial hT hH hsep hZ hhalo x hx F hF heven ha hA hsum henergy).1
  · have hw := (local_analytic_accuracy_and_width n g hδ hM hε hg hbound (gelu_contDiff 2)
      geluKernel one_ne_zero (by intro t; simpa using gelu_second_derivative t)
      geluSpectral geluSpatial hT hH hsep hZ hhalo x hx F hF heven ha hA hsum henergy).2
    have hk : max (1:ℝ) (1/2+1/2) = 1 := by norm_num
    rw [hk, Real.rpow_one, Real.rpow_one] at hw
    simp only [Fintype.card_sum, Fintype.card_fin, Nat.cast_add, Nat.cast_ofNat]
    exact add_le_add hw le_rfl

/-- Standard GELU has κ=1; including its two global affine-anchor neurons,
the actual local construction has width O(|log ε|^d), d=n+1. -/
theorem local_gelu_total_epsilon_bigO (n : ℕ) {T₀ H Z U V a b : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 < a) (hb : 0 < b) :
    (fun ε : ℝ =>
      let q := localAccuracyOrder U a (ε/2)
      let N := localAccuracyOrder V b (ε/2)
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z 2 2 ⊕ Fin 2) : ℝ))
      =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => |Real.log ε| ^ ((n : ℝ) + 1)) := by
  have hh := LocalScaling.local_grid_with_anchors_epsilon_bigO (ν := 2) (μ := 2)
    n 2 hT hH hZ hU hV ha hb
  norm_num only [one_div, invOf_eq_inv, show (2:ℝ)⁻¹+(2:ℝ)⁻¹=1 by norm_num, max_self] at hh
  exact hh

end QIRadon.LocalGELU
