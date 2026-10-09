import QIRadon.Construction.KnownFunctionConstruction

/-! Ordinary MLP construction for a general derivative order, conditional only
on realizing the explicit integration-anchor polynomial. The actual kernel-grid
error is derived from the known-target theorem, never assumed by this wrapper.
Approximate anchors add their own error and actual hidden-unit count. -/
noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap Topology
namespace QIRadon.GenericAnchoredConstruction
open QIRadon.Kernel QIRadon.Gaussian QIRadon.KernelRates QIRadon.Radon
open QIRadon.LocalProfileRate QIRadon.ProfilePreparation QIRadon.KnownFunctionConstruction
open QIRadon.CenteredConstruction

/-- The explicit aggregate of the centered, degree-(r-1) integration anchors.
Its coefficients are the actual preparation-minus-raw-network jets at zero. -/
def centeredSpokeAnchor {d : ℕ} {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) (r : ℕ) (J : Finset ℤ) (θ : ι → Fin d → ℝ)
    (α : ι → ℝ) (ρ g : ι → ℝ → ℝ) (m γ h : ℝ) (x : Fin d → ℝ) : ℝ :=
  ∑ i, α i *
    (anchorPolynomial r (fun k => iteratedDeriv k (g i) 0-
      iteratedDeriv k (rawActivationNetwork φ (ρ i) r m γ h J) 0) (∑ k, θ i k*x k) -
    anchorPolynomial r (fun k => iteratedDeriv k (g i) 0-
      iteratedDeriv k (rawActivationNetwork φ (ρ i) r m γ h J) 0) 0)

@[simp] theorem centeredSpokeAnchor_zero {d : ℕ} {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) (r : ℕ) (J : Finset ℤ) (θ : ι → Fin d → ℝ)
    (α : ι → ℝ) (ρ g : ι → ℝ → ℝ) (m γ h : ℝ) :
    centeredSpokeAnchor φ r J θ α ρ g m γ h 0 = 0 := by
  simp [centeredSpokeAnchor]

/-- Exact decomposition into an ordinary raw-grid MLP and the explicitly
defined anchor. This is valid for arbitrary r, not just first-order activations. -/
theorem centered_spokes_decomposition {d : ℕ} {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) (r : ℕ) (J : Finset ℤ) (θ : ι → Fin d → ℝ)
    (α : ι → ℝ) (ρ g : ι → ℝ → ℝ) (f₀ m γ h : ℝ) :
    ∃ (a : (ι × J) → Fin d → ℝ) (b v : (ι × J) → ℝ) (bias : ℝ),
      ∀ x : Fin d → ℝ, Architecture.network φ a b v bias x+
        centeredSpokeAnchor φ r J θ α ρ g m γ h x =
        f₀+(∑ i, α i*anchoredActivationNetwork φ (ρ i) r m γ h J (g i) (∑ k, θ i k*x k))-
          ∑ i, α i*anchoredActivationNetwork φ (ρ i) r m γ h J (g i) 0 := by
  let a := fun p : ι × J => fun k => γ*θ p.1 k
  let b := fun p : ι × J => -γ*(h*(p.2.val:ℝ))
  let v := fun p : ι × J => α p.1*(h*γ/(m*γ^r))*ρ p.1 (h*p.2.val)
  let S₀ := ∑ i, α i*rawActivationNetwork φ (ρ i) r m γ h J 0
  refine ⟨a,b,v,f₀-S₀,?_⟩
  intro x
  have he := GenericArchitecture.raw_spokes_network φ r J θ α ρ m γ h x
  have hb : Architecture.network φ a b v (f₀-S₀) x =
      f₀+Architecture.network φ a b v 0 x-S₀ := by unfold Architecture.network; ring
  rw [hb,he]
  unfold centeredSpokeAnchor anchoredActivationNetwork
  simp only [mul_add,mul_sub,Finset.sum_add_distrib,Finset.sum_sub_distrib]
  dsimp [S₀]
  ring

/-- The anchor attached to the actual local sample/Fourier construction. It is
calculated solely from that construction's explicit densities and prepared jets. -/
def localAnchor
    (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀)
    (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  let L := localSuppression N T₀
  let B := L/Real.pi
  let γ := B/L^(1/ν)
  let h := 1/(H*B)
  let hγ : γ ≠ 0 := by
    have hL := localSuppression_ge_one hN hT
    have hLp : 0 < L := by dsimp [L]; linarith
    dsimp [γ,B]
    positivity
  let f := fun i : SphereHalfIndex q n => fun t : ℝ => (g (sphereNodeOnSphere q hq n i.val) (t:ℂ)).re
  let ρ := fun i => realReadoutDensity (derivativeSpectrum N (f i) T₀ r) K γ hγ B
  centeredSpokeAnchor φ r (localGrid N T₀ H Z ν μ)
    (fun i k => (sphereNodeOnSphere q hq n i.val).val k)
    (foldedSphereWeight q hq n) ρ (fun i => realPrepared N (f i) T₀) m γ h x

@[simp] theorem localAnchor_zero
    (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    localAnchor q hq n N hN φ g K r m T₀ H Z ν μ hT 0 = 0 := by
  simp [localAnchor]

/-- The actual centered local construction decomposes exactly as raw MLP plus
its known polynomial anchor, in the same physical input coordinates. -/
theorem centered_local_decomposition
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    ∃ (a : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) → Fin (n+1) → ℝ)
      (b v : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) → ℝ) (bias : ℝ),
      ∀ x : EuclideanSpace ℝ (Fin (n+1)), Architecture.network φ a b v bias x+
        localAnchor q hq n N hN φ g K r m T₀ H Z ν μ hT x =
        centeredLocalNetwork f₀ q hq n N hN φ g K r m T₀ H Z ν μ hT x := by
  let L := localSuppression N T₀
  let B := L/Real.pi
  let γ := B/L^(1/ν)
  let h := 1/(H*B)
  have hγ : γ ≠ 0 := by
    have hL := localSuppression_ge_one hN hT
    have hLp : 0 < L := by dsimp [L]; linarith
    dsimp [γ,B]
    positivity
  let θ := fun i : SphereHalfIndex q n => (sphereNodeOnSphere q hq n i.val).val
  let f := fun i : SphereHalfIndex q n => fun t : ℝ => (g (sphereNodeOnSphere q hq n i.val) (t:ℂ)).re
  let ρ := fun i => realReadoutDensity (derivativeSpectrum N (f i) T₀ r) K γ hγ B
  obtain ⟨a,b,v,bias,he⟩ := centered_spokes_decomposition φ r (localGrid N T₀ H Z ν μ)
    (fun i k => θ i k) (foldedSphereWeight q hq n) ρ (fun i => realPrepared N (f i) T₀) f₀ m γ h
  refine ⟨a,b,v,bias,?_⟩
  intro x
  rw [show localAnchor q hq n N hN φ g K r m T₀ H Z ν μ hT x =
    centeredSpokeAnchor φ r (localGrid N T₀ H Z ν μ) (fun i k => θ i k)
      (foldedSphereWeight q hq n) ρ (fun i => realPrepared N (f i) T₀) m γ h x from rfl,he]
  have hi (i : SphereHalfIndex q n) : (∑ k, θ i k*x k)=inner ℝ (θ i) x := by
    simp [PiLp.inner_apply,RCLike.inner_apply,mul_comm]
  simp_rw [hi]
  unfold centeredLocalNetwork LocalMultidimensional.localFiniteNetwork localActivationNetwork
  simp only [inner_zero_right]
  rfl

/-- General-r ordinary MLP endpoint from the actual known target. The only
additional approximation premise concerns the displayed explicit anchor
polynomial, not the target, profiles, angular norm, or raw kernel-grid error.
Its actual units and error are added. Exact anchors preserve the origin value. -/
theorem known_target_with_anchor {n : ℕ} {ιA : Type*} [Fintype ιA]
    (src : LocalAnalyticSource n) {ε : ℝ} (hε : 0 < ε)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ))
    {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0) (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν) (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    let U := angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))
    let V := localErrorConstant E S src.profileBound (scalarEllipse src) T₀ r
    let q := localAccuracyOrder U (angularRate src) (ε/2)
    let hq : 0 < q := localAccuracyOrder_pos _ _ _
    let N := localAccuracyOrder V (localRate (scalarEllipse src) T₀) (ε/4)
    let hN : 0 < N := localAccuracyOrder_pos _ _ _
    ∀ (aa : ιA → Fin (n+2) → ℝ) (ba va : ιA → ℝ) (biasA εanchor : ℝ),
      (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
        |Architecture.network φ aa ba va biasA x-
          localAnchor q hq (n+1) N hN φ src.profile K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) x|
          ≤ εanchor) →
      ∃ (a : (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ ⊕ ιA) → Fin (n+2) → ℝ)
        (b v : (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ ⊕ ιA) → ℝ) (bias : ℝ),
        (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
          |Architecture.network φ a b v bias x-src.target x| ≤ ε+εanchor) ∧
        (εanchor=0 → Architecture.network φ a b v bias (0:Fin (n+2)→ℝ)=src.target 0) ∧
        (Fintype.card (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ ⊕ ιA):ℝ) ≤
          (q:ℝ)^(n+1)*(((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^max 1 (1/μ+1/ν))*
            (N:ℝ)^max 1 (1/μ+1/ν)) + Fintype.card ιA := by
  intro U V q hq N hN aa ba va biasA εanchor hanchor
  obtain ⟨a,b,v,bias,he⟩ := centered_local_decomposition (src.target 0) q hq (n+1) N hN
    φ src.profile K r m T₀ H Z ν μ (lt_trans zero_lt_one hT)
  have hactual := general_kernel_accuracy_and_width src hε hφ K hm hkernel E S hT hH hsep hZ hhalo
  refine ⟨Sum.elim a aa,Sum.elim b ba,Sum.elim v va,bias+biasA,?_,?_,?_⟩
  · intro x hx
    rw [← Architecture.network_sum]
    have hgrid : |Architecture.network φ a b v bias x+
        localAnchor q hq (n+1) N hN φ src.profile K r m T₀ H Z ν μ
          (lt_trans zero_lt_one hT) x-src.target x| ≤ ε := by
      rw [he]
      exact hactual.1 x hx
    exact Architecture.anchor_error hgrid (hanchor x hx)
  · intro hzero
    have hanch := hanchor (0:EuclideanSpace ℝ (Fin (n+2))) (by simp)
    rw [localAnchor_zero,hzero,sub_zero] at hanch
    have hanch0 := abs_eq_zero.mp (le_antisymm hanch (abs_nonneg _))
    have hraw := he (0:EuclideanSpace ℝ (Fin (n+2)))
    rw [localAnchor_zero,add_zero,centeredLocalNetwork_zero] at hraw
    rw [← Architecture.network_sum]
    change Architecture.network φ a b v bias (0:EuclideanSpace ℝ (Fin (n+2))) +
      Architecture.network φ aa ba va biasA (0:EuclideanSpace ℝ (Fin (n+2))) = src.target 0
    rw [hraw,hanch0,add_zero]
  · simp only [Fintype.card_sum,Nat.cast_add]
    exact add_le_add hactual.2 le_rfl

end QIRadon.GenericAnchoredConstruction
