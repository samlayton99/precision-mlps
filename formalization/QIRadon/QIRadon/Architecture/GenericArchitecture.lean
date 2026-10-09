import QIRadon.Construction.CenteredConstruction

/-! Ordinary shallow architecture for general derivative kernels. The residual
integration polynomial is isolated explicitly; for r≤1 it is only an output
bias, and for higher r any supplied anchor network is added with its true
width and approximation error. No full-target fitting assumption is introduced. -/

noncomputable section
open scoped BigOperators Real SchwartzMap
namespace QIRadon.GenericArchitecture
open QIRadon.Kernel QIRadon.Gaussian QIRadon.LocalProfileRate
open QIRadon.LocalMultidimensional QIRadon.CenteredConstruction

theorem raw_spokes_network {d : ℕ} {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) (r : ℕ) (J : Finset ℤ) (θ : ι → Fin d → ℝ)
    (α : ι → ℝ) (ρ : ι → ℝ → ℝ) (m γ h : ℝ) (x : Fin d → ℝ) :
    Architecture.network φ
      (fun p : ι × J => fun k => γ*θ p.1 k)
      (fun p => -γ*(h*(p.2.val:ℝ)))
      (fun p => α p.1*(h*γ/(m*γ^r))*ρ p.1 (h*p.2.val)) 0 x =
    ∑ i, α i*rawActivationNetwork φ (ρ i) r m γ h J (∑ k, θ i k*x k) := by
  simp only [Architecture.network,zero_add,Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro i _
  have harg : ∀ j : ℤ, (∑ k, γ*θ i k*x k)+ -γ*(h*j) =
      γ*((∑ k, θ i k*x k)-h*j) := by
    intro j
    simp only [mul_assoc,← Finset.mul_sum]
    ring
  simp_rw [harg]
  rw [Finset.sum_coe_sort J (fun j : ℤ =>
    α i*(h*γ/(m*γ^r))*ρ i (h*j)*φ (γ*((∑ k, θ i k*x k)-h*j)))]
  unfold rawActivationNetwork
  simp only [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem anchored_difference_firstOrder (φ ρ g : ℝ → ℝ) {r : ℕ} (hr : r ≤ 1)
    (m γ h : ℝ) (J : Finset ℤ) (x : ℝ) :
    anchoredActivationNetwork φ ρ r m γ h J g x-anchoredActivationNetwork φ ρ r m γ h J g 0 =
      rawActivationNetwork φ ρ r m γ h J x-rawActivationNetwork φ ρ r m γ h J 0 := by
  have hh : r=0 ∨ r=1 := by omega
  rcases hh with rfl | rfl
  · simp only [anchoredActivationNetwork_zero]
  · simp only [anchoredActivationNetwork_one]
    ring

/-- The centered first-order or raw-kernel construction has no hidden anchor
polynomial at all: its exact grid is the entire hidden layer. -/
theorem centered_spokes_firstOrder {d : ℕ} {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) {r : ℕ} (hr : r ≤ 1)
    (J : Finset ℤ) (θ : ι → Fin d → ℝ) (α : ι → ℝ) (ρ g : ι → ℝ → ℝ)
    (f₀ m γ h : ℝ) :
    ∃ (a : (ι × J) → Fin d → ℝ) (b v : (ι × J) → ℝ) (bias : ℝ),
      ∀ x : Fin d → ℝ, Architecture.network φ a b v bias x =
        f₀+(∑ i, α i*anchoredActivationNetwork φ (ρ i) r m γ h J (g i) (∑ k, θ i k*x k))-
          ∑ i, α i*anchoredActivationNetwork φ (ρ i) r m γ h J (g i) 0 := by
  let a := fun p : ι × J => fun k => γ*θ p.1 k
  let b := fun p : ι × J => -γ*(h*(p.2.val:ℝ))
  let v := fun p : ι × J => α p.1*(h*γ/(m*γ^r))*ρ p.1 (h*p.2.val)
  let S₀ := ∑ i, α i*rawActivationNetwork φ (ρ i) r m γ h J 0
  refine ⟨a,b,v,f₀-S₀,?_⟩
  intro x
  have he := raw_spokes_network φ r J θ α ρ m γ h x
  have hd : (∑ i, α i*anchoredActivationNetwork φ (ρ i) r m γ h J (g i) (∑ k, θ i k*x k))-
      (∑ i, α i*anchoredActivationNetwork φ (ρ i) r m γ h J (g i) 0) =
        (∑ i, α i*rawActivationNetwork φ (ρ i) r m γ h J (∑ k, θ i k*x k))-S₀ := by
    dsimp only [S₀]
    rw [← Finset.sum_sub_distrib,← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    rw [← mul_sub,← mul_sub,anchored_difference_firstOrder φ (ρ i) (g i) hr]
  have hc : Architecture.network φ a b v (f₀-S₀) x =
      f₀+Architecture.network φ a b v 0 x-S₀ := by unfold Architecture.network; ring
  rw [hc,he]
  linarith [hd]

theorem centered_local_firstOrder
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) {r : ℕ} (hr : r ≤ 1) (g : UnitSphere n → ℂ → ℂ)
    (K : 𝓢(ℝ, ℝ)) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    ∃ (a : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) → Fin (n+1) → ℝ)
      (b v : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) → ℝ) (bias : ℝ),
      ∀ x : EuclideanSpace ℝ (Fin (n+1)), Architecture.network φ a b v bias x =
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
  obtain ⟨a,b,v,bias,he⟩ := centered_spokes_firstOrder φ hr (localGrid N T₀ H Z ν μ)
    (fun i k => θ i k) (foldedSphereWeight q hq n) ρ (fun i => realPrepared N (f i) T₀) f₀ m γ h
  refine ⟨a,b,v,bias,?_⟩
  intro x
  rw [he]
  have hi (i : SphereHalfIndex q n) : (∑ k, θ i k*x k)=inner ℝ (θ i) x := by
    simp [PiLp.inner_apply,RCLike.inner_apply,mul_comm]
  simp_rw [hi]
  unfold centeredLocalNetwork localFiniteNetwork localActivationNetwork
  simp only [inner_zero_right]
  rfl

/-- A supplied anchor network adds its actual neurons and only its own error.
The assumed approximation concerns the explicit integration polynomial, never
the full target or the kernel construction's claimed approximation result. -/
theorem add_anchor_network {d : ℕ} {ι κ : Type*} [Fintype ι] [Fintype κ]
    (φ : ℝ → ℝ) (a : ι → Fin d → ℝ) (b v : ι → ℝ) (bias : ℝ)
    (aa : κ → Fin d → ℝ) (ba va : κ → ℝ) (biasA : ℝ)
    (poly target : (Fin d → ℝ) → ℝ) (D : Set (Fin d → ℝ)) {E Eanchor : ℝ}
    (hgrid : ∀ x ∈ D, |Architecture.network φ a b v bias x+poly x-target x| ≤ E)
    (hanchor : ∀ x ∈ D, |Architecture.network φ aa ba va biasA x-poly x| ≤ Eanchor) :
    (∀ x ∈ D, |Architecture.network φ (Sum.elim a aa) (Sum.elim b ba)
      (Sum.elim v va) (bias+biasA) x-target x| ≤ E+Eanchor) ∧
      Fintype.card (ι ⊕ κ)=Fintype.card ι+Fintype.card κ := by
  refine ⟨?_,Fintype.card_sum⟩
  intro x hx
  rw [← Architecture.network_sum]
  exact Architecture.anchor_error (hgrid x hx) (hanchor x hx)

end QIRadon.GenericArchitecture
