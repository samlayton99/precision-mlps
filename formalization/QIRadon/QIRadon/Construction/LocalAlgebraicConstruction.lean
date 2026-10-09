import QIRadon.Scalar.LocalAlgebraicScaling
import QIRadon.Construction.CenteredConstruction

/-! Centered, uniform multidimensional construction from local real-profile
polynomial approximability and actual canonical angular Sobolev norms. No
global target Fourier moments or analytic target continuations are assumed.
Generic r>=2 still carries its explicit polynomial-anchor realization obligation. -/
noncomputable section
open MeasureTheory FourierTransform Polynomial Filter Asymptotics
open scoped Real SchwartzMap Topology
namespace QIRadon.LocalAlgebraicConstruction
open QIRadon.Gaussian QIRadon.Kernel QIRadon.KernelRates
open QIRadon.LocalProfileRate QIRadon.LocalAlgebraic QIRadon.AlgebraicConstruction
open QIRadon.ProfilePreparation QIRadon.ProfilePreparation.FilteredSamples

/-- Actual finite weighted sum of the local real-sample constructions. -/
def fullNetwork (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℝ → ℝ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀)
    (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  ∑ i : SphereHalfIndex q n,
    let θ := sphereNodeOnSphere q hq n i.val
    foldedSphereWeight q hq n i *
      localAlgebraicNetwork φ (g θ) K r N m T₀ H Z ν μ hN hT (inner ℝ θ.val x)

/-- Exact origin anchoring alters only one output constant. -/
def centeredNetwork (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℝ → ℝ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀)
    (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  f₀ + fullNetwork q hq n N hN φ g K r m T₀ H Z ν μ hT x -
    fullNetwork q hq n N hN φ g K r m T₀ H Z ν μ hT 0

def centeredTarget (n : ℕ) (f₀ : ℝ) (g : UnitSphere n → ℝ → ℝ)
    (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  f₀ + ∫ θ, g θ (inner ℝ θ.val x)-g θ 0 ∂normalizedSphere n

@[simp] theorem centeredNetwork_zero
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℝ → ℝ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    centeredNetwork f₀ q hq n N hN φ g K r m T₀ H Z ν μ hT 0 = f₀ := by
  unfold centeredNetwork
  ring

/-- Centered finite error from actual scalar and actual angular certificates.
Dimension is k+2, so the Sobolev uniform-norm loss is (k+1)/2. -/
theorem local_centered_finite_error
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (k N : ℕ) (hN : 0 < N)
    (g : UnitSphere (k+1) → ℝ → ℝ) {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ A M s T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν)
    (S : SpatialEnvelope K μ) (hA : 0 ≤ A) (hM : 0 ≤ M)
    (hg : ∀ θ x, |x| ≤ 1 → |g θ x| ≤ M)
    (happrox : ∀ θ, ∃ P : ℝ[X], P.degree < N ∧
      ∀ x : ℝ, |x| ≤ 1 → |g θ x-P.eval x| ≤ A*(N:ℝ)^(-s))
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ)
    (x : EuclideanSpace ℝ (Fin (k+1+1))) (hx : ‖x‖ ≤ 1)
    (F : C(UnitSphere (k+1), ℝ))
    (hF : ∀ θ, F θ=g θ (inner ℝ θ.val x)-g θ 0)
    (heven : ∀ θ, F (sphereNeg (k+1) θ)=F θ)
    {sang Aang : ℝ} (hsang : ((k:ℝ)+1)/2 < sang) (hAang : 0 ≤ Aang)
    (hsum : Summable (fun j : ℕ =>
      (((j:ℝ)+1)^sang*‖continuousHarmonicCoordinates (k+1) F j‖)^2))
    (henergy : (∑' j : ℕ,
      (((j:ℝ)+1)^sang*‖continuousHarmonicCoordinates (k+1) F j‖)^2) ≤ Aang^2) :
    |centeredNetwork f₀ q hq (k+1) N hN φ g K r m T₀ H Z ν μ
      (lt_trans zero_lt_one hT) x - centeredTarget (k+1) f₀ g x| ≤
      angularConstant k Aang (sang-((k:ℝ)+1)/2)*(q:ℝ)^(-(sang-((k:ℝ)+1)/2)) +
      2*localAlgebraicConstant E S A M T₀ s r*(N:ℝ)^(-s) := by
  let Q := fun θ t => localAlgebraicNetwork φ (g θ) K r N m T₀ H Z ν μ hN (lt_trans zero_lt_one hT) t
  let e := localAlgebraicConstant E S A M T₀ s r*(N:ℝ)^(-s)
  have hrow : ∀ θ, |(Q θ (inner ℝ θ.val x)-Q θ 0)-F θ| ≤ 2*e := by
    intro θ
    rw [hF]
    have hxerr := local_algebraic_error hφ hN K hm hkernel E S hA hM (hg θ) (happrox θ)
      hT hH hsep hZ hhalo _ (Multidimensional.projection_bound (k+1) θ hx)
    have hzerr := local_algebraic_error hφ hN K hm hkernel E S hA hM (hg θ) (happrox θ)
      hT hH hsep hZ hhalo 0 (by norm_num)
    exact CenteredConstruction.centered_difference_bound hxerr hzerr
  have hang := canonical_sobolev_power_error q hq k F heven hsang hAang hsum henergy
  have htar : centeredTarget (k+1) f₀ g x = f₀+∫ θ, F θ ∂normalizedSphere (k+1) := by
    unfold centeredTarget
    congr 1
    apply integral_congr_ae
    filter_upwards with θ
    exact (hF θ).symm
  have hnet : centeredNetwork f₀ q hq (k+1) N hN φ g K r m T₀ H Z ν μ
      (lt_trans zero_lt_one hT) x = f₀+
      ∑ i : SphereHalfIndex q (k+1), foldedSphereWeight q hq (k+1) i *
        (Q (sphereNodeOnSphere q hq (k+1) i.val) (inner ℝ (sphereNodeOnSphere q hq (k+1) i.val).val x)-
          Q (sphereNodeOnSphere q hq (k+1) i.val) 0) := by
    simp only [centeredNetwork, fullNetwork, inner_zero_right, mul_sub, Finset.sum_sub_distrib, Q]
    ring
  rw [htar, hnet, add_sub_add_left_eq_sub]
  have ht := normalized_approximation_error (foldedSphereWeight q hq (k+1))
    (fun i => Q (sphereNodeOnSphere q hq (k+1) i.val) (inner ℝ (sphereNodeOnSphere q hq (k+1) i.val).val x)-Q (sphereNodeOnSphere q hq (k+1) i.val) 0)
    (fun i => F (sphereNodeOnSphere q hq (k+1) i.val)) (2*e)
    (fun i => (foldedSphereWeight_positive q hq (k+1) i).le)
    (foldedSphereWeight_normalized q hq (k+1)) (fun i => hrow _) hang
  dsimp [e] at ht
  linarith

/-- Fixed finite construction, uniform accuracy, and actual all-halo width
for local polynomial-approximation profiles. No comparator polynomial is used
as network input; each is only a witness of the stated local regularity class. -/
theorem local_centered_uniform_accuracy_and_width
    (f₀ : ℝ) (k : ℕ) (g : UnitSphere (k+1) → ℝ → ℝ)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ A M s T₀ H Z ε : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν)
    (S : SpatialEnvelope K μ) (hA : 0 ≤ A) (hM : 0 ≤ M) (hs : 0 < s) (hε : 0 < ε)
    (hg : ∀ θ x, |x| ≤ 1 → |g θ x| ≤ M)
    (happrox : ∀ θ (N : ℕ), 0 < N → ∃ P : ℝ[X], P.degree < N ∧
      ∀ x : ℝ, |x| ≤ 1 → |g θ x-P.eval x| ≤ A*(N:ℝ)^(-s))
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ)
    (F : EuclideanSpace ℝ (Fin (k+1+1)) → C(UnitSphere (k+1), ℝ))
    (hF : ∀ x, ‖x‖ ≤ 1 → ∀ θ, F x θ=g θ (inner ℝ θ.val x)-g θ 0)
    (heven : ∀ x, ‖x‖ ≤ 1 → ∀ θ, F x (sphereNeg (k+1) θ)=F x θ)
    {sang Aang : ℝ} (hsang : ((k:ℝ)+1)/2 < sang) (hAang : 0 ≤ Aang)
    (hsum : ∀ x, ‖x‖ ≤ 1 → Summable (fun j : ℕ =>
      (((j:ℝ)+1)^sang*‖continuousHarmonicCoordinates (k+1) (F x) j‖)^2))
    (henergy : ∀ x, ‖x‖ ≤ 1 → (∑' j : ℕ,
      (((j:ℝ)+1)^sang*‖continuousHarmonicCoordinates (k+1) (F x) j‖)^2) ≤ Aang^2) :
    let t := sang-((k:ℝ)+1)/2
    let U := max 1 (angularConstant k Aang t)
    let V := max 1 (localAlgebraicConstant E S A M T₀ s r)
    let q := ⌈ScalarScaling.algebraicCutoff U t (ε/2)⌉₊
    let hq : 0 < q := Nat.ceil_pos.mpr (Real.exp_pos _)
    let N := ⌈ScalarScaling.algebraicCutoff V s (ε/4)⌉₊
    let hN : 0 < N := Nat.ceil_pos.mpr (Real.exp_pos _)
    (∀ x : EuclideanSpace ℝ (Fin (k+1+1)), ‖x‖ ≤ 1 →
      |centeredNetwork f₀ q hq (k+1) N hN φ g K r m T₀ H Z ν μ
        (lt_trans zero_lt_one hT) x-centeredTarget (k+1) f₀ g x| ≤ ε) ∧
    (Fintype.card (SphereHalfIndex q (k+1) × localGrid (2*N) T₀ H Z ν μ):ℝ) ≤
      (q:ℝ)^(k+1)*(((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^max 1 (1/μ+1/ν))*
        ((2*N:ℕ):ℝ)^max 1 (1/μ+1/ν)) := by
  let t := sang-((k:ℝ)+1)/2
  let U := max 1 (angularConstant k Aang t)
  let V := max 1 (localAlgebraicConstant E S A M T₀ s r)
  let q := ⌈ScalarScaling.algebraicCutoff U t (ε/2)⌉₊
  let N := ⌈ScalarScaling.algebraicCutoff V s (ε/4)⌉₊
  have hq : 0 < q := Nat.ceil_pos.mpr (Real.exp_pos _)
  have hN : 0 < N := Nat.ceil_pos.mpr (Real.exp_pos _)
  have hU : 0 ≤ U := le_trans zero_le_one (le_max_left _ _)
  have hV : 0 ≤ V := le_trans zero_le_one (le_max_left _ _)
  have ht : 0 < t := sub_pos.mpr hsang
  have heang := power_accuracy_choice hU (half_pos hε) ht
  have hescalar := power_accuracy_choice hV (show 0 < ε/4 by positivity) hs
  constructor
  · intro x hx
    have hh := local_centered_finite_error f₀ q hq k N hN g hφ K hm hkernel E S hA hM hg
      (fun θ => happrox θ N hN) hT hH hsep hZ hhalo x hx (F x) (hF x hx) (heven x hx)
      hsang hAang (hsum x hx) (henergy x hx)
    have hau := mul_le_mul_of_nonneg_right (le_max_right 1 (angularConstant k Aang t))
      (Real.rpow_nonneg (Nat.cast_nonneg q) (-t))
    have hsv := mul_le_mul_of_nonneg_right (le_max_right 1 (localAlgebraicConstant E S A M T₀ s r))
      (Real.rpow_nonneg (Nat.cast_nonneg N) (-s))
    change angularConstant k Aang t*(q:ℝ)^(-t) ≤ U*(q:ℝ)^(-t) at hau
    change localAlgebraicConstant E S A M T₀ s r*(N:ℝ)^(-s) ≤ V*(N:ℝ)^(-s) at hsv
    nlinarith
  · exact LocalMultidimensional.local_total_grid_width q hq (k+1) (lt_trans zero_lt_one hT)
      (by omega : 0 < 2*N) (by linarith) hZ

/-- Budget rescaling only multiplies the fixed error constant. -/
theorem algebraicCutoff_div_budget (C s ε b : ℝ) :
    ScalarScaling.algebraicCutoff C s (ε/b) = ScalarScaling.algebraicCutoff (b*C) s ε := by
  unfold ScalarScaling.algebraicCutoff
  rw [div_div_eq_mul_div, mul_comm C b]

/-- Actual all-halo tensor count for the centered local finite-smooth method.
It has exponent (d-1)/t+kappa/s, not the stronger global-Fourier exponent
(d-1)/t+1/s. Dimension d=n+1 and all constants are fixed as epsilon decreases. -/
theorem local_centered_total_epsilon_bigO (n : ℕ) {T₀ H Z ν μ U V t s : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 < U) (hV : 0 < V) (ht : 0 < t) (hs : 0 < s) :
    (fun ε : ℝ =>
      let q := ⌈ScalarScaling.algebraicCutoff U t (ε/2)⌉₊
      let N := ⌈ScalarScaling.algebraicCutoff V s (ε/4)⌉₊
      (Fintype.card (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ):ℝ))
      =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => ε^(-((n:ℝ)/t+max 1 (1/μ+1/ν)/s))) := by
  have hq := (MultidimensionalScaling.algebraic_direction_order_bigO
    (mul_pos (by norm_num : (0:ℝ)<2) hU) ht).pow n
  have hrow := LocalAlgebraicScaling.local_algebraic_grid_epsilon_bigO
    (ν := ν) (μ := μ) hT hH hZ (mul_pos (by norm_num : (0:ℝ)<4) hV) hs
  have hh := hq.mul hrow
  apply hh.congr'
  · filter_upwards with ε
    have hqpos : 0 < ⌈ScalarScaling.algebraicCutoff (2*U) t ε⌉₊ :=
      Nat.ceil_pos.mpr (Real.exp_pos _)
    simp only [algebraicCutoff_div_budget, Fintype.card_prod, sphereHalfIndex_card _ hqpos,
      Fintype.card_coe, Nat.cast_mul, Nat.cast_pow]
  · filter_upwards [eventually_mem_nhdsWithin] with ε hε
    have hε0 : 0 < ε := hε
    rw [← Real.rpow_natCast (ε^(-(1/t))) n,
      ← Real.rpow_mul hε0.le, ← Real.rpow_add hε0]
    congr 1
    ring

/-- A fixed explicit anchor realization is included in the total dimension law. -/
theorem local_centered_total_with_anchors_epsilon_bigO (n anchors : ℕ)
    {T₀ H Z ν μ U V t s : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 < U) (hV : 0 < V) (ht : 0 < t) (hs : 0 < s) :
    (fun ε : ℝ =>
      let q := ⌈ScalarScaling.algebraicCutoff U t (ε/2)⌉₊
      let N := ⌈ScalarScaling.algebraicCutoff V s (ε/4)⌉₊
      (Fintype.card (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ ⊕ Fin anchors):ℝ))
      =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => ε^(-((n:ℝ)/t+max 1 (1/μ+1/ν)/s))) := by
  have hcount : (fun ε : ℝ =>
      let q := ⌈ScalarScaling.algebraicCutoff U t (ε/2)⌉₊
      let N := ⌈ScalarScaling.algebraicCutoff V s (ε/4)⌉₊
      (Fintype.card (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ ⊕ Fin anchors):ℝ))
      =O[𝓝[>] (0:ℝ)] (fun ε : ℝ =>
        let q := ⌈ScalarScaling.algebraicCutoff U t (ε/2)⌉₊
        let N := ⌈ScalarScaling.algebraicCutoff V s (ε/4)⌉₊
        (Fintype.card (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ):ℝ)) := by
    apply IsBigO.of_bound (1+(anchors:ℝ))
    filter_upwards with ε
    let q := ⌈ScalarScaling.algebraicCutoff U t (ε/2)⌉₊
    let N := ⌈ScalarScaling.algebraicCutoff V s (ε/4)⌉₊
    have hq : 0 < q := Nat.ceil_pos.mpr (Real.exp_pos _)
    have hcard : 0 < Fintype.card (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ) := by
      rw [Fintype.card_prod, sphereHalfIndex_card q hq n, Fintype.card_coe]
      unfold localGrid
      rw [ScalarRate.symmetric_grid_card]
      positivity
    have hcardR : (1:ℝ) ≤ Fintype.card (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ) := by
      exact_mod_cast hcard
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _),
      Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
    simp only [Fintype.card_sum, Fintype.card_fin, Nat.cast_add]
    change _ ≤ (1+(anchors:ℝ))*
      (Fintype.card (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ):ℝ)
    nlinarith [Nat.cast_nonneg anchors (α := ℝ)]
  exact hcount.trans (local_centered_total_epsilon_bigO n hT hH hZ hU hV ht hs)

/-- Exact target interpretation when the real profiles are the actual canonical
Radon profiles on the used interval. No equality outside that interval is added. -/
theorem centeredTarget_canonical_on_ball (n : ℕ)
    (f : 𝓢(EuclideanSpace ℝ (Fin (n+1)), ℝ)) (g : UnitSphere n → ℝ → ℝ)
    (hg : ∀ θ s, |s| ≤ 1 → g θ s=Radon.realRidgeProfile f θ s)
    (x : EuclideanSpace ℝ (Fin (n+1))) (hx : ‖x‖ ≤ 1) :
    centeredTarget n (f 0) g x = f x := by
  have hh := CenteredConstruction.centeredReconstructed_canonical_on_ball n f
    (fun θ z => (g θ z.re:ℂ)) (by intro θ s hs; simpa using hg θ s hs) x hx
  simpa only [CenteredConstruction.centeredReconstructed, centeredTarget, Complex.ofReal_re,
    Complex.zero_re] using hh

/-- The finite-smooth construction also has an exact standard-GELU realization:
all local affine integration anchors combine into two neurons total, and exact
origin anchoring changes only the output bias. -/
theorem local_centered_gelu_is_network
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (g : UnitSphere n → ℝ → ℝ) (K : 𝓢(ℝ, ℝ))
    (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    ∃ (w : (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ ⊕ Fin 2) → Fin (n+1) → ℝ)
      (b v : (SphereHalfIndex q n × localGrid (2*N) T₀ H Z ν μ ⊕ Fin 2) → ℝ) (bias : ℝ),
      ∀ x : EuclideanSpace ℝ (Fin (n+1)),
        Architecture.network Activations.gelu w b v bias x =
          centeredNetwork f₀ q hq n N hN Activations.gelu g K 2 m T₀ H Z ν μ hT x := by
  let gC := fun θ : UnitSphere n => fun z : ℂ => Complex.ofReal ((filteredSamples N (g θ)).eval z.re)
  obtain ⟨w,b,v,bias,heq⟩ := CenteredConstruction.centered_gelu_is_network f₀ q hq n (2*N)
    (by omega) gC K m T₀ H Z ν μ hT
  refine ⟨w,b,v,bias,?_⟩
  intro x
  rw [heq]
  have hfull : ∀ y : EuclideanSpace ℝ (Fin (n+1)),
      LocalMultidimensional.localFiniteNetwork q hq n (2*N) (by omega) Activations.gelu
        gC K 2 m T₀ H Z ν μ hT y = fullNetwork q hq n N hN Activations.gelu g K 2 m T₀ H Z ν μ hT y := by
    intro y
    simp only [LocalMultidimensional.localFiniteNetwork, fullNetwork, localAlgebraicNetwork,
      gC, Complex.ofReal_re]
  unfold CenteredConstruction.centeredLocalNetwork centeredNetwork
  rw [hfull,hfull]

end QIRadon.LocalAlgebraicConstruction
