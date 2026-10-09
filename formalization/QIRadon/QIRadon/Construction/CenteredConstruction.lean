import QIRadon.Construction.LocalMultidimensional
import QIRadon.Construction.LocalGELU

/-! Uniform construction with the approved centered angular integrand.
The origin is anchored exactly by an ordinary output constant. Only the
canonical harmonic energy of gθ(θ·x)-gθ(0) is required, not that of gθ(0).
The scalar profiles use the same actual local sample preparation as before. -/
noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap Topology
namespace QIRadon.CenteredConstruction
open QIRadon.Gaussian QIRadon.Kernel QIRadon.KernelRates
open QIRadon.ProfilePreparation QIRadon.LocalProfileRate QIRadon.LocalMultidimensional

/-- The target reconstructed from centered profiles and a prescribed origin value. -/
def centeredReconstructed (n : ℕ) (f₀ : ℝ) (g : UnitSphere n → ℂ → ℂ)
    (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  f₀ + ∫ θ, (g θ (inner ℝ θ.val x : ℝ)).re - (g θ 0).re ∂normalizedSphere n

/-- One fixed finite local construction, with its constant adjusted to hit f₀.
Subtracting a scalar value at zero does not add hidden neurons. -/
def centeredLocalNetwork (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀)
    (x : EuclideanSpace ℝ (Fin (n+1))) : ℝ :=
  f₀ + localFiniteNetwork q hq n N hN φ g K r m T₀ H Z ν μ hT x -
    localFiniteNetwork q hq n N hN φ g K r m T₀ H Z ν μ hT 0

@[simp] theorem centeredLocalNetwork_zero
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (φ : ℝ → ℝ) (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    centeredLocalNetwork f₀ q hq n N hN φ g K r m T₀ H Z ν μ hT 0 = f₀ := by
  unfold centeredLocalNetwork
  ring

@[simp] theorem centeredReconstructed_zero (n : ℕ) (f₀ : ℝ)
    (g : UnitSphere n → ℂ → ℂ) : centeredReconstructed n f₀ g 0 = f₀ := by
  simp [centeredReconstructed]

/-- Exact centering at the level of each scalar row. -/
theorem centered_difference_bound {u v f g e : ℝ}
    (hu : |u-f| ≤ e) (hv : |v-g| ≤ e) : |(u-v)-(f-g)| ≤ 2*e := by
  have hid : (u-v)-(f-g) = (u-f)-(v-g) := by ring
  rw [hid]
  have hh := abs_sub_le (u-f) 0 (v-g)
  simp only [sub_zero, zero_sub, abs_neg] at hh
  linarith

/-- Derived error of the actual centered network. The factor two is precisely
the approved fallback for centering a certified uncentered scalar approximation. -/
theorem centered_analytic_finite_error
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (g : UnitSphere n → ℂ → ℂ) {δ M : ℝ} (hδ : 0 < δ)
    (hg : ∀ θ, AnalyticOnNhd ℂ (g θ) (bernsteinEllipse δ))
    (hbound : ∀ θ z, z ∈ bernsteinEllipse δ → ‖g θ z‖ ≤ M)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ) (hT : 1 < T₀)
    (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ)
    (x : EuclideanSpace ℝ (Fin (n+1))) (hx : ‖x‖ ≤ 1)
    (F : C(UnitSphere n, ℝ))
    (hF : ∀ θ, F θ = (g θ (inner ℝ θ.val x : ℝ)).re - (g θ 0).re)
    (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ => Real.exp (2*a*k)*‖continuousHarmonicCoordinates n F k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2*a*k)*‖continuousHarmonicCoordinates n F k‖^2) ≤ A^2) :
    |centeredLocalNetwork f₀ q hq n N hN φ g K r m T₀ H Z ν μ
        (lt_trans zero_lt_one hT) x - centeredReconstructed n f₀ g x| ≤
      A*Real.sqrt (SharpAngular.traceConstant n a)*Real.exp (-a*q) +
      2*localErrorConstant E S M δ T₀ r*Real.exp (-localRate δ T₀*N) := by
  let Q := fun θ t => localActivationNetwork φ (fun u => (g θ (u:ℂ)).re) K r N
    m T₀ H Z ν μ hN (lt_trans zero_lt_one hT) t
  let e := localErrorConstant E S M δ T₀ r*Real.exp (-localRate δ T₀*N)
  have hrow : ∀ θ, |(Q θ (inner ℝ θ.val x)-Q θ 0)-F θ| ≤ 2*e := by
    intro θ
    rw [hF]
    have hxerr := local_analytic_activation_error hδ (hg θ) (hbound θ) hφ K hm hkernel E S
      hT hN hH hsep hZ hhalo _ (abs_le.mp (Multidimensional.projection_bound n θ hx))
    have hzerr := local_analytic_activation_error hδ (hg θ) (hbound θ) hφ K hm hkernel E S
      hT hN hH hsep hZ hhalo 0 (by constructor <;> norm_num)
    exact centered_difference_bound hxerr hzerr
  have hang := SharpAngular.canonical_quadrature_error q hq n F heven ha hA hsum henergy
  have heq : centeredReconstructed n f₀ g x = f₀ + ∫ θ, F θ ∂normalizedSphere n := by
    unfold centeredReconstructed
    congr 1
    apply integral_congr_ae
    filter_upwards with θ
    exact (hF θ).symm
  have hnet : centeredLocalNetwork f₀ q hq n N hN φ g K r m T₀ H Z ν μ
      (lt_trans zero_lt_one hT) x = f₀ +
      ∑ i : SphereHalfIndex q n, foldedSphereWeight q hq n i *
        (Q (sphereNodeOnSphere q hq n i.val) (inner ℝ (sphereNodeOnSphere q hq n i.val).val x) -
          Q (sphereNodeOnSphere q hq n i.val) 0) := by
    simp only [centeredLocalNetwork, localFiniteNetwork, inner_zero_right,
      mul_sub, Finset.sum_sub_distrib, Q]
    ring
  rw [heq, hnet]
  rw [add_sub_add_left_eq_sub]
  have ht := normalized_approximation_error (foldedSphereWeight q hq n)
    (fun i => Q (sphereNodeOnSphere q hq n i.val) (inner ℝ (sphereNodeOnSphere q hq n i.val).val x)-Q (sphereNodeOnSphere q hq n i.val) 0)
    (fun i => F (sphereNodeOnSphere q hq n i.val)) (2*e)
    (fun i => (foldedSphereWeight_positive q hq n i).le)
    (foldedSphereWeight_normalized q hq n) (fun i => hrow _) hang
  dsimp [e] at ht
  linarith

/-- Uniform epsilon guarantee for one fixed centered network. All target-side
norm hypotheses are on actual centered canonical harmonic coordinates, uniformly
in x. There is no approximation-error hypothesis and no regularity assumption on
the separate angular origin profile. N receives budget epsilon/4 to pay for the
factor two from centering. -/
theorem centered_analytic_uniform_accuracy_and_width
    (n : ℕ) (f₀ : ℝ) (g : UnitSphere n → ℂ → ℂ) {δ M ε : ℝ}
    (hδ : 0 < δ) (hM : 0 ≤ M) (hε : 0 < ε)
    (hg : ∀ θ, AnalyticOnNhd ℂ (g θ) (bernsteinEllipse δ))
    (hbound : ∀ θ z, z ∈ bernsteinEllipse δ → ‖g θ z‖ ≤ M)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ) (hT : 1 < T₀)
    (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ)
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
    let V := localErrorConstant E S M δ T₀ r
    let q := localAccuracyOrder U a (ε/2)
    let N := localAccuracyOrder V (localRate δ T₀) (ε/4)
    (∀ x : EuclideanSpace ℝ (Fin (n+1)), ‖x‖ ≤ 1 →
      |centeredLocalNetwork f₀ q (localAccuracyOrder_pos _ _ _) n N
        (localAccuracyOrder_pos _ _ _) φ g K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) x -
        centeredReconstructed n f₀ g x| ≤ ε) ∧
    (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ) : ℝ) ≤
      (q:ℝ)^n * (((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^max 1 (1/μ+1/ν)) *
        (N:ℝ)^max 1 (1/μ+1/ν)) := by
  let U := A*Real.sqrt (SharpAngular.traceConstant n a)
  let V := localErrorConstant E S M δ T₀ r
  let q := localAccuracyOrder U a (ε/2)
  let N := localAccuracyOrder V (localRate δ T₀) (ε/4)
  have hq : 0 < q := localAccuracyOrder_pos _ _ _
  have hN : 0 < N := localAccuracyOrder_pos _ _ _
  have hU : 0 ≤ U := by dsimp [U]; positivity
  have hV : 0 ≤ V := localErrorConstant_nonneg E S hM hδ hT r
  constructor
  · intro x hx
    have hh := centered_analytic_finite_error f₀ q hq n N hN g hδ hg hbound hφ K hm hkernel E S
      hT hH hsep hZ hhalo x hx (F x) (hF x hx) (heven x hx) ha hA (hsum x hx) (henergy x hx)
    have hang := localAccuracyOrder_error hU ha (half_pos hε)
    have hscalar := localAccuracyOrder_error hV (localRate_pos hδ hT) (show 0 < ε/4 by positivity)
    apply hh.trans
    change U*Real.exp (-a*q) + 2*V*Real.exp (-localRate δ T₀*N) ≤ ε
    nlinarith
  · exact local_total_grid_width q hq n (lt_trans zero_lt_one hT) hN (by linarith) hZ

/-- A half-sized scalar budget is equivalently a doubled scalar constant.
This identity links the centered construction's orders to the already counted
general local construction. -/
theorem localAccuracyOrder_quarter (V b ε : ℝ) :
    localAccuracyOrder V b (ε/4) = localAccuracyOrder (2*V) b (ε/2) := by
  unfold localAccuracyOrder
  have hh : V/(ε/4) = (2*V)/(ε/2) := by ring
  rw [hh]

/-- Centering only changes the output bias of the ordinary GELU network.
Its exact same grid+2 hidden units work at every input. -/
theorem centered_gelu_is_network
    (f₀ : ℝ) (q : ℕ) (hq : 0 < q) (n N : ℕ) (hN : 0 < N)
    (g : UnitSphere n → ℂ → ℂ) (K : 𝓢(ℝ, ℝ))
    (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    ∃ (w : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ ⊕ Fin 2) → Fin (n+1) → ℝ)
      (b v : (SphereHalfIndex q n × localGrid N T₀ H Z ν μ ⊕ Fin 2) → ℝ) (bias : ℝ),
      ∀ x : EuclideanSpace ℝ (Fin (n+1)),
        Architecture.network Activations.gelu w b v bias x =
          centeredLocalNetwork f₀ q hq n N hN Activations.gelu g K 2 m T₀ H Z ν μ hT x := by
  obtain ⟨w,b,v,bias,heq⟩ := LocalGELU.local_gelu_is_network q hq n N hN g K m T₀ H Z ν μ hT
  refine ⟨w,b,v, f₀+ bias - localFiniteNetwork q hq n N hN Activations.gelu g K 2
    m T₀ H Z ν μ hT 0, ?_⟩
  intro x
  have hb : Architecture.network Activations.gelu w b v
      (f₀+ bias - localFiniteNetwork q hq n N hN Activations.gelu g K 2 m T₀ H Z ν μ hT 0) x =
      f₀ + Architecture.network Activations.gelu w b v bias x -
        localFiniteNetwork q hq n N hN Activations.gelu g K 2 m T₀ H Z ν μ hT 0 := by
    unfold Architecture.network
    ring
  rw [hb, heq]
  rfl

/-- Halos and a fixed explicit anchor realization retain the same geometric
scaling after exact centering and its scalar epsilon/4 budget. -/
theorem centered_total_grid_epsilon_bigO (n anchors : ℕ) {T₀ H Z ν μ U V a b : ℝ}
    (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z)
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 < a) (hb : 0 < b) :
    (fun ε : ℝ =>
      let q := localAccuracyOrder U a (ε/2)
      let N := localAccuracyOrder V b (ε/4)
      (Fintype.card (SphereHalfIndex q n × localGrid N T₀ H Z ν μ ⊕ Fin anchors) : ℝ))
      =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => |Real.log ε|^((n:ℝ)+max 1 (1/μ+1/ν))) := by
  simpa only [localAccuracyOrder_quarter] using
    LocalScaling.local_grid_with_anchors_epsilon_bigO (ν := ν) (μ := μ)
      n anchors hT hH hZ hU (show 0 ≤ 2*V by positivity) ha hb

/-- Integrability of the actual canonical Radon rows follows from Fourier
integrability of the Schwartz target, not from an added angular-profile oracle. -/
theorem canonical_ridge_integrable (n : ℕ)
    (f : 𝓢(EuclideanSpace ℝ (Fin (n+1)), ℝ))
    (x : EuclideanSpace ℝ (Fin (n+1))) :
    Integrable (fun θ : UnitSphere n => Radon.realRidgeProfile f θ (inner ℝ θ.val x))
      (normalizedSphere n) := by
  have hhat : Integrable (𝓕 (fun y => (f y : ℂ))) := by
    change Integrable (𝓕 ((Radon.complexifySchwartz f) : EuclideanSpace ℝ (Fin (n+1)) → ℂ))
    rw [← SchwartzMap.fourier_coe]
    exact (𝓕 (Radon.complexifySchwartz f)).integrable
  have hpos := Radon.integrable_positive_ridge hhat x
  have hr := hpos.re.const_mul (Radon.surfaceArea (EuclideanSpace ℝ (Fin (n+1))))
  exact hr.smul_measure (ENNReal.inv_ne_top.mpr (Radon.surface_mass_ne_zero))

/-- The centered reconstruction is the actual original Schwartz target when
the scalar profile continuation agrees with the proved canonical Radon profile.
This supplies an exact target identity independent of all approximation bounds. -/
theorem centeredReconstructed_canonical (n : ℕ)
    (f : 𝓢(EuclideanSpace ℝ (Fin (n+1)), ℝ))
    (g : UnitSphere n → ℂ → ℂ)
    (hg : ∀ θ (s : ℝ), (g θ (s:ℂ)).re = Radon.realRidgeProfile f θ s)
    (x : EuclideanSpace ℝ (Fin (n+1))) :
    centeredReconstructed n (f 0) g x = f x := by
  have hx := canonical_ridge_integrable n f x
  have hzero : Integrable (fun θ : UnitSphere n => Radon.realRidgeProfile f θ 0)
      (normalizedSphere n) := by simpa only [inner_zero_right] using canonical_ridge_integrable n f 0
  have hid₀ : (∫ θ : UnitSphere n, Radon.realRidgeProfile f θ 0 ∂normalizedSphere n) = f 0 := by
    simpa only [inner_zero_right, normalizedSphere, Radon.normalizedSphere, Radon.surfaceMeasure] using
      (Radon.normalized_real_schwartz_reconstruction f 0).symm
  have hidₓ : (∫ θ : UnitSphere n, Radon.realRidgeProfile f θ (inner ℝ θ.val x)
      ∂normalizedSphere n) = f x := by
    simpa only [normalizedSphere, Radon.normalizedSphere, Radon.surfaceMeasure] using
      (Radon.normalized_real_schwartz_reconstruction f x).symm
  have hgz : ∀ θ, (g θ 0).re = Radon.realRidgeProfile f θ 0 := by
    intro θ
    simpa only [Complex.ofReal_zero] using hg θ 0
  unfold centeredReconstructed
  simp_rw [hg, hgz]
  rw [integral_sub hx hzero, hid₀, hidₓ]
  ring

/-- Local continuation agreement is enough on the target ball; no equality of
the complex continuation with the Fourier profile outside [-1,1] is required. -/
theorem centeredReconstructed_canonical_on_ball (n : ℕ)
    (f : 𝓢(EuclideanSpace ℝ (Fin (n+1)), ℝ))
    (g : UnitSphere n → ℂ → ℂ)
    (hg : ∀ θ (s : ℝ), |s| ≤ 1 → (g θ (s:ℂ)).re = Radon.realRidgeProfile f θ s)
    (x : EuclideanSpace ℝ (Fin (n+1))) (hx : ‖x‖ ≤ 1) :
    centeredReconstructed n (f 0) g x = f x := by
  have hix := canonical_ridge_integrable n f x
  have hiz : Integrable (fun θ : UnitSphere n => Radon.realRidgeProfile f θ 0)
      (normalizedSphere n) := by simpa only [inner_zero_right] using canonical_ridge_integrable n f 0
  have hz : ∀ θ, (g θ 0).re = Radon.realRidgeProfile f θ 0 := by
    intro θ
    simpa only [Complex.ofReal_zero] using hg θ 0 (by norm_num)
  have hp : ∀ θ, (g θ (inner ℝ θ.val x : ℝ)).re =
      Radon.realRidgeProfile f θ (inner ℝ θ.val x) :=
    fun θ => hg θ _ (Multidimensional.projection_bound n θ hx)
  have hid₀ : (∫ θ : UnitSphere n, Radon.realRidgeProfile f θ 0 ∂normalizedSphere n) = f 0 := by
    simpa only [inner_zero_right, normalizedSphere, Radon.normalizedSphere, Radon.surfaceMeasure] using
      (Radon.normalized_real_schwartz_reconstruction f 0).symm
  have hidₓ : (∫ θ : UnitSphere n, Radon.realRidgeProfile f θ (inner ℝ θ.val x)
      ∂normalizedSphere n) = f x := by
    simpa only [normalizedSphere, Radon.normalizedSphere, Radon.surfaceMeasure] using
      (Radon.normalized_real_schwartz_reconstruction f x).symm
  unfold centeredReconstructed
  simp_rw [hp, hz]
  rw [integral_sub hix hiz, hid₀, hidₓ]
  ring

end QIRadon.CenteredConstruction
