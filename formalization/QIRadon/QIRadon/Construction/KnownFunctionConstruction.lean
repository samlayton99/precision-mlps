import QIRadon.Radon.SourceOrbit
import QIRadon.Radon.EllipseTube
import QIRadon.Construction.CenteredGELU
import QIRadon.Architecture.GenericArchitecture

/-! Original bounded local analytic target → actual finite QI–Radon network.
No scalar-profile regularity, angular derivative bounds, harmonic energy,
quadrature accuracy, or network approximation is assumed in the source.
All such properties are derived from its actual finite boundary construction.

The generic theorem keeps its explicit integration-polynomial realization
obligation. The standard GELU corollary is an ordinary one-hidden-layer MLP
with precisely two additional anchor neurons. -/

noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap Topology
namespace QIRadon.KnownFunctionConstruction
open QIRadon.Radon QIRadon.Gaussian QIRadon.KernelRates
open QIRadon.ProfilePreparation QIRadon.LocalProfileRate QIRadon.CenteredConstruction
open QIRadon.Activations

variable {n : ℕ}

def scalarEllipse (src : LocalAnalyticSource n) : ℝ := Real.log (1+src.params.h/2)
def angularScale (src : LocalAnalyticSource n) : ℝ := 4*((n+2:ℕ):ℝ)/src.params.b
def angularRate (src : LocalAnalyticSource n) : ℝ := CasimirDecay.energyRate (angularScale src)
def angularNorm (src : LocalAnalyticSource n) : ℝ :=
  CasimirDecay.energyConstant (2*src.profileBound) (angularScale src)

theorem scalarEllipse_pos (src : LocalAnalyticSource n) : 0 < scalarEllipse src := by
  unfold scalarEllipse
  apply Real.log_pos
  linarith [src.params.h_pos]

theorem angularScale_pos (src : LocalAnalyticSource n) : 0 < angularScale src := by
  unfold angularScale
  exact div_pos (by positivity) src.params.orbit_budget_pos

theorem angularRate_pos (src : LocalAnalyticSource n) : 0 < angularRate src :=
  CasimirDecay.energyRate_pos (angularScale_pos src)

theorem angularNorm_nonneg (src : LocalAnalyticSource n) : 0 ≤ angularNorm src :=
  CasimirDecay.energyConstant_nonneg (mul_nonneg (by norm_num) src.profileBound_nonneg)

/-- The actual scalar profile gets its ellipse and bound from the original
target and the prescribed finite-domain extension. -/
theorem scalar_profile_certificate (src : LocalAnalyticSource n) (θ : UnitSphere (n+1)) :
    AnalyticOnNhd ℂ (src.profile θ) (bernsteinEllipse (scalarEllipse src)) ∧
    ∀ z ∈ bernsteinEllipse (scalarEllipse src), ‖src.profile θ z‖ ≤ src.profileBound := by
  apply ellipse_transfer src.params.one_lt_T.le src.params.h_pos
  · exact (src.profile_analytic θ).differentiableOn.mono
      (scalarTube_mono le_rfl (by linarith [src.params.h_pos]))
  · exact fun z hz => src.profile_bound θ hz

theorem profile_canonical_on_unit (src : LocalAnalyticSource n) (θ : UnitSphere (n+1))
    (s : ℝ) (hs : |s| ≤ 1) :
    (src.profile θ (s:ℂ)).re = realRidgeProfile src.extendedReal θ s := by
  rw [src.profile_eq_realRidge θ (hs.trans src.params.one_lt_T.le),Complex.ofReal_re]

/-- Canonical angular spectral energy is a conclusion from the bounded
complex target, uniformly for every x in the unit ball. -/
theorem angular_energy_certificate (src : LocalAnalyticSource n)
    (x : EuclideanSpace ℝ (Fin (n+2))) (hx : ‖x‖ ≤ 1) :
    Summable (fun k : ℕ => Real.exp (2*angularRate src*k)*
      ‖continuousHarmonicCoordinates (n+1) (canonicalAngular src.extendedReal x) k‖^2) ∧
    (∑' k : ℕ, Real.exp (2*angularRate src*k)*
      ‖continuousHarmonicCoordinates (n+1) (canonicalAngular src.extendedReal x) k‖^2) ≤
      (angularNorm src)^2 := by
  exact (src.boundedOrbit x hx).canonical_energy (by omega) src.params.orbit_budget_pos
    (mul_nonneg (by norm_num) src.profileBound_nonneg)

theorem centered_profile_agreement (src : LocalAnalyticSource n)
    (x : EuclideanSpace ℝ (Fin (n+2))) (hx : ‖x‖ ≤ 1) (θ : UnitSphere (n+1)) :
    canonicalAngular src.extendedReal x θ =
      (src.profile θ (inner ℝ θ.val x:ℝ)).re-(src.profile θ 0).re := by
  rw [canonicalAngular_apply,
    profile_canonical_on_unit src θ _ (Multidimensional.projection_bound (n+1) θ hx)]
  have hh := profile_canonical_on_unit src θ 0 (by norm_num)
  simpa only [Complex.ofReal_zero] using congrArg (fun t => realRidgeProfile src.extendedReal θ
    (inner ℝ θ.val x)-t) hh.symm

theorem centered_profile_even (src : LocalAnalyticSource n)
    (x : EuclideanSpace ℝ (Fin (n+2))) (θ : UnitSphere (n+1)) :
    canonicalAngular src.extendedReal x (sphereNeg (n+1) θ) = canonicalAngular src.extendedReal x θ :=
  canonicalAngular_antipodal src.extendedReal x θ

/-- Actual known-target construction with general derivative kernel.
This is an exact-arithmetic theorem; numerical acquisition/evaluation gets its
own budget. The displayed grid count includes both halos on every direction. -/
theorem general_kernel_accuracy_and_width
    (src : LocalAnalyticSource n) {ε : ℝ} (hε : 0 < ε)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ))
    {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0) (hkernel : ∀ u, iteratedDeriv r φ u = m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν) (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    let U := angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))
    let V := localErrorConstant E S src.profileBound (scalarEllipse src) T₀ r
    let q := localAccuracyOrder U (angularRate src) (ε/2)
    let N := localAccuracyOrder V (localRate (scalarEllipse src) T₀) (ε/4)
    (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
      |centeredLocalNetwork (src.target 0) q (localAccuracyOrder_pos _ _ _) (n+1) N
        (localAccuracyOrder_pos _ _ _) φ src.profile K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) x-
        src.target x| ≤ ε) ∧
    (Fintype.card (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ):ℝ) ≤
      (q:ℝ)^(n+1)*(((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^max 1 (1/μ+1/ν))*
        (N:ℝ)^max 1 (1/μ+1/ν)) := by
  have hzero : src.extendedReal 0=src.target 0 := src.extendedReal_eq_target_unitBall (by simp)
  have hh := centered_analytic_uniform_accuracy_and_width (n+1) (src.target 0) src.profile
    (scalarEllipse_pos src) src.profileBound_nonneg hε
    (fun θ => (scalar_profile_certificate src θ).1) (fun θ => (scalar_profile_certificate src θ).2)
    hφ K hm hkernel E S hT hH hsep hZ hhalo
    (canonicalAngular src.extendedReal) (centered_profile_agreement src)
    (fun x _ θ => centered_profile_even src x θ)
    (angularRate_pos src) (angularNorm_nonneg src)
    (fun x hx => (angular_energy_certificate src x hx).1)
    (fun x hx => (angular_energy_certificate src x hx).2)
  refine ⟨?_,hh.2⟩
  intro x hx
  have heq := centeredReconstructed_canonical_on_ball (n+1) src.extendedReal src.profile
    (profile_canonical_on_unit src) x hx
  rw [hzero,src.extendedReal_eq_target_unitBall hx] at heq
  rw [← heq]
  exact hh.1 x hx

/-- Ordinary one-hidden-layer GELU network from the original analytic target.
The existential arrays are the explicitly flattened finite construction;
every profile/angular hypothesis of the lower-level theorem is proved above. -/
theorem gelu_accuracy_and_width
    (src : LocalAnalyticSource n) {ε T₀ H Z : ℝ} (hε : 0 < ε)
    (hT : 1 < T₀) (hH : 2 ≤ H)
    (hsep : geluSpectral.A+4 ≤ geluSpectral.a*(H/2)^(2:ℝ))
    (hZ : 0 ≤ Z) (hhalo : geluSpectral.A+3 ≤ (geluSpatial.b/2)*Z^(2:ℝ)) :
    let U := angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))
    let V := localErrorConstant geluSpectral geluSpatial src.profileBound (scalarEllipse src) T₀ 2
    let q := localAccuracyOrder U (angularRate src) (ε/2)
    let N := localAccuracyOrder V (localRate (scalarEllipse src) T₀) (ε/4)
    ∃ (w : (SphereHalfIndex q (n+1) × localGrid N T₀ H Z 2 2 ⊕ Fin 2) → Fin (n+2) → ℝ)
      (b v : (SphereHalfIndex q (n+1) × localGrid N T₀ H Z 2 2 ⊕ Fin 2) → ℝ) (bias : ℝ),
      (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
        |Architecture.network gelu w b v bias x-src.target x| ≤ ε) ∧
      Architecture.network gelu w b v bias (0 : Fin (n+2) → ℝ)=src.target 0 ∧
      (Fintype.card (SphereHalfIndex q (n+1) × localGrid N T₀ H Z 2 2 ⊕ Fin 2):ℝ) ≤
        (q:ℝ)^(n+1)*(((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3))*N)+2 := by
  have hh := CenteredGELU.canonical_uniform_accuracy_and_width (n+1) src.extendedReal src.profile
    (scalarEllipse_pos src) src.profileBound_nonneg hε
    (fun θ => (scalar_profile_certificate src θ).1) (fun θ => (scalar_profile_certificate src θ).2)
    (profile_canonical_on_unit src) hT hH hsep hZ hhalo
    (canonicalAngular src.extendedReal) (centered_profile_agreement src)
    (fun x _ θ => centered_profile_even src x θ)
    (angularRate_pos src) (angularNorm_nonneg src)
    (fun x hx => (angular_energy_certificate src x hx).1)
    (fun x hx => (angular_energy_certificate src x hx).2)
  obtain ⟨w,b,v,bias,herror,hzero,hwidth⟩ := hh
  refine ⟨w,b,v,bias,?_,?_,hwidth⟩
  · intro x hx
    rw [← src.extendedReal_eq_target_unitBall hx]
    exact herror x hx
  · rwa [src.extendedReal_eq_target_unitBall (by simp)] at hzero

/-- Actual direction×halo-grid count for the original analytic target,
W_grid=O(|log epsilon|^(d−1+κ)). Extra anchor neurons are separately charged. -/
theorem general_grid_width_bigO
    (src : LocalAnalyticSource n) {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ T₀ H Z : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ) (r : ℕ)
    (hT : 1 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) :
    (fun ε : ℝ =>
      let U := angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))
      let V := localErrorConstant E S src.profileBound (scalarEllipse src) T₀ r
      let q := localAccuracyOrder U (angularRate src) (ε/2)
      let N := localAccuracyOrder V (localRate (scalarEllipse src) T₀) (ε/4)
      (Fintype.card (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ):ℝ))
      =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => |Real.log ε|^(((n+1:ℕ):ℝ)+max 1 (1/μ+1/ν))) := by
  have hU : 0 ≤ angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src)) :=
    mul_nonneg (angularNorm_nonneg src) (Real.sqrt_nonneg _)
  have hV := localErrorConstant_nonneg E S src.profileBound_nonneg (scalarEllipse_pos src) hT r
  have hh := centered_total_grid_epsilon_bigO (ν := ν) (μ := μ) (n+1) 0
    (lt_trans zero_lt_one hT) hH hZ hU hV (angularRate_pos src)
    (localRate_pos (scalarEllipse_pos src) hT)
  simpa only [Fintype.card_sum,Fintype.card_fin,Nat.add_zero] using hh

/-- Complete actual GELU width, including every halo and two anchor units:
W=O(|log epsilon|^d), with d=n+2. Dimension remains in all explicit constants. -/
theorem gelu_total_width_bigO (src : LocalAnalyticSource n) {T₀ H Z : ℝ}
    (hT : 1 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) :
    (fun ε : ℝ =>
      let U := angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))
      let V := localErrorConstant geluSpectral geluSpatial src.profileBound (scalarEllipse src) T₀ 2
      let q := localAccuracyOrder U (angularRate src) (ε/2)
      let N := localAccuracyOrder V (localRate (scalarEllipse src) T₀) (ε/4)
      (Fintype.card (SphereHalfIndex q (n+1) × localGrid N T₀ H Z 2 2 ⊕ Fin 2):ℝ))
      =O[𝓝[>] (0:ℝ)] (fun ε : ℝ => |Real.log ε|^((n+2:ℕ):ℝ)) := by
  have hU : 0 ≤ angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src)) :=
    mul_nonneg (angularNorm_nonneg src) (Real.sqrt_nonneg _)
  have hV := localErrorConstant_nonneg geluSpectral geluSpatial src.profileBound_nonneg
    (scalarEllipse_pos src) hT 2
  have hh := CenteredGELU.canonical_total_epsilon_bigO (n+1) (lt_trans zero_lt_one hT)
    hH hZ hU hV (angularRate_pos src) (localRate_pos (scalarEllipse_pos src) hT)
  simpa only [Nat.cast_add, Nat.cast_one, Nat.cast_ofNat, show (n:ℝ)+1+1=(n:ℝ)+2 by ring] using hh

/-- General raw-kernel/first-derivative activations give an ordinary MLP with
the grid neurons alone. This covers the general r=0 and r=1 classes, rather
than relying on an identity special to GELU or tanh. -/
theorem firstOrder_mlp_accuracy_and_width
    (src : LocalAnalyticSource n) {ε : ℝ} (hε : 0 < ε)
    {φ : ℝ → ℝ} {r : ℕ} (hr : r ≤ 1) (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ))
    {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0) (hkernel : ∀ u, iteratedDeriv r φ u = m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν) (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    let U := angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))
    let V := localErrorConstant E S src.profileBound (scalarEllipse src) T₀ r
    let q := localAccuracyOrder U (angularRate src) (ε/2)
    let N := localAccuracyOrder V (localRate (scalarEllipse src) T₀) (ε/4)
    ∃ (w : (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ) → Fin (n+2) → ℝ)
      (b v : (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ) → ℝ) (bias : ℝ),
      (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
        |Architecture.network φ w b v bias x-src.target x| ≤ ε) ∧
      Architecture.network φ w b v bias (0 : Fin (n+2) → ℝ)=src.target 0 ∧
      (Fintype.card (SphereHalfIndex q (n+1) × localGrid N T₀ H Z ν μ):ℝ) ≤
        (q:ℝ)^(n+1)*(((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^max 1 (1/μ+1/ν))*
          (N:ℝ)^max 1 (1/μ+1/ν)) := by
  let U := angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))
  let V := localErrorConstant E S src.profileBound (scalarEllipse src) T₀ r
  let q := localAccuracyOrder U (angularRate src) (ε/2)
  let N := localAccuracyOrder V (localRate (scalarEllipse src) T₀) (ε/4)
  have hh := general_kernel_accuracy_and_width src hε hφ K hm hkernel E S hT hH hsep hZ hhalo
  obtain ⟨w,b,v,bias,he⟩ := GenericArchitecture.centered_local_firstOrder (src.target 0) q
    (localAccuracyOrder_pos _ _ _) (n+1) N (localAccuracyOrder_pos _ _ _)
    φ hr src.profile K m T₀ H Z ν μ (lt_trans zero_lt_one hT)
  refine ⟨w,b,v,bias,?_,?_,hh.2⟩
  · intro x hx
    rw [he]
    exact hh.1 x hx
  · have hz := he (0 : EuclideanSpace ℝ (Fin (n+2)))
    rw [centeredLocalNetwork_zero] at hz
    exact hz

end QIRadon.KnownFunctionConstruction
