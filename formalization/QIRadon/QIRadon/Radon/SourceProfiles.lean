import QIRadon.Radon.LocalSource

/-! Actual bounded filtered profile families constructed from the original
analytic source, uniformly in directions and complex rotation words. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology
namespace QIRadon.Radon.LocalAnalyticSource
open QIRadon.Radon QIRadon.ComplexRotation
variable {n : ℕ} (src : LocalAnalyticSource n)

def jointProjection {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) : (Fin l → ℂ) × ℂ → ℂ :=
  rotatedCrossSection src.complexTarget pairs θ src.radiusDensity src.params.R₀ src.R

def jointHilbert {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) : (Fin l → ℂ) × ℂ → ℂ :=
  rotatedHilbert src.complexTarget pairs θ src.radiusDensity src.params.R₀ src.R src.params.S

/-- The parity choice is determined entirely by the ambient dimension. -/
def jointProfile {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) : (Fin l → ℂ) × ℂ → ℂ :=
  if Even (n+1) then
    rotatedEvenProfile src.complexTarget pairs θ src.radiusDensity src.params.R₀ src.R ((n+1)/2)
  else rotatedOddProfile src.complexTarget pairs θ src.radiusDensity src.params.R₀ src.R src.params.S ((n+1)/2)

theorem jointProjection_smooth {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) :
    ContDiffOn ℂ ∞ (src.jointProjection pairs θ)
      (rotationSectionDomain l src.params.S (4*src.params.h) src.params.η) := by
  have hh := src.params.h_pos
  exact rotatedCrossSection_smooth pairs θ src.params.S_pos.le (by positivity)
    src.params.projection_margin src.params.Q_lt_R₀ src.params.rotation_budget_pos.le src.bound_nonneg
    src.params.scalar_target_margin src.params.rotation_target_margin src.analytic src.bound
    src.density_smooth.continuous src.density_integrable.integrableOn

theorem jointProjection_bound {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {p : (Fin l → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain l src.params.S (4*src.params.h) src.params.η) :
    ‖src.jointProjection pairs θ p‖ ≤ src.projectionBound := by
  rw [src.projectionBound_eq θ]
  have hh := src.params.h_pos
  exact rotatedCrossSection_norm_le pairs θ src.params.S_pos.le (by positivity)
    src.params.projection_margin src.params.Q_lt_R₀ src.params.rotation_budget_pos.le src.bound_nonneg
    src.params.scalar_target_margin src.params.rotation_target_margin src.analytic src.bound
    src.density_smooth.continuous src.density_integrable.integrableOn src.density_nonneg src.density_normalized hp

theorem jointHilbert_smooth {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) :
    ContDiffOn ℂ ∞ (src.jointHilbert pairs θ)
      (rotationSectionDomain l src.params.T (2*src.params.h) src.params.η) := by
  have hh := src.params.h_pos
  exact rotatedHilbert_smooth pairs θ src.params.S_pos.le (by positivity)
    src.params.projection_margin src.params.Q_lt_R₀ src.params.rotation_budget_pos.le src.bound_nonneg
    src.params.scalar_tube_pos src.params.T_lt_S.le (by linarith) le_rfl src.params.hilbert_margin
    src.params.scalar_target_margin src.params.rotation_target_margin src.analytic src.bound
    src.density_integrable src.density_support src.density_smooth src.density_nonneg src.density_normalized

theorem jointHilbert_bound {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {p : (Fin l → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain l src.params.T (2*src.params.h) src.params.η) :
    ‖src.jointHilbert pairs θ p‖ ≤ src.hilbertBound := by
  unfold hilbertBound
  rw [src.projectionBound_eq θ]
  have hh := src.params.h_pos
  have hT := src.params.T_pos
  exact rotatedHilbert_norm_le pairs θ src.params.S_pos.le (by positivity)
    src.params.projection_margin src.params.Q_lt_R₀ src.params.rotation_budget_pos.le src.bound_nonneg
    src.params.scalar_tube_pos src.params.T_lt_S.le (by linarith) le_rfl (by positivity) src.params.hilbert_margin
    src.params.scalar_target_margin src.params.rotation_target_margin src.analytic src.bound
    src.density_integrable.integrableOn src.density_smooth.continuous src.density_nonneg src.density_normalized hp

theorem jointProfile_smooth {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) :
    ContDiffOn ℂ ∞ (src.jointProfile pairs θ)
      (rotationSectionDomain l src.params.T (2*src.params.h) src.params.η) := by
  unfold jointProfile
  split_ifs
  · exact (rotatedEvenProfile_smooth pairs θ src.radiusDensity src.params.R₀ src.R src.params.S
      (4*src.params.h) src.params.η ((n+1)/2) (src.jointProjection_smooth pairs θ)).mono
        (fun p hp => ⟨hp.1, scalarTube_mono src.params.T_lt_S.le (by linarith [src.params.h_pos]) hp.2⟩)
  · exact rotatedOddProfile_smooth pairs θ src.radiusDensity src.params.R₀ src.R src.params.S src.params.T
      (2*src.params.h) src.params.η ((n+1)/2) (src.jointHilbert_smooth pairs θ)

private theorem norm_even_factor (k : ℕ) (hk : 2*k=n+1) :
    ‖(surfaceArea (EuclideanSpace ℝ (Fin (n+2)))/2:ℂ)*((2*Real.pi*Complex.I)^(2*k))⁻¹‖ = src.filterFactor := by
  have ha := surfaceArea_pos (V := EuclideanSpace ℝ (Fin (n+2)))
  simp [filterFactor, norm_inv, norm_pow, Complex.norm_real,
    abs_of_pos ha, abs_of_pos Real.pi_pos, hk]

private theorem norm_odd_factor (k : ℕ) (hk : 2*k+1=n+1) :
    ‖(surfaceArea (EuclideanSpace ℝ (Fin (n+2)))/2:ℂ)*(Complex.I*((2*Real.pi*Complex.I)^(2*k+1))⁻¹)‖ = src.filterFactor := by
  have ha := surfaceArea_pos (V := EuclideanSpace ℝ (Fin (n+2)))
  simp [filterFactor, norm_inv, norm_pow, Complex.norm_real,
    abs_of_pos ha, abs_of_pos Real.pi_pos, hk]

/-- The actual source's canonical filtered profiles have one explicit bound
uniform in direction, word length, angles, and final scalar position. -/
theorem jointProfile_bound {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {p : (Fin l → ℂ) × ℂ} (hp : p ∈ rotationSectionDomain l src.params.T src.params.h src.params.η) :
    ‖src.jointProfile pairs θ p‖ ≤ src.profileBound := by
  have hh := src.params.h_pos
  have hhalf : (2*src.params.h)/2=src.params.h := by ring
  unfold jointProfile profileBound
  split_ifs with he
  · have hk : 2*((n+1)/2)=n+1 := by have := Nat.even_iff.mp he; omega
    have hsub : rotationSectionDomain l src.params.T (2*src.params.h) src.params.η ⊆
        rotationSectionDomain l src.params.S (4*src.params.h) src.params.η :=
      fun q hq => ⟨hq.1, scalarTube_mono src.params.T_lt_S.le (by linarith) hq.2⟩
    have h := rotatedEvenProfile_norm_le_of_bound pairs θ src.radiusDensity src.params.R₀ src.R
      (L := src.params.T) (δ := 2*src.params.h) (by positivity) ((n+1)/2)
      ((src.jointProjection_smooth pairs θ).mono hsub)
      (fun q hq => src.jointProjection_bound pairs θ (hsub hq))
      (by simpa only [hhalf] using hp)
    rw [src.norm_even_factor _ hk, hk, hhalf] at h
    exact h
  · have hk : 2*((n+1)/2)+1=n+1 := by have := Nat.even_iff.not.mp he; omega
    have h := rotatedOddProfile_norm_le_of_bound pairs θ src.radiusDensity src.params.R₀ src.R src.params.S
      (S := src.params.T) (ε := 2*src.params.h) (by positivity) ((n+1)/2)
      (src.jointHilbert_smooth pairs θ) (fun q hq => src.jointHilbert_bound pairs θ hq)
      (by simpa only [hhalf] using hp)
    rw [src.norm_odd_factor _ hk, hk, hhalf] at h
    exact h

/-- Every actual joint lift agrees with the canonical filtered projection at
real scalar offsets, including the endpoints of the final closed interval. -/
theorem jointProfile_eq_filtered {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {angles : Fin l → ℂ} (ha : angles ∈ rotationBudget l src.params.η)
    {s : ℝ} (hs : |s| ≤ src.params.T) :
    src.jointProfile pairs θ (angles,(s:ℂ)) =
      filteredRadonProfile (complexRadialExtension (rotationTarget src.complexTarget pairs angles)
        realEmbedding src.radiusDensity src.params.R₀ src.R) θ s := by
  have hh := src.params.h_pos
  unfold jointProfile
  split_ifs with he
  · have hk : 2*((n+1)/2)=n+1 := by have := Nat.even_iff.mp he; omega
    exact rotatedEvenProfile_eq_filtered pairs θ src.params.S_pos.le (by positivity)
      src.params.projection_margin src.params.Q_lt_R₀ src.params.rotation_budget_pos.le src.bound_nonneg
      src.params.scalar_tube_pos src.params.scalar_target_margin src.params.rotation_target_margin
      src.analytic src.bound src.density_integrable src.density_support src.density_smooth
      src.density_nonneg src.density_normalized ((n+1)/2) (by simpa using hk.symm)
      ha (hs.trans_lt src.params.T_lt_S)
  · have hk : 2*((n+1)/2)+1=n+1 := by have := Nat.even_iff.not.mp he; omega
    exact rotatedOddProfile_eq_filtered pairs θ src.params.S_pos.le (by positivity)
      src.params.projection_margin src.params.Q_lt_R₀ src.params.rotation_budget_pos.le src.bound_nonneg
      src.params.scalar_tube_pos (by positivity) src.params.T_lt_S.le (by linarith) le_rfl src.params.hilbert_margin
      src.params.scalar_target_margin src.params.rotation_target_margin src.analytic src.bound
      src.density_integrable src.density_support src.density_smooth src.density_nonneg src.density_normalized
      ((n+1)/2) (by simpa using hk.symm) ha hs

/-- Real angle parameters reproduce the original canonical directional profile;
the only transformation is the actual real isometry of the direction. -/
theorem jointProfile_real_agreement {l : ℕ} (pairs : Fin l → Fin (n+2) × Fin (n+2))
    (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) (angles : Fin l → ℝ)
    (ha : (fun i => (angles i:ℂ)) ∈ rotationBudget l src.params.η)
    {s : ℝ} (hs : |s| ≤ src.params.T) :
    src.jointProfile pairs θ ((fun i => (angles i:ℂ)),(s:ℂ)) =
      filteredRadonProfile src.extendedComplex (isometryDirection (totalRealRotationWord pairs angles) θ) s := by
  rw [src.jointProfile_eq_filtered pairs θ ha hs]
  have hf : (complexRadialExtension (rotationTarget src.complexTarget pairs (fun i => (angles i:ℂ)))
      realEmbedding src.radiusDensity src.params.R₀ src.R) =
      (fun x => src.extendedComplex (totalRealRotationWord pairs angles x)) := by
    funext x
    exact complexRadialExtension_rotation_real_total src.complexTarget pairs angles src.radiusDensity src.params.R₀ src.R x
  rw [hf, filteredRadonProfile_comp_isometry src.extendedComplex.integrable]

/-- The scalar continuation used by the one-dimensional finite row is the
zero-word member of the same actual rotation family. -/
def profile (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) (z : ℂ) : ℂ :=
  src.jointProfile (fun i : Fin 0 => Fin.elim0 i) θ (0,z)

theorem profile_analytic (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) :
    AnalyticOnNhd ℂ (src.profile θ) (scalarTube src.params.T (2*src.params.h)) := by
  have hz : (0:Fin 0 → ℂ) ∈ rotationBudget 0 src.params.η := by
    simpa only [rotationBudget, mem_ofPred_eq, Finset.univ_eq_empty, Finset.sum_empty] using src.params.rotation_budget_pos
  have hs : ContDiffOn ℂ ∞ (src.profile θ) (scalarTube src.params.T (2*src.params.h)) :=
    (src.jointProfile_smooth (fun i : Fin 0 => Fin.elim0 i) θ).comp (by fun_prop)
      (fun z hz' => ⟨hz,hz'⟩)
  exact (hs.differentiableOn (by simp)).analyticOnNhd (isOpen_scalarTube _ _)

theorem profile_bound (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {z : ℂ} (hz : z ∈ scalarTube src.params.T src.params.h) :
    ‖src.profile θ z‖ ≤ src.profileBound := by
  apply src.jointProfile_bound
  refine ⟨?_,hz⟩
  simpa only [Finset.univ_eq_empty, Finset.sum_empty] using src.params.rotation_budget_pos

theorem profile_eq_filtered (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {s : ℝ} (hs : |s| ≤ src.params.T) :
    src.profile θ (s:ℂ) = filteredRadonProfile src.extendedComplex θ s := by
  have hz : (0:Fin 0 → ℂ) ∈ rotationBudget 0 src.params.η := by
    simpa only [rotationBudget, mem_ofPred_eq, Finset.univ_eq_empty, Finset.sum_empty] using src.params.rotation_budget_pos
  have h := src.jointProfile_eq_filtered (fun i : Fin 0 => Fin.elim0 i) θ hz hs
  have hf : complexRadialExtension (rotationTarget src.complexTarget (fun i : Fin 0 => Fin.elim0 i) 0)
      realEmbedding src.radiusDensity src.params.R₀ src.R = (src.extendedComplex : EuclideanSpace ℝ (Fin (n+2)) → ℂ) := by
    funext x
    simp [complexRadialExtension, rotationTarget, rotationWord, indexedWord, extendedComplex_apply]
  rw [hf] at h
  exact h

/-- The supplied local continuation agrees with the real canonical profile on
the complete construction interval, not just at isolated sample points. -/
theorem profile_eq_realRidge (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {s : ℝ} (hs : |s| ≤ src.params.T) :
    src.profile θ (s:ℂ) = (realRidgeProfile src.extendedReal θ s:ℂ) := by
  rw [src.profile_eq_filtered θ hs, ← src.complexify_extendedReal]
  exact (real_ridge_eq_filtered_radon src.extendedReal θ s).symm

end QIRadon.Radon.LocalAnalyticSource
