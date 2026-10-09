import QIRadon.Allocation.AllocationPowerSolver
import QIRadon.Construction.CenteredConstruction

/-! The general-power allocation solver attached to the actual finite rows.
Every error constant and cost coefficient is derived from the local analytic
construction. The cost comparison concerns the continuous certificate; neither
integer-global nor arbitrary-network optimality is asserted. -/
noncomputable section
open MeasureTheory FourierTransform
open scoped Real SchwartzMap
namespace QIRadon.AllocatedConstruction
open QIRadon.KernelRates QIRadon.LocalProfileRate QIRadon.ProfilePreparation
open QIRadon.AllocationPower

/-- Finite positive-weight local analytic profiles with independently specified
ellipse widths and magnitude bounds. Bounds can always be chosen positive,
including for a zero profile. No network-error assumption is included. -/
structure AnalyticRows (ι : Type*) where
  α : ι → ℝ
  δ : ι → ℝ
  M : ι → ℝ
  g : ι → ℂ → ℂ
  weight_pos : ∀ i, 0 < α i
  ellipse_pos : ∀ i, 0 < δ i
  magnitude_pos : ∀ i, 0 < M i
  analytic : ∀ i, AnalyticOnNhd ℂ (g i) (bernsteinEllipse (δ i))
  bound : ∀ i z, z ∈ bernsteinEllipse (δ i) → ‖g i z‖ ≤ M i

/-- The exact prefactor from the actual local halo-grid count. -/
def rowCost (T₀ H Z ν μ : ℝ) : ℝ :=
  (2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^max 1 (1/μ+1/ν)

theorem rowCost_pos {T₀ H Z ν μ : ℝ} (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) :
    0 < rowCost T₀ H Z ν μ := by
  have hg := preparationGrowth_pos hT
  unfold rowCost
  positivity

theorem localErrorConstant_pos {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ M δ T₀ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ)
    (hM : 0 < M) (hδ : 0 < δ) (hT : 1 < T₀) (r : ℕ) :
    0 < localErrorConstant E S M δ T₀ r := by
  have hd : 0 < 1-Real.exp (-δ) := sub_pos.mpr (Real.exp_lt_one_iff.mpr (by linarith))
  have hw := windowRate_pos hT
  have hc := localMomentConstant_nonneg (T₀ := T₀) hM.le r
  have he := ScalarRate.envelopeConstant_pos E S
  unfold localErrorConstant interpolationConstant windowConstant
  positivity

/-- Actual row construction data for the solver. The factor two in D pays for
centering each scalar row. Angular weights are in D, not in neuron cost c. -/
def rowSpokeData {ι : Type*} (rows : AnalyticRows ι)
    {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ)
    (r : ℕ) (T₀ H Z : ℝ) (hT : 1 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) : SpokeData ι where
  κ := max 1 (1/μ+1/ν)
  c _ := rowCost T₀ H Z ν μ
  a i := localRate (rows.δ i) T₀
  D i := 2*rows.α i*localErrorConstant E S (rows.M i) (rows.δ i) T₀ r
  base _ := 1
  power_ge_one := le_max_left _ _
  cost_pos _ := rowCost_pos (lt_trans zero_lt_one hT) hH hZ
  rate_pos i := localRate_pos (rows.ellipse_pos i) hT
  error_pos i := mul_pos (mul_pos (by norm_num) (rows.weight_pos i))
    (localErrorConstant_pos E S (rows.magnitude_pos i) (rows.ellipse_pos i) hT r)
  base_ge_one _ := le_rfl

variable {ι : Type*} [Fintype ι]

/-- Each row keeps its own integer preparation order and complete halo grid. -/
abbrev RowIndex (N : ι → ℕ) (T₀ H Z ν μ : ℝ) :=
  (i : ι) × ↥(localGrid (N i) T₀ H Z ν μ)

def rowNetwork (rows : AnalyticRows ι) (N : ι → ℕ) (hN : ∀ i, 0 < N i)
    (φ : ℝ → ℝ) (K : 𝓢(ℝ, ℝ)) (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀)
    (s : ι → ℝ) : ℝ :=
  ∑ i, rows.α i *
    (localActivationNetwork φ (fun t => (rows.g i (t:ℂ)).re) K r (N i)
      m T₀ H Z ν μ (hN i) hT (s i) -
    localActivationNetwork φ (fun t => (rows.g i (t:ℂ)).re) K r (N i)
      m T₀ H Z ν μ (hN i) hT 0)

def rowTarget (rows : AnalyticRows ι) (s : ι → ℝ) : ℝ :=
  ∑ i, rows.α i*((rows.g i (s i:ℂ)).re-(rows.g i 0).re)

/-- The weighted centered-row error is derived from actual sampled row
constructions. The displayed sum is an explicit numeric certificate, never an
assumed approximation bound. No normalization of weights is needed here. -/
theorem actual_rows_error
    (rows : AnalyticRows ι) (N : ι → ℕ) (hN : ∀ i, 0 < N i)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ))
    {m T₀ H Z ν μ : ℝ} (hm : m ≠ 0) (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν) (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ)
    (s : ι → ℝ) (hs : ∀ i, |s i| ≤ 1) :
    |rowNetwork rows N hN φ K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) s-rowTarget rows s| ≤
      ∑ i, (2*rows.α i*localErrorConstant E S (rows.M i) (rows.δ i) T₀ r)*
        Real.exp (-localRate (rows.δ i) T₀*(N i:ℝ)) := by
  let Q := fun i t => localActivationNetwork φ (fun u => (rows.g i (u:ℂ)).re) K r (N i)
    m T₀ H Z ν μ (hN i) (lt_trans zero_lt_one hT) t
  have hrow : ∀ i, |(Q i (s i)-Q i 0)-((rows.g i (s i:ℂ)).re-(rows.g i 0).re)| ≤
      2*(localErrorConstant E S (rows.M i) (rows.δ i) T₀ r*
        Real.exp (-localRate (rows.δ i) T₀*(N i:ℝ))) := by
    intro i
    have hx := local_analytic_activation_error (rows.ellipse_pos i) (rows.analytic i) (rows.bound i)
      hφ K hm hkernel E S hT (hN i) hH hsep hZ hhalo (s i) (abs_le.mp (hs i))
    have hz := local_analytic_activation_error (rows.ellipse_pos i) (rows.analytic i) (rows.bound i)
      hφ K hm hkernel E S hT (hN i) hH hsep hZ hhalo 0 (by constructor <;> norm_num)
    exact CenteredConstruction.centered_difference_bound hx hz
  have hid : rowNetwork rows N hN φ K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) s-rowTarget rows s =
      ∑ i, rows.α i*((Q i (s i)-Q i 0)-((rows.g i (s i:ℂ)).re-(rows.g i 0).re)) := by
    simp only [rowNetwork,rowTarget,mul_sub,Finset.sum_sub_distrib,Q]
  rw [hid]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  apply Finset.sum_le_sum
  intro i _
  rw [abs_mul,abs_of_pos (rows.weight_pos i)]
  exact (mul_le_mul_of_nonneg_left (hrow i) (rows.weight_pos i).le).trans_eq (by ring)

/-- The sum of the actual, possibly unequal halo grids obeys its proved
power-cost certificate. Integration-anchor neurons remain separately charged. -/
theorem actual_rows_width (N : ι → ℕ) (hN : ∀ i, 0 < N i)
    {T₀ H Z ν μ : ℝ} (hT : 0 < T₀) (hH : 0 < H) (hZ : 0 ≤ Z) :
    (Fintype.card (RowIndex N T₀ H Z ν μ):ℝ) ≤
      ∑ i, rowCost T₀ H Z ν μ*(N i:ℝ)^max 1 (1/μ+1/ν) := by
  rw [Fintype.card_sigma]
  simp only [Fintype.card_coe, Nat.cast_sum]
  exact Finset.sum_le_sum (fun i _ => local_analytic_grid_width hT (hN i) hH hZ)

/-- Concrete continuous roots, with the baseline used if it already meets the
budget. The solver's common multiplier is constructed, not assumed. -/
def optimalResolution [Nonempty ι] (p : SpokeData ι) (ε : ℝ) (hε : 0 < ε) : ι → ℝ :=
  if hb : ε < p.baselineError then p.resolution (p.solveLogMultiplier ε hε hb) else p.base

theorem optimalResolution_ge_base [Nonempty ι] (p : SpokeData ι) {ε : ℝ} (hε : 0 < ε) (i : ι) :
    p.base i ≤ optimalResolution p ε hε i := by
  unfold optimalResolution
  split_ifs
  · exact p.resolution_ge_base _ i
  · exact le_rfl

theorem optimalResolution_error [Nonempty ι] (p : SpokeData ι) {ε : ℝ} (hε : 0 < ε) :
    (∑ i, p.D i*Real.exp (-p.a i*optimalResolution p ε hε i)) ≤ ε := by
  unfold optimalResolution
  split_ifs with hb
  · exact (p.solveLogMultiplier_spec hε hb).2.le
  · exact le_of_not_gt hb

theorem optimalResolution_min [Nonempty ι] (p : SpokeData ι) {ε : ℝ} (hε : 0 < ε)
    (z : ι → ℝ) (hz : ∀ i, p.base i ≤ z i)
    (he : (∑ i, p.D i*Real.exp (-p.a i*z i)) ≤ ε) :
    (∑ i, p.c i*(optimalResolution p ε hε i)^p.κ) ≤ ∑ i, p.c i*(z i)^p.κ := by
  unfold optimalResolution
  split_ifs with hb
  · exact p.solver_global_min hε hb z hz he
  · exact p.baseline_global_min z hz

@[simp] theorem rowNetwork_zero
    (rows : AnalyticRows ι) (N : ι → ℕ) (hN : ∀ i, 0 < N i)
    (φ : ℝ → ℝ) (K : 𝓢(ℝ, ℝ)) (r : ℕ) (m T₀ H Z ν μ : ℝ) (hT : 0 < T₀) :
    rowNetwork rows N hN φ K r m T₀ H Z ν μ hT 0 = 0 := by
  simp [rowNetwork]

/-- Full integration of the constructed optimizer with actual sampled rows.
Upward rounding preserves the requested centered weighted error budget, and the
actual total halo-grid count is at most 2^kappa times the optimal continuous
cost certificate. This does not assert exact discrete-global optimality. -/
theorem allocated_rows_certificate [Nonempty ι]
    (rows : AnalyticRows ι) {ε : ℝ} (hε : 0 < ε)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ))
    {m T₀ H Z ν μ : ℝ} (hm : m ≠ 0) (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν) (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    let p := rowSpokeData rows E S r T₀ H Z hT (by linarith) hZ
    let y := optimalResolution p ε hε
    let N := fun i => ⌈y i⌉₊
    let hN : ∀ i, 0 < N i := fun i => Nat.ceil_pos.mpr
      (zero_lt_one.trans_le ((p.base_ge_one i).trans (optimalResolution_ge_base p hε i)))
    (∀ s : ι → ℝ, (∀ i, |s i| ≤ 1) →
      |rowNetwork rows N hN φ K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) s-rowTarget rows s| ≤ ε) ∧
    ((Fintype.card (RowIndex N T₀ H Z ν μ):ℝ) ≤
      (2:ℝ)^p.κ*(∑ i, p.c i*(y i)^p.κ)) ∧
    (∀ z : ι → ℝ, (∀ i, p.base i ≤ z i) →
      (∑ i, p.D i*Real.exp (-p.a i*z i)) ≤ ε →
      (∑ i, p.c i*(y i)^p.κ) ≤ ∑ i, p.c i*(z i)^p.κ) := by
  let p := rowSpokeData rows E S r T₀ H Z hT (by linarith) hZ
  let y := optimalResolution p ε hε
  let N := fun i => ⌈y i⌉₊
  have hy : ∀ i, 1 ≤ y i := fun i => (p.base_ge_one i).trans (optimalResolution_ge_base p hε i)
  have hN : ∀ i, 0 < N i := fun i => Nat.ceil_pos.mpr (zero_lt_one.trans_le (hy i))
  have hbudget := optimalResolution_error p hε
  refine ⟨?_,?_,?_⟩
  · intro s hs
    have hh := actual_rows_error rows N hN hφ K hm hkernel E S hT hH hsep hZ hhalo s hs
    have hr := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
      Allocation.exponential_error_rounding (n := y i) (p.error_pos i).le (p.rate_pos i).le)
    change |rowNetwork rows N hN φ K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) s-rowTarget rows s| ≤
      ∑ i, p.D i*Real.exp (-p.a i*(N i:ℝ)) at hh
    exact hh.trans (hr.trans hbudget)
  · have hc := actual_rows_width (ν := ν) (μ := μ) N hN (lt_trans zero_lt_one hT) (by linarith : 0 < H) hZ
    have hr := power_cost_rounding p.power_pos.le (fun i => (p.cost_pos i).le) hy
    change (Fintype.card (RowIndex N T₀ H Z ν μ):ℝ) ≤ ∑ i, p.c i*(N i:ℝ)^p.κ at hc
    exact hc.trans hr
  · intro z hz he
    exact optimalResolution_min p hε z hz he

/-- Explicit rounding never worsens the actual per-row numeric certificates.
This is useful independently of the particular global allocation choice. -/
theorem allocated_rounding_error_nonincrease [Nonempty ι]
    (p : SpokeData ι) {ε : ℝ} (hε : 0 < ε) :
    (∑ i, p.D i*Real.exp (-p.a i*(⌈optimalResolution p ε hε i⌉₊:ℝ))) ≤
      ∑ i, p.D i*Real.exp (-p.a i*optimalResolution p ε hε i) := by
  exact Finset.sum_le_sum (fun i _ =>
    Allocation.exponential_error_rounding (p.error_pos i).le (p.rate_pos i).le)

/-- Reading the rows at any unit directions gives the same uniform scalar
budget on the input unit ball; weights and row orders are fixed beforehand. -/
theorem allocated_projection_certificate [Nonempty ι]
    (rows : AnalyticRows ι) (d : ℕ) (θ : ι → EuclideanSpace ℝ (Fin d))
    (hθ : ∀ i, ‖θ i‖=1) {ε : ℝ} (hε : 0 < ε)
    {φ : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ))
    {m T₀ H Z ν μ : ℝ} (hm : m ≠ 0) (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν) (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    let p := rowSpokeData rows E S r T₀ H Z hT (by linarith) hZ
    let y := optimalResolution p ε hε
    let N := fun i => ⌈y i⌉₊
    let hN : ∀ i, 0 < N i := fun i => Nat.ceil_pos.mpr
      (zero_lt_one.trans_le ((p.base_ge_one i).trans (optimalResolution_ge_base p hε i)))
    ∀ x : EuclideanSpace ℝ (Fin d), ‖x‖ ≤ 1 →
      |rowNetwork rows N hN φ K r m T₀ H Z ν μ (lt_trans zero_lt_one hT) (fun i => inner ℝ (θ i) x)-
        rowTarget rows (fun i => inner ℝ (θ i) x)| ≤ ε := by
  intro p y N hN x hx
  apply (allocated_rows_certificate rows hε hφ K hm hkernel E S hT hH hsep hZ hhalo).1
  intro i
  have hh := norm_inner_le_norm (𝕜 := ℝ) (θ i) x
  rw [Real.norm_eq_abs, hθ i, one_mul] at hh
  exact hh.trans hx

end QIRadon.AllocatedConstruction
