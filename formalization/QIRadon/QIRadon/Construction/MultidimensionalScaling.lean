import QIRadon.Construction.Multidimensional

noncomputable section
open Filter Asymptotics
open scoped Topology

namespace QIRadon.MultidimensionalScaling

open QIRadon.Gaussian QIRadon.ScalarRate QIRadon.ScalarScaling
open QIRadon.Multidimensional QIRadon.Kernel QIRadon.KernelRates
open MeasureTheory FourierTransform
open scoped Real SchwartzMap

/-- Actual integer Gauss–Jacobi order chosen from an angular exponential rate. -/
def angularOrder (L a : ℝ) : ℕ := ⌈L / a⌉₊

theorem angularOrder_bounds {L a : ℝ} (hL : 1 ≤ L) (ha : 0 < a) :
    0 < angularOrder L a ∧ L ≤ a * (angularOrder L a : ℝ) ∧
      (angularOrder L a : ℝ) ≤ (1 / a + 1) * L := by
  have hL0 : 0 < L := by linarith
  have hc := Nat.le_ceil (L / a)
  have hu := Nat.ceil_lt_add_one (div_pos hL0 ha).le
  have hpos : 0 < angularOrder L a := by
    by_contra hn
    have hz : angularOrder L a = 0 := by omega
    change L / a ≤ (angularOrder L a : ℝ) at hc
    rw [hz, Nat.cast_zero] at hc
    exact (not_le_of_gt (div_pos hL0 ha)) hc
  refine ⟨hpos, ?_, ?_⟩
  · have hh := (div_le_iff₀ ha).mp hc
    simpa only [angularOrder, mul_comm] using hh
  · have heq : L / a + 1 ≤ (1 / a + 1) * L := by
      have hid : (1 / a + 1) * L = L / a + L := by ring
      rw [hid]
      linarith
    exact hu.le.trans heq

/-- The two proved exponential rates meet the allocated tolerance with this
single explicit order. Optimal unequal allocation is handled separately. -/
theorem two_rate_accuracy {U V a ε : ℝ}
    (hU : 0 ≤ U) (hV : 0 ≤ V) (ha : 0 < a) (hε : 0 < ε) :
    let L := max 1 (Real.log ((U + V) / ε))
    U * Real.exp (-a * (angularOrder L a : ℝ)) + V * Real.exp (-L) ≤ ε := by
  dsimp only
  let L := max 1 (Real.log ((U + V) / ε))
  have ho := (angularOrder_bounds (le_max_left 1 (Real.log ((U + V) / ε))) ha).2.1
  have he : Real.exp (-a * (angularOrder L a : ℝ)) ≤ Real.exp (-L) := by
    apply Real.exp_le_exp.mpr
    dsimp [L] at *
    nlinarith
  have hh := log_accuracy_choice (add_nonneg hU hV) hε
  calc
    _ ≤ U * Real.exp (-L) + V * Real.exp (-L) :=
      add_le_add (mul_le_mul_of_nonneg_left he hU) le_rfl
    _ = (U + V) * Real.exp (-L) := by ring
    _ ≤ ε := hh

/-- Finite, fully counted tensor grid bound before Big-O notation.
The exponent is (d−1)+κ because d=n+1. This counts grid neurons; any nontrivial
integration-polynomial realization adds its separately charged anchor units. -/
theorem analytic_total_grid_width (n : ℕ) {T σ μ ν Z H L a : ℝ}
    (hT : 0 ≤ T) (hσ : 0 < σ) (hZ : 0 ≤ Z) (hH : 0 < H)
    (hL : 1 ≤ L) (ha : 0 < a) :
    let q := angularOrder L a
    let B := L / σ
    let γ := B / L ^ (1 / ν)
    let D := Z * L ^ (1 / μ) / γ
    let h := 1 / (H * B)
    let J := Finset.Icc (-(⌊(T + D) / h⌋₊ : ℤ)) (⌊(T + D) / h⌋₊ : ℤ)
    (Fintype.card (SphereHalfIndex q n × J) : ℝ) ≤
      ((1 / a + 1)^n * (2 * H * T / σ + 2 * H * Z + 1)) *
        L ^ ((n : ℝ) + max 1 (1 / μ + 1 / ν)) := by
  obtain ⟨hq, _, hqbound⟩ := angularOrder_bounds hL ha
  have hw := analytic_grid_width_cycles (μ := μ) (ν := ν) hT hσ hL hZ hH
  dsimp only at hw ⊢
  rw [Multidimensional.actual_grid_width _ hq]
  push_cast
  have hp := pow_le_pow_left₀ (Nat.cast_nonneg (angularOrder L a)) hqbound n
  rw [mul_pow] at hp
  have hA : 0 ≤ 2 * H * T / σ + 2 * H * Z + 1 := by positivity
  have hb := mul_le_mul hp hw (Nat.cast_nonneg _) (by positivity)
  apply hb.trans_eq
  rw [Real.rpow_add (lt_of_lt_of_le zero_lt_one hL), Real.rpow_natCast]
  ring

/-- The analytic grid complexity of the actual integer direction and center
choices. The fixed-d exponent is d−1+κ; no dimension-independent constant is claimed. -/
theorem analytic_total_grid_epsilon_bigO (n : ℕ) {T σ μ ν Z H a C : ℝ}
    (hT : 0 ≤ T) (hσ : 0 < σ) (hZ : 0 ≤ Z) (hH : 0 < H)
    (ha : 0 < a) (hC : 0 < C) :
    (fun ε : ℝ =>
      let L := max 1 (Real.log (C / ε))
      let q := angularOrder L a
      ((q^n * gridWidth T (L / σ) L μ ν Z H : ℕ) : ℝ))
      =O[𝓝[>] (0 : ℝ)]
      (fun ε : ℝ => |Real.log ε| ^ ((n : ℝ) + max 1 (1 / μ + 1 / ν))) := by
  let p : ℝ := (n : ℝ) + max 1 (1 / μ + 1 / ν)
  let A : ℝ := (1 / a + 1)^n * (2 * H * T / σ + 2 * H * Z + 1)
  have hA : 0 ≤ A := by dsimp [A]; positivity
  have hp : 0 ≤ p := by dsimp [p]; positivity
  apply IsBigO.of_bound (A * (|Real.log C| + 1)^p)
  filter_upwards [show Set.Ioc 0 (Real.exp (-1)) ∈ 𝓝[>] (0 : ℝ) from
    Ioc_mem_nhdsGT (Real.exp_pos _)] with ε hε
  let L := max 1 (Real.log (C / ε))
  have hL : 1 ≤ L := le_max_left _ _
  have hw := analytic_total_grid_width (μ := μ) (ν := ν) n hT hσ hZ hH hL ha
  dsimp only at hw
  rw [Multidimensional.actual_grid_width _ (angularOrder_bounds hL ha).1,
    symmetric_grid_card] at hw
  have hlo := logarithmic_order_bound hC hε.1 hε.2
  have hpow := Real.rpow_le_rpow (le_trans zero_le_one hL) hlo hp
  rw [Real.mul_rpow (by positivity : 0 ≤ |Real.log C| + 1) (abs_nonneg _)] at hpow
  rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _), Real.norm_eq_abs,
    abs_of_nonneg (Real.rpow_nonneg (abs_nonneg _) _)]
  change (↑((angularOrder L a)^n * gridWidth T (L / σ) L μ ν Z H) : ℝ) ≤ _
  apply hw.trans
  have hb := mul_le_mul_of_nonneg_left hpow hA
  simpa only [A, p, mul_assoc] using hb

/-- Ceiling an explicit algebraic direction cutoff costs only a bounded factor. -/
theorem algebraic_direction_order_bigO {U t : ℝ} (hU : 0 < U) (ht : 0 < t) :
    (fun ε : ℝ => (⌈algebraicCutoff U t ε⌉₊ : ℝ)) =O[𝓝[>] (0 : ℝ)]
      (fun ε : ℝ => ε ^ (-(1/t))) := by
  have hcut := (algebraicCutoff_eventually_eq hU ht).isBigO.trans
    (isBigO_const_mul_self (U ^ (1/t)) (fun ε : ℝ => ε ^ (-(1/t))) (𝓝[>] 0))
  have hceil : (fun ε : ℝ => (⌈algebraicCutoff U t ε⌉₊ : ℝ)) =O[𝓝[>] (0 : ℝ)]
      (fun ε : ℝ => algebraicCutoff U t ε) := by
    apply IsBigO.of_bound 2
    filter_upwards [(algebraicCutoff_tendsto hU ht).eventually_ge_atTop 1] with ε hε
    have hh := Nat.ceil_lt_add_one (show 0 ≤ algebraicCutoff U t ε by linarith)
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _), Real.norm_eq_abs,
      abs_of_nonneg (show 0 ≤ algebraicCutoff U t ε by linarith)]
    linarith
  exact hceil.trans hcut

/-- Algebraic total width for the actual integer grid. The exponent is the sum
of the angular and scalar costs, with n=d−1 and scalar smoothness s. -/
theorem algebraic_total_grid_epsilon_bigO (n : ℕ) {T μ ν Z H s t U V : ℝ}
    (hT : 0 ≤ T) (hμ : 0 < μ) (hν : 0 < ν) (hZ : 0 ≤ Z) (hH : 0 < H)
    (hs : 0 < s) (ht : 0 < t) (hU : 0 < U) (hV : 0 < V) :
    (fun ε : ℝ =>
      let q := ⌈algebraicCutoff U t ε⌉₊
      let B := algebraicCutoff V s ε
      ((q^n * gridWidth T B (max 1 ((s/2)*Real.log (1+B))) μ ν Z H : ℕ) : ℝ))
      =O[𝓝[>] (0 : ℝ)] (fun ε : ℝ => ε ^ (-((n:ℝ)/t + 1/s))) := by
  have hq := (algebraic_direction_order_bigO hU ht).pow n
  have hw := algebraic_gridWidth_epsilon_bigO hT hμ hν hZ hH hs hV
  have hh := hq.mul hw
  apply hh.congr'
  · filter_upwards with ε
    simp only [Nat.cast_mul, Nat.cast_pow]
  · filter_upwards [eventually_mem_nhdsWithin] with ε hε
    have hε0 : 0 < ε := hε
    rw [← Real.rpow_natCast (ε^(-(1/t))) n,
      ← Real.rpow_mul hε0.le, ← Real.rpow_add hε0]
    congr 1
    ring

/-- Joint accuracy and actual grid count for the explicit finite construction.
The only target-side hypotheses are the stated angular and scalar regularity classes. -/
theorem analytic_finite_accuracy_and_width
    (n : ℕ) {φ : ℝ → ℝ} {r : ℕ}
    (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m ν μ : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m * K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u : ℂ))) ν)
    (S : SpatialEnvelope K μ)
    (g : UnitSphere n → ℝ → ℝ) (pHat : UnitSphere n → ℝ → ℂ)
    {σ H T Z A ε : ℝ} (hσ : 0 < σ) (hε : 0 < ε) (hH : 2 ≤ H)
    (hsep : E.A + 4 ≤ E.a * (H / 2) ^ ν)
    (hZ : 0 ≤ Z) (hhalo : E.A + 3 ≤ (S.b / 2) * Z ^ μ)
    (hT : 0 < T) (hA : 0 ≤ A)
    (hp : ∀ θ, Integrable (pHat θ))
    (hW : ∀ θ, Integrable (fun ξ : ℝ => Real.exp (σ * |ξ|) * ‖pHat θ ξ‖))
    (hWbound : ∀ θ, (∫ ξ : ℝ, Real.exp (σ * |ξ|) * ‖pHat θ ξ‖) ≤ A)
    (hg : ∀ θ, ContDiffOn ℝ r (g θ) (Set.Icc (-T) T))
    (hprofile : ∀ θ s, s ∈ Set.Icc (-T) T →
      iteratedDeriv r (g θ) s = (synthesize (pHat θ) s).re)
    (x : EuclideanSpace ℝ (Fin (n + 1))) (hx : ‖x‖ ≤ T)
    (v : (k : ℕ) → PolynomialL2Space n k)
    (hF : Integrable (fun θ => g θ (inner ℝ θ.val x)) (Gaussian.normalizedSphere n))
    (heven : ∀ θ, g (sphereNeg n θ) (inner ℝ (sphereNeg n θ).val x) = g θ (inner ℝ θ.val x))
    (hrep : ∀ θ, g θ (inner ℝ θ.val x) = ∑' k, polynomialEvaluation n k θ (v k))
    {a Aang : ℝ} (ha : 0 < a) (hAang : 0 ≤ Aang)
    (hsum : Summable (fun k : ℕ => Real.exp (2 * a * k) * ‖v k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2 * a * k) * ‖v k‖^2) ≤ Aang^2) :
    let U := 2 * (Aang * (((n+1).factorial : ℝ) / (a/2)^(n+1) * Real.exp (a/2))) /
      (1 - Real.exp (-(a/2)))
    let V := A * envelopeConstant E S * T^r / (r.factorial : ℝ)
    let L := max 1 (Real.log ((U+V)/ε))
    let q := angularOrder L a
    let B := L / σ
    let γ := B / L^(1/ν)
    let h := 1/(H*B)
    let D := Z*L^(1/μ)/γ
    let J := Finset.Icc (-(⌊(T+D)/h⌋₊ : ℤ)) (⌊(T+D)/h⌋₊ : ℤ)
    (|finiteNetwork q (angularOrder_bounds (le_max_left _ _) ha).1 n φ g pHat K r m γ B h
      (by positivity) J x - reconstructed n g x| ≤ ε) ∧
    (Fintype.card (SphereHalfIndex q n × J) : ℝ) ≤
      ((1/a+1)^n * (2*H*T/σ + 2*H*Z + 1)) * L^((n:ℝ)+max 1 (1/μ+1/ν)) := by
  let U := 2 * (Aang * (((n+1).factorial : ℝ) / (a/2)^(n+1) * Real.exp (a/2))) /
    (1 - Real.exp (-(a/2)))
  let V := A * envelopeConstant E S * T^r / (r.factorial : ℝ)
  let L := max 1 (Real.log ((U+V)/ε))
  let q := angularOrder L a
  have hL : 1 ≤ L := le_max_left _ _
  have hq : 0 < q := (angularOrder_bounds hL ha).1
  have hU : 0 ≤ U := by
    have hd : 0 < 1 - Real.exp (-(a/2)) :=
      sub_pos.mpr (Real.exp_lt_one_iff.mpr (by linarith))
    dsimp [U]
    positivity
  have hV : 0 ≤ V := by
    have hc := (envelopeConstant_pos E S).le
    dsimp [V]
    positivity
  constructor
  · have hh := analytic_finite_error q hq n hφ K hm hkernel E S g pHat hσ hL hH
      hsep hZ hhalo hT hA hp hW hWbound hg hprofile x hx v hF heven hrep ha hAang hsum henergy
    apply hh.trans
    calc
      _ = U * Real.exp (-a*q) + V * Real.exp (-L) := by dsimp [U,V]; ring
      _ ≤ ε := two_rate_accuracy hU hV ha hε
  · exact analytic_total_grid_width n hT.le hσ hZ (by linarith) hL ha

end QIRadon.MultidimensionalScaling
