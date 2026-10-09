import QIRadon.Scalar.Preparation.FilteredSamples
import QIRadon.Scalar.LocalProfileRate
import QIRadon.Construction.AlgebraicConstruction

/-! Actual local finite-smooth construction from real samples. A stable filtered
polynomial preserves the supplied polynomial-approximation exponent. Only this
computed polynomial is analytically continued for Fourier preparation; the
original nonanalytic target is never evaluated at a complex argument. -/
noncomputable section
open MeasureTheory FourierTransform Polynomial
open scoped Real SchwartzMap Topology
namespace QIRadon.LocalAlgebraic
open QIRadon.ProfilePreparation QIRadon.ProfilePreparation.FilteredSamples
open QIRadon.LocalProfileRate QIRadon.Kernel QIRadon.KernelRates

/-- Actual activation row computed from the stable 2N-sample polynomial.
The second interpolation is polynomial reproduction, not further access to f. -/
def localAlgebraicNetwork (φ f : ℝ → ℝ) (K : 𝓢(ℝ, ℝ))
    (r N : ℕ) (m T₀ H Z ν μ : ℝ) (hN : 0 < N) (hT : 0 < T₀) : ℝ → ℝ :=
  localActivationNetwork φ (filteredSamples N f).eval K r (2*N)
    m T₀ H Z ν μ (by omega) hT

/-- No evaluations of the original target outside the prescribed 2N real
samples enter any readout, bandwidth, or integration anchor. -/
theorem localAlgebraicNetwork_congr {φ f g : ℝ → ℝ} (K : 𝓢(ℝ, ℝ))
    (r N : ℕ) (m T₀ H Z ν μ : ℝ) (hN : 0 < N) (hT : 0 < T₀)
    (hfg : ∀ j < 2*N, f (sampleNode (2*N) j)=g (sampleNode (2*N) j)) :
    localAlgebraicNetwork φ f K r N m T₀ H Z ν μ hN hT =
      localAlgebraicNetwork φ g K r N m T₀ H Z ν μ hN hT := by
  unfold localAlgebraicNetwork
  rw [filteredSamples_congr hfg]

/-- Re-interpolation in the existing Fourier preparation reproduces the
already computed degree-2N polynomial exactly. -/
theorem filtered_reinterpolation (N : ℕ) (f : ℝ → ℝ) :
    samplePolynomial (2*N+1) (filteredSamples N f).eval = filteredSamples N f := by
  apply samplePolynomial_reproduces (by omega)
  exact (filteredSamples_degree_le N f).trans_lt (by exact_mod_cast Nat.lt_succ_self (2*N))

/-- Actual preparation on real inputs, with no implicit unknown function in it. -/
theorem realPrepared_filtered (N : ℕ) (f : ℝ → ℝ) (T₀ x : ℝ) :
    realPrepared (2*N) (filteredSamples N f).eval T₀ x =
      (filteredSamples N f).eval x*flatWindow (2*N) ((x/T₀)^2) := by
  unfold realPrepared
  rw [preparedProfile_ofReal, Complex.ofReal_re, filtered_reinterpolation]

/-- Actual finite row error, retaining all three contributions before reducing
them to an algebraic rate: polynomial approximation, finite window, and the
kernel cutoff/alias/halo construction. -/
theorem local_algebraic_error_terms
    {φ f : ℝ → ℝ} {r N : ℕ} (hφ : ContDiff ℝ r φ) (hN : 0 < N)
    (K : 𝓢(ℝ, ℝ)) {m ν μ A M s T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν)
    (S : SpatialEnvelope K μ) (hA : 0 ≤ A) (hM : 0 ≤ M)
    (hf : ∀ x : ℝ, |x| ≤ 1 → |f x| ≤ M)
    (happrox : ∃ P : ℝ[X], P.degree < N ∧
      ∀ x : ℝ, |x| ≤ 1 → |f x-P.eval x| ≤ A*(N:ℝ)^(-s))
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    ∀ x : ℝ, |x| ≤ 1 →
      |localAlgebraicNetwork φ f K r N m T₀ H Z ν μ hN (lt_trans zero_lt_one hT) x-f x| ≤
      4*A*(N:ℝ)^(-s) + 3*M*Real.exp (-(2*(N:ℝ))*windowRate T₀) +
      (localMomentConstant (3*M) T₀ r*ScalarRate.envelopeConstant E S/(r.factorial:ℝ))*
        Real.exp (-4*(N:ℝ)) := by
  have hT0 : 0 < T₀ := lt_trans zero_lt_one hT
  have hpoly : ∀ x : ℝ, |x| ≤ 1 → |(filteredSamples N f).eval x| ≤ 3*M := by
    intro x hx
    exact filteredSamples_bound hN hM (fun j _ => hf _ (sampleNode_abs_le_one _ _)) hx
  have herr := filteredSamples_algebraic_error hN hA happrox
  have hsample : ∀ j < 2*N+1,
      |(filteredSamples N f).eval (sampleNode (2*N+1) j)| ≤ 3*M :=
    fun j _ => hpoly _ (sampleNode_abs_le_one _ _)
  have hnet := prepared_activation_geometric_error (f := (filteredSamples N f).eval) hφ K hm hkernel E S
    (show 0 ≤ 3*M by positivity) hT0 (show 0 < 2*N by omega) hsample hH hsep hZ hhalo
  intro x hx
  have hu : 0 ≤ (x/T₀)^2 := sq_nonneg _
  have huv : (x/T₀)^2 ≤ (1/T₀)^2 := by
    rw [div_pow, div_pow, one_pow]
    apply (div_le_div_iff_of_pos_right (sq_pos_of_pos hT0)).mpr
    have hh := mul_self_le_mul_self (abs_nonneg x) hx
    nlinarith [sq_abs x]
  have hv : 0 < (1/T₀)^2 := sq_pos_of_pos (div_pos zero_lt_one hT0)
  have hv1 : (1/T₀)^2 < 1 := by
    have hh := (div_lt_one hT0).mpr hT
    have hh0 := div_pos zero_lt_one hT0
    nlinarith
  have hprep := flatWindow_preparation_error (2*N) (herr x hx) (hpoly x hx) hu huv hv hv1
  rw [← realPrepared_filtered] at hprep
  have hrow := hnet x (abs_le.mp hx)
  change |localAlgebraicNetwork φ f K r N m T₀ H Z ν μ hN hT0 x -
    realPrepared (2*N) (filteredSamples N f).eval T₀ x| ≤ _ at hrow
  have htri := abs_sub_le (localAlgebraicNetwork φ f K r N m T₀ H Z ν μ hN hT0 x)
    (realPrepared (2*N) (filteredSamples N f).eval T₀ x) (f x)
  rw [abs_sub_comm (realPrepared (2*N) (filteredSamples N f).eval T₀ x)] at htri
  change _ ≤ 4*A*(N:ℝ)^(-s) + 3*M*Real.exp (-((2*N:ℕ):ℝ)*windowRate T₀) at hprep
  norm_num only [Nat.cast_mul, Nat.cast_ofNat] at hprep hrow
  have hexp : (-2:ℝ)*(2*(N:ℝ)) = -4*(N:ℝ) := by ring
  rw [hexp] at hrow
  linarith

/-- Explicit cost of absorbing an exponential remainder into the desired
algebraic rate. The integer ceiling affects only this fixed constant. -/
def expPowerConstant (b s : ℝ) : ℝ := (⌈s⌉₊.factorial:ℝ)/b^⌈s⌉₊

theorem exponential_le_algebraic {b s x : ℝ} (hb : 0 < b) (hx : 1 ≤ x) :
    Real.exp (-b*x) ≤ expPowerConstant b s*x^(-s) := by
  let k := ⌈s⌉₊
  have hx0 : 0 < x := lt_of_lt_of_le zero_lt_one hx
  have hkfac : (0:ℝ) < k.factorial := by positivity
  have hpow := Real.pow_div_factorial_le_exp (b*x) (by positivity) k
  have hp := (div_le_iff₀ hkfac).mp hpow
  have hm := mul_le_mul_of_nonneg_right hp (Real.exp_pos (-b*x)).le
  have he : Real.exp (b*x)*Real.exp (-b*x) = 1 := by rw [← Real.exp_add]; simp
  have hbase : (b*x)^k*Real.exp (-b*x) ≤ (k.factorial:ℝ) := by
    calc
      _ ≤ Real.exp (b*x)*(k.factorial:ℝ)*Real.exp (-b*x) := hm
      _ = (k.factorial:ℝ)*(Real.exp (b*x)*Real.exp (-b*x)) := by ring
      _ = _ := by rw [he,mul_one]
  have hd : Real.exp (-b*x) ≤ (k.factorial:ℝ)/(b*x)^k := by
    apply (le_div_iff₀ (pow_pos (mul_pos hb hx0) k)).mpr
    simpa only [mul_comm] using hbase
  have hid : (k.factorial:ℝ)/(b*x)^k = expPowerConstant b s*x^(-(k:ℝ)) := by
    rw [Real.rpow_neg hx0.le, Real.rpow_natCast, mul_pow]
    unfold expPowerConstant
    dsimp [k]
    ring
  rw [hid] at hd
  apply hd.trans
  apply mul_le_mul_of_nonneg_left
    (Real.rpow_le_rpow_of_exponent_le hx (neg_le_neg (Nat.le_ceil s)))
  unfold expPowerConstant
  positivity

def localAlgebraicConstant {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ)
    (A M T₀ s : ℝ) (r : ℕ) : ℝ :=
  4*A + 3*M*expPowerConstant (2*windowRate T₀) s +
    (localMomentConstant (3*M) T₀ r*ScalarRate.envelopeConstant E S/(r.factorial:ℝ))*
      expPowerConstant 4 s

theorem localAlgebraicConstant_nonneg {Khat : ℝ → ℂ} {K : ℝ → ℝ} {ν μ A M T₀ s : ℝ}
    (E : SpectralEnvelope Khat ν) (S : SpatialEnvelope K μ)
    (hA : 0 ≤ A) (hM : 0 ≤ M) (hT : 1 < T₀) (r : ℕ) :
    0 ≤ localAlgebraicConstant E S A M T₀ s r := by
  have hw := windowRate_pos hT
  have hm := localMomentConstant_nonneg (T₀ := T₀) (show 0 ≤ 3*M by positivity) r
  have he := ScalarRate.envelopeConstant_pos E S
  unfold localAlgebraicConstant expPowerConstant
  positivity

/-- The actual real-sample activation construction preserves the local
polynomial-approximation exponent s, with no loss from interpolation. -/
theorem local_algebraic_error
    {φ f : ℝ → ℝ} {r N : ℕ} (hφ : ContDiff ℝ r φ) (hN : 0 < N)
    (K : 𝓢(ℝ, ℝ)) {m ν μ A M s T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν)
    (S : SpatialEnvelope K μ) (hA : 0 ≤ A) (hM : 0 ≤ M)
    (hf : ∀ x : ℝ, |x| ≤ 1 → |f x| ≤ M)
    (happrox : ∃ P : ℝ[X], P.degree < N ∧
      ∀ x : ℝ, |x| ≤ 1 → |f x-P.eval x| ≤ A*(N:ℝ)^(-s))
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    ∀ x : ℝ, |x| ≤ 1 →
      |localAlgebraicNetwork φ f K r N m T₀ H Z ν μ hN (lt_trans zero_lt_one hT) x-f x| ≤
        localAlgebraicConstant E S A M T₀ s r*(N:ℝ)^(-s) := by
  intro x hx
  have ht := local_algebraic_error_terms hφ hN K hm hkernel E S hA hM hf happrox
    hT hH hsep hZ hhalo x hx
  have hN1 : 1 ≤ (N:ℝ) := by exact_mod_cast hN
  have hw := exponential_le_algebraic (s := s) (mul_pos (by norm_num : (0:ℝ)<2) (windowRate_pos hT)) hN1
  have he := exponential_le_algebraic (s := s) (by norm_num : (0:ℝ)<4) hN1
  have hwid : -(2*(N:ℝ))*windowRate T₀ = -(2*windowRate T₀)*(N:ℝ) := by ring
  rw [hwid] at ht
  have hmw := mul_le_mul_of_nonneg_left hw (show 0 ≤ 3*M by positivity)
  have hC : 0 ≤ localMomentConstant (3*M) T₀ r*ScalarRate.envelopeConstant E S/(r.factorial:ℝ) := by
    have hm := localMomentConstant_nonneg (T₀ := T₀) (show 0 ≤ 3*M by positivity) r
    have he := ScalarRate.envelopeConstant_pos E S
    positivity
  have hme := mul_le_mul_of_nonneg_left he hC
  unfold localAlgebraicConstant
  linarith

/-- Epsilon-selected actual real-sample row, and its complete finite halo count.
The target's local polynomial approximation class supplies comparators only for
the proof; the construction itself reads only the stated 2N values of f. -/
theorem local_algebraic_accuracy_and_width
    {φ f : ℝ → ℝ} {r : ℕ} (hφ : ContDiff ℝ r φ)
    (K : 𝓢(ℝ, ℝ)) {m ν μ A M s T₀ H Z ε : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u=m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν)
    (S : SpatialEnvelope K μ) (hA : 0 ≤ A) (hM : 0 ≤ M) (hs : 0 < s) (hε : 0 < ε)
    (hf : ∀ x : ℝ, |x| ≤ 1 → |f x| ≤ M)
    (happrox : ∀ N : ℕ, 0 < N → ∃ P : ℝ[X], P.degree < N ∧
      ∀ x : ℝ, |x| ≤ 1 → |f x-P.eval x| ≤ A*(N:ℝ)^(-s))
    (hT : 1 < T₀) (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ) :
    let C := max 1 (localAlgebraicConstant E S A M T₀ s r)
    let N := ⌈ScalarScaling.algebraicCutoff C s ε⌉₊
    let hN : 0 < N := Nat.ceil_pos.mpr (Real.exp_pos _)
    (∀ x : ℝ, |x| ≤ 1 →
      |localAlgebraicNetwork φ f K r N m T₀ H Z ν μ hN (lt_trans zero_lt_one hT) x-f x| ≤ ε) ∧
    ((localGrid (2*N) T₀ H Z ν μ).card:ℝ) ≤
      ((2*H/Real.pi+2*H*Z+1)*(preparationGrowth T₀+3)^max 1 (1/μ+1/ν))*
        ((2*N:ℕ):ℝ)^max 1 (1/μ+1/ν) := by
  let C := max 1 (localAlgebraicConstant E S A M T₀ s r)
  let N := ⌈ScalarScaling.algebraicCutoff C s ε⌉₊
  have hN : 0 < N := Nat.ceil_pos.mpr (Real.exp_pos _)
  have hC : 0 ≤ C := le_trans zero_le_one (le_max_left _ _)
  have heps := AlgebraicConstruction.power_accuracy_choice hC hε hs
  constructor
  · intro x hx
    apply (local_algebraic_error hφ hN K hm hkernel E S hA hM hf (happrox N hN)
      hT hH hsep hZ hhalo x hx).trans
    exact (mul_le_mul_of_nonneg_right (le_max_right 1 _)
      (Real.rpow_nonneg (Nat.cast_nonneg N) _)).trans heps
  · exact local_analytic_grid_width (lt_trans zero_lt_one hT) (by omega : 0 < 2*N) (by linarith) hZ

end QIRadon.LocalAlgebraic
