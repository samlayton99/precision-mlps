import QIRadon.Quadrature.CasimirJet
import QIRadon.Quadrature.CasimirProjection
import QIRadon.Quadrature.CasimirDecay

open MeasureTheory MvPolynomial
open scoped BigOperators

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 200000

open Gaussian

def casimirEigenvalue (n k : ℕ) : ℝ := 2*(k:ℝ)*((k:ℝ)+(n+1:ℕ)-2)

theorem generator_smul {n : ℕ} (i j : Fin (n+1)) (c : ℝ)
    (p : MvPolynomial (Fin (n+1)) ℝ) : generator i j (c • p) = c • generator i j p := by
  unfold generator
  rw [(pderiv j).map_smul, (pderiv i).map_smul, mul_smul_comm, mul_smul_comm, smul_sub]

theorem orderedCasimir_smul {n : ℕ} (c : ℝ) (p : MvPolynomial (Fin (n+1)) ℝ) :
    orderedCasimir (c • p) = c • orderedCasimir p := by
  simp only [orderedCasimir, generator_smul, ← Finset.smul_sum]

theorem polynomialCasimirIterate_smul {n : ℕ} (r : ℕ) (c : ℝ)
    (p : MvPolynomial (Fin (n+1)) ℝ) :
    polynomialCasimirIterate r (c • p) = c • polynomialCasimirIterate r p := by
  induction r generalizing p with
  | zero => rfl
  | succ r ih => rw [polynomialCasimirIterate_succ, orderedCasimir_smul, ih,
      polynomialCasimirIterate_succ]

theorem polynomialCasimirIterate_eigen (n k r : ℕ) (v : HarmonicSpace n k) :
    polynomialCasimirIterate r (harmonicPolynomial n k v).val =
      (-casimirEigenvalue n k)^r • (harmonicPolynomial n k v).val := by
  induction r with
  | zero => simp [polynomialCasimirIterate]
  | succ r ih =>
    rw [polynomialCasimirIterate_succ, harmonicPolynomial_eigen,
      polynomialCasimirIterate_smul, ih, smul_smul]
    rw [pow_succ']
    rfl

theorem polynomialCasimirIterate_harmonic_onSphere (n k r : ℕ)
    (v : HarmonicSpace n k) (θ : UnitSphere n) :
    polynomialOnSphere n (polynomialCasimirIterate r (harmonicPolynomial n k v).val) θ =
      (-casimirEigenvalue n k)^r * harmonicEvaluation n k θ v := by
  rw [polynomialCasimirIterate_eigen]
  change eval (fun i ↦ θ.val i) ((-casimirEigenvalue n k)^r • (harmonicPolynomial n k v).val) = _
  rw [smul_eq_C_mul, map_mul, eval_C]
  have h := congrArg (fun F : C(UnitSphere n, ℝ) ↦ F θ) (harmonicPolynomial_representative n k v)
  change eval (fun i ↦ θ.val i) (harmonicPolynomial n k v).val = harmonicEvaluation n k θ v at h
  rw [h]

theorem casimirEigenvalue_ge_square {n k : ℕ} (hn : 1 ≤ n) :
    (k:ℝ)^2 ≤ casimirEigenvalue n k := by
  have hnR : 1 ≤ (n:ℝ) := by exact_mod_cast hn
  have hkR : 0 ≤ (k:ℝ) := Nat.cast_nonneg _
  unfold casimirEigenvalue
  push_cast
  nlinarith

theorem canonical_jet_projection_estimate {n : ℕ} (hn : 1 ≤ n)
    (D : RotationJet n) (hD : IsRotationJet D) (k r : ℕ) :
    (casimirEigenvalue n k)^r * ‖continuousHarmonicCoordinates n (D []) k‖ ≤
      ‖jetCasimir D r []‖ := by
  let v := continuousHarmonicCoordinates n (D []) k
  have heigen_nonneg : 0 ≤ casimirEigenvalue n k :=
    (sq_nonneg (k:ℝ)).trans (casimirEigenvalue_ge_square hn)
  have hpair := jetCasimir_pairing D hD r [] (harmonicPolynomial n k v).val
  have hp (θ : UnitSphere n) : polynomialOnSphere n (harmonicPolynomial n k v).val θ =
      harmonicEvaluation n k θ v := congrArg (fun F : C(UnitSphere n, ℝ) ↦ F θ)
        (harmonicPolynomial_representative n k v)
  simp_rw [hp, polynomialCasimirIterate_harmonic_onSphere] at hpair
  have hfactor : (fun θ : UnitSphere n ↦ D [] θ * ((-casimirEigenvalue n k)^r *
      harmonicEvaluation n k θ v)) =
      (fun θ ↦ (-casimirEigenvalue n k)^r * (D [] θ * harmonicEvaluation n k θ v)) := by
    funext θ; ring
  rw [hfactor, integral_const_mul, continuous_harmonic_pairing_self] at hpair
  have hb := continuous_harmonic_pairing_bound n k (jetCasimir D r []) v
  rw [hpair, abs_mul, abs_pow, abs_neg, abs_of_nonneg heigen_nonneg,
    abs_of_nonneg (sq_nonneg _)] at hb
  by_cases hv : ‖v‖ = 0
  · change _ * ‖v‖ ≤ _
    rw [hv, mul_zero]
    exact norm_nonneg _
  · have hvpos : 0 < ‖v‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hv)
    apply (mul_le_mul_iff_right₀ hvpos).mp
    convert hb using 1 <;> first | rfl | ring

/-- Actual canonical harmonic coefficients obey the all-orders bound once the
 real rotation derivatives have the mixed-Cauchy word bound. No coefficient
 decay or approximation rate is a hypothesis. -/
theorem canonical_jet_all_orders {n : ℕ} (hn : 1 ≤ n)
    (D : RotationJet n) (hD : IsRotationJet D) {B δ : ℝ} (_hB : 0 ≤ B) (_hδ : 0 < δ)
    (hword : ∀ w : List (RotationPair n),
      ‖D w‖ ≤ B * (2*(w.length:ℝ)/δ)^w.length)
    (k : ℕ) (hk : 0 < k) (r : ℕ) :
    ‖continuousHarmonicCoordinates n (D []) k‖ ≤
      B * ((4*(n+1:ℕ)/δ)*(r:ℝ)/(k:ℝ))^(2*r) := by
  have hproj := canonical_jet_projection_estimate hn D hD k r
  have htarget := jetCasimir_norm_bound D r [] (B*(4*(r:ℝ)/δ)^(2*r)) (by
    intro w hw
    simp only [List.length_nil, zero_add] at hw
    have h := hword w
    rw [hw] at h
    convert h using 1
    push_cast
    ring)
  have hkR : 0 < (k:ℝ) := Nat.cast_pos.mpr hk
  have hpow : (k:ℝ)^(2*r) ≤ (casimirEigenvalue n k)^r := by
    rw [pow_mul]
    exact pow_le_pow_left₀ (sq_nonneg _) (casimirEigenvalue_ge_square hn) r
  have hh := (mul_le_mul_of_nonneg_right hpow (norm_nonneg
    (continuousHarmonicCoordinates n (D []) k))).trans (hproj.trans htarget)
  have hdivide : ‖continuousHarmonicCoordinates n (D []) k‖ ≤
      (((n+1:ℕ):ℝ)^(2*r) * (B*(4*(r:ℝ)/δ)^(2*r))) / (k:ℝ)^(2*r) := by
    apply (le_div_iff₀ (pow_pos hkR _)).mpr
    simpa only [mul_comm] using hh
  convert hdivide using 1
  simp only [div_pow, mul_pow, Nat.cast_add, Nat.cast_one]
  ring


/-- The angular analytic energy is derived from actual bounded rotation
derivatives. `OrbitFamily.lean` constructs this jet and proves its word bound
from the bounded complex rotation family. -/
theorem canonical_jet_energy {n : ℕ} (hn : 1 ≤ n)
    (D : RotationJet n) (hD : IsRotationJet D) {B δ : ℝ} (hB : 0 ≤ B) (hδ : 0 < δ)
    (hword : ∀ w : List (RotationPair n),
      ‖D w‖ ≤ B * (2*(w.length:ℝ)/δ)^w.length) :
    Summable (fun k : ℕ ↦
      Real.exp (2*CasimirDecay.energyRate (4*(n+1:ℕ)/δ)*k) *
        ‖continuousHarmonicCoordinates n (D []) k‖^2) ∧
    (∑' k : ℕ, Real.exp (2*CasimirDecay.energyRate (4*(n+1:ℕ)/δ)*k) *
        ‖continuousHarmonicCoordinates n (D []) k‖^2) ≤
      (CasimirDecay.energyConstant B (4*(n+1:ℕ)/δ))^2 := by
  apply CasimirDecay.all_orders_energy_certificate hB (by positivity)
    (fun k ↦ norm_nonneg _) ?_
    (fun k hk r ↦ canonical_jet_all_orders hn D hD hB hδ hword k hk r)
  have h0 := hword []
  simp only [List.length_nil, Nat.cast_zero, mul_zero, zero_div, pow_zero, mul_one] at h0
  exact (continuous_harmonic_norm_le n 0 (D [])).trans h0

end
end QIRadon.Casimir
