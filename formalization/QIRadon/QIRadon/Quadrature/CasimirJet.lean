import QIRadon.Quadrature.CasimirSmooth
import QIRadon.Quadrature.CasimirHarmonic

open MeasureTheory
open scoped BigOperators

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 100000

open Gaussian

abbrev RotationPair (n : ℕ) := Fin (n+1) × Fin (n+1)
abbrev RotationJet (n : ℕ) := List (RotationPair n) → C(UnitSphere n, ℝ)

def IsRotationJet {n : ℕ} (D : RotationJet n) : Prop :=
  ∀ w i j, HasRotationDerivative i j (D w) (D ((i,j)::w))

def jetCasimir {n : ℕ} (D : RotationJet n) : ℕ → List (RotationPair n) → C(UnitSphere n, ℝ)
  | 0, w => D w
  | j+1, w => ∑ i, ∑ l, jetCasimir D j ((i,l)::(i,l)::w)

def polynomialCasimirIterate {n : ℕ} (r : ℕ) (p : MvPolynomial (Fin (n+1)) ℝ) :
    MvPolynomial (Fin (n+1)) ℝ := (orderedCasimir^[r]) p

theorem polynomialCasimirIterate_succ {n : ℕ} (r : ℕ) (p : MvPolynomial (Fin (n+1)) ℝ) :
    polynomialCasimirIterate (r+1) p = polynomialCasimirIterate r (orderedCasimir p) :=
  Function.iterate_succ_apply orderedCasimir r p

theorem jet_pairing_step {n : ℕ} (D : RotationJet n) (hD : IsRotationJet D)
    (w : List (RotationPair n)) (i j : Fin (n+1)) (p : MvPolynomial (Fin (n+1)) ℝ) :
    (∫ θ, D ((i,j)::(i,j)::w) θ * polynomialOnSphere n p θ ∂normalizedSphere n) =
    (∫ θ, D w θ * polynomialOnSphere n (generator i j (generator i j p)) θ
      ∂normalizedSphere n) := by
  rw [rotationDerivative_skew i j (D ((i,j)::w)) (polynomialOnSphere n p)
    (D ((i,j)::(i,j)::w)) (polynomialOnSphere n (generator i j p))
    (hD _ i j) (polynomial_hasRotationDerivative n i j p)]
  rw [rotationDerivative_skew i j (D w) (polynomialOnSphere n (generator i j p))
    (D ((i,j)::w)) (polynomialOnSphere n (generator i j (generator i j p)))
    (hD _ i j) (polynomial_hasRotationDerivative n i j (generator i j p)), neg_neg]


theorem jet_pairing_orderedCasimir {n : ℕ} (D : RotationJet n) (hD : IsRotationJet D)
    (w : List (RotationPair n)) (p : MvPolynomial (Fin (n+1)) ℝ) :
    (∫ θ, jetCasimir D 1 w θ * polynomialOnSphere n p θ ∂normalizedSphere n) =
    (∫ θ, D w θ * polynomialOnSphere n (orderedCasimir p) θ ∂normalizedSphere n) := by
  have hp (θ : UnitSphere n) : D w θ * polynomialOnSphere n (orderedCasimir p) θ =
      ∑ i, ∑ j, D w θ * polynomialOnSphere n (generator i j (generator i j p)) θ := by
    simp [polynomialOnSphere, orderedCasimir, Finset.mul_sum]
  simp only [jetCasimir, ContinuousMap.sum_apply, Finset.sum_mul]
  simp_rw [hp]
  rw [integral_finsetSum, integral_finsetSum]
  · apply Finset.sum_congr rfl
    intro i _
    rw [integral_finsetSum, integral_finsetSum]
    · apply Finset.sum_congr rfl
      intro j _
      exact jet_pairing_step D hD w i j p
    all_goals
      intro j _
      exact (by fun_prop : Continuous _).integrable_of_hasCompactSupport
        (HasCompactSupport.of_compactSpace _)
  all_goals
    intro i _
    exact integrable_finsetSum _ (fun j _ ↦
      (by fun_prop : Continuous _).integrable_of_hasCompactSupport
        (HasCompactSupport.of_compactSpace _))



theorem integral_continuous_sum_pairing {n : ℕ} {α : Type*} [Fintype α]
    (F : α → C(UnitSphere n, ℝ)) (G : C(UnitSphere n, ℝ)) :
    (∫ θ, (∑ a, F a) θ * G θ ∂normalizedSphere n) =
      ∑ a, ∫ θ, F a θ * G θ ∂normalizedSphere n := by
  simp only [ContinuousMap.sum_apply, Finset.sum_mul]
  exact integral_finsetSum _ (fun a _ ↦
    ((F a).continuous.mul G.continuous).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _))

theorem polynomialCasimirIterate_commute {n : ℕ} (r : ℕ)
    (p : MvPolynomial (Fin (n+1)) ℝ) :
    polynomialCasimirIterate r (orderedCasimir p) = orderedCasimir (polynomialCasimirIterate r p) :=
  (Function.iterate_succ_apply orderedCasimir r p).symm.trans
    (Function.iterate_succ_apply' orderedCasimir r p)

theorem integral_jetCasimir_succ {n : ℕ} (D : RotationJet n) (r : ℕ)
    (w : List (RotationPair n)) (p : MvPolynomial (Fin (n+1)) ℝ) :
    (∫ θ, jetCasimir D (r+1) w θ * polynomialOnSphere n p θ ∂normalizedSphere n) =
      ∑ i, ∑ j, ∫ θ, jetCasimir D r ((i,j)::(i,j)::w) θ * polynomialOnSphere n p θ
        ∂normalizedSphere n := by
  rw [jetCasimir, integral_continuous_sum_pairing]
  apply Finset.sum_congr rfl
  intro i _
  exact integral_continuous_sum_pairing _ _


theorem jetCasimir_pairing {n : ℕ} (D : RotationJet n) (hD : IsRotationJet D)
    (r : ℕ) (w : List (RotationPair n)) (p : MvPolynomial (Fin (n+1)) ℝ) :
    (∫ θ, jetCasimir D r w θ * polynomialOnSphere n p θ ∂normalizedSphere n) =
    (∫ θ, D w θ * polynomialOnSphere n (polynomialCasimirIterate r p) θ
      ∂normalizedSphere n) := by
  induction r generalizing w p with
  | zero => rfl
  | succ r ih =>
    rw [integral_jetCasimir_succ]
    calc
      _ = ∑ i, ∑ j, ∫ θ, D ((i,j)::(i,j)::w) θ *
          polynomialOnSphere n (polynomialCasimirIterate r p) θ ∂normalizedSphere n := by
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        exact ih _ p
      _ = ∫ θ, D w θ * polynomialOnSphere n (orderedCasimir (polynomialCasimirIterate r p)) θ
          ∂normalizedSphere n := by
        have hs := jet_pairing_orderedCasimir D hD w (polynomialCasimirIterate r p)
        rw [integral_jetCasimir_succ D 0] at hs
        exact hs
      _ = _ := by rw [polynomialCasimirIterate_succ, polynomialCasimirIterate_commute]


theorem jetCasimir_norm_bound {n : ℕ} (D : RotationJet n) (r : ℕ)
    (w : List (RotationPair n)) (B : ℝ)
    (hB : ∀ u : List (RotationPair n), u.length = w.length + 2*r → ‖D u‖ ≤ B) :
    ‖jetCasimir D r w‖ ≤ ((n+1:ℕ):ℝ)^(2*r) * B := by
  induction r generalizing w with
  | zero => simpa [jetCasimir] using hB w (by simp)
  | succ r ih =>
    calc
      _ ≤ ∑ i, ∑ j, ‖jetCasimir D r ((i,j)::(i,j)::w)‖ := by
        apply (norm_sum_le _ _).trans
        apply Finset.sum_le_sum
        intro i _
        exact norm_sum_le _ _
      _ ≤ ∑ i : Fin (n+1), ∑ j : Fin (n+1), ((n+1:ℕ):ℝ)^(2*r) * B := by
        apply Finset.sum_le_sum
        intro i _
        apply Finset.sum_le_sum
        intro j _
        apply ih
        intro u hu
        apply hB u
        simp only [List.length_cons] at hu
        omega
      _ = _ := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        rw [show 2*(r+1) = 2*r+2 by omega, pow_add]
        ring

end
end QIRadon.Casimir
