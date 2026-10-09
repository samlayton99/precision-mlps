import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic

/-!
# Positive quadrature and transport of scalar approximation errors

These are supporting theorems for the QI–Radon angular construction. In particular,
`positive_quadrature_error` is CONDITIONAL on exact integration of the supplied
approximating function. It does not assert that Gauss–Jacobi nodes have already
been constructed, nor that the desired spherical polynomial approximation exists.
-/

open scoped BigOperators
open MeasureTheory

namespace QIRadon

section FiniteSums

variable {ι X : Type*} [Fintype ι]

/-- Positive weights transport heterogeneous pointwise errors without multiplying
the bound by the number of nodes. Normalization is not needed for this version. -/
theorem weighted_error_bound
    (weight : ι → ℝ) (a b : ι → ℝ) (err : ι → ℝ)
    (hweight : ∀ i, 0 ≤ weight i) (herr : ∀ i, |a i - b i| ≤ err i) :
    |(∑ i, weight i * a i) - ∑ i, weight i * b i| ≤ ∑ i, weight i * err i := by
  rw [← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  calc
    |∑ i, weight i * (a i - b i)| ≤ ∑ i, |weight i * (a i - b i)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, weight i * |a i - b i| := by
      simp_rw [abs_mul, abs_of_nonneg (hweight _)]
    _ ≤ ∑ i, weight i * err i :=
      Finset.sum_le_sum fun i _ ↦ mul_le_mul_of_nonneg_left (herr i) (hweight i)

/-- A common error is preserved by a positive normalized weighted average. -/
theorem normalized_weighted_error_bound
    (weight : ι → ℝ) (a b : ι → ℝ) (err : ℝ)
    (hweight : ∀ i, 0 ≤ weight i) (hnorm : ∑ i, weight i = 1)
    (herr : ∀ i, |a i - b i| ≤ err) :
    |(∑ i, weight i * a i) - ∑ i, weight i * b i| ≤ err := by
  calc
    |(∑ i, weight i * a i) - ∑ i, weight i * b i|
        ≤ ∑ i, weight i * err := weighted_error_bound weight a b (fun _ ↦ err) hweight herr
    _ = err := by rw [← Finset.sum_mul, hnorm, one_mul]

/-- The common construction step: an angular quadrature error plus the
positive weighted average of the certified row errors. The target here is a
number (usually the exact integral). The caller supplies the already proved
angular and row certificates; this lemma only combines them. -/
theorem weighted_approximation_error
    (weight : ι → ℝ) (approx exact : ι → ℝ) (err : ι → ℝ)
    {target angular : ℝ}
    (hweight : ∀ i, 0 ≤ weight i) (hrow : ∀ i, |approx i-exact i| ≤ err i)
    (hangular : |target-∑ i, weight i*exact i| ≤ angular) :
    |(∑ i, weight i*approx i)-target| ≤ angular+∑ i, weight i*err i := by
  have hw := weighted_error_bound weight approx exact err hweight hrow
  have ha : |(∑ i, weight i*exact i)-target| ≤ angular := by
    rwa [abs_sub_comm]
  exact (abs_sub_le _ (∑ i, weight i*exact i) _).trans
    ((add_le_add hw ha).trans_eq (add_comm _ _))

/-- With normalized positive weights, a common row error is not multiplied
by the number of directions. -/
theorem normalized_approximation_error
    (weight : ι → ℝ) (approx exact : ι → ℝ) (err : ℝ)
    {target angular : ℝ}
    (hweight : ∀ i, 0 ≤ weight i) (hnorm : ∑ i, weight i=1)
    (hrow : ∀ i, |approx i-exact i| ≤ err)
    (hangular : |target-∑ i, weight i*exact i| ≤ angular) :
    |(∑ i, weight i*approx i)-target| ≤ angular+err := by
  simpa only [← Finset.sum_mul,hnorm,one_mul] using
    weighted_approximation_error weight approx exact (fun _ => err) hweight hrow hangular

/-- Anchoring an approximation by subtracting its value at one reference point
costs at most twice the original uniform error. -/
theorem anchored_error_bound (f g : X → ℝ) (x x₀ : X) (err : ℝ)
    (hx : |f x - g x| ≤ err) (hx₀ : |f x₀ - g x₀| ≤ err) :
    |(f x - f x₀) - (g x - g x₀)| ≤ 2 * err := by
  calc
    |(f x - f x₀) - (g x - g x₀)| = |(f x - g x) - (f x₀ - g x₀)| := by
      congr 1; ring
    _ ≤ |f x - g x| + |f x₀ - g x₀| := abs_sub _ _
    _ ≤ 2 * err := by linarith

/-- Algebraic identity separating the angular error and each anchored spoke error. -/
theorem anchored_transport_identity
    (weight : ι → ℝ) (f : X → ℝ) (g q : ι → X → ℝ) (x x₀ : X) :
    (f x₀ + ∑ i, weight i * (q i x - q i x₀)) - f x =
      ((∑ i, weight i * g i x) - f x) -
      ((∑ i, weight i * g i x₀) - f x₀) +
      ∑ i, weight i * ((q i x - q i x₀) - (g i x - g i x₀)) := by
  simp_rw [mul_sub, Finset.sum_sub_distrib]
  ring

/-- End-to-end error transport. The final network is anchored once; the angular
and unanchored per-spoke certificates each incur their explicit factor of two. -/
theorem anchored_transport_bound
    (weight : ι → ℝ) (f : X → ℝ) (g q : ι → X → ℝ)
    (x x₀ : X) (angular : ℝ) (err : ι → ℝ)
    (hweight : ∀ i, 0 ≤ weight i)
    (hax : |(∑ i, weight i * g i x) - f x| ≤ angular)
    (ha₀ : |(∑ i, weight i * g i x₀) - f x₀| ≤ angular)
    (hsx : ∀ i, |q i x - g i x| ≤ err i)
    (hs₀ : ∀ i, |q i x₀ - g i x₀| ≤ err i) :
    |(f x₀ + ∑ i, weight i * (q i x - q i x₀)) - f x|
      ≤ 2 * angular + 2 * ∑ i, weight i * err i := by
  have ha : |(f x-f x₀)-∑ i, weight i*(g i x-g i x₀)| ≤ 2*angular := by
    rw [abs_sub_comm]
    simpa only [mul_sub,Finset.sum_sub_distrib] using
      anchored_error_bound (fun y => ∑ i, weight i*g i y) f x x₀ angular hax ha₀
  have ht := weighted_approximation_error weight
    (fun i => q i x-q i x₀) (fun i => g i x-g i x₀) (fun i => 2*err i) hweight
    (fun i => anchored_error_bound (q i) (g i) x x₀ (err i) (hsx i) (hs₀ i)) ha
  have hsum : (∑ i, weight i*(2*err i)) = 2*∑ i, weight i*err i := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hsum] at ht
  convert! ht using 1
  congr 1
  ring


end FiniteSums

section ProductAndFolding

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- The actual tensor-product weights used in recursive sphere quadrature. -/
def productWeight (a : ι → ℝ) (b : κ → ℝ) (ij : ι × κ) : ℝ := a ij.1 * b ij.2

omit [Fintype ι] [Fintype κ] in
theorem productWeight_nonneg (a : ι → ℝ) (b : κ → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ j, 0 ≤ b j) (ij : ι × κ) :
    0 ≤ productWeight a b ij := mul_nonneg (ha ij.1) (hb ij.2)

theorem productWeight_sum (a : ι → ℝ) (b : κ → ℝ)
    (ha : ∑ i, a i = 1) (hb : ∑ j, b j = 1) :
    ∑ ij, productWeight a b ij = 1 := by
  simp only [productWeight, Fintype.sum_prod_type, ← Finset.mul_sum, hb, mul_one, ha]

/-- Exactness of each factor implies exactness for a separated product. Finite
linear combinations give the algebraic building blocks of tensor cubature. -/
theorem product_sum_separable (a : ι → ℝ) (b : κ → ℝ)
    (f : ι → ℝ) (g : κ → ℝ) :
    ∑ ij, productWeight a b ij * (f ij.1 * g ij.2) =
      (∑ i, a i * f i) * ∑ j, b j * g j := by
  simp only [productWeight, Fintype.sum_prod_type]
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Sum the weights of a known antipodal pair. Pair identification and the
Radon-profile parity identity are separate mathematical obligations. -/
def foldedWeight (a : ι × Bool → ℝ) (i : ι) : ℝ := a (i, false) + a (i, true)

omit [Fintype ι] in
theorem foldedWeight_nonneg (a : ι × Bool → ℝ) (ha : ∀ ij, 0 ≤ a ij) (i : ι) :
    0 ≤ foldedWeight a i := add_nonneg (ha _) (ha _)

theorem folded_sum (a : ι × Bool → ℝ) (f : ι × Bool → ℝ)
    (heven : ∀ i, f (i, true) = f (i, false)) :
    ∑ ij, a ij * f ij = ∑ i, foldedWeight a i * f (i, false) := by
  simp only [Fintype.sum_prod_type, Fintype.sum_bool, heven, foldedWeight, add_mul, add_comm]

theorem foldedWeight_sum (a : ι × Bool → ℝ) (ha : ∑ ij, a ij = 1) :
    ∑ i, foldedWeight a i = 1 := by
  have h := folded_sum a (fun _ ↦ 1) (by intro i; rfl)
  simpa using h.symm.trans (by simpa using ha)

/-- Cardinality of the retained tuple index set. To apply this count to actual
sphere nodes one still needs the recursive geometric construction and pairing. -/
theorem product_index_card (q k : ℕ) : Fintype.card (Fin k → Fin q) = q ^ k := by
  simp

theorem paired_product_index_card (q k : ℕ) :
    Fintype.card ((Fin k → Fin q) × Bool) = 2 * q ^ k := by
  simp [Nat.mul_comm]

end ProductAndFolding

section Quadrature

variable {ι X : Type*} [Fintype ι] [MeasurableSpace X]
variable (μ : Measure X) [IsProbabilityMeasure μ]

/-- Exactness for a comparison function and its uniform approximation error imply
a quadrature error of at most twice that error. The actual construction of an
exact quadrature rule and comparison polynomial is a separate obligation. -/
theorem positive_quadrature_error
    (node : ι → X) (weight : ι → ℝ) (f p : X → ℝ) (err : ℝ)
    (hweight : ∀ i, 0 ≤ weight i) (hnorm : ∑ i, weight i = 1)
    (hf : Integrable f μ) (hp : Integrable p μ)
    (hexact : ∫ x, p x ∂μ = ∑ i, weight i * p (node i))
    (happrox : ∀ x, |f x - p x| ≤ err) :
    |(∫ x, f x ∂μ) - ∑ i, weight i * f (node i)| ≤ 2 * err := by
  have hint : |∫ x, f x - p x ∂μ| ≤ err := by
    have h := norm_integral_le_of_norm_le_const
      (μ := μ) (f := fun x ↦ f x - p x) (C := err)
      (Filter.Eventually.of_forall fun x ↦ by simpa only [Real.norm_eq_abs] using happrox x)
    simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using h
  rw [integral_sub hf hp,hexact] at hint
  have ht := normalized_approximation_error weight
    (fun i => f (node i)) (fun i => p (node i)) err hweight hnorm
    (fun i => happrox (node i)) hint
  rw [abs_sub_comm] at ht
  linarith


end Quadrature

end QIRadon
