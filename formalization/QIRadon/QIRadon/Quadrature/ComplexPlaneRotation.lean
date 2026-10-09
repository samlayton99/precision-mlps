import QIRadon.Quadrature.PlaneRotation
import Mathlib.Analysis.Analytic.WithLp
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-! Actual complex coordinate-plane rotations, including quantitative bounds
for arbitrary products of rotations. Diagonal index pairs act as the identity. -/

noncomputable section
open scoped BigOperators
namespace QIRadon.ComplexRotation
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Canonical coordinatewise real-to-complex isometric embedding. -/
def realEmbedding : EuclideanSpace ℝ ι →ₗᵢ[ℝ] EuclideanSpace ℂ ι where
  toFun x := WithLp.toLp 2 (fun i => (x i : ℂ))
  map_add' x y := by ext i; simp
  map_smul' c x := by ext i; simp
  norm_map' x := by
    apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
    rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
    apply Finset.sum_congr rfl
    intro i _
    simp

omit [DecidableEq ι] in
@[simp] theorem realEmbedding_apply (x : EuclideanSpace ℝ ι) (i : ι) :
    realEmbedding x i = (x i : ℂ) := rfl

def planeProjection (i j : ι) (x : EuclideanSpace ℂ ι) : EuclideanSpace ℂ ι :=
  WithLp.toLp 2 (fun k => if k=i ∨ k=j then x k else 0)

def planeGenerator (i j : ι) (x : EuclideanSpace ℂ ι) : EuclideanSpace ℂ ι :=
  WithLp.toLp 2 (fun k => if k=i then -x j else if k=j then x i else 0)

theorem planeProjection_norm_le (i j : ι) (x : EuclideanSpace ℂ ι) :
    ‖planeProjection i j x‖ ≤ ‖x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
  apply Finset.sum_le_sum
  intro k _
  change ‖if k=i ∨ k=j then x k else (0 : ℂ)‖^2 ≤ ‖x k‖^2
  split_ifs <;> simp

theorem planeGenerator_norm_eq (i j : ι) (hij : i ≠ j) (x : EuclideanSpace ℂ ι) :
    ‖planeGenerator i j x‖ = ‖planeProjection i j x‖ := by
  apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
  have hjterm (k : ι) : ‖planeGenerator i j x k‖^2 =
      (if k=i then ‖x j‖^2 else 0) + (if k=j then ‖x i‖^2 else 0) := by
    by_cases hi : k=i
    · subst k; simp [planeGenerator, hij]
    · by_cases hj : k=j
      · subst k; simp [planeGenerator, hi]
      · simp [planeGenerator, hi, hj]
  have hpterm (k : ι) : ‖planeProjection i j x k‖^2 =
      (if k=i then ‖x i‖^2 else 0) + (if k=j then ‖x j‖^2 else 0) := by
    by_cases hi : k=i
    · subst k; simp [planeProjection, hij]
    · by_cases hj : k=j
      · subst k; simp [planeProjection, hi]
      · simp [planeProjection, hi, hj]
  simp_rw [hjterm, hpterm]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true]
  ring

theorem planeGenerator_norm_le (i j : ι) (hij : i ≠ j) (x : EuclideanSpace ℂ ι) :
    ‖planeGenerator i j x‖ ≤ ‖x‖ := by
  rw [planeGenerator_norm_eq i j hij]
  exact planeProjection_norm_le i j x

/-- Exact complexification of the real coordinate rotation, with the same
sign convention and identity on diagonal pairs. -/
def complexPlaneRotate (i j : ι) (z : ℂ) (x : EuclideanSpace ℂ ι) : EuclideanSpace ℂ ι :=
  if i=j then x else
    WithLp.toLp 2 (fun k => if k=i then Complex.cos z*x i-Complex.sin z*x j
      else if k=j then Complex.sin z*x i+Complex.cos z*x j else x k)

omit [Fintype ι] in
theorem complexPlaneRotate_decomposition (i j : ι) (hij : i ≠ j) (z : ℂ)
    (x : EuclideanSpace ℂ ι) :
    complexPlaneRotate i j z x = x + (Complex.cos z-1) • planeProjection i j x +
      Complex.sin z • planeGenerator i j x := by
  ext k
  simp only [complexPlaneRotate, hij, if_false, PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
  by_cases hi : k=i
  · subst k; simp [planeProjection, planeGenerator, hij]; ring
  · by_cases hj : k=j
    · subst k; simp [planeProjection, planeGenerator, hi]; ring
    · simp [planeProjection, planeGenerator, hi, hj]

theorem exp_difference_bound (z : ℂ) : ‖Complex.exp z-1‖ ≤ Real.exp ‖z‖-1 := by
  simpa using Complex.norm_exp_sub_sum_le_exp_norm_sub_sum z 1

theorem cos_difference_bound (z : ℂ) : ‖Complex.cos z-1‖ ≤ Real.exp ‖z‖-1 := by
  have hid : Complex.cos z-1 = ((Complex.exp (z*Complex.I)-1)+
      (Complex.exp (-z*Complex.I)-1))/2 := by rw [Complex.cos]; ring
  rw [hid, norm_div]
  have hh := norm_add_le (Complex.exp (z*Complex.I)-1) (Complex.exp (-z*Complex.I)-1)
  have h1 := exp_difference_bound (z*Complex.I)
  have h2 := exp_difference_bound (-z*Complex.I)
  simp only [norm_mul, norm_neg, Complex.norm_I, mul_one] at h1 h2
  rw [Complex.norm_two]
  linarith

theorem sin_bound (z : ℂ) : ‖Complex.sin z‖ ≤ Real.exp ‖z‖-1 := by
  have hid : Complex.sin z = ((Complex.exp (-z*Complex.I)-1)-
      (Complex.exp (z*Complex.I)-1))*Complex.I/2 := by rw [Complex.sin]; ring
  rw [hid, norm_div, norm_mul, Complex.norm_I, mul_one]
  have hh := norm_sub_le (Complex.exp (-z*Complex.I)-1) (Complex.exp (z*Complex.I)-1)
  have h1 := exp_difference_bound (z*Complex.I)
  have h2 := exp_difference_bound (-z*Complex.I)
  simp only [norm_mul, norm_neg, Complex.norm_I, mul_one] at h1 h2
  rw [Complex.norm_two]
  linarith

theorem doubled_exponential_margin (r : ℝ) : 2*(Real.exp r-1) ≤ Real.exp (2*r)-1 := by
  rw [show 2*r=r+r by ring, Real.exp_add]
  nlinarith [sq_nonneg (Real.exp r-1)]

/-- Actual displacement bound, uniform in the number of ambient coordinates. -/
theorem complexPlaneRotate_displacement (i j : ι) (z : ℂ) (x : EuclideanSpace ℂ ι) :
    ‖complexPlaneRotate i j z x-x‖ ≤ (Real.exp (2*‖z‖)-1)*‖x‖ := by
  by_cases hij : i=j
  · simp only [complexPlaneRotate, hij, if_true, sub_self, norm_zero]
    exact mul_nonneg (sub_nonneg.mpr (Real.one_le_exp_iff.mpr (by positivity))) (norm_nonneg _)
  · rw [complexPlaneRotate_decomposition i j hij]
    have heq : x + (Complex.cos z-1) • planeProjection i j x +
        Complex.sin z • planeGenerator i j x-x =
        (Complex.cos z-1) • planeProjection i j x + Complex.sin z • planeGenerator i j x := by abel
    rw [heq]
    calc
      _ ≤ ‖Complex.cos z-1‖*‖planeProjection i j x‖ + ‖Complex.sin z‖*‖planeGenerator i j x‖ := by
        simpa only [norm_smul] using norm_add_le ((Complex.cos z-1) • planeProjection i j x)
          (Complex.sin z • planeGenerator i j x)
      _ ≤ (Real.exp ‖z‖-1)*‖x‖ + (Real.exp ‖z‖-1)*‖x‖ := by
        have hm : 0 ≤ Real.exp ‖z‖-1 := sub_nonneg.mpr (Real.one_le_exp_iff.mpr (norm_nonneg z))
        exact add_le_add
          (mul_le_mul (cos_difference_bound z) (planeProjection_norm_le i j x) (norm_nonneg _) hm)
          (mul_le_mul (sin_bound z) (planeGenerator_norm_le i j hij x) (norm_nonneg _) hm)
      _ ≤ _ := by nlinarith [mul_le_mul_of_nonneg_right (doubled_exponential_margin ‖z‖) (norm_nonneg x)]

theorem complexPlaneRotate_norm (i j : ι) (z : ℂ) (x : EuclideanSpace ℂ ι) :
    ‖complexPlaneRotate i j z x‖ ≤ Real.exp (2*‖z‖)*‖x‖ := by
  have ht := norm_le_norm_sub_add (complexPlaneRotate i j z x) x
  have hb := complexPlaneRotate_displacement i j z x
  nlinarith

omit [Fintype ι] in
@[simp] theorem complexPlaneRotate_zero (i j : ι) (x : EuclideanSpace ℂ ι) :
    complexPlaneRotate i j 0 x = x := by
  by_cases hij : i=j
  · simp [complexPlaneRotate, hij]
  · rw [complexPlaneRotate_decomposition i j hij]
    simp

theorem complexPlaneRotate_real (i j : ι) (hij : i ≠ j) (t : ℝ)
    (x : EuclideanSpace ℝ ι) :
    complexPlaneRotate i j (t : ℂ) (realEmbedding x) =
      realEmbedding (QIRadon.Gaussian.planeRotate i j t x) := by
  ext k
  simp only [complexPlaneRotate, hij, if_false, realEmbedding_apply, QIRadon.Gaussian.planeRotate_apply]
  split_ifs <;> simp


/-- Joint holomorphic dependence, suitable for products whose vector input
already depends holomorphically on all preceding angles. -/
theorem complexPlaneRotate_analytic_comp {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℂ E]
    {t : E → ℂ} {v : E → EuclideanSpace ℂ ι} {a : E}
    (ht : AnalyticAt ℂ t a) (hv : AnalyticAt ℂ v a) (i j : ι) :
    AnalyticAt ℂ (fun z => complexPlaneRotate i j (t z) (v z)) a := by
  by_cases hij : i=j
  · simpa only [complexPlaneRotate, hij, if_true] using hv
  · have hcoord : ∀ k : ι, AnalyticAt ℂ (fun z => v z k) a := by
      intro k
      exact ((PiLp.proj 2 (fun _ : ι => ℂ) k).analyticAt (v a)).comp hv
    have hh : AnalyticAt ℂ (fun z => fun k : ι =>
        if k=i then Complex.cos (t z)*v z i-Complex.sin (t z)*v z j
        else if k=j then Complex.sin (t z)*v z i+Complex.cos (t z)*v z j else v z k) a := by
      apply AnalyticAt.pi
      intro k
      split_ifs
      · exact ((Complex.analyticAt_cos.comp ht).mul (hcoord i)).sub
          ((Complex.analyticAt_sin.comp ht).mul (hcoord j))
      · exact ((Complex.analyticAt_sin.comp ht).mul (hcoord i)).add
          ((Complex.analyticAt_cos.comp ht).mul (hcoord j))
      · exact hcoord k
    have hto := ((PiLp.continuousLinearEquiv 2 ℂ (fun _ : ι => ℂ)).symm.analyticAt _).comp hh
    simp only [complexPlaneRotate, hij, if_false]
    convert! hto using 1

theorem complexPlaneRotate_entire (i j : ι) (x : EuclideanSpace ℂ ι) :
    AnalyticOnNhd ℂ (fun z => complexPlaneRotate i j z x) Set.univ := by
  intro z _
  exact complexPlaneRotate_analytic_comp analyticAt_id analyticAt_const i j

/-- Ordered rotation word. Each list entry selects a coordinate-plane pair
and its complex angle from a finite parameter vector. Repeated indices work. -/
def indexedWord {n : ℕ} (pairs : Fin n → ι × ι) :
    List (Fin n) → (Fin n → ℂ) → EuclideanSpace ℂ ι → EuclideanSpace ℂ ι
  | [], _, x => x
  | k :: w, z, x => complexPlaneRotate (pairs k).1 (pairs k).2 (z k) (indexedWord pairs w z x)

def wordBudget {n : ℕ} (w : List (Fin n)) (z : Fin n → ℂ) : ℝ :=
  (w.map (fun k => ‖z k‖)).sum

theorem wordBudget_nonneg {n : ℕ} (w : List (Fin n)) (z : Fin n → ℂ) :
    0 ≤ wordBudget w z := by
  induction w with
  | nil => simp [wordBudget]
  | cons k w ih => simpa [wordBudget] using add_nonneg (norm_nonneg (z k)) ih

theorem indexedWord_norm {n : ℕ} (pairs : Fin n → ι × ι) (w : List (Fin n))
    (z : Fin n → ℂ) (x : EuclideanSpace ℂ ι) :
    ‖indexedWord pairs w z x‖ ≤ Real.exp (2*wordBudget w z)*‖x‖ := by
  induction w with
  | nil => simp [indexedWord, wordBudget]
  | cons k w ih =>
      have hh := complexPlaneRotate_norm (pairs k).1 (pairs k).2 (z k) (indexedWord pairs w z x)
      calc
        _ ≤ Real.exp (2*‖z k‖)*‖indexedWord pairs w z x‖ := hh
        _ ≤ Real.exp (2*‖z k‖)*(Real.exp (2*wordBudget w z)*‖x‖) :=
          mul_le_mul_of_nonneg_left ih (Real.exp_pos _).le
        _ = _ := by
          simp only [wordBudget, List.map_cons, List.sum_cons]
          rw [← mul_assoc, ← Real.exp_add]
          congr 2
          ring

theorem indexedWord_displacement {n : ℕ} (pairs : Fin n → ι × ι) (w : List (Fin n))
    (z : Fin n → ℂ) (x : EuclideanSpace ℂ ι) :
    ‖indexedWord pairs w z x-x‖ ≤ (Real.exp (2*wordBudget w z)-1)*‖x‖ := by
  induction w with
  | nil => simp [indexedWord, wordBudget]
  | cons k w ih =>
      let y := indexedWord pairs w z x
      have hd := complexPlaneRotate_displacement (pairs k).1 (pairs k).2 (z k) y
      have hn := indexedWord_norm pairs w z x
      have hm : 0 ≤ Real.exp (2*‖z k‖)-1 :=
        sub_nonneg.mpr (Real.one_le_exp_iff.mpr (by positivity))
      calc
        _ ≤ ‖complexPlaneRotate (pairs k).1 (pairs k).2 (z k) y-y‖+‖y-x‖ := by
          simpa only [dist_eq_norm, indexedWord, y] using! dist_triangle
            (complexPlaneRotate (pairs k).1 (pairs k).2 (z k) y) y x
        _ ≤ (Real.exp (2*‖z k‖)-1)*‖y‖ + (Real.exp (2*wordBudget w z)-1)*‖x‖ := add_le_add hd ih
        _ ≤ (Real.exp (2*‖z k‖)-1)*(Real.exp (2*wordBudget w z)*‖x‖) +
            (Real.exp (2*wordBudget w z)-1)*‖x‖ :=
          add_le_add (mul_le_mul_of_nonneg_left hn hm) le_rfl
        _ = _ := by
          have he : Real.exp (2*‖z k‖)*Real.exp (2*wordBudget w z) =
              Real.exp (2*wordBudget (k::w) z) := by
            rw [← Real.exp_add]
            congr 1
            simp only [wordBudget, List.map_cons, List.sum_cons]
            ring
          rw [show (Real.exp (2*‖z k‖)-1)*(Real.exp (2*wordBudget w z)*‖x‖) +
              (Real.exp (2*wordBudget w z)-1)*‖x‖ =
              (Real.exp (2*‖z k‖)*Real.exp (2*wordBudget w z)-1)*‖x‖ by ring, he]

set_option maxHeartbeats 800000 in
theorem indexedWord_entire {n : ℕ} (pairs : Fin n → ι × ι) (w : List (Fin n))
    (x : EuclideanSpace ℂ ι) :
    AnalyticOnNhd ℂ (fun z : Fin n → ℂ => indexedWord pairs w z x) Set.univ := by
  induction w with
  | nil => intro z _; exact analyticAt_const
  | cons k w ih =>
      intro z _
      have ht : AnalyticAt ℂ (fun u : Fin n → ℂ => u k) z :=
        (ContinuousLinearMap.proj (R := ℂ) (φ := fun _ : Fin n => ℂ) k).analyticAt z
      have hv : AnalyticAt ℂ (fun u : Fin n → ℂ => indexedWord pairs w u x) z := ih z (Set.mem_univ _)
      change AnalyticAt ℂ (fun u : Fin n → ℂ =>
        complexPlaneRotate (pairs k).1 (pairs k).2 (u k) (indexedWord pairs w u x)) z
      exact complexPlaneRotate_analytic_comp ht hv (pairs k).1 (pairs k).2

omit [Fintype ι] in
@[simp] theorem indexedWord_zero {n : ℕ} (pairs : Fin n → ι × ι) (w : List (Fin n))
    (x : EuclideanSpace ℂ ι) : indexedWord pairs w 0 x = x := by
  induction w with
  | nil => rfl
  | cons k w ih => simpa only [indexedWord, Pi.zero_apply, complexPlaneRotate_zero] using ih

/-- One independently controlled angle per position in a rotation word. -/
def rotationWord {n : ℕ} (pairs : Fin n → ι × ι) (z : Fin n → ℂ)
    (x : EuclideanSpace ℂ ι) : EuclideanSpace ℂ ι :=
  indexedWord pairs (List.ofFn (fun i : Fin n => i)) z x

theorem rotationWord_norm {n : ℕ} (pairs : Fin n → ι × ι) (z : Fin n → ℂ)
    (x : EuclideanSpace ℂ ι) :
    ‖rotationWord pairs z x‖ ≤ Real.exp (2*∑ i, ‖z i‖)*‖x‖ := by
  simpa [rotationWord, wordBudget, List.map_ofFn, List.sum_ofFn] using
    indexedWord_norm pairs (List.ofFn (fun i : Fin n => i)) z x

theorem rotationWord_displacement {n : ℕ} (pairs : Fin n → ι × ι) (z : Fin n → ℂ)
    (x : EuclideanSpace ℂ ι) :
    ‖rotationWord pairs z x-x‖ ≤ (Real.exp (2*∑ i, ‖z i‖)-1)*‖x‖ := by
  simpa [rotationWord, wordBudget, List.map_ofFn, List.sum_ofFn] using
    indexedWord_displacement pairs (List.ofFn (fun i : Fin n => i)) z x

theorem rotationWord_entire {n : ℕ} (pairs : Fin n → ι × ι) (x : EuclideanSpace ℂ ι) :
    AnalyticOnNhd ℂ (fun z : Fin n → ℂ => rotationWord pairs z x) Set.univ :=
  indexedWord_entire pairs _ x


/-- The same coordinate formula bundled as an actual complex-linear map. -/
def complexRotationLinear (i j : ι) (z : ℂ) :
    EuclideanSpace ℂ ι →ₗ[ℂ] EuclideanSpace ℂ ι where
  toFun := complexPlaneRotate i j z
  map_add' x y := by
    by_cases hij : i=j
    · simp [complexPlaneRotate, hij]
    · ext k
      simp only [complexPlaneRotate, hij, if_false, PiLp.add_apply]
      split_ifs <;> ring
  map_smul' c x := by
    by_cases hij : i=j
    · simp [complexPlaneRotate, hij]
    · ext k
      simp only [complexPlaneRotate, hij, if_false, PiLp.smul_apply, smul_eq_mul, RingHom.id_apply]
      split_ifs <;> ring

/-- Bundled bounded operator, with an explicit dimension-independent bound. -/
def complexPlaneRotation (i j : ι) (z : ℂ) :
    EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι :=
  (complexRotationLinear i j z).mkContinuous (Real.exp (2*‖z‖))
    (complexPlaneRotate_norm i j z)

@[simp] theorem complexPlaneRotation_apply (i j : ι) (z : ℂ) (x : EuclideanSpace ℂ ι) :
    complexPlaneRotation i j z x = complexPlaneRotate i j z x := rfl

theorem complexPlaneRotation_operator_norm (i j : ι) (z : ℂ) :
    ‖complexPlaneRotation i j z‖ ≤ Real.exp (2*‖z‖) := by
  apply ContinuousLinearMap.opNorm_le_bound _ (Real.exp_pos _).le
  exact complexPlaneRotate_norm i j z

theorem complexPlaneRotation_operator_displacement (i j : ι) (z : ℂ) :
    ‖complexPlaneRotation i j z - ContinuousLinearMap.id ℂ (EuclideanSpace ℂ ι)‖ ≤
      Real.exp (2*‖z‖)-1 := by
  apply ContinuousLinearMap.opNorm_le_bound _
    (sub_nonneg.mpr (Real.one_le_exp_iff.mpr (by positivity)))
  intro x
  exact complexPlaneRotate_displacement i j z x

omit [Fintype ι] in
@[simp] theorem rotationWord_zero {n : ℕ} (pairs : Fin n → ι × ι)
    (x : EuclideanSpace ℂ ι) : rotationWord pairs 0 x = x := indexedWord_zero pairs _ x


/-- Actual infinitesimal complex rotation, with zero diagonal generator. -/
theorem complexPlaneRotate_hasDerivAt_zero (i j : ι) (x : EuclideanSpace ℂ ι) :
    HasDerivAt (fun z : ℂ => complexPlaneRotate i j z x)
      (if i=j then 0 else planeGenerator i j x) 0 := by
  by_cases hij : i=j
  · simpa only [complexPlaneRotate, hij, if_true] using hasDerivAt_const (0 : ℂ) x
  · simp only [hij, if_false]
    simp_rw [complexPlaneRotate_decomposition i j hij]
    have hc := ((Complex.hasDerivAt_cos (0 : ℂ)).sub_const 1).smul_const (planeProjection i j x)
    have hs := (Complex.hasDerivAt_sin (0 : ℂ)).smul_const (planeGenerator i j x)
    convert! (hc.const_add x).add hs using 1
    simp

end QIRadon.ComplexRotation
